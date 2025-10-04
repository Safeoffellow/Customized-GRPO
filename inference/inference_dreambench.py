import os
import dataclasses
from typing import Literal

from accelerate import Accelerator
import transformers
from transformers import HfArgumentParser
import json
import itertools

from uno.flux.pipeline import UNOPipeline, preprocess_ref
# from uno.reward.clip import CLIPScore, single_eval_clipt_score, single_eval_clipi_score
from uno.reward.dino import Dinov2Score, single_eval_dino_score
from uno.reward.dreambench import clipeval, dinoeval_image, clipeval_image
from tqdm import tqdm
import torch
from PIL import Image, ImageDraw, ImageFont
from uno.reward.hpsv3.inference import HPSv3RewardInferencer
from uno.reward.vision_transformer import vit_small
import clip
from collections import OrderedDict

# 图像拼接
def horizontal_concat(images):
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_im.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return new_im

def tensor_to_float(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return x

@dataclasses.dataclass
class InferenceArgs:
    prompt: str | None = None
    image_paths: list[str] | None = None
    eval_json_path: str | None = None
    offload: bool = False
    num_images_per_prompt: int = 1
    model_type: Literal["flux-dev", "flux-dev-fp8", "flux-schnell"] = "flux-dev"
    width: int = 512
    height: int = 512
    ref_size: int = -1
    num_steps: int = 25
    guidance: float = 4
    seed: int = 3407
    save_path: str = "output/inference"
    only_lora: bool = True
    concat_refs: bool = False
    lora_rank: int = 512
    data_resolution: int = 512
    pe: Literal['d', 'h', 'w', 'o'] = 'd'
    lora_path: str | None = None
    clip_path: str = "./checkpoints/clip-vit-large-patch14/"
    dino_path: str = "./checkpoints/dinov2-base/"

def main(args: InferenceArgs):
    
    accelerator = Accelerator()
    device = accelerator.device
    print(f"accelerator.device: {accelerator.device}")
    clip_model, _ = clip.load("./checkpoints/ViT-B-32.pt", device=device, jit=False)
    clip_model.eval()
    
    dino_model_v1 = torch.hub.load(
                "./checkpoints/dino-main",
                "dino_vits16",
                source='local',
                pretrained=False
            ).to(device, dtype=torch.float32)
    local_weights_path = "./checkpoints/dino_deitsmall16_pretrain.pth"
    checkpoint = torch.load(local_weights_path, map_location=device)
    if 'student' in checkpoint:
        state_dict = checkpoint['student']
    else:
        state_dict = checkpoint
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v


    dino_model_v1.load_state_dict(new_state_dict)

    dino_model_v1.eval()

    dino_model_v2 = Dinov2Score(model_or_name_path = args.dino_path, device = device)
    hps_v3_model = HPSv3RewardInferencer('./checkpoints/HPSv3/HPSv3_7B.yaml', "./checkpoints/HPSv3/HPSv3.safetensors", device = device)

    pipeline = UNOPipeline(
        args.model_type,
        device,
        args.offload,
        only_lora=args.only_lora,
        lora_rank=args.lora_rank,
        lora_path=args.lora_path
    )

    assert args.prompt is not None or args.eval_json_path is not None, \
        "Please provide either prompt or eval_json_path"
    
    if args.eval_json_path is not None:
        with open(args.eval_json_path, "rt") as f:
            data_dicts = json.load(f)
        data_root = os.path.dirname(args.eval_json_path)
    else:
        data_root = "./"
        data_dicts = [{"prompt": args.prompt, "image_paths": args.image_paths}]

    clipi_score_all = []
    clipt_score_all = []
    dino_v1_score_all = []
    dino_v2_score_all = []
    # hps_score_all = []

    # 嵌套索引
    for (i, data_dict), j in itertools.product(enumerate(data_dicts), range(args.num_images_per_prompt)):
        if (i * args.num_images_per_prompt + j) % accelerator.num_processes != accelerator.process_index:
            continue
        ref_img_base = os.path.join(args.save_path, "ref_img")
        gen_img_base = os.path.join(args.save_path, "gen_img")
        prompt_base = os.path.join(args.save_path, "prompt")
        json_files_base = os.path.join(args.save_path, "json_files")
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(ref_img_base, exist_ok=True)
        os.makedirs(gen_img_base, exist_ok=True)
        os.makedirs(prompt_base, exist_ok=True)
        os.makedirs(json_files_base, exist_ok=True)

        ref_imgs = [
            Image.open(os.path.join(data_root, img_path))
            for img_path in data_dict["image_paths"]
        ]
        ref_img_paths = [
            os.path.join(data_root, img_path)
            for img_path in data_dict["image_paths"]
        ]

        if len(ref_imgs) == 1:
            ref_imgs[0].save(os.path.join(ref_img_base, f"{i}_{j}.png"))
        else:
            for k, ref_img in ref_imgs:
                ref_img.save(os.path.join(ref_img_base, f"{i}_{k}_{j}.png"))
        

        if args.ref_size==-1:
            args.ref_size = 512 if len(ref_imgs)==1 else 320

        ref_imgs = [preprocess_ref(img, args.ref_size) for img in ref_imgs]

        image_gen = pipeline(
            prompt=data_dict["prompt"],
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed + j,
            ref_imgs=ref_imgs,
            pe=args.pe,
        )
        clipt_score = float(clipeval([image_gen], [data_dict["prompt"]], clip_model, device))
        clipi_score = float(clipeval_image([image_gen], ref_img_paths, clip_model, device))
        dino_v1_score = float(dinoeval_image([image_gen], ref_img_paths, dino_model_v1, device))
        dino_v2_score = single_eval_dino_score(ref_img_paths, image_gen, dino_model_v2)
        hps_score_r = hps_v3_model.reward([image_gen], data_dict["prompt"])
        hps_score = hps_score_r[0][0]
        dino_v2_score = tensor_to_float(dino_v2_score)

        clipt_score_all.append(clipt_score)
        clipi_score_all.append(clipi_score)
        dino_v1_score_all.append(dino_v1_score)
        dino_v2_score_all.append(dino_v2_score)
        hps_score_all.append(hps_score)
        

        if args.concat_refs:
            image_gen = horizontal_concat([image_gen, *ref_imgs])

        prompt = data_dict["prompt"]
        image_gen.save(os.path.join(gen_img_base, f"{i}_{j}.png"))
        # save config and image
        temp_dict = {}
        temp_dict["prompt"] = data_dict["prompt"]
        temp_dict['reference_image'] = data_dict["image_paths"]
        temp_dict["generated_image"] = os.path.join(gen_img_base, f"{i}_{j}.png")
        temp_dict["reward"] = {"clip_t": clipt_score, "clip_i": clipi_score, "dino_v1":dino_v1_score, "dino_v2": dino_v2_score, "HPS-V3": hps_score} 
        with open(os.path.join(json_files_base, f"{i}_{j}.json"), 'w') as f:
            json.dump(temp_dict, f, indent=4)        
        with open(os.path.join(prompt_base, f"{i}_{j}.txt"), 'w') as f:
            f.write(data_dict["prompt"])
    
    with open(os.path.join(args.save_path, "all_score.txt"), 'w') as f:
        f.write(f"Avg CLIP-T: {sum(clipt_score_all)/len(clipt_score_all)}\n")
        f.write(f"Avg CLIP-I: {sum(clipi_score_all)/len(clipi_score_all)}\n")
        f.write(f"Avg DINO_v1: {sum(dino_v1_score_all)/len(dino_v1_score_all)}\n")
        f.write(f"Avg DINO_v2: {sum(dino_v2_score_all)/len(dino_v2_score_all)}\n")
        f.write(f"Avg HPS: {sum(hps_score_all)/len(hps_score_all)}\n")

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, help="config yaml path", required=True)
    args = parser.parse_args()
    args_tuple = transformers.HfArgumentParser([InferenceArgs]).parse_yaml_file(args.config)
    main(*args_tuple)