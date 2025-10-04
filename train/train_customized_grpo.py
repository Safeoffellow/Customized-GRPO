import dataclasses
import gc
import itertools
import logging
import math
import os
import random
from copy import deepcopy
from typing import TYPE_CHECKING, Literal
import time
import torch
import torch.nn.functional as F
import transformers
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from uno.flux.util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from uno.dataset.uno import FluxPairedDataset_My, FluxPairedDataset_without_text
from uno.flux.sampling import denoise, get_noise, get_schedule, prepare_multi_ip, unpack, prepare_multi_without_text
from uno.flux.util import load_ae, load_clip, load_flow_model, load_t5, set_lora, load_flow_model_only_lora, load_flow_model_merge
from uno.flux.logging_ import main_print
from uno.reward.clip import CLIPScore, single_eval_clipt_score
from uno.reward.dino import Dinov2Score, single_eval_dino_score
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data.distributed import DistributedSampler
from uno.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
    sp_parallel_dataloader_wrapper,
    sp_parallel_dataloader_wrapper_without_text
)
from uno.utils.checkpoint import save_checkpoint, save_lora_my
from uno.utils.grpo_states import GRPOTrainingStates
import torch.distributed as dist
from diffusers.image_processor import VaeImageProcessor
if TYPE_CHECKING:
    from uno.flux.model import Flux
    from uno.flux.modules.autoencoder import AutoEncoder
    from uno.flux.modules.conditioner import HFEmbedder
import numpy as np
from typing import List
from diffusers import FluxTransformer2DModel, AutoencoderKL
from collections import deque

import wandb
import datetime
import torchvision.transforms.functional as TVF
from torch.nn.utils import clip_grad_norm_
from hpsv3 import HPSv3RewardInferencer
import shutil

def get_weights_sigmoid_final(t, num_timesteps=25):

    HPS_MAX_WEIGHT = 0.7
    HPS_MIN_WEIGHT = 0.3  

    # 阶段划分
    TRANSITION_START_STEP = 6
    TRANSITION_END_STEP = 14 
    FINAL_INTEGRATION_STEP = 22

    TRANSITION_CENTER = (TRANSITION_START_STEP + TRANSITION_END_STEP) / 2

    TRANSITION_SCALE = 0.8 
    
    # 最终整合期权重
    FINAL_HPS_WEIGHT = 0.5 



    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


    # 阶段 1: 结构定义期
    if t < TRANSITION_START_STEP:
        w_hps = HPS_MAX_WEIGHT
    
    # 阶段 3: 最终整合期
    elif t >= FINAL_INTEGRATION_STEP:
        w_hps = FINAL_HPS_WEIGHT
        
    # 阶段 2: 平滑过渡期
    else:

        sigmoid_input = (t - TRANSITION_CENTER) * TRANSITION_SCALE
        

        sigmoid_value = _sigmoid(sigmoid_input)
        

        reversed_sigmoid = 1.0 - sigmoid_value
        
       
        hps_range = HPS_MAX_WEIGHT - HPS_MIN_WEIGHT
        w_hps = HPS_MIN_WEIGHT + hps_range * reversed_sigmoid

    # DINO的权重始终是与HPS互补的
    w_dino = 1.0 - w_hps
    
    return w_hps, w_dino

def save_tensor(tensor_dict, filename, save_path):
    if save_path is not None:
        torch.save(tensor_dict, os.path.join(save_path, filename))


def flux_step(
    model_output: torch.Tensor,   
    latents: torch.Tensor,  
    eta: float, 
    sigmas: torch.Tensor,
    index: int,   
    prev_sample: torch.Tensor,  
    grpo: bool,    
    sde_solver: bool,   
):

    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma

    prev_sample_mean = latents + dsigma * model_output
    
    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]    
    std_dev_t = eta * math.sqrt(delta_t)  
    
    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        if sde_solver:
            prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
        else:
            prev_sample = prev_sample_mean
        
    if grpo:
        log_prob = (
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean,pred_original_sample

def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)

logger = get_logger(__name__)

def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

def get_models(name: str, device, offload: bool=False):
    model = load_flow_model_merge(
        name,
        device=device,
        local=False
    )
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae

def run_sample_step(
        args,
        z,    
        progress_bar,   
        sigma_schedule,   
        dit,
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        text_ids,      
        image_ids,    
        ref_img,
        ref_img_ids,
        grpo_sample,
        determistic
    ):
    if grpo_sample:
        all_latents = [z] 
        all_log_probs = []
        for i in progress_bar:  
            B = encoder_hidden_states.shape[0]
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
            dit.eval()
            with torch.autocast("cuda", torch.bfloat16):  
                pred= dit(
                    img=z, 
                    img_ids = image_ids, 
                    ref_img = ref_img,
                    ref_img_ids = ref_img_ids,
                    txt=encoder_hidden_states,
                    txt_ids = text_ids,
                    y = pooled_prompt_embeds,
                    timesteps=timesteps/1000,
                    guidance=torch.tensor(
                        [3.5],
                        device=z.device,
                        dtype=torch.bfloat16
                    ),
                ) 
            if determistic[i]:
                z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=False)
            else:
                z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original 
        all_latents = torch.stack(all_latents, dim=1)  
        all_log_probs = torch.stack(all_log_probs, dim=1)  
        return z, latents, all_latents, all_log_probs

def grpo_one_step(
            args,
            latents,
            pre_latents,
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            image_ids,
            dit,
            timesteps,
            ref_img,
            ref_img_ids,
            i,
            sigma_schedule,
):
    B = encoder_hidden_states.shape[0]
    dit.train()
    with torch.autocast("cuda", torch.bfloat16):
        pred= dit(
            img = latents,
            img_ids = image_ids,
            ref_img = ref_img,
            ref_img_ids = ref_img_ids,

            txt=encoder_hidden_states,
            txt_ids = text_ids,
            y = pooled_prompt_embeds,
            timesteps=timesteps/1000,
            guidance=torch.tensor(
                [3.5], 
                device=latents.device,
                dtype=torch.bfloat16
            ),
        )[0]
    z, pred_original, log_prob = flux_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
    return log_prob

def sample_reference_model(
    args,
    device, 
    dit,
    vae,
    encoder_hidden_states, 
    pooled_prompt_embeds,
    img_ids,
    text_ids,
    ref_img,
    ref_img_ids,
    reward_model,
    prompts,
    ids,
    ref_image_paths,
    timesteps_train,
    global_step,
):
    w, h = args.w, args.h
    sample_steps = args.sampling_steps 
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1).to(device) 
    
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )
    
    B = encoder_hidden_states.shape[0] 
    SPATIAL_DOWNSAMPLE = 8 
    IN_CHANNELS = 16    
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)   

    all_latents = [] 
    all_log_probs = [] 
    all_rewards = []  
    all_image_ids = [] 
    all_ref_image_ids = []
    all_ref_img = []

    if args.init_same_noise:
        input_latents = torch.randn(
                (1, IN_CHANNELS, latent_h, latent_w),  #（b,c,h,w)
                device=device,
                dtype=torch.bfloat16,
            )
    if dist.get_rank() == 0:
        sampling_time = 0
    for index, batch_idx in enumerate(batch_indices):
        if dist.get_rank() == 0:
            meta_sampling_time = time.time()
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx]
        batch_text_ids = text_ids[batch_idx]

        batch_ref_img = [r[batch_idx] for r in ref_img]     
        batch_ref_img_ids = [ids[batch_idx] for ids in ref_img_ids] 

        batch_prompt = [prompts[i] for i in batch_idx]
        batch_ids = [ids[i] for i in batch_idx]
        batch_img_ids = img_ids[batch_idx]
        batch_ref_img_paths = [ref_image_paths[i] for i in batch_idx]
        
        if not args.init_same_noise:
            input_latents = torch.randn(
                    (len(batch_idx), IN_CHANNELS, latent_h, latent_w),  
                    device=device,
                    dtype=torch.bfloat16,
                )
        input_latents_new = pack_latents(input_latents, len(batch_idx), IN_CHANNELS, latent_h, latent_w)
        grpo_sample=True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")

        if args.training_strategy == "part":
            determistic = [True] * sample_steps
            for i in timesteps_train:
                determistic[i] = False
        elif args.training_strategy == "all":
            determistic = [False] * sample_steps

        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs = run_sample_step(
                args,
                input_latents_new,
                progress_bar,
                sigma_schedule,
                dit,
                batch_encoder_hidden_states,
                batch_pooled_prompt_embeds,
                batch_text_ids,
                batch_img_ids,
                batch_ref_img,
                batch_ref_img_ids,
                grpo_sample,
                determistic=determistic
            )

        if dist.get_rank() == 0:
            sampling_time += time.time() - meta_sampling_time
            main_print(f"##### Sampling time per data: {sampling_time/(index+1)} seconds")
                       
        all_image_ids.append(batch_img_ids)
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        all_ref_img.append(batch_ref_img)
        all_ref_image_ids.append(batch_ref_img_ids)
        
        image_processor = VaeImageProcessor(16)  
        rank = int(os.environ["RANK"])
        temp_path = os.path.join(args.temp_image_path, f"single_syncd_1w_aug_mixgrpo_eta{args.eta}_lr{args.learning_rate}_hps_dino_{args.clip_weight}_{args.dino_weight}_gamma_{args.gamma}_{args.conflict_strategy}_{args.weight_strategy}_{args.hps_high}_{args.hps_low}")
        image_base = os.path.join(temp_path, "images", str(global_step))
        os.makedirs(image_base, exist_ok=True) 

        
        with torch.inference_mode(): 
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents = unpack_latents(latents, h, w, 8)
                image = vae.decode(latents)[0]
                if image.dim() == 3:
                    image = image.unsqueeze(0)

                decoded_image = image_processor.postprocess(image)
        
        name = batch_ids[0]
        image_temp = os.path.join(image_base, name)
        os.makedirs(image_temp, exist_ok=True)

        reward_temp = []

        if args.use_clip:
            with torch.no_grad():
                clip_m = reward_model[0]
                if isinstance(decoded_image, list):
                    clip_score = single_eval_clipt_score(batch_prompt, decoded_image[0], clip_m)
                else:
                    clip_score = single_eval_clipt_score(batch_prompt, decoded_image, clip_m)
                print("Prompt:", batch_prompt)
                reward_temp.append(clip_score)
                print("CLIP_score:", clip_score)
        if args.use_hps_v3:
            with torch.no_grad():
                hps_model = reward_model[0]
                if isinstance(decoded_image, list):
                    hps_score_r = hps_model.reward(decoded_image, batch_prompt)
                else:
                    hps_score_r = hps_model.reward([decoded_image], batch_prompt)
                hps_score = hps_score_r[0][0]
                print("Prompt:", batch_prompt)
                reward_temp.append(hps_score)
                print("HPS_score:", hps_score)
        
        ref_img_path_temp = batch_ref_img_paths[0][0].replace(".png", "_seg.png")          
        if args.use_dino:
            with torch.no_grad():
                dino_m = reward_model[1]
                if os.path.exists(ref_img_path_temp):
                    ref_img_path_temp = [ref_img_path_temp]
                else:
                    ref_img_path_temp = batch_ref_img_paths[0]

                if isinstance(decoded_image, list):
                    dino_score = single_eval_dino_score(ref_img_path_temp, decoded_image[0], dino_m)
                else:
                    dino_score = single_eval_dino_score(ref_img_path_temp, decoded_image, dino_m)
                print("Reference_img", batch_ref_img_paths[0])
                
                reward_temp.append(dino_score)
                print("DINO_score:", dino_score)
        

        image_path = os.path.join(image_temp, f"flux_{rank}_global{global_step}_dino{dino_score.item():.2f}_hps{hps_score.item():.2f}_{index}.png")
        with open(os.path.join(image_temp, "prompt.txt"), 'w') as writer:
            writer.write(batch_prompt[0])
        ref_img_seg_save = os.path.join(image_temp, f"reference_img_seg.png")
        ref_img_save = os.path.join(image_temp, f"reference_img.png")
        print("Generated_img:", image_path)
        if isinstance(decoded_image, list):
            if not os.path.exists(image_path):  
                decoded_image[0].save(image_path)

        else:
            if not os.path.exists(image_path): 
                decoded_image.save(image_path)
        if not os.path.exists(ref_img_save):
            shutil.copy(batch_ref_img_paths[0][0], ref_img_save)

        
        reward_temp = torch.stack(reward_temp) 
        all_rewards.append(reward_temp)
    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_rewards = torch.stack(all_rewards, dim=0)    
    all_image_ids = torch.cat(all_image_ids, dim=0)  

    return all_rewards, all_latents, all_log_probs, sigma_schedule, all_image_ids

def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def train_one_step(
    args,
    device,
    dit,
    vae,
    reward_model,
    optimizer,
    lr_scheduler,
    loader,
    max_grad_norm,
    timesteps_train,
    global_step,
):
    temp_path = os.path.join(args.temp_image_path, f"single_syncd_1w_aug_mixgrpo_eta{args.eta}_lr{args.learning_rate}_hps_dino_{args.clip_weight}_{args.dino_weight}_gamma_{args.gamma}_{args.conflict_strategy}_{args.weight_strategy}_{args.hps_high}_{args.hps_low}")
    total_loss = 0.0
    optimizer.zero_grad()

    prompts, ids, ref_imgs, ref_img_paths, encoder_hidden_states, pool_prompt_embed = next(loader)
    ref_imgs_num = len(ref_imgs)

    with torch.no_grad():
        x_ref = [vae.encode(ref_img.to(device).to(torch.float32)) for ref_img in ref_imgs]

        x_1_ = torch.randn_like(x_ref[0], device = device)
        inp = prepare_multi_without_text(encoder_hidden_states=encoder_hidden_states, pool_prompt_embed=pool_prompt_embed, img=x_1_, prompt=prompts, ref_imgs=tuple(x_ref), pe=args.pe)
        x_1_ = rearrange(x_1_, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        x_ref = [rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) for x in x_ref]
            
        encoder_hidden_states = inp["txt"]
        pooled_prompt_embeds = inp["vec"]
        text_ids = inp["txt_ids"]
        
        ref_img_ids = inp["ref_img_ids"]
        ref_img = inp["ref_img"]
        img_ids = inp["img_ids"]

    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)
        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        pooled_prompt_embeds = repeat_tensor(pooled_prompt_embeds)
        text_ids = repeat_tensor(text_ids)

        if isinstance(ref_img, (list, tuple)):
            ref_img = [repeat_tensor(t) for t in ref_img]
        else:  
            ref_img = repeat_tensor(ref_img)

        if isinstance(ref_img_ids, (list, tuple)):
            ref_img_ids = [repeat_tensor(t) for t in ref_img_ids]
        else:
            ref_img_ids = repeat_tensor(ref_img_ids)

        img_ids = repeat_tensor(img_ids)
        ref_image_paths = [item for item in ref_img_paths for _ in range(args.num_generations)]

        if isinstance(prompts, str):
            prompts = [prompts] * args.num_generations
        elif isinstance(prompts, list):
            prompts = [item for item in prompts for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported prompts type: {type(prompts)}")
        
        if isinstance(ids, str):
            ids = [ids] * args.num_generations
        elif isinstance(ids, list):
            ids = [item for item in ids for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported ids type: {type(ids)}")

    reward, all_latents, all_log_probs, sigma_schedule, all_image_ids = sample_reference_model(
            args,
            device, 
            dit,
            vae,
            encoder_hidden_states,
            pooled_prompt_embeds,
            img_ids,
            text_ids,
            ref_img,
            ref_img_ids,
            reward_model,
            prompts,
            ids,
            ref_image_paths,
            timesteps_train,
            global_step
        )
    batch_size = all_latents.shape[0]
    
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps =  torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)
    
    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[:, :-1][:, :-1].detach().clone(),
        "next_latents": all_latents[:, 1:][:, :-1].detach().clone(),
        "log_probs": all_log_probs[:, :-1].detach().clone(),
        "rewards": reward,
        "image_ids": all_image_ids,
        "text_ids": text_ids,
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "ref_img": ref_img,
        "ref_img_ids": ref_img_ids,
    }

    gathered_reward = gather_tensor(samples["rewards"])     # [world*N, 2]
    means = None
    reward_std = None

    if dist.get_rank() == 0:
        means = gathered_reward.mean(dim=0)
        reward_std = gathered_reward.std(dim=0)
        main_print(f"gathered clip: {means[0].item():.4f}, gathered dino: {means[1].item():.4f}")
        reward_file_path  = os.path.join(temp_path, "reward.txt")
        with open(reward_file_path, 'a') as f: 
            f.write(f"{means[0].item()} {means[1].item()}\n")

    if args.use_group:   
        n_original_batch = len(samples["rewards"]) // args.num_generations

        advantages = torch.zeros(samples["rewards"].shape[0], device=samples["rewards"].device)
        advantages_clip = torch.zeros(samples["rewards"].shape[0], device=samples["rewards"].device)
        advantages_dino = torch.zeros(samples["rewards"].shape[0], device=samples["rewards"].device)
        advantages_bonus = torch.zeros(samples["rewards"].shape[0], device=samples["rewards"].device)

        for i in range(n_original_batch):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_reward_clip = samples["rewards"][start_idx:end_idx, 0]
            group_reward_dino = samples["rewards"][start_idx:end_idx, 1]
            group_mean_clip = group_reward_clip.mean()
            group_mean_dino = group_reward_dino.mean()
            group_std_clip = group_reward_clip.std() + 1e-8
            group_std_dino = group_reward_dino.std() + 1e-8

            clip_adv = (group_reward_clip - group_mean_clip) / group_std_clip
            dino_adv = (group_reward_dino - group_mean_dino) / group_std_dino
            if args.conflict_strategy == "tanh":
                prod = clip_adv * dino_adv
                at_least_one_positive = (clip_adv > 0) | (dino_adv > 0)

                conflict_bonus = torch.empty_like(prod)
                conflict_bonus[at_least_one_positive] = args.gamma * torch.tanh(prod[at_least_one_positive])
                conflict_bonus[~at_least_one_positive] = -args.gamma * torch.tanh(prod[~at_least_one_positive])

                advantages_dino[start_idx:end_idx] = dino_adv
                advantages_clip[start_idx:end_idx] = clip_adv
                advantages_bonus[start_idx:end_idx] = conflict_bonus
            elif args.conflict_strategy == "tanh_only":
                prod = clip_adv * dino_adv
                at_least_one_positive = (clip_adv > 0) | (dino_adv > 0)

                conflict_bonus = torch.empty_like(prod)
                conflict_bonus[at_least_one_positive] = 1 * torch.tanh(prod[at_least_one_positive])
                conflict_bonus[~at_least_one_positive] = -1 * torch.tanh(prod[~at_least_one_positive])
                advantages_dino[start_idx:end_idx] = dino_adv
                advantages_clip[start_idx:end_idx] = clip_adv
                advantages_bonus[start_idx:end_idx] = conflict_bonus
            elif args.conflict_strategy == "min":
                advantages_dino[start_idx:end_idx] = dino_adv
                advantages_clip[start_idx:end_idx] = clip_adv
                conflict_bonus = torch.zeros_like(clip_adv)
                advantages[start_idx:end_idx] = torch.min(clip_adv, dino_adv)
            elif args.conflict_strategy == "max":
                advantages_dino[start_idx:end_idx] = dino_adv
                advantages_clip[start_idx:end_idx] = clip_adv        
                conflict_bonus = torch.zeros_like(clip_adv)
                advantages[start_idx:end_idx] = torch.max(clip_adv, dino_adv)
            elif args.conflict_strategy == "harmonic":
                eps = 1e-6
                conflict_bonus = args.gamma * (clip_adv * dino_adv) / (clip_adv + dino_adv + eps)
                advantages_dino[start_idx:end_idx] = dino_adv
                advantages_clip[start_idx:end_idx] = clip_adv 
                advantages_bonus[start_idx:end_idx] = conflict_bonus
            else:
                clip_adv_pos = torch.nn.functional.softplus(clip_adv)
                dino_adv_pos = torch.nn.functional.softplus(dino_adv)
                
                advantages_dino[start_idx:end_idx] = dino_adv
                advantages_clip[start_idx:end_idx] = clip_adv
                advantages_bonus[start_idx:end_idx] = dino_adv


        samples["advantages"] = advantages
        samples["advantages_clip"] = advantages_clip
        samples["advantages_dino"] = advantages_dino
        samples["advantages_bonus"] = advantages_bonus
    if args.training_strategy == "all":
        perms = torch.stack(
            [
                torch.randperm(len(samples["timesteps"][0]))
                for _ in range(batch_size)
            ]
        ).to(device) 
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            samples[key] = samples[key][
                torch.arange(batch_size).to(device) [:, None],
                perms,
            ]

    # --debug_2
    batch_size = next(v.shape[0] for v in samples.values() if isinstance(v, torch.Tensor))

    samples_batched = {}
    for k, v in samples.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
            samples_batched[k] = [v[i:i+1] for i in range(batch_size)]
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            samples_batched[k] = [ [t[i:i+1] for t in v] for i in range(batch_size)]
        else:
            samples_batched[k] = [v for _ in range(batch_size)]  

    samples_batched_list = [
        {k: v[i] for k, v in samples_batched.items()}
        for i in range(batch_size)
    ]

    if args.training_strategy == "part":
        train_timesteps = timesteps_train
    elif args.training_strategy == "all":
        train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)
        train_timesteps = range(train_timesteps)

    conflict_num = 0
    print("Sample_List Len:", len(samples_batched_list))

    if dist.get_rank() == 0:
        optimize_sampling_time = 0

    for i,sample in list(enumerate(samples_batched_list)):
        print(f"==== grpo one step on sample {i}")
        for _ in train_timesteps:
            main_print(f"Current Optimaze Timestep: {_}" )
            if dist.get_rank() == 0:
                meta_optimize_sampling_time = time.time()
            clip_range = float(args.clip_range)
            adv_clip_max = args.adv_clip_max

            new_log_probs = grpo_one_step(
                args,
                sample["latents"][:,_],
                sample["next_latents"][:,_],
                sample["encoder_hidden_states"],
                sample["pooled_prompt_embeds"],
                sample["text_ids"],
                sample["image_ids"],
                dit,
                sample["timesteps"][:,_],   
                sample["ref_img"],
                sample["ref_img_ids"],
                perms[i][_] if args.training_strategy == "all" else _,
                sigma_schedule,
            )
            
            if dist.get_rank() == 0:
                meta_optimize_sampling_time = time.time() - meta_optimize_sampling_time
                optimize_sampling_time += meta_optimize_sampling_time

            if args.weight_strategy == 'dynamic':
                clip_weight, dino_weight = get_weights_sigmoid_final(_, num_timesteps=25) 
                main_print(f"Current HPS_weight {clip_weight}, Current DINO_weight {dino_weight}")
            elif args.weight_strategy == "fix":
                clip_weight = args.clip_weight
                dino_weight = args.dino_weight

            if args.conflict_strategy == "tanh":
                timestep_specific_advantage = (
                    sample["advantages_clip"] * clip_weight +
                    sample["advantages_dino"] * dino_weight +
                    sample["advantages_bonus"]
                )
            elif args.conflict_strategy == "tanh":
                timestep_specific_advantage = sample["advantages_bonus"]
            elif args.conflict_strategy == "min":
                timestep_specific_advantage = sample["advantages"]
            elif args.conflict_strategy == "max":
                timestep_specific_advantage = sample["advantages"]
            elif args.conflict_strategy == "harmonic":
                timestep_specific_advantage = (
                    sample["advantages_clip"] * clip_weight +
                    sample["advantages_dino"] * dino_weight +
                    sample["advantages_bonus"] 
                )
            else:
                clip_adv_pos = torch.nn.functional.softplus(sample["advantages_clip"])
                dino_adv_pos = torch.nn.functional.softplus(sample["advantages_dino"])
                
                product_advantages = (clip_adv_pos ** (clip_weight*2)) * (dino_adv_pos ** (dino_weight*2))
                
                baseline_score = torch.nn.functional.softplus(torch.tensor(0.0).to(clip_adv.device)) ** (clip_weight*2) * \
                                torch.nn.functional.softplus(torch.tensor(0.0).to(clip_adv.device)) ** (dino_weight*2)
                                
                final_advantages_raw = product_advantages - baseline_score
                
                final_advantages_mean = final_advantages_raw.mean()
                final_advantages_std = final_advantages_raw.std() + 1e-8
                timestep_specific_advantage = (final_advantages_raw - final_advantages_mean) / final_advantages_std

            advantages = torch.clamp(
                timestep_specific_advantage,
                -adv_clip_max,
                adv_clip_max,
            )

            ratio = torch.exp(new_log_probs - sample["log_probs"][:,_])

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * len(train_timesteps))

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()

        if dist.get_rank() == 0:
            main_print(f"##### Optimize sampling time per step: {optimize_sampling_time / (i+1)} seconds")
        if (i+1)%args.gradient_accumulation_steps==0:
            grad_norm = clip_grad_norm_(
                filter(lambda p: p.requires_grad and p.grad is not None, dit.parameters()),
                max_grad_norm
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        dist.barrier()
    return total_loss, grad_norm.item(), means, reward_std

now = datetime.datetime.now()

now_name = now.strftime("%Y-%m-%d_%H-%M-%S")

@dataclasses.dataclass
class TrainArgs:

    project_dir: str | None = None
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    gradient_accumulation_steps: int = 4
    seed: int = 42
    wandb_project_name: str | None = "flux_grpo_multisubject"
    wandb_run_name: str | None = None
    wandb_tags: list[str] = dataclasses.field(default_factory=list)
    load_pretrain_lora: bool = True
    model_name: Literal["flux-dev", "flux-schnell"] = "flux-dev"

    lora_rank: int = 0

    double_blocks_indices: list[int] | None = dataclasses.field(
        default=None,
        metadata={"help": "Indices of double blocks to apply LoRA. None means all double blocks."}
    )
    single_blocks_indices: list[int] | None = dataclasses.field(
        default=None,
        metadata={"help": "Indices of double blocks to apply LoRA. None means all single blocks."}
    )
    pe: Literal["d", "h", "w", "o"] = "d"
    gradient_checkpoint: bool = True
    ema: bool = False 
    ema_interval: int = 1
    ema_decay: float = 0.99

    fsdp_sharding_startegy: str = "full"
    sampling_steps: int = 16 
    eta: float = 0.3
    sde_solver: float = True 
    shift: float = 3
    guidance_scale: float = 3.5 
    init_same_noise: bool = True 

    use_clip: bool = False
    use_dino: bool = True
    use_hps_v3: bool = True
    text_dropout: float = 0.0
    clip_range: float = 1e-4 
    adv_clip_max: float = 5.0 

    use_group: bool = True 
    num_generations: int = 12 
    timestep_fraction: float = 0.6
    selective_checkpointing: bool = True
    gradient_checkpointing: bool = True
    clip_weight: float = 0.5
    dino_weight: float = 0.5

    ## optimizer
    learning_rate: float = 1e-5
    adam_betas: list[float] = dataclasses.field(default_factory=lambda: [0.9, 0.999])
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.0001
    max_grad_norm: float = 1.0

    ## lr_scheduler
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 0
    max_train_steps: int = 1000000
    lr_num_cycles: int = 1

    ## dataloader
    train_data_json: str = "./datasets/syncd_1w/train_1w_cleaned.json"
    train_batch_size: int = 1
    train_image_base: str= "./datasets/syncd"
    resolution: int = 512
    resolution_ref: int | None = None
    sampler_seed: int = 1223627
    dataloader_num_workers: int = 4
    lr_power: float = 1.0 

    ## misc
    resume_from_checkpoint: str | None | Literal["latest"] = None
    checkpointing_steps: int = 40

    ### MixGRPO
    training_strategy: str = "part"
    iters_per_group: int = 25
    group_size: int = 4
    sample_strategy: str = "progressive"
    prog_overlap: bool = True
    prog_overlap_step: int = 0
    max_iters_per_group : int = 10
    min_iters_per_group: int = 1
    roll_back: bool = True

    # other pa
    sp_size: int = 1
    train_sp_batch_size: int = 1
    use_cpu_offload: bool = False
    master_weight_type: str = "fp32"
    w: int = 512
    h: int = 512
    temp_image_path: str = "./Single_syncd_1w_grpo/"
    # conflict training strategy
    gamma: float = 0.5
    conflict_strategy: str = "tanh"
    weight_strategy: str = "dynamic"
    hps_high: float = 0.7
    hps_low: float = 0.3

    log_file: str = "./log_file"

def main(
    args: TrainArgs,
):
    torch.backends.cuda.matmul.allow_tf32 = True
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.project_dir is not None:
        os.makedirs(args.project_dir, exist_ok=True)

    if rank <= 0:
        project = args.wandb_project_name
        project_name = args.wandb_run_name
        project_tags = args.wandb_tags
        wandb.init(project=project, config=args, name = project_name, tags = project_tags)
    # load reward model TODO
    if args.use_clip:
        reward_clip = CLIPScore(model_or_name_path = "./checkpoints/clip-vit-large-patch14", device=device)
    if args.use_hps_v3:
        reward_clip = HPSv3RewardInferencer('./checkpoints/HPSv3/HPSv3_7B.yaml', "./checkpoints/HPSv3/HPSv3.safetensors", device)
    if args.use_dino:
        reward_dino = Dinov2Score(model_or_name_path = "./checkpoints/dinov2-base", device = device)

    reward_model = [reward_clip, reward_dino]
    ## 模型加载
    dit, vae = get_models(
        name=args.model_name,
        device=device,
    )

    # 冻结 vae
    vae.requires_grad_(False)
    dit.requires_grad_(False)

    dit = set_lora(dit, args.lora_rank, None, None, device)

    for name, param in dit.named_parameters():
        if "lora" in name.lower():  
            param.requires_grad = True
            # print(name)
        else:
            param.requires_grad = False

    # fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
    #     dit,
    #     args.fsdp_sharding_startegy,
    #     False,    #是否使用LoRA
    #     args.use_cpu_offload,
    #     args.master_weight_type,
    # )

    # dit = FSDP(dit, **fsdp_kwargs)

    # 冻结dit的初始参数
    # dit.requires_grad_(False)
    # 使用LoRA在dit模型上添加可训练的低秩适应层
    # dit = set_lora(dit, 512, None, None, device)      # check
    # dit.to(torch.float32)
    # 启用梯度检查点以节省显存

    # 使用训练模式
    dit.train()

    if args.gradient_checkpointing:
        # apply_fsdp_checkpointing(
        #     dit, no_split_modules, args.selective_checkpointing
        # )
        # dit.enable_gradient_checkpointing()
        dit.gradient_checkpointing = True

    noise_scheduler = None

    params_to_optimize = dit.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    ## optimizer and lr scheduler
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=args.adam_betas,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_eps,
    )
    init_steps = 0
    main_print(f"optimizer: {optimizer}")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )
    dataset = FluxPairedDataset_without_text(
        json_file=args.train_data_json,
        resolution_ref=args.resolution_ref,
        image_base = args.train_image_base
    )
    sampler = DistributedSampler(
            dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )
    train_dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(f"  Master weight dtype: {dit.parameters().__next__().dtype}")

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper_without_text(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,     
        args.train_sp_batch_size,    
    )

    step_times = deque(maxlen=100)

    if args.training_strategy == "part":
        grpo_states = GRPOTrainingStates(
            iters_per_group=args.iters_per_group,
            group_size=args.group_size,
            max_timesteps=args.sampling_steps-2,  # Because the max timestep index is args.sampling_steps - 2
            cur_timestep=0,
            cur_iter_in_group=0,
            sample_strategy=args.sample_strategy,
            prog_overlap=args.prog_overlap,
            prog_overlap_step=args.prog_overlap_step,
            max_iters_per_group=args.max_iters_per_group,
            min_iters_per_group=args.min_iters_per_group,
            roll_back=args.roll_back,
        )

    global_step = -1
    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch
        
        for step in range(init_steps+1, args.max_train_steps+1):
            global_step += 1
            start_time = time.time()
            if step % args.checkpointing_steps == 0:
                save_lora_my(dit, rank, args.project_dir,
                                step, epoch)

                dist.barrier()

            if args.training_strategy == "part":
                timesteps_train = grpo_states.get_current_timesteps()
                grpo_states.update_iteration()
            elif args.training_strategy == "all":
                timesteps_train = [ti for ti in range(args.sampling_steps)]
            
            main_print(f"Optimation Timesteps: {timesteps_train}",)

            
            print(f"[debug] allocated before: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")
            loss, grad_norm, means, reward_std = train_one_step(
                args,
                device, 
                dit,
                vae,
                reward_model,
                # processor,
                optimizer,
                lr_scheduler,
                loader,
                # noise_scheduler,
                args.max_grad_norm,
                timesteps_train,
                global_step,
            )

            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                if args.use_clip:
                    wandb.log(
                        {
                            "train_loss": loss,
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "step_time": step_time,
                            "avg_step_time": avg_step_time,
                            "grad_norm": grad_norm,
                            "clip_t": means[0].item(),
                            "dino": means[1].item(),
                            "clip_t_std": reward_std[0].item(),
                            "dino_std": reward_std[1].item()
                        },
                        step=step,
                    )
                else:
                    wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                        "hps_v3": means[0].item(),
                        "dino": means[1].item(),
                        "hps_v3_std": reward_std[0].item(),
                        "dino_std": reward_std[1].item(),
                    },
                    step=step,
                )

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, help="config yaml path", required=True)
    args = parser.parse_args()

    args_tuple = transformers.HfArgumentParser([TrainArgs]).parse_yaml_file(args.config)
    main(*args_tuple)
