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
import requests
import base64

from uno.reward.process_prompt import process_prompt
import re
import io
import json


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode("utf-8")

def pil_image_to_base64(decoded_image, format="PNG"):
    buffer = io.BytesIO()
    decoded_image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

def vlm_api(msg, max_retries=3, delay=2):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f'Your_Key',
        'Content-Type': 'application/json',
    }

    data = {
        "messages": msg,
        "model": "gpt-4o-0806" #
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            # print("Response:", response.json())
            # print("Response:", response)
            if response.status_code == 200:
                res = response.json()['choices'][0]['message']['content']
                if res:
                    return res  # 返回有效内容
                else:
                    retries += 1
            else:
                retries += 1

        except requests.exceptions.RequestException as e:
            retries += 1
            print(f"Request failed , error: {str(e)}")
        
        # 重试前的延迟
        time.sleep(delay)

    return "None"  # 请求失败时返回空字符串，避免返回 None


def save_tensor(tensor_dict, filename, save_path):
    if save_path is not None:
        torch.save(tensor_dict, os.path.join(save_path, filename))

# flux_step 是 扩散模型反向去噪过程中的单步执行器。它根据模型预测和当前噪声水平，计算下一个去噪后的潜在样本，并支持引入随机性和对数概率计算
def flux_step(
    model_output: torch.Tensor,   # 模型预测的噪声
    latents: torch.Tensor,  # 当前时间步的潜在样本
    eta: float,  # 随机性参数
    sigmas: torch.Tensor, # 噪声水平
    index: int,   # 当前时间步在sigmas中的索引
    prev_sample: torch.Tensor,  # 上一步采样的结果
    grpo: bool,    # 是否启用GRPO模式
    sde_solver: bool,   # 是否启用SDE求解模式
):
    # 获取当前时间步的噪声水平
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma

    # 计算下一时间步潜在样本的平均值(predictive mean)
    prev_sample_mean = latents + dsigma * model_output
    
    # 预测原始样本
    pred_original_sample = latents - sigma * model_output

    # 计算随机性相关的标准差
    delta_t = sigma - sigmas[index + 1]    # 当前去噪步骤中噪声水平的绝对下降量
    std_dev_t = eta * math.sqrt(delta_t)  
    
    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
        
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

# 时间步的非线形转换或偏移
def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)

logger = get_logger(__name__)

# 提供网络坐标ID
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
    # model = load_flow_model(name, device=device) # check
    model = load_flow_model_merge(
        name,
        device=device,
        local=False
    )
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae

def run_sample_step(
        args,
        z,    # 最开始是随机纯噪声
        progress_bar,   
        sigma_schedule,   
        dit,
        encoder_hidden_states, # T5输出的文本隐藏状态
        pooled_prompt_embeds,  # CLIP输出的池化嵌入
        text_ids,      # 文本的ID
        image_ids,     # 图像ID
        ref_img,
        ref_img_ids,
        grpo_sample,
    ):
    if grpo_sample:
        all_latents = [z] # 存储每个采样步骤的潜在样本 z，记录轨迹
        all_log_probs = []
        for i in progress_bar:  # Add progress bar
            B = encoder_hidden_states.shape[0]
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
            dit.eval()
            with torch.autocast("cuda", torch.bfloat16):   # ----check
            # with torch.autocast("cuda", torch.float32):
                
                # pred 模型预测的下一步轨迹
                pred= dit(
                    img=z, # 是否要加.to(weight_dtype)
                    img_ids = image_ids, # 是否要加.to(weight_dtype),
                    ref_img = ref_img,
                    ref_img_ids = ref_img_ids,
                    txt=encoder_hidden_states,
                    txt_ids = text_ids,
                    y = pooled_prompt_embeds,
                    timesteps=timesteps/1000,
                    guidance=torch.tensor(
                        [3.5],# 这里训练为1
                        device=z.device,
                        dtype=torch.bfloat16
                    ),
                    # joint_attention_kwargs=None,
                    # return_dict=False,
                ) # ----check
            z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original # 最终去噪后的、未被噪声扰动的原始图像潜在变量
        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
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
    # 设置为训练模式
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
                [3.5], # 这里训练为1
                device=latents.device,
                dtype=torch.bfloat16
            ),
            # joint_attention_kwargs=None,
            # return_dict=False,
        )[0]
    #  参数被传入了 `pre_latents`，这意味着 `flux_step` 不会自己随机生成 `prev_sample`，而是使用外部提供的 `pre_latents`。
    z, pred_original, log_prob = flux_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
    return log_prob

# 主要用于 从当前策略（即参考模型）生成一批图像（或其潜在表示），并收集强化学习训练所需的所有轨迹数据。它是一个数据收集（Rollout）函数
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
    # reward_model,
    prompts,
    ref_image_paths
):
    w, h = args.w, args.h
    sample_steps = args.sampling_steps 
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1) # sigma表示当前噪声强度
    
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )
    
    B = encoder_hidden_states.shape[0] # Dataloader中的batch_size
    SPATIAL_DOWNSAMPLE = 8 # VAE编码器下采样倍数
    IN_CHANNELS = 16    # 潜空间的通道数
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1   # 内部采样循环的“小”批次大小，这里设置为1，意味着逐个样本处理
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)   # 将总批次索引分成多个小批次

    all_latents = [] # 存储每个批次生成的完整潜在样本轨迹
    all_log_probs = [] # 存储每个批次生成的每一步对数概率
    all_rewards = []  # 存储每个批次生成图像的奖励值
    all_image_ids = [] # 存储每个批次对应的图像位置编码
    all_ref_image_ids = []
    all_ref_img = []

    # 如果使用相同随机初始噪声
    if args.init_same_noise:
        input_latents = torch.randn(
                (1, IN_CHANNELS, latent_h, latent_w),  #（b,c,h,w)
                device=device,
                dtype=torch.bfloat16,
            )

    # 循环处理每个小批次
    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx]
        batch_text_ids = text_ids[batch_idx]

        batch_ref_img = [r[batch_idx] for r in ref_img]     # ref_img: list, 每个元素 shape [B_total, ...]
        batch_ref_img_ids = [ids[batch_idx] for ids in ref_img_ids] # ref_img_ids: 同理 

        batch_prompt = [prompts[i] for i in batch_idx]
        batch_img_ids = img_ids[batch_idx]
        batch_ref_img_paths = [ref_image_paths[i] for i in batch_idx]
        
        if not args.init_same_noise:
            input_latents = torch.randn(
                    (len(batch_idx), IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
                    device=device,
                    dtype=torch.bfloat16,
                )
        input_latents_new = pack_latents(input_latents, len(batch_idx), IN_CHANNELS, latent_h, latent_w)
        # image_ids = prepare_latent_image_ids(len(batch_idx), latent_h // 2, latent_w // 2, device, torch.bfloat16)
        grpo_sample=True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")

        with torch.no_grad():
            # 这里的batch_latents是每一步的轨迹预测，batch_log_probs是每一步的概率预测
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
            )
        
        all_image_ids.append(batch_img_ids)
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        all_ref_img.append(batch_ref_img)
        all_ref_image_ids.append(batch_ref_img_ids)
        
        # 启用 VAE 的 tiling 模式，以处理大图像时节省显存
        # vae.enable_tiling()
        
        image_processor = VaeImageProcessor(16)  
        rank = int(os.environ["RANK"])
        image_base = args.temp_image_path
        os.makedirs(image_base, exist_ok=True) 
        save_path = image_base
        # save_tensor({'latents': latents}, f'flux_latents_{rank}_{index}.pt', save_path)
        # save_tensor({'batch_img_ids': batch_img_ids}, f'batch_img_ids_{rank}_{index}.pt', save_path)
        
        with torch.inference_mode(): 
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents = unpack_latents(latents, h, w, 8)
                # latents = (latents / 0.3611) + 0.1159

                # image = vae.decode(latents, return_dict=False)[0]       -----check-----
                image = vae.decode(latents)[0]

                # 保证4维
                if image.dim() == 3:
                    image = image.unsqueeze(0)

                decoded_image = image_processor.postprocess(image)

        name = batch_ref_img_paths[0][0].split("/")[-2]
        image_temp = os.path.join(image_base, name)
        os.makedirs(image_temp, exist_ok=True)
        image_path = os.path.join(image_temp, f"flux_{rank}_{index}.png")
        print(batch_ref_img_paths)

        # 检查 decoded_image 类型
        if isinstance(decoded_image, list):
            if not os.path.exists(image_path):  # 如果文件不存在才保存
                decoded_image[0].save(image_path)
                # print("Save list Successfully:", image_path)
        else:
            if not os.path.exists(image_path):  # 如果文件不存在才保存
                decoded_image.save(image_path)
                # print("Save single Successfully:", image_path)

        # for ref_idx, ref_path in enumerate(batch_ref_img_paths[0]):
        #     ref_image_temp = os.path.join(image_temp, f"ref_{ref_idx}.png")
        #     if not os.path.exists(ref_image_temp):  # 如果文件不存在才保存
        #         temp_img = Image.open(ref_path)
        #         temp_img.save(ref_image_temp)

        reward_temp = []
        # 计算奖励

        # prompt following
        prompt_following = process_prompt(batch_prompt[0], args.task_following)
        system_prompt = "You are an image analysis assistant."

        gen_img_base64 = pil_image_to_base64(decoded_image[0])

        msg_pf = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": [
                {"type": "text", "text": prompt_following},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{gen_img_base64}"
                    }
                }
                ] 
            }
        ]
        answer_pf = vlm_api(msg_pf)
        print("Answer:", answer_pf)
        s_pf = re.search(r'Score:\s*([0-9\.]+)', answer_pf)
        if s_pf:
            pf_score = float(s_pf.group(1))
        else:
            pf_score = float(0)
        pf_score = torch.tensor(pf_score, dtype=torch.float32, device = device)  # 转为1D tensor
        reward_temp.append(pf_score)

        # concept  
        ref_img_path_temp = batch_ref_img_paths[0][0].replace(".png", "_seg.png")          # list change
        concept_preservation = process_prompt("", args.task_concept)

        if os.path.exists(ref_img_path_temp):
            ref_img_path_temp = ref_img_path_temp
        else:
            ref_img_path_temp = batch_ref_img_paths[0][0]

        ref_img_base64 = image_to_base64(ref_img_path_temp)

        msg_cp = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": [
                {"type": "text", "text": concept_preservation},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{ref_img_base64}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{gen_img_base64}"
                    }
                }
                ] 
            }
        ]
        answer_cp = vlm_api(msg_cp)
        print("Answer:", answer_cp)
        s_cp = re.search(r'Score:\s*([0-9\.]+)', answer_cp)
        if s_cp:
            cp_score = float(s_cp.group(1))
        else:
            cp_score = float(0)
        cp_score = torch.tensor(cp_score, dtype=torch.float32, device = device)
        reward_temp.append(cp_score)

        reward_temp = torch.stack(reward_temp) # [n_reward_type] ----check----
        all_rewards.append(reward_temp)
    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)

    # all_rewards = torch.cat(all_rewards, dim=0)
    all_rewards = torch.stack(all_rewards, dim=0)    # [N, n_reward_type] ---check
    # all_image_ids = torch.stack(all_image_ids, dim=0)
    all_image_ids = torch.cat(all_image_ids, dim=0)   # --debug

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
    # reward_model,
    optimizer,
    lr_scheduler,
    loader,
    max_grad_norm,
):
    # 用于累计梯度损失
    total_loss = 0.0
    optimizer.zero_grad()

    prompts, ref_imgs, ref_img_paths, encoder_hidden_states, pool_prompt_embed = next(loader)

    ref_imgs_num = len(ref_imgs)
    # if isinstance(prompts, list):
    #     print(f"Length of prompts list: {len(prompts)}")
    #     if prompts: print(f"Type of first element in prompts: {type(prompts[0])}")
    # print(f"Type of ref_imgs: {type(ref_imgs)}")
    # if isinstance(ref_imgs, list):
    #     print(f"Length of ref_imgs list: {len(ref_imgs)}")
    #     if ref_imgs: print(f"Type of first element in ref_imgs: {type(ref_imgs[0])}, Shape: {ref_imgs[0].shape}")
    # else:
    #     print(f"Ref_imgs is not a list. Type: {type(ref_imgs)}")

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
        
        # 扩充 size变为 (B * num_generations, *)
        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        pooled_prompt_embeds = repeat_tensor(pooled_prompt_embeds)
        text_ids = repeat_tensor(text_ids)

        # ref_img = repeat_tensor(ref_img)
        # ref_img_ids = repeat_tensor(ref_img_ids)
        # ref_img 和 ref_img_ids 是 list of tensors, 需分开处理每个slot
        if isinstance(ref_img, (list, tuple)):
            ref_img = [repeat_tensor(t) for t in ref_img]
        else:  # 正常情况下应为list
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
            # reward_model,
            prompts,
            ref_image_paths
        )
    # print("*********************REWARD:", reward)
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

    # gathered_reward = gather_tensor(samples["rewards"])
    # if dist.get_rank()==0:
    #     main_print(f"gathered dino and clip: {gathered_reward.mean().item():.4f}")
    #     with open('/data/oss_bucket_0/ziwei/hps_reward.txt', 'a') as f: 
    #     f.write(f"{gathered_reward.mean().item()}\n")

    gathered_reward = gather_tensor(samples["rewards"])     # [world*N, 2]
    means = None

    if dist.get_rank() == 0:
        means = gathered_reward.mean(dim=0)
        main_print(f"gathered clip: {means[0].item():.4f}, gathered dino: {means[1].item():.4f}")
        # with open('/data/oss_bucket_0/ziwei/hps_reward.txt', 'a') as f: 
        #     # 分别写clip和dino均值
        #     f.write(f"{means[0].item()} {means[1].item()}\n")

    if args.use_group:
        clip_weight = args.clip_weight
        dino_weight = args.dino_weight
        n_original_batch = len(samples["rewards"]) // args.num_generations
        
        advantages = torch.zeros(samples["rewards"].shape[0], device=samples["rewards"].device)

        for i in range(n_original_batch):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_reward_clip = samples["rewards"][start_idx:end_idx, 0]
            group_reward_dino = samples["rewards"][start_idx:end_idx, 1]
            group_mean_clip = group_reward_clip.mean()
            group_mean_dino = group_reward_dino.mean()
            group_std_clip = group_reward_clip.std() + 1e-8
            group_std_dino = group_reward_dino.std() + 1e-8
            advantages[start_idx:end_idx] = (
                (group_reward_clip - group_mean_clip) / group_std_clip * clip_weight
            + (group_reward_dino - group_mean_dino) / group_std_dino * dino_weight
            )
        
        samples["advantages"] = advantages
    else:
        advantages = (samples["rewards"] - gathered_reward.mean())/(gathered_reward.std()+1e-8)
        samples["advantages"] = advantages

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

    train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)


    print("Sample_List Len:", len(samples_batched_list))
    for i,sample in list(enumerate(samples_batched_list)):
        print(f"==== grpo one step on sample {i}")
        for _ in range(train_timesteps):
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
                perms[i][_],
                sigma_schedule,
            )

            advantages = torch.clamp(
                sample["advantages"],
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
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * train_timesteps)

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
        
        if (i+1)%args.gradient_accumulation_steps==0:
            grad_norm = clip_grad_norm_(
                filter(lambda p: p.requires_grad and p.grad is not None, dit.parameters()),
                max_grad_norm
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if dist.get_rank()%8==0:
            print("ratio", ratio)
            print("advantage", sample["advantages"].item())
            print("final loss", loss.item())
        dist.barrier()
    return total_loss, grad_norm.item(), means

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

    use_clip: bool = True
    use_dino: bool = True
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

    # other pa
    sp_size: int = 1
    train_sp_batch_size: int = 1
    use_cpu_offload: bool = False
    master_weight_type: str = "fp32"
    w: int = 512
    h: int = 512
    temp_image_path: str = "./Single_syncd_1w_grpo/" + now_name + "/images"
    task_concept: str = "concept"
    task_following: str = "text_cot"

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

    if args.seed is not None:
        set_seed(args.seed + rank)

    # Handle the repository creation
    if rank <= 0 and args.project_dir is not None:
        os.makedirs(args.project_dir, exist_ok=True)

    dit, vae = get_models(
        name=args.model_name,
        device=device,
    )

    vae.requires_grad_(False)
    dit.requires_grad_(False)

    dit = set_lora(dit, args.lora_rank, None, None, device)

    for name, param in dit.named_parameters():
        if "lora" in name.lower():  
            param.requires_grad = True
            # print(name)
        else:
            param.requires_grad = False

    dit.train()

    if args.gradient_checkpointing:
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
    if rank <= 0:
        project = args.wandb_project_name
        project_name = args.wandb_run_name
        project_tags = args.wandb_tags
        wandb.init(project=project, config=args, name = project_name, tags = project_tags)

    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        disable=local_rank > 0,
    )

    # 数据加载器序列并行包装
    loader = sp_parallel_dataloader_wrapper_without_text(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,     
        args.train_sp_batch_size,    
    )

    step_times = deque(maxlen=100)

    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch
        
        for step in range(init_steps+1, args.max_train_steps+1):
            start_time = time.time()
            if step % args.checkpointing_steps == 0:
                save_lora_my(dit, rank, args.project_dir,
                                step, epoch)

                dist.barrier()
            print(f"[debug] allocated before: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")
            loss, grad_norm, means = train_one_step(
                args,
                device, 
                dit,
                vae,
                # reward_model,
                # processor,
                optimizer,
                lr_scheduler,
                loader,
                # noise_scheduler,
                args.max_grad_norm,
                # preprocess_val,
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
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                        "following": means[0].item(),
                        "concept": means[1].item()
                    },
                    step=step,
                )
            print(f"[debug] allocated after: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, help="config yaml path", required=True)
    args = parser.parse_args()

    args_tuple = transformers.HfArgumentParser([TrainArgs]).parse_yaml_file(args.config)
    main(*args_tuple)
