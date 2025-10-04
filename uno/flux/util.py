# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass

import torch
import json
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file as load_sft

from .model import Flux, FluxParams
from .modules.autoencoder import AutoEncoder, AutoEncoderParams
from .modules.conditioner import HFEmbedder
# from peft.utils.other import fsdp_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel, FluxTransformerBlock, FluxSingleTransformerBlock
import functools
from functools import partial
import re
from uno.flux.modules.layers import DoubleStreamBlockLoraProcessor, SingleStreamBlockLoraProcessor

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

def load_model(ckpt, device='cpu'):
    print(f"DEBUG: load_model Inside load_ae - received 'device' argument: {device}")
    print(f"DEBUG: load_model Type of 'device': {type(device)}")
    print(f"DEBUG: load_model String representation of 'device' for safetensors: {str(device)}")
    if ckpt.endswith('safetensors'):
        from safetensors import safe_open
        pl_sd = {}
        with safe_open(ckpt, framework="pt", device=device) as f:
            for k in f.keys():
                pl_sd[k] = f.get_tensor(k)
    else:
        pl_sd = torch.load(ckpt, map_location=device)
    return pl_sd

def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def get_lora_rank(checkpoint):
    for k in checkpoint.keys():
        if k.endswith(".down.weight"):
            return checkpoint[k].shape[0]

def load_checkpoint(local_path, repo_id, name):
    if local_path is not None:
        if '.safetensors' in local_path:
            print(f"Loading .safetensors checkpoint from {local_path}")
            checkpoint = load_safetensors(local_path)
        else:
            print(f"Loading checkpoint from {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu')
    elif repo_id is not None and name is not None:
        print(f"Loading checkpoint {name} from repo id {repo_id}")
        checkpoint = load_from_repo_id(repo_id, name)
    else:
        raise ValueError(
            "LOADING ERROR: you must specify local_path or repo_id with name in HF to download"
        )
    return checkpoint


def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None
    repo_id_ae: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path="./checkpoints/FLUX.1_dev/flux1-dev.safetensors",
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path="./checkpoints/FLUX.1_dev/ae.safetensors",
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fp8": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_FP8"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-uno-fused": ModelSpec(
        repo_id="model.alimama_aigc_algo.FLUX.1-dev/version=uno/ckpt_id=lora_fused_1.0",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path="./checkpoints/UNO/merge_dit.safetensors",
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path="./checkpoints/FLUX.1_dev/ae.safetensors",
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    )
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))

def load_from_repo_id(repo_id, checkpoint_name):
    ckpt_path = hf_hub_download(repo_id, checkpoint_name)
    sd = load_sft(ckpt_path, device='cpu')
    return sd

def load_flow_model(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)
    
    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params).to(torch.bfloat16)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_model(ckpt_path, device=str(device))
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return model

def load_flow_model_merge(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")

    ckpt_path = configs[name].ckpt_path

    print("Load_flow_model_merge CKPT_PATH:", ckpt_path)
    # if (
    #     ckpt_path is None
    #     and configs[name].repo_id is not None
    #     and configs[name].repo_flow is not None
    #     and hf_download
    # ):
    #     ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)
    
    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params).to(torch.bfloat16)

    if isinstance(device, int):
        device = str(device)
    
    if "cuda" not in str(device):
        device = f"cuda:{device}"

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_model(ckpt_path, device=str(device))
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return model

def load_flow_model_only_lora(
    name: str,
    device: str | torch.device = "cuda",
    hf_download: bool = False,
    lora_rank: int = 16,
    use_fp8: bool = False,
):
    print(f"DEBUG: load_model Inside only lora - received 'device' argument: {device}")
    print(f"DEBUG: load_model Type of 'device': {type(device)}")
    print(f"DEBUG: load_model String representation of 'device' for safetensors: {str(device)}")
    # Loading Flux
    print("Init model")

    if isinstance(device, int):
        device = str(device)
    
    if "cuda" not in device:
        device = f"cuda:{device}"

    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow.replace("sft", "safetensors"))
    
    if hf_download:
        try:
            lora_ckpt_path = hf_hub_download("bytedance-research/UNO", "dit_lora.safetensors")
        except:
            lora_ckpt_path = os.environ.get("LORA", None)
    else:
        lora_ckpt_path = "./checkpoints/UNO/dit_lora.safetensors"

    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params)


    model = set_lora(model, lora_rank, device="meta" if lora_ckpt_path is not None else device)

    if ckpt_path is not None:
        print("Loading lora")
        lora_sd = load_sft(lora_ckpt_path, device=str(device)) if lora_ckpt_path.endswith("safetensors")\
            else torch.load(lora_ckpt_path, map_location='cpu')
        
        print("Loading main checkpoint")
        # load_sft doesn't support torch.device

        if ckpt_path.endswith('safetensors'):
            if use_fp8:
                print(
                    "####\n"
                    "We are in fp8 mode right now, since the fp8 checkpoint of XLabs-AI/flux-dev-fp8 seems broken\n"
                    "we convert the fp8 checkpoint on flight from bf16 checkpoint\n"
                    "If your storage is constrained"
                    "you can save the fp8 checkpoint and replace the bf16 checkpoint by yourself\n"
                )
                sd = load_sft(ckpt_path, device="cpu")
                sd = {k: v.to(dtype=torch.float8_e4m3fn, device=device) for k, v in sd.items()}
            else:
                sd = load_sft(ckpt_path, device=str(device))
            
            sd.update(lora_sd)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        else:
            dit_state = torch.load(ckpt_path, map_location='cpu')
            sd = {}
            for k in dit_state.keys():
                sd[k.replace('module.','')] = dit_state[k]
            sd.update(lora_sd)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            model.to(str(device))
        print_load_warning(missing, unexpected)
    return model

def load_flow_model_only_lora_fuse(
    name: str,
    device: str | torch.device = "cuda",
    hf_download: bool = False,
    lora_rank: int = 16,
    use_fp8: bool = False,
    lora_path: str = None,
):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow.replace("sft", "safetensors"))
    
    if hf_download:
        try:
            lora_ckpt_path = hf_hub_download("bytedance-research/UNO", "dit_lora.safetensors")
        except:
            lora_ckpt_path = os.environ.get("LORA", None)
    else:
        lora_ckpt_path = lora_path

    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params)


    model = set_lora(model, lora_rank, device="meta" if lora_ckpt_path is not None else device)

    if ckpt_path is not None:
        print("Loading lora")
        lora_sd = load_sft(lora_ckpt_path, device=str(device)) if lora_ckpt_path.endswith("safetensors")\
            else torch.load(lora_ckpt_path, map_location='cpu')
        
        print("Loading main checkpoint")
        # load_sft doesn't support torch.device

        if ckpt_path.endswith('safetensors'):
            if use_fp8:
                print(
                    "####\n"
                    "We are in fp8 mode right now, since the fp8 checkpoint of XLabs-AI/flux-dev-fp8 seems broken\n"
                    "we convert the fp8 checkpoint on flight from bf16 checkpoint\n"
                    "If your storage is constrained"
                    "you can save the fp8 checkpoint and replace the bf16 checkpoint by yourself\n"
                )
                sd = load_sft(ckpt_path, device="cpu")
                sd = {k: v.to(dtype=torch.float8_e4m3fn, device=device) for k, v in sd.items()}
            else:
                sd = load_sft(ckpt_path, device=str(device))
            
            sd.update(lora_sd)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        else:
            dit_state = torch.load(ckpt_path, map_location='cpu')
            sd = {}
            for k in dit_state.keys():
                sd[k.replace('module.','')] = dit_state[k]
            sd.update(lora_sd)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            model.to(str(device))
        print_load_warning(missing, unexpected)
    return model


def set_lora(
    model: Flux,
    lora_rank: int,
    double_blocks_indices: list[int] | None = None,
    single_blocks_indices: list[int] | None = None,
    device: str | torch.device = "cpu",
) -> Flux:
    double_blocks_indices = list(range(model.params.depth)) if double_blocks_indices is None else double_blocks_indices
    single_blocks_indices = list(range(model.params.depth_single_blocks)) if single_blocks_indices is None \
                            else single_blocks_indices
    
    lora_attn_procs = {}
    with torch.device(device):
        for name, attn_processor in  model.attn_processors.items():
            # 判断属于哪一层 如果 name 是 "double_blocks.5.attn.q"，那么 match.group(1) 将会是 "5"
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))

            if name.startswith("double_blocks") and layer_index in double_blocks_indices:
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=model.params.hidden_size, rank=lora_rank)
            elif name.startswith("single_blocks") and layer_index in single_blocks_indices:
                lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=model.params.hidden_size, rank=lora_rank)
            else:
                lora_attn_procs[name] = attn_processor
    model.set_attn_processor(lora_attn_procs)
    return model


def load_flow_model_quintized(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    from optimum.quanto import requantize
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)
    # json_path = hf_hub_download(configs[name].repo_id, 'flux_dev_quantization_map.json')


    model = Flux(configs[name].params).to(torch.bfloat16)

    print("Loading checkpoint")
    # load_sft doesn't support torch.device
    sd = load_sft(ckpt_path, device='cpu')
    sd = {k: v.to(dtype=torch.float8_e4m3fn, device=device) for k, v in sd.items()}
    model.load_state_dict(sd, assign=True)
    return model
    with open(json_path, "r") as f:
        quantization_map = json.load(f)
    print("Start a quantization process...")
    requantize(model, sd, quantization_map, device=device)
    print("Model is quantized!")
    return model

def load_t5(device: str | torch.device = "cuda", max_length: int = 512, local: bool = False) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    version = "./checkpoints/xflux_text_encoders"
    return HFEmbedder(version, max_length=max_length, torch_dtype=torch.bfloat16).to(device)

def load_clip(device: str | torch.device = "cuda", local: bool = False) -> HFEmbedder:

    version = "./checkpoints/clip-vit-large-patch14"
    return HFEmbedder(version, max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True, local: bool = False) -> AutoEncoder:

    ckpt_path = configs[name].ae_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id_ae, configs[name].repo_ae)


    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        print(f"DEBUG: Inside load_ae - received 'device' argument: {device}")
        print(f"DEBUG: Type of 'device': {type(device)}")
        print(f"DEBUG: String representation of 'device' for safetensors: {str(device)}")
        print(f"Loading AE on device: {str(device)}")
        if isinstance(device, int):
            device = "cuda:" + str(device)
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return ae


# FSDP
def fsdp_auto_wrap_policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
    # 只包 LoRA module (根据名字或类型筛选)
    return any("lora" in n.lower() for n, _ in module.named_parameters(recurse=False))


def get_mixed_precision(master_weight_type="fp32"):
    weight_type = torch.float32 if master_weight_type == "fp32" else torch.bfloat16
    mixed_precision = MixedPrecision(
        param_dtype=weight_type,
        # Gradient communication precision.
        reduce_dtype=weight_type,
        # Buffer precision.
        buffer_dtype=weight_type,
        cast_forward_inputs=False,
    )
    return mixed_precision

def get_dit_fsdp_kwargs(
    transformer,
    sharding_strategy,
    use_lora=False,
    cpu_offload=False,
    master_weight_type="fp32",
):
    # ---1. 获取不需要分片的模块，这里返回了(FluxTransformerBlock, FluxSingleTransformerBlock)
    no_split_modules = (FluxTransformerBlock, FluxSingleTransformerBlock)

    # ---2. 定义自动包装策略

    if use_lora:
        auto_wrap_policy = fsdp_auto_wrap_policy
    else:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=no_split_modules,
        )
    
    # we use float32 for fsdp but autocast during training
    mixed_precision = get_mixed_precision(master_weight_type)

    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "hybrid_full":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    device_id = torch.cuda.current_device()
    cpu_offload = (torch.distributed.fsdp.CPUOffload(
        offload_params=True) if cpu_offload else None)
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        "device_id": device_id,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
    }

    # Add LoRA-specific settings when LoRA is enabled
    if use_lora:
        fsdp_kwargs.update({
            "use_orig_params": False,  # Required for LoRA memory savings
            "sync_module_states": True,
        })

    return fsdp_kwargs, no_split_modules

def apply_fsdp_checkpointing(model, no_split_modules, p=1):
    # https://github.com/foundation-model-stack/fms-fsdp/blob/408c7516d69ea9b6bcd4c0f5efab26c0f64b3c2d/fms_fsdp/policies/ac_handler.py#L16
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print("--> applying fdsp activation checkpointing...")
    block_idx = 0
    cut_off = 1 / 2
    # when passing p as a fraction number (e.g. 1/3), it will be interpreted
    # as a string in argv, thus we need eval("1/3") here for fractions.
    p = eval(p) if isinstance(p, str) else p

    def selective_checkpointing(submodule):
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, no_split_modules):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )

def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))