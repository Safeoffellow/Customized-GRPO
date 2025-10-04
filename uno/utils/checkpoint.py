import json
import os

import torch
import torch.distributed.checkpoint as dist_cp
from peft import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.default_planner import (DefaultLoadPlanner,
                                                          DefaultSavePlanner)
from torch.distributed.checkpoint.optimizer import \
    load_sharded_optimizer_state_dict
from torch.distributed.fsdp import (FullOptimStateDictConfig,
                                    FullStateDictConfig)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from uno.flux.logging_ import main_print

def save_checkpoint(transformer, rank, output_dir, step, epoch):
    main_print(f"--> saving checkpoint at step {step}")
    with FSDP.state_dict_type(
            transformer,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = transformer.state_dict()
    # todo move to get_state_dict
    if rank <= 0:
        save_dir = os.path.join(output_dir, f"checkpoint-{step}-{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        # save using safetensors
        weight_path = os.path.join(save_dir,
                                   "dit.safetensors")
        save_file(cpu_state, weight_path)
        config_dict = dict(transformer.config)
        if "dtype" in config_dict:
            del config_dict["dtype"]  # TODO
        config_path = os.path.join(save_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
    main_print(f"--> checkpoint saved at step {step}")

def save_lora_my(dit, rank, output_dir, step, epoch):

    if rank <= 0:
        main_print(f"--> saving checkpoint at step {step}")
        # save lora
        lora_rl_path = os.path.join(output_dir, 'ckpt', f'checkpoint-{step}-{epoch}')
        os.makedirs(lora_rl_path, exist_ok=True)
        # 1. 获取所有 LoRA 可训练参数
        state_dict = dit.state_dict()
        requires_grad_key = [k for k, v in dit.named_parameters() if v.requires_grad]
        lora_state_dict = {k: state_dict[k] for k in requires_grad_key}

        # 2. 保存为 safetensors 格式
        save_file(lora_state_dict, os.path.join(lora_rl_path, 'dit_lora.safetensors'))


        # del base_state_dict, base_converted_state_dict, ref_state_dict, ref_converted_state_dict
        main_print(f"--> checkpoint saved at step {step}, epoch {epoch}, in {lora_rl_path}")