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

def save_checkpoint(dit, rank, output_dir, step, epoch):
    main_print(f"--> saving checkpoint at step {step}")
    with FSDP.state_dict_type(
            dit,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = dit.state_dict()
    # todo move to get_state_dict
    if rank <= 0:
        save_dir = os.path.join(output_dir, f"checkpoint-{step}-{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        # save using safetensors
        weight_path = os.path.join(save_dir,
                                   "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        config_dict = dict(dit.config)
        if "dtype" in config_dict:
            del config_dict["dtype"]  # TODO
        config_path = os.path.join(save_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
    main_print(f"--> checkpoint saved at step {step}")