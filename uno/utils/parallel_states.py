#This code file is from [https://github.com/hao-ai-lab/FastVideo], which is licensed under Apache License 2.0.

import os

from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

class COMM_INFO:

    def __init__(self):
        self.group = None
        self.sp_size = 1
        self.global_rank = 0
        self.rank_within_group = 0
        self.group_id = 0


nccl_info = COMM_INFO()
_SEQUENCE_PARALLEL_STATE = False


def initialize_sequence_parallel_state(sequence_parallel_size):
    global _SEQUENCE_PARALLEL_STATE
    if sequence_parallel_size > 1:
        _SEQUENCE_PARALLEL_STATE = True
        initialize_sequence_parallel_group(sequence_parallel_size)
    else:
        nccl_info.sp_size = 1
        nccl_info.global_rank = int(os.getenv("RANK", "0"))
        nccl_info.rank_within_group = 0
        nccl_info.group_id = int(os.getenv("RANK", "0"))


def set_sequence_parallel_state(state):
    global _SEQUENCE_PARALLEL_STATE
    _SEQUENCE_PARALLEL_STATE = state


def get_sequence_parallel_state():
    return _SEQUENCE_PARALLEL_STATE


def initialize_sequence_parallel_group(sequence_parallel_size):
    """Initialize the sequence parallel group."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    assert (
        world_size % sequence_parallel_size == 0
    ), "world_size must be divisible by sequence_parallel_size, but got world_size: {}, sequence_parallel_size: {}".format(
        world_size, sequence_parallel_size)
    nccl_info.sp_size = sequence_parallel_size
    nccl_info.global_rank = rank
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size,
                      (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            nccl_info.group = group
            nccl_info.rank_within_group = rank - i * sequence_parallel_size
            nccl_info.group_id = i


def destroy_sequence_parallel_group():
    """Destroy the sequence parallel group."""
    dist.destroy_process_group()

# def sp_parallel_dataloader_wrapper(
#     dataloader, device, train_batch_size, sp_size, train_sp_batch_size
# ):
#     while True:
#         for data_item in dataloader:
#             prompts, ref_imgs = data_item
#             frame = 19
#             if frame == 1:
#                 yield prompts, ref_imgs
#             else:
#                 prompts, ref_imgs = prepare_sequence_parallel_data(
#                     prompts, ref_imgs
#                 )
#                 assert (
#                     train_batch_size * sp_size >= train_sp_batch_size
#                 ), "train_batch_size * sp_size should be greater than train_sp_batch_size"
#                 for iter in range(train_batch_size * sp_size // train_sp_batch_size):
#                     yield (
#                             prompts,
#                             ref_imgs
#                       )

def sp_parallel_dataloader_wrapper(
    dataloader, device, train_batch_size, sp_size, train_sp_batch_size
):
    while True:
        for data_item in dataloader:
            prompts = data_item["text"]
            ref_imgs = data_item["ref_imgs"]
            ref_img_paths = data_item["ref_img_paths"]
            print(f"DEBUG (wrapper): After unpacking, type of prompts: {type(prompts)}")
            print(f"DEBUG (wrapper): After unpacking, type of ref_imgs: {type(ref_imgs)}") # 应该是一个列表或张量
            yield prompts, ref_imgs, ref_img_paths

def sp_parallel_dataloader_wrapper_without_text(
    dataloader, device, train_batch_size, sp_size, train_sp_batch_size
):
    while True:
        for data_item in dataloader:
            prompts = data_item["text"]
            ids = data_item["ids"]
            ref_imgs = data_item["ref_imgs"]
            ref_img_paths = data_item["ref_img_paths"]
            encoder_hidden_states = data_item["encoder_hidden_states"]
            pool_prompt_embed = data_item["pool_prompt_embed"]
            print(f"DEBUG (wrapper): After unpacking, type of prompts: {type(prompts)}")
            print(f"DEBUG (wrapper): After unpacking, type of ref_imgs: {type(ref_imgs)}") # 应该是一个列表或张量
            yield prompts, ids, ref_imgs, ref_img_paths, encoder_hidden_states, pool_prompt_embed