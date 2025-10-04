#!/bin/bash

CONFIG_PATH=${1:-config/train/syncd_1w_grpo_vlm.yaml}
NUM_GPU=${2:-8}      # default 8 GPU

accelerate launch --num_processes $NUM_GPU train/train_grpo_vlm.py --config $CONFIG_PATH
