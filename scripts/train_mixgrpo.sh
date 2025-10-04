#!/bin/bash

CONFIG_PATH=${1:-config/train/syncd_1w_customized_grpo.yaml}
NUM_GPU=${2:-8}      # default 8 GPU

accelerate launch --num_processes $NUM_GPU train/train_customized_grpo.py --config $CONFIG_PATH
