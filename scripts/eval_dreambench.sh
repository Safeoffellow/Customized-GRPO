#!/bin/bash

CONFIG_PATH=${1:-config/eval/eval_dreambench_grpo.yaml}
NUM_GPU=${2:-8}      # default 8 GPU

accelerate launch --num_processes $NUM_GPU inference/inference_dreambench.py --config $CONFIG_PATH
