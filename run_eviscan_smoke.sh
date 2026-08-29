#!/usr/bin/env bash
set -euo pipefail
cd /comp_robot/lxr/EviScan
source /home/liuxiangrui/miniconda3/etc/profile.d/conda.sh
conda activate timescope
export CUDA_VISIBLE_DEVICES=0
python infer.py \
  --video /comp_robot/lxr/nvidia/Cosmos3-Edge/assets/example_action_id_av_0_input.mp4 \
  --question "When does the robot move an object?" \
  --max-frames 8
