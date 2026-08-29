#!/usr/bin/env bash
python infer.py \
  --video /comp_robot/lxr/nvidia/Cosmos3-Edge/assets/example_action_id_av_0_input.mp4 \
  --question "When does the robot move an object?" \
  --max-frames 8
