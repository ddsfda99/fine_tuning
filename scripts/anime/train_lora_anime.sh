#!/bin/bash
accelerate launch training/train_dreambooth_lora.py \
  --pretrained_model_name_or_path /home/jenny/fine_tuning/models/stable-diffusion-v1-5 \
  --instance_data_dir /home/jenny/fine_tuning/trainsets/anime-style/instance \
  --output_dir /home/jenny/fine_tuning/outputs/lora_anime \
  --instance_prompt "<anime-style>" \
  --resolution 512 \
  --train_batch_size 1 \
  --max_train_steps 200 \
  --mixed_precision fp16
