#!/bin/bash

# 启动 textual_inversion_training 训练
accelerate launch training/textual_inversion.py \
  --pretrained_model_name_or_path /home/jenny/fine_tuning/models/stable-diffusion-v1-5 \
  --train_data_dir /home/jenny/fine_tuning/trainsets/anime-style/instance \
  --learnable_property style \
  --placeholder_token "<anime-style>" \
  --initializer_token "style" \
  --output_dir /home/jenny/fine_tuning/outputs/textual_inversion_anime \
  --resolution 512 \
  --train_batch_size 1 \
  --max_train_steps 800 \
  --learning_rate 5e-4 \
  --scale_lr \
  --mixed_precision fp16
