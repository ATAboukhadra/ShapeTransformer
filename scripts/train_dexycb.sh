#!/bin/bash

python main.py \
--data_root /ds-av/public_datasets/DexYCB/td_s1_sequential/ \
--meta_root /data/DexYCB/ \
--batch_size 16 \
--epochs 5 \
--log_interval 1000 \
# --data_root /ds-av/public_datasets/DexYCB/td \
# --output_folder checkpoints/transformer
