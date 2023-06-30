#!/bin/bash

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export USE_OPENMP=1  # prevents openblas to override OMP_NUM_THREADS

python main.py \
--data_root /ds-av/public_datasets/DexYCB/td_s1_sequential/ \
--meta_root /data/DexYCB/ \
--batch_size 16 \
--epochs 5 \
--log_interval 10 \
--window_size 9 \
# --data_root /ds-av/public_datasets/DexYCB/td \
# --output_folder checkpoints/transformer
