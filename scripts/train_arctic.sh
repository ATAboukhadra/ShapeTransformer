#!/bin/bash

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export USE_OPENMP=1  # prevents openblas to override OMP_NUM_THREADS

python train_arctic.py \
--data_root '/ds-av/public_datasets/arctic/td_p1_sequential_nocropped/' \
--meta_root 'dataset/arctic_objects' \
--output_folder checkpoints/Stohrmer \
--batch_size 1 \
--log_interval 100 \
--epoch 1 \
--window_size 9 \
--num_workers 0 \
# --causal \

