#!/bin/bash

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export USE_OPENMP=1  # prevents openblas to override OMP_NUM_THREADS

python main.py \
--data_root /data/HO3D/ \
--batch_size 32 \
--epochs 5 \
--hdf5 \
--window_size 21 \
--log_interval 100 \
--num_workers 0 \
# --output_folder checkpoints/transformer
