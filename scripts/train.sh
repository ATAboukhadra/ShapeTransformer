#!/bin/bash

python main.py \
--data_root /data/HO3D/ \
--batch_size 32 \
--epochs 5 \
--hdf5 \
--window_size 9 \
--log_interval 100 \
# --output_folder checkpoints/transformer
