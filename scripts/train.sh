#!/bin/bash

python main.py \
--data_root /data/HO3D/ \
--batch_size 8 \
--epochs 5 \
--hdf5 \
--log_interval 1000 \
# --output_folder checkpoints/transformer
