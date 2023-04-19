#!/bin/bash

python3.8 visualize.py \
--data_root /data/DexYCB/td_s1_sequential/ \
--meta_root /data/DexYCB/ \
--batch_size 1 \
--pretrained_model checkpoints/transformer/transformer_1.pkl
# --hdf5 \
# --data_root /data/HO3D/ \
# --output_folder checkpoints/transformer
