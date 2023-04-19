#!/bin/bash
pip install h5py

python main.py \
--data_root /datasets/ho/ \
--batch_size 128 \
--num_workers 32 \
--epochs 1 \
--hdf5 \
--output_folder /checkpoints/transformer
