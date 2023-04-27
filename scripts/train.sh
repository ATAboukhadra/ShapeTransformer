#!/bin/bash
pip install chumpy
pip install h5py
pip install trimesh
cd manopth
pip install .
cd ..

python main.py \
--data_root /datasets/ho/ \
--batch_size 128 \
--num_workers 32 \
--output_folder /checkpoints/transformer \
--epochs 5 \
--hdf5 \
--log_interval 1000 \
