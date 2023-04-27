#!/bin/bash
pip install chumpy
pip install h5py
pip install trimesh
cd manopth
pip install .
cd ..

python main.py \
--data_root /data/HO3D/ \
--batch_size 32 \
--epochs 5 \
--hdf5 \
--window_size 9 \
--log_interval 100 \
# --output_folder checkpoints/transformer
