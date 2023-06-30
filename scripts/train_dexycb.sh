#!/bin/bash
#pip freeze
pip install --no-deps -e datapipes
pip install chumpy
pip install h5py
pip install trimesh
cd manopth
pip install .
cd ..

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export USE_OPENMP=1  # prevents openblas to override OMP_NUM_THREADS

python main.py \
--data_root /ds-av/public_datasets/DexYCB/td_s1_sequential/ \
--meta_root /datasets/DexYCB/ \
--output_folder /checkpoints/st_graformer/ \
--batch_size 128 \
--epochs 5 \
--log_interval 1000 \
--num_workers 32 \
--window_size 21 \
# --data_root /ds-av/public_datasets/DexYCB/td \
# --output_folder checkpoints/transformer
