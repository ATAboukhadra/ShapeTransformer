#!/bin/bash
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
--data_root /datasets/ho/ \
--batch_size 32 \
--num_workers 32 \
--output_folder /checkpoints/transformer_ho3d \
--epochs 20 \
--hdf5 \
--log_interval 1000 \
--window_size 21 \
--d_model 64 \
#--pretrained_model /checkpoints/transformer_ho3d/transformer_best.pkl \
