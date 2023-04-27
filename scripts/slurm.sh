srun -K --gpus=1 --time=3-0 \
--container-mounts=/netscratch/aboukhadra/checkpoints:/checkpoints/,/netscratch/aboukhadra/datasets:/datasets/,/ds-av/public_datasets/honotate_v3/raw:/home2/HO3D_v3,"`pwd`":"`pwd`" \
--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.07-py3.sqsh \
--container-workdir="`pwd`" --cpus-per-gpu=32 --mem=200G --partition=A100-40GB \
./train.sh
