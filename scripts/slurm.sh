srun -K --gpus=1 --time=3-0 \
--container-mounts=/netscratch/aboukhadra/checkpoints:/checkpoints/,/netscratch/aboukhadra/datasets:/datasets/,/netscratch/aboukhadra/datasets/HO3D_v3/:/home2/HO3D_v3,/ds-av/public_datasets/DexYCB/:/ds-av/public_datasets/DexYCB/,"`pwd`":"`pwd`",/home/aboukhadra/datapipes:"`pwd`/datapipes" \
--container-image=/netscratch/aboukhadra/pytorch3d.sqsh \
--container-workdir="`pwd`" --cpus-per-task=32 --mem=300G --partition=A100-80GB \
./scripts/train_cluster.sh
