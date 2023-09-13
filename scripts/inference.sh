python infer_model.py \
--data_root /ds-av/public_datasets/arctic/td/sequential_resized_allocentric/ \
--meta_root dataset/arctic_objects \
--model_name shapethor \
--split val \
--batch_size 1 \
--window_size 1 \
--num_workers 4 \
--input_dim 538 \
--weights /data/checkpoints/thor_shape/model_14.pth \
--num_seqs 1 \
--visualize \
# --rcnn_path /data/checkpoints/thor/rcnn.pth \
