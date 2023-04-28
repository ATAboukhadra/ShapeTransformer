import time
import h5py
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import torch
import itertools
from dataset.HO3D_dataset import Dataset
from dataset.DexYCB_pipeline import create_pipe
from models.poseformer import PoseFormer, PoseGraFormer
from models.graformer import GraFormer
from utils import mpjpe, AverageMeter, parse_args, initialize_masks, create_logger
import cProfile
import pstats

args = parse_args()

if not os.path.exists(args.output_folder): os.mkdir(args.output_folder)
logger = create_logger(args.output_folder)

logger.info(f'combination: {args.causal, args.window_size, args.skip}')

# HO3D
if 'HO3D' in args.data_root or 'ho' in args.data_root:
    trainset = Dataset(args.data_root, T=args.window_size, skip=args.skip, causal=args.causal, hdf5=args.hdf5)
    valset = Dataset(args.data_root, load_set='val', T=args.window_size, skip=args.skip, causal=args.causal, hdf5=args.hdf5)
    length = len(trainset) // args.batch_size
# DexYCB
elif 'DexYCB' in args.data_root:
    dataset, trainset, train_factory = create_pipe(args.data_root, 'train', args, sequential=True)
    _, valset, val_factory = create_pipe(args.data_root, 'val', args, sequential=True)
    length = len(dataset) // args.batch_size

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

# iterator = iter(trainloader)
# sample = next(iterator)
# sample = next(iterator)

# profiler = cProfile.Profile()
# profiler.enable()
# sample = next(iterator)
# profiler.disable()

# stats = pstats.Stats(profiler)
# stats.strip_dirs()
# stats.sort_stats('tottime')
# stats.print_stats()

# exit()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PoseGraFormer(input_dim=2, output_dim=3, d_model=args.d_model, num_frames=args.window_size, normalize_before=True).to(device)
# model = PoseFormer(input_dim=2, output_dim=3, d_model=args.d_model, num_frames=w, normalize_before=True).to('cuda')
# model = GraFormer(hid_dim=128, coords_dim=(2, 3),  num_pts=21, temporal=False).to('cuda')

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f'total number of parameters: {num_params}')

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

pose_idx = args.window_size-1 if args.causal else args.window_size // 2

best_err = 1000
for epoch in range(args.epochs):

    total_loss = 0.0
    pose_errors = AverageMeter()
    for idx, batch in enumerate(trainloader):
        src = batch['pose2d'].to(device)
        trg = batch['pose3d'].to(device)
        bs = src.shape[0]

        out = model(src)

        loss = mpjpe(out, trg)

        # Evaluate pose in mm
        pose_err = loss.item()
        pose_errors.update(pose_err, bs)

        total_loss += loss.item()
        if (idx+1) % args.log_interval == 0:
            logger.info(f'[{idx+1}/{length}] train loss (mm): {total_loss / args.log_interval:.4f}')
            total_loss = 0.0
            pose_errors = AverageMeter()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pose_errors = AverageMeter()
    total_loss = 0.0
    for idx, batch in enumerate(valloader):
        src = batch['pose2d'].to(device)
        trg = batch['pose3d'].to(device)

        bs = src.shape[0]
        out = model(src)

        trg_pose = trg[:, pose_idx]
        pred_pose = out[:, pose_idx]

        pose_err = mpjpe(pred_pose, trg_pose).item()
        pose_errors.update(pose_err, bs)

    
    logger.info(f'val loss (mm): {pose_errors.avg:.4f}')
    if pose_errors.avg < best_err:
        best_err = pose_errors.avg
        logger.info('Saving best model...')
        torch.save(model.state_dict(), os.path.join(args.output_folder, 'transformer_best.pkl'))
        
    logger.info('Saving model...')
    torch.save(model.state_dict(), os.path.join(args.output_folder, 'transformer_'+str(epoch+1)+'.pkl'))

logger.info(f'val average error (mm): {pose_errors.avg}' )
