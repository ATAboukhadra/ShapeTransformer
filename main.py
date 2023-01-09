import h5py
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import torch
import itertools
from dataset import Dataset
from transformer import PoseTransformer
from utils import mpjpe, AverageMeter, parse_args

args = parse_args()

causal = [False, True]
windows = [1, 3, 5, 7, 9, 11, 13]
skip = [1, 2, 5, 10, 20]

best_err = 1000
best_comb = None

for c, w, s in itertools.product(causal, windows, skip):
    args.causal = c
    args.window_size = w
    args.skip = s
    print(f'combination: {c, w, s}')
    trainset = Dataset(args.data_root, T=args.window_size, skip=args.skip, causal=args.causal, hdf5=args.hdf5)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    valset = Dataset(args.data_root, load_set='val', T=args.window_size, skip=args.skip, causal=args.causal, hdf5=args.hdf5)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = PoseTransformer(input_dim=2, output_dim=3, d_model=args.d_model).to('cuda')
    # print('total number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    pose_idx = args.window-1 if args.causal else args.window_size // 2
    if not os.path.exists(args.output_folder): os.mkdir(args.output_folder)

    for epoch in range(args.epochs):

        # total_loss = 0.0
        # pose_errors = AverageMeter()
        for idx, batch in enumerate(trainloader):
            src = batch['pose2d'].to('cuda')
            bs = src.shape[0]
            trg = batch['pose3d'].to('cuda')
            out = model(src)

            loss = criterion(trg, out) 

            # Evaluate pose in mm
            # trg_pose = trg[:, pose_idx]
            # pred_pose = out[:, pose_idx]
            # pose_err = mpjpe(pred_pose, trg_pose).item()
            # pose_errors.update(pose_err, bs)

            # total_loss += loss.item()

            # if (idx+1) % args.log_interval == 0:
                # print(f'train loss [{idx+1}/{len(trainloader)}]: {total_loss / args.log_interval:.4f}, average error (mm): {pose_errors.avg}' )
                # total_loss = 0.0
                # pose_errors = AverageMeter()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print('-' * 40)

        pose_errors = AverageMeter()
        total_loss = 0.0
        for idx, batch in enumerate(valloader):
            src = batch['pose2d'].to('cuda')
            bs = src.shape[0]
            trg = batch['pose3d'].to('cuda')
            out = model(src)

            trg_pose = trg[:, pose_idx]
            pred_pose = out[:, pose_idx]

            pose_err = mpjpe(pred_pose, trg_pose).item()
            pose_errors.update(pose_err, bs)
            loss = criterion(trg, out)
            total_loss += loss.item()
            # if (idx+1) % args.log_interval == 0:
        # print(f'val loss [{idx+1}/{len(valloader)}]: {total_loss / args.log_interval:.4f}, average error (mm): {pose_errors.avg}' )
    print(f'val average error (mm): {pose_errors.avg}' )
    if pose_errors.avg < best_err:
        print('updating best error ..')
        best_err = pose_errors.avg
        best_comb = (c, w, s)

print('best combination:', best_comb, 'best error', best_err)
        # torch.save(model.state_dict(), os.path.join(args.output_folder, 'transformer_'+str(epoch+1)+'.pkl'))
