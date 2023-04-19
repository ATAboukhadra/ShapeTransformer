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
from utils import mpjpe, AverageMeter, parse_args, initialize_masks

args = parse_args()

# causal = [False, True]
# windows = [1, 3, 5, 7, 9, 11, 13]
# skip = [1, 2, 5, 10, 20]

causal = [False]
windows = [21]
skip = [1] #, 2, 5, 10, 20]

best_err = 1000
best_comb = None

for c, w, s in itertools.product(causal, windows, skip):
    args.causal = c
    args.window_size = w
    args.skip = s
    print(f'combination: {c, w, s}')

    # HO3D
    if 'HO3D' in args.data_root:
        trainset = Dataset(args.data_root, T=args.window_size, skip=args.skip, causal=args.causal, hdf5=args.hdf5)
        valset = Dataset(args.data_root, load_set='val', T=args.window_size, skip=args.skip, causal=args.causal, hdf5=args.hdf5)
    # DexYCB
    elif 'DexYCB' in args.data_root:
        _, trainset, train_factory = create_pipe(args.data_root, 'train', args, sequential=True)
        _, valset, val_factory = create_pipe(args.data_root, 'val', args, sequential=True)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    # trainset, dexycb_pipeline, factory = create_pipe(args.dataset_dir, 'train', args)
    # trainset = torch.utils.data.DataLoader(dexycb_pipeline, batch_size=1, num_workers=0, shuffle=True)
    
    model = PoseGraFormer(input_dim=2, output_dim=3, d_model=args.d_model, num_frames=w, normalize_before=True).to('cuda')
    # model = PoseFormer(input_dim=2, output_dim=3, d_model=args.d_model, num_frames=w, normalize_before=True).to('cuda')
    # model = GraFormer(hid_dim=128, coords_dim=(2, 3),  num_pts=21, temporal=False).to('cuda')

    print('total number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # criterion = torch.nn.L1Loss()

    pose_idx = args.window_size-1 if args.causal else args.window_size // 2
    if not os.path.exists(args.output_folder): os.mkdir(args.output_folder)

    for epoch in range(args.epochs):

        total_loss = 0.0
        pose_errors = AverageMeter()
        for idx, batch in enumerate(trainloader):
            src = batch['pose2d'].to('cuda')
            trg = batch['pose3d'].to('cuda')
            bs = src.shape[0]

            # if model.isinstance(GraFormer):
            #     src = src.squeeze(1)
            #     trg = trg.squeeze(1)

            out = model(src)
            # trg_pose = trg[:, pose_idx]

            # loss = criterion(trg_pose, out)
            loss = mpjpe(out, trg)

            # Evaluate pose in mm
            # pred_pose = out[:, pose_idx]
            pose_err = loss.item()
            pose_errors.update(pose_err, bs)

            total_loss += loss.item()
            if (idx+1) % args.log_interval == 0:
                # print(f'[{idx+1}/{len(trainloader)}] train loss (mm): {total_loss / args.log_interval:.4f}')#, average error (mm): {pose_errors.avg:.4f}' )
                print(f'[{idx+1}] train loss (mm): {total_loss / args.log_interval:.4f}')#, average error (mm): {pose_errors.avg:.4f}' )

                total_loss = 0.0
                pose_errors = AverageMeter()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print('-' * 40)

        pose_errors = AverageMeter()
        total_loss = 0.0
        for idx, batch in enumerate(valloader):
            src = batch['pose2d'].to('cuda')
            trg = batch['pose3d'].to('cuda')

            bs = src.shape[0]
            # if model.isinstance(GraFormer):
            #     src = src.squeeze(1)
                # trg = trg.squeeze(1)
            out = model(src)

            trg_pose = trg[:, pose_idx]
            pred_pose = out[:, pose_idx]

            pose_err = mpjpe(pred_pose, trg_pose).item()
            pose_errors.update(pose_err, bs)
            # loss = criterion(trg, out)
            # total_loss += loss.item()
            # if (idx+1) % args.log_interval == 0:
        
        # print(f'[{idx+1}/{len(valloader)}] val loss (mm): {pose_errors.avg:.4f}')#, average error (mm): {pose_errors.avg:.4f}' )
        print(f'[{idx+1}] val loss (mm): {pose_errors.avg:.4f}')#, average error (mm): {pose_errors.avg:.4f}' )
        
        # print(f'val loss [{idx+1}/{len(valloader)}]: {total_loss / args.log_interval:.4f}, average error (mm): {pose_errors.avg}' )
        print('Saving model...')
        torch.save(model.state_dict(), os.path.join(args.output_folder, 'transformer_'+str(epoch+1)+'.pkl'))
    print(f'val average error (mm): {pose_errors.avg}' )
    if pose_errors.avg < best_err:
        print('updating best error ..')
        best_err = pose_errors.avg
        best_comb = (c, w, s)

print('best combination:', best_comb, 'best error', best_err)
