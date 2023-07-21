import matplotlib.pyplot as plt
import torch
import numpy as np
from render_utils import create_renderer, render_arctic_mesh
from vis_utils import showHandJoints
from dataset.arctic_pipeline import create_pipe, temporal_batching
from tqdm import tqdm
from models.Stohrmer import Stohrmer
import os
from utils import AverageMeter, parse_args, create_logger, calculate_loss, calculate_error, run_val, load_model
from tqdm import tqdm
from models.model_poseformer import PoseTransformer
from multigpu_helpers.dist_helper import DistributedHelper

def main():

    np.set_printoptions(precision=2)

    args = parse_args()

    dh = DistributedHelper()
    target_idx = args.window_size-1 if args.causal else args.window_size // 2

    if not os.path.exists(args.output_folder): os.mkdir(args.output_folder)
    logger = create_logger(args.output_folder)

    train_pipeline, train_count, decoder, factory = create_pipe(args.data_root, args.meta_root, 'train', 'cpu', args.window_size, args.num_seqs)
    trainloader = torch.utils.data.DataLoader(train_pipeline, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=temporal_batching, drop_last=True)

    val_pipeline, val_count, _, _ = create_pipe(args.data_root, args.meta_root, 'val', 'cpu', args.window_size, args.num_seqs, factory=factory, arctic_decoder=decoder)
    valloader = torch.utils.data.DataLoader(val_pipeline, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=temporal_batching, drop_last=True)

    dataset = decoder.dataset
    hand_faces = dataset.hand_faces

    if args.model_name == 'stohrmer':
        model = Stohrmer(dh.local_rank, num_kps=42, num_frames=args.window_size)
    else:
        model = PoseTransformer(num_frame=args.window_size, num_joints=42, in_chans=2)

    start_epoch = 0
    if args.weights:
        if dh.is_master: 
            logger.info(f'Loading model from {args.weights} if exists')
        model, start_epoch = load_model(model, args.weights)
    
    model = dh.wrap_model_for_ddp(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if dh.is_master: logger.info(f'total number of parameters: {num_params}')

    if args.run_val:
        with torch.no_grad():
            errors = run_val(valloader, val_count, args.batch_size, dataset, target_idx, model, logger, start_epoch, dh.local_rank, dh)
        if dh.is_master:
            error_list = [f'{k}: {v.avg:.2f}' for k, v in errors.items()]
            logger.info(f'\nEpoch {start_epoch-1} Val Err: {error_list}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    keys = ['left_mesh_err', 'left_pose_err', 'right_mesh_err', 'right_pose_err', 'top_obj_err', 'bottom_obj_err', 'obj_acc']

    for e in range(start_epoch, args.epochs):

        errors = {k: AverageMeter() for k in keys}
        total_count = train_count // (args.batch_size * dh.world_size)
        loader = tqdm(enumerate(trainloader), total=train_count // args.batch_size) if dh.is_master else enumerate(trainloader)
        for i, data_dict in loader:
            if data_dict is None: continue

            data_dict['rgb'] = [img_batch.to(dh.local_rank) for img_batch in data_dict['rgb']]

            for k in data_dict.keys():
                if isinstance(data_dict[k], torch.Tensor):
                    data_dict[k] = data_dict[k].to(dh.local_rank)

            outputs = model(data_dict)
            loss = calculate_loss(outputs, data_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metrics = calculate_error(outputs, data_dict, dataset, target_idx, model.module)
            
            dh.sync_distributed_values(metrics)
            if dh.is_master:
                for k in metrics.keys():
                    errors[k].update(metrics[k].item(), args.batch_size)

            if (i+1) % args.log_interval == 0 and dh.is_master:
                error_list = [f'{k}: {v.avg:.2f}' for k, v in errors.items()]
                logger.info(f'\n[{i+1} / {total_count}]: {error_list}')
                errors = {k: AverageMeter() for k in keys}
                torch.save(model.module.state_dict(), f'{args.output_folder}/model_{e}.pth')

        if dh.is_master:
            logger.info(f'Saving model at epoch {e}')
            torch.save(model.module.state_dict(), f'{args.output_folder}/model_{e}.pth')

        errors = run_val(valloader, val_count, args.batch_size, dataset, target_idx, model, logger, e, dh.local_rank, dh)
        if dh.is_master:
            error_list = [f'{k}: {v.avg:.2f}' for k, v in errors.items()]
            logger.info(f'\nEpoch {e} Val Err: {error_list}')
            errors = {k: AverageMeter() for k in keys}


if __name__ == '__main__':
    main()