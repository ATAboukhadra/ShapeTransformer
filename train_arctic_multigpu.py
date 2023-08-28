import matplotlib.pyplot as plt
import torch
import numpy as np
from render_utils import create_renderer, render_arctic_mesh
from vis_utils import showHandJoints
from dataset.arctic_pipeline import create_pipe
from tqdm import tqdm
from models.Stohrmer import Stohrmer
import os
from utils import AverageMeter, parse_args, create_logger, calculate_loss, calculate_error, run_val, load_model, load_weights
from tqdm import tqdm
from models.model_poseformer import PoseTransformer
from multigpu_helpers.dist_helper import DistributedHelper
from datapipes.utils.collation_functions import collate_sequences_as_dicts
from torch import distributed as dist
import os

class Terminate():
    def __init__(self, terminate_log_path):
        self.terminate_log_path = terminate_log_path

    def terminate(self):
        with open(self.terminate_log_path, 'w') as f:
            f.write('')

    def isTerminated(self):
        
        return os.path.exists(self.terminate_log_path)

def main():

    np.set_printoptions(precision=2)

    args = parse_args()

    dh = DistributedHelper()
    target_idx = args.window_size-1 if args.causal else args.window_size // 2

    if dh.is_master and not os.path.exists(args.output_folder): os.mkdir(args.output_folder)
    logger = create_logger(args.output_folder)

    train_pipeline, train_count, decoder, factory = create_pipe(args.data_root, args.meta_root, 'train', args.mode, 'cpu', args.window_size, args.num_seqs)
    trainloader = torch.utils.data.DataLoader(train_pipeline, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_sequences_as_dicts, drop_last=True)

    val_pipeline, val_count, _, _ = create_pipe(args.data_root, args.meta_root, 'val', args.mode, 'cpu', args.window_size, args.num_seqs, factory=factory, arctic_decoder=decoder)
    valloader = torch.utils.data.DataLoader(val_pipeline, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_sequences_as_dicts, drop_last=True)

    dataset = decoder.dataset
    hand_faces = dataset.hand_faces

    model = load_model(args, dh.local_rank, target_idx)

    start_epoch = 1
    if args.weights:
        if dh.is_master: 
            logger.info(f'Loading model from {args.weights} if exists')
        model, start_epoch = load_weights(model, args.weights)
    
    model = dh.wrap_model_for_ddp(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if dh.is_master: 
        logger.info(f'saving outputs to {args.output_folder}')
        logger.info(f'total number of parameters: {num_params}')

    if args.run_val:
        with torch.no_grad():
            errors = run_val(valloader, val_count, args.batch_size, dataset, target_idx, model, logger, start_epoch, dh.local_rank, dh)
        if dh.is_master:
            error_list = [f'{k}: {v.avg:.2f}' for k, v in errors.items()]
            logger.info(f'\nEpoch {start_epoch-1} Val Err: {error_list}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    keys = ['lm', 'lp', 'lpc', 'rm', 'rp', 'rpc', 'tm', 'bm', 'tk', 'bk', 'acc']

    total_count = train_count // (args.batch_size * dh.world_size)

    for e in range(start_epoch, args.epochs):

        errors = {k: AverageMeter() for k in keys}
        loader = tqdm(enumerate(trainloader), total=total_count) if dh.is_master else enumerate(trainloader)
        # t = Terminate(f'{args.output_folder}/terminate_{e}.txt')

        for i, (_, data_dict) in loader:

            if data_dict is None: continue
            data_dict['rgb'] = [img_batch.to(dh.local_rank) for img_batch in data_dict['rgb']]

            for k in data_dict.keys():
                if isinstance(data_dict[k], torch.Tensor):
                    data_dict[k] = data_dict[k].to(dh.local_rank)

            outputs = model(data_dict)
            loss = calculate_loss(outputs, data_dict)
            
            if i / total_count > 0.75: 
                # if t.isTerminated():
                #     print('gpu', dh.local_rank, flush=True)
                break
            
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
                logger.info(f'\nEpoch {e} [{i+1} / {total_count}]: {error_list}')
                # errors = {k: AverageMeter() for k in keys}
                torch.save(model.module.state_dict(), f'{args.output_folder}/model_{e}.pth')

            # if dh.is_master: break

        # print('terminate on gpu', dh.local_rank, flush=True)
        # t.terminate()

        if dh.is_master:
            logger.info(f'Saving model at epoch {e+1}')
            torch.save(model.module.state_dict(), f'{args.output_folder}/model_{e}.pth')

        errors = run_val(valloader, val_count, args.batch_size, dataset, target_idx, model, logger, e, dh.local_rank, dh)
        if dh.is_master:
            error_list = [f'{k}: {v.avg:.2f}' for k, v in errors.items()]
            logger.info(f'\nEpoch {e} Val Err: {error_list}')
            errors = {k: AverageMeter() for k in keys}


if __name__ == '__main__':
    main()