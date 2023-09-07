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

from datapipes.utils.collation_functions import collate_sequences_as_dicts


np.set_printoptions(precision=2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

args = parse_args()
target_idx = args.window_size-1 if args.causal else args.window_size // 2

if not os.path.exists(args.output_folder): os.mkdir(args.output_folder)
logger = create_logger(args.output_folder)

train_pipeline, train_count, decoder, factory = create_pipe(args.data_root, args.meta_root, 'train', args.mode, 'cpu', args.window_size, args.num_seqs)
trainloader = torch.utils.data.DataLoader(train_pipeline, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_sequences_as_dicts)

val_pipeline, val_count, _, _ = create_pipe(args.data_root, args.meta_root, 'val', args.mode, 'cpu', args.window_size, args.num_seqs, factory=factory, arctic_decoder=decoder)
valloader = torch.utils.data.DataLoader(val_pipeline, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_sequences_as_dicts)

dataset = decoder.dataset
hand_faces = dataset.hand_faces

model = load_model(args, device, target_idx)
start_epoch = 1
if args.weights:
    logger.info(f'Loading model from {args.weights}')
    model, start_epoch = load_weights(model, args.weights)

# model.eval()
# for param in model.parameters():
#     param.requires_grad = False


num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f'total number of parameters: {num_params}')

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

L1 = torch.nn.L1Loss()
CEL = torch.nn.CrossEntropyLoss()

keys = ['lm', 'lp', 'lpc', 'rm', 'rp', 'rpc', 'tm', 'bm', 'tk', 'bk', 'acc']

for e in range(start_epoch, args.epochs):

    errors = {k: AverageMeter() for k in keys}
    for i, (_, data_dict) in tqdm(enumerate(trainloader), total=train_count // args.batch_size):
        if data_dict is None: continue

        data_dict['rgb'] = [img_batch.to(device) for img_batch in data_dict['rgb']] if isinstance(model, Stohrmer) else data_dict['rgb']
        for k in data_dict.keys():
            if isinstance(data_dict[k], torch.Tensor):
                data_dict[k] = data_dict[k].to(device)

        outputs = model(data_dict)
        loss = calculate_loss(outputs, data_dict)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        metrics = calculate_error(outputs, data_dict, dataset, target_idx, model)
        for k in metrics.keys():
            errors[k].update(metrics[k], args.batch_size)

        if (i+1) % args.log_interval == 0:
            error_list = [f'{k}: {v.avg:.2f}' for k, v in errors.items()]
            logger.info(f'\nEpoch {e} [{i+1} / {train_count // args.batch_size}]: {error_list}')
            # errors = {k: AverageMeter() for k in keys}

    torch.save(model.state_dict(), f'{args.output_folder}/model_{e}.pth')
    with torch.no_grad():
        errors = run_val(valloader, val_count, args.batch_size, dataset, target_idx, model, logger, e, device)  
    error_list = [f'{k}: {v.avg:.2f}' for k, v in errors.items()]  
    logger.info(f'\nepoch {e} val err: {error_list}')
    errors = {k: AverageMeter() for k in keys}


