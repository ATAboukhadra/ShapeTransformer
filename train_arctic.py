import matplotlib.pyplot as plt
import torch
import numpy as np
from render_utils import create_renderer, render_arctic_mesh
from vis_utils import showHandJoints
from dataset.arctic_pipeline import create_pipe, batch_samples
from tqdm import tqdm
from models.Stohrmer import Stohrmer
import os
from utils import AverageMeter, parse_args, create_logger, calculate_loss, calculate_error

np.set_printoptions(precision=2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

args = parse_args()
target_idx = args.window_size-1 if args.causal else args.window_size // 2

if not os.path.exists(args.output_folder): os.mkdir(args.output_folder)
logger = create_logger(args.output_folder)

train_pipeline, decoder, factory = create_pipe(args.data_root, args.meta_root, 'train', torch.device('cpu'), args.window_size)
trainloader = torch.utils.data.DataLoader(train_pipeline, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=batch_samples)

val_pipeline, _, _ = create_pipe(args.data_root, args.meta_root, 'val', torch.device('cpu'), args.window_size, factory=factory, arctic_decoder=decoder)
valloader = torch.utils.data.DataLoader(train_pipeline, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=batch_samples)

dataset = decoder.dataset
hand_faces = dataset.hand_faces

model = Stohrmer(device, num_kps=42, num_frames=args.window_size).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f'total number of parameters: {num_params}')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

L2 = torch.nn.MSELoss()
CEL = torch.nn.CrossEntropyLoss()

keys = ['left_mesh_err', 'left_pose_err', 'right_mesh_err', 'right_pose_err', 'top_obj_err', 'bottom_obj_err', 'obj_acc']

for e in range(args.epochs):

    errors = {k: AverageMeter() for k in keys}
    for i, data_dict in enumerate(trainloader):

        for k in data_dict.keys():
            data_dict[k] = data_dict[k].to(device) if isinstance(data_dict[k], torch.Tensor) else data_dict[k]

        outputs = model(data_dict)
        loss = calculate_loss(outputs, data_dict)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        calculate_error(outputs, data_dict, errors, dataset, target_idx, model)

        if (i+1) % args.log_interval == 0:
            error_list = [f'{k}: {v.avg:.2f}' for k, v in errors.items()]
            logger.info(f'{i+1}: {error_list}')
            errors = {k: AverageMeter() for k in keys}

    torch.save(model.state_dict(), f'{args.output_folder}/model_{e}.pth')

    for i, data_dict in enumerate(valloader):
        outputs = model(data_dict)
        calculate_error(outputs, data_dict, errors, dataset, target_idx, model)
    
    error_list = [f'{k}: {v.avg:.2f}' for k, v in errors.items()]
    logger.info(f'epoch {e+1} val: {error_list}')


