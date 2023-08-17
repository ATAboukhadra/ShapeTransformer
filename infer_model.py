import matplotlib.pyplot as plt
import torch
import numpy as np
from render_utils import create_renderer, render_arctic_mesh
from vis_utils import plot_pose3d, plot_pose2d
from dataset.arctic_pipeline import create_pipe
from tqdm import tqdm
from models.Stohrmer import Stohrmer
from utils import parse_args, load_model, load_weights, mpjpe, p_mpjpe
from tqdm import tqdm

from datapipes.utils.collation_functions import collate_sequences_as_dicts

np.set_printoptions(precision=2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

args = parse_args()
target_idx = args.window_size-1 if args.causal else args.window_size // 2

pipeline, count, decoder, factory = create_pipe(args.data_root, args.meta_root, 'val', args.mode, 'cpu', args.window_size, args.num_seqs)
loader = torch.utils.data.DataLoader(pipeline, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_sequences_as_dicts)

dataset = decoder.dataset
hand_faces = dataset.hand_faces

model = load_model(args, device)
model, start_epoch = load_weights(model, args.weights)

min_error = 30

for i, (_, data_dict) in tqdm(enumerate(loader), total=count // args.batch_size):
    if data_dict is None: continue

    data_dict['rgb'] = [img_batch.to(device) for img_batch in data_dict['rgb']] if isinstance(model, Stohrmer) else data_dict['rgb']
    for k in data_dict.keys():
        if isinstance(data_dict[k], torch.Tensor):
            data_dict[k] = data_dict[k].to(device)

    outputs = model(data_dict)
    # for i in tqdm(range(data_dict['rgb'][0].shape[0])):
    # print(outputs[:, target_idx, :21].shape, data_dict['left_pose3d'][:, target_idx].shape)
    error = (mpjpe(outputs[:, target_idx, :21], data_dict['left_pose3d'][:, target_idx]) + \
             (mpjpe(outputs[:, target_idx, 21:42], data_dict['right_pose3d'][:, target_idx]))) * 500
    if error < 25:
        min_error = error
        print(f'New min error: {min_error}')
        fig = plt.figure(figsize=(20, 20))
        img = data_dict['rgb'][0][target_idx].cpu().numpy().transpose(1, 2, 0)
        img = np.ascontiguousarray(img * 255, np.uint8)
        plot_pose2d(img, outputs[0][target_idx].cpu().detach().numpy(), data_dict['cam_int'][0][0].cpu().numpy(), (fig, 1, 2), 1, '2D Pose')

        # Plot 3D pose
        plot_pose3d((fig, 1, 2), 2, outputs[0][target_idx].cpu().detach().numpy(), '3D pose', mode='pred')
        plt.show()
    # print(outputs[0].shape)
