from vis_utils import plot_pose2d, plot_pose3d, plot_temporal_sample
from dataset.HO3D_dataset import Dataset
import torch
import matplotlib
# matplotlib.use('gtk3agg')

import matplotlib.pyplot as plt
from models.poseformer import PoseFormer
from utils import parse_args
from dataset.DexYCB_pipeline import create_pipe


def visualize2d(img, pose3d, pose3d_gt=None):
    
    fig = plt.figure(figsize=(20, 10))
    
    H = 1
    W = 2

    fig_config = (fig, H, W)
    plot_id = 1
    plt_image = plot_pose2d(img, pose3d, fig_config, plot_id, 'predicted 2D pose')
    plot_pose2d(plt_image, pose3d_gt, fig_config, plot_id, 'predicted 2D pose')

    plot_id += 1
    plot_pose3d(fig_config, plot_id, pose3d, 'predicted pose')
    # plot_pose3d(fig_config, plot_id, pose3d_gt, 'predicted pose', mode='gt')

    fig.tight_layout()
    plt.show()
    # plt.savefig(filename)
    # plt.clf()
    # plt.close(fig)

args = parse_args()

# HO3D
if 'HO3D' in args.data_root:
    trainset = Dataset(args.data_root, T=args.window_size, skip=args.skip, causal=args.causal, hdf5=args.hdf5)
    valset = Dataset(args.data_root, load_set='val', T=args.window_size, skip=args.skip, causal=args.causal, hdf5=args.hdf5)
# DexYCB
elif 'DexYCB' in args.data_root:
    _, trainset, train_factory = create_pipe(args.data_root, 'train', args, sequential=True)
    _, valset, val_factory = create_pipe(args.data_root, 'val', args, sequential=True)

valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

model = PoseFormer(input_dim=2, output_dim=3, d_model=args.d_model, num_frames=args.window_size, normalize_before=True).to('cuda')
model.load_state_dict(torch.load(args.pretrained_model))
model.eval()
# print(model)
pose_idx = args.window-1 if args.causal else args.window_size // 2
for idx, batch in enumerate(valloader):
    plot_temporal_sample(batch)
    src = batch['pose2d'].to('cuda')
    trg = batch['pose3d'].to('cuda')
    bs = src.shape[0]
    out = model(src)[0].cpu().detach().numpy()
    trg_pose = trg[0, pose_idx].cpu().detach().numpy()

    img = batch['images'][0, pose_idx]

    visualize2d(img, out, trg_pose)

