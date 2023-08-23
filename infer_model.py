import matplotlib.pyplot as plt
import torch
import numpy as np
from render_utils import create_renderer, render_arctic_mesh
from vis_utils import plot_pose3d, plot_pose2d, plot_mesh3d, save_mesh
from dataset.arctic_pipeline import create_pipe
from tqdm import tqdm
from models.Stohrmer import Stohrmer
from utils import parse_args, load_model, load_weights, mpjpe, p_mpjpe, calculate_error
from tqdm import tqdm

from datapipes.utils.collation_functions import collate_sequences_as_dicts

def get_mesh(dict, side, cam_ext, bs, t):
    bs, t = dict[f'left_pose'].shape[:2]

    mano = [dict[f'{side}_pose'], dict[f'{side}_shape'], dict[f'{side}_trans']]

    mano[1] = mano[1].repeat(1, t, 1)
    if mano[2].shape[1] == 1:
        mano[2] = mano[2].repeat(1, t, 1)
        mano[0] = mano[0].repeat(1, t, 1)

    for i in range(len(mano)):
        mano[i] = mano[i].view(bs * t, mano[i].shape[-1])

    mesh, _ = model.decode_mano(mano[0], mano[1], mano[2], side, cam_ext)
    mesh = mesh.view(bs, t, -1, 3)

    return mesh

def infer_mesh(outputs, cam_ext, targets=None):
    bs, t = outputs[f'left_pose'].shape[:2]

    for side in ['left', 'right']:
        mesh_pred = get_mesh(outputs, side, cam_ext, bs, t)
        outputs[f'{side}_mesh'] = mesh_pred
        
        mesh_gt = get_mesh(targets, side, cam_ext, bs, t) if f'{side}_pose' in targets.keys() else None
        targets[f'{side}_mesh'] = mesh_gt
    
    mesh = torch.cat((outputs['left_mesh'], outputs['right_mesh']), dim=2)
    mesh_gt = torch.cat((targets['left_mesh'], targets['right_mesh']), dim=2) if targets['left_mesh'] is not None else None

    return mesh[0, target_idx].cpu().detach().numpy(), mesh_gt[0, target_idx].cpu().detach().numpy() if mesh_gt is not None else None

np.set_printoptions(precision=2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

args = parse_args()
target_idx = args.window_size-1 if args.causal else args.window_size // 2

pipeline, count, decoder, factory = create_pipe(args.data_root, args.meta_root, 'val', args.mode, 'cpu', args.window_size, args.num_seqs)
loader = torch.utils.data.DataLoader(pipeline, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_sequences_as_dicts)

dataset = decoder.dataset
hand_faces = dataset.hand_faces

model = load_model(args, device, target_idx)
model, start_epoch = load_weights(model, args.weights)

min_error = 150

for i, (_, data_dict) in tqdm(enumerate(loader), total=count // args.batch_size):
    if data_dict is None: continue

    data_dict['rgb'] = [img_batch.to(device) for img_batch in data_dict['rgb']] if isinstance(model, Stohrmer) else data_dict['rgb']
    for k in data_dict.keys():
        if isinstance(data_dict[k], torch.Tensor):
            data_dict[k] = data_dict[k].to(device)

    outputs = model(data_dict)
    # for i in tqdm(range(data_dict['rgb'][0].shape[0])):
    # print(outputs[:, target_idx, :21].shape, data_dict['left_pose3d'][:, target_idx].shape)
    if isinstance(outputs, dict):
        errors = calculate_error(outputs, data_dict, dataset, target_idx, model)
        error = (errors['lm'] + errors['rm']) / 2 
        pose = torch.cat((outputs['left_pose3d'][0, target_idx], outputs['right_pose3d'][0, target_idx], \
                         outputs['top_kps3d'][0, target_idx], outputs['bottom_kps3d'][0, target_idx]),
                         dim=0).cpu().detach().numpy()
        mesh, mesh_gt = infer_mesh(outputs, data_dict['cam_ext'][0], data_dict)
    else:    
        outputs = outputs[0]
        error = (mpjpe(outputs[0, target_idx, :21], data_dict['left_pose3d'][0, target_idx]) + \
             (mpjpe(outputs[0, target_idx, 21:42], data_dict['right_pose3d'][0, target_idx]))) * 500
        pose = outputs[0, target_idx].cpu().detach().numpy()
    
    if error < min_error:
        min_error = error
        print(f'New min error: {min_error}')
        H, W = 2, 2
        fig = plt.figure(figsize=(20, 20))
        img = data_dict['rgb'][0][target_idx].cpu().numpy().transpose(1, 2, 0)
        img = np.ascontiguousarray(img * 255, np.uint8)
        ax = fig.add_subplot(H, W, 1)
        ax.imshow(img)

        plot_pose2d(img, pose, data_dict['cam_int'][0][0].cpu().numpy(), (fig, H, W), 2, '2D Pose')

        # Plot 3D pose
        plot_pose3d((fig, H, W), 3, pose, '3D pose', mode='pred')

        plot_mesh3d(mesh, hand_faces, (fig, H, W), 4, '3D mesh')
        save_mesh(mesh, hand_faces, data_dict['key'][0][target_idx], error)
        if mesh_gt is not None: save_mesh(mesh_gt, hand_faces, data_dict['key'][0][target_idx])

        seq_name = '_'.join(data_dict['key'][0][target_idx].split('/')[:-1])
        frame_name = data_dict['key'][0][target_idx].split('/')[-1].split('.')[0]

        plt.savefig(f'output/meshes/{seq_name}/{frame_name}.png')
        plt.close()
        # plt.show()
    # print(outputs[0].shape)
