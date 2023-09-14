import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from render_utils import create_renderer, render_arctic_mesh
from vis_utils import plot_pose3d, plot_pose2d, plot_mesh3d, save_mesh
from dataset.arctic_pipeline import create_pipe
from tqdm import tqdm
from models.Stohrmer import Stohrmer
from utils import parse_args, load_model, load_weights, mpjpe, p_mpjpe, calculate_error
from tqdm import tqdm
from pytorch3d.transforms import axis_angle_to_matrix
from datapipes.utils.collation_functions import collate_sequences_as_dicts


def get_mesh(dict, side, cam_ext, bs, t):
    t = dict[f'left_pose'].shape[1]

    mano = [dict[f'{side}_pose'][0], dict[f'{side}_shape'][0], dict[f'{side}_trans'][0]]

    mano[1] = mano[1].repeat(1, t, 1)
    if mano[2].shape[1] == 1:
        mano[2] = mano[2].repeat(1, t, 1)
        mano[0] = mano[0].repeat(1, t, 1)
    # print(cam_ext.shape)
    for i in range(len(mano)):
        mano[i] = mano[i].view(t, mano[i].shape[-1])

    mesh, _ = model.decode_mano(mano[0], mano[1], mano[2], side, cam_ext)
    mesh = mesh.view(1, t, -1, 3)

    return mesh

def infer_mesh(outputs, cam_ext, targets):
    bs, t = outputs[f'left_pose'].shape[:2]
    isTest = 'left_pose' not in targets.keys()

    for side in ['left', 'right']:
        mesh_pred = get_mesh(outputs, side, cam_ext, bs, t)
        outputs[f'{side}_mesh'] = mesh_pred
        
        mesh_gt = get_mesh(targets, side, cam_ext, bs, t) if not isTest else None
        targets[f'{side}_mesh'] = mesh_gt
    
    mesh = torch.cat((outputs['left_mesh'], outputs['right_mesh']), dim=2)
    mesh_gt = torch.cat((targets['left_mesh'], targets['right_mesh']), dim=2) if not isTest else None

    return mesh[0, target_idx].cpu().detach().numpy(), mesh_gt[0, target_idx].cpu().detach().numpy() if not isTest else None

def infer_object_mesh(outputs, cam_ext, faces, faces_gt, targets, mesh=None, mesh_gt=None):

    bs, t = outputs[f'left_pose'].shape[:2]
    isTest = 'obj_pose' not in targets.keys()

    obj_pose, obj_class = outputs['obj_pose'], outputs['obj_class']
    if obj_pose.shape[1] == 1: obj_pose = obj_pose.repeat(1, t, 1)

    pred_labels = torch.argmax(obj_class, dim=1)
    obj_pred = obj_pose[:, :, :1], obj_pose[:, :, 1:4], obj_pose[:, :, 4:]
    pred_object_names = [dataset.object_names[l] for l in pred_labels]
    
    i = 0
    obj_verts_pred, _ = dataset.transform_obj(pred_object_names[i], obj_pred[0][i], obj_pred[1][i], obj_pred[2][i], cam_ext)
    if not isTest:
        obj_pose_gt = targets['obj_pose']
        obj_gt = obj_pose_gt[:, :, :1], obj_pose_gt[:, :, 1:4], obj_pose_gt[:, :, 4:]
        object_names = [dataset.object_names[l] for l in targets['label'][:, 0]]
        obj_verts_gt, _ = dataset.transform_obj(object_names[i], obj_gt[0][i], obj_gt[1][i], obj_gt[2][i], cam_ext)

    offset, offset_gt = 778 * 2, 778 * 2
    for part in ['top', 'bottom']:
        obj_mesh = obj_verts_pred[part][target_idx].cpu().detach().numpy()
        mesh = np.concatenate((mesh, obj_mesh), axis=0) if mesh is not None else obj_mesh
        obj_faces = dataset.objects[pred_object_names[i]][part][1] + offset
        faces = np.concatenate((faces, obj_faces), axis=0)
        offset = mesh.shape[0]

        if not isTest:
            obj_mesh_gt = obj_verts_gt[part][target_idx].cpu().detach().numpy()
            mesh_gt = np.concatenate((mesh_gt, obj_mesh_gt), axis=0) if mesh_gt is not None else obj_mesh_gt
            obj_faces = dataset.objects[object_names[i]][part][1] + offset_gt
            faces_gt = np.concatenate((faces_gt, obj_faces), axis=0)
            offset_gt = mesh_gt.shape[0]

    return mesh, mesh_gt, faces, faces_gt


def store_results(outputs, keys, output_dict, target_idx=0):

    for i, sample in enumerate(keys):
        key = sample[target_idx].split('/')
        subject = key[0]
        seq_name = key[1]
        camera = key[2]
        frame = key[3].split('.')[0]
        seq_id = '_'.join(key[:3])

        if seq_id not in output_dict.keys():
            output_dict[seq_id] = {}

        img_path = f'./data/arctic_data/data/images/{subject}/{seq_name}/{camera}/{frame}.jpg'
        output_dict[seq_id][frame] = {'meta_info.imgname': img_path}
        for k in outputs.keys():
            if 'pose' in k and '3d' not in k and 'obj' not in k:
                pose = outputs[k][i, target_idx].reshape(16, 3)
                matrix = axis_angle_to_matrix(pose)
                output_dict[seq_id][frame][f'pred.mano.pose.{k[0]}'] = matrix
            elif 'shape' in k:
                output_dict[seq_id][frame][f'pred.mano.beta.{k[0]}'] = outputs[k][i, target_idx]
            elif 'trans' in k:
                output_dict[seq_id][frame][f'pred.mano.cam_t.{k[0]}'] = outputs[k][i, target_idx] # ?
            elif 'obj_pose' in k:
                output_dict[seq_id][frame][f'pred.object.radian'] = outputs[k][i, target_idx, 0]
                output_dict[seq_id][frame][f'pred.object.rot'] = outputs[k][i, target_idx, 1:4]
                output_dict[seq_id][frame][f'pred.object.cam_t'] = outputs[k][i, target_idx, 4:] # ?

def convert_to_directory(output_dict):

    path = f'output/inference/ex_patch_features/pose_p1_test/eval'
    os.makedirs(path, exist_ok=True)
    for seq_id in output_dict.keys():
        os.makedirs(f'{path}/{seq_id}/meta_info', exist_ok=True)
        os.makedirs(f'{path}/{seq_id}/preds', exist_ok=True)

        # Group all values inside output_dict[seq_id] for each key in a tensor
        files = {}
        for frame in sorted(output_dict[seq_id].keys()):
            for k in output_dict[seq_id][frame].keys():
                if k not in files.keys():
                    files[k] = []
                files[k].append(output_dict[seq_id][frame][k])
        
        for k in files.keys():
            subdir = 'meta_info' if 'meta' in k else 'preds'
            t = torch.stack(files[k]).to(torch.float16) if 'meta' not in k else files[k]
            torch.save(t, f'{path}/{seq_id}/{subdir}/{k}.pt')


np.set_printoptions(precision=2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

args = parse_args()
target_idx = args.window_size-1 if args.causal else args.window_size // 2

pipeline, count, decoder, factory = create_pipe(args.data_root, args.meta_root, args.batch_size, args.split, args.mode, 'cpu', args.window_size, args.num_seqs)
loader = torch.utils.data.DataLoader(pipeline, batch_size=None, num_workers=args.num_workers, pin_memory=True)


dataset = decoder.dataset
hand_faces = np.concatenate((dataset.hand_faces['left'], dataset.hand_faces['right'] + 778), axis=0)
model = load_model(args, device, target_idx)
model, start_epoch = load_weights(model, args.weights)

min_error = 100
outputs_dict = {}

for i, (_, data_dict) in tqdm(enumerate(loader), total=count // args.batch_size):
    if data_dict is None: continue
    data_dict['rgb'] = [img_batch.to(device) for img_batch in data_dict['rgb']]

    for k in data_dict.keys():
        if isinstance(data_dict[k], torch.Tensor):
            data_dict[k] = data_dict[k].to(device)

    with torch.no_grad():
        outputs = model(data_dict)
    store_results(outputs, data_dict['key'], outputs_dict)
    # if i % 10 == 1: break
    error = None
    if isinstance(outputs, dict):
        if args.split in ['train', 'val']: 
            errors = calculate_error(outputs, data_dict, dataset, target_idx, model)
            if errors['acc'] != 1: continue
            error = (errors['lm'] + errors['rm'] + errors['tm'] + errors['bm']) / 4 

        pose = torch.cat((outputs['left_pose3d'][0, target_idx], outputs['right_pose3d'][0, target_idx], \
                        outputs['top_kps3d'][0, target_idx], outputs['bottom_kps3d'][0, target_idx]),
                        dim=0).cpu().detach().numpy()
        mesh, mesh_gt = infer_mesh(outputs, data_dict['cam_ext'][0], data_dict)
        mesh, mesh_gt, faces, faces_gt = infer_object_mesh(outputs, data_dict['cam_ext'][0], np.copy(hand_faces), np.copy(hand_faces), data_dict, mesh, mesh_gt)
    else: 
        outputs = outputs[0]
        if args.split in ['train', 'val']:
            error = (mpjpe(outputs[0, target_idx, :21], data_dict['left_pose3d'][0, target_idx]) + \
                (mpjpe(outputs[0, target_idx, 21:42], data_dict['right_pose3d'][0, target_idx]))) * 500
        pose = outputs[0, target_idx].cpu().detach().numpy()
    
    if args.visualize and ((error is None and i % 1000 == 0) or (error is not None and error < 50)):
        min_error = error
        if error is not None: print(f'New min error: {min_error:.2f}')
        H, W = 2, 2
        fig = plt.figure(figsize=(20, 20))
        img = data_dict['rgb'][0][target_idx].cpu().numpy().transpose(1, 2, 0)
        img = np.ascontiguousarray(img * 255, np.uint8)
        ax = fig.add_subplot(H, W, 1)
        ax.imshow(img)

        plot_pose2d(img, pose, data_dict['cam_int'][0][0].cpu().numpy(), (fig, H, W), 2, '2D Pose')

        # Plot 3D pose
        plot_pose3d((fig, H, W), 3, pose, '3D pose', mode='pred')

        plot_mesh3d(mesh, faces, (fig, H, W), 4, '3D mesh')
        save_mesh(mesh, faces, data_dict['key'][0][target_idx], error)
        if mesh_gt is not None: save_mesh(mesh_gt, faces_gt, data_dict['key'][0][target_idx])

        seq_name = '_'.join(data_dict['key'][0][target_idx].split('/')[:-1])
        frame_name = data_dict['key'][0][target_idx].split('/')[-1].split('.')[0]
        print('Exporting:', seq_name, frame_name)
        plt.savefig(f'output/meshes/{seq_name}/{frame_name}.png')
        plt.close()

convert_to_directory(outputs_dict)