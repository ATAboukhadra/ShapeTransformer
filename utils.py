import torch
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import logging
import torch.nn.functional as F

from nimble.utils import batch_to_tensor_device
from nimble.NIMBLELayer import NIMBLELayer
from pytorch3d.renderer import Textures
from pytorch3d.structures.meshes import Meshes
from pytorch3d.io import load_obj
from manopth.manolayer import ManoLayer
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from multiprocessing import Value

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# data_path = '/home2/HO3D_v3'
# cam_intr = torch.tensor([
#   [614.627,   0.,    320.262],
#  [  0. ,   614.101 ,238.469],
#  [  0. ,     0. ,     1.   ]] 
# , device=device
#  )
# mano_layer = ManoLayer(mano_root='mano_v1_2/models', ncomps=45, flat_hand_mean=True, use_pca=False).to(device)

def parse_args():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, help="Directory containing data", default='/data/ho')
    ap.add_argument("--meta_root", type=str, help="Directory containing additional data", default='/data/DexYCB')
    ap.add_argument("--mode", type=str, help="Cameras to be included (all, allocentric, egocentric)", default='all')
    ap.add_argument("--output_folder", type=str, help="relative path to save checkpoint", default='checkpoints/transformer')
    ap.add_argument("--batch_size", type=int, help="batch size", default='8')
    ap.add_argument("--model_name", type=str, help="name of the model", default='stohrmer')
    ap.add_argument("--log_interval", type=int, help="How many batches needes to log", default='50')
    ap.add_argument("--val_interval", type=int, help="How many batches needes to log", default='100000')
    ap.add_argument("--epochs", type=int, help="number of epochs after which to stop", default='10')
    ap.add_argument("--window_size", type=int, help="How many batches needes to log", default='9')
    ap.add_argument("--skip", type=int, help="how many frames to jump", default='1')
    ap.add_argument("--d_model", type=int, help="number of features in transformer", default='32')
    ap.add_argument("--num_workers", type=int, help="number of workers", default='8')
    ap.add_argument("--num_seqs", type=int, help="number of sequences in each workers shuffle buffer", default='4')
    ap.add_argument("--num_gpus", type=int, help="number of gpus to be used in multigpu case", default='1')
    ap.add_argument("--input_dim", type=int, help="number of spatial features", default='2')
    ap.add_argument("--run_val", action='store_true', help="run validation epoch once before training")
    ap.add_argument("--hdf5", action='store_true', help="Load data from HDF5 file") 
    ap.add_argument("--causal", action='store_true', help="Use only previous frames")     
    ap.add_argument("--weights", type=str, default='', help="path to pretrained weights")
    ap.add_argument("--backbone_path", type=str, default='', help="path to backbone network or subnetwork")
     

    return ap.parse_args()

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    # Convert to Numpy because this metric needs numpy array
    predicted = predicted.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def initialize_masks(batch, heads, size, p):
  # probability of each element being 1 (True)

    size = (heads * size, batch, batch)  # shape of the boolean mask
    mask = torch.rand(size) < p
    return mask

# for i in range(batch['patches'].shape[0]):
#     patch = batch['patches'][i]
#     # print(patch.shape)
#     ax = fig.add_subplot(5, 5, i+1)
#     ax.imshow(patch)
    # print(batch['patches'].shape)

# plt.imshow(im)
# plt.show()

def init_nimble(device):

    pm_dict_name = r"nimble/assets/NIMBLE_DICT_9137.pkl"
    tex_dict_name = r"nimble/assets/NIMBLE_TEX_DICT.pkl"

    if os.path.exists(pm_dict_name):
        pm_dict = np.load(pm_dict_name, allow_pickle=True)
        pm_dict = batch_to_tensor_device(pm_dict, device)

    if os.path.exists(tex_dict_name):
        tex_dict = np.load(tex_dict_name, allow_pickle=True)
        tex_dict = batch_to_tensor_device(tex_dict, device)

    if os.path.exists(r"nimble/assets/NIMBLE_MANO_VREG.pkl"):
        nimble_mano_vreg = np.load("nimble/assets/NIMBLE_MANO_VREG.pkl", allow_pickle=True)
        nimble_mano_vreg = batch_to_tensor_device(nimble_mano_vreg, device)
    else:
        nimble_mano_vreg=None

    nlayer = NIMBLELayer(pm_dict, tex_dict, device, use_pose_pca=True, pose_ncomp=30, shape_ncomp=20, nimble_mano_vreg=nimble_mano_vreg)
    return nlayer


def create_logger(dir, name='mylog'):

    # Create a logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create file handler
    fh = logging.FileHandler(f'{dir}/{name}.log')
    fh.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def calculate_photo_loss(segmented_image, rendered_image):
    # Disable all background pixels in the rendered image
    rendered_image[:, :, :, :3][rendered_image[:, :, :, 3] == 0] = 0.
    # Calculate the difference between the two images
    mse_loss = F.mse_loss(segmented_image, rendered_image[:, :, :, :3]) * 1e4
    
    return mse_loss

def load_texture(textured_pkl, device):

    f_uv = np.load(textured_pkl, allow_pickle=True)

    faces = []
    uvs = []
    for line in f_uv[:-3]: # avoid last 3 elements
        if 'vt' in line:
            uv = line[:-1].split(' ')
            u = float(uv[1]) 
            v = float(uv[2])
            uvs.append([u, v])
        if 'f' in line:
            face_sides = line[:-1].split(' ')
            face = [0] * 3
            for i in range(1, 4):
                # Face
                face_side = face_sides[i].split('/')
                vt = int(face_side[1]) - 1
                face[i-1] = vt

            faces.append(face)
    
    return torch.tensor(faces).to(device).unsqueeze(0), torch.tensor(uvs, dtype=torch.float32).to(device).unsqueeze(0)


def transform_vertices(vertices, rot_matrices, tranlsation_vectors):
    # vertices: (N, V, 3)
    # rot_matrices: (N, 3, 3)
    # tranlsation_vectors: (N, 3)
    # return: (N, V, 3)
    return vertices @ rot_matrices + tranlsation_vectors.unsqueeze(1)

def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''

    proj_pts = pts3D @ cam_mat.T
    proj_pts2d = proj_pts[:, :, :2] / proj_pts[:, :, 2:]
    
    return proj_pts2d

def load_ycb_obj(ycb_dataset_path, obj_name, rot=None, trans=None):
    ''' Load a YCB mesh based on the name '''
    path = os.path.join(ycb_dataset_path, obj_name, f'textured_simple.obj')
    verts, faces, aux = load_obj(path)
    verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
    faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
    tex_maps = aux.texture_images

    # tex_maps is a dictionary of {material name: texture image}.
    # Take the first image:
    texture_image = list(tex_maps.values())[0]
    texture_image = texture_image[None, ...]  # (1, H, W, 3)

    # Create a textures object
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

    # # apply current pose to the object model
    verts = torch.matmul(verts, rot.T) + trans
    coordChangeMat = torch.tensor([[-1., 0., 0.], [0, 1., 0.], [0., 0., -1.]], dtype=torch.float32)
    verts = torch.matmul(verts, coordChangeMat.T)
    
    # # Initialise the mesh with textures
    obj_mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex.to(device))

    return obj_mesh

def load_sample(data_path, split, seq_name, frame_num, device):
    
    """ Load a sample from the dataset """
    meta_file = os.path.join(data_path, split, seq_name, 'meta', '%04d.pkl' % frame_num)
    data = np.load(meta_file, allow_pickle=True)
    
    pose, shape, trans = data['handPose'], data['handBeta'], data['handTrans']
    pose = torch.tensor(pose, dtype=torch.float32, device=device).unsqueeze(0)
    shape = torch.tensor(shape, dtype=torch.float32, device=device).unsqueeze(0) 
    trans = torch.tensor(trans, dtype=torch.float32, device=device).unsqueeze(0)
    
    img_path = meta_file.replace('meta', 'rgb').replace('pkl', 'jpg')
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    obj_mesh = load_ycb_obj('../ycb_models', data['objName'], torch.tensor(cv2.Rodrigues(data['objRot'])[0]), torch.tensor(data['objTrans']))
    
    hand_verts3d, hand_joints3d = mano_layer(pose, shape, trans)
    hand_verts2d = project_3D_points(cam_intr, hand_verts3d)
    hand_verts2d = hand_verts2d[0].cpu().detach().numpy()

    return img, pose, shape, trans, hand_verts3d, hand_verts2d, obj_mesh


def visualize_image_with_mesh(img, rendered_image):

    # Load the RGB image
    # Create a figure and axes for plotting
    ax = plt.subplot(1, 2, 1)
    ax.imshow(img)
    
    rendered_image[:, :, :, 3][rendered_image[:, :, :, 3] != 0] = 1.
    # rendered_image[:, :, :, :3][rendered_image[:, :, :, 3] == 0] = 0.
    
    img2 = rendered_image[0].detach().cpu().numpy()
    # plt.imshow(img2[:, :, :3])
    ax = plt.subplot(1, 3, 2)
    ax.imshow(img2)

    # ax = plt.subplot(1, 3, 3)

    # img3 = seg_mask[0].detach().cpu().numpy()

    # ax.imshow(img3)
    # plt.show()

    # Show the plot
    plt.savefig('output/projection.png')
    plt.clf()
    plt.close()

L1 = torch.nn.L1Loss()
L2 = torch.nn.MSELoss()
CEL = torch.nn.CrossEntropyLoss()

def calculate_pose_loss(outputs, targets, target_idx=0, mode='loss'):
    left_pose_gt = targets['left_pose3d'][:, target_idx]
    right_pose_gt = targets['right_pose3d'][:, target_idx]

    output_target_idx = target_idx if outputs.shape[1] > 1 else 0
    left_pose_loss = mpjpe(outputs[:, output_target_idx, :21], left_pose_gt)
    right_pose_loss = mpjpe(outputs[:, output_target_idx, 21:42], right_pose_gt)
    pose_loss = left_pose_loss + right_pose_loss

    errors = [left_pose_loss * 1000, right_pose_loss * 1000]

    if outputs.shape[2] > 42:
        top_pose_gt = targets['top_kps3d'][:, target_idx]
        bottom_pose_gt = targets['bottom_kps3d'][:, target_idx]
        top_pose_loss = mpjpe(outputs[:, output_target_idx, 42:63], top_pose_gt)
        bottom_pose_loss = mpjpe(outputs[:, output_target_idx, 63:], bottom_pose_gt)
        pose_loss += top_pose_loss + bottom_pose_loss
        errors += [top_pose_loss * 1000, bottom_pose_loss * 1000]

    return pose_loss if mode == 'loss' else errors

def calculate_loss(outputs, targets, target_idx=0):
    if not isinstance(outputs, dict):
        return calculate_pose_loss(outputs[0], targets, target_idx)

    losses = {}
    hw = 0.01
    for side in ['left', 'right']:
        if outputs[f'{side}_pose'].shape[1] > 1:
            mano_gt = targets[f'{side}_pose'], targets[f'{side}_shape'][:, 0], targets[f'{side}_trans']#, targets[f'{side}_pose3d']
            mano_pred = outputs[f'{side}_pose'], outputs[f'{side}_shape'], outputs[f'{side}_trans']#, outputs[f'{side}_pose3d']
        else:
            mano_gt = targets[f'{side}_pose'][:, target_idx], targets[f'{side}_shape'][:, 0], targets[f'{side}_trans'][:, target_idx]#, targets[f'{side}_pose3d'][:, target_idx]
            mano_pred = outputs[f'{side}_pose'][:, 0], outputs[f'{side}_shape'][:, 0], outputs[f'{side}_trans'][:, 0]#, outputs[f'{side}_pose3d'][:, 0]

        loss = sum(L1(mano_gt[i], mano_pred[i]) * hw for i in range(len(mano_gt)))
        losses[f'{side}_mano'] = loss

    obj_pose_pred, obj_class = outputs['obj_pose'], outputs['obj_class']
    obj_pose_gt = targets['obj_pose']

    ow = 0.1
    loss = L1(obj_pose_gt, obj_pose_pred) * ow if obj_pose_pred.shape[1] > 1 else L1(obj_pose_gt[:, target_idx], obj_pose_pred[:, 0]) * ow
    # if outputs['bottom_kps3d'].shape[1] > 1:
    #     top_pose_loss = mpjpe(outputs['top_kps3d'], targets['top_kps3d'])
    #     bottom_pose_loss = mpjpe(outputs['bottom_kps3d'], targets['bottom_kps3d'])
        
    # else:
    #     top_pose_loss = mpjpe(outputs['top_kps3d'][:, 0], targets['top_kps3d'][:, target_idx])
    #     bottom_pose_loss = mpjpe(outputs['bottom_kps3d'][:, 0], targets['bottom_kps3d'][:, target_idx])

    # loss += top_pose_loss + bottom_pose_loss 
    losses['obj'] = loss
        
    total_loss = sum(loss for loss in losses.values())

    return total_loss

def calculate_error(outputs, targets, dataset, target_idx, model):
    bs, t = targets[f'left_pose'].shape[:2]

    metrics = {}

    if not isinstance(outputs, dict):
        errors = calculate_pose_loss(outputs[0], targets, target_idx, mode='error')

        metrics['lpc'] = errors[0]
        metrics['rpc'] = errors[1]

        if len(errors) > 2:
            metrics['tk'] = errors[2]
            metrics['bk'] = errors[3]

        return metrics

    # Calculate hand mesh error
    cam_ext = targets['cam_ext'].unsqueeze(1).view(bs * t, 4, 4)

    for side in ['left', 'right']:
        mano_gt = [targets[f'{side}_pose'], targets[f'{side}_shape'], targets[f'{side}_trans']]
        mano_pred = [outputs[f'{side}_pose'], outputs[f'{side}_shape'], outputs[f'{side}_trans']]

        mano_pred[1] = mano_pred[1].repeat(1, t, 1)
        if mano_pred[2].shape[1] == 1:
            mano_pred[2] = mano_pred[2].repeat(1, t, 1)
            mano_pred[0] = mano_pred[0].repeat(1, t, 1)

        for i in range(len(mano_gt)):
            mano_gt[i] = mano_gt[i].view(bs * t, mano_gt[i].shape[-1])
            mano_pred[i] = mano_pred[i].view(bs * t, mano_pred[i].shape[-1])

        mesh_gt, pose_gt = model.decode_mano(mano_gt[0], mano_gt[1], mano_gt[2], side, cam_ext)
        mesh_pred, pose_pred = model.decode_mano(mano_pred[0], mano_pred[1], mano_pred[2], side, cam_ext)
        
        mesh_gt = mesh_gt.view(bs, t, -1, 3)
        mesh_pred = mesh_pred.view(bs, t, -1, 3)

        pose_gt = pose_gt.view(bs, t, -1, 3)
        pose_pred = pose_pred.view(bs, t, -1, 3)

        # Calculate hand mesh error for only the middle frame or the last frame
        mesh_err = mpjpe(mesh_pred[:, target_idx], mesh_gt[:, target_idx]) * 1000
        pose_err = mpjpe(pose_pred[:, target_idx], pose_gt[:, target_idx]) * 1000
        
        pose3d_err = mpjpe(outputs[f'{side}_pose3d'][:, target_idx], targets[f'{side}_pose3d'][:, target_idx]) * 1000

        metrics[f'{side[0]}m'] = mesh_err
        metrics[f'{side[0]}p'] = pose_err
        metrics[f'{side[0]}pc'] = pose3d_err

    obj_pose, obj_class = outputs['obj_pose'], outputs['obj_class']
    if obj_pose.shape[1] == 1:
        obj_pose = obj_pose.repeat(1, t, 1)

    # Calculate object class accuracy
    pred_labels = torch.argmax(obj_class, dim=1)
    pred_object_names = [dataset.object_names[l] for l in pred_labels]
    acc = (pred_labels == targets['label'][:, 0]).float().mean()
    metrics['acc'] = acc

    # Calculate object mesh error
    obj_pred = obj_pose[:, :, :1], obj_pose[:, :, 1:4], obj_pose[:, :, 4:]
    obj_pose_gt = targets['obj_pose']
    obj_gt = obj_pose_gt[:, :, :1], obj_pose_gt[:, :, 1:4], obj_pose_gt[:, :, 4:]
    
    object_names = [dataset.object_names[l] for l in targets['label'][:, 0]]

    for i in range(len(object_names)):
        cam_ext_i = cam_ext.view(bs, t, 4, 4)[i]
        obj_verts_pred, _ = dataset.transform_obj(pred_object_names[i], obj_pred[0][i], obj_pred[1][i], obj_pred[2][i], cam_ext_i)
        obj_verts_gt, _ = dataset.transform_obj(object_names[i], obj_gt[0][i], obj_gt[1][i], obj_gt[2][i], cam_ext_i)
        for part in ['top', 'bottom']:
            if obj_verts_pred[part].shape[1] != obj_verts_gt[part].shape[1]:
                # Calculate chamfer distance in case of wrong classification
                obj_mesh_err = chamfer_distance(obj_verts_pred[part][target_idx].unsqueeze(0), obj_verts_gt[part][target_idx].unsqueeze(0))[0] * 1000
            else:
                obj_mesh_err = mpjpe(obj_verts_pred[part][target_idx], obj_verts_gt[part][target_idx]) * 1000

            metrics[f'{part[0]}m'] = metrics[f'{part}_mesh'] + obj_mesh_err if f'{part}_mesh' in metrics.keys() else obj_mesh_err
            kps_err = mpjpe(outputs[f'{part}_kps3d'][i, target_idx], targets[f'{part}_kps3d'][i, target_idx]) * 1000
            metrics[f'{part[0]}k'] = kps_err

    return metrics

def run_val(valloader, val_count, batch_size, dataset, target_idx, model, logger, e, device, dh=None):

    keys = ['lm', 'lp', 'lpc', 'rm', 'rp', 'rpc', 'tm', 'bm', 'tk', 'bk', 'acc']
    errors = {k: AverageMeter() for k in keys}
    
    master_condition = dh is None or (dh is not None and dh.is_master)
    total_samples = val_count // batch_size if dh is None else val_count // (batch_size * dh.world_size)

    if master_condition: logger.info(f'Running validation for epoch {e}')
    iterable_loader = tqdm(enumerate(valloader), total=total_samples) if master_condition else enumerate(valloader)
    
    for i, (_, data_dict) in iterable_loader:
        if dh is not None and i / total_samples > 0.75: break # Due to unbalanced dataloaders between GPUs

        if data_dict is None: continue

        data_dict['rgb'] = [img_batch.to(device) for img_batch in data_dict['rgb']]

        for k in data_dict.keys():
            data_dict[k] = data_dict[k].to(device) if isinstance(data_dict[k], torch.Tensor) else data_dict[k]

        outputs = model(data_dict)

        # if master_condition:
        model_obj = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        metrics = calculate_error(outputs, data_dict, dataset, target_idx, model_obj)
        if dh is not None: dh.sync_distributed_values(metrics) # For multi-GPU training

        for k in metrics.keys():
            errors[k].update(metrics[k].item(), batch_size)

        if (i+1) % 1000 == 0 and master_condition:
            error_list = [f'{k}: {v.avg:.2f}' for k, v in errors.items()]
            logger.info(f'\nValidation [{i+1}/{total_samples}] Err: {error_list}')

    return errors

def load_weights(model, weights_path):
    start_epoch = 0
    if os.path.isfile(weights_path):
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint)
        start_epoch = int(weights_path.split('/')[-1].split('_')[-1].split('.')[0]) + 1
        model.reload_backbone()
    return model, start_epoch

def get_keypoints(outputs, i):
    keypoints = outputs[i]['keypoints']
    labels = outputs[i]['labels']
    boxes = outputs[i]['boxes']
    left_hand, right_hand, top_object, bottom_object = None, None, None, None
    left_hand_label, right_hand_label, top_object_label, bottom_object_label = None, None, None, None
    left_hand_box, right_hand_box, top_object_box, bottom_object_box = None, None, None, None
    # Hands
    left_hand_idx = torch.where(labels == 22)[0]
    right_hand_idx = torch.where(labels == 23)[0]
    if len(left_hand_idx) > 0:
        left_hand = keypoints[left_hand_idx[0]]
        left_hand_label = labels[left_hand_idx[0]]
        left_hand_box = boxes[left_hand_idx[0]]

    if len(right_hand_idx) > 0:
        right_hand = keypoints[right_hand_idx[0]]
        right_hand_label = labels[right_hand_idx[0]]
        right_hand_box = boxes[right_hand_idx[0]]

    # Objects
    top_object_idx = torch.where((labels < 22) & (labels % 2 == 0))[0]
    bottom_object_idx = torch.where((labels < 22) & (labels % 2 == 1))[0]

    if len(top_object_idx) > 0:
        top_object = keypoints[top_object_idx[0]]
        top_object_label = labels[top_object_idx[0]]
        top_object_box = boxes[top_object_idx[0]]

    if len(bottom_object_idx) > 0:
        bottom_object = keypoints[bottom_object_idx[0]]
        bottom_object_label = labels[bottom_object_idx[0]]
        bottom_object_box = boxes[bottom_object_idx[0]]

    kps = left_hand, right_hand, top_object, bottom_object
    labels = left_hand_label, right_hand_label, top_object_label, bottom_object_label
    boxes = left_hand_box, right_hand_box, top_object_box, bottom_object_box

    return kps, labels, boxes


def load_model(args, device, target_idx):
    from models.model_poseformer import PoseTransformer
    from models.thor import THOR
    from models.Stohrmer import Stohrmer
    from models.ShapeTHOR import ShapeTHOR
    from models.TemporalTHOR import TemporalTHOR
    if args.model_name == 'stohrmer':
        model = Stohrmer(device, num_kps=42, num_frames=args.window_size).to(device)
    elif args.model_name == 'poseformer':
        model = PoseTransformer(num_frame=args.window_size, num_joints=42, in_chans=2).to(device)
    elif args.model_name == 'thor':
        if args.window_size == 1:
            model = THOR(device, input_dim=args.input_dim, num_frames=args.window_size, num_kps=84, rcnn_path=args.backbone_path, target_idx=target_idx).to(device)
        else:
            model = TemporalTHOR(device, input_dim=args.input_dim, num_frames=args.window_size, num_kps=84, thor_path=args.backbone_path).to(device)
    
    elif args.model_name == 'shapethor':
        model = ShapeTHOR(device, input_dim=args.input_dim, num_frames=args.window_size, num_kps=84, thor_path=args.backbone_path, target_idx=target_idx).to(device)
    return model
