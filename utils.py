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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_path = '/home2/HO3D_v3'
cam_intr = torch.tensor([
  [614.627,   0.,    320.262],
 [  0. ,   614.101 ,238.469],
 [  0. ,     0. ,     1.   ]] 
, device=device
 )
mano_layer = ManoLayer(mano_root='mano_v1_2/models', ncomps=45, flat_hand_mean=True, use_pca=False).to(device)

def parse_args():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, help="Directory containing data", default='/data/ho')
    ap.add_argument("--meta_root", type=str, help="Directory containing additional data", default='/data/DexYCB')
    ap.add_argument("--output_folder", type=str, help="relative path to save checkpoint", default='checkpoints/transformer')
    ap.add_argument("--batch_size", type=int, help="batch size", default='8')
    ap.add_argument("--log_interval", type=int, help="How many batches needes to log", default='50')
    ap.add_argument("--epochs", type=int, help="number of epochs after which to stop", default='10')
    ap.add_argument("--window_size", type=int, help="How many batches needes to log", default='9')
    ap.add_argument("--skip", type=int, help="how many frames to jump", default='1')
    ap.add_argument("--d_model", type=int, help="number of features in transformer", default='32')
    ap.add_argument("--num_workers", type=int, help="number of workers", default='8')
    ap.add_argument("--hdf5", action='store_true', help="Load data from HDF5 file") 
    ap.add_argument("--causal", action='store_true', help="Use only previous frames")     
    ap.add_argument("--pretrained_model", type=str, help="path to pretrained weights")     

    return ap.parse_args()

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

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


def create_logger(dir):

    # Create a logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create file handler
    fh = logging.FileHandler(f'{dir}/mylog.log')
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

def calculate_loss(outputs, targets):
    losses = {}
    w = 0.1
    for side in ['left', 'right']:
        mano_gt = targets[f'{side}_pose'], targets[f'{side}_shape'], targets[f'{side}_trans']
        mano_pred = outputs[f'{side}_pose'], outputs[f'{side}_shape'], outputs[f'{side}_trans']
        loss = sum(L2(mano_gt[i], mano_pred[i]) * w for i in range(len(mano_gt)))
        losses[f'{side}_mano'] = loss

    obj_pose_pred, obj_class = outputs['obj_pose'], outputs['obj_class']
    obj_pose_gt = targets['obj_pose']

    # loss = L2(obj_pose_gt, obj_pose_pred) * w 
    # obj_class_loss = CEL(obj_class, targets['label'])
    # loss += obj_class_loss
    # losses['obj'] = loss
        
    total_loss = sum(loss for loss in losses.values())

    return total_loss

def calculate_error(outputs, targets, errors, dataset, target_idx, model):

    # Calculate hand mesh error
    bs, t = targets[f'left_pose'].shape[:2]
    cam_ext = targets['cam_ext'].unsqueeze(1).view(bs * t, 4, 4)

    for side in ['left', 'right']:
        mano_gt = [targets[f'{side}_pose'], targets[f'{side}_shape'], targets[f'{side}_trans']]
        mano_pred = [outputs[f'{side}_pose'], outputs[f'{side}_shape'], outputs[f'{side}_trans']]

        mano_gt[1] = mano_gt[1].unsqueeze(1).repeat(1, t, 1)
        mano_pred[1] = mano_pred[1].unsqueeze(1).repeat(1, t, 1)

        for i in range(len(mano_gt)):
            mano_gt[i] = mano_gt[i].view(bs * t, mano_gt[i].shape[-1])
            mano_pred[i] = mano_pred[i].view(bs * t, mano_pred[i].shape[-1])

        mesh_gt, pose_gt = model.decode_mano(mano_gt[0], mano_gt[1], mano_gt[2], side, cam_ext)
        mesh_pred, pose_pred = model.decode_mano(mano_gt[0], mano_pred[1], mano_pred[2], side, cam_ext)

        # Calculate hand mesh error for only the middle frame or the last frame
        mesh_err = mpjpe(mesh_pred[target_idx], mesh_gt[target_idx]) * 1000
        pose_err = mpjpe(pose_pred[target_idx], pose_gt[target_idx]) * 1000
        errors[f'{side}_mesh_err'].update(mesh_err.item(), bs)
        errors[f'{side}_pose_err'].update(pose_err.item(), bs)

    obj_pose, obj_class = outputs['obj_pose'], outputs['obj_class']

    # Calculate object class accuracy
    # pred_labels = torch.argmax(obj_class, dim=1)
    # pred_object_names = [dataset.object_names[l] for l in pred_labels]
    # acc = (pred_labels == targets['label']).float().mean()
    # errors['obj_acc'].update(acc, bs)

    # # Calculate object mesh error
    # obj_pred = obj_pose[:, :, :1], obj_pose[:, :, 1:4], obj_pose[:, :, 4:]
    # obj_pose_gt = targets['obj_pose']
    # obj_gt = obj_pose_gt[:, :, :1], obj_pose_gt[:, :, 1:4], obj_pose_gt[:, :, 4:]
    
    # object_names = [dataset.object_names[l] for l in targets['label']]

    # for i in range(len(object_names)):
    #     cam_ext_i = cam_ext.view(bs, t, 4, 4)[i]
    #     obj_verts_pred, _ = dataset.transform_obj(object_names[i], obj_pred[0][i], obj_pred[1][i], obj_pred[2][i], cam_ext_i)
    #     obj_verts_gt, _ = dataset.transform_obj(pred_object_names[i], obj_gt[0][i], obj_gt[1][i], obj_gt[2][i], cam_ext_i)
    #     for part in ['top', 'bottom']:
    #         if obj_verts_pred[part].shape[1] != obj_verts_gt[part].shape[1]:
    #             continue
    #         obj_mesh_err = mpjpe(obj_verts_pred[part][target_idx], obj_verts_gt[part][target_idx]) * 1000
    #         errors[f'{part}_obj_err'].update(obj_mesh_err.item(), 1)

    return errors
