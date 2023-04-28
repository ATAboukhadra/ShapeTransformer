import time
from dataset.DexYCB_dataset import DexYCBDataset
import numpy as np
from io import BytesIO
import torch
import os
import yaml
import cv2
from torch.utils.data.datapipes.datapipe import DataChunk

dexycb_dataset_dict = {'train': None, 'val': None, 'test': None}

def decode_rgb(extension: str, data: bytes):
    # rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    # print('rgb', rgb.shape, rgb.dtype)

    return rgb

def calculate_corners(verts):

    min_coords = torch.min(verts, dim=0).values
    max_coords = torch.max(verts, dim=0).values

    # Create the bounding box vertices
    obj_bb = torch.Tensor([    
        [min_coords[0], min_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]],
    ])

    return obj_bb

def decode_labels(extension: str, data: bytes):

    splits = extension.split('_')
    subject_id = splits[-4].split('/')[-1]
    seq_id = subject_id + '/' + splits[-3] + '_' + splits[-2]
    cam_id = splits[-1].split('.')[0]
    subset = extension.split('/')[-3]
    dexycb_dataset = dexycb_dataset_dict[subset]
    # Extract per sequence annotations
    intr = dexycb_dataset.depth_intrinsics[cam_id]
    cam_mat = torch.Tensor([
        [intr['fx'], 0, intr['ppx']],
        [0, intr['fy'], intr['ppy']],
        [0, 0, 1]
    ])
    meta = dexycb_dataset._meta_dict[seq_id]
    visible_objs = meta['ycb_ids']
    grasp_ind = meta['ycb_grasp_ind']
    object_id = visible_objs[grasp_ind]
    object_name = dexycb_dataset.YCB_CLASSES[object_id]
    
    # Extract per-frame annotations    
    labels = np.load(BytesIO(data))
    obj_pose = labels['pose_y'][grasp_ind]
    rotation, translation = obj_pose[:, :3], obj_pose[:, 3]

    mano_params = labels['pose_m']
    hand_pose2d = labels['joint_2d']
    hand_pose3d = labels['joint_3d'][0] * 1000
    
    mano_calib = dexycb_dataset.mano_calibration(meta)
    mano_betas = mano_calib['betas']
    mano_side = meta['mano_sides'][0]

    hand_verts, _ = dexycb_dataset.decode_mano(mano_params, mano_betas, mano_side)

    annotations = {
        'cam_mat': cam_mat, 
        'hand_pose2d': hand_pose2d,
        'hand_pose3d': hand_pose3d,
        'hand_verts3d': hand_verts
    }
    return annotations

def decode_depth(extension: str, data: bytes):
    """Decode the depth image to depth map in millimeters"""
    # dpt = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)

    # # threshold at 2 meters
    # depth_threshold = 2000
    # dpt[dpt>depth_threshold] = depth_threshold
    # dpt[dpt==0] = depth_threshold
    # depth = torch.from_numpy(dpt.astype(np.float32)) 
    # print('depth', depth.shape, depth.dtype)
    depth = torch.zeros((480, 640), dtype=torch.float32)
    return depth

def preprocess_sequential(sample: DataChunk):
    seq_samples = [preprocess(s) for s in sample]

    # Concatenate the tensors for each key across all dictionaries
    keys = list(seq_samples[0].keys())
    temporal_sample = {}
    for key in keys:
        if isinstance(seq_samples[0][key], torch.Tensor):
            temporal_sample[key] = torch.cat([s[key] for s in seq_samples], dim=0)
        else:
            temporal_sample[key] = np.concatenate([d[key] for d in seq_samples], axis=0)
    return temporal_sample

def preprocess(sample: DataChunk):
    for i in range(len(sample)):
        if 'color' in sample[i][0]:
            color_idx = i
        if 'depth' in sample[i][0]:
            depth_idx = i
        if 'labels' in sample[i][0]:
            labels_idx = i
    
    subset = sample[labels_idx][0].split('/')[-3]
    dexycb_dataset = dexycb_dataset_dict[subset]
    
    depth = sample[depth_idx][1]
    labels = sample[labels_idx][1]
    rgb = sample[color_idx][1]
    cam_mat = labels['cam_mat']
    # print(labels['hand_pose2d'][0])
    boxes, lbls = dexycb_dataset.calculate_bounding_box(torch.Tensor(labels['hand_pose2d'][0]))
    # print(boxes)
    fx, fy, ux, uy = cam_mat[0, 0], cam_mat[1, 1], cam_mat[0, 2], cam_mat[1, 2]
    
    sample = {
        'images': rgb.reshape((1, *rgb.shape)),
        'boxes': torch.Tensor(boxes).unsqueeze(0),
        'pose2d': torch.Tensor(labels['hand_pose2d']),
        'pose3d': torch.Tensor(labels['hand_pose3d']).unsqueeze(0)
    }
    return sample