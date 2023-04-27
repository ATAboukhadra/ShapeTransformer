import torch
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import logging

from nimble.utils import batch_to_tensor_device
from nimble.NIMBLELayer import NIMBLELayer

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

def init_nimble():
    device = torch.zeros(1).device

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