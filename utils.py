import torch
import cv2
import matplotlib.pyplot as plt
import argparse

def parse_args():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, help="Directory containing additional data", default='/data/ho')
    ap.add_argument("--output_folder", type=str, help="relative path to save checkpoint", default='checkpoints/transformer')
    ap.add_argument("--batch_size", type=int, help="batch size", default='8')
    ap.add_argument("--log_interval", type=int, help="How many batches needes to log", default='50')
    ap.add_argument("--epochs", type=int, help="number of epochs after which to stop", default='10')
    ap.add_argument("--window_size", type=int, help="How many batches needes to log", default='9')
    ap.add_argument("--skip", type=int, help="how many frames to jump", default='1')
    ap.add_argument("--d_model", type=int, help="number of features in transformer", default='128')
    ap.add_argument("--num_workers", type=int, help="number of features in transformer", default='8')
    ap.add_argument("--hdf5", action='store_true', help="Load data from HDF5 file") 
    ap.add_argument("--causal", action='store_true', help="Use only previous frames")     

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

def plot_sample(batch):
    
    fig = plt.figure(figsize=(15, 15))
    for i in range(batch['images'].shape[0]):
        im = batch['images'][i].cpu().detach().numpy()
        # img = img.transpose(1, 2, 0) * 255
        # img = np.ascontiguousarray(img, np.uint8) 
        for bb in batch['boxes'][i]:
            bb = bb.detach().numpy()
            pt1 = (bb[0], bb[1])
            pt2 = (bb[2], bb[3])
            cv2.rectangle(im, pt1, pt2, color=(255, 0, 0))
        ax = fig.add_subplot(3, 3, i+1)
        ax.imshow(im)

# for i in range(batch['patches'].shape[0]):
#     patch = batch['patches'][i]
#     # print(patch.shape)
#     ax = fig.add_subplot(5, 5, i+1)
#     ax.imshow(patch)
    # print(batch['patches'].shape)

# plt.imshow(im)
# plt.show()
