# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import os
import torch.utils.data as data
import cv2
import os.path
import io
import torch 
import h5py
from PIL import Image
# from utils.utils import calculate_bounding_box, create_rcnn_data


class Dataset(data.Dataset):
    """# Dataset Class """

    def __init__(self, root='./', load_set='train', transform=None, num_keypoints=21,  hdf5=False, causal=True, T=1, skip=1, num_kps=21, patch_size=16):

        self.root = root
        self.transform = transform
        self.num_keypoints = num_keypoints
        self.hdf5_file = None
        if hdf5:
            self.hdf5_file = h5py.File(os.path.join(root, f'images-{load_set}.hdf5'), 'r')

        self.causal = causal
        self.T = T
        self.skip = skip
        self.num_kps = num_kps
        self.patch_size = patch_size
        # TODO: add depth transformation
        self.load_set = load_set  # 'train','val','test'
        self.images = np.load(os.path.join(root, 'images-%s.npy' % self.load_set))
        self.points2d = np.load(os.path.join(root, 'points2d-%s.npy' % self.load_set))
        self.points3d = np.load(os.path.join(root, 'points3d-%s.npy' % self.load_set))

        self.mesh2d = np.load(os.path.join(root, 'mesh2d-%s.npy' % self.load_set))
        self.mesh3d = np.load(os.path.join(root, 'mesh3d-%s.npy' % self.load_set))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """

        image_path = self.images[index]
        temporal_images, temporal_pose2d, temporal_boxes, temporal_pose3d = self.temporal_batch(index, self.T)
        
        # inputs = self.transform(original_image)  # [:3]
        # if self.load_set != 'test':
            # Loading 2D Mesh for bounding box calculation
        if self.num_keypoints == 21 or self.num_keypoints == 778: #i.e. hand
            mesh3d = self.mesh3d[index][:778]
        else: # i.e. object
            mesh3d = self.mesh3d[index]
        # boxes, _ = calculate_bounding_box(point2d[:21])
        # patches = create_patches(original_image, boxes)
        data = {
            'path': image_path,
            'images': temporal_images,
            'pose2d': temporal_pose2d,
            'boxes': temporal_boxes,
            # 'patches': patches,
            'pose3d': temporal_pose3d,
            'mesh3d': mesh3d,
        }

        return data
    
    def read_img(self, index):
        # Load image and apply preprocessing if any
        image_path = self.images[index]

        if self.hdf5_file is not None:
            data = np.array(self.hdf5_file[image_path])
            original_image = np.array(Image.open(io.BytesIO(data)))[..., :3]
        else:
            original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        return original_image

    def __len__(self):
        return len(self.images)


    def single_sample(self, index):
        img = torch.from_numpy(self.read_img(index)).to(torch.float)
        point2d = torch.from_numpy(self.points2d[index][:self.num_kps]).to(torch.float)
        boxes, _ = self.calculate_bounding_box(index)
        boxes = torch.from_numpy(boxes).to(torch.float)
        point3d = torch.from_numpy(self.points3d[index][:self.num_kps]).to(torch.float)

        return img, point2d, boxes, point3d

    def initalize_temporal_tensors(self, index, T):
        
        img, pose2d, boxes, pose3d = self.single_sample(index)

        temporal_images = img.unsqueeze(0).repeat(T, 1, 1, 1)
        temporal_pose2d = pose2d.unsqueeze(0).repeat(T, 1, 1)
        temporal_boxes = boxes.unsqueeze(0).repeat(T, 1, 1)
        temporal_pose3d = pose3d.unsqueeze(0).repeat(T, 1, 1)
        
        return temporal_images, temporal_pose2d, temporal_boxes, temporal_pose3d

    def causal_temporal_batch(self, index, T):
        # In case of live stream we can only use the history
        orig_seq = self.images[index].split("/")[-3]

        temporal_images, temporal_pose2d, temporal_boxes, temporal_pose3d = self.initalize_temporal_tensors(index, T)

        for i in range(1, T):
            if index - (i * self.skip) < 0:
                break
            seq = self.images[index - (i*self.skip)].split("/")[-3]
            if seq != orig_seq:
                break

            prev_img, prev_point2d, boxes, prev_point3d = self.single_sample(index - (i*self.skip))
            temporal_images[T-1-i] = prev_img
            temporal_pose2d[T-1-i] = prev_point2d
            temporal_boxes[T-1-i] = boxes
            temporal_pose3d[T-1-i] = prev_point3d

        return temporal_images, temporal_pose2d, temporal_boxes, temporal_pose3d

    def temporal_batch(self, index, T):
        # In case of vides we can also use future frames

        if self.causal:
            history_images, history_pose2d, history_boxes, history_pose3d = self.causal_temporal_batch(index, T)
            return history_images, history_pose2d, history_boxes, history_pose3d

        history_images, history_pose2d, history_boxes, history_pose3d = self.causal_temporal_batch(index, T//2+1)

        temporal_images, temporal_pose2d, temporal_boxes, temporal_pose3d = self.initalize_temporal_tensors(index, T)
        
        temporal_images[:T//2+1] = history_images
        temporal_pose2d[:T//2+1] = history_pose2d
        temporal_boxes[:T//2+1] = history_boxes
        temporal_pose3d[:T//2+1] = history_pose3d

        orig_seq = self.images[index].split("/")[-3]
        for i in range(1, T//2):
            
            if index + (i * self.skip) >= self.points2d.shape[0]:
                break

            seq = self.images[index + (i * self.skip)].split("/")[-3]
            if seq != orig_seq:
                break
            
            next_img, next_point2d, next_boxes, next_point3d = self.single_sample(index + (i * self.skip))

            temporal_images[T//2 + 1 + i] = next_img
            temporal_boxes[T//2 + 1 + i] = next_boxes
            temporal_pose2d[T//2 + 1 + i] = next_point2d
            temporal_pose3d[T//2 + 1 + i] = next_point3d

        return temporal_images, temporal_pose2d, temporal_boxes, temporal_pose3d



    def calculate_bounding_box(self, index):

        point2d = self.points2d[index][:self.num_kps]

        boxes = []
        labels = []
        size = self.patch_size
        for i in range(point2d.shape[0]):
            x, y = point2d[i]
            boxes.append((x - size, y - size, x + size, y + size))
            labels.append(i+1)

        boxes = np.array(boxes).reshape(-1, 4).astype(int)
        # labels = torch.tensor(labels, dtype=torch.long)

        return boxes, labels 



# def calculate_bounding_box(point2d, increase=False):
#     pad_size = 15
#     x_min = int(min(point2d[:,0]))
#     y_min = int(min(point2d[:,1]))
#     x_max = int(max(point2d[:,0]))
#     y_max = int(max(point2d[:,1]))
    
#     if increase:
#         return np.array([x_min - pad_size, y_min - pad_size, x_max + pad_size, y_max + pad_size])
#     else:
#         return np.array([x_min, y_min, x_max, y_max])

def create_patches(original_image, boxes):
    patches = []
    for bb in boxes:
        patches.append(original_image[bb[1]:bb[3], bb[0]:bb[2]])
    
    return np.array(patches)
