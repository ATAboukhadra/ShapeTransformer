import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import json
import cv2
from arctic_dataset import ArcticDataset
from torch.utils.data import DataLoader
from render_utils import create_renderer, render_arctic_mesh
from mp_hand_segmenter import detect_hand
from vis_utils import showHandJoints
from tqdm import tqdm
np.set_printoptions(precision=2)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

colors = {'right': 'blue', 'left': 'red', 'top': 'green', 'bottom': 'yellow'}

root = '../arctic/data/arctic_data/data'
objects_root = '../arctic_objects'

img = cv2.cvtColor(cv2.imread(os.path.join(root, 'images/s01/box_use_02/1/00175.jpg')), cv2.COLOR_BGR2RGB)
_, x, y, crop, _,_ = detect_hand(img, segment=False)
plt.imshow(img)
# plt.scatter(x[0], y[0], c='r')
# plt.scatter(x[1], y[1], c='b')
# plt.show()
# print('------')
# splits = np.load(os.path.join(root, ''), allow_pickle=True).item()
# p1 = json.load(open(os.path.join(root, 'splits_json/protocol_p1.json')))
# p2 = json.load(open(os.path.join(root, 'splits_json/protocol_p2.json')))

dataset = ArcticDataset(root, objects_root, device=device)
hand_faces = dataset.hand_faces

loader = DataLoader(dataset, batch_size=1, shuffle=True)
iterable = iter(loader)

for i in tqdm(range(10000)):
    data_dict = next(iterable)
    if data_dict['articulation'][0] < 0.2:
        continue

    cam_int = data_dict['cam_int'][0]
    fx, fy, cx, cy = cam_int[0, 0], cam_int[1, 1], cam_int[0, 2], cam_int[1, 2]
    K = torch.tensor([[[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 0, 1], [0, 0, 1, 0]]], device=device)
    image_sizes = [(img.shape[0], img.shape[1]) for img in data_dict['img']]
    renderer = create_renderer(K, device, image_size=image_sizes)

    img = data_dict['img'][0].cpu().numpy()
    # plt.imshow(img)
    if data_dict['hands_pose2d'][0].shape[0] > 0:
        for hand in data_dict['hands_pose2d'][0]:
            img = showHandJoints(img, hand.cpu().numpy())
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    # plt.show()
    art = data_dict['articulation'].cpu().numpy()
    rot = data_dict['rot'][0].cpu().numpy()
    trans = data_dict['trans'][0].cpu().numpy()
    print('Sample Name:', data_dict['key'][0], 'Articulation: ', art, ' Rotation:', rot, 'Translation:', trans)
    # print('Articuldata_dict['articulation'][0], data_dict['rot'][0], data_dict['trans'][0])
    # Plot 2D hand pose
    pose2d = data_dict['hands_pose2d'][0].cpu().numpy().reshape(-1, 2)
    # print(pose2d.shape, pose2d)
    # plt.scatter(pose2d[:, 0], pose2d[:, 1], c='r')

    verts_list, faces_list, textures_list = [], [], []
    for mesh in colors.keys():

        verts = data_dict[f'{mesh}_verts_cam']
        object_name = data_dict['object_name'][0]
        faces = dataset.objects[object_name][mesh][1].verts_idx if mesh in ['top', 'bottom'] else torch.tensor(hand_faces[mesh], device=device)
        texture = dataset.objects[object_name][mesh][2] if mesh in ['top', 'bottom'] else None
        verts_list.append(verts)
        faces_list.append(faces)
        textures_list.append(texture)        

    rendered_image, _ = render_arctic_mesh(verts_list, faces_list, textures_list, renderer)
    plt.subplot(1, 2, 2)
    plt.imshow(rendered_image[0].cpu().numpy())
    plt.axis('off')
    plt.show()

