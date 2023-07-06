import matplotlib.pyplot as plt
import torch
import numpy as np
from render_utils import create_renderer, render_arctic_mesh
from vis_utils import showHandJoints
from dataset.arctic_pipeline import create_pipe, batch_samples
from tqdm import tqdm
from torchvision.transforms.functional import resize

np.set_printoptions(precision=2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

colors = {'right': 'blue', 'left': 'red', 'top': 'green', 'bottom': 'yellow'}

root = '/ds-av/public_datasets/arctic/td_p1_sequential_nocropped/'
objects_root = 'dataset/arctic_objects'
batch_size = 1
num_workers = 1
sliding_window_size = 9
scale_factor = 4

train_pipeline, num_samples, decoder, factory = create_pipe(root, objects_root, 'train', 'cpu', sliding_window_size)
trainloader = torch.utils.data.DataLoader(train_pipeline, batch_size=batch_size, num_workers=0, collate_fn=batch_samples)

val_pipeline, _, _, _ = create_pipe(root, objects_root, 'val', 'cpu', sliding_window_size, factory=factory, arctic_decoder=decoder)
valloader = torch.utils.data.DataLoader(train_pipeline, batch_size=batch_size, num_workers=0, collate_fn=batch_samples)

# dataset = ArcticDataset(root, objects_root, device=device)
dataset = decoder.dataset
hand_faces = dataset.hand_faces

for i, data_dict in tqdm(enumerate(trainloader), total=num_samples // batch_size):
    articulation = data_dict['obj_pose'][0][0][0]
    if articulation < 0.1 or data_dict['img'][0][0].shape[-2:][0] == 700:
        continue
    
    data_dict['img'] = [torch.stack(img_batch, dim=0).to(device) for img_batch in data_dict['img']]

    cam_int = data_dict['cam_int'][0]
    fx, fy, cx, cy = cam_int[0, 0] / scale_factor, cam_int[1, 1] / scale_factor, cam_int[0, 2] / scale_factor, cam_int[1, 2] / scale_factor
    K = torch.tensor([[[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 0, 1], [0, 0, 1, 0]]], device=device)
    image_sizes = ((data_dict['img'][0][0].shape[-2:]), )
    fig_dim = (3, 3) if image_sizes[0][0] != 700 else (1, 9)
    renderer = create_renderer(K, device, image_size=image_sizes)

    # Plot 2D hand pose
    obj_pose = data_dict['obj_pose'][0]
    articulation = obj_pose[:, 0].unsqueeze(1)
    rot = obj_pose[:, 1:4]
    trans = obj_pose[:, 4:] / 1000

    object_name = data_dict['object_name'][0]
    cam_ext = data_dict['cam_ext'][0]
    obj_verts = dataset.transform_obj(object_name, articulation, rot, trans, cam_ext)

    for i in tqdm(range(data_dict['img'][0].shape[0])):
        img = data_dict['img'][0][i].cpu().numpy().transpose(1, 2, 0)
        img = np.ascontiguousarray(img * 255, np.uint8)

        if data_dict['hands_pose2d'][0][0].shape[0] > 0:
            for hand in data_dict['hands_pose2d'][0][i]:
                img = showHandJoints(img, hand.cpu().numpy() / scale_factor)
        plt.subplot(*fig_dim, i+1)
        plt.imshow(img)

        verts_list, faces_list, textures_list = [], [], []
    
        for mesh in colors.keys():
            if mesh in ['left' , 'right']:
                pose = data_dict[f'{mesh}_pose'][0][i].unsqueeze(0)
                shape = data_dict[f'{mesh}_shape'][0].unsqueeze(0)
                trans = data_dict[f'{mesh}_trans'][0][i].unsqueeze(0)
                verts = dataset.decode_mano(pose, shape, trans, mesh, cam_ext[i].unsqueeze(0))[0].to(device)
            else:
                verts = obj_verts[mesh][i].unsqueeze(0).to(device)

            faces = dataset.objects[object_name][mesh][1].verts_idx if mesh in ['top', 'bottom'] else torch.tensor(hand_faces[mesh], device=device)
            texture = dataset.objects[object_name][mesh][2] if mesh in ['top', 'bottom'] else None
            verts_list.append(verts)
            faces_list.append(faces)
            textures_list.append(texture)        

        rendered_image, _ = render_arctic_mesh(verts_list, faces_list, textures_list, renderer)
        plt.imshow(rendered_image[0].cpu().numpy())
        plt.axis('off')
    plt.show()

