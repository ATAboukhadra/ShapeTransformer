import matplotlib.pyplot as plt
import torch
import numpy as np
from render_utils import create_renderer, render_arctic_mesh
from vis_utils import showHandJoints
from dataset.arctic_pipeline import create_pipe, batch_samples
from tqdm import tqdm

np.set_printoptions(precision=2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

colors = {'right': 'blue', 'left': 'red', 'top': 'green', 'bottom': 'yellow'}

root = '/ds-av/public_datasets/arctic/td_p1_sequential_nocropped/'
objects_root = 'dataset/arctic_objects'
batch_size = 1
num_workers = 0
sliding_window_size = 9

train_pipeline, decoder, factory = create_pipe(root, objects_root, 'train', device, sliding_window_size)
trainloader = torch.utils.data.DataLoader(train_pipeline, batch_size=batch_size, num_workers=num_workers, collate_fn=batch_samples)

val_pipeline, _, _ = create_pipe(root, objects_root, 'val', device, sliding_window_size, factory=factory, arctic_decoder=decoder)
valloader = torch.utils.data.DataLoader(train_pipeline, batch_size=batch_size, num_workers=num_workers, collate_fn=batch_samples)

# dataset = ArcticDataset(root, objects_root, device=device)
dataset = decoder.dataset
hand_faces = dataset.hand_faces

for i, data_dict in enumerate(trainloader):

    if data_dict['articulation'][0][0] < 0.2:
        continue
    
    cam_int = data_dict['cam_int'][0]
    fx, fy, cx, cy = cam_int[0, 0], cam_int[1, 1], cam_int[0, 2], cam_int[1, 2]
    K = torch.tensor([[[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 0, 1], [0, 0, 1, 0]]], device=device)
    # image_sizes = [(img.shape[0], img.shape[1]) for img in data_dict['img'][0]]
    image_sizes = (data_dict['img'][0][0].shape[-2:], )
    renderer = create_renderer(K, device, image_size=image_sizes)

    # plt.imshow(img)
    # Plot 2D hand pose

    # plt.show()
    articulation = data_dict['articulation'][0]
    rot = data_dict['rot'][0]
    trans = data_dict['trans'][0]
    object_name = data_dict['object_name'][0]
    cam_ext = data_dict['cam_ext'][0]
    # print('Sample Name:', data_dict['key'][0], 'Articulation: ', art, ' Rotation:', rot, 'Translation:', trans)

    obj_verts = dataset.transform_obj(object_name, articulation, rot, trans, cam_ext)

    for i in tqdm(range(data_dict['img'][0].shape[0])):
        img = data_dict['img'][0][i].cpu().numpy().transpose(1, 2, 0)
        img = np.ascontiguousarray(img * 255, np.uint8)

        if data_dict['hands_pose2d'][0][0].shape[0] > 0:
            for hand in data_dict['hands_pose2d'][0][i]:
                img = showHandJoints(img, hand.cpu().numpy())
        plt.subplot(3, 3, i+1)
        plt.imshow(img)

        verts_list, faces_list, textures_list = [], [], []
    
        for mesh in colors.keys():

            verts = data_dict[f'{mesh}_verts'][0][i].unsqueeze(0) if mesh in ['right', 'left'] else obj_verts[mesh][i].unsqueeze(0)
            faces = dataset.objects[object_name][mesh][1].verts_idx if mesh in ['top', 'bottom'] else torch.tensor(hand_faces[mesh], device=device)
            texture = dataset.objects[object_name][mesh][2] if mesh in ['top', 'bottom'] else None
            verts_list.append(verts)
            faces_list.append(faces)
            textures_list.append(texture)        

        # rendered_image, _ = render_arctic_mesh(verts_list, faces_list, textures_list, renderer)
        # plt.subplot(1, 2, 2)
        # plt.imshow(rendered_image[0].cpu().numpy())
        plt.axis('off')
    plt.show()

