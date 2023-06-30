import torch
import os
import numpy as np
from utils import init_nimble, calculate_photo_loss
from nimble.utils import smooth_mesh, save_textured_nimble, write_obj
from pytorch3d.structures.meshes import Meshes
from pytorch3d.loss import chamfer_distance
from manopth.manolayer import ManoLayer
from pytorch3d.ops import iterative_closest_point
from pytorch3d.transforms import Transform3d
import pytorch3d.transforms as T
import torch.nn.functional as F

import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from utils import load_texture, project_3D_points, visualize_image_with_mesh, load_sample
from render_utils import create_mesh, create_renderer, render_mesh
from models.mp_hand_segmenter import detect_hand
from vis_utils import show3DHandJoints
from pytorch3d.io import load_obj
from pytorch3d.structures import join_meshes_as_scene

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_path = '/home2/HO3D_v3'
cam_intr = torch.tensor([
  [614.627,   0.,    320.262],
 [  0. ,   614.101 ,238.469],
 [  0. ,     0. ,     1.   ]] 
, device=device
 )

# Create a Nimble model
nlayer = init_nimble(device)

# Create a textures object
faces_uvs, verts_uvs = load_texture('nimble/assets/NIMBLE_TEX_FUV.pkl', device)

mano_layer = ManoLayer(mano_root='mano_v1_2/models', ncomps=45, flat_hand_mean=True, use_pca=False).to(device)
handFaces = mano_layer.th_faces.cpu().detach().numpy()

min_dist = 200
min_photo_loss = 100
output_folder = 'output'

# seq_name, frame_num = 'SS2', 885
# seq_name, frame_num = 'ShSu13', 2
seq_name, frame_num = 'ABF10', 882

img, pose, shape, trans, hand_verts, hand_verts2d, obj_mesh = load_sample(data_path, 'train', seq_name, frame_num, device)

segmented_img = detect_hand(img)[-1] / 255
segmented_img_tensor = torch.tensor(segmented_img, dtype=torch.float32).to(device).unsqueeze(0)

coordChangeMat = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32, device=device)       

tex_param = torch.zeros(1, 10, device=device, requires_grad=True)
nimble_params = torch.zeros(1, 50, device=device, requires_grad=True)

params = [nimble_params, tex_param]#, translation, rotation]
optimizer = torch.optim.Adam(params, lr=0.1, weight_decay=0.01)

write_obj(hand_verts[0].cpu().detach().numpy(), handFaces, 'output/mano')

cam_intr_renderer = torch.tensor([[[614.627, 0., 320.262, 0.], [0., 614.101, 238.469, 0.], [0., 0., 0., 1.], [0., 0., 1., 0.]]], device=device)
renderer = create_renderer(cam_intr_renderer, device)


for i in range(1000):
    # NIMBLE
    bn = nimble_params.shape[0]
    # tex_param = torch.rand(bn, 10, device=device) - 0.5
    tex_param.data.clamp_(0, 1)
    nimble_params.data.clamp_(0, 1)
    pose_param = nimble_params[:, :30] * 2 - 1 
    shape_param = nimble_params[:, 30:50] * 2 - 1 
    
    skin_v, muscle_v, bone_v, bone_joints, tex_img = nlayer.forward(pose_param, shape_param, tex_param, handle_collision=True)

    output = iterative_closest_point(skin_v, hand_verts)
    skin_v = output.Xt
    
    # transformation = output.RTs
    # transformed_vertices = transform_vertices(skin_v, transformation.R, transformation.T)
    dist = chamfer_distance(skin_v, hand_verts)[0]

    rendered_image, _ = render_mesh(skin_v, nlayer.skin_f, renderer, verts_uvs, faces_uvs, tex_img, bn=bn, obj_mesh=obj_mesh)
    photo_loss = calculate_photo_loss(segmented_img_tensor, rendered_image)
    
    loss = dist + photo_loss

    optimizer.zero_grad()

    # Compute the gradients of the chamfer distance with respect to the parameters
    loss.backward()
    # Update the parameters
    optimizer.step()
    
    if dist < min_dist: # or photo_loss < min_photo_loss:
        min_dist = min(dist, min_dist)
        print(f'min chamfer dist: {min_dist.item():.02f}, photo loss: {photo_loss.item():.02f}')
        min_photo_loss = min(photo_loss, min_photo_loss)
        skin_v2d = project_3D_points(cam_intr, skin_v)

        # Invert X and Y axis to match PyTorch3D's coordinate system.
        skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(bn, 1, 1))
        skin_p3dmesh = smooth_mesh(skin_p3dmesh)
        skin_v_smooth = skin_p3dmesh.verts_padded().detach().cpu().numpy()
        tex_img_numpy = tex_img.detach().cpu().numpy()

        visualize_image_with_mesh(img, rendered_image)
        # visualize_image_with_mesh(segmented_img, skin_v2d, nlayer.skin_f.detach().cpu().numpy(), rendered_image)

        save_textured_nimble("{:s}/rand_{}_{}.obj".format(output_folder, i, int(dist)), skin_v_smooth[0], tex_img_numpy[0])
