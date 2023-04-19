import torch
import os
import numpy as np
from utils import init_nimble
from nimble.utils import smooth_mesh, save_textured_nimble, write_obj
from pytorch3d.structures.meshes import Meshes
from pytorch3d.loss import chamfer_distance
from manopth.manolayer import ManoLayer
from pytorch3d.ops import iterative_closest_point

nlayer = init_nimble()
mano_layer = ManoLayer(mano_root='mano_v1_2/models', ncomps=45, flat_hand_mean=False, use_pca=True)
handFaces = mano_layer.th_faces.cpu().detach().numpy()

min_dist = 2000
output_folder = 'output'
for i in range(1000):
    # NIMBLE
    bn = 1
    pose_param = torch.rand(bn, 30) * 2 - 1 
    shape_param = torch.rand(bn, 20) * 2 - 1 
    tex_param = torch.rand(bn, 10) - 0.5

    skin_v, muscle_v, bone_v, bone_joints, tex_img = nlayer.forward(pose_param, shape_param, tex_param, handle_collision=True)
    # skin_v[:, :, 2] *= -1
    
    # MANO
    random_shape = torch.rand(bn, 10)
    random_pose = torch.rand(bn, 48)

    # Forward pass through MANO layer
    hand_verts, hand_joints = mano_layer(random_pose, random_shape)

    output = iterative_closest_point(hand_verts, skin_v)
    hand_verts = output.Xt
    # hand_verts[:, :, 2] *= -1
    # demo.display_hand({'verts': hand_verts, 'joints': hand_joints}, mano_faces=mano_layer.th_faces)
    dist = chamfer_distance(skin_v, hand_verts)[0]
    if dist < min_dist:
        min_dist = dist
        print(dist)
        i=0
        skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(bn, 1, 1))
        skin_p3dmesh = smooth_mesh(skin_p3dmesh)
        skin_v_smooth = skin_p3dmesh.verts_padded().detach().cpu().numpy()
        save_textured_nimble("{:s}/rand_{:d}.obj".format(output_folder, i), skin_v_smooth[i], tex_img[i])
        write_obj(hand_verts[0], handFaces, 'output/mano')
        write_obj(skin_v_smooth[0], nlayer.skin_f.detach().cpu().numpy(), 'output/nimble')
        


# tex_img = tex_img.detach().cpu().numpy()

# output_folder = "output"
# os.makedirs(output_folder, exist_ok=True)
# for i in range(bn):