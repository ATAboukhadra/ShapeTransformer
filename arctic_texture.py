import os
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from tqdm import tqdm

root = '../arctic/data/arctic_data/data'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
obj = 'box'

verts, faces, aux = load_obj(os.path.join(root, 'meta/object_vtemplates', obj, f'top.obj'), device=device)
verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
tex_maps = aux.texture_images
# tex_maps is a dictionary of {material name: texture image}.
# Take the first image:
texture_image = list(tex_maps.values())[0]
texture_image = texture_image[None, ...].to(device)  # (1, H, W, 3)

# Create a textures object
tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

# Initialise the mesh with textures
meshes = Meshes(verts=[verts], faces=[faces.verts_idx], textures=tex)

# top_verts, _, _ = load_obj(os.path.join(root, 'meta/object_vtemplates', obj, f'top.obj'), device=device)
# # map_top_to_main = {i: j}

# # print(tex_maps)

# bottom_verts, _, _ = load_obj(os.path.join(root, 'meta/object_vtemplates', obj, f'bottom.obj'), device=device)

# # dist = torch.cdist(bottom_verts, verts)

# print(verts_uvs.shape)
# # bottom_verts_uvs 
# bottom_verts_uvs = torch.zeros((1, bottom_verts.shape[0], 2), device=device)
# map_main_top_bottom = {}
# for i, v in tqdm(enumerate(bottom_verts)):
#     dist = torch.norm(v - verts, dim=1)
#     min_idx = torch.argmin(dist)
#     bottom_verts_uvs[0, i] = verts_uvs[0, min_idx]
#     map_main_top_bottom[min_idx] = i

# bottom_faces_uvs = faces_uvs.clone()

