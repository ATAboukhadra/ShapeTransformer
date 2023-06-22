# import os
# import torch
# import matplotlib.pyplot as plt

# # Util function for loading meshes
# from pytorch3d.io import load_objs_as_meshes, load_obj

# # Data structures and functions for rendering
# from pytorch3d.structures import Meshes
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
# from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
# from pytorch3d.renderer import (
#     PerspectiveCameras,
#     PointLights, 
#     RasterizationSettings, 
#     MeshRenderer, 
#     MeshRasterizer,  
#     SoftPhongShader,
#     Textures
# )
# import numpy as np
# import torch
# from pytorch3d.renderer import MeshRenderer, MeshRasterizer, SoftPhongShader
# from nimble.utils import load_texture
# # add path for demo utils functions 
# import os

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # Load obj file
# # mesh = load_objs_as_meshes(["output/rand_5_89_skin.obj"], device=device)

# tex_maps = aux.texture_images
# print(tex_maps.keys(), tex_maps['material_0'].shape)
# # tex_maps is a dictionary of {material name: texture image}.
# # Take the first image:
# texture_image = list(tex_maps.values())[0]
# texture_image = texture_image[None, ...].to(device)  # (1, H, W, 3)

# # Compute the view transform matrix using the camera position, target, and up direction.
# # view_transform_matrix = look_at_view_transform(position=position, at=target, up=up)

# # Create a texture object from the texture map.
# # texture = TexturesUV(maps=tex_img[None, :], faces_uvs=faces[None, :])

# # Create a Meshes object from the vertices, faces, and texture.
# # mesh = Meshes(verts=[skin_v], faces=[faces], textures=texture)


# # plt.figure(figsize=(7,7))
# # texture_image=mesh.textures.maps_padded()
# # plt.imshow(texture_image.squeeze().cpu().numpy())
# # plt.axis("off")

# # plt.figure(figsize=(7,7))
# # texturesuv_image_matplotlib(mesh.textures, subsample=None)
# # plt.axis("off")
# fig = plot_scene({
#     "subplot1": {
#         "hand": mesh
#     }
# })
# fig.show()

# images = render_mesh(mesh, cam_intr, device, ((480, 640),))
# plt.figure(figsize=(10, 10))
# plt.imshow(images[0, ..., :3].cpu().numpy())
# # plt.axis("off")
# plt.show()