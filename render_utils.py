from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    Textures
)
from pytorch3d.structures.meshes import Meshes
from pytorch3d.structures import join_meshes_as_scene
import torch
import matplotlib.colors as mcolors

# def create_mesh(verts, faces, verts_uvs=None, faces_uvs=None, texture_image=None, color=(1, 1, 1)):
#     if texture_image is not None:
#         texture_image_rgb = texture_image[..., :3]
#         texture_image_rgb = texture_image_rgb.flip(dims=(3,))  # BGR to RGB
#         tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image_rgb)
#     else:
#         dummy_verts_uvs = torch.zeros_like(verts[:, :, :2], device=verts.device) 
#         r, g, b = color
#         dummy_texture_image = torch.tensor([[[[r, g, b], [r, g, b]]]], device=verts.device)  # White dummy texture image
#         # print(dummy_texture_image.shape, dummy_texture_image)
#         tex = Textures(verts_uvs=dummy_verts_uvs, faces_uvs=faces, maps=dummy_texture_image)
#     # Create a meshes object with textures

#     mesh = Meshes(verts=verts, faces=faces, textures=tex)
#     return mesh

def create_mesh(verts, faces, tex=None, color=(1, 1, 1)):
    if tex is None:
        
        dummy_verts_uvs = torch.zeros_like(verts[:, :, :2], device=verts.device) 
        r, g, b = color
        dummy_texture_image = torch.tensor([[[[r, g, b], [r, g, b]]]], device=verts.device)  # White dummy texture image
        # print(dummy_texture_image.shape, dummy_texture_image)
        tex = Textures(verts_uvs=dummy_verts_uvs, faces_uvs=faces, maps=dummy_texture_image)
    # Create a meshes object with textures

    mesh = Meshes(verts=verts, faces=faces, textures=tex)
    return mesh

def create_renderer(K, device, R=None, T=None, image_size=((480, 640),)):
    
    raster_settings = RasterizationSettings(
        image_size=image_size[0], 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        bin_size=0
    )

    # R = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], device=device) if R is None else R
    # T = torch.tensor([[0., 0., 0.]], device=device) if T is None else T

    cameras = PerspectiveCameras(K=K, device=device, in_ndc=False, image_size=image_size)
    # Create a MeshRenderer object with a SoftPhongShader to render the mesh
    lights = PointLights(device=device, location=[[0, 0, 0]])

    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights))
    # Render the mesh and return the output image
    return renderer

def render_nimble_mesh(verts, faces, renderer, verts_uvs=None, faces_uvs=None, tex_img=None, bn=1, obj_mesh=None):
    
    verts_renderer = verts.clone()
    verts_renderer[:, :, 0] *= -1
    verts_renderer[:, :, 2] *= -1
    verts_renderer /= 1000
    
    texture_image_rgb = tex_img[..., :3]
    texture_image_rgb = texture_image_rgb.flip(dims=(3,))  # BGR to RGB
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image_rgb)

    mesh = create_mesh(verts_renderer, faces.repeat(bn, 1, 1), tex)
    
    mesh = mesh if obj_mesh is None else join_meshes_as_scene([mesh, obj_mesh])

    rendered_image = renderer(mesh)
    mask = rendered_image[..., 3] > 0

    return rendered_image, mask

def render_arctic_mesh(verts, faces, textures, renderer, bn=1):
    meshes = []
    colors = [mcolors.to_rgb('#849DAB'), mcolors.to_rgb('#24788F'), mcolors.to_rgb('#F18A85'), mcolors.to_rgb('#282130')]
    
    for i in range(len(verts)):
        verts_renderer = verts[i].clone()
        verts_renderer[:, :, 0] *= -1
        verts_renderer[:, :, 1] *= -1
        if textures[i] is None:
            meshes.append(create_mesh(verts_renderer, faces[i].repeat(bn, 1, 1), color=colors[i]))
        else:
            meshes.append(create_mesh(verts_renderer, faces[i].repeat(bn, 1, 1), tex=textures[i]))
            
    meshes = join_meshes_as_scene(meshes)

    rendered_image = renderer(meshes)
    mask = rendered_image[..., 3] > 0

    return rendered_image, mask
