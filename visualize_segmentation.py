import torch
from utils import load_sample, mano_layer, device, data_path
from render_utils import create_renderer, render_mesh
import numpy as np
import matplotlib.pyplot as plt

# seq_name, frame_num = 'SS2', 885
seq_name, frame_num = 'ShSu13', 2
seq_name, frame_num = 'ABF10', 882

img, pose, shape, trans, hand_verts, hand_verts2d, obj_mesh = load_sample(data_path, 'train', seq_name, frame_num, device)

cam_intr_renderer = torch.tensor([[[614.627, 0., 320.262, 0.], [0., 614.101, 238.469, 0.], [0., 0., 0., 1.], [0., 0., 1., 0.]]], device=device)
renderer = create_renderer(cam_intr_renderer, device)

rendered_mano_obj, mano_obj_mask = render_mesh(hand_verts, mano_layer.th_faces, renderer, None, None, None, bn=1, obj_mesh=obj_mesh)
rendered_mano, mano_mask = render_mesh(hand_verts, mano_layer.th_faces, renderer, None, None, None, bn=1)
rendered_obj = renderer(obj_mesh)
obj_mask = rendered_obj[..., 3] > 0

# hand_mask = mano_obj_mask & mano_mask & ~obj_mask
# obj_mask = mano_obj_mask & obj_mask & ~mano_mask

seg_mask = torch.zeros_like(obj_mask, dtype=torch.long)
# seg_mask[mano_mask == 1] = 1
seg_mask[obj_mask == 1] = 1
# print(seg_mask.shape, seg_mask.unique())
seg_mask_np = seg_mask[0].cpu().detach().numpy()

ax = plt.subplot(2, 2, 1)
ax.imshow(img)
# ax.imshow(rendered_obj[0][..., :3].cpu().detach().numpy())

ax = plt.subplot(2, 2, 2)
ax.imshow(seg_mask_np)

hand_seg = np.copy(img)
hand_seg[seg_mask_np == 0] = 0
ax = plt.subplot(2, 2, 3)
ax.imshow(hand_seg)

# obj_seg = np.copy(img)
# obj_seg[seg_mask_np != 2] = 0
# ax = plt.subplot(2, 2, 4)
# ax.imshow(obj_seg)


plt.show()