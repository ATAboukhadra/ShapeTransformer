import torch
import torch.nn as nn
from models.thor import THOR
from models.graformer import GraFormer
from manopth.manolayer import ManoLayer
from dataset.arctic_utils import transform_points_batch

class ShapeTHOR(nn.Module):

    def __init__(self, device, num_kps=84, input_dim=26, num_frames=1, target_idx=0, thor_path='') -> None:
        super().__init__()

        mano_layer_right = ManoLayer(mano_root='mano_v1_2/models', ncomps=45, flat_hand_mean=False, use_pca=False).to(device)
        mano_layer_left = ManoLayer(mano_root='mano_v1_2/models', ncomps=45, flat_hand_mean=False, use_pca=False, side='left').to(device)
        self.mano_layers = {'right': mano_layer_right, 'left': mano_layer_left}

        self.backbone_path = thor_path
        self.thor = THOR(device, input_dim=input_dim, num_frames=num_frames, num_kps=num_kps)
        if thor_path != '': 
            self.thor.load_state_dict(torch.load(thor_path))
            self.thor.eval()

            if hasattr(self.thor, 'resnet18'):
                self.thor.resnet18.train()

        self.target_idx = target_idx
        self.input_dim = input_dim
        self.mano_params = 48 + 10 + 3
        self.obj_pose = 7
        self.shape_graformer = GraFormer(num_pts=num_kps, coords_dim=(self.input_dim + 1, self.mano_params + self.obj_pose), trainable_adj=True)

    def reload_backbone(self):
        if self.backbone_path != '': 
            self.thor.load_state_dict(torch.load(self.backbone_path))
            self.thor.eval()

            if hasattr(self.thor, 'resnet18'):
                self.thor.resnet18.train()
        
        
    def decode_mano(self, pose, shape, trans, side, cam_ext):
        verts_world, kps_world = self.mano_layers[side](pose, shape, trans)
        verts_world /= 1000
        kps_world /= 1000
        verts_cam = transform_points_batch(cam_ext, verts_world)
        kps_cam = transform_points_batch(cam_ext, kps_world)
        return verts_cam, kps_cam
    
    def forward(self, batch_dict):
        with torch.no_grad():
            pose3d, graph, labels = self.thor(batch_dict)

        bs, t = len(batch_dict['rgb']), batch_dict['rgb'][0].shape[0]
        graph = graph.view(bs, t, 4 * 21, self.input_dim)[:, self.target_idx, :, 2:]
        graph = torch.cat((pose3d[:, self.target_idx], graph), dim=-1).view(bs, 4 * 21, self.input_dim + 1)

        params = self.shape_graformer(graph)

        left_hand_mano = torch.mean(params[:, :21, :self.mano_params], dim=1).unsqueeze(1)
        right_hand_mano = torch.mean(params[:, 21:42, :self.mano_params], dim=1).unsqueeze(1)
        obj_pose = torch.mean(params[:, :, self.mano_params:], dim=1).unsqueeze(1)
        
        outputs_dict = {
            'left_pose': left_hand_mano[:, :, :48],
            'left_shape': left_hand_mano[:, :, 48:58],
            'left_trans': left_hand_mano[:, :, 58:61],
            'left_pose3d': pose3d[:, :, :21],

            'right_pose': right_hand_mano[:, :, :48],
            'right_shape': right_hand_mano[:, :, 48:58],
            'right_trans': right_hand_mano[:, :, 58:61],
            'right_pose3d': pose3d[:, :, 21:42],
            
            'top_kps3d': pose3d[:, :, 42:63],
            'bottom_kps3d': pose3d[:, :, 63:84],
            'obj_pose': obj_pose,
            'obj_class': labels
        }
        # check for nan in left pose
        if torch.isnan(outputs_dict['left_pose']).any():
            print('left pose is nan')
            # check if the inputs in batch_dict are tensors and if the contain nan
            for k in batch_dict.keys():
                if isinstance(batch_dict[k], torch.Tensor) and torch.isnan(batch_dict[k]).any():
                    print(k, 'is nan')
        
        return outputs_dict
        