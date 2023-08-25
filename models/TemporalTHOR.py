import torch
import torch.nn as nn
from models.thor import THOR
from models.graformer import GraFormer
from manopth.manolayer import ManoLayer
from dataset.arctic_utils import transform_points_batch

class TemporalTHOR(nn.Module):

    def __init__(self, device, num_kps=84, input_dim=26, num_frames=9, temporal_dim=128, thor_path='') -> None:
        super().__init__()

        mano_layer_right = ManoLayer(mano_root='mano_v1_2/models', ncomps=45, flat_hand_mean=False, use_pca=False).to(device)
        mano_layer_left = ManoLayer(mano_root='mano_v1_2/models', ncomps=45, flat_hand_mean=False, use_pca=False, side='left').to(device)
        self.mano_layers = {'right': mano_layer_right, 'left': mano_layer_left}

        self.thor = THOR(device, input_dim=input_dim, num_frames=num_frames, num_kps=num_kps)
        self.backbone_path = thor_path
        if thor_path != '': self.thor.load_state_dict(torch.load(thor_path))
        self.thor.eval()

        self.input_dim = input_dim
        pose_dim = 3 * num_kps
        # self.temporal_encoder = Transformer(num_frames, num_kps=num_kps, input_dim=num_features, hid_dim=hid_dim, num_layers=4, normalize_before=True)
        self.temporal_encoder = GraFormer(hid_dim=temporal_dim, coords_dim=(pose_dim, pose_dim), num_pts=num_frames, temporal=True)

    def reload_backbone(self):
        if self.backbone_path != '': self.thor.load_state_dict(torch.load(self.backbone_path))

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

        bs, t = pose3d.shape[:2]
        temporal_out = self.temporal_encoder(pose3d.view(bs, t, -1)).view(bs, t, -1, 3)
        pose3d = temporal_out

        return pose3d, graph, labels
        