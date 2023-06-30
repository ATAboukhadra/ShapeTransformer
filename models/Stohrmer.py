import torch.nn as nn
import torch
from models.poseformer import PoseGraFormer
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from models.graformer import GraFormer

class Stohrmer(nn.Module):
    def __init__(self, device, num_kps=21, num_frames=9, spatial_dim=32, temporal_dim=128, extra_features=32):
        super().__init__()
        self.device = device

        # self.graformer = PoseGraFormer(num_kps=num_kps, num_frames=num_frames, d_model=32)
        self.spatial_encoder = GraFormer(num_pts=num_kps, coords_dim=(2, spatial_dim))
        num_features = spatial_dim * num_kps + 512

        self.resize = transforms.Resize(500, antialias=True)
        weights = ResNet18_Weights.DEFAULT
        full_resnet18 = resnet18(weights=weights, progress=False)
        self.resnet18 = torch.nn.Sequential(*list(full_resnet18.children())[:-1])
        
        frame_output_dim = 3 * 2 + 45 * 2 + 3 * 2 + 7 + extra_features # 48 MANO Pose, 3 Translation, 7 Object Pose, additional per-frame features
        self.temporal_encoder = GraFormer(hid_dim=temporal_dim, coords_dim=(num_features, frame_output_dim), num_pts=num_frames, temporal=True)

        output_dim = 10 * 2 + 11 # 10 MANO Shape, 11 Object Classes
        self.output = nn.Linear(extra_features * num_frames, output_dim)
    
    def forward(self, batch_dict):
        # 2D Pose Spatial Features
        pose2d = batch_dict['hands_pose2d'].to(self.device)
        bs, t, n, k, d = pose2d.shape
        pose2d = pose2d.view(bs * t, n * k, d)
        spatial_pose_features = self.spatial_encoder(pose2d).view(bs, t, -1)

        # Image Features
        img_list = batch_dict['img']
        features_list = []
        for img_batch in img_list:
            img_batch = self.resize(img_batch.to(self.device))
            features = self.resnet18(img_batch).squeeze(-1).squeeze(-1)
            features_list.append(features)

        features_tensor = torch.stack(features_list, dim=0)

        # Temporal Features
        features_tensor = torch.cat([features_tensor, spatial_pose_features], dim=-1)
        frame_outputs = self.temporal_encoder(features_tensor)

        left_mano_rot, left_mano_pose, left_mano_trans = frame_outputs[:, :, :3], frame_outputs[:, :, 3:48], frame_outputs[:, :, 48:51]
        right_mano_rot, right_mano_pose, right_mano_trans = frame_outputs[:, :, 51:54], frame_outputs[:, :, 54:99], frame_outputs[:, :, 99:102]        
        obj_pose = frame_outputs[:, :, 102:109]

        # Output
        final_features = frame_outputs[:, :, 109:].reshape(bs, -1)
        outputs = self.output(final_features)
        left_mano_shape, right_mano_shape, obj_class = outputs[:, :10], outputs[:, 10:20], outputs[:, 20:]

        outputs_dict = {
            'left_rot': left_mano_rot,
            'left_pose': left_mano_pose,
            'left_shape': left_mano_shape,
            'left_trans': left_mano_trans,
            'right_rot': right_mano_rot,
            'right_pose': right_mano_pose,
            'right_shape': right_mano_shape,
            'right_trans': right_mano_trans,
            'obj_pose': obj_pose,
            'obj_class': obj_class

        }

        return outputs_dict

