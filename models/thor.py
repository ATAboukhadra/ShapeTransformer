import torch.nn as nn
import torch
from models.poseformer import PoseGraFormer, Transformer
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from models.graformer import GraFormer
from models.poseformer import Transformer
import torchvision
from utils import get_keypoints

class THOR(nn.Module):
    def __init__(self, device, num_kps=84, input_dim=2, num_frames=9, spatial_dim=32, temporal_dim=128, extra_features=32, rcnn_path='', target_idx=0):
        super().__init__()
        self.device = device

        # self.graformer = PoseGraFormer(num_kps=num_kps, num_frames=num_frames, d_model=32)
        
        self.obj_rcnn = torchvision.models.detection.keypointrcnn_resnet50_fpn(num_keypoints=21, num_classes=24)
        if rcnn_path != '':
            self.obj_rcnn.load_state_dict(torch.load(rcnn_path))
        self.obj_rcnn.eval()

        self.spatial_output = 3 if num_frames == 1 else spatial_dim
        self.input_dim = input_dim
        self.spatial_encoder = GraFormer(num_pts=num_kps, coords_dim=(self.input_dim, self.spatial_output))


        num_features = spatial_dim * num_kps
        if num_frames > 1:
            hid_dim = 128
            # self.temporal_encoder = Transformer(num_frames, num_kps=num_kps, input_dim=num_features, hid_dim=hid_dim, num_layers=4, normalize_before=True)
            self.temporal_encoder = GraFormer(hid_dim=temporal_dim, coords_dim=(num_features, 3 * num_kps), num_pts=num_frames, temporal=True)

        if self.input_dim > 2 + 24:
            weights = ResNet18_Weights.DEFAULT
            full_resnet18 = resnet18(weights=weights, progress=False)
            self.resnet18 = torch.nn.Sequential(*list(full_resnet18.children())[:-1])
 
        # # 48 MANO Pose, 3 Translation, 7 Object Pose, additional per-frame features, 21 3D KPs
        # frame_output_dim = 3 * 2 + 45 * 2 + 3 * 2 + 7 + extra_features #+ 2 * 21 * 3
        
        # output_dim = 10 * 2 + 11 # 10 MANO Shape, 11 Object Classes
        # self.output = nn.Linear(extra_features * num_frames, output_dim)

    def one_hot(self, label):
        one_hot = torch.zeros(24)#.to(self.device)
        one_hot[label] = 1
        one_hot = one_hot.unsqueeze(0).repeat(21, 1).to(self.device)
        return one_hot

    def patch_features(self, image, box):
        # Crop image around box
        cropped_image = image[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])].unsqueeze(0)
        features = self.resnet18(cropped_image).squeeze(-1).squeeze(-1).squeeze(0)
        return features


    def forward(self, batch_dict):
        bs, t = len(batch_dict['rgb']), batch_dict['rgb'][0].shape[0]

        # 2D Keypoints
        images = batch_dict['rgb']
        # Expand list of temporal batches to list of list of frames
        images = [img for temporal_batch in images for img in temporal_batch]
        with torch.no_grad():
            rcnn_outputs = self.obj_rcnn(images)

        graph = torch.zeros(bs * t, 4, 21, self.input_dim).to(self.device)
        obj_class = torch.zeros((bs, 11)).to(self.device)

        for i in range(bs * t):
            kps, labels, boxes = get_keypoints(rcnn_outputs, i)
            for o in range(4):
                if kps[o] is not None:
                    graph[i, o, :, :2] = kps[o][:, :2]
                    if i % t == 0 and o == 2: obj_class[i // t, labels[o] // 2] = 1 # One hot
                    if self.input_dim > 2: graph[i, o, :, 2:26] = self.one_hot(labels[o])
                    if self.input_dim > 26: graph[i, o, :, 26:] = self.patch_features(images[i], boxes[o])               

        graph = graph.view(bs * t, 4 * 21, self.input_dim)
        spatial_out = self.spatial_encoder(graph).view(bs, t, -1, self.spatial_output)

        if t > 1:
            temporal_out = self.temporal_encoder(spatial_out.view(bs, t, -1)).view(bs, t, -1, 3)
            pose3d = temporal_out
        else:
            pose3d = spatial_out
        
        return pose3d, graph, obj_class