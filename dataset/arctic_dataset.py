import os
import json
import numpy as np
import torch
import cv2
import zipfile

from torch.utils.data import Dataset
from manopth.manolayer import ManoLayer
from dataset.arctic_utils import transform_points_batch
# from arctic_utils import transform_points_batch

from utils import project_3D_points
from pytorch3d.io import load_obj
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_apply
from pytorch3d.renderer import Textures
from torch.utils.data import DataLoader
from models.mp_hand_segmenter import detect_hand
from torchvision import transforms

class ArcticDataset(Dataset):
    def __init__(self, root, objects_root, device, iterable=False):
        self.root = root
        mano_layer_right = ManoLayer(mano_root='mano_v1_2/models', ncomps=45, flat_hand_mean=False, use_pca=False).to(device)
        faces_right = mano_layer_right.th_faces.cpu().detach().numpy()

        mano_layer_left = ManoLayer(mano_root='mano_v1_2/models', ncomps=45, flat_hand_mean=False, use_pca=False, side='left').to(device)
        faces_left = mano_layer_left.th_faces.cpu().detach().numpy()
        self.mano_layers = {'right': mano_layer_right, 'left': mano_layer_left}
        self.hand_faces = {'right': faces_right, 'left': faces_left}

        self.annotations = zipfile.ZipFile(os.path.join(self.root, 'raw_seqs.zip'))
        self.meta_archive = zipfile.ZipFile(os.path.join(self.root, 'meta.zip'))
        self.meta = json.load(self.meta_archive.open('meta/misc.json'))
        self.objects_root = objects_root
        self.device = device
        self.objects = {name: {} for name in os.listdir(objects_root)}
        self.load_objects()
        self.object_names = sorted(list(self.objects.keys()))
        self.transform = transforms.Compose([transforms.ToTensor()])
        # self.iterable = iterable
        if not iterable: self.scan_dataset()

    def scan_dataset(self):
        self.dataset_keys = []
        subjects = sorted(os.listdir(os.path.join(self.root, 'images')))
        for subject in subjects:
            seqs = sorted(os.listdir(os.path.join(self.root, 'images', subject)))
            for seq in seqs:
                obj = seq.split('_')[0]
                if obj not in self.objects.keys(): self.objects[obj] = {}
                cameras = sorted(os.listdir(os.path.join(self.root, 'images', subject, seq)))
                for camera in cameras:
                    frames = sorted(os.listdir(os.path.join(self.root, 'images', subject, seq, camera)))
                    for frame in frames:
                        self.dataset_keys.append('/'.join([subject, seq, camera, frame]))

    def load_objects(self):
        for obj in self.objects.keys():
            for part in ['bottom', 'top']:
                verts, faces, aux = load_obj(os.path.join(self.objects_root, obj, f'{part}.obj'), device=self.device)
                tex = self.create_texture(faces, aux)
                self.objects[obj][part] = (verts, faces, tex)

    def __len__(self):
        return len(self.dataset_keys)

    def load_camera_matrix(self, subject, seq_name, camera, frame_num):

        ego_annotations_path = os.path.join('raw_seqs', subject, seq_name+f'.egocam.dist.npy')
        
        # ego_annotations_path = os.path.join(self.root, 'raw_seqs', subject, seq_name+f'.egocam.dist.npy')
        ego_annotations = np.load(self.annotations.open(ego_annotations_path), allow_pickle=True).item()

        if camera > 0:
            cam_ext = torch.tensor(self.meta[subject]['world2cam'][camera-1], device=self.device).unsqueeze(0)
            cam_int = torch.tensor(self.meta[subject]['intris_mat'][camera-1], device=self.device)
        else:
            num_frames = ego_annotations[f'R_k_cam_np'].shape[0]-1
            R = torch.tensor(ego_annotations[f'R_k_cam_np'][min(frame_num, num_frames)], device=self.device)
            T = torch.tensor(ego_annotations[f'T_k_cam_np'][min(frame_num, num_frames)], device=self.device) 
            cam_ext = torch.cat((torch.cat((R, T), dim=1), torch.tensor([[0, 0, 0, 1]], device=self.device)), dim=0).unsqueeze(0)
            cam_int = torch.tensor(ego_annotations['intrinsics'], device=self.device)

        return cam_ext, cam_int

    def decode_mano(self, pose, shape, trans, side, cam_ext):
        verts_world, kps_world = self.mano_layers[side](pose, shape, trans)
        verts_world /= 1000
        kps_world /= 1000
        verts_cam = transform_points_batch(cam_ext, verts_world)
        kps_cam = transform_points_batch(cam_ext, kps_world)
        return verts_cam, kps_cam
            
    def load_hand_annotations(self, subject, seq_name, frame_num, cam_ext, cam_int):

        # hand_annotations_path = os.path.join(self.root, 'raw_seqs', subject, seq_name+f'.mano.npy')
        hand_annotations_path = os.path.join('raw_seqs', subject, seq_name+f'.mano.npy')
        hand_annotations = np.load(self.annotations.open(hand_annotations_path), allow_pickle=True).item()
        hand_dict = {}
        for side in ['right', 'left']:

            anno = hand_annotations[side]
            hand = []
            for component in ['rot', 'pose', 'shape', 'trans']:
                if component != 'shape':
                    anno_comp = anno[component]
                    comp_tensor = torch.tensor(anno_comp[min(frame_num, anno_comp.shape[0]-1)], device=self.device)
                else:
                    comp_tensor = torch.tensor(anno[component], device=self.device)
                hand.append(comp_tensor.unsqueeze(0))
                hand_dict[f'{side}_{component}'] = comp_tensor

            verts, kps = self.decode_mano(torch.cat((hand[0], hand[1]), dim=1), hand[2], hand[3], side, cam_ext)
            hand_dict[f'{side}_verts'] = verts[0]

        return hand_dict

    def load_obj_annotations(self, subject, seq_name, frame_num, cam_ext, cam_int, obj):

        # obj_annotations_path = os.path.join(self.root, 'raw_seqs', subject, seq_name+f'.object.npy')
        obj_annotations_path = os.path.join('raw_seqs', subject, seq_name+f'.object.npy')
        obj_annotations = np.load(self.annotations.open(obj_annotations_path), allow_pickle=True)
        
        obj_dict = {}
        num_frames = obj_annotations.shape[0] - 1
        articulation = torch.tensor(obj_annotations[min(frame_num, num_frames)][0], device=self.device)
        rot = torch.tensor(obj_annotations[min(frame_num, num_frames)][1:4], device=self.device)
        trans = torch.tensor(obj_annotations[min(frame_num, num_frames)][4:7], device=self.device)

        obj_dict['articulation'] = articulation.unsqueeze(0)
        obj_dict['rot'] = rot
        obj_dict['trans'] = trans / 1000
        obj_dict['object_name'] = obj
        obj_dict['label'] = torch.tensor(self.object_names.index(obj), dtype=torch.long, device=self.device)

        return obj_dict

    def transform_obj(self, obj, articulation, rot, trans, cam_ext):
        obj_verts = {}
        z_axis = torch.FloatTensor(np.array([0, 0, -1])).view(1, 3).to(self.device)
        quat_arti = axis_angle_to_quaternion(z_axis * articulation).unsqueeze(1)
        quat_global = axis_angle_to_quaternion(rot).unsqueeze(1)

        bs = rot.shape[0]
        for part in ['top', 'bottom']:
            verts = self.objects[obj][part][0].unsqueeze(0).repeat(bs, 1, 1)#.unsqueeze(0)
            num_verts = verts.shape[1]
            quat_arti_mesh, quat_global_mesh = quat_arti.repeat(1, num_verts, 1), quat_global.repeat(1, num_verts, 1)
            verts = quaternion_apply(quat_arti_mesh, verts) if part == 'top' else verts
            verts = quaternion_apply(quat_global_mesh, verts) 
            trans_mesh = trans.unsqueeze(1).repeat(1, num_verts, 1) * 1000
            verts += trans_mesh 
            verts /= 1000

            verts_cam = transform_points_batch(cam_ext, verts)
            obj_verts[part] = verts_cam

        return obj_verts

    def create_texture(self, faces, aux):
        verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
        faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
        tex_maps = aux.texture_images
        texture_image = list(tex_maps.values())[0]
        texture_image = texture_image[None, ...].to(self.device)  # (1, H, W, 3)

        tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)
        return tex
    
    def get_anno(self, img, key):
        subject, seq_name, camera, frame = key.split('/')

        camera_num = int(camera)
        frame_num = int(frame.split('.')[0]) - 1
        obj = seq_name.split('_')[0]
        
        _, pose2d, _, _, _ = detect_hand(img, segment=False)

        img = self.transform(img).to(self.device)

        # hands_pose2d_placeholder = torch.zeros((2, 21, 2))
        # hands_pose2d = torch.stack((torch.tensor(x).view(-1, 21), torch.tensor(y).view(-1, 21)), dim=2)
        # hands_pose2d_placeholder[:hands_pose2d.shape[0]] = hands_pose2d

        cam_ext, cam_int = self.load_camera_matrix(subject, seq_name, camera_num, frame_num)
        hand_dict = self.load_hand_annotations(subject, seq_name, frame_num, cam_ext, cam_int)
        obj_dict = self.load_obj_annotations(subject, seq_name, frame_num, cam_ext, cam_int, obj)

        hand_dict.update(obj_dict)
        data_dict = hand_dict
        data_dict['img'] = img
        data_dict['key'] = key
        data_dict['hands_pose2d'] = pose2d.to(self.device)
        data_dict['cam_ext'] = cam_ext[0]
        data_dict['cam_int'] = cam_int

        return data_dict

    def __getitem__(self, idx):

        key = self.dataset_keys[idx]
        subject, seq_name, camera, frame = key.split('/')
        img_path = os.path.join(self.root, 'images', subject, seq_name, camera, frame)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        data_dict = self.get_anno(img, key)
        
        return data_dict


if __name__ == '__main__':
    root = '../arctic/data/arctic_data/data'
    objects_root = '../arctic_objects/'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = ArcticDataset(root, objects_root, device=device)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # print(len(dataset))
    anno_dict = next(iter(loader))
    for k, v in anno_dict.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)
