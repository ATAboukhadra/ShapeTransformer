import os
import json
import numpy as np
import torch
import cv2
import zipfile
import time

from torch.utils.data import Dataset
from manopth.manolayer import ManoLayer
from dataset.arctic_utils import transform_points_batch
# from arctic_utils import transform_points_batch

from utils import project_3D_points
from pytorch3d.io import load_obj
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_apply
from pytorch3d.renderer import Textures
from torch.utils.data import DataLoader
# from models.mp_hand_segmenter import detect_hand, init_hand_kp_model
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

        # self.hand_detector = init_hand_kp_model()
        self.annotations = zipfile.ZipFile(os.path.join(self.root, 'raw_seqs.zip'))
        self.prefetch_annotations()
        self.meta_archive = zipfile.ZipFile(os.path.join(self.root, 'meta.zip'))
        self.meta = json.load(self.meta_archive.open('meta/misc.json'))
        self.objects_root = objects_root
        self.device = device
        self.objects = {name: {} for name in os.listdir(objects_root)}
        self.object_keypoints = {name: {} for name in os.listdir(objects_root)}
        self.num_kps_obj = 30
        self.resize_factor = 4
        self.load_objects()
        self.object_names = sorted(list(self.objects.keys()))

        self.total, self.bad = 0, 0
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
        sampled_indices = np.arange(0, 300, 300 // self.num_kps_obj)
        for obj in self.objects.keys():
            for part in ['bottom', 'top']:
                verts, faces, aux = load_obj(os.path.join(self.objects_root, obj, f'{part}.obj'), device=self.device)
                tex = self.create_texture(faces, aux)
                self.objects[obj][part] = (verts, faces, tex)
                keypoints = json.load(open(os.path.join(self.objects_root, obj, f'{part}_keypoints_300.json')))['keypoints']
                keypoints = torch.tensor(keypoints)#[:self.num_kps_obj]
                self.object_keypoints[obj][part] = keypoints


    def prefetch_annotations(self):
        print('Prefetching Annotations ..', flush=True)

        self.ego_annotations_dict = {}
        self.hand_annotations_dict = {}
        self.obj_annotations_dict = {}

        for file in self.annotations.filelist:
            if 'smpl' in file.filename: continue
            key = file.filename
            with self.annotations.open(file, 'r') as f:
                if 'mano' in key:
                    self.hand_annotations_dict[key] = np.load(f, allow_pickle=True).item()
                elif 'object' in key:
                    self.obj_annotations_dict[key] = np.load(f, allow_pickle=True)
                elif 'dist' in key:
                    self.ego_annotations_dict[key] = np.load(f, allow_pickle=True).item()


    def __len__(self):
        return len(self.dataset_keys)

    def downscale_cam_int(self, cam_int):
        fx, fy = cam_int[0, 0], cam_int[1, 1]
        cx, cy = cam_int[0, 2], cam_int[1, 2]
        fx, fy = fx / self.resize_factor, fy / self.resize_factor
        cx, cy = cx / self.resize_factor, cy / self.resize_factor
        cam_int[0, 0], cam_int[1, 1] = fx, fy
        cam_int[0, 2], cam_int[1, 2] = cx, cy
        return cam_int

    def load_camera_matrix(self, subject, seq_name, camera, frame_num):
        ego_annotations_path = os.path.join('raw_seqs', subject, seq_name+f'.egocam.dist.npy')
        
        self.total += 1
        valid = True

        if ego_annotations_path not in self.ego_annotations_dict.keys():
            ego_annotations = {}
            ego_annotations['R_k_cam_np'] = np.zeros((1000, 3, 3))
            ego_annotations['T_k_cam_np'] = np.zeros((1000, 3, 1))
            ego_annotations['intrinsics'] = np.zeros((3, 3))
            self.bad += 1
            valid = False
        else:
            ego_annotations = self.ego_annotations_dict[ego_annotations_path]

        if camera > 0:
            cam_ext = torch.tensor(self.meta[subject]['world2cam'][camera-1], device=self.device).unsqueeze(0)
            cam_int = torch.tensor(self.meta[subject]['intris_mat'][camera-1], device=self.device)
        else:
            num_frames = ego_annotations[f'R_k_cam_np'].shape[0]-1
            R = torch.tensor(ego_annotations[f'R_k_cam_np'][min(frame_num, num_frames)], device=self.device, dtype=torch.float32)
            T = torch.tensor(ego_annotations[f'T_k_cam_np'][min(frame_num, num_frames)], device=self.device, dtype=torch.float32) 
            cam_ext = torch.cat((torch.cat((R, T), dim=1), torch.tensor([[0, 0, 0, 1]], device=self.device)), dim=0).unsqueeze(0)
            cam_int = torch.tensor(ego_annotations['intrinsics'], device=self.device, dtype=torch.float32)

        cam_int = self.downscale_cam_int(cam_int)

        return cam_ext, cam_int, valid

    def decode_mano(self, pose, shape, trans, side, cam_ext):
        verts_world, kps_world = self.mano_layers[side](pose, shape, trans)
        verts_world /= 1000
        kps_world /= 1000
        verts_cam = transform_points_batch(cam_ext, verts_world)
        kps_cam = transform_points_batch(cam_ext, kps_world)
        return verts_cam, kps_cam
            
    def load_hand_annotations(self, subject, seq_name, frame_num, cam_ext, cam_int, valid):
        # t1 = time.time()
        # hand_annotations_path = os.path.join(self.root, 'raw_seqs', subject, seq_name+f'.mano.npy')
        hand_annotations_path = os.path.join('raw_seqs', subject, seq_name+f'.mano.npy')
        self.total += 1
        hand_dict = {}
        if hand_annotations_path not in self.hand_annotations_dict.keys():
            hand_annotations = {
                'left': {'rot': np.zeros((1000, 3)), 'pose': np.zeros((1000, 45)), 'shape': np.zeros((10)), 'trans': np.zeros((1000, 3))},
                'right': {'rot': np.zeros((1000, 3)), 'pose': np.zeros((1000, 45)), 'shape': np.zeros((10)), 'trans': np.zeros((1000, 3))}
            }
            self.bad += 1
            valid = False
        else:
            hand_annotations = self.hand_annotations_dict[hand_annotations_path]
            

        pose2d = torch.zeros((2, 21, 2), device=self.device, dtype=torch.float32)
        for i, side in enumerate(['left', 'right']):

            anno = hand_annotations[side]
            hand = []
            for component in ['rot', 'pose', 'shape', 'trans']:
                if component != 'shape':
                    anno_comp = anno[component]
                    comp_tensor = torch.tensor(anno_comp[min(frame_num, anno_comp.shape[0]-1)], dtype=torch.float32, device=self.device)
                else:
                    comp_tensor = torch.tensor(anno[component], dtype=torch.float32, device=self.device)
                hand.append(comp_tensor)#.unsqueeze(0))

            hand_dict[f'{side}_pose'] = torch.cat((hand[0], hand[1]), dim=0)
            hand_dict[f'{side}_shape'] = hand[2]
            hand_dict[f'{side}_trans'] = hand[3]

            # print(hand_dict[f'{side}_mano'].shape)
            verts, kps = self.decode_mano(hand_dict[f'{side}_pose'].unsqueeze(0), hand[2].unsqueeze(0), hand[3].unsqueeze(0), side, cam_ext)
            kps_2d = project_3D_points(cam_int, kps)
            if kps_2d.isnan().any():
                # print(subject, seq_name, frame_num, kps.max(), cam_int, kps_2d.max(), valid)
                valid = False
            pose2d[i] = kps_2d[0]
            hand_dict[f'{side}_pose3d'] = kps[0]

        hand_dict['hands_pose2d'] = pose2d
        # print(f'load_hand_annotations: {time.time()-t1}')

        return hand_dict, valid

    def load_obj_annotations(self, subject, seq_name, frame_num, cam_ext, cam_int, obj, valid):
        # obj_annotations_path = os.path.join(self.root, 'raw_seqs', subject, seq_name+f'.object.npy')
        obj_annotations_path = os.path.join('raw_seqs', subject, seq_name+f'.object.npy')
        self.total += 1
        obj_dict = {}

        if obj_annotations_path not in self.obj_annotations_dict.keys():
            obj_annotations = np.zeros((1000, 7), dtype=np.float32)
            self.bad += 1
            valid = False
        else:
            obj_annotations = self.obj_annotations_dict[obj_annotations_path]

        num_frames = obj_annotations.shape[0] - 1
        obj_pose = torch.tensor(obj_annotations[min(frame_num, num_frames)], device=self.device)
        obj_dict['obj_pose'] = obj_pose
        obj_dict['object_name'] = obj
        obj_dict['label'] = torch.tensor(self.object_names.index(obj), dtype=torch.long, device=self.device)
        _, obj_kps = self.transform_obj(obj, obj_pose[0].unsqueeze(0), obj_pose[1:4].unsqueeze(0), obj_pose[4:].unsqueeze(0) / 1000, cam_ext)
        obj_kps2d = project_3D_points(cam_int, obj_kps.view(2, -1, 3))
        top_bb = self.calculate_bounding_box(obj_kps2d[0])
        bottom_bb = self.calculate_bounding_box(obj_kps2d[1])
        bbs = torch.stack((top_bb, bottom_bb), dim=0)
        visibility = torch.ones((2, self.num_kps_obj, 1), dtype=torch.float32)
        # obj_dict['keypoints'] = 
        obj_dict['keypoints'] = torch.cat((obj_kps2d[:, :self.num_kps_obj], visibility), dim=2)
        obj_dict['boxes'] = bbs
        obj_dict['labels'] = self.create_obj_labels(obj)
        # print(obj_kps.shape)
        return obj_dict, valid

    def create_obj_labels(self, obj_name):
        obj_label_top = self.object_names.index(obj_name) * 2 
        obj_label_bottom = obj_label_top + 1
        obj_labels = torch.tensor([obj_label_top, obj_label_bottom], dtype=torch.long)
        return obj_labels

    def transform_points(self, points, cam_ext, part, quat_arti, quat_global, trans):
        num_verts = points.shape[1]
        quat_arti_mesh, quat_global_mesh = quat_arti.repeat(1, num_verts, 1), quat_global.repeat(1, num_verts, 1)
        points = quaternion_apply(quat_arti_mesh, points) if part == 'top' else points
        points = quaternion_apply(quat_global_mesh, points) 
        trans_mesh = trans.unsqueeze(1).repeat(1, num_verts, 1) * 1000
        points += trans_mesh 
        points /= 1000

        points = transform_points_batch(cam_ext, points)
        return points

    def transform_obj(self, obj, articulation, rot, trans, cam_ext):
        obj_verts = {}

        z_axis = torch.FloatTensor(np.array([0, 0, -1])).view(1, 3).to(articulation.device)
        quat_arti = axis_angle_to_quaternion(z_axis * articulation).unsqueeze(1)
        quat_global = axis_angle_to_quaternion(rot).unsqueeze(1)

        bs = rot.shape[0]
        obj_kps = torch.zeros((2, bs, 300, 3), device=self.device, dtype=torch.float32)
        
        for i, part in enumerate(['top', 'bottom']):
            verts = self.objects[obj][part][0].unsqueeze(0).repeat(bs, 1, 1).to(articulation.device)#.unsqueeze(0)
            kps = self.object_keypoints[obj][part].unsqueeze(0).repeat(bs, 1, 1).to(articulation.device)#.unsqueeze(0)
            
            verts_cam = self.transform_points(verts, cam_ext, part, quat_arti, quat_global, trans)
            kps_cam = self.transform_points(kps, cam_ext, part, quat_arti, quat_global, trans)

            obj_verts[part] = verts_cam
            obj_kps[i] = kps_cam

        return obj_verts, obj_kps

    def create_texture(self, faces, aux):
        verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
        faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
        tex_maps = aux.texture_images
        texture_image = list(tex_maps.values())[0]
        texture_image = texture_image[None, ...].to(self.device)  # (1, H, W, 3)

        tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)
        return tex
    
    def calculate_bounding_box(self, point2d):
        
        x_min, y_min = int(min(point2d[:,0])), int(min(point2d[:,1]))
        x_max, y_max = int(max(point2d[:,0])), int(max(point2d[:,1]))
        
        return torch.tensor([x_min, y_min, x_max, y_max])

    def get_anno(self, key):
        subject, seq_name, camera, frame = key.split('/')
        camera_num = int(camera)
        frame_num = int(frame.split('.')[0]) - 1
        obj = seq_name.split('_')[0]
        
        # _, pose2d, _, _, _ = detect_hand(img, detector=self.hand_detector)
        # pose2d = torch.zeros((2, 21, 2))
        cam_ext, cam_int, valid = self.load_camera_matrix(subject, seq_name, camera_num, frame_num)
        hand_dict, valid = self.load_hand_annotations(subject, seq_name, frame_num, cam_ext, cam_int, valid)
        obj_dict, valid = self.load_obj_annotations(subject, seq_name, frame_num, cam_ext, cam_int, obj, valid)

        hand_dict.update(obj_dict)
        data_dict = hand_dict
        data_dict['valid'] = valid
        data_dict['key'] = key
        # data_dict['hands_pose2d'] = pose2d.to(self.device)
        data_dict['cam_ext'] = cam_ext[0]
        data_dict['cam_int'] = cam_int
        return data_dict

    def __getitem__(self, idx):

        key = self.dataset_keys[idx]
        subject, seq_name, camera, frame = key.split('/')
        img_path = os.path.join(self.root, 'images', subject, seq_name, camera, frame)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        data_dict = self.get_anno(img, key)
        data_dict['rgb'] = img

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
