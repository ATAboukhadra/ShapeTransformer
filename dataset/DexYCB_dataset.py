import os
import yaml
from manopth.manolayer import ManoLayer
import torch
import numpy as np
# from dataset.data_util import *
# from voxelization_torch import VoxelizationTorch
# from mesh_torch import MeshTorch

class DexYCBDataset():
    "Adapted from https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/dex_ycb_toolkit/dex_ycb.py"
    def __init__(self, data_dir, patch_size=16, num_kps=21):
        self._SERIALS = [
            '836212060125',
            '839512060362',
            '840412060917',
            '841412060263',
            '932122060857',
            '932122060861',
            '932122061900',
            '932122062010',
        ]

        self._SUBJECTS = [
            '20200709-subject-01',
            '20200813-subject-02',
            '20200820-subject-03',
            '20200903-subject-04',
            '20200908-subject-05',
            '20200918-subject-06',
            '20200928-subject-07',
            '20201002-subject-08',
            '20201015-subject-09',
            '20201022-subject-10',
        ]
        
        self.YCB_CLASSES = {
            1: '002_master_chef_can',
            2: '003_cracker_box',
            3: '004_sugar_box',
            4: '005_tomato_soup_can',
            5: '006_mustard_bottle',
            6: '007_tuna_fish_can',
            7: '008_pudding_box',
            8: '009_gelatin_box',
            9: '010_potted_meat_can',
            10: '011_banana',
            11: '019_pitcher_base',
            12: '021_bleach_cleanser',
            13: '024_bowl',
            14: '025_mug',
            15: '035_power_drill',
            16: '036_wood_block',
            17: '037_scissors',
            18: '040_large_marker',
            19: '051_large_clamp',
            20: '052_extra_large_clamp',
            21: '061_foam_brick',
        }

        self.jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]

        self.obj_model_path = 'dataset/object_models_HO3D/displacement_model/'
        # self.obj_resolution = obj_resolution
        self.data_dir = data_dir
        # self.augmentation = augmentation
        self.color_intrinsics = {}
        self.depth_intrinsics = {}
        
        self._meta_dict = {}
        self.mano_calibs = {}
        self._calib_dir = os.path.join(data_dir, "calibration") 
        self._h = 480
        self._w = 640
        self.device = 'cpu'
        self.dtype = torch.float
        self.patch_size = patch_size
        self.num_kps = num_kps
        # self.tsdf = tsdf
             
        for s in self._SERIALS:
            intr_file = os.path.join(self._calib_dir, "intrinsics",
                                "{}_{}x{}.yml".format(s, self._w, self._h))
            with open(intr_file, 'r') as f:
                intr = yaml.load(f, Loader=yaml.FullLoader)
            self.color_intrinsics[s] = intr['color']
            self.depth_intrinsics[s] = intr['depth']
        
        self._init_meta_dict()
        self.rh_model = ManoLayer(mano_root='mano_v1_2/models', ncomps=45, flat_hand_mean=False, use_pca=True)
        self.lh_model = ManoLayer(mano_root='mano_v1_2/models', ncomps=45, side='left', flat_hand_mean=False, use_pca=True)
        
        # TODO: Change this to handle left hand mesh as well
        # self.hand_mesh = MeshTorch(self.device, faces=self.rh_model.th_faces.cpu().detach().numpy())
        # vertices, faces = MeshTorch.read_obj(self.obj_model_path + obj_resolution + "/" + "init_" + obj_resolution + ".obj")
        # self.object_mesh = MeshTorch(self.device, vertices = vertices, faces = faces)

        # self.HAND_JOINTS, self.HAND_VERTS, self.OBJ_CORNERS, self.OBJ_VERTS = 21, 778, 8, self.object_mesh.vertices.shape[0]
        # if mode == 'hand':
        #     self.OBJ_CORNERS=0
        #     self.OBJ_VERTS=0
        # elif mode == 'object':
        #     self.HAND_JOINTS=0
        #     self.HAND_VERTS=0
        
        # self.voxelization = VoxelizationTorch('cpu', cubic_size=cubic_size, n_voxels=n_voxels)
        # self.voxelization_half = VoxelizationTorch('cpu', cubic_size=cubic_size, n_voxels=int(n_voxels/2))

    def _init_meta_dict(self):
        for subject_id in self._SUBJECTS:
            sequences = sorted(os.listdir(os.path.join(self.data_dir, subject_id)))
            sequences = [os.path.join(subject_id, s) for s in sequences]
            for seq in sequences:
                meta_file = os.path.join(self.data_dir, seq, "meta.yml")
                with open(meta_file, 'r') as f:
                    self._meta_dict[seq] = yaml.load(f, Loader=yaml.FullLoader)
    
    # def project_points(self, points, camMat, ux, uy, fx, fy, map_hand=False):

    #     if map_hand:
    #         points = points[self.jointsMapManoToSimple]

    #     # print(points.dtype)
    #     uvd = torch.from_numpy(project_3D_points(camMat, points, is_OpenGL_coords=False)).to(self.device).to(self.dtype)
    #     world = Main_pixelToworld(uvd.clone(), ux, uy, fx, fy)

    #     return uvd, world
    
    def mano_calibration(self, meta):
    # Load mano calibration if they are not loaded (only loads 10 files once)
        mano_calib_name = meta['mano_calib'][0]
        if mano_calib_name not in self.mano_calibs.keys():
            mano_calib_file = os.path.join(self.data_dir, "calibration",
                                        "mano_{}".format(mano_calib_name),
                                        "mano.yml")
            with open(mano_calib_file, 'r') as f:
                mano_calib = yaml.load(f, Loader=yaml.FullLoader)
            self.mano_calibs[mano_calib_name] = mano_calib
        else:
            mano_calib = self.mano_calibs[mano_calib_name]

        return mano_calib

    def decode_mano(self, mano_params, mano_betas, mano_side):

        mano_betas = torch.tensor(mano_betas, dtype=torch.float32).unsqueeze(0)
        # print(mano_betas.shape)
        # mano_betas = torch.zeros((1, 10))
        mano_pose = torch.tensor(mano_params)
        if mano_side == 'right':
            verts, pose = self.rh_model(mano_pose[:, 0:48], mano_betas, mano_pose[:, 48:51])
        else:
            verts, pose = self.lh_model(mano_pose[:, 0:48], mano_betas, mano_pose[:, 48:51])
        
        return verts[0].detach().numpy() / 1000, pose[0].detach().numpy() / 1000

    def calculate_bounding_box(self, point2d):
        point2d = point2d[:self.num_kps]
        boxes = []
        labels = []
        size = self.patch_size
        for i in range(point2d.shape[0]):
            x, y = point2d[i]
            boxes.append((x - size, y - size, x + size, y + size))
            labels.append(i+1)

        boxes = np.array(boxes).reshape(-1, 4).astype(int)

        return boxes, labels 


    # def voxelize_depth(self, depth, refpoint, cam_mat, aug_translate, aug_angle, aug_scale):
    #     fx, fy, ux, uy = cam_mat[0, 0], cam_mat[1, 1], cam_mat[0, 2], cam_mat[1, 2]
    #     threshold = self.voxelization.voxel_size
    #     clamp_to_one=True
    #     if self.tsdf:
    #         clamp_to_one = False
    #         N = 7
    #         threshold = self.voxelization.voxel_size * N
        
    #     cubic_depth, _ = self.voxelization.depth(depth, refpoint, ux, uy, fx, fy, 
    #                                             threshold=threshold, clamp_to_one=clamp_to_one, 
    #                                             translate=aug_translate, angle=aug_angle, scale=aug_scale)
    #     cubic_depth_half, _ = self.voxelization_half.depth(depth, refpoint, ux, uy, fx, fy, 
    #                                                         threshold = threshold // 2, clamp_to_one=clamp_to_one, 
    #                                                         translate=aug_translate, angle=aug_angle, scale=aug_scale)
    #     return cubic_depth.unsqueeze(0), cubic_depth_half.unsqueeze(0)

    # def get_object_mesh(self, object_name, cam_mat, rotation, translation):
    #     # def read_object_mesh_data(self, annot, camMat, ux, uy, fx, fy):
    #     fx, fy, ux, uy = cam_mat[0, 0], cam_mat[1, 1], cam_mat[0, 2], cam_mat[1, 2]

    #     objMesh_v = self.object_mesh.vertices.copy()
    #     objDF = np.load(os.path.join(self.obj_model_path, self.obj_resolution, object_name + '.npy'), allow_pickle=True).tolist()
    #     displ = objDF['displacements']
    #     scale = objDF['scale']
    #     transl = objDF['translation']
    #     objMesh_v = objMesh_v + displ
    #     objMesh_v = objMesh_v * scale + transl
    #     objMesh_v = np.matmul(objMesh_v, rotation.T) + translation
    #     objMesh_uvd = torch.from_numpy(project_3D_points(cam_mat, objMesh_v, is_OpenGL_coords=False)).to(self.device).to(self.dtype)
    #     objMesh_world = Main_pixelToworld(objMesh_uvd.clone(), ux, uy, fx, fy)

    #     return objMesh_uvd, objMesh_world
    
    # def get_pose_channels(self):
    #     input_channels=1
    #     output_channels=self.HAND_JOINTS+self.OBJ_CORNERS

    #     return input_channels, output_channels

    # def get_voxel_channels(self):
    #     input_channels=1+self.HAND_JOINTS+self.OBJ_CORNERS
    #     output_channels=1

    #     return input_channels, output_channels

    # def get_shape_channels(self):
    #     input_channels=1+self.HAND_JOINTS+self.OBJ_CORNERS
    #     output_channels=self.HAND_VERTS+self.OBJ_VERTS

    #     return input_channels, output_channels