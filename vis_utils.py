from __future__ import print_function, unicode_literals
import numpy as np
import json
import os
import time
import skimage.io as io
import pickle
import math
import sys
import matplotlib.pyplot as plt
import cv2
import pymeshlab
from manopth.manolayer import ManoLayer


""" General util functions. """
def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def showHandJoints(imgInOrg, gtIn, filename=None, dataset_name='ho', mode='pred'):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param filename: dump image name
    :return:
    '''
    import cv2

    imgIn = np.copy(imgInOrg)
    # Set color for each finger

    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    if mode == 'gt':
        joint_color_code = [[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]

    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    cf = 35 

    gtIn = np.round(gtIn).astype(np.int)

    if gtIn.shape[0]==1:
        imgIn = cv2.circle(imgIn, center=(gtIn[0][0], gtIn[0][1]), radius=3, color=joint_color_code[0],
                             thickness=-1)
    else:
        max_length=500
        for joint_num in range(gtIn.shape[0]):

            color_code_num = (joint_num // 4)
            joint_color = list(map(lambda x: x + cf * (joint_num % 4), joint_color_code[color_code_num]))[::-1]    
            
            cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=1, color=joint_color, thickness=-1)
        
        for limb_num in range(len(limbs)):
            x1 = gtIn[limbs[limb_num][0], 1]
            y1 = gtIn[limbs[limb_num][0], 0]
            x2 = gtIn[limbs[limb_num][1], 1]
            y2 = gtIn[limbs[limb_num][1], 0]
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                        (int(length / 2), 1),
                                        int(deg),
                                        0, 360, 1)
            color_code_num = limb_num // 4

            limb_color = list(map(lambda x: x  + cf * (limb_num % 4), joint_color_code[color_code_num]))[::-1]


            cv2.fillConvexPoly(imgIn, polygon, color=limb_color)

    if filename is not None:
        cv2.imwrite(filename, cv2.cvtColor(imgIn, cv2.COLOR_RGB2BGR))

    return imgIn

def showObjJoints(imgIn, gtIn, estIn=None, filename=None, upscale=1, lineThickness=2):
    '''
    Utility function for displaying object annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param estIn: estimated keypoints
    :param filename: dump image name
    :param upscale: scale factor
    :param lineThickness:
    :return:
    '''
    import cv2
    jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3,7]]
    jointColsGt = (255,255,0)
    jointColsEst  = (0, 0, 0)

    # draws lines connected using jointConns
    img = np.zeros((imgIn.shape[0], imgIn.shape[1], imgIn.shape[2]), dtype=np.uint8)
    img[:, :, :] = (imgIn).astype(np.uint8)

    img = cv2.resize(img, (upscale * imgIn.shape[1], upscale * imgIn.shape[0]), interpolation=cv2.INTER_CUBIC)
    if gtIn is not None:
        gt = gtIn.copy() * upscale
    if estIn is not None:
        est = estIn.copy() * upscale

    for i in range(len(jointConns)):
        for j in range(len(jointConns[i]) - 1):
            jntC = jointConns[i][j]
            jntN = jointConns[i][j+1]
            if gtIn is not None:
                cv2.line(img, (int(gt[jntC,0]), int(gt[jntC,1])), (int(gt[jntN,0]), int(gt[jntN,1])), jointColsGt, lineThickness)
            if estIn is not None:
                cv2.line(img, (int(est[jntC,0]), int(est[jntC,1])), (int(est[jntN,0]), int(est[jntN,1])), jointColsEst, lineThickness)

    if filename is not None:
        cv2.imwrite(filename, img)

    return img

def draw_bb(img, bb, color):
    """ Show bounding box on the image"""
    bb_img = np.copy(img)

    # print(bb, bb_img.shape, bb_img.dtype)
    bb = bb.astype(int)
    cv2.rectangle(bb_img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
    return bb_img

def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

def plot3dVisualize(ax, m, faces, flip_x=False, c="b", alpha=0.1, camPose=np.eye(4, dtype=np.float32), isOpenGLCoords=False):
    '''
    Create 3D visualization
    :param ax: matplotlib axis
    :param m: mesh
    :param flip_x: flix x axis?
    :param c: mesh color
    :param alpha: transperency
    :param camPose: camera pose
    :param isOpenGLCoords: is mesh in openGL coordinate system?
    :return:
    '''
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if hasattr(m, 'r'):
        verts = np.copy(m.r) * 1000
    elif hasattr(m, 'v'):
        verts = np.copy(m.v) * 1000
    elif isinstance(m, np.ndarray): # In case of an output of a Mano layer (no need to scale)
        verts = np.copy(m)
    else:
        raise Exception('Unknown Mesh format')
    vertsHomo = np.concatenate([verts, np.ones((verts.shape[0],1), dtype=np.float32)], axis=1)
    verts = vertsHomo.dot(camPose.T)[:,:3]

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        verts = verts.dot(coordChangeMat.T)

    # ax.view_init(elev=-90, azim=-180)
    ax.view_init(elev=-90, azim=-90)
    # ax.view_init(elev=90, azim=-90)
    # ax.view_init(elev=120, azim=-90)


    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    if c == "b":
        face_color = (141 / 255, 184 / 255, 226 / 255)
        face_color = np.tile(np.array([[0., 0., 1., 1.]]), [verts.shape[0], 1])
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "r":
        face_color = (226 / 255, 141 / 255, 141 / 255)
        face_color = np.tile(np.array([[1., 0., 0., 1.]]), [verts.shape[0], 1])
        edge_color = (112 / 255, 0 / 255, 0 / 255)
    elif c == "viridis":
        face_color = plt.cm.viridis(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "plasma":
        face_color = plt.cm.plasma(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        # edge_color = (0 / 255, 0 / 255, 112 / 255)
    else:
        face_color = c
        edge_color = c

    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    cam_equal_aspect_3d(ax, verts, flip_x=flip_x)
    # plt.tight_layout()

def show3DHandJoints(ax, verts, mode='pred', isOpenGLCoords=False):
    '''
    Utility function for displaying hand 3D annotations
    :param ax: matplotlib axis
    :param verts: ground truth annotation
    '''
    # ax.axis('off')
    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        verts = verts.dot(coordChangeMat.T)

    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]


    joint_color_code = ['b', 'g', 'r', 'c', 'm']

    if mode == 'gt':
        joint_color_code = ['k'] * 5

    ax.view_init(elev=120, azim=-90)
    for limb_num in range(len(limbs)):
        x1 = verts[limbs[limb_num][0], 0]
        y1 = verts[limbs[limb_num][0], 1]
        z1 = verts[limbs[limb_num][0], 2]
        x2 = verts[limbs[limb_num][1], 0]
        y2 = verts[limbs[limb_num][1], 1]
        z2 = verts[limbs[limb_num][1], 2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color=joint_color_code[limb_num//4])

    x = verts[:,0]
    y = verts[:,1]
    z = verts[:,2]
    ax.scatter(x, y, z)

def show3DObjCorners(ax, verts, mode='pred', isOpenGLCoords=False):
    '''
    Utility function for displaying Object 3D annotations
    :param ax: matplotlib axis
    :param verts: ground truth annotation
    '''

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        verts = verts.dot(coordChangeMat.T)

    # jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3,7]]
    
    # for i in range(len(jointConns)):
    #     for j in range(len(jointConns[i]) - 1):
    #         jntC = jointConns[i][j]
    #         jntN = jointConns[i][j+1]
    #         if mode == 'gt':
    #             ax.plot([verts[jntC][0], verts[jntN][0]], [verts[jntC][1], verts[jntN][1]], [verts[jntC][2], verts[jntN][2]], color='k')
    #         else:    
    #             ax.plot([verts[jntC][0], verts[jntN][0]], [verts[jntC][1], verts[jntN][1]], [verts[jntC][2], verts[jntN][2]], color='y')

    x = verts[:,0]
    y = verts[:,1]
    z = verts[:,2]

    ax.scatter(x, y, z)

def show2DMesh(fig, ax, img, mesh2DPoints, gt=False, filename=None):
    ax.imshow(img)
    if gt:
        ax.scatter(mesh2DPoints[:, 0], mesh2DPoints[:, 1], alpha=0.3, s=20, color="black", marker='.')
    else:
        ax.scatter(mesh2DPoints[:778, 0], mesh2DPoints[:778, 1], alpha=0.3, s=20, marker='.')
        if mesh2DPoints.shape[0] > 778:
            ax.scatter(mesh2DPoints[778:, 0], mesh2DPoints[778:, 1], alpha=0.3, s=20, color="red", marker='.')
    
    # Save just the portion _inside_ the second axis's boundaries
    if filename is not None:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{filename}', bbox_inches=extent)

def draw_confidence(image, keypoints, scores):
    keypoints = np.round(keypoints).astype(np.int)

    high_confidence = np.where(scores >= 2)[0]
    low_confidence = np.where(scores < 2)[0]
    
    for idx in high_confidence:
        cv2.circle(image, center=(keypoints[idx][0], keypoints[idx][1]), radius=3, color=[43, 140, 237], thickness=-1)
    for idx in low_confidence:
        cv2.circle(image, center=(keypoints[idx][0], keypoints[idx][1]), radius=3, color=[0, 0, 0], thickness=-1)
    
    return image

def plot_bb_ax(img, outputs, fig_config, subplot_id, plot_txt):
    fig, H, W = fig_config
    bb_image = np.copy(img)
    ax = fig.add_subplot(H, W, subplot_id)
    
    labels = list(outputs['labels'])
    
    if max(labels) > 1:
        required_bbs = [
            outputs['boxes'][labels.index(1)],
            outputs['boxes'][labels.index(2)],
            outputs['boxes'][labels.index(3)]
        ]
    else:
        required_bbs = outputs['boxes']
    
    for bb in required_bbs:
        bb_image = draw_bb(bb_image, bb, [229, 255, 204])    
    
    ax.title.set_text(plot_txt)
    ax.imshow(bb_image)

def plot_pose2d(img, keypoints3d, cam_mat, fig_config, subplot_id, plot_txt):

    keypoints = project_3D_points(cam_mat, keypoints3d, is_OpenGL_coords=False)
    # print(keypoints)
    fig, H, W = fig_config
    # plt_image = np.copy(img)
    
    ax = fig.add_subplot(H, W, subplot_id)

    # if not np.isnan(keypoints[0][0]):
    plt_image = showHandJoints(img, keypoints[:21])
    plt_image = showHandJoints(plt_image, keypoints[21:42])
    # ax.scatter(obj_kps2d[i, 2, :, 0], obj_kps2d[i, 2, :, 1], c='lightgreen', s=1)
    # ax.scatter(obj_kps2d[i, 3, :, 0], obj_kps2d[i, 3, :, 1], c='gold', s=1)

    ax.scatter(keypoints[42:63, 0], keypoints[42:63, 1], c='lightblue', s=1)
    ax.scatter(keypoints[63:, 0], keypoints[63:, 1], c='peachpuff', s=1)
    
    # If pose is only 1 hand and object (HO3D)
    # if not np.isnan(keypoints[-1][0]):
    # plt_image = showObjJoints(plt_image, keypoints[-8:])
 
    ax.title.set_text(plot_txt)
    ax.imshow(plt_image)
    return plt_image
    

# def plot_pose3d(labels, fig_config, subplot_id, plot_txt, mode='pred', center=None, idx=0):
    
#     fig, H, W = fig_config
#     keypoints3d = labels['keypoints3d'][idx]
#     if center is not None:
#         keypoints3d += center

#     ax = fig.add_subplot(H, W, subplot_id, projection="3d")
    
#     # Hide grid lines
#     ax.grid(False)

#     # Hide axes ticks
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])

#     show3DHandJoints(ax, keypoints3d[:21], mode=mode, isOpenGLCoords=True)
    
#     # If pose is only 1 hand and object (HO3D)
#     if keypoints3d.shape[0] == 29:
#         show3DObjCorners(ax, keypoints3d[21:], mode=mode, isOpenGLCoords=True)
    
#     # If pose is only 2 hands and object (H2O)
#     if keypoints3d.shape[0] == 50:
#         show3DHandJoints(ax, keypoints3d[21:42], mode=mode, isOpenGLCoords=True)
#         show3DObjCorners(ax, keypoints3d[42:], mode=mode, isOpenGLCoords=True)

#     ax.title.set_text(plot_txt)

def plot_pose3d(fig_config, plot_id, pose3d, text, mode='pred'):
    
    fig, H, W = fig_config

    ax = fig.add_subplot(H, W, plot_id, projection="3d")
    
    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    show3DHandJoints(ax, pose3d[:21], mode=mode, isOpenGLCoords=True)
    show3DHandJoints(ax, pose3d[21:42], mode=mode, isOpenGLCoords=True)
    show3DObjCorners(ax, pose3d[42:63], mode='gt', isOpenGLCoords=True)
    show3DObjCorners(ax, pose3d[63:], mode=mode, isOpenGLCoords=True)

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    cam_equal_aspect_3d(ax, pose3d.dot(coordChangeMat.T), flip_x=False)

    ax.view_init(elev=-90, azim=-90)
    ax.title.set_text(text)

def plot_mesh3d(mesh3d, faces, fig_config, subplot_id, plot_txt):
    
    fig, H, W = fig_config
    ax = fig.add_subplot(H, W, subplot_id, projection="3d")
    
    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plot3dVisualize(ax, mesh3d[:778], faces[:1538], flip_x=False, isOpenGLCoords=True, c="r")
    plot3dVisualize(ax, mesh3d[778:778*2], faces[1538:1538*2] - 778, flip_x=False, isOpenGLCoords=True, c="g")
    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    cam_equal_aspect_3d(ax, mesh3d.dot(coordChangeMat.T), flip_x=False)
    
    ax.view_init(elev=-90, azim=-90)
    ax.title.set_text(plot_txt)

def plot_pose_heatmap(img, predictions, idx, center, fig_config, plot_id):

    # porject to 2D 
    cam_mat = np.array(
        [[617.343,0,      312.42],
        [0,       617.343,241.42],
        [0,       0,       1]
    ])
    
    keypoints3d = predictions['keypoints3d'][idx]
    keypoints = project_3D_points(cam_mat, keypoints3d + center, is_OpenGL_coords=False)
    keypoints = np.round(keypoints).astype(np.int)
    
    fig, H, W = fig_config
    ax = fig.add_subplot(H, W, plot_id)

    heatmap = create_heatmap(img, keypoints)

    ax.imshow(heatmap, cmap='viridis')

def create_heatmap(img, keypoints):

    heatmap = np.zeros_like(img[:, :, 0]) 
    r = 6
    translations = []
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            distance = i**2 + j**2
            if (distance <= r*r):
                translations.append((i, j, 1 - math.sqrt(distance)/r))
    for t in translations:
        if np.all(keypoints[:, 1] + t[0] < heatmap.shape[0]) and np.all(keypoints[:, 0] + t[1] < heatmap.shape[1]):
            heatmap[keypoints[:, 1] + t[0], keypoints[:, 0] + t[1]] = t[2] * 10
    
    return heatmap


def plot_temporal_sample(batch):
    # print(batch['boxes'].shape)
    fig = plt.figure(figsize=(15, 15))
    for i in range(batch['images'][0].shape[0]):
        im = batch['images'][0][i].cpu().detach().numpy()
        # img = img.transpose(1, 2, 0) * 255
        # img = np.ascontiguousarray(img, np.uint8) 
        for bb in batch['boxes'][0][i]:
            bb = bb.detach().numpy()
            pt1 = (bb[0], bb[1])
            pt2 = (bb[2], bb[3])
            # print(pt1, pt2)
            cv2.rectangle(im, pt1, pt2, color=(255, 0, 0))
        ax = fig.add_subplot(3, 3, i+1)
        ax.imshow(im)

class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs
        
def read_obj(filename):
    """ Reads the Obj file. Function reused from Matthew Loper's OpenDR package"""

    lines = open(filename).read().split('\n')

    d = {'v': [], 'f': []}

    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0])-1 for l in spl[:3]], dtype=np.uint32)])

    for k, v in d.items():
        if k in ['v','f']:
            if v:
                d[k] = np.vstack(v)
            else:
                print(k)
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result

def load_pickle_data(f_name):
    """ Loads the pickle data """
    if not os.path.exists(f_name):
        raise Exception('Unable to find annotations picle file at %s. Aborting.'%(f_name))
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data

def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if is_OpenGL_coords:
        pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2]],axis=1)

    assert len(proj_pts.shape) == 2

    return proj_pts

def save_mesh(mesh3d, faces, key, err=0):
    seq_name = '_'.join(key.split('/')[:-1])
    if not os.path.exists(f'output/meshes/{seq_name}'):
        os.makedirs(f'output/meshes/{seq_name}')
    
    pred=False
    frame_name = key.split('/')[-1].split('.')[0]
    file_name = f'output/meshes/{seq_name}/{frame_name}'
    if err > 0: 
        file_name += f'_{int(err)}'
        pred = True
    
    # faces = np.concatenate((faces['left'], faces['right'] + 778), axis = 0)

    write_obj(mesh3d, faces, file_name, pred)


def write_obj(verts, faces, filename, pred=True):
    """Saves and obj file using vertices and faces"""
    # if 'gt' in filename:
    texture = np.zeros_like(verts)

    if pred:
        texture[:778, 0] = 0
        texture[:778, 1] = 0.5
        texture[:778, 2] = 1

        texture[778:778*2, 0] = 0
        texture[778:778*2, 1] = 1
        texture[778:778*2, 2] = 0.5

        texture[778*2:, 0] = 1
        texture[778*2:, 1] = 0.5
        texture[778*2:, 2] = 0
    else:
        texture[:, 0] = 0.7
        texture[:, 1] = 0.7
        texture[:, 2] = 0.7

    if texture is not None:
        alpha = np.ones((verts.shape[0], 1))
        v_color_matrix = np.append(texture, alpha, axis=1)
        m = pymeshlab.Mesh(verts, faces, v_color_matrix=v_color_matrix)
    else:
        m = pymeshlab.Mesh(verts, faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, f'{filename}')
    ms.save_current_mesh(f'{filename}.obj', save_vertex_normal=True, save_vertex_color=True, save_polygonal=True)

