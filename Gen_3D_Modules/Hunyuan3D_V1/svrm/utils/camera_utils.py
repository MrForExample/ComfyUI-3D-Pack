import math
import numpy as np

def compute_extrinsic_matrix(elevation, azimuth, camera_distance):
    # 将角度转换为弧度
    elevation_rad = np.radians(elevation)
    azimuth_rad = np.radians(azimuth)

    R = np.array([
        [np.cos(azimuth_rad), 0, -np.sin(azimuth_rad)],
        [0, 1, 0],
        [np.sin(azimuth_rad), 0, np.cos(azimuth_rad)],
    ], dtype=np.float32)

    R = R @ np.array([
        [1, 0, 0],
        [0, np.cos(elevation_rad), -np.sin(elevation_rad)],
        [0, np.sin(elevation_rad), np.cos(elevation_rad)]
    ], dtype=np.float32)

    # 构建平移矩阵 T (3x1)
    T = np.array([[camera_distance], [0], [0]], dtype=np.float32)
    T = R @ T

    # 组合成 4x4 的变换矩阵
    extrinsic_matrix = np.vstack((np.hstack((R, T)), np.array([[0, 0, 0, 1]], dtype=np.float32)))

    return extrinsic_matrix


def transform_camera_pose(im_pose, ori_pose, new_pose):
    T = new_pose @ ori_pose.T
    transformed_poses = []

    for pose in im_pose:
        transformed_pose = T @ pose
        transformed_poses.append(transformed_pose)

    return transformed_poses

def compute_fov(intrinsic_matrix):
    # 获取内参矩阵中的焦距值
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    h, w = intrinsic_matrix[0,2]*2, intrinsic_matrix[1,2]*2

    # 计算水平和垂直方向的FOV值
    fov_x = 2 * math.atan(w / (2 * fx)) * 180 / math.pi
    fov_y = 2 * math.atan(h / (2 * fy)) * 180 / math.pi

    return fov_x, fov_y



def rotation_matrix_to_quaternion(rotation_matrix):
    rot = Rotation.from_matrix(rotation_matrix)
    quaternion = rot.as_quat()
    return quaternion

def quaternion_to_rotation_matrix(quaternion):
    rot = Rotation.from_quat(quaternion)
    rotation_matrix = rot.as_matrix()
    return rotation_matrix

def remap_points(img_size, match, size=512):
    H, W, _ = img_size

    S = max(W, H)
    new_W = int(round(W * size / S))
    new_H = int(round(H * size / S))
    cx, cy = new_W // 2, new_H // 2

    # 计算变换后的图像中心点坐标
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8

    dw, dh = cx - halfw, cy - halfh

    # 初始化一个新的数组来存储映射回原始图像的点坐标
    new_match = np.zeros_like(match)

    # 将变换后的点坐标映射回原始图像
    new_match[:, 0] = (match[:, 0] + dw) / new_W * W
    new_match[:, 1] = (match[:, 1] + dh) / new_H * H

    #print(dw,new_W,W,dh,new_H,H)

    return new_match

    