from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

import torch

from kiui.cam import orbit_camera

#{Key: [elevation, azimuth], ...}
ORBITPOSE_PRESET_DICT = OrderedDict([
    ("Custom",           [[0.0, 90.0, 0.0, 0.0, -90.0, 0.0], [-90.0, 0.0, 180.0, 90.0, 0.0, 0.0]]),
    ("CRM(6)",           [[0.0, 90.0, 0.0, 0.0, -90.0, 0.0], [-90.0, 0.0, 180.0, 90.0, 0.0, 0.0]]),
    ("Wonder3D(6)",      [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 45.0, 90.0, 180.0, -90.0, -45.0]]),
    ("Zero123Plus(6)",   [[-20.0, 10.0, -20.0, 10.0, -20.0, 10.0], [30.0, 90.0, 150.0, -150.0, -90.0, -30.0]]),
    ("Era3D(6)",         [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 45.0, 90.0, 180.0, -90.0, -45.0]]),
    ("MVDream(4)",       [[0.0, 0.0, 0.0, 0.0], [0.0, 90.0, 180.0, -90.0]]),
    ("Unique3D(4)",      [[0.0, 0.0, 0.0, 0.0], [0.0, 90.0, 180.0, -90.0]]),
    ("CharacterGen(4)",  [[0.0, 0.0, 0.0, 0.0], [-90.0, 180.0, 90.0, 0.0]]),
])
ELEVATION_MIN = -89.999
ELEVATION_MAX = 89.999
AZIMUTH_MIN = -180.0
AZIMUTH_MAX = 180.0

def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R

def get_look_at_camera_pose(target, target_to_cam_offset, look_distance=0.1, opengl=True):
    """
    Calculate the pose (cam2world) matrix from target position the camera suppose to look at and offset vector from target to camera
    
    Args:
        target (NDArray[float32], shape: 3): the target position the camera suppose to look at
        target_to_cam_dir (NDArray[float32], shape: 3): offset direction from target to camera
        look_distance (float, optional): length of offset vector from target to camera.

    Returns:
        NDArray[float32]: shape: (4, 4), pose (cam2world) matrix
    """
    
    norm=np.linalg.norm(target_to_cam_offset)
    if norm==0:
        norm=np.finfo(np.float32).eps
    target_to_cam_offset = look_distance * target_to_cam_offset / norm
    
    campos = target_to_cam_offset + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!

    @property
    def fovx(self):
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])
        
def calculate_fovX(H, W, fovy):
    return 2 * np.arctan(np.tan(fovy / 2) * W / H)
        
def get_projection_matrix(znear, zfar, fovX, fovY, z_sign=1.0):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, projection_matrix=None):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            get_projection_matrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        ) if projection_matrix is None else projection_matrix
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

class BaseCameraController(ABC):
    def __init__(self, renderer, cam_size_W, cam_size_H, reference_orbit_camera_fovy, invert_bg_prob=1.0, static_bg=None, device='cuda'):
        self.device = torch.device(device)
        
        self.renderer = renderer
        self.cam = OrbitCamera(cam_size_W, cam_size_H, fovy=reference_orbit_camera_fovy)
        
        self.invert_bg_prob = invert_bg_prob
        self.black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        self.white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
        self.static_bg = None if static_bg is None else torch.tensor(static_bg, dtype=torch.float32, device=self.device)

        self.post_init()

        super().__init__()
    
    def post_init(self):
        # Calls after default initialize at the end of __init__()
        pass

    @abstractmethod
    def get_render_result(self, render_pose, bg_color, **kwargs):
        pass
        
    def render_at_pose(self, cam_pose, **kwargs):
        radius, elevation, azimuth, center_X, center_Y, center_Z = cam_pose
        
        orbit_target = np.array([center_X, center_Y, center_Z], dtype=np.float32)
        render_pose = orbit_camera(elevation, azimuth, radius, target=orbit_target)
        
        if self.static_bg is None:
            bg_color = self.white_bg if np.random.rand() > self.invert_bg_prob else self.black_bg
        else:
            bg_color = self.static_bg
            
        return self.get_render_result(render_pose, bg_color, **kwargs)
    
    def render_all_pose(self, all_cam_poses, **kwargs):
        all_rendered_images, all_rendered_masks = [], []
        extra_outputs = {}
        for cam_pose in all_cam_poses:
            out = self.render_at_pose(cam_pose, **kwargs)
            
            image = out["image"] # [3, H, W] in [0, 1]
            mask = out["alpha"] # [1, H, W] in [0, 1]
            
            all_rendered_images.append(image)
            all_rendered_masks.append(mask)
            
            for k in out:
                if k not in extra_outputs:
                    extra_outputs[k] = []
                extra_outputs[k].append(out[k])
                
        for k in extra_outputs:
            extra_outputs[k] = torch.stack(extra_outputs[k], dim=0)
            
        # [Number of Poses, 3, H, W], [Number of Poses, 1, H, W] both in [0, 1]
        return torch.stack(all_rendered_images, dim=0), torch.stack(all_rendered_masks, dim=0), extra_outputs
    
def compose_orbit_camposes(orbit_radius, orbit_elevations, orbit_azimuths, orbit_center_x, orbit_center_y, orbit_center_z):
    orbit_camposes = []
    
    campose_num = len(orbit_radius)
    for i in range(campose_num):
        orbit_camposes.append([
            orbit_radius[i], 
            np.clip(orbit_elevations[i], ELEVATION_MIN, ELEVATION_MAX), 
            np.clip(orbit_azimuths[i], AZIMUTH_MIN, AZIMUTH_MAX),
            orbit_center_x[i], orbit_center_y[i], orbit_center_z[i]
        ])
        
    return orbit_camposes