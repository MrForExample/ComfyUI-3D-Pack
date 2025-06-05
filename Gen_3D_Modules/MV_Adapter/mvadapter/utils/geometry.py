from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F


def get_position_map_from_depth(depth, mask, intrinsics, extrinsics, image_wh=None):
    """Compute the position map from the depth map and the camera parameters for a batch of views.

    Args:
        depth (torch.Tensor): The depth maps with the shape (B, H, W, 1).
        mask (torch.Tensor): The masks with the shape (B, H, W, 1).
        intrinsics (torch.Tensor): The camera intrinsics matrices with the shape (B, 3, 3).
        extrinsics (torch.Tensor): The camera extrinsics matrices with the shape (B, 4, 4).
        image_wh (Tuple[int, int]): The image width and height.

    Returns:
        torch.Tensor: The position maps with the shape (B, H, W, 3).
    """
    if image_wh is None:
        image_wh = depth.shape[2], depth.shape[1]

    B, H, W, _ = depth.shape
    depth = depth.squeeze(-1)

    u_coord, v_coord = torch.meshgrid(
        torch.arange(image_wh[0]), torch.arange(image_wh[1]), indexing="xy"
    )
    u_coord = u_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)
    v_coord = v_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)

    # Compute the position map by back-projecting depth pixels to 3D space
    x = (
        (u_coord - intrinsics[:, 0, 2].unsqueeze(-1).unsqueeze(-1))
        * depth
        / intrinsics[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    )
    y = (
        (v_coord - intrinsics[:, 1, 2].unsqueeze(-1).unsqueeze(-1))
        * depth
        / intrinsics[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    )
    z = depth

    # Concatenate to form the 3D coordinates in the camera frame
    camera_coords = torch.stack([x, y, z], dim=-1)

    # Apply the extrinsic matrix to get coordinates in the world frame
    coords_homogeneous = torch.nn.functional.pad(
        camera_coords, (0, 1), "constant", 1.0
    )  # Add a homogeneous coordinate
    world_coords = torch.matmul(
        coords_homogeneous.view(B, -1, 4), extrinsics.transpose(1, 2)
    ).view(B, H, W, 4)

    # Apply the mask to the position map
    position_map = world_coords[..., :3] * mask

    return position_map


def get_position_map_from_depth_ortho(
    depth, mask, extrinsics, ortho_scale, image_wh=None
):
    """Compute the position map from the depth map and the camera parameters for a batch of views
    using orthographic projection with a given ortho_scale.

    Args:
        depth (torch.Tensor): The depth maps with the shape (B, H, W, 1).
        mask (torch.Tensor): The masks with the shape (B, H, W, 1).
        extrinsics (torch.Tensor): The camera extrinsics matrices with the shape (B, 4, 4).
        ortho_scale (torch.Tensor): The scaling factor for the orthographic projection with the shape (B, 1, 1, 1).
        image_wh (Tuple[int, int]): Optional. The image width and height.

    Returns:
        torch.Tensor: The position maps with the shape (B, H, W, 3).
    """
    if image_wh is None:
        image_wh = depth.shape[2], depth.shape[1]

    B, H, W, _ = depth.shape
    depth = depth.squeeze(-1)

    # Generating grid of coordinates in the image space
    u_coord, v_coord = torch.meshgrid(
        torch.arange(0, image_wh[0]), torch.arange(0, image_wh[1]), indexing="xy"
    )
    u_coord = u_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)
    v_coord = v_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)

    # Compute the position map using orthographic projection with ortho_scale
    x = (u_coord - image_wh[0] / 2) * ortho_scale / image_wh[0]
    y = (v_coord - image_wh[1] / 2) * ortho_scale / image_wh[1]
    z = depth

    # Concatenate to form the 3D coordinates in the camera frame
    camera_coords = torch.stack([x, y, z], dim=-1)

    # Apply the extrinsic matrix to get coordinates in the world frame
    coords_homogeneous = torch.nn.functional.pad(
        camera_coords, (0, 1), "constant", 1.0
    )  # Add a homogeneous coordinate
    world_coords = torch.matmul(
        coords_homogeneous.view(B, -1, 4), extrinsics.transpose(1, 2)
    ).view(B, H, W, 4)

    # Apply the mask to the position map
    position_map = world_coords[..., :3] * mask

    return position_map


def get_opencv_from_blender(matrix_world, fov=None, image_size=None):
    # convert matrix_world to opencv format extrinsics
    opencv_world_to_cam = matrix_world.inverse()
    opencv_world_to_cam[1, :] *= -1
    opencv_world_to_cam[2, :] *= -1
    R, T = opencv_world_to_cam[:3, :3], opencv_world_to_cam[:3, 3]

    if fov is None:  # orthographic camera
        return R, T

    R, T = R.unsqueeze(0), T.unsqueeze(0)
    # convert fov to opencv format intrinsics
    focal = 1 / np.tan(fov / 2)
    intrinsics = np.diag(np.array([focal, focal, 1])).astype(np.float32)
    opencv_cam_matrix = (
        torch.from_numpy(intrinsics).unsqueeze(0).float().to(matrix_world.device)
    )
    opencv_cam_matrix[:, :2, -1] += torch.tensor([image_size / 2, image_size / 2]).to(
        matrix_world.device
    )
    opencv_cam_matrix[:, [0, 1], [0, 1]] *= image_size / 2

    return R, T, opencv_cam_matrix


def get_ray_directions(
    H: int,
    W: int,
    focal: float,
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
) -> torch.Tensor:
    """
    Get ray directions for all pixels in camera coordinate.
    Args:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0
    cx, cy = W / 2, H / 2 if principal is None else principal
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )
    directions = torch.stack(
        [(i - cx) / focal, -(j - cy) / focal, -torch.ones_like(i)], -1
    )
    return F.normalize(directions, dim=-1)


def get_rays(
    directions: torch.Tensor, c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get ray origins and directions from camera coordinates to world coordinates
    Args:
        directions: (H, W, 3) ray directions in camera coordinates
        c2w: (4, 4) camera-to-world transformation matrix
    Outputs:
        rays_o, rays_d: (H, W, 3) ray origins and directions in world coordinates
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def compute_plucker_embed(
    c2w: torch.Tensor, image_width: int, image_height: int, focal: float
) -> torch.Tensor:
    """
    Computes Plucker coordinates for a camera.
    Args:
        c2w: (4, 4) camera-to-world transformation matrix
        image_width: Image width
        image_height: Image height
        focal: Focal length of the camera
    Returns:
        plucker: (6, H, W) Plucker embedding
    """
    directions = get_ray_directions(image_height, image_width, focal)
    rays_o, rays_d = get_rays(directions, c2w)
    # Cross product to get Plucker coordinates
    cross = torch.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat((rays_d, cross), dim=-1)
    return plucker.permute(2, 0, 1)


def get_plucker_embeds_from_cameras(
    c2w: List[torch.Tensor], fov: List[float], image_size: int
) -> torch.Tensor:
    """
    Given lists of camera transformations and fov, returns the batched plucker embeddings.
    Args:
        c2w: list of camera-to-world transformation matrices
        fov: list of field of view values
        image_size: size of the image
    Returns:
        plucker_embeds: (B, 6, H, W) batched plucker embeddings
    """
    plucker_embeds = []
    for cam_matrix, cam_fov in zip(c2w, fov):
        focal = 0.5 * image_size / np.tan(0.5 * cam_fov)
        plucker = compute_plucker_embed(cam_matrix, image_size, image_size, focal)
        plucker_embeds.append(plucker)
    return torch.stack(plucker_embeds)


def get_plucker_embeds_from_cameras_ortho(
    c2w: List[torch.Tensor], ortho_scale: List[float], image_size: int
):
    """
    Given lists of camera transformations and fov, returns the batched plucker embeddings.

    Parameters:
        c2w: list of camera-to-world transformation matrices
        fov: list of field of view values
        image_size: size of the image

    Returns:
        plucker_embeds: plucker embeddings (B, 6, H, W)
    """
    plucker_embeds = []
    # compute pairwise mask and plucker embeddings
    for cam_matrix, scale in zip(c2w, ortho_scale):
        # blender to opencv to pytorch3d
        R, T = get_opencv_from_blender(cam_matrix)
        cam_pos = -R.T @ T
        view_dir = R.T @ torch.tensor([0, 0, 1]).float().to(cam_matrix.device)
        # normalize camera position
        cam_pos = F.normalize(cam_pos, dim=0)
        plucker = torch.concat([view_dir, cam_pos])
        plucker = plucker.unsqueeze(-1).unsqueeze(-1).repeat(1, image_size, image_size)
        plucker_embeds.append(plucker)

    plucker_embeds = torch.stack(plucker_embeds)

    return plucker_embeds
