from partcrafter_src.utils.typing_utils import *

import numpy as np
import torch
import torch.nn as nn
import scipy.ndimage
from skimage import measure
from einops import repeat
import torch.nn.functional as F

def generate_dense_grid_points(
    bbox_min: np.ndarray, bbox_max: np.ndarray, octree_depth: int, indexing: str = "ij"
):
    length = bbox_max - bbox_min
    num_cells = np.exp2(octree_depth)
    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length

def generate_dense_grid_points_gpu(
    bbox_min: torch.Tensor,                        
    bbox_max: torch.Tensor,
    octree_depth: int,
    indexing: str = "ij", 
    dtype: torch.dtype = torch.float16
):
    length = bbox_max - bbox_min
    num_cells = 2 ** octree_depth
    device = bbox_min.device
    
    x = torch.linspace(bbox_min[0], bbox_max[0], int(num_cells), dtype=dtype, device=device)
    y = torch.linspace(bbox_min[1], bbox_max[1], int(num_cells), dtype=dtype, device=device)
    z = torch.linspace(bbox_min[2], bbox_max[2], int(num_cells), dtype=dtype, device=device)
    
    xs, ys, zs = torch.meshgrid(x, y, z, indexing=indexing)
    xyz = torch.stack((xs, ys, zs), dim=-1)
    xyz = xyz.view(-1, 3)
    grid_size = [int(num_cells), int(num_cells), int(num_cells)]

    return xyz, grid_size, length

def find_mesh_grid_coordinates_fast_gpu(
    occupancy_grid, 
    n_limits=-1
):
    core_grid = occupancy_grid[1:-1, 1:-1, 1:-1]
    occupied = core_grid > 0

    neighbors_unoccupied = (
        (occupancy_grid[:-2, :-2, :-2] < 0)
        | (occupancy_grid[:-2, :-2, 1:-1] < 0)
        | (occupancy_grid[:-2, :-2, 2:] < 0)  # x-1, y-1, z-1/0/1
        | (occupancy_grid[:-2, 1:-1, :-2] < 0)
        | (occupancy_grid[:-2, 1:-1, 1:-1] < 0)
        | (occupancy_grid[:-2, 1:-1, 2:] < 0)  # x-1, y0, z-1/0/1
        | (occupancy_grid[:-2, 2:, :-2] < 0)
        | (occupancy_grid[:-2, 2:, 1:-1] < 0)
        | (occupancy_grid[:-2, 2:, 2:] < 0)  # x-1, y+1, z-1/0/1
        | (occupancy_grid[1:-1, :-2, :-2] < 0)
        | (occupancy_grid[1:-1, :-2, 1:-1] < 0)
        | (occupancy_grid[1:-1, :-2, 2:] < 0)  # x0, y-1, z-1/0/1
        | (occupancy_grid[1:-1, 1:-1, :-2] < 0)
        | (occupancy_grid[1:-1, 1:-1, 2:] < 0)  # x0, y0, z-1/1
        | (occupancy_grid[1:-1, 2:, :-2] < 0)
        | (occupancy_grid[1:-1, 2:, 1:-1] < 0)
        | (occupancy_grid[1:-1, 2:, 2:] < 0)  # x0, y+1, z-1/0/1
        | (occupancy_grid[2:, :-2, :-2] < 0)
        | (occupancy_grid[2:, :-2, 1:-1] < 0)
        | (occupancy_grid[2:, :-2, 2:] < 0)  # x+1, y-1, z-1/0/1
        | (occupancy_grid[2:, 1:-1, :-2] < 0)
        | (occupancy_grid[2:, 1:-1, 1:-1] < 0)
        | (occupancy_grid[2:, 1:-1, 2:] < 0)  # x+1, y0, z-1/0/1
        | (occupancy_grid[2:, 2:, :-2] < 0)
        | (occupancy_grid[2:, 2:, 1:-1] < 0)
        | (occupancy_grid[2:, 2:, 2:] < 0)  # x+1, y+1, z-1/0/1
    )
    core_mesh_coords = torch.nonzero(occupied & neighbors_unoccupied, as_tuple=False) + 1

    if n_limits != -1 and core_mesh_coords.shape[0] > n_limits:
        print(f"core mesh coords {core_mesh_coords.shape[0]} is too large, limited to {n_limits}")
        ind = np.random.choice(core_mesh_coords.shape[0], n_limits, True)
        core_mesh_coords = core_mesh_coords[ind]

    return core_mesh_coords

def find_candidates_band(
    occupancy_grid: torch.Tensor, 
    band_threshold: float, 
    n_limits: int = -1
) -> torch.Tensor:
    """
    Returns the coordinates of all voxels in the occupancy_grid where |value| < band_threshold.

    Args:
        occupancy_grid (torch.Tensor): A 3D tensor of SDF values.
        band_threshold (float): The threshold below which |SDF| must be to include the voxel.
        n_limits (int): Maximum number of points to return (-1 for no limit)

    Returns:
        torch.Tensor: A 2D tensor of coordinates (N x 3) where each row is [x, y, z].
    """
    core_grid = occupancy_grid[1:-1, 1:-1, 1:-1]  
    # logits to sdf
    core_grid = torch.sigmoid(core_grid) * 2 - 1  
    # Create a boolean mask for all cells in the band
    in_band = torch.abs(core_grid) < band_threshold

    # Get coordinates of all voxels in the band
    core_mesh_coords = torch.nonzero(in_band, as_tuple=False) + 1

    if n_limits != -1 and core_mesh_coords.shape[0] > n_limits:
        print(f"core mesh coords {core_mesh_coords.shape[0]} is too large, limited to {n_limits}")
        ind = np.random.choice(core_mesh_coords.shape[0], n_limits, True)
        core_mesh_coords = core_mesh_coords[ind]

    return core_mesh_coords 

def expand_edge_region_fast(edge_coords, grid_size, dtype):
    expanded_tensor = torch.zeros(grid_size, grid_size, grid_size, device='cuda', dtype=dtype, requires_grad=False)
    expanded_tensor[edge_coords[:, 0], edge_coords[:, 1], edge_coords[:, 2]] = 1
    if grid_size < 512:
        kernel_size = 5
        pooled_tensor = torch.nn.functional.max_pool3d(expanded_tensor.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=2).squeeze()
    else:
        kernel_size = 3
        pooled_tensor = torch.nn.functional.max_pool3d(expanded_tensor.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=1).squeeze()
    expanded_coords_low_res = torch.nonzero(pooled_tensor, as_tuple=False).to(torch.int16)

    expanded_coords_high_res = torch.stack([
        torch.cat((expanded_coords_low_res[:, 0] * 2, expanded_coords_low_res[:, 0] * 2, expanded_coords_low_res[:, 0] * 2, expanded_coords_low_res[:, 0] * 2, expanded_coords_low_res[:, 0] * 2 + 1, expanded_coords_low_res[:, 0] * 2 + 1, expanded_coords_low_res[:, 0] * 2 + 1, expanded_coords_low_res[:, 0] * 2 + 1)),
        torch.cat((expanded_coords_low_res[:, 1] * 2, expanded_coords_low_res[:, 1] * 2, expanded_coords_low_res[:, 1] * 2+1, expanded_coords_low_res[:, 1] * 2 + 1, expanded_coords_low_res[:, 1] * 2, expanded_coords_low_res[:, 1] * 2, expanded_coords_low_res[:, 1] * 2 + 1, expanded_coords_low_res[:, 1] * 2 + 1)),
        torch.cat((expanded_coords_low_res[:, 2] * 2, expanded_coords_low_res[:, 2] * 2+1, expanded_coords_low_res[:, 2] * 2, expanded_coords_low_res[:, 2] * 2 + 1, expanded_coords_low_res[:, 2] * 2, expanded_coords_low_res[:, 2] * 2+1, expanded_coords_low_res[:, 2] * 2, expanded_coords_low_res[:, 2] * 2 + 1))
    ], dim=1)

    return expanded_coords_high_res

def zoom_block(block, scale_factor, order=3):
    block = block.astype(np.float32)
    return scipy.ndimage.zoom(block, scale_factor, order=order)

def parallel_zoom(occupancy_grid, scale_factor):
    result = torch.nn.functional.interpolate(occupancy_grid.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor)
    return result.squeeze(0).squeeze(0)


@torch.no_grad()
def hierarchical_extract_geometry(
    geometric_func: Callable,
    device: torch.device,
    dtype: torch.dtype,
    bounds: Union[Tuple[float], List[float], float] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
    dense_octree_depth: int = 8,
    hierarchical_octree_depth: int = 9, 
    max_num_expanded_coords: int = 1e8, 
    verbose: bool = False,
):
    """
    Args:
        geometric_func:
        device:
        bounds:
        dense_octree_depth:
        hierarchical_octree_depth:
    Returns:
    """
    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

    bbox_min = torch.tensor(bounds[0:3]).to(device)
    bbox_max = torch.tensor(bounds[3:6]).to(device)
    bbox_size = bbox_max - bbox_min

    xyz_samples, grid_size, length = generate_dense_grid_points_gpu(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        octree_depth=dense_octree_depth,
        indexing="ij",
        dtype=dtype
    )
    
    if verbose:
        print(f'step 1 query num: {xyz_samples.shape[0]}')
    grid_logits = geometric_func(xyz_samples.unsqueeze(0)).to(dtype).view(grid_size[0], grid_size[1], grid_size[2])
    # print(f'step 1 grid_logits shape: {grid_logits.shape}')
    for i in range(hierarchical_octree_depth - dense_octree_depth):
        curr_octree_depth = dense_octree_depth + i + 1
        # upsample
        grid_size = 2**curr_octree_depth
        normalize_offset = grid_size / 2
        high_res_occupancy = parallel_zoom(grid_logits, 2).to(dtype)

        band_threshold = 1.0
        edge_coords = find_candidates_band(grid_logits, band_threshold)
        expanded_coords = expand_edge_region_fast(edge_coords, grid_size=int(grid_size/2), dtype=dtype).to(dtype)
        if verbose:
            print(f'step {i+2} query num: {len(expanded_coords)}')
        if max_num_expanded_coords > 0 and len(expanded_coords) > max_num_expanded_coords:
            raise ValueError(f"expanded_coords is too large, {len(expanded_coords)} > {max_num_expanded_coords}")
        expanded_coords_norm = (expanded_coords - normalize_offset) * (abs(bounds[0]) / normalize_offset)

        all_logits = None

        all_logits = geometric_func(expanded_coords_norm.unsqueeze(0)).to(dtype)
        all_logits = torch.cat([expanded_coords_norm, all_logits[0]], dim=1)
        # print("all logits shape = ", all_logits.shape)

        indices = all_logits[..., :3]
        indices = indices * (normalize_offset / abs(bounds[0]))  + normalize_offset
        indices = indices.type(torch.IntTensor)
        values = all_logits[:, 3]
        # breakpoint()
        high_res_occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = values
        grid_logits = high_res_occupancy
        # torch.cuda.empty_cache()

    if verbose:
        print("final grids shape = ", grid_logits.shape)
    vertices, faces, normals, _ = measure.marching_cubes(grid_logits.float().cpu().numpy(), 0, method="lewiner")
    vertices = vertices / (2**hierarchical_octree_depth) * bbox_size.cpu().numpy() + bbox_min.cpu().numpy()
    mesh_v_f = (vertices.astype(np.float32), np.ascontiguousarray(faces))

    return mesh_v_f