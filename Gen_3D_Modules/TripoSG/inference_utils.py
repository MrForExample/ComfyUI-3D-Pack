import numpy as np
import torch
import torch.nn as nn
import scipy.ndimage
from skimage import measure
from einops import repeat
from diso import DiffDMC
import torch.nn.functional as F

from TripoSG.utils.typing import *

def generate_dense_grid_points_gpu(bbox_min: torch.Tensor,
                                   bbox_max: torch.Tensor,
                                   octree_depth: int,
                                   indexing: str = "ij"):
    length = bbox_max - bbox_min
    num_cells = 2 ** octree_depth
    device = bbox_min.device
    
    x = torch.linspace(bbox_min[0], bbox_max[0], int(num_cells), dtype=torch.float16, device=device)
    y = torch.linspace(bbox_min[1], bbox_max[1], int(num_cells), dtype=torch.float16, device=device)
    z = torch.linspace(bbox_min[2], bbox_max[2], int(num_cells), dtype=torch.float16, device=device)
    
    xs, ys, zs = torch.meshgrid(x, y, z, indexing=indexing)
    xyz = torch.stack((xs, ys, zs), dim=-1)
    xyz = xyz.view(-1, 3)
    grid_size = [int(num_cells), int(num_cells), int(num_cells)]

    return xyz, grid_size, length

def find_mesh_grid_coordinates_fast_gpu(occupancy_grid, n_limits=-1):
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

def find_candidates_band(occupancy_grid: torch.Tensor, band_threshold: float, n_limits: int = -1) -> torch.Tensor:
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

def expand_edge_region_fast(edge_coords, grid_size):
    expanded_tensor = torch.zeros(grid_size, grid_size, grid_size, device='cuda', dtype=torch.float16, requires_grad=False)
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
def hierarchical_extract_geometry(geometric_func: Callable,
                     device: torch.device,
                     bounds: Union[Tuple[float], List[float], float] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
                     dense_octree_depth: int = 8,
                     hierarchical_octree_depth: int = 9,
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
        indexing="ij"
    )
    
    print(f'step 1 query num: {xyz_samples.shape[0]}')
    grid_logits = geometric_func(xyz_samples.unsqueeze(0)).to(torch.float16).view(grid_size[0], grid_size[1], grid_size[2])
    # print(f'step 1 grid_logits shape: {grid_logits.shape}')
    for i in range(hierarchical_octree_depth - dense_octree_depth):
        curr_octree_depth = dense_octree_depth + i + 1
        # upsample
        grid_size = 2**curr_octree_depth
        normalize_offset = grid_size / 2
        high_res_occupancy = parallel_zoom(grid_logits, 2)

        band_threshold = 1.0
        edge_coords = find_candidates_band(grid_logits, band_threshold)
        expanded_coords = expand_edge_region_fast(edge_coords, grid_size=int(grid_size/2)).to(torch.float16)
        print(f'step {i+2} query num: {len(expanded_coords)}')
        expanded_coords_norm = (expanded_coords - normalize_offset) * (abs(bounds[0]) / normalize_offset)

        all_logits = None

        all_logits = geometric_func(expanded_coords_norm.unsqueeze(0)).to(torch.float16)
        all_logits = torch.cat([expanded_coords_norm, all_logits[0]], dim=1)
        # print("all logits shape = ", all_logits.shape)

        indices = all_logits[..., :3]
        indices = indices * (normalize_offset / abs(bounds[0]))  + normalize_offset
        indices = indices.type(torch.IntTensor)
        values = all_logits[:, 3]
        # breakpoint()
        high_res_occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = values
        grid_logits = high_res_occupancy
        torch.cuda.empty_cache()
    mesh_v_f = []
    try:
        print("final grids shape = ", grid_logits.shape)
        vertices, faces, normals, _ = measure.marching_cubes(grid_logits.float().cpu().numpy(), 0, method="lewiner")
        vertices = vertices / (2**hierarchical_octree_depth) * bbox_size.cpu().numpy() + bbox_min.cpu().numpy()
        mesh_v_f = (vertices.astype(np.float32), np.ascontiguousarray(faces))
    except Exception as e:
        print(e)
        torch.cuda.empty_cache()
        mesh_v_f = (None, None)

    return [mesh_v_f]

def extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float):
    """
    Args:
        input_tensor: shape [D, D, D], torch.float16
        alpha: isosurface offset
    Returns:
        mask: shape [D, D, D], torch.int32
    """
    device = input_tensor.device
    D = input_tensor.shape[0]
    signed_val = 0.0

    # add isosurface offset and exclude invalid value
    val = input_tensor + alpha
    valid_mask = val > -9000

    # obtain neighbors
    def get_neighbor(t, shift, axis):
        if shift == 0:
            return t.clone()

        pad_dims = [0, 0, 0, 0, 0, 0]  # [x_front，x_back，y_front，y_back，z_front，z_back]

        if axis == 0:  # x axis
            pad_idx = 0 if shift > 0 else 1
            pad_dims[pad_idx] = abs(shift)
        elif axis == 1:  # y axis
            pad_idx = 2 if shift > 0 else 3
            pad_dims[pad_idx] = abs(shift)
        elif axis == 2:  # z axis
            pad_idx = 4 if shift > 0 else 5
            pad_dims[pad_idx] = abs(shift)

        # Apply padding with replication at boundaries
        padded = F.pad(t.unsqueeze(0).unsqueeze(0), pad_dims[::-1], mode='replicate')

        # Create dynamic slicing indices
        slice_dims = [slice(None)] * 3
        if axis == 0:  # x axis
            if shift > 0:
                slice_dims[0] = slice(shift, None)
            else:
                slice_dims[0] = slice(None, shift)
        elif axis == 1:  # y axis
            if shift > 0:
                slice_dims[1] = slice(shift, None)
            else:
                slice_dims[1] = slice(None, shift)
        elif axis == 2:  # z axis
            if shift > 0:
                slice_dims[2] = slice(shift, None)
            else:
                slice_dims[2] = slice(None, shift)

        # Apply slicing and restore dimensions
        padded = padded.squeeze(0).squeeze(0)
        sliced = padded[slice_dims]
        return sliced

    # Get neighbors in all directions
    left = get_neighbor(val, 1, axis=0)  # x axis
    right = get_neighbor(val, -1, axis=0)
    back = get_neighbor(val, 1, axis=1)  # y axis
    front = get_neighbor(val, -1, axis=1)
    down = get_neighbor(val, 1, axis=2)  # z axis
    up = get_neighbor(val, -1, axis=2)

    # Handle invalid boundary values
    def safe_where(neighbor):
        return torch.where(neighbor > -9000, neighbor, val)

    left = safe_where(left)
    right = safe_where(right)
    back = safe_where(back)
    front = safe_where(front)
    down = safe_where(down)
    up = safe_where(up)

    # Calculate sign consistency
    sign = torch.sign(val.to(torch.float32))
    neighbors_sign = torch.stack([
        torch.sign(left.to(torch.float32)),
        torch.sign(right.to(torch.float32)),
        torch.sign(back.to(torch.float32)),
        torch.sign(front.to(torch.float32)),
        torch.sign(down.to(torch.float32)),
        torch.sign(up.to(torch.float32))
    ], dim=0)

    # Check if all signs are consistent
    same_sign = torch.all(neighbors_sign == sign, dim=0)

    # Generate final mask
    mask = (~same_sign).to(torch.int32)
    return mask * valid_mask.to(torch.int32)


def generate_dense_grid_points_2(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_resolution: int,
    indexing: str = "ij",
):
    length = bbox_max - bbox_min
    num_cells = octree_resolution

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length

@torch.no_grad()
def flash_extract_geometry(
    latents: torch.FloatTensor,
    vae: Callable,
    bounds: Union[Tuple[float], List[float], float] = 1.01,
    num_chunks: int = 10000,
    mc_level: float = 0.0,
    octree_depth: int = 9,
    min_resolution: int = 63,
    mini_grid_num: int = 4,
    **kwargs,
):
    geo_decoder = vae.decoder
    device = latents.device
    dtype = latents.dtype
    # resolution to depth
    octree_resolution = 2 ** octree_depth
    resolutions = []
    if octree_resolution < min_resolution:
        resolutions.append(octree_resolution)
    while octree_resolution >= min_resolution:
        resolutions.append(octree_resolution)
        octree_resolution = octree_resolution // 2
    resolutions.reverse()
    resolutions[0] = round(resolutions[0] / mini_grid_num) * mini_grid_num - 1
    for i, resolution in enumerate(resolutions[1:]):
        resolutions[i + 1] = resolutions[0] * 2 ** (i + 1)


    # 1. generate query points
    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
    bbox_min = np.array(bounds[0:3])
    bbox_max = np.array(bounds[3:6])
    bbox_size = bbox_max - bbox_min

    xyz_samples, grid_size, length = generate_dense_grid_points_2(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        octree_resolution=resolutions[0],
        indexing="ij"
    )

    dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
    dilate.weight = torch.nn.Parameter(torch.ones(dilate.weight.shape, dtype=dtype, device=device))

    grid_size = np.array(grid_size)

    # 2. latents to 3d volume
    xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype)
    batch_size = latents.shape[0]
    mini_grid_size = xyz_samples.shape[0] // mini_grid_num
    xyz_samples = xyz_samples.view(
        mini_grid_num, mini_grid_size,
        mini_grid_num, mini_grid_size,
        mini_grid_num, mini_grid_size, 3
    ).permute(
        0, 2, 4, 1, 3, 5, 6
    ).reshape(
        -1, mini_grid_size * mini_grid_size * mini_grid_size, 3
    )
    batch_logits = []
    num_batchs = max(num_chunks // xyz_samples.shape[1], 1)
    for start in range(0, xyz_samples.shape[0], num_batchs):
        queries = xyz_samples[start: start + num_batchs, :]
        batch = queries.shape[0]
        batch_latents = repeat(latents.squeeze(0), "p c -> b p c", b=batch)
        # geo_decoder.set_topk(True)
        geo_decoder.set_topk(False)
        logits = vae.decode(batch_latents, queries).sample
        batch_logits.append(logits)
    grid_logits = torch.cat(batch_logits, dim=0).reshape(
        mini_grid_num, mini_grid_num, mini_grid_num,
        mini_grid_size, mini_grid_size,
        mini_grid_size
    ).permute(0, 3, 1, 4, 2, 5).contiguous().view(
        (batch_size, grid_size[0], grid_size[1], grid_size[2])
    )

    for octree_depth_now in resolutions[1:]:
        grid_size = np.array([octree_depth_now + 1] * 3)
        resolution = bbox_size / octree_depth_now
        next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=device)
        next_logits = torch.full(next_index.shape, -10000., dtype=dtype, device=device)
        curr_points = extract_near_surface_volume_fn(grid_logits.squeeze(0), mc_level)
        curr_points += grid_logits.squeeze(0).abs() < 0.95

        if octree_depth_now == resolutions[-1]:
            expand_num = 0
        else:
            expand_num = 1
        for i in range(expand_num):
            curr_points = dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)
            curr_points = dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)
        (cidx_x, cidx_y, cidx_z) = torch.where(curr_points > 0)

        next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
        for i in range(2 - expand_num):
            next_index = dilate(next_index.unsqueeze(0)).squeeze(0)
        nidx = torch.where(next_index > 0)

        next_points = torch.stack(nidx, dim=1)
        next_points = (next_points * torch.tensor(resolution, dtype=torch.float32, device=device) +
                        torch.tensor(bbox_min, dtype=torch.float32, device=device))

        query_grid_num = 6
        min_val = next_points.min(axis=0).values
        max_val = next_points.max(axis=0).values
        vol_queries_index = (next_points - min_val) / (max_val - min_val) * (query_grid_num - 0.001)
        index = torch.floor(vol_queries_index).long()
        index = index[..., 0] * (query_grid_num ** 2) + index[..., 1] * query_grid_num + index[..., 2]
        index = index.sort()
        next_points = next_points[index.indices].unsqueeze(0).contiguous()
        unique_values = torch.unique(index.values, return_counts=True)
        grid_logits = torch.zeros((next_points.shape[1]), dtype=latents.dtype, device=latents.device)
        input_grid = [[], []]
        logits_grid_list = []
        start_num = 0
        sum_num = 0
        for grid_index, count in zip(unique_values[0].cpu().tolist(), unique_values[1].cpu().tolist()):
            if sum_num + count < num_chunks or sum_num == 0:
                sum_num += count
                input_grid[0].append(grid_index)
                input_grid[1].append(count)
            else:
                # geo_decoder.set_topk(input_grid)
                geo_decoder.set_topk(False)
                logits_grid = vae.decode(latents,next_points[:, start_num:start_num + sum_num]).sample
                start_num = start_num + sum_num
                logits_grid_list.append(logits_grid)
                input_grid = [[grid_index], [count]]
                sum_num = count
        if sum_num > 0:
            # geo_decoder.set_topk(input_grid)
            geo_decoder.set_topk(False)
            logits_grid = vae.decode(latents,next_points[:, start_num:start_num + sum_num]).sample
            logits_grid_list.append(logits_grid)
        logits_grid = torch.cat(logits_grid_list, dim=1)
        grid_logits[index.indices] = logits_grid.squeeze(0).squeeze(-1)
        next_logits[nidx] = grid_logits
        grid_logits = next_logits.unsqueeze(0)
    
    grid_logits[grid_logits == -10000.] = float('nan')
    torch.cuda.empty_cache()
    mesh_v_f = []
    grid_logits = grid_logits[0]
    try:
        print("final grids shape = ", grid_logits.shape)
        dmc = DiffDMC(dtype=torch.float32).to(grid_logits.device)
        sdf = -grid_logits / octree_resolution
        sdf = sdf.to(torch.float32).contiguous()
        vertices, faces = dmc(sdf, deform=None, return_quads=False, normalize=False)
        vertices = vertices.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()[:, ::-1]        
        vertices = vertices / (2 ** octree_depth) * bbox_size + bbox_min
        mesh_v_f = (vertices.astype(np.float32), np.ascontiguousarray(faces))
    except Exception as e:
        print(e)
        torch.cuda.empty_cache()
        mesh_v_f = (None, None)

    return [mesh_v_f]