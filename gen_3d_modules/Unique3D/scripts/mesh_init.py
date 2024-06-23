from PIL import Image
import torch
import numpy as np
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex
from .utils import meshlab_mesh_to_py3dmesh, py3dmesh_to_meshlab_mesh
import pymeshlab

_MAX_THREAD = 8

def fast_geo(front_normal: Image.Image, back_normal: Image.Image, side_normal: Image.Image, clamp=0., init_type="std"):
    front_normal = front_normal.resize((192, 192))
    back_normal = back_normal.resize((192, 192))
    side_normal = side_normal.resize((192, 192))
    
    # build mesh with front back projection # ~3s
    side_w_over_h = calc_w_over_h(side_normal)
    mesh_front = build_mesh(front_normal, front_normal, clamp_min=clamp, scale=side_w_over_h, init_type=init_type)
    mesh_back = build_mesh(back_normal, back_normal, is_back=True, clamp_min=clamp, scale=side_w_over_h, init_type=init_type)
    meshes = join_meshes_as_scene([mesh_front, mesh_back])
    meshes = fix_border_with_pymeshlab_fast(meshes, poissson_depth=6, simplification=2000)
    return meshes

# rgb and depth to mesh
def get_ortho_ray_directions_origins(W, H, use_pixel_centers=True, device="cuda"):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i).to(device), torch.from_numpy(j).to(device)

    origins = torch.stack([(i/W-0.5)*2, (j/H-0.5)*2 * H / W, torch.zeros_like(i)], dim=-1) # W, H, 3
    directions = torch.stack([torch.zeros_like(i), torch.zeros_like(j), torch.ones_like(i)], dim=-1) # W, H, 3

    return origins, directions

def depth_and_color_to_mesh(rgb_BCHW, pred_HWC, valid_HWC=None, is_back=False):
    if valid_HWC is None:
        valid_HWC = torch.ones_like(pred_HWC).bool()
    H, W = rgb_BCHW.shape[-2:]
    rgb_BCHW = rgb_BCHW.flip(-2)
    pred_HWC = pred_HWC.flip(0)
    valid_HWC = valid_HWC.flip(0)
    rays_o, rays_d = get_ortho_ray_directions_origins(W, H, device=rgb_BCHW.device)
    verts = rays_o + rays_d * pred_HWC  # [H, W, 3]
    verts = verts.reshape(-1, 3)    # [V, 3]
    indexes = torch.arange(H * W).reshape(H, W).to(rgb_BCHW.device)
    faces1 = torch.stack([indexes[:-1, :-1], indexes[:-1, 1:], indexes[1:, :-1]], dim=-1)
    # faces1_valid = valid_HWC[:-1, :-1] | valid_HWC[:-1, 1:] | valid_HWC[1:, :-1]
    faces1_valid = valid_HWC[:-1, :-1] & valid_HWC[:-1, 1:] & valid_HWC[1:, :-1]
    faces2 = torch.stack([indexes[1:, 1:], indexes[1:, :-1], indexes[:-1, 1:]], dim=-1)
    # faces2_valid = valid_HWC[1:, 1:] | valid_HWC[1:, :-1] | valid_HWC[:-1, 1:]
    faces2_valid = valid_HWC[1:, 1:] & valid_HWC[1:, :-1] & valid_HWC[:-1, 1:]
    faces = torch.cat([faces1[faces1_valid.expand_as(faces1)].reshape(-1, 3), faces2[faces2_valid.expand_as(faces2)].reshape(-1, 3)], dim=0)  # (F, 3)
    colors = (rgb_BCHW[0].permute((1,2,0)) / 2 + 0.5).reshape(-1, 3)  # (V, 3)
    if is_back:
        verts = verts * torch.tensor([-1, 1, -1], dtype=verts.dtype, device=verts.device) 
    
    used_verts = faces.unique()
    old_to_new_mapping = torch.zeros_like(verts[..., 0]).long()
    old_to_new_mapping[used_verts] = torch.arange(used_verts.shape[0], device=verts.device)
    new_faces = old_to_new_mapping[faces]
    mesh = Meshes(verts=[verts[used_verts]], faces=[new_faces], textures=TexturesVertex(verts_features=[colors[used_verts]]))
    return mesh

def normalmap_to_depthmap(normal_np):
    from .normal_to_height_map import estimate_height_map
    height = estimate_height_map(normal_np, raw_values=True, thread_count=_MAX_THREAD, target_iteration_count=96)
    return height

def transform_back_normal_to_front(normal_pil):
    arr = np.array(normal_pil)  # in [0, 255]
    arr[..., 0] = 255-arr[..., 0]
    arr[..., 2] = 255-arr[..., 2]
    return Image.fromarray(arr.astype(np.uint8))

def calc_w_over_h(normal_pil):
    if isinstance(normal_pil, Image.Image):
        arr = np.array(normal_pil)
    else:
        assert isinstance(normal_pil, np.ndarray)
        arr = normal_pil
    if arr.shape[-1] == 4:
        alpha = arr[..., -1] / 255.
        alpha[alpha >= 0.5] = 1
        alpha[alpha < 0.5] = 0
    else:
        alpha = ~(arr.min(axis=-1) >= 250)
    h_min, w_min = np.min(np.where(alpha), axis=1)
    h_max, w_max = np.max(np.where(alpha), axis=1)
    return (w_max - w_min) / (h_max - h_min)

def build_mesh(normal_pil, rgb_pil, is_back=False, clamp_min=-1, scale=0.3, init_type="std", offset=0):
    if is_back:
        normal_pil = transform_back_normal_to_front(normal_pil)
    normal_img = np.array(normal_pil)
    rgb_img = np.array(rgb_pil)
    if normal_img.shape[-1] == 4:
        valid_HWC = normal_img[..., [3]] / 255
    elif rgb_img.shape[-1] == 4:
        valid_HWC = rgb_img[..., [3]] / 255
    else:
        raise ValueError("invalid input, either normal or rgb should have alpha channel")
    
    real_height_pix = np.max(np.where(valid_HWC>0.5)[0]) - np.min(np.where(valid_HWC>0.5)[0])

    heights = normalmap_to_depthmap(normal_img)
    rgb_BCHW = torch.from_numpy(rgb_img[..., :3] / 255.).permute((2,0,1))[None]
    valid_HWC[valid_HWC < 0.5] = 0
    valid_HWC[valid_HWC >= 0.5] = 1
    valid_HWC = torch.from_numpy(valid_HWC).bool()
    if init_type == "std":
        # accurate but not stable
        pred_HWC = torch.from_numpy(heights / heights.max() * (real_height_pix / heights.shape[0]) * scale * 2).float()[..., None]
    elif init_type == "thin":
        heights = heights - heights.min()
        heights = (heights / heights.max() * 0.2)
        pred_HWC = torch.from_numpy(heights * scale).float()[..., None]
    else:
        # stable but not accurate
        heights = heights - heights.min()
        heights = (heights / heights.max() * (1-offset)) + offset # to [0.2, 1]
        pred_HWC = torch.from_numpy(heights * scale).float()[..., None]
    
    # set the boarder pixels to 0 height
    import cv2
    # edge filter
    edge = cv2.Canny((valid_HWC[..., 0] * 255).numpy().astype(np.uint8), 0, 255)
    edge = torch.from_numpy(edge).bool()[..., None]
    pred_HWC[edge] = 0
    
    valid_HWC[pred_HWC < clamp_min] = False
    return depth_and_color_to_mesh(rgb_BCHW.cuda(), pred_HWC.cuda(), valid_HWC.cuda(), is_back)

def fix_border_with_pymeshlab_fast(meshes: Meshes, poissson_depth=6, simplification=0):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(py3dmesh_to_meshlab_mesh(meshes), "cube_vcolor_mesh")
    if simplification > 0:
        ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=simplification, preservetopology=True)
    ms.apply_filter('generate_surface_reconstruction_screened_poisson', threads = 6, depth = poissson_depth, preclean = True)
    if simplification > 0:
        ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=simplification, preservetopology=True)
    return meshlab_mesh_to_py3dmesh(ms.current_mesh())
