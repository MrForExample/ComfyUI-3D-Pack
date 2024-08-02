import os
import math
import copy
from enum import Enum
from collections import OrderedDict
import folder_paths as comfy_paths
from omegaconf import OmegaConf
import json

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import numpy as np
from safetensors.torch import load_file
from einops import rearrange

from diffusers import (
    DiffusionPipeline, 
    StableDiffusionPipeline
)

from diffusers import (
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    DDIMParallelScheduler,
    LCMScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
)
from huggingface_hub import snapshot_download

from plyfile import PlyData
from PIL import Image

from .mesh_processer.mesh import Mesh
from .mesh_processer.mesh_utils import (
    ply_to_points_cloud, 
    get_target_axis_and_scale, 
    switch_ply_axis_and_scale, 
    switch_mesh_axis_and_scale, 
    calculate_max_sh_degree_from_gs_ply,
    marching_cubes_density_to_mesh,
    color_func_to_albedo,
)

from FlexiCubes.flexicubes_trainer import FlexiCubesTrainer
from DiffRastMesh.diff_mesh import DiffMesh, DiffMeshCameraController
from DiffRastMesh.diff_mesh import DiffRastRenderer
from GaussianSplatting.main_3DGS import GaussianSplatting3D, GaussianSplattingCameraController, GSParams
from GaussianSplatting.main_3DGS_renderer import GaussianSplattingRenderer
from NeRF.Instant_NGP import InstantNGP

from TriplaneGaussian.triplane_gaussian_transformers import TGS
from TriplaneGaussian.utils.config import ExperimentConfig as ExperimentConfigTGS, load_config as load_config_tgs
from TriplaneGaussian.data import CustomImageOrbitDataset
from TriplaneGaussian.utils.misc import todevice, get_device
from LGM.core.options import config_defaults
from LGM.mvdream.pipeline_mvdream import MVDreamPipeline
from LGM.large_multiview_gaussian_model import LargeMultiviewGaussianModel
from LGM.nerf_marching_cubes_converter import GSConverterNeRFMarchingCubes
from TripoSR.system import TSR
from StableFast3D.sf3d import utils as sf3d_utils
from StableFast3D.sf3d.system import SF3D
from InstantMesh.utils.camera_util import oribt_camera_poses_to_input_cameras
from CRM.model.crm.model import ConvolutionalReconstructionModel
from CRM.model.crm.sampler import CRMSampler
from Wonder3D.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
from Wonder3D.data.single_image_dataset import SingleImageDataset as MVSingleImageDataset
from Wonder3D.utils.misc import load_config as load_config_wonder3d
from Zero123Plus.pipeline import Zero123PlusPipeline
from Era3D.mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from Era3D.mvdiffusion.data.single_image_dataset import SingleImageDataset as Era3DSingleImageDataset
from Era3D.utils.misc import load_config as load_config_era3d
from Unique3D.custum_3d_diffusion.custum_pipeline.unifield_pipeline_img2mvimg import StableDiffusionImage2MVCustomPipeline
from Unique3D.custum_3d_diffusion.custum_pipeline.unifield_pipeline_img2img import StableDiffusionImageCustomPipeline
from Unique3D.scripts.mesh_init import fast_geo
from Unique3D.scripts.utils import from_py3d_mesh, to_py3d_mesh, to_pyml_mesh, simple_clean_mesh
from Unique3D.scripts.project_mesh import multiview_color_projection, get_cameras_list
from Unique3D.mesh_reconstruction.recon import reconstruct_stage1
from Unique3D.mesh_reconstruction.refine import run_mesh_refine
from CharacterGen.character_inference import Inference2D_API, Inference3D_API
from CharacterGen.Stage_3D.lrm.utils.config import load_config as load_config_cg3d
import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.config import ExperimentConfig as ExperimentConfigCraftsman, load_config as load_config_craftsman

from .shared_utils.image_utils import (
    prepare_torch_img, torch_imgs_to_pils, troch_image_dilate, 
    pils_rgba_to_rgb, pil_make_image_grid, pil_split_image, pils_to_torch_imgs, pils_resize_foreground
)
from .shared_utils.log_utils import cstr
from .shared_utils.common_utils import parse_save_filename, get_list_filenames, resume_or_download_model_from_hf

DIFFUSERS_PIPE_DICT = OrderedDict([
    ("MVDreamPipeline", MVDreamPipeline),
    ("Wonder3DMVDiffusionPipeline", MVDiffusionImagePipeline),
    ("Zero123PlusPipeline", Zero123PlusPipeline),
    ("DiffusionPipeline", DiffusionPipeline),
    ("StableDiffusionPipeline", StableDiffusionPipeline),
    ("Era3DPipeline", StableUnCLIPImg2ImgPipeline),
    ("Unique3DImage2MVCustomPipeline", StableDiffusionImage2MVCustomPipeline),
    ("Unique3DImageCustomPipeline", StableDiffusionImageCustomPipeline),
])

DIFFUSERS_SCHEDULER_DICT = OrderedDict([
    ("EulerAncestralDiscreteScheduler", EulerAncestralDiscreteScheduler),
    ("Wonder3DMVDiffusionPipeline", MVDiffusionImagePipeline),
    ("EulerDiscreteScheduler,", EulerDiscreteScheduler),
    ("DDIMScheduler,", DDIMScheduler),
    ("DDIMParallelScheduler,", DDIMParallelScheduler),
    ("LCMScheduler,", LCMScheduler),
    ("KDPM2AncestralDiscreteScheduler,", KDPM2AncestralDiscreteScheduler),
    ("KDPM2DiscreteScheduler,", KDPM2DiscreteScheduler),
])

ROOT_PATH = os.path.join(comfy_paths.get_folder_paths("custom_nodes")[0], "ComfyUI-3D-Pack")
CKPT_ROOT_PATH = os.path.join(ROOT_PATH, "Checkpoints")
CKPT_DIFFUSERS_PATH = os.path.join(CKPT_ROOT_PATH, "Diffusers")
CONFIG_ROOT_PATH = os.path.join(ROOT_PATH, "Configs")
MODULE_ROOT_PATH = os.path.join(ROOT_PATH, "Gen_3D_Modules")

MANIFEST = {
    "name": "ComfyUI-3D-Pack",
    "version": (0,0,2),
    "author": "Mr. For Example",
    "project": "https://github.com/MrForExample/ComfyUI-3D-Pack",
    "description": "An extensive node suite that enables ComfyUI to process 3D inputs (Mesh & UV Texture, etc) using cutting edge algorithms (3DGS, NeRF, etc.)",
}

SUPPORTED_3D_EXTENSIONS = (
    '.obj',
    '.ply',
    '.glb',
)

SUPPORTED_3DGS_EXTENSIONS = (
    '.ply',
)

SUPPORTED_CHECKPOINTS_EXTENSIONS = (
    '.ckpt', 
    '.bin', 
    '.safetensors',
)

ELEVATION_MIN = -90
ELEVATION_MAX = 90.0
AZIMUTH_MIN = -180.0
AZIMUTH_MAX = 180.0

WEIGHT_DTYPE = torch.float16

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

class Preview_3DGS:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_file_path": ("STRING", {"default": '', "multiline": False}),
            },
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "preview_gs"
    CATEGORY = "Comfy3D/Visualize"
    
    def preview_gs(self, gs_file_path):
        
        gs_folder_path, filename = os.path.split(gs_file_path)
        
        if not os.path.isabs(gs_file_path):
            gs_file_path = os.path.join(comfy_paths.output_directory, gs_folder_path)
        
        if not filename.lower().endswith(SUPPORTED_3DGS_EXTENSIONS):
            cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3DGS file extensions: {SUPPORTED_3DGS_EXTENSIONS}").error.print()
            gs_file_path = ""
        
        previews = [
            {
                "filepath": gs_file_path,
            }
        ]
        return {"ui": {"previews": previews}, "result": ()}
    
class Preview_3DMesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
            },
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "preview_mesh"
    CATEGORY = "Comfy3D/Visualize"
    
    def preview_mesh(self, mesh_file_path):
        
        mesh_folder_path, filename = os.path.split(mesh_file_path)
        
        if not os.path.isabs(mesh_file_path):
            mesh_file_path = os.path.join(comfy_paths.output_directory, mesh_folder_path)
        
        if not filename.lower().endswith(SUPPORTED_3D_EXTENSIONS):
            cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3D file extensions: {SUPPORTED_3D_EXTENSIONS}").error.print()
            mesh_file_path = ""
        
        previews = [
            {
                "filepath": mesh_file_path,
            }
        ]
        return {"ui": {"previews": previews}, "result": ()}

class Load_3D_Mesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
                "resize":  ("BOOLEAN", {"default": False},),
                "renormal":  ("BOOLEAN", {"default": True},),
                "retex":  ("BOOLEAN", {"default": False},),
                "optimizable": ("BOOLEAN", {"default": False},),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "load_mesh"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_mesh(self, mesh_file_path, resize, renormal, retex, optimizable):
        mesh = None
        
        if not os.path.isabs(mesh_file_path):
            mesh_file_path = os.path.join(comfy_paths.input_directory, mesh_file_path)
        
        if os.path.exists(mesh_file_path):
            folder, filename = os.path.split(mesh_file_path)
            if filename.lower().endswith(SUPPORTED_3D_EXTENSIONS):
                with torch.inference_mode(not optimizable):
                    mesh = Mesh.load(mesh_file_path, resize, renormal, retex)
            else:
                cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3D file extensions: {SUPPORTED_3D_EXTENSIONS}").error.print()
        else:        
            cstr(f"[{self.__class__.__name__}] File {mesh_file_path} does not exist").error.print()
        return (mesh, )
    
class Load_3DGS:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_file_path": ("STRING", {"default": '', "multiline": False}),
            },
        }

    RETURN_TYPES = (
        "GS_PLY",
    )
    RETURN_NAMES = (
        "gs_ply",
    )
    FUNCTION = "load_gs"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_gs(self, gs_file_path):
        gs_ply = None
        
        if not os.path.isabs(gs_file_path):
            gs_file_path = os.path.join(comfy_paths.input_directory, gs_file_path)
        
        if os.path.exists(gs_file_path):
            folder, filename = os.path.split(gs_file_path)
            if filename.lower().endswith(SUPPORTED_3DGS_EXTENSIONS):
                gs_ply = PlyData.read(gs_file_path)
            else:
                cstr(f"[{self.__class__.__name__}] File name {filename} does not end with supported 3DGS file extensions: {SUPPORTED_3DGS_EXTENSIONS}").error.print()
        else:        
            cstr(f"[{self.__class__.__name__}] File {gs_file_path} does not exist").error.print()
        return (gs_ply, )
    
class Save_3D_Mesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "save_path": ("STRING", {"default": 'Mesh_%Y-%m-%d-%M-%S-%f.glb', "multiline": False}),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "save_path",
    )
    FUNCTION = "save_mesh"
    CATEGORY = "Comfy3D/Import|Export"
    
    def save_mesh(self, mesh, save_path):
        save_path = parse_save_filename(save_path, comfy_paths.output_directory, SUPPORTED_3D_EXTENSIONS, self.__class__.__name__)
        
        if save_path is not None:
            mesh.write(save_path)

        return (save_path, )
    
class Save_3DGS:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_ply": ("GS_PLY",),
                "save_path": ("STRING", {"default": '3DGS_%Y-%m-%d-%M-%S-%f.ply', "multiline": False}),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "save_path",
    )
    FUNCTION = "save_gs"
    CATEGORY = "Comfy3D/Import|Export"
    
    def save_gs(self, gs_ply, save_path):
        
        save_path = parse_save_filename(save_path, comfy_paths.output_directory, SUPPORTED_3DGS_EXTENSIONS, self.__class__.__name__)
        
        if save_path is not None:
            gs_ply.write(save_path)
        
        return (save_path, )

class Image_Add_Pure_Color_Background:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "R": ("INT", {"default": 255, "min": 0, "max": 255}),
                "G": ("INT", {"default": 255, "min": 0, "max": 255}),
                "B": ("INT", {"default": 255, "min": 0, "max": 255}),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "images",
    )
    
    FUNCTION = "image_add_bg"
    CATEGORY = "Comfy3D/Preprocessor"

    def image_add_bg(self, images, masks, R, G, B):
        """
        bg_mask = bg_mask.unsqueeze(3)
        inv_bg_mask = torch.ones_like(bg_mask) - bg_mask
        color = torch.tensor([R, G, B]).to(image.dtype) / 255
        color_bg = color.repeat(bg_mask.shape)
        image = inv_bg_mask * image + bg_mask * color_bg
        """

        image_pils = torch_imgs_to_pils(images, masks)
        image_pils = pils_rgba_to_rgb(image_pils, (R, G, B))

        images = pils_to_torch_imgs(image_pils, images.device)
        return (images,)
    
class Resize_Image_Foreground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "foreground_ratio": ("FLOAT", {"default": 0.85, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "images",
        "masks",
    )
    
    FUNCTION = "resize_img_foreground"
    CATEGORY = "Comfy3D/Preprocessor"

    def resize_img_foreground(self, images, masks, foreground_ratio):
        image_pils = torch_imgs_to_pils(images, masks)
        image_pils = pils_resize_foreground(image_pils, foreground_ratio)
        
        images = pils_to_torch_imgs(image_pils, images.device, force_rgb=False)
        images, masks = images[:, :, :, 0:-1], images[:, :, :, -1]
        return (images, masks,)
    
class Make_Image_Grid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "grid_side_num": ("INT", {"default": 1, "min": 1, "max": 8192}),
                "use_rows": ("BOOLEAN", {"default": True},),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "image_grid",
    )
    
    FUNCTION = "make_image_grid"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def make_image_grid(self, images, grid_side_num, use_rows):
        pil_image_list = torch_imgs_to_pils(images)

        if use_rows:
            rows = grid_side_num
            clos = None
        else:
            clos = grid_side_num
            rows = None

        image_grid = pil_make_image_grid(pil_image_list, rows, clos)

        image_grid = TF.to_tensor(image_grid).permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 3]

        return (image_grid,)

class Split_Image_Grid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grid_side_num": ("INT", {"default": 1, "min": 1, "max": 8192}),
                "use_rows": ("BOOLEAN", {"default": True},),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "images",
    )
    
    FUNCTION = "split_image_grid"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def split_image_grid(self, image, grid_side_num, use_rows):
        images = []
        for image_pil in torch_imgs_to_pils(image):

            if use_rows:
                rows = grid_side_num
                clos = None
            else:
                clos = grid_side_num
                rows = None

            image_pils = pil_split_image(image_pil, rows, clos)

            images.append(pils_to_torch_imgs(image_pils, image.device))
            
        images = torch.cat(images, dim=0)
        return (images,)

class Get_Masks_From_Normal_Maps:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_maps": ("IMAGE",),
            },
        }
        
    RETURN_TYPES = (
        "MASK",
    )
    RETURN_NAMES = (
        "normal_masks",
    )
    
    FUNCTION = "make_image_grid"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def make_image_grid(self, normal_maps):
        from Unique3D.scripts.utils import get_normal_map_masks
        pil_normal_list = torch_imgs_to_pils(normal_maps)
        normal_masks = get_normal_map_masks(pil_normal_list)
        normal_masks = torch.stack(normal_masks, dim=0).to(normal_maps.dtype).to(normal_maps.device)
        return (normal_masks,)

class Rotate_Normal_Maps_Horizontally:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_maps": ("IMAGE",),
                "normal_masks": ("MASK",),
                "clockwise": ("BOOLEAN", {"default": True},),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "normal_maps",
    )
    
    FUNCTION = "make_image_grid"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def make_image_grid(self, normal_maps, normal_masks, clockwise):
        if normal_maps.shape[0] > 1:
            from Unique3D.scripts.utils import rotate_normals_torch
            pil_image_list = torch_imgs_to_pils(normal_maps, normal_masks)
            pil_image_list = rotate_normals_torch(pil_image_list, return_types='pil', rotate_direction=int(clockwise))
            normal_maps = pils_to_torch_imgs(pil_image_list, normal_maps.device)
        return (normal_maps,)
    
class Fast_Clean_Mesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "apply_smooth": ("BOOLEAN", {"default": True},),
                "smooth_step": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "apply_sub_divide": ("BOOLEAN", {"default": True},),
                "sub_divide_threshold": ("FLOAT", {"default": 0.25, "step": 0.001}),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "clean_mesh"
    CATEGORY = "Comfy3D/Preprocessor"

    def clean_mesh(self, mesh, apply_smooth, smooth_step, apply_sub_divide, sub_divide_threshold):

        meshes = simple_clean_mesh(to_pyml_mesh(mesh.v, mesh.f), apply_smooth=apply_smooth, stepsmoothnum=smooth_step, apply_sub_divide=apply_sub_divide, sub_divide_threshold=sub_divide_threshold).to(DEVICE)
        vertices, faces, _ = from_py3d_mesh(meshes)

        mesh = Mesh(v=vertices, f=faces, device=DEVICE)

        return (mesh,)

class Switch_3DGS_Axis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_ply": ("GS_PLY",),
                "axis_x_to": (["+x", "-x", "+y", "-y", "+z", "-z"],),
                "axis_y_to": (["+y", "-y", "+z", "-z", "+x", "-x"],),
                "axis_z_to": (["+z", "-z", "+x", "-x", "+y", "-y"],),
            },
        }

    RETURN_TYPES = (
        "GS_PLY",
    )
    RETURN_NAMES = (
        "switched_gs_ply",
    )
    FUNCTION = "switch_axis_and_scale"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def switch_axis_and_scale(self, gs_ply, axis_x_to, axis_y_to, axis_z_to):
        switched_gs_ply = None
        if axis_x_to[1] != axis_y_to[1] and axis_x_to[1] != axis_z_to[1] and axis_y_to[1] != axis_z_to[1]:
            target_axis, target_scale, coordinate_invert_count = get_target_axis_and_scale([axis_x_to, axis_y_to, axis_z_to])
            switched_gs_ply = switch_ply_axis_and_scale(gs_ply, target_axis, target_scale, coordinate_invert_count)
        else:
            cstr(f"[{self.__class__.__name__}] axis_x_to: {axis_x_to}, axis_y_to: {axis_y_to}, axis_z_to: {axis_z_to} have to be on separated axis").error.print()
        
        return (switched_gs_ply, )
    
class Switch_Mesh_Axis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "axis_x_to": (["+x", "-x", "+y", "-y", "+z", "-z"],),
                "axis_y_to": (["+y", "-y", "+z", "-z", "+x", "-x"],),
                "axis_z_to": (["+z", "-z", "+x", "-x", "+y", "-y"],),
                "flip_normal": ("BOOLEAN", {"default": False},),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100, "step": 0.01}),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "switched_mesh",
    )
    FUNCTION = "switch_axis_and_scale"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def switch_axis_and_scale(self, mesh, axis_x_to, axis_y_to, axis_z_to, flip_normal, scale):
        
        switched_mesh = None
        
        if axis_x_to[1] != axis_y_to[1] and axis_x_to[1] != axis_z_to[1] and axis_y_to[1] != axis_z_to[1]:
            target_axis, target_scale, coordinate_invert_count = get_target_axis_and_scale([axis_x_to, axis_y_to, axis_z_to], scale)
            switched_mesh = switch_mesh_axis_and_scale(mesh, target_axis, target_scale, flip_normal)
        else:
            cstr(f"[{self.__class__.__name__}] axis_x_to: {axis_x_to}, axis_y_to: {axis_y_to}, axis_z_to: {axis_z_to} have to be on separated axis").error.print()
        
        return (switched_mesh, )
    
class Convert_3DGS_To_Pointcloud:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_ply": ("GS_PLY",),
            },
        }

    RETURN_TYPES = (
        "POINTCLOUD",
    )
    RETURN_NAMES = (
        "points_cloud",
    )
    FUNCTION = "convert_gs_ply"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def convert_gs_ply(self, gs_ply):
        
        points_cloud = ply_to_points_cloud(gs_ply)
        
        return (points_cloud, )
    
class Convert_Mesh_To_Pointcloud:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
            },
        }

    RETURN_TYPES = (
        "POINTCLOUD",
    )
    RETURN_NAMES = (
        "points_cloud",
    )
    FUNCTION = "convert_mesh"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def convert_mesh(self, mesh):
        
        points_cloud = mesh.convert_to_pointcloud()
        
        return (points_cloud, )
    
class Stack_Orbit_Camera_Poses:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "orbit_radius_start": ("FLOAT", {"default": 1.75, "step": 0.0001}),
                "orbit_radius_stop": ("FLOAT", {"default": 1.75, "step": 0.0001}),
                "orbit_radius_step": ("FLOAT", {"default": 0.1, "step": 0.0001}),
                "elevation_start": ("FLOAT", {"default": 0.0, "min": ELEVATION_MIN, "max": ELEVATION_MAX, "step": 0.0001}),
                "elevation_stop": ("FLOAT", {"default": 0.0, "min": ELEVATION_MIN, "max": ELEVATION_MAX, "step": 0.0001}),
                "elevation_step": ("FLOAT", {"default": 0.0, "min": ELEVATION_MIN, "max": ELEVATION_MAX, "step": 0.0001}),
                "azimuth_start": ("FLOAT", {"default": 0.0, "min": AZIMUTH_MIN, "max": AZIMUTH_MAX, "step": 0.0001}),
                "azimuth_stop": ("FLOAT", {"default": 0.0, "min": AZIMUTH_MIN, "max": AZIMUTH_MAX, "step": 0.0001}),
                "azimuth_step": ("FLOAT", {"default": 0.0, "min": AZIMUTH_MIN, "max": AZIMUTH_MAX, "step": 0.0001}),
                "orbit_center_X_start": ("FLOAT", {"default": 0.0, "step": 0.0001}),
                "orbit_center_X_stop": ("FLOAT", {"default": 0.0, "step": 0.0001}),
                "orbit_center_X_step": ("FLOAT", {"default": 0.1, "step": 0.0001}),
                "orbit_center_Y_start": ("FLOAT", {"default": 0.0, "step": 0.0001}),
                "orbit_center_Y_stop": ("FLOAT", {"default": 0.0, "step": 0.0001}),
                "orbit_center_Y_step": ("FLOAT", {"default": 0.1, "step": 0.0001}),
                "orbit_center_Z_start": ("FLOAT", {"default": 0.0, "step": 0.0001}),
                "orbit_center_Z_stop": ("FLOAT", {"default": 0.0, "step": 0.0001}),
                "orbit_center_Z_step": ("FLOAT", {"default": 0.1, "step": 0.0001}),
            },
        }

    RETURN_TYPES = (
        "ORBIT_CAMPOSES",   # [orbit radius, elevation, azimuth, orbit center X, orbit center Y, orbit center Z]
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
    )
    RETURN_NAMES = (
        "orbit_camposes",  # List of 6 lists
        "orbit_radius_list",
        "elevation_list", 
        "azimuth_list", 
        "orbit_center_X_list",  
        "orbit_center_Y_list",  
        "orbit_center_Z_list",
    )
    OUTPUT_IS_LIST = (
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    )
    
    FUNCTION = "get_camposes"
    CATEGORY = "Comfy3D/Preprocessor"
    
    class Pose_Config(Enum):
        STOP_LARGER_STEP_POS = 0
        START_LARGER_STEP_POS = 1
        START_LARGER_STEP_NEG = 2
        STOP_LARGER_STEP_NEG = 3
    
    class Pose_Type:
        def __init__(self, start, stop, step, min_value=-math.inf, max_value=math.inf, is_linear = True):
            if abs(step) < 0.0001:
                step = 0.0001 * (-1.0 if step < 0 else 1.0)
            
            if is_linear and ( (step > 0 and stop < start) or (step < 0 and stop > start)):
                cstr(f"[{self.__class__.__name__}] stop value: {stop} cannot be reached from start value {start} with step value {step}, will reverse the sign of step value to {-step}").warning.print()
                self.step = -step
            else:
                self.step = step
                      
            self.start = start
            self.stop = stop
            
            self.min = min_value
            self.max = max_value
            
            self.is_linear = is_linear  # linear or circular (i.e. min and max value are connected, e.g. -180 & 180 degree in azimuth angle) value
                
    
    def stack_camposes(self, pose_type_index=None, last_camposes=[[]]):
        if pose_type_index == None:
            pose_type_index = len(self.all_pose_types) - 1
            
        if pose_type_index == -1:
            return last_camposes
        else:
            current_pose_type = self.all_pose_types[pose_type_index]
            
            all_camposes = []
            
            # There are four different kind of situation we need to deal with to make this function generalize for any combination of inputs
            if current_pose_type.step > 0:
                if current_pose_type.start < current_pose_type.stop or current_pose_type.is_linear:
                    pose_config = Stack_Orbit_Camera_Poses.Pose_Config.STOP_LARGER_STEP_POS
                else:
                    pose_config = Stack_Orbit_Camera_Poses.Pose_Config.START_LARGER_STEP_POS
            else:
                if current_pose_type.start > current_pose_type.stop or current_pose_type.is_linear:
                    pose_config = Stack_Orbit_Camera_Poses.Pose_Config.START_LARGER_STEP_NEG
                else:
                    pose_config = Stack_Orbit_Camera_Poses.Pose_Config.STOP_LARGER_STEP_NEG
                    
            p = current_pose_type.start
            p_passed_min_max_seam = False
            
            while ( (pose_config == Stack_Orbit_Camera_Poses.Pose_Config.STOP_LARGER_STEP_POS and p <= current_pose_type.stop) or 
                    (pose_config == Stack_Orbit_Camera_Poses.Pose_Config.START_LARGER_STEP_POS and (not p_passed_min_max_seam or p <= current_pose_type.stop)) or
                    (pose_config == Stack_Orbit_Camera_Poses.Pose_Config.START_LARGER_STEP_NEG and p >= current_pose_type.stop) or 
                    (pose_config == Stack_Orbit_Camera_Poses.Pose_Config.STOP_LARGER_STEP_NEG and (not p_passed_min_max_seam or p >= current_pose_type.stop)) ):
                
                # If current pose value surpass the either min/max value then we map its vaule to the oppsite sign
                if pose_config == Stack_Orbit_Camera_Poses.Pose_Config.START_LARGER_STEP_POS and p > current_pose_type.max:
                    p = current_pose_type.min + p % current_pose_type.max
                    p_passed_min_max_seam = True    
                elif pose_config == Stack_Orbit_Camera_Poses.Pose_Config.STOP_LARGER_STEP_NEG and p < current_pose_type.min:
                    p = current_pose_type.max + p % current_pose_type.min
                    p_passed_min_max_seam = True
                    
                new_camposes = copy.deepcopy(last_camposes)
                    
                for campose in new_camposes:
                    campose.insert(0, p)
                    
                all_camposes.extend(new_camposes)
                
                p += current_pose_type.step
                    
            return self.stack_camposes(pose_type_index-1, all_camposes)
    
    def get_camposes(self, 
                     orbit_radius_start, 
                     orbit_radius_stop, 
                     orbit_radius_step, 
                     elevation_start, 
                     elevation_stop, 
                     elevation_step, 
                     azimuth_start, 
                     azimuth_stop, 
                     azimuth_step, 
                     orbit_center_X_start, 
                     orbit_center_X_stop, 
                     orbit_center_X_step, 
                     orbit_center_Y_start, 
                     orbit_center_Y_stop, 
                     orbit_center_Y_step, 
                     orbit_center_Z_start, 
                     orbit_center_Z_stop, 
                     orbit_center_Z_step):
        
        """
            Return the combination of all the pose types interpolation values
            Return values in two ways:
            orbit_camposes: CAMPOSES type list can directly input to other 3D process node (e.g. GaussianSplatting)
            all the camera pose types seperated in different list, becasue some 3D model's conditioner only takes a sub set of all camera pose types (e.g. StableZero123)
        """
        
        orbit_radius_list = []
        elevation_list = []
        azimuth_list = []
        orbit_center_X_list = []
        orbit_center_Y_list = []
        orbit_center_Z_list = []
        
        self.all_pose_types = []
        self.all_pose_types.append( Stack_Orbit_Camera_Poses.Pose_Type(orbit_radius_start, orbit_radius_stop, orbit_radius_step) )
        self.all_pose_types.append( Stack_Orbit_Camera_Poses.Pose_Type(elevation_start, elevation_stop, elevation_step, ELEVATION_MIN, ELEVATION_MAX) )
        self.all_pose_types.append( Stack_Orbit_Camera_Poses.Pose_Type(azimuth_start, azimuth_stop, azimuth_step, AZIMUTH_MIN, AZIMUTH_MAX, False) )
        self.all_pose_types.append( Stack_Orbit_Camera_Poses.Pose_Type(orbit_center_X_start, orbit_center_X_stop, orbit_center_X_step) )
        self.all_pose_types.append( Stack_Orbit_Camera_Poses.Pose_Type(orbit_center_Y_start, orbit_center_Y_stop, orbit_center_Y_step) )
        self.all_pose_types.append( Stack_Orbit_Camera_Poses.Pose_Type(orbit_center_Z_start, orbit_center_Z_stop, orbit_center_Z_step) )
        
        orbit_camposes = self.stack_camposes()
        
        for campose in orbit_camposes:
            orbit_radius_list.append(campose[0])
            elevation_list.append(campose[1])
            azimuth_list.append(campose[2])
            orbit_center_X_list.append(campose[3])
            orbit_center_Y_list.append(campose[4])
            orbit_center_Z_list.append(campose[5])
        
        return (orbit_camposes, orbit_radius_list, elevation_list, azimuth_list, orbit_center_X_list, orbit_center_Y_list, orbit_center_Z_list, )

class Get_Camposes_From_List_Indexed:
    
    RETURN_TYPES = ("ORBIT_CAMPOSES",)
    FUNCTION = "get_indexed_camposes"
    CATEGORY = "Comfy3D/Preprocessor"
    DESCRIPTION = """
        Selects and returns the camera poses at the specified indices as an list.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True}),
            },
        }
    
    def get_indexed_camposes(self, original_orbit_camera_poses, indexes):
        
        # Parse the indexes string into a list of integers
        index_list = [int(index.strip()) for index in indexes.split(',')]
        
        # Select the camposes at the specified indices
        orbit_camera_poses = []
        for pose_list in original_orbit_camera_poses:
            new_pose_list = [pose_list[i] for i in index_list]
            orbit_camera_poses.append(new_pose_list)

        return (orbit_camera_poses,)

class Mesh_Orbit_Renderer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "render_image_width": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "render_image_height": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "render_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "render_orbit_camera_fovy": ("FLOAT", {"default": 49.1, "min": 0.0, "max": 180.0, "step": 0.1}),
                "render_background_color_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "render_background_color_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "render_background_color_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "force_cuda_rasterize": ("BOOLEAN", {"default": False},),
            },
            
            "optional": {
                "render_depth": ("BOOLEAN", {"default": False},),
                "render_normal": ("BOOLEAN", {"default": False},),
            }
        }
        
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "IMAGE",
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "rendered_mesh_images",   # [Number of Poses, H, W, 3]
        "rendered_mesh_masks",    # [Number of Poses, H, W, 1]
        "all_rendered_depths",    # [Number of Poses, H, W, 3]
        "all_rendered_normals",   # [Number of Poses, H, W, 3]
        "all_rendered_viewcos",   # [Number of Poses, H, W, 3]
    )
    
    FUNCTION = "render_mesh"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def render_mesh(
        self, 
        mesh, 
        render_image_width, 
        render_image_height, 
        render_orbit_camera_poses, 
        render_orbit_camera_fovy,
        render_background_color_r, 
        render_background_color_g, 
        render_background_color_b,
        force_cuda_rasterize,
        render_depth=False,
        render_normal=False,
    ):
        
        renderer = DiffRastRenderer(mesh, force_cuda_rasterize)
        
        optional_render_types = []
        if render_depth:
            optional_render_types.append('depth')
        if render_normal:
            optional_render_types.append('normal')
        
        cam_controller = DiffMeshCameraController(
            renderer, 
            render_image_width, 
            render_image_height, 
            render_orbit_camera_fovy, 
            static_bg=[render_background_color_r, render_background_color_g, render_background_color_b]
        )
        
        extra_kwargs = {"optional_render_types": optional_render_types}
        all_rendered_images, all_rendered_masks, extra_outputs = cam_controller.render_all_pose(render_orbit_camera_poses, **extra_kwargs)
        all_rendered_masks = all_rendered_masks.squeeze(-1)  # [N, H, W, 1] -> [N, H, W]
        if 'depth' in extra_outputs:
            all_rendered_depths = extra_outputs['depth'].repeat(1, 1, 1, 3)   # [N, H, W, 1] -> [N, H, W, 3]
        else:
            all_rendered_depths = None
        
        if 'normal' in extra_outputs:
            all_rendered_normals = extra_outputs['normal']
            all_rendered_viewcos = extra_outputs['viewcos'].repeat(1, 1, 1, 3)
        else:
            all_rendered_normals = None
            all_rendered_viewcos = None
        
        return (all_rendered_images, all_rendered_masks, all_rendered_depths, all_rendered_normals, all_rendered_viewcos)
        
   
class Gaussian_Splatting_Orbit_Renderer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_ply": ("GS_PLY",),
                "render_image_width": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "render_image_height": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "render_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "render_orbit_camera_fovy": ("FLOAT", {"default": 49.1, "min": 0.0, "max": 180.0, "step": 0.1}),
                "render_background_color_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "render_background_color_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "render_background_color_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }
        
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "IMAGE",
    )
    RETURN_NAMES = (
        "rendered_gs_images",    # [Number of Poses, H, W, 3]
        "rendered_gs_masks",     # [Number of Poses, H, W, 1]
        "rendered_gs_depths",   # [Number of Poses, H, W, 3]
    )
    
    FUNCTION = "render_gs"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def render_gs(
        self, 
        gs_ply, 
        render_image_width, 
        render_image_height, 
        render_orbit_camera_poses, 
        render_orbit_camera_fovy,
        render_background_color_r, 
        render_background_color_g, 
        render_background_color_b,
    ):
        
        sh_degree, _ = calculate_max_sh_degree_from_gs_ply(gs_ply)
        renderer = GaussianSplattingRenderer(sh_degree=sh_degree)
        renderer.initialize(gs_ply)
        
        cam_controller = GaussianSplattingCameraController(
            renderer, 
            render_image_width, 
            render_image_height, 
            render_orbit_camera_fovy, 
            static_bg=[render_background_color_r, render_background_color_g, render_background_color_b]
        )
        
        all_rendered_images, all_rendered_masks, extra_outputs = cam_controller.render_all_pose(render_orbit_camera_poses)
        all_rendered_images = all_rendered_images.permute(0, 2, 3, 1)   # [N, 3, H, W] -> [N, H, W, 3]
        all_rendered_masks = all_rendered_masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
        
        if 'depth' in extra_outputs:
            all_rendered_depths = extra_outputs['depth'].permute(0, 2, 3, 1).repeat(1, 1, 1, 3)   # [N, 1, H, W] -> [N, H, W, 3]
        else:
            all_rendered_depths = None
        
        return (all_rendered_images, all_rendered_masks, all_rendered_depths)
    
class Gaussian_Splatting_3D:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE",), 
                "reference_masks": ("MASK",),
                "reference_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "reference_orbit_camera_fovy": ("FLOAT", {"default": 49.1, "min": 0.0, "max": 180.0, "step": 0.1}),
                "training_iterations": ("INT", {"default": 30_000, "min": 1, "max": 0xffffffffffffffff}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
                "ms_ssim_loss_weight": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, }),
                "alpha_loss_weight": ("FLOAT", {"default": 3, "min": 0.0, }),
                "offset_loss_weight": ("FLOAT", {"default": 0.0, "min": 0.0, }),
                "offset_opacity_loss_weight": ("FLOAT", {"default": 0.0, "min": 0.0, }),
                "invert_background_probability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "feature_learning_rate": ("FLOAT", {"default": 0.0025, "min": 0.000001, "step": 0.000001}),
                "opacity_learning_rate": ("FLOAT", {"default": 0.05,  "min": 0.000001, "step": 0.000001}),
                "scaling_learning_rate": ("FLOAT", {"default": 0.005,  "min": 0.000001, "step": 0.000001}),
                "rotation_learning_rate": ("FLOAT", {"default": 0.001,  "min": 0.000001, "step": 0.000001}),
                "position_learning_rate_init": ("FLOAT", {"default": 0.00016,  "min": 0.000001, "step": 0.000001}),
                "position_learning_rate_final": ("FLOAT", {"default": 0.0000016, "min": 0.0000001, "step": 0.0000001}),
                "position_learning_rate_delay_mult": ("FLOAT", {"default": 0.01, "min": 0.000001, "step": 0.000001}),
                "position_learning_rate_max_steps": ("INT", {"default": 30_000, "min": 1, "max": 0xffffffffffffffff}),
                "initial_gaussians_num": ("INT", {"default": 10_000, "min": 1, "max": 0xffffffffffffffff}),
                "K_nearest_neighbors": ("INT", {"default": 3, "min": 1, "max": 0xffffffffffffffff}),
                "percent_dense": ("FLOAT", {"default": 0.01,  "min": 0.00001, "step": 0.00001}),
                "density_start_iterations": ("INT", {"default": 500, "min": 0, "max": 0xffffffffffffffff}),
                "density_end_iterations": ("INT", {"default": 15_000, "min": 0, "max": 0xffffffffffffffff}),
                "densification_interval": ("INT", {"default": 100, "min": 1, "max": 0xffffffffffffffff}),
                "opacity_reset_interval": ("INT", {"default": 3000, "min": 1, "max": 0xffffffffffffffff}),
                "densify_grad_threshold": ("FLOAT", {"default": 0.0002, "min": 0.00001, "step": 0.00001}),
                "gaussian_sh_degree": ("INT", {"default": 3, "min": 0}),
            },
            
            "optional": {
                "points_cloud_to_initialize_gaussian": ("POINTCLOUD",),
                "ply_to_initialize_gaussian": ("GS_PLY",),
                "mesh_to_initialize_gaussian": ("MESH",),
            }
        }

    RETURN_TYPES = (
        "GS_PLY",
    )
    RETURN_NAMES = (
        "gs_ply",
    )
    FUNCTION = "run_gs"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_gs(
        self,
        reference_images,
        reference_masks,
        reference_orbit_camera_poses,
        reference_orbit_camera_fovy,
        training_iterations,
        batch_size,
        ms_ssim_loss_weight,
        alpha_loss_weight,
        offset_loss_weight,
        offset_opacity_loss_weight,
        invert_background_probability,
        feature_learning_rate,
        opacity_learning_rate,
        scaling_learning_rate,
        rotation_learning_rate,
        position_learning_rate_init,
        position_learning_rate_final,
        position_learning_rate_delay_mult,
        position_learning_rate_max_steps,
        initial_gaussians_num,
        K_nearest_neighbors,
        percent_dense,
        density_start_iterations,
        density_end_iterations,
        densification_interval,
        opacity_reset_interval,
        densify_grad_threshold,
        gaussian_sh_degree,
        points_cloud_to_initialize_gaussian=None,
        ply_to_initialize_gaussian=None,
        mesh_to_initialize_gaussian=None,
    ):
        
        gs_ply = None
        
        ref_imgs_num = len(reference_images)
        ref_masks_num = len(reference_masks)
        if ref_imgs_num == ref_masks_num:
            
            ref_cam_poses_num = len(reference_orbit_camera_poses)
            if ref_imgs_num == ref_cam_poses_num:
                
                if batch_size > ref_imgs_num:
                    cstr(f"[{self.__class__.__name__}] Batch size {batch_size} is bigger than number of reference images {ref_imgs_num}! Set batch size to {ref_imgs_num} instead").warning.print()
                    batch_size = ref_imgs_num
                
                with torch.inference_mode(False):
                
                    gs_params = GSParams(
                        training_iterations,
                        batch_size,
                        ms_ssim_loss_weight,
                        alpha_loss_weight,
                        offset_loss_weight,
                        offset_opacity_loss_weight,
                        invert_background_probability,
                        feature_learning_rate,
                        opacity_learning_rate,
                        scaling_learning_rate,
                        rotation_learning_rate,
                        position_learning_rate_init,
                        position_learning_rate_final,
                        position_learning_rate_delay_mult,
                        position_learning_rate_max_steps,
                        initial_gaussians_num,
                        K_nearest_neighbors,
                        percent_dense,
                        density_start_iterations,
                        density_end_iterations,
                        densification_interval,
                        opacity_reset_interval,
                        densify_grad_threshold,
                        gaussian_sh_degree
                    )
                    
                    
                    if points_cloud_to_initialize_gaussian is not None:
                        gs_init_input = points_cloud_to_initialize_gaussian
                    elif ply_to_initialize_gaussian is not None:
                        gs_init_input = ply_to_initialize_gaussian
                    else:
                        gs_init_input = mesh_to_initialize_gaussian
                    
                    gs = GaussianSplatting3D(gs_params, gs_init_input)
                    gs.prepare_training(reference_images, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy)
                    gs.training()

                    gs_ply = gs.renderer.gaussians.to_ply()
                
            else:
                cstr(f"[{self.__class__.__name__}] Number of reference images {ref_imgs_num} does not equal to number of reference camera poses {ref_cam_poses_num}").error.print()
        else:
            cstr(f"[{self.__class__.__name__}] Number of reference images {ref_imgs_num} does not equal to number of masks {ref_masks_num}").error.print()

        return (gs_ply, )
    
class Fitting_Mesh_With_Multiview_Images:
    
    def __init__(self):
        self.need_update = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE",), 
                "reference_masks": ("MASK",),
                "reference_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "reference_orbit_camera_fovy": ("FLOAT", {"default": 49.1, "min": 0.0, "max": 180.0, "step": 0.1}),
                "mesh": ("MESH",),
                "mesh_albedo_width": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "mesh_albedo_height": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "training_iterations": ("INT", {"default": 1024, "min": 1, "max": 100000}),
                "batch_size": ("INT", {"default": 3, "min": 1, "max": 0xffffffffffffffff}),
                "texture_learning_rate": ("FLOAT", {"default": 0.001, "min": 0.00001, "step": 0.00001}),
                "train_mesh_geometry": ("BOOLEAN", {"default": False},),
                "geometry_learning_rate": ("FLOAT", {"default": 0.0001, "min": 0.00001, "step": 0.00001}),
                "ms_ssim_loss_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "remesh_after_n_iteration": ("INT", {"default": 512, "min": 128, "max": 100000}),
                "invert_background_probability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "force_cuda_rasterize": ("BOOLEAN", {"default": False},),
            },
        }

    RETURN_TYPES = (
        "MESH",
        "IMAGE",
    )
    RETURN_NAMES = (
        "trained_mesh",
        "baked_texture",    # [1, H, W, 3]
    )
    FUNCTION = "fitting_mesh"
    CATEGORY = "Comfy3D/Algorithm"
    
    def fitting_mesh(
        self, 
        reference_images, 
        reference_masks, 
        reference_orbit_camera_poses, 
        reference_orbit_camera_fovy, 
        mesh, 
        mesh_albedo_width,
        mesh_albedo_height,
        training_iterations, 
        batch_size, 
        texture_learning_rate, 
        train_mesh_geometry, 
        geometry_learning_rate, 
        ms_ssim_loss_weight, 
        remesh_after_n_iteration,
        invert_background_probability,
        force_cuda_rasterize,
    ):
        
        mesh.set_new_albedo(mesh_albedo_width, mesh_albedo_height)
        
        trained_mesh = None
        baked_texture = None
        
        ref_imgs_num = len(reference_images)
        ref_masks_num = len(reference_masks)
        if ref_imgs_num == ref_masks_num:
            
            ref_cam_poses_num = len(reference_orbit_camera_poses)
            if ref_imgs_num == ref_cam_poses_num:
                
                if batch_size > ref_imgs_num:
                    cstr(f"[{self.__class__.__name__}] Batch size {batch_size} is bigger than number of reference images {ref_imgs_num}! Set batch size to {ref_imgs_num} instead").warning.print()
                    batch_size = ref_imgs_num
                    
                with torch.inference_mode(False):
                
                    mesh_fitter = DiffMesh(
                        mesh, 
                        training_iterations, 
                        batch_size, 
                        texture_learning_rate, 
                        train_mesh_geometry, 
                        geometry_learning_rate, 
                        ms_ssim_loss_weight, 
                        remesh_after_n_iteration, 
                        invert_background_probability, 
                        force_cuda_rasterize
                    )
                    
                    mesh_fitter.prepare_training(reference_images, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy)
                    mesh_fitter.training()
                    
                    trained_mesh, baked_texture = mesh_fitter.get_mesh_and_texture()
                    
            else:
                cstr(f"[{self.__class__.__name__}] Number of reference images {ref_imgs_num} does not equal to number of reference camera poses {ref_cam_poses_num}").error.print()
        else:
            cstr(f"[{self.__class__.__name__}] Number of reference images {ref_imgs_num} does not equal to number of masks {ref_masks_num}").error.print()
        
        return (trained_mesh, baked_texture, )

class Load_Triplane_Gaussian_Transformers:
    
    checkpoints_dir = "TriplaneGaussian"
    default_ckpt_name = "model_lvis_rel.ckpt"
    default_repo_id = "VAST-AI/TriplaneGaussian"
    config_path = "TriplaneGaussian_config.yaml"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)
        cls.config_path_abs = os.path.join(CONFIG_ROOT_PATH, cls.config_path)
        all_models_names = get_list_filenames(cls.checkpoints_dir_abs, SUPPORTED_CHECKPOINTS_EXTENSIONS)
        if cls.default_ckpt_name not in all_models_names:
            all_models_names += [cls.default_ckpt_name]
        
        return {
            "required": {
                "model_name": (all_models_names, ),
            },
        }
    
    RETURN_TYPES = (
        "TGS_MODEL",
    )
    RETURN_NAMES = (
        "tgs_model",
    )
    FUNCTION = "load_TGS"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_TGS(self, model_name):

        device = get_device()

        cfg: ExperimentConfigTGS = load_config_tgs(self.config_path_abs)

        ckpt_path = resume_or_download_model_from_hf(self.checkpoints_dir_abs, self.default_repo_id, model_name, self.__class__.__name__)
            
        cfg.system.weights=ckpt_path
        tgs_model = TGS(cfg=cfg.system).to(device)
        
        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {ckpt_path}").msg.print()

        return (tgs_model, )
    
class Triplane_Gaussian_Transformers:
    
    config_path = "TriplaneGaussian_config.yaml"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.config_path_abs = os.path.join(CONFIG_ROOT_PATH, cls.config_path)
        return {
            "required": {
                "reference_image": ("IMAGE", ),
                "reference_mask": ("MASK",),
                "tgs_model": ("TGS_MODEL", ),
                "cam_dist": ("FLOAT", {"default": 1.9, "min": 0.01, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = (
        "GS_PLY",
    )
    RETURN_NAMES = (
        "gs_ply",
    )
    FUNCTION = "run_TGS"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_TGS(self, reference_image, reference_mask, tgs_model, cam_dist):        
        cfg: ExperimentConfigTGS = load_config_tgs(self.config_path_abs)

        cfg.data.cond_camera_distance = cam_dist
        cfg.data.eval_camera_distance = cam_dist
        dataset = CustomImageOrbitDataset(reference_image, reference_mask, cfg.data)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.data.eval_batch_size, 
            shuffle=False,
            collate_fn=dataset.collate
        )

        gs_ply = []
        for batch in dataloader:
            batch = todevice(batch)
            gs_ply.extend(tgs_model(batch))
        
        return (gs_ply[0], )
    
class Load_Diffusers_Pipeline:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusers_pipeline_name": (list(DIFFUSERS_PIPE_DICT.keys()),),
                "repo_id": ("STRING", {"default": "ashawkey/imagedream-ipmv-diffusers", "multiline": False}),
                "custom_pipeline": ("STRING", {"default": "", "multiline": False}),
                "force_download": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "checkpoint_sub_dir": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = (
        "DIFFUSERS_PIPE",
    )
    RETURN_NAMES = (
        "pipe",
    )
    FUNCTION = "load_diffusers_pipe"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_diffusers_pipe(self, diffusers_pipeline_name, repo_id, custom_pipeline, force_download, checkpoint_sub_dir=""):
        
        # resume download pretrained checkpoint
        ckpt_download_dir = os.path.join(CKPT_DIFFUSERS_PATH, repo_id)
        snapshot_download(repo_id=repo_id, local_dir=ckpt_download_dir, force_download=force_download, repo_type="model", ignore_patterns=["*.json", "*.py"])
        
        diffusers_pipeline_class = DIFFUSERS_PIPE_DICT[diffusers_pipeline_name]
        
        # load diffusers pipeline
        if not custom_pipeline:
            custom_pipeline = None
            
        ckpt_path = ckpt_download_dir if not checkpoint_sub_dir else os.path.join(ckpt_download_dir, checkpoint_sub_dir)
        pipe = diffusers_pipeline_class.from_pretrained(
            ckpt_path,
            torch_dtype=WEIGHT_DTYPE,
            custom_pipeline=custom_pipeline,
        ).to(DEVICE)
        
        pipe.enable_xformers_memory_efficient_attention()
        
        return (pipe, )
    
class Set_Diffusers_Pipeline_Scheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("DIFFUSERS_PIPE",),
                "diffusers_scheduler_name": (list(DIFFUSERS_SCHEDULER_DICT.keys()),),
            },
        }

    RETURN_TYPES = (
        "DIFFUSERS_PIPE",
    )
    RETURN_NAMES = (
        "pipe",
    )
    FUNCTION = "set_pipe_scheduler"
    CATEGORY = "Comfy3D/Import|Export"

    def set_pipe_scheduler(self, pipe, diffusers_scheduler_name):

        diffusers_scheduler_class = DIFFUSERS_SCHEDULER_DICT[diffusers_scheduler_name]

        pipe.scheduler = diffusers_scheduler_class.from_config(
            pipe.scheduler.config, timestep_spacing='trailing'
        )
        return (pipe, )

class Set_Diffusers_Pipeline_State_Dict:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("DIFFUSERS_PIPE",),
                "repo_id": ("STRING", {"default": "TencentARC/InstantMesh", "multiline": False}),
                "model_name": ("STRING", {"default": "diffusion_pytorch_model.bin", "multiline": False}),
            },
        }

    RETURN_TYPES = (
        "DIFFUSERS_PIPE",
    )
    RETURN_NAMES = (
        "pipe",
    )
    FUNCTION = "set_pipe_state_dict"
    CATEGORY = "Comfy3D/Import|Export"

    def set_pipe_state_dict(self, pipe, repo_id, model_name):

        checkpoints_dir_abs = os.path.join(CKPT_DIFFUSERS_PATH, repo_id)
        ckpt_path = resume_or_download_model_from_hf(checkpoints_dir_abs, repo_id, model_name, self.__class__.__name__)

        state_dict = torch.load(ckpt_path, map_location='cpu')
        pipe.unet.load_state_dict(state_dict, strict=True)
        pipe.enable_xformers_memory_efficient_attention()
        pipe = pipe.to(DEVICE)

        return (pipe, )

class Wonder3D_MVDiffusion_Model:
    
    config_path = "Wonder3D_config.yaml"
    fix_cam_pose_dir = "Wonder3D/data/fixed_poses/nine_views"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.config_path_abs = os.path.join(CONFIG_ROOT_PATH, cls.config_path)
        cls.fix_cam_pose_dir_abs = os.path.join(MODULE_ROOT_PATH, cls.fix_cam_pose_dir)
        return {
            "required": {
                "mvdiffusion_pipe": ("DIFFUSERS_PIPE",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mv_guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1}),
            },
        }
        
    RETURN_TYPES = (
        "IMAGE",
        "IMAGE", 
    )
    RETURN_NAMES = (
        "multiview_images",
        "multiview_normals",
    )
    FUNCTION = "run_mvdiffusion"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_mvdiffusion(
        self, 
        mvdiffusion_pipe, 
        reference_image, # [1, H, W, 3]
        reference_mask,  # [1, H, W]
        seed,
        mv_guidance_scale, 
        num_inference_steps, 
    ):

        cfg = load_config_wonder3d(self.config_path_abs)

        batch = self.prepare_data(reference_image, reference_mask)

        mvdiffusion_pipe.set_progress_bar_config(disable=True)
        seed = int(seed)
        generator = torch.Generator(device=mvdiffusion_pipe.unet.device).manual_seed(seed)

        # repeat  (2B, Nv, 3, H, W)
        imgs_in = torch.cat([batch['imgs_in']] * 2, dim=0).to(WEIGHT_DTYPE)

        # (2B, Nv, Nce)
        camera_embeddings = torch.cat([batch['camera_embeddings']] * 2, dim=0).to(WEIGHT_DTYPE)

        task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0).to(WEIGHT_DTYPE)

        camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1).to(WEIGHT_DTYPE)

        # (B*Nv, 3, H, W)
        imgs_in = rearrange(imgs_in, "Nv C H W -> (Nv) C H W")
        # (B*Nv, Nce)
        # camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

        out = mvdiffusion_pipe(
            imgs_in,
            # camera_embeddings,
            generator=generator,
            guidance_scale=mv_guidance_scale,
            num_inference_steps=num_inference_steps,
            output_type='pt',
            num_images_per_prompt=1,
            **cfg.pipe_validation_kwargs,
        ).images

        num_views = out.shape[0] // 2
        # [N, 3, H, W] -> [N, H, W, 3]
        mv_images = out[num_views:].permute(0, 2, 3, 1)
        mv_normals = out[:num_views].permute(0, 2, 3, 1)
    
        return (mv_images, mv_normals, )
    
    def prepare_data(self, ref_image, ref_mask):
        single_image = torch_imgs_to_pils(ref_image, ref_mask)[0]
        dataset = MVSingleImageDataset(fix_cam_pose_dir=self.fix_cam_pose_dir_abs, num_views=6, img_wh=[256, 256], bg_color='white', single_image=single_image)
        return dataset[0]

class MVDream_Model:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mvdream_pipe": ("DIFFUSERS_PIPE",),
                "reference_image": ("IMAGE",), 
                "reference_mask": ("MASK",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "prompt_neg": ("STRING", {
                    "default": "ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate", 
                    "multiline": True
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mv_guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1}),
                "elevation": ("FLOAT", {"default": 0.0, "min": ELEVATION_MIN, "max": ELEVATION_MAX, "step": 0.0001}),
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "ORBIT_CAMPOSES",   # [orbit radius, elevation, azimuth, orbit center X, orbit center Y, orbit center Z]
    )
    RETURN_NAMES = (
        "multiview_images",
        "orbit_camposes",
    )
    FUNCTION = "run_mvdream"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_mvdream(
        self, 
        mvdream_pipe, 
        reference_image, # [1, H, W, 3]
        reference_mask,  # [1, H, W]
        prompt, 
        prompt_neg, 
        seed,
        mv_guidance_scale, 
        num_inference_steps, 
        elevation,
    ):
        if len(reference_image.shape) == 4:
            reference_image = reference_image.squeeze(0)
        if len(reference_mask.shape) == 3:
            reference_mask = reference_mask.squeeze(0)
            
        generator = torch.manual_seed(seed)
            
        reference_mask = reference_mask.unsqueeze(2)
        # give the white background to reference_image
        reference_image = (reference_image * reference_mask + (1 - reference_mask)).detach().cpu().numpy()

        # generate multi-view images
        mv_images = mvdream_pipe(prompt, reference_image, generator=generator, negative_prompt=prompt_neg, guidance_scale=mv_guidance_scale, num_inference_steps=num_inference_steps, elevation=elevation)
        mv_images = torch.from_numpy(np.stack([mv_images[1], mv_images[2], mv_images[3], mv_images[0]], axis=0)).float() # [4, H, W, 3], float32
        
        azimuths = [0, 90, 180, -90]
        elevations = [0, 0, 0, 0]
        radius = [4.0] * 4
        center = [0.0] * 4

        orbit_camposes = [azimuths, elevations, radius, center, center, center]

        return (mv_images, orbit_camposes)
    
class Load_Large_Multiview_Gaussian_Model:
    
    checkpoints_dir = "LGM"
    default_ckpt_name = "model_fp16.safetensors"
    default_repo_id = "ashawkey/LGM"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)
        all_models_names = get_list_filenames(cls.checkpoints_dir_abs, SUPPORTED_CHECKPOINTS_EXTENSIONS)
        if cls.default_ckpt_name not in all_models_names:
            all_models_names += [cls.default_ckpt_name]
            
        return {
            "required": {
                "model_name": (all_models_names, ),
                "lgb_config": (['big', 'default', 'small', 'tiny'], )
            },
        }
    
    RETURN_TYPES = (
        "LGM_MODEL",
    )
    RETURN_NAMES = (
        "lgm_model",
    )
    FUNCTION = "load_LGM"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_LGM(self, model_name, lgb_config):

        lgm_model = LargeMultiviewGaussianModel(config_defaults[lgb_config])
        
        ckpt_path = resume_or_download_model_from_hf(self.checkpoints_dir_abs, self.default_repo_id, model_name, self.__class__.__name__)
            
        if ckpt_path.endswith('safetensors'):
            ckpt = load_file(ckpt_path, device='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')

        lgm_model.load_state_dict(ckpt, strict=False)

        lgm_model = lgm_model.half().to(DEVICE)
        lgm_model.eval()
        
        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {ckpt_path}").msg.print()
        
        return (lgm_model, )
    
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    
class Large_Multiview_Gaussian_Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiview_images": ("IMAGE", ),
                "lgm_model": ("LGM_MODEL", ),
            },
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = (
        "GS_PLY",
    )
    RETURN_NAMES = (
        "gs_ply",
    )
    FUNCTION = "run_LGM"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_LGM(self, multiview_images, lgm_model):
        ref_image_torch = prepare_torch_img(multiview_images, lgm_model.opt.input_size, lgm_model.opt.input_size, DEVICE_STR) # [4, 3, 256, 256]
        ref_image_torch = TF.normalize(ref_image_torch, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        rays_embeddings = lgm_model.prepare_default_rays(DEVICE_STR)
        ref_image_torch = torch.cat([ref_image_torch, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, 256, 256]
        
        with torch.autocast(device_type=DEVICE_STR, dtype=WEIGHT_DTYPE):
            # generate gaussians
            gaussians = lgm_model.forward_gaussians(ref_image_torch)
        
        # convert gaussians to ply
        gs_ply = lgm_model.gs.to_ply(gaussians)
            
        return (gs_ply, )
    
class Convert_3DGS_to_Mesh_with_NeRF_and_Marching_Cubes:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_ply": ("GS_PLY",),
                "gs_config": (['big', 'default', 'small', 'tiny'], ),
                "training_nerf_iterations": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff}),
                "training_nerf_resolution": ("INT", {"default": 128, "min": 1, "max": 0xffffffffffffffff}),
                "marching_cude_grids_resolution": ("INT", {"default": 256, "min": 1, "max": 0xffffffffffffffff}),
                "marching_cude_grids_batch_size": ("INT", {"default": 128, "min": 1, "max": 0xffffffffffffffff}),
                "marching_cude_threshold": ("FLOAT", {"default": 10.0, "min": 0.0, "step": 0.01}),
                "training_mesh_iterations": ("INT", {"default": 2048, "min": 1, "max": 0xffffffffffffffff}),
                "training_mesh_resolution": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff}),
                "remesh_after_n_iteration": ("INT", {"default": 512, "min": 128, "max": 100000}),
                "training_albedo_iterations": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff}),
                "training_albedo_resolution": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff}),
                "texture_resolution": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "force_cuda_rast": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (
        "MESH",
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "mesh",
        "imgs",
        "alphas",
    )
    FUNCTION = "convert_gs_ply"
    CATEGORY = "Comfy3D/Algorithm"
    
    def convert_gs_ply(
        self, 
        gs_ply, 
        gs_config, 
        training_nerf_iterations,
        training_nerf_resolution,
        marching_cude_grids_resolution,
        marching_cude_grids_batch_size,
        marching_cude_threshold,
        training_mesh_iterations,
        training_mesh_resolution,
        remesh_after_n_iteration,
        training_albedo_iterations,
        training_albedo_resolution,
        texture_resolution,
        force_cuda_rast,
    ):
        with torch.inference_mode(False):
            chosen_config = config_defaults[gs_config]
            chosen_config.force_cuda_rast = force_cuda_rast
            converter = GSConverterNeRFMarchingCubes(config_defaults[gs_config], gs_ply).cuda()
            imgs, alphas = converter.fit_nerf(training_nerf_iterations, training_nerf_resolution)
            converter.fit_mesh(
                training_mesh_iterations, remesh_after_n_iteration, training_mesh_resolution, 
                marching_cude_grids_resolution, marching_cude_grids_batch_size, marching_cude_threshold
            )
            converter.fit_mesh_uv(training_albedo_iterations, training_albedo_resolution, texture_resolution)
        
            return(converter.get_mesh(), imgs, alphas)
    
class Load_TripoSR_Model:
    checkpoints_dir = "TripoSR"
    default_ckpt_name = "model.ckpt"
    default_repo_id = "stabilityai/TripoSR"
    config_path = "TripoSR_config.yaml"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)
        all_models_names = get_list_filenames(cls.checkpoints_dir_abs, SUPPORTED_CHECKPOINTS_EXTENSIONS)
        if cls.default_ckpt_name not in all_models_names:
            all_models_names += [cls.default_ckpt_name]
            
        cls.config_path_abs = os.path.join(CONFIG_ROOT_PATH, cls.config_path)
        return {
            "required": {
                "model_name": (all_models_names, ),
                "chunk_size": ("INT", {"default": 8192, "min": 1, "max": 10000})
            },
        }
    
    RETURN_TYPES = (
        "TSR_MODEL",
    )
    RETURN_NAMES = (
        "tsr_model",
    )
    FUNCTION = "load_TSR"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_TSR(self, model_name, chunk_size):
        
        ckpt_path = resume_or_download_model_from_hf(self.checkpoints_dir_abs, self.default_repo_id, model_name, self.__class__.__name__)

        tsr_model = TSR.from_pretrained(
            weight_path=ckpt_path,
            config_path=self.config_path_abs
        )
        
        tsr_model.renderer.set_chunk_size(chunk_size)
        tsr_model.to(DEVICE)
        
        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {ckpt_path}").msg.print()
        
        return (tsr_model, )
    
class TripoSR:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tsr_model": ("TSR_MODEL", ),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "geometry_extract_resolution": ("INT", {"default": 256, "min": 1, "max": 0xffffffffffffffff}),
                "marching_cude_threshold": ("FLOAT", {"default": 25.0, "min": 0.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    
    FUNCTION = "run_TSR"
    CATEGORY = "Comfy3D/Algorithm"

    @torch.no_grad()
    def run_TSR(self, tsr_model, reference_image, reference_mask, geometry_extract_resolution, marching_cude_threshold):
        mesh = None
        
        image = reference_image[0]
        mask = reference_mask[0].unsqueeze(2)
        image = torch.cat((image, mask), dim=2).detach().cpu().numpy()
        
        image = Image.fromarray(np.clip(255. * image, 0, 255).astype(np.uint8))
        image = self.fill_background(image)
        image = image.convert('RGB')
        
        scene_codes = tsr_model([image], DEVICE)
        meshes = tsr_model.extract_mesh(scene_codes, resolution=geometry_extract_resolution, threshold=marching_cude_threshold)
        mesh = Mesh.load_trimesh(given_mesh=meshes[0])

        return (mesh,)
    
    # Default model are trained on images with this background 
    def fill_background(self, image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image
    
class Load_SF3D_Model:
    checkpoints_dir = "StableFast3D"
    default_ckpt_name = "model.safetensors"
    default_repo_id = "stabilityai/stable-fast-3d"
    config_path = "StableFast3D_config.yaml"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)
        all_models_names = get_list_filenames(cls.checkpoints_dir_abs, SUPPORTED_CHECKPOINTS_EXTENSIONS)
        if cls.default_ckpt_name not in all_models_names:
            all_models_names += [cls.default_ckpt_name]
            
        cls.config_path_abs = os.path.join(CONFIG_ROOT_PATH, cls.config_path)
        return {
            "required": {
                "model_name": (all_models_names, ),
            },
        }
    
    RETURN_TYPES = (
        "SF3D_MODEL",
    )
    RETURN_NAMES = (
        "sf3d_model",
    )
    FUNCTION = "load_SF3D"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_SF3D(self, model_name):
        
        ckpt_path = resume_or_download_model_from_hf(self.checkpoints_dir_abs, self.default_repo_id, model_name, self.__class__.__name__)

        sf3d_model = SF3D.from_pretrained(
            config_path=self.config_path_abs,
            weight_path=ckpt_path
        )
        
        sf3d_model.eval()
        sf3d_model.to(DEVICE)
        
        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {ckpt_path}").msg.print()
        
        return (sf3d_model, )
    
class StableFast3D:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sf3d_model": ("SF3D_MODEL", ),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "texture_resolution": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "remesh_option": (["None", "Triangle"], ),
            }
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    
    FUNCTION = "run_SF3D"
    CATEGORY = "Comfy3D/Algorithm"

    @torch.no_grad()
    def run_SF3D(self, sf3d_model, reference_image, reference_mask, texture_resolution, remesh_option):
        single_image = torch_imgs_to_pils(reference_image, reference_mask)[0]
        
        with torch.autocast(device_type=DEVICE_STR, dtype=WEIGHT_DTYPE):
            model_batch = self.create_batch(single_image)
            model_batch = {k: v.cuda() for k, v in model_batch.items()}
            trimesh_mesh, _ = sf3d_model.generate_mesh(
                model_batch, texture_resolution, remesh_option
            )
        mesh = Mesh.load_trimesh(given_mesh=trimesh_mesh[0])

        return (mesh,)
    
    # Default model are trained on images with this background 
    def create_batch(self, input_image: Image):
        COND_WIDTH = 512
        COND_HEIGHT = 512
        COND_DISTANCE = 1.6
        COND_FOVY_DEG = 40
        BACKGROUND_COLOR = [0.5, 0.5, 0.5]

        # Cached. Doesn't change
        c2w_cond = sf3d_utils.default_cond_c2w(COND_DISTANCE)
        intrinsic, intrinsic_normed_cond = sf3d_utils.create_intrinsic_from_fov_deg(
            COND_FOVY_DEG, COND_HEIGHT, COND_WIDTH
        )
        
        img_cond = (
            torch.from_numpy(
                np.asarray(input_image.resize((COND_WIDTH, COND_HEIGHT))).astype(np.float32)
                / 255.0
            )
            .float()
            .clip(0, 1)
        )
        mask_cond = img_cond[:, :, -1:]
        rgb_cond = torch.lerp(
            torch.tensor(BACKGROUND_COLOR)[None, None, :], img_cond[:, :, :3], mask_cond
        )

        batch_elem = {
            "rgb_cond": rgb_cond,
            "mask_cond": mask_cond,
            "c2w_cond": c2w_cond.unsqueeze(0),
            "intrinsic_cond": intrinsic.unsqueeze(0),
            "intrinsic_normed_cond": intrinsic_normed_cond.unsqueeze(0),
        }
        # Add batch dim
        batched = {k: v.unsqueeze(0) for k, v in batch_elem.items()}
        return batched
    
class Load_CRM_MVDiffusion_Model:
    checkpoints_dir = "CRM"
    default_ckpt_name = ["pixel-diffusion.pth", "ccm-diffusion.pth"]
    default_conf_name = ["sd_v2_base_ipmv_zero_SNR.yaml", "sd_v2_base_ipmv_chin8_zero_snr.yaml"]
    default_repo_id = "Zhengyi/CRM"
    config_path = "CRM_configs"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)
        all_models_names = get_list_filenames(cls.checkpoints_dir_abs, SUPPORTED_CHECKPOINTS_EXTENSIONS)
        for ckpt_name in cls.default_ckpt_name:
            if ckpt_name not in all_models_names:
                all_models_names += [ckpt_name]
            
        cls.config_root_path_abs = os.path.join(CONFIG_ROOT_PATH, cls.config_path)
        return {
            "required": {
                "model_name": (all_models_names, ),
                "crm_config_path": (cls.default_conf_name, ),
            },
        }
    
    RETURN_TYPES = (
        "CRM_MVDIFFUSION_SAMPLER",
    )
    RETURN_NAMES = (
        "crm_mvdiffusion_sampler",
    )
    FUNCTION = "load_CRM"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_CRM(self, model_name, crm_config_path):
        
        from CRM.imagedream.ldm.util import (
            instantiate_from_config,
            get_obj_from_str,
        )

        crm_config_path = os.path.join(self.config_root_path_abs, crm_config_path)
        
        ckpt_path = resume_or_download_model_from_hf(self.checkpoints_dir_abs, self.default_repo_id, model_name, self.__class__.__name__)
            
        crm_config = OmegaConf.load(crm_config_path)

        crm_mvdiffusion_model = instantiate_from_config(crm_config.model)
        crm_mvdiffusion_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        crm_mvdiffusion_model = crm_mvdiffusion_model.to(DEVICE).to(WEIGHT_DTYPE)
        crm_mvdiffusion_model.device = DEVICE
        
        crm_mvdiffusion_sampler = get_obj_from_str(crm_config.sampler.target)(
            crm_mvdiffusion_model, device=DEVICE, dtype=WEIGHT_DTYPE, **crm_config.sampler.params
        )
        
        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {ckpt_path}").msg.print()
        
        return (crm_mvdiffusion_sampler, )
    
class CRM_Images_MVDiffusion_Model:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crm_mvdiffusion_sampler": ("CRM_MVDIFFUSION_SAMPLER",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "prompt": ("STRING", {
                    "default": "3D assets",
                    "multiline": True
                }),
                "prompt_neg": ("STRING", {
                    "default": "uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  obscure, unnatural colors, poor lighting, dull, and unclear.", 
                    "multiline": True
                }),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "mv_guidance_scale": ("FLOAT", {"default": 5.5, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1}),
                
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "ORBIT_CAMPOSES",   # [orbit radius, elevation, azimuth, orbit center X, orbit center Y, orbit center Z]
    )
    RETURN_NAMES = (
        "multiview_images",
        "orbit_camposes",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_model(
        self, 
        crm_mvdiffusion_sampler, 
        reference_image, # [1, H, W, 3]
        reference_mask,  # [1, H, W]
        prompt, 
        prompt_neg, 
        seed,
        mv_guidance_scale, 
        num_inference_steps, 
    ):
        pixel_img = torch_imgs_to_pils(reference_image, reference_mask)[0]
        pixel_img = CRMSampler.process_pixel_img(pixel_img)
        
        multiview_images = CRMSampler.stage1_sample(
            crm_mvdiffusion_sampler,
            pixel_img,
            prompt,
            prompt_neg,
            seed,
            mv_guidance_scale, 
            num_inference_steps
        )

        azimuths = [-90, 0, 180, 90, 0, 0]
        elevations = [0, 90, 0, 0, -90, 0]
        radius = [4.0] * 6
        center = [0.0] * 6

        orbit_camposes = [azimuths, elevations, radius, center, center, center]

        return (multiview_images, orbit_camposes)
    
class CRM_CCMs_MVDiffusion_Model:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crm_mvdiffusion_sampler": ("CRM_MVDIFFUSION_SAMPLER",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "multiview_images": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "3D assets",
                    "multiline": True
                }),
                "prompt_neg": ("STRING", {
                    "default": "uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  obscure, unnatural colors, poor lighting, dull, and unclear.", 
                    "multiline": True
                }),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "mv_guidance_scale": ("FLOAT", {"default": 5.5, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1}),
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "multiview_CCMs",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self, 
        crm_mvdiffusion_sampler, 
        reference_image, # [1, H, W, 3]
        reference_mask,  # [1, H, W]
        multiview_images, # [6, H, W, 3]
        prompt, 
        prompt_neg, 
        seed,
        mv_guidance_scale, 
        num_inference_steps, 
    ):
        pixel_img = torch_imgs_to_pils(reference_image, reference_mask)[0]
        pixel_img = CRMSampler.process_pixel_img(pixel_img)
        
        multiview_CCMs = CRMSampler.stage2_sample(
            crm_mvdiffusion_sampler,
            pixel_img,
            multiview_images,
            prompt,
            prompt_neg,
            seed,
            mv_guidance_scale, 
            num_inference_steps
        )
        
        return(multiview_CCMs, )
    
class Load_Convolutional_Reconstruction_Model:
    checkpoints_dir = "CRM"
    default_ckpt_name = "CRM.pth"
    default_repo_id = "Zhengyi/CRM"
    config_path = "CRM_configs/specs_objaverse_total.json"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)
        all_models_names = get_list_filenames(cls.checkpoints_dir_abs, SUPPORTED_CHECKPOINTS_EXTENSIONS)
        if cls.default_ckpt_name not in all_models_names:
            all_models_names += [cls.default_ckpt_name]
            
        cls.config_path_abs = os.path.join(CONFIG_ROOT_PATH, cls.config_path)
        return {
            "required": {
                "model_name": (all_models_names, ),
            },
        }
    
    RETURN_TYPES = (
        "CRM_MODEL",
    )
    RETURN_NAMES = (
        "crm_model",
    )
    FUNCTION = "load_CRM"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_CRM(self, model_name):
        
        ckpt_path = resume_or_download_model_from_hf(self.checkpoints_dir_abs, self.default_repo_id, model_name, self.__class__.__name__)
        
        crm_conf = json.load(open(self.config_path_abs))
        crm_model = ConvolutionalReconstructionModel(crm_conf).to(DEVICE)
        crm_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        
        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {ckpt_path}").msg.print()
        
        return (crm_model, )
    
class Convolutional_Reconstruction_Model:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "crm_model": ("CRM_MODEL", ),
                "multiview_images": ("IMAGE",),
                "multiview_CCMs": ("IMAGE",),
            }
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    
    FUNCTION = "run_CRM"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_CRM(self, crm_model, multiview_images, multiview_CCMs):

        np_imgs = np.concatenate(multiview_images.cpu().numpy(), 1) # (256, 256*6==1536, 3)
        np_xyzs = np.concatenate(multiview_CCMs.cpu().numpy(), 1) # (256, 1536, 3)
        
        mesh = CRMSampler.generate3d(crm_model, np_imgs, np_xyzs, DEVICE)
        
        return (mesh,)
    
class Zero123Plus_Diffusion_Model:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zero123plus_pipe": ("DIFFUSERS_PIPE",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1}),
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "ORBIT_CAMPOSES",   # [orbit radius, elevation, azimuth, orbit center X, orbit center Y, orbit center Z]
    )
    RETURN_NAMES = (
        "multiviews",
        "orbit_camposes",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self,
        zero123plus_pipe,
        reference_image,
        reference_mask,
        seed,
        guidance_scale,
        num_inference_steps,
    ):
        
        single_image = torch_imgs_to_pils(reference_image, reference_mask)[0]

        seed = int(seed)
        generator = torch.Generator(device=zero123plus_pipe.unet.device).manual_seed(seed)

        # sampling
        output_image = zero123plus_pipe(
            single_image, 
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps, 
        ).images[0]

        multiview_images = np.asarray(output_image, dtype=np.float32) / 255.0
        multiview_images = torch.from_numpy(multiview_images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
        multiview_images = rearrange(multiview_images, 'c (n h) (m w) -> (n m) h w c', n=3, m=2)        # (6, 320, 320, 3)

        azimuths = [30, 90, 150, -150, -90, -30]
        elevations = [-20, 10, -20, 10, -20, 10]
        radius = [4.0] * 6
        center = [0.0] * 6

        orbit_camposes = [azimuths, elevations, radius, center, center, center]

        return (multiview_images, orbit_camposes)
    
class Load_InstantMesh_Reconstruction_Model:
    checkpoints_dir = "InstantMesh"
    default_ckpt_names = ["instant_mesh_large.ckpt", "instant_mesh_base.ckpt", "instant_nerf_large.ckpt", "instant_nerf_base.ckpt"]
    default_repo_id = "TencentARC/InstantMesh"
    config_root_dir = "InstantMesh_configs"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)
        all_models_names = get_list_filenames(cls.checkpoints_dir_abs, SUPPORTED_CHECKPOINTS_EXTENSIONS)
        for ckpt_name in cls.default_ckpt_names:
            if ckpt_name not in all_models_names:
                all_models_names += [ckpt_name]
                
        cls.config_root_path_abs = os.path.join(CONFIG_ROOT_PATH, cls.config_root_dir)
        return {
            "required": {
                "model_name": (all_models_names, ),
            },
        }
    
    RETURN_TYPES = (
        "LRM_MODEL",
    )
    RETURN_NAMES = (
        "lrm_model",
    )
    FUNCTION = "load_LRM"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_LRM(self, model_name):

        from InstantMesh.utils.train_util import instantiate_from_config

        is_flexicubes = True if model_name.startswith('instant_mesh') else False
        
        config_name = model_name.split(".")[0] + ".yaml"
        config_path = os.path.join(self.config_root_path_abs, config_name)
        config = OmegaConf.load(config_path)

        lrm_model = instantiate_from_config(config.model_config)
        ckpt_path = resume_or_download_model_from_hf(self.checkpoints_dir_abs, self.default_repo_id, model_name, self.__class__.__name__)

        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        lrm_model.load_state_dict(state_dict, strict=True)

        lrm_model = lrm_model.to(DEVICE)
        if is_flexicubes:
            lrm_model.init_flexicubes_geometry(DEVICE, fovy=30.0)
        lrm_model = lrm_model.eval()
        
        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {ckpt_path}").msg.print()

        return (lrm_model, )
    
class InstantMesh_Reconstruction_Model:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lrm_model": ("LRM_MODEL", ),
                "multiview_images": ("IMAGE",),
                "orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "orbit_camera_fovy": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 180.0, "step": 0.1}),
                "texture_resolution": ("INT", {"default": 1024, "min": 128, "max": 8192}),
            }
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    
    FUNCTION = "run_LRM"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_LRM(self, lrm_model, multiview_images, orbit_camera_poses, orbit_camera_fovy, texture_resolution):

        images = multiview_images.permute(0, 3, 1, 2).unsqueeze(0).to(DEVICE)   # [N, H, W, 3] -> [1, N, 3, H, W]
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

        # convert camera format from orbit to lrm inputs
        azimuths, elevations, radius = orbit_camera_poses[0], orbit_camera_poses[1], orbit_camera_poses[2]
        input_cameras = oribt_camera_poses_to_input_cameras(azimuths, elevations, radius=radius, fov=orbit_camera_fovy).to(DEVICE)

        # get triplane
        planes = lrm_model.forward_planes(images, input_cameras)

        # get mesh
        mesh_out = lrm_model.extract_mesh(
            planes,
            use_texture_map=True,
            texture_resolution=texture_resolution,
        )

        vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
        tex_map = troch_image_dilate(tex_map.permute(1, 2, 0))  # [3, H, W] -> [H, W, 3]
        
        mesh = Mesh(v=vertices, f=faces, vt=uvs, ft=mesh_tex_idx, albedo=tex_map, device=DEVICE)
        mesh.auto_normal()
        return (mesh,)

class Era3D_MVDiffusion_Model:
    
    config_path = "Era3D_config.yaml"
    @classmethod
    def INPUT_TYPES(cls):
        cls.config_path_abs = os.path.join(CONFIG_ROOT_PATH, cls.config_path)
        return {
            "required": {
                "era3d_pipe": ("DIFFUSERS_PIPE",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "image_crop_size": ("INT", {"default": 420, "min": 400, "max": 8192}),
                "seed": ("INT", {"default": 600, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 40, "min": 1}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.01}),
                "radius": ("FLOAT", {"default": 4.0, "min": 0.1, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "ORBIT_CAMPOSES",   # [orbit radius, elevation, azimuth, orbit center X, orbit center Y, orbit center Z]
    )
    RETURN_NAMES = (
        "multiviews",
        "multiview_normals",
        "orbit_camposes",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self,
        era3d_pipe,
        reference_image,
        reference_mask,
        image_crop_size,
        seed,
        guidance_scale,
        num_inference_steps,
        eta,
        radius,
    ):
        cfg = load_config_era3d(self.config_path_abs)
        
        single_image = torch_imgs_to_pils(reference_image, reference_mask)[0]

        # Get the dataset
        cfg.dataset.prompt_embeds_path = os.path.join(ROOT_PATH, cfg.dataset.prompt_embeds_path)
        dataset = Era3DSingleImageDataset(
            single_image=single_image,
            crop_size=image_crop_size,
            dtype=WEIGHT_DTYPE,
            **cfg.dataset
        )

        # Get input data
        img_batch = dataset.__getitem__(0)

        imgs_in = torch.cat([img_batch['imgs_in']]*2, dim=0).to(DEVICE).to(WEIGHT_DTYPE)    # (B*Nv, 3, H, W) B==1
        #num_views = imgs_in.shape[1]

        normal_prompt_embeddings, clr_prompt_embeddings = img_batch['normal_prompt_embeddings'], img_batch['color_prompt_embeddings'] 
        prompt_embeddings = torch.cat([normal_prompt_embeddings, clr_prompt_embeddings], dim=0).to(DEVICE).to(WEIGHT_DTYPE)    # (B*Nv, N, C) B==1

        generator = torch.Generator(device=era3d_pipe.unet.device).manual_seed(seed)

        # sampling
        with torch.autocast(DEVICE_STR):
            unet_out = era3d_pipe(
                imgs_in, None, prompt_embeds=prompt_embeddings,
                generator=generator, guidance_scale=guidance_scale, output_type='pt', num_images_per_prompt=1, 
                num_inference_steps=num_inference_steps, eta=eta
            )
        
        out = unet_out.images
        bsz = out.shape[0] // 2

        # (1, 3, 512, 512)
        normals_pred = out[:bsz]    
        images_pred = out[bsz:] 
        
        # [N, 3, H, W] -> [N, H, W, 3]
        multiview_images = images_pred.permute(0, 2, 3, 1).to(reference_image.dtype)   
        multiview_normals = normals_pred.permute(0, 2, 3, 1).to(reference_image.dtype)   

        azimuths = [0, 45, 90, 180, -90, -45]
        elevations = [0.0] * 6
        radius = [radius] * 6
        center = [0.0] * 6

        orbit_camposes = [azimuths, elevations, radius, center, center, center]

        return (multiview_images, multiview_normals, orbit_camposes)
    
class Instant_NGP:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "reference_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "reference_orbit_camera_fovy": ("FLOAT", {"default": 49.1, "min": 0.0, "max": 180.0, "step": 0.1}),
                "training_iterations": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff}),
                "training_resolution": ("INT", {"default": 128, "min": 128, "max": 8192}),
                "marching_cude_grids_resolution": ("INT", {"default": 256, "min": 1, "max": 0xffffffffffffffff}),
                "marching_cude_grids_batch_size": ("INT", {"default": 128, "min": 1, "max": 0xffffffffffffffff}),
                "marching_cude_threshold": ("FLOAT", {"default": 10.0, "min": 0.0, "step": 0.01}),
                "texture_resolution": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "background_color": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "force_cuda_rast": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_instant_ngp"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_instant_ngp(
        self, 
        reference_image, 
        reference_mask, 
        reference_orbit_camera_poses, 
        reference_orbit_camera_fovy,
        training_iterations,
        training_resolution,
        marching_cude_grids_resolution,
        marching_cude_grids_batch_size,
        marching_cude_threshold,
        texture_resolution,
        background_color,
        force_cuda_rast
    ):
        with torch.inference_mode(False):
            
            ngp = InstantNGP(training_resolution).to(DEVICE)
            ngp.prepare_training(reference_image, reference_mask, reference_orbit_camera_poses, reference_orbit_camera_fovy)
            ngp.fit_nerf(training_iterations, background_color)
            
            vertices, triangles = marching_cubes_density_to_mesh(ngp.get_density, marching_cude_grids_resolution, marching_cude_grids_batch_size, marching_cude_threshold)

            v = torch.from_numpy(vertices).contiguous().float().to(DEVICE)
            f = torch.from_numpy(triangles).contiguous().int().to(DEVICE)

            mesh = Mesh(v=v, f=f, device=DEVICE)
            mesh.auto_normal()
            mesh.auto_uv()
            
            mesh.albedo = color_func_to_albedo(mesh, ngp.get_color, texture_resolution, device=DEVICE, force_cuda_rast=force_cuda_rast)
            
            return (mesh, )
        
class FlexiCubes_MVS:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_depth_maps": ("IMAGE",),
                "reference_masks": ("MASK",),
                "reference_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
                "reference_orbit_camera_fovy": ("FLOAT", {"default": 49.1, "min": 0.0, "max": 180.0, "step": 0.1}),
                "training_iterations": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 0xffffffffffffffff}),
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.001, "step": 0.001}),
                "voxel_grids_resolution": ("INT", {"default": 128, "min": 1, "max": 0xffffffffffffffff}),
                "depth_min_distance": ("FLOAT", {"default": 0.5, "min": 0.0, "step": 0.01}),
                "depth_max_distance": ("FLOAT", {"default": 5.5, "min": 0.0, "step": 0.01}),
                "mask_loss_weight": ("FLOAT", {"default": 1.0, "min": 0.01, "step": 0.01}),
                "depth_loss_weight": ("FLOAT", {"default": 100.0, "min": 0.01, "step": 0.01}),
                "normal_loss_weight": ("FLOAT", {"default": 1.0, "min": 0.01, "step": 0.01}),
                "sdf_regularizer_weight": ("FLOAT", {"default": 0.2, "min": 0.01, "step": 0.01}),
                "remove_floaters_weight": ("FLOAT", {"default": 0.5, "min": 0.01, "step": 0.01}),
                "cube_stabilizer_weight": ("FLOAT", {"default": 0.1, "min": 0.01, "step": 0.01}),
                "force_cuda_rast": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference_normal_maps": ("IMAGE",), 
            }
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_flexicubes"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_flexicubes(
        self,
        reference_depth_maps,
        reference_masks,
        reference_orbit_camera_poses,
        reference_orbit_camera_fovy,
        training_iterations,
        batch_size,
        learning_rate,
        voxel_grids_resolution,
        depth_min_distance,
        depth_max_distance,
        mask_loss_weight,
        depth_loss_weight,
        normal_loss_weight,
        sdf_regularizer_weight,
        remove_floaters_weight,
        cube_stabilizer_weight,
        force_cuda_rast,
        reference_normal_maps=None
    ):
        
        with torch.inference_mode(False):
            
            fc_trainer = FlexiCubesTrainer(
                training_iterations,
                batch_size,
                learning_rate,
                voxel_grids_resolution,
                depth_min_distance,
                depth_max_distance,
                mask_loss_weight,
                depth_loss_weight,
                normal_loss_weight,
                sdf_regularizer_weight,
                remove_floaters_weight,
                cube_stabilizer_weight,
                force_cuda_rast,
                device=DEVICE
            )
            
            fc_trainer.prepare_training(reference_depth_maps, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy, reference_normal_maps)
            
            fc_trainer.training()
            
            mesh = fc_trainer.get_mesh()
            
            return (mesh, )

class Load_Unique3D_Custom_UNet:
    default_repo_id = "MrForExample/Unique3D"
    config_root_dir = "Unique3D_configs"

    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(CKPT_DIFFUSERS_PATH, cls.default_repo_id)
        cls.config_path_abs = os.path.join(CONFIG_ROOT_PATH, cls.config_root_dir)
        return {
            "required": {
                "pipe": ("DIFFUSERS_PIPE",),
                "config_name": (["image2mvimage", "image2normal"],),
            },
        }
    
    RETURN_TYPES = (
        "DIFFUSERS_PIPE",
    )
    RETURN_NAMES = (
        "pipe",
    )
    FUNCTION = "load_diffusers_unet"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_diffusers_unet(self, pipe, config_name):

        from Unique3D.custum_3d_diffusion.trainings.config_classes import ExprimentConfig
        from Unique3D.custum_3d_diffusion.custum_modules.unifield_processor import AttnConfig, ConfigurableUNet2DConditionModel
        from Unique3D.custum_3d_diffusion.trainings.utils import load_config
        # Download models and configs
        cfg_path = os.path.join(self.config_path_abs, config_name + ".yaml")
        checkpoint_dir_path = os.path.join(self.checkpoints_dir_abs, config_name)
        checkpoint_path = os.path.join(checkpoint_dir_path, "unet_state_dict.pth")

        cfg: ExprimentConfig = load_config(ExprimentConfig, cfg_path)
        if cfg.init_config.init_unet_path == "":
            cfg.init_config.init_unet_path = checkpoint_dir_path
        init_config: AttnConfig = load_config(AttnConfig, cfg.init_config)
        configurable_unet = ConfigurableUNet2DConditionModel(init_config, WEIGHT_DTYPE)
        configurable_unet.enable_xformers_memory_efficient_attention()

        state_dict = torch.load(checkpoint_path)
        configurable_unet.unet.load_state_dict(state_dict, strict=False)
        # Move unet, vae and text_encoder to device and cast to weight_dtype
        configurable_unet.unet.to(DEVICE, dtype=WEIGHT_DTYPE)

        pipe.unet = configurable_unet.unet
        
        cstr(f"[{self.__class__.__name__}] loaded unet ckpt from {checkpoint_path}").msg.print()
        return (pipe, )
    
class Unique3D_MVDiffusion_Model:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unique3d_pipe": ("DIFFUSERS_PIPE",),
                "reference_image": ("IMAGE",),
                "seed": ("INT", {"default": 1145, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1}),
                "image_resolution": ([256, 512],),
                "radius": ("FLOAT", {"default": 4.0, "min": 0.1, "step": 0.01}),
                "preprocess_images":  ("BOOLEAN", {"default": True},),
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "ORBIT_CAMPOSES",   # [orbit radius, elevation, azimuth, orbit center X, orbit center Y, orbit center Z]
    )
    RETURN_NAMES = (
        "multiviews",
        "orbit_camposes",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self,
        unique3d_pipe,
        reference_image,    # Need to have white background
        seed,
        guidance_scale,
        num_inference_steps,
        image_resolution,
        radius,
        preprocess_images,
    ):
        from Unique3D.scripts.utils import simple_image_preprocess

        pil_image_list = torch_imgs_to_pils(reference_image)
        for i in range(len(pil_image_list)):
            if preprocess_images:
                pil_image_list[i] = simple_image_preprocess(pil_image_list[i])

        pil_image_list = pils_rgba_to_rgb(pil_image_list, bkgd="WHITE")

        generator = torch.Generator(device=unique3d_pipe.unet.device).manual_seed(seed)

        image_pils = unique3d_pipe(
            image=pil_image_list,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=image_resolution,
            height=image_resolution,
            height_cond=image_resolution,
            width_cond=image_resolution,
        ).images

        # [N, H, W, 3]
        multiview_images = pils_to_torch_imgs(image_pils, reference_image.device)

        azimuths = [0, 90, 180, -90]
        elevations = [0.0] * 4
        radius = [radius] * 4
        center = [0.0] * 4

        orbit_camposes = [azimuths, elevations, radius, center, center, center]

        return (multiview_images, orbit_camposes)

class Fast_Normal_Maps_To_Mesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front_side_back_normal_maps": ("IMAGE",),
                "front_side_back_normal_masks": ("MASK",),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_fast_recon"
    CATEGORY = "Comfy3D/Algorithm"

    def run_fast_recon(self, front_side_back_normal_maps, front_side_back_normal_masks):
        pil_normal_list = torch_imgs_to_pils(front_side_back_normal_maps, front_side_back_normal_masks)
        meshes = fast_geo(pil_normal_list[0], pil_normal_list[2], pil_normal_list[1])
        vertices, faces, _ = from_py3d_mesh(meshes)

        mesh = Mesh(v=vertices, f=faces, device=DEVICE)
        return (mesh,)

class ExplicitTarget_Mesh_Optimization:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "normal_maps": ("IMAGE",),
                "normal_masks": ("MASK",),
                "reconstruction_steps": ("INT", {"default": 200, "min": 0, "max": 0xffffffffffffffff}),
                "coarse_reconstruct_resolution": ("INT", {"default": 512, "min": 128, "max": 8192}),
                "loss_expansion_weight": ("FLOAT", {"default": 0.1, "min": 0.01, "step": 0.01}),
                "refinement_steps": ("INT", {"default": 100, "min": 0, "max": 0xffffffffffffffff}),
                "target_warmup_update_num": ("INT", {"default": 5, "min": 1, "max": 0xffffffffffffffff}),
                "target_update_interval": ("INT", {"default": 20, "min": 1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "normal_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
            }
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_ET_mesh_optimization"
    CATEGORY = "Comfy3D/Algorithm"

    def run_ET_mesh_optimization(
        self, 
        mesh,
        normal_maps, 
        normal_masks,  
        reconstruction_steps,
        coarse_reconstruct_resolution,
        loss_expansion_weight,
        refinement_steps, 
        target_warmup_update_num,
        target_update_interval,
        normal_orbit_camera_poses=None,
    ):
        #TODO For now only support four orthographic view with elevation equals zero
        #azimuths, elevations, radius = normal_orbit_camera_poses[0], normal_orbit_camera_poses[1], normal_orbit_camera_poses[2]
        pil_normal_list = torch_imgs_to_pils(normal_maps, normal_masks)
        normal_stg1 = [img.resize((coarse_reconstruct_resolution, coarse_reconstruct_resolution)) for img in pil_normal_list]
        with torch.inference_mode(False):
            vertices, faces = mesh.v.detach().clone().to(DEVICE), mesh.f.detach().clone().to(DEVICE).type(torch.int64)
            if reconstruction_steps > 0:
                vertices, faces = reconstruct_stage1(normal_stg1, steps=reconstruction_steps, vertices=vertices, faces=faces, loss_expansion_weight=loss_expansion_weight)
                
            if refinement_steps > 0:
                vertices, faces = run_mesh_refine(vertices, faces, pil_normal_list, steps=refinement_steps, update_normal_interval=target_update_interval, update_warmup=target_warmup_update_num, )

            mesh = Mesh(v=vertices, f=faces, device=DEVICE)
            return (mesh,)

class ExplicitTarget_Color_Projection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "reference_images": ("IMAGE",), 
                "reference_masks": ("MASK",),
                "projection_resolution": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "complete_unseen_rgb":  ("BOOLEAN", {"default": True},),
            },
            "optional": {
                "reference_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
            }
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_color_projection"
    CATEGORY = "Comfy3D/Algorithm"

    def run_color_projection(
        self, 
        mesh, 
        reference_images, 
        reference_masks,  
        projection_resolution, 
        complete_unseen_rgb,
        reference_orbit_camera_poses=None,
    ):
        pil_image_list = torch_imgs_to_pils(reference_images, reference_masks)

        meshes = to_py3d_mesh(mesh.v, mesh.f)

        #TODO Convert camera format, currently only support elevation equal to zero
        if reference_orbit_camera_poses is None:
            img_num = len(reference_images)
            interval = 360 / img_num
            angle = 0
            azimuths = []
            for i in range(0, img_num):
                azimuths.append(angle)
                angle += interval
        else:
            azimuths = [360 + angle if angle < 0 else angle for angle in reference_orbit_camera_poses[0]]
        cam_list = get_cameras_list(azimuths, DEVICE, focal=1)
        
        new_meshes = multiview_color_projection(meshes, pil_image_list, resolution=projection_resolution, device=DEVICE, complete_unseen=complete_unseen_rgb, confidence_threshold=0.2, cameras_list=cam_list)
        vertices, faces, vertex_colors = from_py3d_mesh(new_meshes)
        vertices = vertices / 2 * 1.35

        mesh = Mesh(v=vertices, f=faces, vc=vertex_colors, device=DEVICE)
        mesh.auto_normal()
        return (mesh,)
    
class Load_CharacterGen_MVDiffusion_Model:
    checkpoints_dir = "CharacterGen"
    default_repo_id = "zjpshadow/CharacterGen"
    config_path = "CharacterGen_configs/Stage_2D_infer.yaml"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)
        cls.config_root_path_abs = os.path.join(CONFIG_ROOT_PATH, cls.config_path)
        return {
            "required": {
                "force_download": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = (
        "CHARACTER_MV_GEN_PIPE",
    )
    RETURN_NAMES = (
        "character_mv_gen_pipe",
    )
    FUNCTION = "load_model"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_model(self, force_download):
        # Download checkpoints
        snapshot_download(repo_id=self.default_repo_id, local_dir=self.checkpoints_dir_abs, force_download=force_download, repo_type="model", ignore_patterns=["*.json", "*.py"])
        # Load pre-trained models
        character_mv_gen_pipe = Inference2D_API(checkpoint_root_path=self.checkpoints_dir_abs, **OmegaConf.load(self.config_root_path_abs))
        
        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {self.checkpoints_dir_abs}").msg.print()
        return (character_mv_gen_pipe,)
    
class CharacterGen_MVDiffusion_Model:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_mv_gen_pipe": ("CHARACTER_MV_GEN_PIPE",),
                "reference_image": ("IMAGE", ),
                "reference_mask": ("MASK",),
                "target_image_width": ("INT", {"default": 512, "min": 128, "max": 8192}),
                "target_image_height": ("INT", {"default": 768, "min": 128, "max": 8192}),
                "seed": ("INT", {"default": 2333, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 40, "min": 1}),
                "prompt": ("STRING", {
                    "default": "high quality, best quality",
                    "multiline": True
                }),
                "prompt_neg": ("STRING", {
                    "default": "", 
                    "multiline": True
                }),
                "radius": ("FLOAT", {"default": 1.5, "min": 0.1, "step": 0.01})
            },
        }
    
    RETURN_TYPES = (
        "IMAGE",
        "ORBIT_CAMPOSES",   # [orbit radius, elevation, azimuth, orbit center X, orbit center Y, orbit center Z]
    )
    RETURN_NAMES = (
        "multiviews",
        "orbit_camposes",
    )
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(
        self,
        character_mv_gen_pipe,
        reference_image,
        reference_mask,
        target_image_width,
        target_image_height,
        seed,
        guidance_scale,
        num_inference_steps,
        prompt,
        prompt_neg,
        radius
    ):
        single_image = torch_imgs_to_pils(reference_image, reference_mask)[0]

        multiview_images = character_mv_gen_pipe.inference(
            single_image, target_image_width, target_image_height, prompt=prompt, prompt_neg=prompt_neg, 
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, seed=seed
        )
        
        azimuths = [-90, 90, 180, 0]
        elevations = [0.0] * 4
        radius = [radius] * 4
        center = [0.0] * 4

        orbit_camposes = [azimuths, elevations, radius, center, center, center]

        return (multiview_images, orbit_camposes)
    
class Load_CharacterGen_Reconstruction_Model:
    checkpoints_dir = "CharacterGen"
    default_repo_id = "zjpshadow/CharacterGen"
    config_path = "CharacterGen_configs/Stage_3D_infer.yaml"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)
        cls.config_root_path_abs = os.path.join(CONFIG_ROOT_PATH, cls.config_path)
        return {
            "required": {
                "force_download": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = (
        "CHARACTER_LRM_PIPE",
    )
    RETURN_NAMES = (
        "character_lrm_pipe",
    )
    FUNCTION = "load_model"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_model(self, force_download):
        # Download checkpoints
        snapshot_download(repo_id=self.default_repo_id, local_dir=self.checkpoints_dir_abs, force_download=force_download, repo_type="model", ignore_patterns=["*.json", "*.py"])
        # Load pre-trained models
        character_lrm_pipe = Inference3D_API(checkpoint_root_path=self.checkpoints_dir_abs, cfg=load_config_cg3d(self.config_root_path_abs))
        
        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {self.checkpoints_dir_abs}").msg.print()
        return (character_lrm_pipe,)
    
class CharacterGen_Reconstruction_Model:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "character_lrm_pipe": ("CHARACTER_LRM_PIPE", ),
                "multiview_images": ("IMAGE",),
                "multiview_masks": ("MASK",),
            }
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    
    FUNCTION = "run_LRM"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_LRM(self, character_lrm_pipe, multiview_images, multiview_masks):
        pil_mv_image_list = torch_imgs_to_pils(multiview_images, multiview_masks, alpha_min=0.2)
        
        vertices, faces = character_lrm_pipe.inference(pil_mv_image_list)

        mesh = Mesh(v=vertices, f=faces.to(torch.int64), device=DEVICE)
        mesh.auto_normal()
        mesh.auto_uv()
        
        return (mesh,)
    
class Load_Craftsman_Shape_Diffusion_Model:
    checkpoints_dir = "Craftsman"
    default_repo_id = "wyysf/CraftsMan"
    default_ckpt_name = "image-to-shape-diffusion/clip-mvrgb-modln-l256-e64-ne8-nd16-nl6-aligned-vae/model.ckpt"
    config_path = "Craftsman_config.yaml"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(CKPT_ROOT_PATH, cls.checkpoints_dir)
        cls.config_root_path_abs = os.path.join(CONFIG_ROOT_PATH, cls.config_path)
        all_models_names = get_list_filenames(cls.checkpoints_dir_abs, SUPPORTED_CHECKPOINTS_EXTENSIONS, recursive=True)
        if cls.default_ckpt_name not in all_models_names:
            all_models_names += [cls.default_ckpt_name]

        return {
            "required": {
                "model_name": (all_models_names, ),
            },
        }
    
    RETURN_TYPES = (
        "CRAFTSMAN_MODEL",
    )
    RETURN_NAMES = (
        "craftsman_model",
    )
    FUNCTION = "load_model"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_model(self, model_name):
        ckpt_path = resume_or_download_model_from_hf(self.checkpoints_dir_abs, self.default_repo_id, model_name, self.__class__.__name__)
        
        cfg: ExperimentConfigCraftsman
        cfg = load_config_craftsman(self.config_root_path_abs)

        craftsman_model: BaseSystem = craftsman.find(cfg.system_type)(
            cfg.system, 
        )
        
        craftsman_model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'])
        craftsman_model = craftsman_model.to(DEVICE).eval()
        
        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {self.checkpoints_dir_abs}").msg.print()
        return (craftsman_model,)
    
class Craftsman_Shape_Diffusion_Model:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "craftsman_model": ("CRAFTSMAN_MODEL", ),
                "multiview_images": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1}),
                "marching_cude_grids_resolution": ("INT", {"default": 256, "min": 1, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    
    FUNCTION = "run_model"
    CATEGORY = "Comfy3D/Algorithm"
    
    @torch.no_grad()
    def run_model(self, craftsman_model, multiview_images, seed, guidance_scale, num_inference_steps, marching_cude_grids_resolution):
        pil_mv_image_list = torch_imgs_to_pils(multiview_images)
        
        sample_inputs = {"mvimages": [pil_mv_image_list]}   # view order: front, right, back, left
        
        latents = craftsman_model.sample(
            sample_inputs,
            sample_times=1,
            steps=num_inference_steps,
            guidance_scale=guidance_scale,
            return_intermediates=False,
            seed=seed
        )[0]
        
        cstr(f"[{self.__class__.__name__}] Starting to extract mesh...").msg.print()
        # decode the latents to mesh
        box_v = 1.1
        mesh_outputs, _ = craftsman_model.shape_model.extract_geometry(
            latents,
            bounds=[-box_v, -box_v, -box_v, box_v, box_v, box_v],
            grids_resolution=marching_cude_grids_resolution
        )
        vertices, faces = torch.from_numpy(mesh_outputs[0][0]).to(DEVICE), torch.from_numpy(mesh_outputs[0][1]).to(torch.int64).to(DEVICE)

        mesh = Mesh(v=vertices, f=faces, device=DEVICE)
        mesh.auto_normal()
        mesh.auto_uv()
        
        return (mesh,)
    


ORBITPOSE_PRESET = ["Custom", "CRM(6)", "Zero123Plus(6)", "Wonder3D(6)", "Era3D(6)", "MVDream(4)", "Unique3D(4)", "CharacterGen(4)"]

OrbitPosesList = {
    "Custom":           [[-90.0, 0.0, 180.0, 90.0, 0.0, 0.0], [0.0, 90.0, 0.0, 0.0, -90.0, 0.0], [4.0, 4.0, 4.0, 4.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    "CRM(6)":           [[-90.0, 0.0, 180.0, 90.0, 0.0, 0.0], [0.0, 90.0, 0.0, 0.0, -90.0, 0.0], [4.0, 4.0, 4.0, 4.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    "Wonder3D(6)":      [[0.0, 45.0, 90.0, 180.0, -90.0, -45.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 4.0, 4.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    "Zero123Plus(6)":   [[30.0, 90.0, 150.0, -150.0, -90.0, -30.0], [-20.0, 10.0, -20.0, 10.0, -20.0, 10.0], [4.0, 4.0, 4.0, 4.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    "Era3D(6)":         [[0.0, 45.0, 90.0, 180.0, -90.0, -45.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], #[[radius], [radius], [radius], [radius], [radius], [radius]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    "MVDream(4)":       [[0.0, 90.0, 180.0, -90.0], [0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    "Unique3D(4)":      [[0.0, 90.0, 180.0, -90.0], [0.0, 0.0, 0.0, 0.0]], #[[radius], [radius], [radius], [radius]], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
    "CharacterGen(4)":  [[-90.0, 180.0, 90.0, 0.0], [0.0, 0.0, 0.0, 0.0]], #[[radius], [radius], [radius], [radius]], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
}

class OrbitPoses_JK:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "orbitpose_preset": (ORBITPOSE_PRESET, {"default": "Custom"}),
                "azimuths": ("STRING", {"default": "-90.0, 0.0, 180.0, 90.0, 0.0, 0.0"}),
                "elevations": ("STRING", {"default": "0.0, 90.0, 0.0, 0.0, -90.0, 0.0"}),
                "radius": ("STRING", {"default": "4.0, 4.0, 4.0, 4.0, 4.0, 4.0"}),
                "center": ("STRING", {"default": "0.0, 0.0, 0.0, 0.0, 0.0, 0.0"}),
            },
        }
    
    RETURN_TYPES = ("ORBIT_CAMPOSES", "ORBIT_CAMPOSES",)
    RETURN_NAMES = ("orbit_lists", "orbit_camposes",)
    
    FUNCTION = "get_orbit_poses"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def get_orbit_poses(self, orbitpose_preset, azimuths, elevations, radius, center):
        
        orbit_lists = OrbitPosesList.get(f"{orbitpose_preset}")
        
        if orbitpose_preset == "Custom":
            azimuths = azimuths.split(",")
            elevations = elevations.split(",")
            radius = radius.split(",")
            center = center.split(",")
            orbit_azimuths = [float(item) for item in azimuths]
            orbit_elevations = [float(item) for item in elevations]
            orbit_radius = [float(item) for item in radius]
            orbit_center = [float(item) for item in center]
            orbit_lists = [orbit_azimuths, orbit_elevations, orbit_radius, orbit_center, orbit_center, orbit_center]
        elif orbitpose_preset == "Era3D(6)":
            radius = radius.split(",")
            orbit_radius = [float(item) for item in radius]
            orbit_center = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            orbit_lists = [orbit_lists[0], orbit_lists[1], orbit_radius, orbit_center, orbit_center, orbit_center]
        elif orbitpose_preset == "Unique3D(4)" or orbitpose_preset == "CharacterGen(4)":
            radius = radius.split(",")
            orbit_radius = [float(item) for item in radius]
            orbit_radius.pop(4)
            orbit_radius.pop(4)
            orbit_center = [0.0, 0.0, 0.0, 0.0]
            orbit_lists = [orbit_lists[0], orbit_lists[1], orbit_radius, orbit_center, orbit_center, orbit_center]
        
        orbit_camposes = []

        for i in range(0, len(orbit_lists[0])):
            orbit_camposes.append([orbit_lists[2][i], orbit_lists[1][i], orbit_lists[0][i], orbit_lists[3][i], orbit_lists[4][i], orbit_lists[5][i]])
        
        return (orbit_lists, orbit_camposes,)

class OrbitLists_to_OrbitPoses_JK:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "orbit_lists": ("ORBIT_CAMPOSES",),
            },
        }
    
    RETURN_TYPES = ("ORBIT_CAMPOSES",)
    RETURN_NAMES = ("orbit_camposes",)
    
    FUNCTION = "convert_orbit_poses"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def convert_orbit_poses(self, orbit_lists):
        
        orbit_camposes = []

        for i in range(0, len(orbit_lists[0])):
            orbit_camposes.append([orbit_lists[2][i], orbit_lists[1][i], orbit_lists[0][i], orbit_lists[3][i], orbit_lists[4][i], orbit_lists[5][i]])
        
        return (orbit_camposes,)

class OrbitPoses_to_OrbitLists_JK:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "orbit_camposes": ("ORBIT_CAMPOSES",),
            },
        }
    
    RETURN_TYPES = ("ORBIT_CAMPOSES",)
    RETURN_NAMES = ("orbit_lists",)
    
    FUNCTION = "convert_orbit_poses"
    CATEGORY = "Comfy3D/Preprocessor"
    
    def convert_orbit_poses(self, orbit_camposes):
        
        orbit_azimuths = []
        orbit_elevations = []
        orbit_radius = []
        orbit_center0 = []
        orbit_center1 = []
        orbit_center2 = []

        for i in range(0, len(orbit_camposes)):
            orbit_azimuths.append(orbit_camposes[i][2])
            orbit_elevations.append(orbit_camposes[i][1])
            orbit_radius.append(orbit_camposes[i][0])
            orbit_center0.append(orbit_camposes[i][3])
            orbit_center1.append(orbit_camposes[i][4])
            orbit_center2.append(orbit_camposes[i][5])
        
        orbit_lists = [orbit_azimuths, orbit_elevations, orbit_radius, orbit_center0, orbit_center1, orbit_center2]
        
        return (orbit_lists,)
