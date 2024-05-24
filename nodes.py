import os
import sys
import re
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
from .algorithms.main_3DGS import GaussianSplatting, GaussianSplattingCameraController, GSParams
from .algorithms.main_3DGS_renderer import GaussianSplattingRenderer
from .algorithms.diff_mesh import DiffMesh, DiffMeshCameraController
from .algorithms.diff_mesh import DiffRastRenderer
from .algorithms.dmtet import DMTetMesh
from .algorithms.triplane_gaussian_transformers import TGS
from .algorithms.large_multiview_gaussian_model import LGM
from .algorithms.nerf_marching_cubes_converter import GSConverterNeRFMarchingCubes
from .algorithms.NeuS_runner import NeuSParams, NeuSRunner
from .algorithms.convolutional_reconstruction_model import CRMSampler
from .algorithms.Instant_NGP import InstantNGP
from .algorithms.flexicubes_trainer import FlexiCubesTrainer

from .tgs.utils.config import ExperimentConfig, load_config as load_config_tgs
from .tgs.data import CustomImageOrbitDataset
from .tgs.utils.misc import todevice, get_device
from .lgm.core.options import config_defaults
from .lgm.mvdream.pipeline_mvdream import MVDreamPipeline
from .mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
from .mvdiffusion.data.single_image_dataset import SingleImageDataset
from .mvdiffusion.utils.misc import load_config as load_config_wonder3d
from .tsr.system import TSR
from .crm.model.crm.model import CRM
from .zero123plus.pipeline import Zero123PlusPipeline
from .instant_mesh.utils.camera_util import oribt_camera_poses_to_input_cameras

from .shared_utils.image_utils import prepare_torch_img, torch_img_to_pil_rgba, troch_image_dilate
from .shared_utils.common_utils import cstr, parse_save_filename, get_list_filenames, resume_or_download_model_from_hf

DIFFUSERS_PIPE_DICT = OrderedDict([
    ("MVDreamPipeline", MVDreamPipeline),
    ("Wonder3DMVDiffusionPipeline", MVDiffusionImagePipeline),
    ("Zero123PlusPipeline", Zero123PlusPipeline),
    ("DiffusionPipeline", DiffusionPipeline),
    ("StableDiffusionPipeline", StableDiffusionPipeline),
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                "save_path": ("STRING", {"default": 'Mesh_%Y-%m-%d-%M-%S-%f.obj', "multiline": False}),
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
    
    def switch_axis_and_scale(self, mesh, axis_x_to, axis_y_to, axis_z_to, flip_normal):
        
        switched_mesh = None
        
        if axis_x_to[1] != axis_y_to[1] and axis_x_to[1] != axis_z_to[1] and axis_y_to[1] != axis_z_to[1]:
            target_axis, target_scale, coordinate_invert_count = get_target_axis_and_scale([axis_x_to, axis_y_to, axis_z_to])
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
                "orbit_radius_start": ("FLOAT", {"default": 0.0, 'step': 0.0001}),
                "orbit_radius_stop": ("FLOAT", {"default": 0.0, 'step': 0.0001}),
                "orbit_radius_step": ("FLOAT", {"default": 0.1, 'step': 0.0001}),
                "elevation_start": ("FLOAT", {"default": 0.0, "min": ELEVATION_MIN, "max": ELEVATION_MAX, 'step': 0.0001}),
                "elevation_stop": ("FLOAT", {"default": 0.0, "min": ELEVATION_MIN, "max": ELEVATION_MAX, 'step': 0.0001}),
                "elevation_step": ("FLOAT", {"default": 0.0, "min": ELEVATION_MIN, "max": ELEVATION_MAX, 'step': 0.0001}),
                "azimuth_start": ("FLOAT", {"default": 0.0, "min": AZIMUTH_MIN, "max": AZIMUTH_MAX, 'step': 0.0001}),
                "azimuth_stop": ("FLOAT", {"default": 0.0, "min": AZIMUTH_MIN, "max": AZIMUTH_MAX, 'step': 0.0001}),
                "azimuth_step": ("FLOAT", {"default": 0.0, "min": AZIMUTH_MIN, "max": AZIMUTH_MAX, 'step': 0.0001}),
                "orbit_center_X_start": ("FLOAT", {"default": 0.0, 'step': 0.0001}),
                "orbit_center_X_stop": ("FLOAT", {"default": 0.0, 'step': 0.0001}),
                "orbit_center_X_step": ("FLOAT", {"default": 0.1, 'step': 0.0001}),
                "orbit_center_Y_start": ("FLOAT", {"default": 0.0, 'step': 0.0001}),
                "orbit_center_Y_stop": ("FLOAT", {"default": 0.0, 'step': 0.0001}),
                "orbit_center_Y_step": ("FLOAT", {"default": 0.1, 'step': 0.0001}),
                "orbit_center_Z_start": ("FLOAT", {"default": 0.0, 'step': 0.0001}),
                "orbit_center_Z_stop": ("FLOAT", {"default": 0.0, 'step': 0.0001}),
                "orbit_center_Z_step": ("FLOAT", {"default": 0.1, 'step': 0.0001}),
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
    
class Generate_Orbit_Camera_Poses:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE",), 
                "generate_pose_command": ("STRING", {
                    "default": "#([start_reference_image_index : end_reference_image_index], orbit_radius, elevation_angle [-90, 90], start_azimuth_angle [0, 360], end_azimuth_angle [0, 360])\n([0:30], 1.75, 0, 0, 360)", 
                    "multiline": True
                }),
            },
        }

    RETURN_TYPES = (
        "ORBIT_CAMPOSES",   # [orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z]
    )
    RETURN_NAMES = (
        "orbit_camposes",
    )
    FUNCTION = "get_camposes"
    CATEGORY = "Comfy3D/Preprocessor"
    
    class Slice_Camposes:
        def __init__(self, start_reference_image_index, end_reference_image_index, camposes_start_to_end):
            self.start_reference_image_index = start_reference_image_index
            self.end_reference_image_index = end_reference_image_index
            self.camposes_start_to_end = camposes_start_to_end
    
    def get_camposes(self, reference_images, generate_pose_command):
        orbit_camposes = []
        
        self.ref_imgs_num_minus_1 = len(reference_images) - 1
        
        # To match pattern ( [ start_reference_image_index : end_reference_image_index ] , orbit_radius, elevation_angle , start_azimuth_angle , end_azimuth_angle )
        pattern = re.compile(r"\([ \t]*\[[ \t]*(\d+)[ \t]*:[ \t]*(\d+)[ \t]*\][ \t]*,[ \t]*([\d]+\.?[\d]*)[ \t]*,[ \t]*([\d]+\.?[\d]*)[ \t]*,[ \t]*([\d]+\.?[\d]*)[ \t]*,[ \t]*([\d]+\.?[\d]*)[ \t]*\)")
        all_matches = pattern.findall(generate_pose_command)

        all_slice_camposes = []
        for match in all_matches:
            start_reference_image_index, end_reference_image_index, orbit_radius, elevation_angle, start_azimuth_angle, end_azimuth_angle = (int(s) if i < 2 else float(s) for i, s in enumerate(match))
            
            end_reference_image_index = min(end_reference_image_index, self.ref_imgs_num_minus_1)
            
            if start_reference_image_index <= end_reference_image_index:
            
                azimuth_imgs_num = end_reference_image_index - start_reference_image_index + 1
                # calculate all the reference camera azimuth angles
                camposes_start_to_end = []
                if start_azimuth_angle > end_azimuth_angle:
                    azimuth_angle_interval = -(end_azimuth_angle + 360 - start_azimuth_angle) / azimuth_imgs_num
                else:
                    azimuth_angle_interval = (end_azimuth_angle - start_azimuth_angle) / azimuth_imgs_num
                    
                now_azimuth_angle = start_azimuth_angle
                for _ in range(azimuth_imgs_num):
                    camposes_start_to_end.append((orbit_radius, elevation_angle, now_azimuth_angle, 0.0, 0.0, 0.0))
                    now_azimuth_angle = (now_azimuth_angle + azimuth_angle_interval) % 360
                    
                all_slice_camposes.append(Generate_Orbit_Camera_Poses.Slice_Camposes(start_reference_image_index, end_reference_image_index, camposes_start_to_end))
                    
            else:
                cstr(f"[{self.__class__.__name__}] start_reference_image_index: {start_reference_image_index} must smaller than or equal to end_reference_image_index: {end_reference_image_index}").error.print()
        
        all_slice_camposes = sorted(all_slice_camposes, key=lambda slice_camposes:slice_camposes.start_reference_image_index)
        
        last_end_index_plus_1 = 0
        for slice_camposes in all_slice_camposes:
            if last_end_index_plus_1 == slice_camposes.start_reference_image_index:
                orbit_camposes.extend(slice_camposes.camposes_start_to_end)
                last_end_index_plus_1 = slice_camposes.end_reference_image_index + 1
            else:
                orbit_camposes = []
                cstr(f"[{self.__class__.__name__}] Last end_reference_image_index: {end_reference_image_index} plus 1 must equal to current start_reference_image_index: {start_reference_image_index}").error.print()
        
        return (orbit_camposes, )
    
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
            }
        }
        
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "rendered_mesh_images",    # [Number of Poses, H, W, 3]
        "rendered_mesh_masks",    # [Number of Poses, H, W, 1]
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
    ):
        
        renderer = DiffRastRenderer(mesh, force_cuda_rasterize)
        
        cam_controller = DiffMeshCameraController(
            renderer, 
            render_image_width, 
            render_image_height, 
            render_orbit_camera_fovy, 
            static_bg=[render_background_color_r, render_background_color_g, render_background_color_b]
        )
        
        all_rendered_images, all_rendered_masks = cam_controller.render_all_pose(render_orbit_camera_poses)
        all_rendered_masks = all_rendered_masks.squeeze(-1)  # [N, H, W, 1] -> [N, H, W]
        
        return (all_rendered_images, all_rendered_masks)    # [N, H, W, 3], [N, H, W]
        
   
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
    )
    RETURN_NAMES = (
        "rendered_gs_images",    # [Number of Poses, H, W, 3]
        "rendered_gs_masks",    # [Number of Poses, H, W, 1]
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
        
        all_rendered_images, all_rendered_masks = cam_controller.render_all_pose(render_orbit_camera_poses)
        all_rendered_images = all_rendered_images.permute(0, 2, 3, 1)   # [N, 3, H, W] -> [N, H, W, 3]
        all_rendered_masks = all_rendered_masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
        
        return (all_rendered_images, all_rendered_masks)
    
class Gaussian_Splatting:

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
                    
                    gs = GaussianSplatting(gs_params, gs_init_input)
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
        force_cuda_rasterize
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
                
                    mesh_fitter = DiffMesh(mesh, training_iterations, batch_size, texture_learning_rate, train_mesh_geometry, geometry_learning_rate, ms_ssim_loss_weight, remesh_after_n_iteration, force_cuda_rasterize)
                    
                    mesh_fitter.prepare_training(reference_images, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy)
                    mesh_fitter.training()
                    
                    trained_mesh, baked_texture = mesh_fitter.get_mesh_and_texture()
                    
            else:
                cstr(f"[{self.__class__.__name__}] Number of reference images {ref_imgs_num} does not equal to number of reference camera poses {ref_cam_poses_num}").error.print()
        else:
            cstr(f"[{self.__class__.__name__}] Number of reference images {ref_imgs_num} does not equal to number of masks {ref_masks_num}").error.print()
        
        return (trained_mesh, baked_texture, )
    
class Deep_Marching_Tetrahedrons:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_iterations": ("INT", {"default": 5000, "min": 1, "max": 100000}),
                "points_cloud_fitting_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.01}),
                "mesh_smoothing_weight": ("FLOAT", {"default": 0.1, "min": 0.0, "step": 0.01}),
                "chamfer_faces_sample_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "step": 0.01}),
                "mesh_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "step": 0.01}),
                "grid_resolution": ([128, 64, 32], ),
                "geometry_learning_rate": ("FLOAT", {"default": 0.0001, "min": 0.00001, "step": 0.00001}),
                "positional_encoding_multires": ("INT", {"default": 2, "min": 2}),
                "mlp_internal_dims": ("INT", {"default": 128, "min": 8}),
                "mlp_hidden_layer_num": ("INT", {"default": 5, "min": 1}),
            },
            
            "optional": {
                "reference_points_cloud": ("POINTCLOUD",),
                "reference_images": ("IMAGE",), 
                "reference_masks": ("MASK",),
            }
        }
    
    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "trained_mesh",
    )
    FUNCTION = "run_dmtet"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_dmtet(self, training_iterations, points_cloud_fitting_weight, mesh_smoothing_weight, chamfer_faces_sample_scale, mesh_scale, grid_resolution, geometry_learning_rate,
                  positional_encoding_multires, mlp_internal_dims, mlp_hidden_layer_num, 
                  reference_points_cloud=None, reference_images=None, reference_masks=None):
        
        with torch.inference_mode(False):
            dmtet_mesh = DMTetMesh(
                training_iterations, 
                points_cloud_fitting_weight, 
                mesh_smoothing_weight, 
                chamfer_faces_sample_scale, 
                mesh_scale, 
                grid_resolution, 
                geometry_learning_rate, 
                positional_encoding_multires, 
                mlp_internal_dims, 
                mlp_hidden_layer_num
            )
            
            if reference_points_cloud is None and reference_images is None:
                cstr(f"[{self.__class__.__name__}] reference_points_cloud and reference_images cannot both be None, you need at least provide one of them!").error.print()
                raise Exception("User didn't provide necessary inputs, stop executing the code!")
            
            dmtet_mesh.training(reference_points_cloud, reference_images, reference_masks)
            mesh = dmtet_mesh.get_mesh()
            return (mesh, )

class Load_Triplane_Gaussian_Transformers:
    
    checkpoints_dir = "checkpoints/tgs"
    default_ckpt_name = "model_lvis_rel.ckpt"
    default_repo_id = "VAST-AI/TriplaneGaussian"
    tgs_config_path = "configs/tgs_config.yaml"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(ROOT_PATH, cls.checkpoints_dir)
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
        
        config_path = os.path.join(ROOT_PATH, self.tgs_config_path)
        cfg: ExperimentConfig = load_config_tgs(config_path)

        ckpt_path = resume_or_download_model_from_hf(self.checkpoints_dir_abs, self.default_repo_id, model_name, self.__class__.__name__)
            
        cfg.system.weights=ckpt_path
        tgs_model = TGS(cfg=cfg.system).to(device)
        
        save_path = os.path.join(ROOT_PATH, "outputs")
        tgs_model.set_save_dir(save_path)
        
        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {ckpt_path}").msg.print()

        return (tgs_model, )
    
class Triplane_Gaussian_Transformers:
    
    tgs_config_path = "configs/tgs_config.yaml"
    
    @classmethod
    def INPUT_TYPES(cls):
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
        config_path = os.path.join(ROOT_PATH, self.tgs_config_path)
        cfg: ExperimentConfig = load_config_tgs(config_path)
        
        #save_path = os.path.join(ROOT_PATH, "outputs")
        #tgs_model.set_save_dir(save_path)

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
            
        """
        # Output rendered video, for testing this node only
        tgs_model.save_img_sequences(
            "video",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            delete=True,
        )
        """
        
        return (gs_ply[0], )
    
class Load_Diffusers_Pipeline:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusers_pipeline_name": (list(DIFFUSERS_PIPE_DICT.keys()),),
                "model_name": ("STRING", {"default": "ashawkey/imagedream-ipmv-diffusers", "multiline": False}),
                "custom_pipeline": ("STRING", {"default": "", "multiline": False}),
            },
        }
    
    RETURN_TYPES = (
        "DIFFUSERS_PIPE",
    )
    RETURN_NAMES = (
        "pipe",
    )
    FUNCTION = "load_diffusers_pipe"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_diffusers_pipe(self, diffusers_pipeline_name, model_name, custom_pipeline):
        
        # resume pretrained checkpoint
        ckpt_path = os.path.join(ROOT_PATH, "checkpoints", model_name)
        if not os.path.exists(ckpt_path):
            cstr(f"[{self.__class__.__name__}] can't find checkpoint {ckpt_path}, will download it from repo {model_name}").warning.print()
            
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_name, local_dir=ckpt_path, repo_type="model")
        
        diffusers_pipeline_class = DIFFUSERS_PIPE_DICT[diffusers_pipeline_name]
        
        # load diffusers pipeline
        if not custom_pipeline:
            custom_pipeline = None
            
        pipe = diffusers_pipeline_class.from_pretrained(
            ckpt_path,
            torch_dtype=WEIGHT_DTYPE,
            custom_pipeline=custom_pipeline,
            trust_remote_code=True,
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
    
    checkpoints_dir = "checkpoints"

    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_base_dir_abs = os.path.join(ROOT_PATH, cls.checkpoints_dir)
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

        checkpoints_dir_abs = os.path.join(self.checkpoints_base_dir_abs, repo_id)
        ckpt_path = resume_or_download_model_from_hf(checkpoints_dir_abs, repo_id, model_name, self.__class__.__name__)

        state_dict = torch.load(ckpt_path, map_location='cpu')
        pipe.unet.load_state_dict(state_dict, strict=True)

        pipe = pipe.to(DEVICE)

        return (pipe, )

class Wonder3D_MVDiffusion_Model:
    
    wonder3d_config_path = "configs/wonder3d_config.yaml"
    fix_cam_pose_dir = "mvdiffusion/data/fixed_poses/nine_views"
    
    @classmethod
    def INPUT_TYPES(cls):
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

        config_path = os.path.join(ROOT_PATH, self.wonder3d_config_path)
        cfg = load_config_wonder3d(config_path)

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
        mv_images = out[num_views:].permute(0, 2, 3, 1)   # [N, 3, H, W] -> [N, H, W, 3]
        mv_normals = out[:num_views].permute(0, 2, 3, 1)   # [N, 3, H, W] -> [N, H, W, 3]
    
        return (mv_images, mv_normals, )
    
    def prepare_data(self, ref_image, ref_mask):
        single_image = torch_img_to_pil_rgba(ref_image, ref_mask)
        abs_fix_cam_pose_dir = os.path.join(ROOT_PATH, self.fix_cam_pose_dir)
        dataset = SingleImageDataset(fix_cam_pose_dir=abs_fix_cam_pose_dir, num_views=6, img_wh=[256, 256], bg_color='white', single_image=single_image)
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
                "elevation": ("FLOAT", {"default": 0.0, "min": ELEVATION_MIN, "max": ELEVATION_MAX, 'step': 0.0001}),
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
    
    checkpoints_dir = "checkpoints/lgm"
    default_ckpt_name = "model_fp16.safetensors"
    default_repo_id = "ashawkey/LGM"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(ROOT_PATH, cls.checkpoints_dir)
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

        lgm_model = LGM(config_defaults[lgb_config])
        
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
        device = "cuda"
        ref_image_torch = prepare_torch_img(multiview_images, lgm_model.opt.input_size, lgm_model.opt.input_size, device) # [4, 3, 256, 256]
        ref_image_torch = TF.normalize(ref_image_torch, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        rays_embeddings = lgm_model.prepare_default_rays(device)
        ref_image_torch = torch.cat([ref_image_torch, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, 256, 256]
        
        with torch.autocast(device_type=device, dtype=WEIGHT_DTYPE):
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
    
    def convert_gs_ply(self, gs_ply, gs_config, force_cuda_rast):
        with torch.inference_mode(False):
            chosen_config = config_defaults[gs_config]
            chosen_config.force_cuda_rast = force_cuda_rast
            converter = GSConverterNeRFMarchingCubes(config_defaults[gs_config], gs_ply).cuda()
            imgs, alphas = converter.fit_nerf()
            converter.fit_mesh()
            converter.fit_mesh_uv()
        
            return(converter.get_mesh(), imgs, alphas)
    
class NeuS:
    NeuS_config_path = "configs/NeuS.conf"
    fix_cam_pose_dir = "NeuS/models/fixed_poses"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "reference_normals": ("IMAGE",),
                "training_iterations": ("INT", {"default": 1000, "min": 1, "max": 100000}), # longer time, better result. 1w will be ok for most cases
                "batch_size": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff}),
                "learning_rate": ("FLOAT", {"default": 5e-4, "min": 0.00001, "step": 0.00001}),
                "learning_rate_alpha": ("FLOAT", {"default": 5e-4, "min": 0.00001, "step": 0.00001}),
                "color_loss_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.01}),
                "mesh_smoothing_weight": ("FLOAT", {"default": 0.1, "min": 0.0, "step": 0.01}),
                "mask_loss_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.01}),
                "normal_loss_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.01}),
                "sparse_loss_weight": ("FLOAT", {"default": 0.1, "min": 0.0, "step": 0.01}),
                "warm_up_end": ("INT", {"default": 500, "min": 0, "max": 0xffffffffffffffff}),
                "anneal_end": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "use_white_background":  ("BOOLEAN", {"default": True},),
                "geometry_extract_resolution": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff}),
                "marching_cude_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "run_NeuS"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_NeuS(
        self, 
        reference_image,    # [num_views, H, W, 3]
        reference_mask,     # [num_views, H, W]
        reference_normals,  # [num_views, H, W, 3]
        training_iterations,
        batch_size,
        learning_rate,
        learning_rate_alpha,
        color_loss_weight,
        mesh_smoothing_weight,
        mask_loss_weight,
        normal_loss_weight,
        sparse_loss_weight,
        warm_up_end,
        anneal_end,
        use_white_background,
        geometry_extract_resolution,
        marching_cude_threshold
    ):
        mesh = None
        num_views = reference_image.shape[0]
        
        if num_views in [6, 5, 4]:
            config_path = os.path.join(ROOT_PATH, self.NeuS_config_path)
            abs_fix_cam_pose_dir = os.path.join(ROOT_PATH, self.fix_cam_pose_dir)
            
            with torch.inference_mode(False):
                NeuS_params = NeuSParams(
                    training_iterations,
                    batch_size,
                    learning_rate,
                    learning_rate_alpha,
                    color_loss_weight,
                    mesh_smoothing_weight,
                    mask_loss_weight,
                    normal_loss_weight,
                    sparse_loss_weight,
                    warm_up_end,
                    anneal_end,
                    use_white_background
                )
                
                runner = NeuSRunner(
                    reference_image,
                    reference_mask,
                    reference_normals,
                    config_path,
                    abs_fix_cam_pose_dir,
                    num_views,
                    NeuS_params
                )
                
                cstr(f"[{self.__class__.__name__}] Training NeuS...").msg.print()
                runner.train()
                
                cstr(f"[{self.__class__.__name__}] Extracting Mesh...").msg.print()
                mesh = runner.extract_mesh(resolution=geometry_extract_resolution, threshold=marching_cude_threshold)
        else:
            cstr(f"[{self.__class__.__name__}] Number of views must be one of the following: 6, 5, 4. But got {num_views}").error.print()
            
        return(mesh, )
    
class Load_TripoSR_Model:
    checkpoints_dir = "checkpoints/tsr"
    default_ckpt_name = "model.ckpt"
    default_repo_id = "stabilityai/TripoSR"
    tsr_config_path = "configs/tsr_config.yaml"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(ROOT_PATH, cls.checkpoints_dir)
        all_models_names = get_list_filenames(cls.checkpoints_dir_abs, SUPPORTED_CHECKPOINTS_EXTENSIONS)
        if cls.default_ckpt_name not in all_models_names:
            all_models_names += [cls.default_ckpt_name]
            
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
            weight_path = ckpt_path,
            config_path = os.path.join(ROOT_PATH, self.tsr_config_path)
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
        cstr(f"[{self.__class__.__name__}] Running TripoSR...").msg.print()
        
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
    
class Load_CRM_MVDiffusion_Model:
    checkpoints_dir = "checkpoints/crm"
    default_ckpt_name = ["pixel-diffusion.pth", "ccm-diffusion.pth"]
    default_conf_name = ["configs/crm_configs/sd_v2_base_ipmv_zero_SNR.yaml", "configs/crm_configs/sd_v2_base_ipmv_chin8_zero_snr.yaml"]
    default_repo_id = "Zhengyi/CRM"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(ROOT_PATH, cls.checkpoints_dir)
        all_models_names = get_list_filenames(cls.checkpoints_dir_abs, SUPPORTED_CHECKPOINTS_EXTENSIONS)
        for ckpt_name in cls.default_ckpt_name:
            if ckpt_name not in all_models_names:
                all_models_names += [ckpt_name]
            
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
        
        from .crm.imagedream.ldm.util import (
            instantiate_from_config,
            get_obj_from_str,
        )
        
        if not os.path.isabs(crm_config_path):
            crm_config_path = os.path.join(ROOT_PATH, crm_config_path)
        
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
        pixel_img = torch_img_to_pil_rgba(reference_image, reference_mask)
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
        pixel_img = torch_img_to_pil_rgba(reference_image, reference_mask)
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
    checkpoints_dir = "checkpoints/crm"
    default_ckpt_name = "CRM.pth"
    default_repo_id = "Zhengyi/CRM"
    config_path = "configs/crm_configs/specs_objaverse_total.json"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(ROOT_PATH, cls.checkpoints_dir)
        all_models_names = get_list_filenames(cls.checkpoints_dir_abs, SUPPORTED_CHECKPOINTS_EXTENSIONS)
        if cls.default_ckpt_name not in all_models_names:
            all_models_names += [cls.default_ckpt_name]
            
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
        
        crm_conf = json.load(open(os.path.join(ROOT_PATH, self.config_path)))
        crm_model = CRM(crm_conf).to(DEVICE)
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
        
        single_image = torch_img_to_pil_rgba(reference_image, reference_mask)

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
    checkpoints_dir = "checkpoints/crm"
    default_ckpt_names = ["instant_mesh_large.ckpt", "instant_mesh_base.ckpt", "instant_nerf_large.ckpt", "instant_nerf_base.ckpt"]
    default_repo_id = "TencentARC/InstantMesh"
    config_root_dir = "configs/InstantMesh_configs"
    
    @classmethod
    def INPUT_TYPES(cls):
        cls.checkpoints_dir_abs = os.path.join(ROOT_PATH, cls.checkpoints_dir)
        all_models_names = get_list_filenames(cls.checkpoints_dir_abs, SUPPORTED_CHECKPOINTS_EXTENSIONS)
        for ckpt_name in cls.default_ckpt_names:
            if ckpt_name not in all_models_names:
                all_models_names += [ckpt_name]
            
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

        from .instant_mesh.utils.train_util import instantiate_from_config

        is_flexicubes = True if model_name.startswith('instant_mesh') else False
        
        config_name = model_name.split(".")[0] + ".yaml"
        config_path = os.path.join(ROOT_PATH, self.config_root_dir, config_name)
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
        force_cuda_rast
    ):
        with torch.inference_mode(False):
            
            ngp = InstantNGP(training_resolution).to(DEVICE)
            ngp.prepare_training(reference_image, reference_mask, reference_orbit_camera_poses, reference_orbit_camera_fovy)
            ngp.fit_nerf(training_iterations)
            
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