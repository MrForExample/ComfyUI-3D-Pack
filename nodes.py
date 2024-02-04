import os
import sys
import re
import math
import copy
from enum import Enum
import folder_paths as comfy_paths

import torch
from torch.utils.data import DataLoader
import numpy as np

from plyfile import PlyData

from .shared_utils.common_utils import cstr, parse_save_filename
from .mesh_processer.mesh import Mesh
from .mesh_processer.mesh_utils import (
    ply_to_points_cloud, 
    get_target_axis_and_scale, 
    switch_ply_axis_and_scale, 
    switch_mesh_axis_and_scale, 
    calculate_max_sh_degree_from_gs_ply,
)
from .algorithms.main_3DGS import GaussianSplatting, GaussianSplattingCameraController, GSParams
from .algorithms.main_3DGS_renderer import GaussianSplattingRenderer
from .algorithms.diff_texturing import DiffTextureBaker
from .algorithms.dmtet import DMTetMesh
from .algorithms.triplane_gaussian_transformers import TGS
from .tgs.utils.config import ExperimentConfig, load_config
from .tgs.data import CustomImageOrbitDataset
from .tgs.utils.misc import todevice, get_device

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

ELEVATION_MIN = -90
ELEVATION_MAX = 90.0
AZIMUTH_MIN = -180.0
AZIMUTH_MAX = 180.0

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
    
    def switch_axis_and_scale(self, mesh, axis_x_to, axis_y_to, axis_z_to):
        
        switched_mesh = None
        
        if axis_x_to[1] != axis_y_to[1] and axis_x_to[1] != axis_z_to[1] and axis_y_to[1] != axis_z_to[1]:
            target_axis, target_scale, coordinate_invert_count = get_target_axis_and_scale([axis_x_to, axis_y_to, axis_z_to])
            switched_mesh = switch_mesh_axis_and_scale(mesh, target_axis, target_scale)
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
        "ORBIT_CAMPOSES",   # (orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z)
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "FLOAT",
    )
    RETURN_NAMES = (
        "orbit_camposes",
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
        "ORBIT_CAMPOSES",   # (orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z)
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
   
class Gaussian_Splatting_Orbit_Renderer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_ply": ("GS_PLY",),
                "render_image_width": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "render_image_height": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "render_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # (orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z)
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
        all_rendered_masks = all_rendered_masks.permute(0, 2, 3, 1).squeeze(3)  # [N, 1, H, W] -> [N, H, W]
        
        return (all_rendered_images, all_rendered_masks)
    
class Gaussian_Splatting:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE",), 
                "reference_masks": ("MASK",),
                "reference_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # (orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z)
                "reference_orbit_camera_fovy": ("FLOAT", {"default": 49.1, "min": 0.0, "max": 180.0, "step": 0.1}),
                "training_iterations": ("INT", {"default": 1000, "min": 1, "max": 100000}),
                "batch_size": ("INT", {"default": 3, "min": 1, "max": 0xffffffffffffffff}),
                "loss_value_scale": ("FLOAT", {"default": 10000.0, "min": 1.0}),
                "ms_ssim_loss_weight": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, }),
                "alpha_loss_weight": ("FLOAT", {"default": 3, "min": 0.0, }),
                "offset_loss_weight": ("FLOAT", {"default": 0.0, "min": 0.0, }),
                "offset_opacity_loss_weight": ("FLOAT", {"default": 0.0, "min": 0.0, }),
                "invert_background_probability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "feature_learning_rate": ("FLOAT", {"default": 0.01, "min": 0.00001, "step": 0.00001}),
                "opacity_learning_rate": ("FLOAT", {"default": 0.05,  "min": 0.00001, "step": 0.00001}),
                "scaling_learning_rate": ("FLOAT", {"default": 0.005,  "min": 0.00001, "step": 0.00001}),
                "rotation_learning_rate": ("FLOAT", {"default": 0.005,  "min": 0.00001, "step": 0.00001}),
                "position_learning_rate_init": ("FLOAT", {"default": 0.001,  "min": 0.00001, "step": 0.00001}),
                "position_learning_rate_final": ("FLOAT", {"default": 0.00002, "min": 0.00001, "step": 0.00001}),
                "position_learning_rate_delay_mult": ("FLOAT", {"default": 0.02, "min": 0.00001, "step": 0.00001}),
                "position_learning_rate_max_steps": ("INT", {"default": 500, "min": 1}),
                "initial_gaussians_num": ("INT", {"default": 5000, "min": 1}),
                "K_nearest_neighbors": ("INT", {"default": 3, "min": 1}),
                "percent_dense": ("FLOAT", {"default": 0.01,  "min": 0.00001, "step": 0.00001}),
                "density_start_iterations": ("INT", {"default": 100, "min": 0}),
                "density_end_iterations": ("INT", {"default": 100000, "min": 0}),
                "densification_interval": ("INT", {"default": 100, "min": 1}),
                "opacity_reset_interval": ("INT", {"default": 700, "min": 1}),
                "densify_grad_threshold": ("FLOAT", {"default": 0.01, "min": 0.00001, "step": 0.00001}),
                "gaussian_sh_degree": ("INT", {"default": 0, "min": 0}),
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
        loss_value_scale,
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
                
                    gs_params = GSParams(training_iterations,
                                        batch_size,
                                        loss_value_scale,
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
                                        gaussian_sh_degree)
                    
                    
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
    
class Bake_Texture_To_Mesh:
    
    def __init__(self):
        self.need_update = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE",), 
                "reference_masks": ("MASK",),
                "reference_orbit_camera_poses": ("ORBIT_CAMPOSES",),    # (orbit radius, elevation, azimuth, orbit center X,  orbit center Y,  orbit center Z)
                "reference_orbit_camera_fovy": ("FLOAT", {"default": 49.1, "min": 0.0, "max": 180.0, "step": 0.1}),
                "mesh": ("MESH",),
                "mesh_albedo_width": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "mesh_albedo_height": ("INT", {"default": 1024, "min": 128, "max": 8192}),
                "training_iterations": ("INT", {"default": 1000, "min": 1, "max": 100000}),
                "batch_size": ("INT", {"default": 3, "min": 1, "max": 0xffffffffffffffff}),
                "texture_learning_rate": ("FLOAT", {"default": 0.1, "min": 0.00001, "step": 0.00001}),
                "train_mesh_geometry": ("BOOLEAN", {"default": False},),
                "geometry_learning_rate": ("FLOAT", {"default": 0.01, "min": 0.00001, "step": 0.00001}),
                "ms_ssim_loss_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
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
    FUNCTION = "bake_texture"
    CATEGORY = "Comfy3D/Algorithm"
    
    def bake_texture(self, 
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
    force_cuda_rasterize):
        
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
                
                    texture_baker = DiffTextureBaker(mesh, training_iterations, batch_size, texture_learning_rate, train_mesh_geometry, geometry_learning_rate, ms_ssim_loss_weight, force_cuda_rasterize)
                    
                    texture_baker.prepare_training(reference_images, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy)
                    texture_baker.training()
                    
                    trained_mesh, baked_texture = texture_baker.get_mesh_and_texture()
                    
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
            dmtet_mesh = DMTetMesh(training_iterations, 
                                points_cloud_fitting_weight, 
                                mesh_smoothing_weight, 
                                chamfer_faces_sample_scale, 
                                mesh_scale, 
                                grid_resolution, 
                                geometry_learning_rate, 
                                positional_encoding_multires, 
                                mlp_internal_dims, 
                                mlp_hidden_layer_num)
            
            if reference_points_cloud is None and reference_images is None:
                cstr(f"[{self.__class__.__name__}] reference_points_cloud and reference_images cannot both be None, you need at least provide one of them!").error.print()
                raise Exception("User didn't provide necessary inputs, stop executing the code!")
            
            dmtet_mesh.training(reference_points_cloud, reference_images, reference_masks)
            mesh = dmtet_mesh.get_mesh()
            return (mesh, )
        
TGS_CONFIG_PATH = "configs/tgs_config.yaml"

class Load_Triplane_Gaussian_Transformers:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },

            "optional": {
            }
        }
    
    RETURN_TYPES = (
        "TGS_MODEL",
    )
    RETURN_NAMES = (
        "tgs_model",
    )
    FUNCTION = "load_TGS"
    CATEGORY = "Comfy3D/Import|Export"
    
    def load_TGS(self):

        device = get_device()
        
        config_path = os.path.join(ROOT_PATH, TGS_CONFIG_PATH)
        cfg: ExperimentConfig = load_config(config_path)
        
        # Download pre-trained model if it not exist locally
        model_name = "model_lvis_rel.ckpt"
        model_path = os.path.join(ROOT_PATH, f"checkpoints/{model_name}")
        if not os.path.isfile(model_path):
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id="VAST-AI/TriplaneGaussian", local_dir="./checkpoints", filename=model_name, repo_type="model")
        
        cfg.system.weights=model_path
        tgs_model = TGS(cfg=cfg.system).to(device)
        
        save_path = os.path.join(ROOT_PATH, "outputs")
        tgs_model.set_save_dir(save_path)
        
        cstr(f"[{self.__class__.__name__}] loaded model ckpt from {model_path}").msg.print()

        return (tgs_model, )
    
class Triplane_Gaussian_Transformers:
    
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
    
    OUTPUT_NODE = True
    RETURN_TYPES = (
        "GS_PLY",
    )
    RETURN_NAMES = (
        "gs_ply",
    )
    FUNCTION = "run_TGS"
    CATEGORY = "Comfy3D/Algorithm"
    
    def run_TGS(self, reference_image, reference_mask, tgs_model, cam_dist):        
        config_path = os.path.join(ROOT_PATH, TGS_CONFIG_PATH)
        cfg: ExperimentConfig = load_config(config_path)
        
        #save_path = os.path.join(ROOT_PATH, "outputs")
        #tgs_model.set_save_dir(save_path)

        cfg.data.cond_camera_distance = cam_dist
        cfg.data.eval_camera_distance = cam_dist
        dataset = CustomImageOrbitDataset(reference_image, reference_mask, cfg.data)
        dataloader = DataLoader(dataset,
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
        
        return (gs_ply[0],)
        