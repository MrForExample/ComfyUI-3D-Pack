import os
import re
import math
import copy
from enum import Enum
import folder_paths as comfy_paths

import torch
import numpy as np

from .diff_rast.diff_texturing import DiffTextureBaker
from .shared_utils.common_utils import cstr
from .mesh_processer.mesh import Mesh
from .gaussian_splatting.main_3DGS import GaussianSplatting, GSParams

MANIFEST = {
    "name": "ComfyUI-3D-Pack",
    "version": (0,0,1),
    "author": "Mr. For Example",
    "project": "https://github.com/MrForExample/ComfyUI-3D-Pack",
    "description": "An extensive node suite that enables ComfyUI to process 3D inputs (Mesh & UV Texture, etc) using cutting edge algorithms (3DGS, NeRF, etc.)",
}

SUPPORTED_3D_EXTENSIONS = (
    '.obj',
    '.ply',
    '.glb',
)

ELEVATION_MIN = -90
ELEVATION_MAX = 90.0
AZIMUTH_MIN = -180.0
AZIMUTH_MAX = 180.0

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
    CATEGORY = "ComfyUI3D/Import|Export"
    
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
    
class Save_3D_Mesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "mesh_save_path": ("STRING", {"default": 'Mesh_[time(%Y-%m-%d)].obj', "multiline": False}),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_mesh"
    CATEGORY = "ComfyUI3D/Import|Export"
    
    def save_mesh(self, mesh, mesh_save_path):
        
        mesh_folder_path, filename = os.path.split(mesh_save_path)
        
        if not os.path.isabs(mesh_save_path):
            mesh_folder_path = os.path.join(comfy_paths.output_directory, mesh_folder_path)
            mesh_save_path = os.path.join(mesh_folder_path, filename)
        
        os.makedirs(mesh_folder_path, exist_ok=True)
        
        if filename.lower().endswith(SUPPORTED_3D_EXTENSIONS):
            mesh.write(mesh_save_path)
            cstr(f"[{self.__class__.__name__}] saved model to {mesh_save_path}").msg.print()
        else:
            cstr(f"File name {filename} does not end with supported 3D file extensions: {SUPPORTED_3D_EXTENSIONS}").error.print()
        
        return ()
    
class Save_3DGS:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_3DGS": ("GS_RAW",),
                "save_path": ("STRING", {"default": '3DGS_[time(%Y-%m-%d)].ply', "multiline": False}),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_3DGS"
    CATEGORY = "ComfyUI3D/Import|Export"
    
    def save_3DGS(self, raw_3DGS, save_path):
        
        folder_path, filename = os.path.split(save_path)
        
        if not os.path.isabs(save_path):
            folder_path = os.path.join(comfy_paths.output_directory, folder_path)
            save_path = os.path.join(folder_path, filename)
        
        os.makedirs(folder_path, exist_ok=True)
        
        raw_3DGS.renderer.gaussians.save_ply(save_path)
        
        return ()
    
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
    CATEGORY = "ComfyUI3D/Preprocessor"
    
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
    CATEGORY = "ComfyUI3D/Preprocessor"
    
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
                "batch_size": ("INT", {"default": 5, "min": 1, "max": 0xffffffffffffffff}),
                "loss_value_scale": ("FLOAT", {"default": 10000.0, }),
                "ms_ssim_loss_weight": ("FLOAT", {"default": 0.2, }),
                "alpha_loss_weight": ("FLOAT", {"default": 3, }),
                "offset_loss_weight": ("FLOAT", {"default": 0.0, }),
                "offset_opacity_loss_weight": ("FLOAT", {"default": 0.0, }),
                "invert_background_probability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "feature_learning_rate": ("FLOAT", {"default": 0.01,  }),
                "opacity_learning_rate": ("FLOAT", {"default": 0.05,  }),
                "scaling_learning_rate": ("FLOAT", {"default": 0.005,  }),
                "rotation_learning_rate": ("FLOAT", {"default": 0.005,  }),
                "position_learning_rate_init": ("FLOAT", {"default": 0.001,  }),
                "position_learning_rate_final": ("FLOAT", {"default": 0.00002, }),
                "position_learning_rate_delay_mult": ("FLOAT", {"default": 0.02, }),
                "position_learning_rate_max_steps": ("INT", {"default": 500, }),
                "initial_gaussians_num": ("INT", {"default": 5000, }),
                "K_nearest_neighbors": ("INT", {"default": 3, }),
                "percent_dense": ("FLOAT", {"default": 0.01, }),
                "density_start_iterations": ("INT", {"default": 100, }),
                "density_end_iterations": ("INT", {"default": 100000, }),
                "densification_interval": ("INT", {"default": 100, }),
                "opacity_reset_interval": ("INT", {"default": 700, }),
                "densify_grad_threshold": ("FLOAT", {"default": 0.01 }),
                "gaussian_sh_degree": ("INT", {"default": 0, }),
            },
            
            "optional": {
                "mesh_to_initialize_gaussian": ("MESH",),
            }
        }

    RETURN_TYPES = (
        "GS_RAW",
    )
    RETURN_NAMES = (
        "raw_3DGS",
    )
    FUNCTION = "run_3DGS"
    CATEGORY = "ComfyUI3D/Algorithm"
    
    def run_3DGS(self,
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
                 mesh_to_initialize_gaussian=None):
        
        
        raw_3DGS = None
        
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
                    
                    gs = GaussianSplatting(reference_images, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy, gs_params, mesh_to_initialize_gaussian)
                    
                    gs.training()

                    raw_3DGS = gs
                
            else:
                cstr(f"[{self.__class__.__name__}] Number of reference images {ref_imgs_num} does not equal to number of reference camera poses {ref_cam_poses_num}").error.print()
        else:
            cstr(f"[{self.__class__.__name__}] Number of reference images {ref_imgs_num} does not equal to number of masks {ref_masks_num}").error.print()

        return (raw_3DGS, )
    
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
                "training_iterations": ("INT", {"default": 1000, "min": 1, "max": 100000}),
                "batch_size": ("INT", {"default": 5, "min": 1, "max": 0xffffffffffffffff}),
                "texture_learning_rate": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "train_mesh_geometry": ("BOOLEAN", {"default": False},),
                "geometry_learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.1, "step": 0.001}),
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
    CATEGORY = "ComfyUI3D/Algorithm"
    
    def bake_texture(self, reference_images, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy, mesh, 
                     training_iterations, batch_size, texture_learning_rate, train_mesh_geometry, geometry_learning_rate, ms_ssim_loss_weight, force_cuda_rasterize):
        
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
                
                    texture_baker = DiffTextureBaker(reference_images, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy, mesh, 
                                        training_iterations, batch_size, texture_learning_rate, train_mesh_geometry, geometry_learning_rate, ms_ssim_loss_weight, force_cuda_rasterize)
                    
                    texture_baker.training()
                    
                    trained_mesh, baked_texture = texture_baker.get_mesh_and_texture()
                    
            else:
                cstr(f"[{self.__class__.__name__}] Number of reference images {ref_imgs_num} does not equal to number of reference camera poses {ref_cam_poses_num}").error.print()
        else:
            cstr(f"[{self.__class__.__name__}] Number of reference images {ref_imgs_num} does not equal to number of masks {ref_masks_num}").error.print()
        
        return (trained_mesh, baked_texture, )