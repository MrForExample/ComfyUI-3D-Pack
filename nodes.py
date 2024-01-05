import nodes
import torch
import torch.nn.functional as F
import numpy as np
import os
import math

from pytorch_msssim import SSIM, MS_SSIM

import diff_rast.diff_texturing
from shared_utils.common_utils import cstr
from mesh_processer.mesh import Mesh

MANIFEST = {
    "name": "ComfyUI-3D-Pack",
    "version": (0,0,1),
    "author": "Mr. For Example",
    "project": "https://github.com/MrForExample/ComfyUI-3D-Pack",
    "description": "An extensive node suite that enables ComfyUI to process 3D inputs (Mesh & UV Texture, etc) using cutting edge algorithms (3DGS, NeRF, etc.)",
}

class Load3DFile:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file_path": ("STRING", {"default": '', "multiline": False}),
                "resize":  ("BOOLEAN", {"default": False},),
                "renormal":  ("BOOLEAN", {"default": True},),
                "retex":  ("BOOLEAN", {"default": False},),
            },
        }

    RETURN_TYPES = (
        "MESH",
    )
    RETURN_NAMES = (
        "mesh",
    )
    FUNCTION = "load_mesh"
    CATEGORY = "ComfyUI-3D"
    
    SUPPORTED_3D_EXTENSIONS = (
        '.obj',
        '.ply',
        '.glb',
    )
    
    def load_mesh(self, mesh_file_path, resize, renormal, retex):
        mesh = None
        
        if os.path.exists(mesh_file_path):
            folder, file = os.path.split(self.filepath)
            if file.lower().endswith(self.SUPPORTED_3D_EXTENSIONS):
                mesh = Mesh.load(self.opt.mesh, resize, renormal, retex)
            else:
                cstr(f"File name {file} does not end with supported 3D file extensions: {self.SUPPORTED_3D_EXTENSIONS}").error.print()
        else:        
            cstr(f"File {mesh_file_path} does not exist").error.print()
        return (mesh, )

class GaussianSplatting:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE",), 
            },
        }

    RETURN_TYPES = (
        "GS_RAW",
    )
    RETURN_NAMES = (
        "raw_3DGS",
    )
    FUNCTION = "run_3DGS"
    CATEGORY = "ComfyUI-3D"
    
    def run_3DGS(self, reference_images):
        raw_3DGS = None

        return (raw_3DGS, )
    
class BakeTextureToMesh:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_images": ("IMAGE",), 
                "mesh": ("MESH")
            },
        }

    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "baked_texture",
    )
    FUNCTION = "bake_texture"
    CATEGORY = "ComfyUI-3D"
    
    def bake_texture(self, reference_images, mesh):
        baked_texture = None

        return (baked_texture, )

NODE_CLASS_MAPPINGS = {
    "Load3DFile": Load3DFile,
    "3DGS": GaussianSplatting,
    "BakeTextureToMesh": BakeTextureToMesh
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Load3DFile": "Load 3D File",
    "3DGS": "3D Gaussian Splatting",
    "BakeTextureToMesh": "Bake Texture To Mesh",
}