# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys, os

# Absolute path to ComfyUI root (where main.py or execution.py is)
comfy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Add ComfyUI/custom_nodes to sys.path
custom_nodes_path = os.path.join(comfy_root, "custom_nodes")
if custom_nodes_path not in sys.path:
    sys.path.insert(0, custom_nodes_path)


import os
import torch
import rembg
import numpy as np
from pathlib import PureWindowsPath
from .PartPacker.flow.configs.schema import ModelConfig
from .PartPacker.flow.model import Model
from .PartPacker.app import process_3d
from .node_utils import tensor2cv,gc_clear,add_mask,tensor2pil_upscale
from comfy_extras.nodes_hunyuan3d import MESH


import folder_paths


MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))


device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

# add checkpoints dir
PartPacker_Weigths_Path = os.path.join(folder_paths.models_dir, "PartPacker")
if not os.path.exists(PartPacker_Weigths_Path):
    os.makedirs(PartPacker_Weigths_Path)
folder_paths.add_model_folder_path("PartPacker", PartPacker_Weigths_Path)


class PartPacker_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint": (["none"] + [i for i in folder_paths.get_filename_list("PartPacker") if i.endswith(".pt")],),
                "vae":  (folder_paths.get_filename_list("vae"),),
                "dino":("STRING", { "default": "facebook/dinov2-giant"}),
                "cpu_offload":  ("BOOLEAN", {"default": True},),             
            },
        }

    RETURN_TYPES = ("PartPacker_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "PartPacker"

    def loader_main(self, checkpoint,vae,dino,cpu_offload,):
        if checkpoint == "none":
            raise ValueError("No checkpoint selected")
        
        flow_ckpt_path=folder_paths.get_full_path("PartPacker", checkpoint)
        vae_ckpt_path=folder_paths.get_full_path("vae", vae)

        # load model
        print("***********Load model ***********")
        TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)
        bg_remover = rembg.new_session()
        if not dino:
            raise ValueError("No dino model path fill")
        else:
            if dino.count('/')!=1:
                dino = PureWindowsPath(dino).as_posix()
        dino_model="dinov2_vitg14" if "giant" in dino.lower() else "dinov2_vitl14_reg"

        # model config
        model_config = ModelConfig(
            vae_conf="ComfyUI_3D_Pack.PartPacker.PartPacker.vae.configs.part_woenc",
            vae_ckpt_path=vae_ckpt_path,
            qknorm=True,
            qknorm_type="RMSNorm",
            use_pos_embed=False,
            dino_model=dino_model,
            hidden_dim=1536,
            flow_shift=3.0,
            logitnorm_mean=1.0,
            logitnorm_std=1.0,
            latent_size=4096,
            use_parts=True,
        )

        # instantiate model
        # model = Model(model_config,device,dino,cpu_offload=cpu_offload).eval().cuda().bfloat16()
        model = Model(model_config, device, dino, cpu_offload=cpu_offload).eval()
        if not cpu_offload:
            model = model.cuda().bfloat16()

        # load weight
        ckpt_dict = torch.load(flow_ckpt_path, weights_only=True)
        model.load_state_dict(ckpt_dict, strict=True)

        print("***********Load model done ***********")
        gc_clear()
        return ({"pipe": model, "bg_remover": bg_remover, "TRIMESH_GLB_EXPORT": TRIMESH_GLB_EXPORT},)



class PartPacker_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("PartPacker_MODEL",),
                "image": ("IMAGE",),# BHWC
                "target_num_faces" : ("INT", {"default": 100000, "min": 1000, "max": MAX_SEED, "step": 1}),
                "grid_res": ("INT", {"default": 384, "min": 128, "max": 2048, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 50, "min": 3, "max": 1024, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 1, "max": 20, "step": 0.1}),
                "simplify_mesh":  ("BOOLEAN", {"default": False},),},
            "optional":{ "mask": ("MASK",),   # B H W 默认的mask是遮罩区黑色，而非传统的白色
                            }
            }

    RETURN_TYPES = ("TRIMESH","MESH","STRING",)
    RETURN_NAMES = ("trimesh","mesh","model_path", )
    FUNCTION = "sampler_main"
    CATEGORY = "PartPacker"

    def sampler_main(self, model,image,target_num_faces,grid_res,seed,steps,cfg_scale,simplify_mesh,**kwargs):
        if isinstance(kwargs.get("mask"),torch.Tensor):
            mask=add_mask(kwargs.get("mask"),image)
            mask=tensor2cv(mask)
        else:
            mask=None
        input_image=tensor2cv(image,RGB2BGR=False)
        trimesh,model_path=process_3d(model.get("pipe"),model.get("bg_remover"),input_image,model.get("TRIMESH_GLB_EXPORT"), mask,folder_paths.get_output_directory(),num_steps=steps, cfg_scale=cfg_scale, grid_res=grid_res, seed=seed, simplify_mesh=simplify_mesh, target_num_faces=target_num_faces)
        gc_clear()
       
        return (trimesh,MESH(torch.tensor(trimesh.vertices, dtype=torch.float32).unsqueeze(0),torch.tensor(trimesh.faces, dtype=torch.long).unsqueeze(0)),model_path,)


# WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "PartPacker_Loader": PartPacker_Loader,
    "PartPacker_Sampler": PartPacker_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PartPacker_Loader": "PartPacker_Loader",
    "PartPacker_Sampler": "PartPacker_Sampler",
}
