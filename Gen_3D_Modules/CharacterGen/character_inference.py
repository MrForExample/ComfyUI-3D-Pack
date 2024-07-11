import sys
import os
from os.path import dirname
from typing import Dict, List
import numpy as np

import torch
import torch.utils.checkpoint

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision import transforms

from CharacterGen.Stage_2D.tuneavideo.models.unet_mv2d_condition import UNetMV2DConditionModel
from CharacterGen.Stage_2D.tuneavideo.models.unet_mv2d_ref import UNetMV2DRefModel
from CharacterGen.Stage_2D.tuneavideo.models.PoseGuider import PoseGuider
from CharacterGen.Stage_2D.tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline

from einops import rearrange
from PIL import Image
import json

import trimesh
from CharacterGen.Stage_3D import lrm
from datetime import datetime
from pygltflib import GLTF2
import pymeshlab

CHARACTER_GEN_ROOT_ABS_PATH = dirname(__file__)
CHARACTER_GEN_2D_DATA_ABS_PATH = os.path.join(CHARACTER_GEN_ROOT_ABS_PATH, "Stage_2D/material")
CHARACTER_GEN_3D_ABS_PATH = os.path.join(CHARACTER_GEN_ROOT_ABS_PATH, "Stage_3D")
CHARACTER_GEN_3D_DATA_ABS_PATH = os.path.join(CHARACTER_GEN_3D_ABS_PATH, "material")

sys.path.append(CHARACTER_GEN_3D_ABS_PATH)

def get_bg_color(bg_color):
    if bg_color == 'white':
        bg_color = np.array([1., 1., 1.], dtype=np.float32)
    elif bg_color == 'black':
        bg_color = np.array([0., 0., 0.], dtype=np.float32)
    elif bg_color == 'gray':
        bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    elif bg_color == 'random':
        bg_color = np.random.rand(3)
    elif isinstance(bg_color, float):
        bg_color = np.array([bg_color] * 3, dtype=np.float32)
    else:
        raise NotImplementedError
    return bg_color

def process_image(image, totensor, w=512, h=768):
    if not image.mode == "RGBA":
        image = image.convert("RGBA")

    # Find non-transparent pixels
    non_transparent = np.nonzero(np.array(image)[..., 3])
    min_x, max_x = non_transparent[1].min(), non_transparent[1].max()
    min_y, max_y = non_transparent[0].min(), non_transparent[0].max()    
    image = image.crop((min_x, min_y, max_x, max_y))

    # paste to center
    max_dim = max(image.width, image.height)
    max_height = max_dim
    max_width = int(max_dim / 3 * 2)
    new_image = Image.new("RGBA", (max_width, max_height))
    left = (max_width - image.width) // 2
    top = (max_height - image.height) // 2
    new_image.paste(image, (left, top))

    image = new_image.resize((w, h), resample=Image.Resampling.BICUBIC)
    image = np.array(image)
    image = image.astype(np.float32) / 255.
    assert image.shape[-1] == 4  # RGBA
    alpha = image[..., 3:4]
    bg_color = get_bg_color("gray")
    image = image[..., :3] * alpha + bg_color * (1 - alpha)
    # save image
    new_image = Image.fromarray((image * 255).astype(np.uint8))
    return totensor(image)

class Inference2D_API:

    def __init__(self,
            pretrained_model_path: str,
            checkpoint_root_path: str,
            image_encoder_path: str,
            ckpt_dir: str,
            validation: Dict,
            local_crossattn: bool = True,
            unet_from_pretrained_kwargs=None,
            unet_condition_type=None,
            use_pose_guider=False,
            use_shifted_noise=False,
            use_noise=True,
            device="cuda",
            weight_dtype=torch.float16
    ):
        self.validation = validation
        self.use_noise = use_noise
        self.use_shifted_noise = use_shifted_noise
        self.unet_condition_type = unet_condition_type
        self.device = device
        self.weight_dtype = weight_dtype
        image_encoder_path = os.path.join(checkpoint_root_path, image_encoder_path)
        ckpt_dir = os.path.join(checkpoint_root_path, ckpt_dir)

        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
        feature_extractor = CLIPImageProcessor()
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        unet = UNetMV2DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", local_crossattn=local_crossattn, **unet_from_pretrained_kwargs)
        ref_unet = UNetMV2DRefModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", local_crossattn=local_crossattn, **unet_from_pretrained_kwargs)
        if use_pose_guider:
            pose_guider = PoseGuider(noise_latent_channels=4).to(device)
        else:
            pose_guider = None

        unet_params = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
        if use_pose_guider:
            pose_guider_params = torch.load(os.path.join(ckpt_dir, "pytorch_model_1.bin"), map_location="cpu")
            ref_unet_params = torch.load(os.path.join(ckpt_dir, "pytorch_model_2.bin"), map_location="cpu")
            pose_guider.load_state_dict(pose_guider_params)
        else:
            ref_unet_params = torch.load(os.path.join(ckpt_dir, "pytorch_model_1.bin"), map_location="cpu")
        unet.load_state_dict(unet_params)
        ref_unet.load_state_dict(ref_unet_params)

        text_encoder.to(device, dtype=weight_dtype)
        image_encoder.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        ref_unet.to(device, dtype=weight_dtype)
        unet.to(device, dtype=weight_dtype)

        vae.requires_grad_(False)
        unet.requires_grad_(False)
        ref_unet.requires_grad_(False)

        noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
        self.validation_pipeline = TuneAVideoPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=self.tokenizer, unet=unet, ref_unet=ref_unet,feature_extractor=feature_extractor,image_encoder=image_encoder,
            scheduler=noise_scheduler
        )
        self.validation_pipeline.enable_vae_slicing()
        self.validation_pipeline.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def inference(
        self, input_image, val_width, val_height, prompt="high quality, best quality", prompt_neg=None, 
        guidance_scale=7.5, num_inference_steps=40, seed=100, use_shifted_noise=False, crop=False
    ):
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        totensor = transforms.ToTensor()

        metas = json.load(open(os.path.join(CHARACTER_GEN_2D_DATA_ABS_PATH, "pose.json"), "r"))
        cameras = []
        pose_images = []
        for lm in metas:
            cameras.append(torch.tensor(np.array(lm[0]).reshape(4, 4).transpose(1,0)[:3, :4]).reshape(-1))
            if not crop:
                pose_images.append(totensor(np.asarray(Image.open(os.path.join(CHARACTER_GEN_2D_DATA_ABS_PATH, lm[1])).resize(
                    (val_height, val_width), resample=Image.Resampling.BICUBIC)).astype(np.float32) / 255.))
            else:
                pose_image = Image.open(os.path.join(CHARACTER_GEN_2D_DATA_ABS_PATH, lm[1]))
                crop_area = (128, 0, 640, 768)
                pose_images.append(totensor(np.array(pose_image.crop(crop_area)).astype(np.float32)) / 255.)
        camera_matrixs = torch.stack(cameras).unsqueeze(0).to(self.device)
        pose_imgs_in = torch.stack(pose_images).to(self.device)
        prompt_ids = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids[0]

        # (B*Nv, 3, H, W) B==1
        imgs_in = process_image(input_image, totensor, val_width, val_height)
        imgs_in = rearrange(imgs_in.unsqueeze(0).unsqueeze(0), "B Nv C H W -> (B Nv) C H W")
                
        with torch.autocast(self.device, dtype=self.weight_dtype):
            imgs_in = imgs_in.to(self.device)
            # B*Nv images
            out = self.validation_pipeline(prompt=prompt, negative_prompt=prompt_neg, image=imgs_in.to(self.weight_dtype), generator=generator, 
                                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                                        camera_matrixs=camera_matrixs.to(self.weight_dtype), prompt_ids=prompt_ids, 
                                        height=val_height, width=val_width, unet_condition_type=self.unet_condition_type, 
                                        pose_guider=None, pose_image=pose_imgs_in, use_noise=self.use_noise, 
                                        use_shifted_noise=use_shifted_noise, **self.validation).videos
            out = rearrange(out, "B C f H W -> (B f) H W C", f=self.validation.video_length)

        torch.cuda.empty_cache()
        return out

class Inference3D_API:

    def __init__(self, checkpoint_root_path, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.cfg.system.weights = os.path.join(checkpoint_root_path, self.cfg.system.weights)
        self.cfg.system.image_tokenizer.pretrained_model_name_or_path = os.path.join(checkpoint_root_path, self.cfg.system.image_tokenizer.pretrained_model_name_or_path)
        self.cfg.system.renderer.tet_dir = os.path.join(CHARACTER_GEN_3D_DATA_ABS_PATH, self.cfg.system.renderer.tet_dir)
        self.system = lrm.find(self.cfg.system_cls)(self.cfg.system).to(self.device)
        self.system.eval()

    def inference(self, mv_imgs: List[Image.Image]):
        meta = json.load(open(os.path.join(CHARACTER_GEN_3D_DATA_ABS_PATH, "meta.json")))
        c2w_cond = [np.array(loc["transform_matrix"]) for loc in meta["locations"]]
        c2w_cond = torch.from_numpy(np.stack(c2w_cond, axis=0)).float()[None].to(self.device)

        rgb_cond = []
        new_images = []
        for pil_img in mv_imgs:
            image = np.array(pil_img)
            image = Image.fromarray(image)
            if image.width != image.height:
                max_dim = max(image.width, image.height)
                new_image = Image.new("RGBA", (max_dim, max_dim))
                left = (max_dim - image.width) // 2
                top = (max_dim - image.height) // 2
                new_image.paste(image, (left, top))
                image = new_image

            image = image.convert('RGB')
            rgb = image.resize((self.cfg.data.cond_width, self.cfg.data.cond_height), Image.Resampling.BILINEAR)
            rgb = np.array(rgb).astype(np.float32) / 255.0
            new_images.append(image)
            rgb_cond.append(rgb)
        assert len(rgb_cond) == 4, "Please provide 4 images"

        rgb_cond = torch.from_numpy(np.stack(rgb_cond, axis=0)).float()[None].to(self.device)

        scene_code = self.system({"rgb_cond": rgb_cond, "c2w_cond": c2w_cond})[0]
        mesh = self.system.get_geometry(scene_code)
        
        # Flip mesh horizontally
        #mesh.v_pos[:, 0] = -mesh.v_pos[:, 0]
            
        return mesh.v_pos.detach(), mesh.t_pos_idx.detach()
