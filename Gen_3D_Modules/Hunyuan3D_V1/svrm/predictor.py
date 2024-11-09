# Open Source Model Licensed under the Apache License Version 2.0 and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved. 
# The below software and/or models in this distribution may have been 
# modified by THL A29 Limited ("Tencent Modifications"). 
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT 
# except for the third-party components listed below. 
# Hunyuan 3D does not impose any additional limitations beyond what is outlined 
# in the repsective licenses of these third-party components. 
# Users must comply with all terms and conditions of original licenses of these third-party 
# components and must ensure that the usage of the third party components adheres to 
# all relevant laws and regulations. 

# For avoidance of doubts, Hunyuan 3D means the large language models and 
# their software and algorithms, including trained model weights, parameters (including 
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code, 
# fine-tuning enabling code and other elements of the foregoing made publicly available 
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import math
import time
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageSequence
from omegaconf import OmegaConf
from torchvision import transforms
from safetensors.torch import save_file, load_file
from .ldm.util import instantiate_from_config
from .ldm.vis_util import render

class MV23DPredictor(object):
	def __init__(self, ckpt_path, cfg_path, elevation=15, number_view=60, 
				render_size=256, device="cuda:0") -> None:
		self.device = device
		self.elevation = elevation
		self.number_view = number_view
		self.render_size = render_size

		self.elevation_list = [0, 0, 0, 0, 0, 0, 0]
		self.azimuth_list = [0, 60, 120, 180, 240, 300, 0]

		st = time.time()
		self.model = self.init_model(ckpt_path, cfg_path)
		print(f"=====> mv23d model init time: {time.time() - st}")

		self.input_view_transform = transforms.Compose([
			transforms.Resize(504, interpolation=Image.BICUBIC),
			transforms.ToTensor(),
		])
		self.final_input_view_transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

	def init_model(self, ckpt_path, cfg_path):
		config = OmegaConf.load(cfg_path)
		model = instantiate_from_config(config.model)

		weights = load_file(ckpt_path)
		model.load_state_dict(weights)

		model.to(self.device)
		model = model.eval()
		model.render.half()
		print(f'Load model successfully')
		return model

	def create_camera_to_world_matrix(self, elevation, azimuth, cam_dis=1.5):
		# elevation azimuth are radians
		# Convert elevation and azimuth angles to Cartesian coordinates on a unit sphere
		x = np.cos(elevation) * np.cos(azimuth)
		y = np.cos(elevation) * np.sin(azimuth)
		z = np.sin(elevation)

		# Calculate camera position, target, and up vectors
		camera_pos = np.array([x, y, z]) * cam_dis
		target = np.array([0, 0, 0])
		up = np.array([0, 0, 1])

		# Construct view matrix
		forward = target - camera_pos
		forward /= np.linalg.norm(forward)
		right = np.cross(forward, up)
		right /= np.linalg.norm(right)
		new_up = np.cross(right, forward)
		new_up /= np.linalg.norm(new_up)
		cam2world = np.eye(4)
		cam2world[:3, :3] = np.array([right, new_up, -forward]).T
		cam2world[:3, 3] = camera_pos
		return cam2world

	def refine_mask(self, mask, k=16):
		mask /= 255.0
		boder_mask = (mask >= -math.pi / 2.0 / k + 0.5) & (mask <= math.pi / 2.0 / k + 0.5)
		mask[boder_mask] = 0.5 * np.sin(k * (mask[boder_mask] - 0.5)) + 0.5
		mask[mask < -math.pi / 2.0 / k + 0.5] = 0.0
		mask[mask > math.pi / 2.0 / k + 0.5] = 1.0
		return (mask * 255.0).astype(np.uint8)

	def load_images_and_cameras(self, input_imgs, elevation_list, azimuth_list):
		input_image_list = []
		input_cam_list = []
		for input_view_image, elevation, azimuth in zip(input_imgs, elevation_list, azimuth_list):
			input_view_image = self.input_view_transform(input_view_image)  
			input_image_list.append(input_view_image)

			input_view_cam_pos = self.create_camera_to_world_matrix(np.radians(elevation), np.radians(azimuth))
			input_view_cam_intrinsic = np.array([35. / 32, 35. /32, 0.5, 0.5])
			input_view_cam = torch.from_numpy(
				np.concatenate([input_view_cam_pos.reshape(-1), input_view_cam_intrinsic], 0)
			).float()
			input_cam_list.append(input_view_cam)

		pixels_input = torch.stack(input_image_list, dim=0)
		input_images = self.final_input_view_transform(pixels_input)
		input_cams = torch.stack(input_cam_list, dim=0)
		return input_images, input_cams

	def load_data(self, intput_imgs):
		assert (6+1) == len(intput_imgs)
		
		input_images, input_cams = self.load_images_and_cameras(intput_imgs, self.elevation_list, self.azimuth_list)
		input_cams[-1, :] = 0 # for user input view
		
		data = {}
		data["input_view"] = input_images.unsqueeze(0).to(self.device)    # 1 4 3 512 512
		data["input_view_cam"] = input_cams.unsqueeze(0).to(self.device)  # 1 4 20
		return data

	@torch.no_grad()
	def predict(
		self, 
		intput_imgs, 
		save_dir = "outputs/", 
		image_input = None,
		target_face_count = 10000,
		do_texture_mapping = True,
	):
		
		with torch.amp.autocast('cuda'):
			vtx_refine, faces_refine, vtx_colors = self.model.generate_mesh(
				data = self.load_data(intput_imgs),
				target_face_count = target_face_count,
			)
			
			#self.model.export_mesh_with_uv(
			#	vtx_refine, faces_refine, vtx_colors,
			#	do_texture_mapping = do_texture_mapping,
			#	out_dir = save_dir,
			#)
		
		return vtx_refine, faces_refine, vtx_colors
