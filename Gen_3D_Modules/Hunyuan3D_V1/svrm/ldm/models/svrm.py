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
import time
import math
import cv2
import numpy as np
import itertools
import shutil
from tqdm import tqdm
import torch
import torch.nn.functional as F
from einops import rearrange
try:
	import trimesh
	import mcubes
	import xatlas
except:
	raise "failed to import 3d libraries "

from ..modules.rendering_neus.mesh import Mesh
from ..modules.rendering_neus.rasterize import NVDiffRasterizerContext

from ..utils.ops import scale_tensor
from ..util import count_params, instantiate_from_config
from ..vis_util import render


def unwrap_uv(v_pos, t_pos_idx):
	print("Using xatlas to perform UV unwrapping, may take a while ...")
	atlas = xatlas.Atlas()
	atlas.add_mesh(v_pos, t_pos_idx)
	atlas.generate(xatlas.ChartOptions(), xatlas.PackOptions())
	_, indices, uvs = atlas.get_mesh(0)
	indices = indices.astype(np.int64, casting="same_kind")
	return uvs, indices


def uv_padding(image, hole_mask, uv_padding_size = 2):
	return cv2.inpaint(
		(image.detach().cpu().numpy() * 255).astype(np.uint8),
		(hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
		uv_padding_size,
		cv2.INPAINT_TELEA
	)

def refine_mesh(vtx_refine, faces_refine, device):
	import open3d as o3d

	mesh = o3d.geometry.TriangleMesh(
		vertices=o3d.utility.Vector3dVector(vtx_refine), 
		triangles=o3d.utility.Vector3iVector(faces_refine))

	mesh = mesh.remove_unreferenced_vertices()
	mesh = mesh.remove_duplicated_triangles()
	mesh = mesh.remove_duplicated_vertices()

	voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound())

	mesh = mesh.simplify_vertex_clustering(
		voxel_size=0.007, # 0.005
		contraction=o3d.geometry.SimplificationContraction.Average)

	mesh = mesh.filter_smooth_simple(number_of_iterations=2)

	vtx_refine = np.asarray(mesh.vertices).astype(np.float32)
	faces_refine = np.asarray(mesh.triangles)

	# reduce faces
	if faces_refine.shape[0] > target_face_count:
		print(f"reduce face: {faces_refine.shape[0]} -> {target_face_count}")
		mesh = o3d.geometry.TriangleMesh(
			vertices = o3d.utility.Vector3dVector(vtx_refine),
			triangles = o3d.utility.Vector3iVector(faces_refine)
		)
		
		# Function to simplify mesh using Quadric Error Metric Decimation by Garland and Heckbert
		mesh = mesh.simplify_quadric_decimation(target_face_count, boundary_weight=1.0)

		mesh = Mesh(
			v_pos = torch.from_numpy(np.asarray(mesh.vertices)).to(device),
			t_pos_idx = torch.from_numpy(np.asarray(mesh.triangles)).to(device),
			v_rgb = torch.from_numpy(np.asarray(mesh.vertex_colors)).to(device)
		)
		vtx_refine = mesh.v_pos.cpu().numpy()
		faces_refine = mesh.t_pos_idx.cpu().numpy()
	return vtx_refine, faces_refine


class SVRMModel(torch.nn.Module):
	def __init__(
		self,
		img_encoder_config,
		img_to_triplane_config,
		render_config,
		device = "cuda:0",
		**kwargs
	):
		super().__init__()

		self.img_encoder = instantiate_from_config(img_encoder_config).half()
		self.img_to_triplane_decoder = instantiate_from_config(img_to_triplane_config).half()
		self.render = instantiate_from_config(render_config).half()
		self.device = device
		count_params(self, verbose=True)

	@torch.no_grad()
	def generate_mesh(
		self, 
		data, 
		mesh_size: int = 384, 
		target_face_count = 10000,
	):
		"""
		color_type: 0 for ray texture, 1 for vertices texture
		"""
		st = time.time()
		here = {'device': self.device, 'dtype': torch.float16}
		input_view_image = data["input_view"].to(**here)    # [b, m, c, h, w]
		input_view_cam = data["input_view_cam"].to(**here)  # [b, m, 20]

		batch_size, input_view_num, *_ = input_view_image.shape
		assert batch_size == 1, "batch size should be 1"

		input_view_image = rearrange(input_view_image, 'b m c h w -> (b m) c h w')
		input_view_cam = rearrange(input_view_cam, 'b m d -> (b m) d')
		input_view_feat = self.img_encoder(input_view_image, input_view_cam)
		input_view_feat = rearrange(input_view_feat, '(b m) l d -> b (l m) d', m=input_view_num)

		# -- decoder
		torch.cuda.empty_cache()
		triplane_gen = self.img_to_triplane_decoder(input_view_feat)  # [b, 3, tri_dim, h, w]
		del input_view_feat
		torch.cuda.empty_cache()

		# --- triplane nerf render

		cur_triplane = triplane_gen[0:1]
		
		aabb = torch.tensor([[-0.6, -0.6, -0.6], [0.6, 0.6, 0.6]]).unsqueeze(0).to(**here)
		grid_out = self.render.forward_grid(planes=cur_triplane, grid_size=mesh_size, aabb=aabb)

		print(f"=====> LRM forward time: {time.time() - st}")
		st = time.time()
		
		vtx, faces = mcubes.marching_cubes(0. - grid_out['sdf'].squeeze(0).squeeze(-1).cpu().float().numpy(), 0)
		
		bbox = aabb[0].cpu().numpy()
		vtx = vtx / (mesh_size - 1)  
		vtx = vtx * (bbox[1] - bbox[0]) + bbox[0]

		print(f"=====> generate mesh with marching cube time: {time.time() - st}")

		# refine mesh
		#st = time.time()
		#vtx, faces = refine_mesh(vtx, faces)
		#print(f"=====> refine mesh time: {time.time() - st}")

		vtx_colors = self.render.forward_points(cur_triplane, torch.tensor(vtx).unsqueeze(0).to(**here))
		vtx_colors = vtx_colors['rgb'].float().squeeze(0).cpu().numpy()

		return vtx, faces, vtx_colors
		
	def export_mesh_with_uv(
		self,
		vtx_refine,
		faces_refine,
		vtx_colors,
		ctx = None, 
		context_type = 'cuda', 
		texture_res = 1024,
		do_texture_mapping = True,
		out_dir = 'outputs/test'
	):
		os.makedirs(out_dir, exist_ok=True)

		color_ratio = 0.8 # increase brightness
		with open(f'{out_dir}/mesh_with_colors.obj', 'w') as fid:
			verts = vtx_refine[:, [1,2,0]] 
			for pidx, pp in enumerate(verts):
				color = vtx_colors[pidx]
				color = [color[0]**color_ratio, color[1]**color_ratio, color[2]**color_ratio]
				fid.write('v %f %f %f %f %f %f\n' % (pp[0], pp[1], pp[2], color[0], color[1], color[2]))
			for i, f in enumerate(faces_refine):
				f1 = f + 1
				fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
				
		mesh = trimesh.load_mesh(f'{out_dir}/mesh_with_colors.obj')
		st = time.time()

		if not do_texture_mapping:
			shutil.copy(f'{out_dir}/mesh_with_colors.obj', f'{out_dir}/mesh.obj')
			mesh.export(f'{out_dir}/mesh.glb', file_type='glb')
			return None

		##########  export texture  ########
		st = time.time()
		
		# uv unwrap 
		vtx_tex, t_tex_idx = unwrap_uv(vtx_refine, faces_refine)      
		vtx_refine   = torch.from_numpy(vtx_refine).to(self.device)   
		faces_refine = torch.from_numpy(faces_refine).to(self.device)  
		t_tex_idx    = torch.from_numpy(t_tex_idx).to(self.device)    
		uv_clip      = torch.from_numpy(vtx_tex * 2.0 - 1.0).to(self.device) 

		# rasterize
		ctx = NVDiffRasterizerContext(context_type, cur_triplane.device) if ctx is None else ctx
		rast = ctx.rasterize_one(
			torch.cat([
				uv_clip, 
				torch.zeros_like(uv_clip[..., 0:1]), 
				torch.ones_like(uv_clip[..., 0:1])
			], dim=-1), 
			t_tex_idx,  
			(texture_res, texture_res)
		)[0]
		hole_mask = ~(rast[:, :, 3] > 0)

		# Interpolate world space position
		gb_pos = ctx.interpolate_one(vtx_refine, rast[None, ...], faces_refine)[0][0]
		with torch.no_grad():
			gb_mask_pos_scale = scale_tensor(gb_pos.unsqueeze(0).view(1, -1, 3), (-1, 1), (-1, 1))
			tex_map = self.render.forward_points(cur_triplane, gb_mask_pos_scale)['rgb']
			tex_map = tex_map.float().squeeze(0)  # (0, 1)
			tex_map = tex_map.view((texture_res, texture_res, 3)) 
			img = uv_padding(tex_map, hole_mask)
			img = ((img/255.0) ** color_ratio) * 255  # increase brightness
			img = img.clip(0, 255).astype(np.uint8)
	
		verts = vtx_refine.cpu().numpy()[:, [1,2,0]] 
		faces = faces_refine.cpu().numpy()

		with open(f'{out_dir}/texture.mtl', 'w') as fid:
			fid.write('newmtl material_0\n')
			fid.write("Ka 1.000 1.000 1.000\n")
			fid.write("Kd 1.000 1.000 1.000\n")
			fid.write("Ks 0.000 0.000 0.000\n")
			fid.write("d 1.0\n")
			fid.write("illum 2\n")
			fid.write(f'map_Kd texture.png\n')
		
		with open(f'{out_dir}/mesh.obj', 'w') as fid:
			fid.write(f'mtllib texture.mtl\n')
			for pidx, pp in enumerate(verts):
				fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))
			for pidx, pp in enumerate(vtx_tex): 
				fid.write('vt %f %f\n' % (pp[0], 1 - pp[1]))
			fid.write('usemtl material_0\n')
			for i, f in enumerate(faces):
				f1 = f + 1
				f2 = t_tex_idx[i] + 1
				fid.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2],))

		cv2.imwrite(f'{out_dir}/texture.png', img[..., [2, 1, 0]])   
		mesh = trimesh.load_mesh(f'{out_dir}/mesh.obj')
		mesh.export(f'{out_dir}/mesh.glb', file_type='glb')
		print(f"=====> generate mesh with texture shading time: {time.time() - st}")
  