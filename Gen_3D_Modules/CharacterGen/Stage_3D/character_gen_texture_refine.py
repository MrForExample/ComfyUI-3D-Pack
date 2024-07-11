import os
import numpy as np
import torch
import nvdiffrast.torch as dr
import json
import torch.nn.functional as F
from PIL import Image
import pymeshlab
import cv2

def back_to_texture(glctx, look_at, pos, tri, tex, uv, uv_idx, idx, vn):
    rast_out, rast_out_db = dr.rasterize(glctx, pos, tri, resolution=[tex.shape[0],tex.shape[1]])
    gb_normal, _ = dr.interpolate(vn[None], rast_out, tri)
    gb_normal = F.normalize(gb_normal, dim=-1)
    if idx == 2 or idx == 0:
        filter_camera = [torch.tensor([[[[1,0.,0.]]]]).cuda(), torch.tensor([[[[-1,0.,0.]]]]).cuda()]
    else:
        filter_camera = [torch.tensor([[[[0,-1.,0.]]]]).cuda(), torch.tensor([[[[0,1.,0.]]]]).cuda()]
    nmasks = []
    for fc in filter_camera:
        nmasks.append(((gb_normal * fc) > 0.75).int().sum(keepdim=True, dim=-1))
    gb_normal_mask = 1 - (nmasks[0] | nmasks[1])
   #Image.fromarray(np.clip(gb_normal_mask[0,...,0].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)).save(f"mask_normal_{idx}.png")
    gb_mask = rast_out[...,3:4] > 0
    tri_list = torch.unique(rast_out[...,3:4].reshape(-1))
    tri_list = (tri_list[1:] - 1).to(torch.int32)
    pos = pos[0]

    depth_map = rast_out[...,3:4].clone()
    depth_map[depth_map > 0] = 1
    depth_map = depth_map.to(torch.float32)
    dmax = (rast_out[...,2:3] * gb_mask).max()
    uv = torch.cat([uv * 2 - 1, torch.zeros(uv.shape[0], 1).cuda(), torch.ones(uv.shape[0], 1).cuda()], dim=1).unsqueeze(0)
    uv_idx = uv_idx[tri_list.to(torch.long)]
    rast_uv, rast_uv_db = dr.rasterize(glctx, uv, uv_idx, resolution=(1024, 1024))
    pos_clip = torch.cat([pos[...,:2], pos[...,3:]], -1)
    pos_2d, _ = dr.interpolate(pos_clip, rast_uv, tri[tri_list.to(torch.long)]) # pos (x, y, z, w)
    pos_coord = (pos_2d[...,:2] / (pos_2d[...,2:3] + 1e-6) + 1) / 2.
    texture_mask = (rast_uv[...,3:4] > 0).int()
    color = dr.texture(tex[None, ...] * gb_normal_mask, pos_coord, filter_mode='linear')
    color_mask = dr.texture(gb_normal_mask.to(torch.float32), pos_coord, filter_mode='linear')
    color_mask[color_mask > 0.82] = 1
    color_mask[color_mask <= 0.82] = 0
    color_mask = color_mask.to(torch.int32)
   #Image.fromarray(np.clip(color_mask[0].repeat(1,1,3).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)).save(f"depth_{idx}.png")
    texture_mask = texture_mask * color_mask
   #Image.fromarray(np.clip(color[0].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)).save(f"{idx}.png")
   #Image.fromarray(np.clip(texture_mask[0].repeat(1,1,3).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)).convert("RGB").save(f"mask-{idx}.png")
    return color, texture_mask, rast_uv

def perspective(fovy=0.6913, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    return torch.tensor([[1/(y*aspect),    0,            0,              0], 
                         [           0, 1/-y,            0,              0], 
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [           0,    0,           -1,              0]]).to(torch.float32).cuda()

def rec_mvp(trans, h, w):
    mv = trans
    fov = 40. / 180. * np.pi
    proj = perspective(fov, h / w, n=0.1, f=1000)
    mvp = proj @ mv
    return mvp

def aggregate_texture(kd_map, textures, texture_masks, rast_uvs):
    texture = torch.zeros_like(textures[0])
    texture_mask = torch.zeros_like(texture_masks[0])
    ctex = []
    for idx in range(len(textures)):
        ctex.append(textures[idx] * texture_masks[idx] + 10 * (1 - texture_masks[idx]))
    cat_textures = torch.stack(ctex, dim=-2)
    dis_measure = (cat_textures - kd_map.unsqueeze(-2)).abs().sum(-1)
    _, choose_idx = dis_measure.min(-1)

    choose_idx = choose_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, 3)
    final_texture_map = torch.gather(cat_textures, 3, choose_idx).squeeze(-2)
    #cv2.imwrite("final_texture_map.png", cv2.cvtColor((final_texture_map[0].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    #cv2.imwrite("final_texture_mask.png", (texture_mask[0].cpu().numpy() * 255).astype(np.uint8))
    zero_mask = (final_texture_map.max(dim=-1, keepdim=True)[0] > 0.1)
    close_mask = ((final_texture_map[0] - kd_map).abs().sum(dim=-1, keepdim=True) < 1.0).int()
    for idx in range(len(textures)):
        texture += textures[idx] * texture_masks[idx]
        texture_mask |= texture_masks[idx]
    texture_mask = texture_mask * zero_mask * close_mask[None]
    optimize_mask = (texture_mask == 0).int()

   #import pdb; pdb.set_trace()
   #mask = (texture_mask[0].cpu().numpy() * 255).astype(np.uint8)
   #cv2.imwrite("mask.png", mask)
   #kernel = np.ones((5,5), np.uint8)
   #dilated = cv2.dilate(mask, kernel, iterations=1)
   #cv2.imwrite("di_mask.png", dilated)
   #texture_mask[0] = torch.from_numpy(dilated).unsqueeze(-1).to(torch.float32) / 255.

    final_texture_map = final_texture_map[0] * texture_mask[0]
    Image.fromarray(np.rint(final_texture_map.cpu().numpy() * 255).astype(np.uint8)).save(f"final_texture.png")

   #cv2.imwrite("kd_map.png", cv2.cvtColor((kd_map.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
   #cv2.imwrite("texture_map.png", cv2.cvtColor((final_texture_map.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
   #result = cv2.seamlessClone((final_texture_map.cpu().numpy() * 255).astype(np.uint8), (kd_map.cpu().numpy() * 255).astype(np.uint8), mask, (mask.shape[1]//2, mask.shape[0]//2), cv2.NORMAL_CLONE)
   #cv2.imwrite("result.png", cv2.cvtColor(result * 255, cv2.COLOR_BGR2RGB))

    kd_map = kd_map * (1 - texture_mask[0]) + final_texture_map
    return kd_map, optimize_mask

def refine(save_path, front_image, back_image, left_image, right_image):
    ms = pymeshlab.MeshSet()
    mesh_path = f"{save_path}/model-00.obj"
    ms.load_new_mesh(mesh_path)
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=10)
    tl = open(mesh_path, "r").readlines()
    tex_uv = []
    uv_idx = []
    for line in tl:
        if line.startswith("vt"):
            uvs = line.split(" ")[1:3]
            tex_uv += [float(uvs[0]), 1.0-float(uvs[1])]
    tex_uv = torch.from_numpy(np.array(tex_uv)).to(torch.float32).cuda().reshape(-1, 2)
    m = ms.current_mesh()
    v_matrix = m.vertex_matrix()
    f_matrix = m.face_matrix()
    vn = m.vertex_normal_matrix()
    uv_idx = torch.arange(f_matrix.shape[0] * 3).reshape(-1, 3).to(torch.int32).cuda()
    vn = torch.tensor(vn).contiguous().cuda().to(torch.float32)

    frames = []
    front_camera = torch.tensor([[
        1,0,0,0,
        0,0,1,0,
        0,-1,0,-1.5,
        0,0,0,1,
    ]]).to(torch.float32).reshape(4,4).cuda()
    back_camera = torch.tensor([[
        1,0,0,0,
        0,0,1,0,
        0,1,0,-1.5,
        0,0,0,1,
    ]]).to(torch.float32).reshape(4,4).cuda()
    right_camera = torch.tensor([[
        0,-1,0,0,
        0,0,1,0,
        1,0,0,-1.5,
        0,0,0,1,
    ]]).to(torch.float32).reshape(4,4).cuda()
    left_camera = torch.tensor([[
        0,1,0,0,
        0,0,1,0,
        -1,0,0,-1.5,
        0,0,0,1,
    ]]).to(torch.float32).reshape(4,4).cuda()
    frames = [front_camera, left_camera, back_camera, right_camera]

    target_images = []
    for target_image in [front_image, left_image, back_image, right_image]:
        target_images.append(torch.from_numpy(np.asarray(target_image.convert("RGB"))).to(torch.float32).cuda() / 255.)

    pos = torch.tensor(v_matrix, dtype=torch.float32).contiguous().cuda()
    tri = torch.tensor(f_matrix, dtype=torch.int32).contiguous().cuda()

    kd_map = (torch.tensor(np.asarray(Image.open(f"{save_path}/texture_kd.jpg"))) / 255.).cuda()
    translate_tensor = torch.zeros((1,1,3)).cuda()
    pos = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()],-1).unsqueeze(0)
    glctx = dr.RasterizeCudaContext()
    target_texture = []
    target_mask = []
    rast_uvs = []
    with torch.no_grad():
        for idx, trans in enumerate(frames):
            target_image = target_images[idx]
            look_at = -torch.linalg.inv(trans)[:3,2]
            mvp = rec_mvp(trans, h=target_images[0].shape[0], w=target_images[0].shape[1])
            trans_pos = pos.clone()
            trans_pos[...,:3] += translate_tensor
            view_pos = torch.matmul(mvp, trans_pos.unsqueeze(-1)).squeeze(-1) 
            texture, mask, rast_uv = back_to_texture(glctx, look_at, view_pos, tri, target_image, tex_uv, uv_idx, idx, vn)
            target_texture.append(texture)
            target_mask.append(mask)
            rast_uvs.append(rast_uv)
        kd_map, opt_mask = aggregate_texture(kd_map, target_texture, target_mask, rast_uvs)
        opt_mask = opt_mask[0]
    Image.fromarray((np.clip(kd_map.detach().cpu().numpy() * 255, 0, 255)).astype(np.uint8)).save(f"{save_path}/refined_texture_kd.jpg")

   #ms.save_current_mesh(f"{save_path}/model-00.obj")
    with open(f"{save_path}/model-00.mtl", "w") as f:
        f.write(f"newmtl default\nKa 0.0 0.0 0.0\nmap_Kd refined_texture_kd.jpg\nKs 0.0 0.0 0.0")