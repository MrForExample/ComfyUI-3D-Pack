import os
from typing import List, Optional
from PIL import Image
import imageio
import time
import torch
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.common.datatypes import Device
from pytorch3d.structures import Meshes
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    AmbientLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    camera_position_from_spherical_angles,
    BlendParams,
)


def render(
    obj_filename, 
    elev=0, 
    azim=0, 
    resolution=512, 
    gif_dst_path='', 
    n_views=120, 
    fps=30, 
    device="cuda:0", 
    rgb=False
):
    '''
        obj_filename: path to obj file
        gif_dst_path: 
            if set a path, will render n_views frames, then save it to a gif file
            if not set, will render single frame, then return PIL.Image instance
        rgb: if set true, will convert result to rgb image/frame
    '''
    # load mesh
    mesh = load_objs_as_meshes([obj_filename], device=device)
    meshes = mesh.extend(n_views)
    
    if gif_dst_path != '':
        elev = torch.linspace(elev, elev, n_views+1)[:-1]
        azim = torch.linspace(0, 360, n_views+1)[:-1]

    # prepare R,T  then compute cameras
    R, T = look_at_view_transform(dist=1.5, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=49.1)

    # init pytorch3d renderer instance
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=resolution,
                blur_radius=0.0,
                faces_per_pixel=1,
            ),
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=AmbientLights(device=device),
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        )
    )
    images = renderer(meshes)

    # single frame rendering
    if gif_dst_path == '': 
        frame = images[0, ..., :3] if rgb else images[0, ...]
        frame = Image.fromarray((frame.cpu().squeeze(0) * 255).numpy().astype("uint8"))
        return frame

    # orbit frames rendering
    with imageio.get_writer(uri=gif_dst_path, mode='I', duration=1. / fps * 1000, loop=0) as writer:
        for i in range(n_views):
            frame = images[i, ..., :3] if rgb else images[i, ...]
            frame = Image.fromarray((frame.cpu().squeeze(0) * 255).numpy().astype("uint8"))
            writer.append_data(frame)
        return gif_dst_path
