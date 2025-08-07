from partcrafter_src.utils.typing_utils import *

import os
import numpy as np
from PIL import Image
import trimesh
from trimesh.transformations import rotation_matrix
# import pyrender  # Commented out - not needed for core functionality
from diffusers.utils import export_to_video
from diffusers.utils.loading_utils import load_video
import torch
from torchvision.utils import make_grid
import math

# os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Commented out - not needed without pyrender

def explode_mesh(mesh, explosion_scale=0.4):
    # ensure we have a Scene
    if isinstance(mesh, trimesh.Trimesh):
        scene = trimesh.Scene(mesh)
    elif isinstance(mesh, trimesh.Scene):
        scene = mesh
    else:
        raise ValueError(f"Expected Trimesh or Scene, got {type(mesh)}")

    if len(scene.geometry) <= 1:
        print("Nothing to explode")
        return scene

    # 1) collect (name, geom, world_center)
    parts = []
    for name, geom in scene.geometry.items():
        # ← get(name) returns (4×4 world‐space matrix, parent_frame)
        world_tf, _ = scene.graph.get(name)
        pts = trimesh.transformations.transform_points(geom.vertices, world_tf)
        center = pts.mean(axis=0)
        parts.append((name, geom, center))

    # compute global center
    all_centers = np.stack([c for _,_,c in parts], axis=0)
    global_center = all_centers.mean(axis=0)

    exploded = trimesh.Scene()
    for name, geom, center in parts:
        dir_vec = center - global_center
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-6:
            dir_vec = np.random.randn(3)
            dir_vec /= np.linalg.norm(dir_vec)
        else:
            dir_vec /= norm

        offset = dir_vec * explosion_scale

        # fetch the same 4×4, then bump just the translation
        world_tf, _ = scene.graph.get(name)
        world_tf = world_tf.copy()
        world_tf[:3, 3] += offset

        exploded.add_geometry(geom, transform=world_tf, geom_name=name)
        print(f"[explode] {name} moved by {np.linalg.norm(offset):.4f}")

    return exploded


# All render functions below use pyrender and are commented out
# They are not needed for core PartCrafter functionality

# def render(
#     scene: pyrender.Scene,
#     renderer: pyrender.Renderer,
#     camera: pyrender.Camera,
#     pose: np.ndarray,
#     light: Optional[pyrender.Light] = None,
#     normalize_depth: bool = False,
#     flags: int = pyrender.constants.RenderFlags.NONE,
#     return_type: Literal['pil', 'ndarray'] = 'pil'
# ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Image.Image, Image.Image]]:
#     camera_node = scene.add(camera, pose=pose)
#     if light is not None:
#         light_node = scene.add(light, pose=pose)
#     image, depth = renderer.render(
#         scene, 
#         flags=flags
#     )
#     scene.remove_node(camera_node)
#     if light is not None:
#         scene.remove_node(light_node)
#     if normalize_depth or return_type == 'pil':
#         depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
#     if return_type == 'pil':
#         image = Image.fromarray(image)
#         depth = Image.fromarray(depth.astype(np.uint8))
#     return image, depth

# def rotation_matrix_from_vectors(vec1, vec2):
#     a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
#     v = np.cross(a, b)
#     c = np.dot(a, b)
#     s = np.linalg.norm(v)
#     if s == 0:
#         return np.eye(3) if c > 0 else -np.eye(3)
#     kmat = np.array([
#         [0, -v[2], v[1]],
#         [v[2], 0, -v[0]],
#         [-v[1], v[0], 0]
#     ])
#     return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))

# def create_circular_camera_positions(
#     num_views: int,
#     radius: float,
#     axis: np.ndarray = np.array([0.0, 1.0, 0.0])
# ) -> List[np.ndarray]:
#     # Create a list of positions for a circular camera trajectory
#     # around the given axis with the given radius.
#     positions = []
#     axis = axis / np.linalg.norm(axis)
#     for i in range(num_views):
#         theta = 2 * np.pi * i / num_views
#         position = np.array([
#             np.sin(theta) * radius,
#             0.0,
#             np.cos(theta) * radius
#         ])
#         if not np.allclose(axis, np.array([0.0, 1.0, 0.0])):
#             R = rotation_matrix_from_vectors(np.array([0.0, 1.0, 0.0]), axis)
#             position = R @ position
#         positions.append(position)
#     return positions

# def create_circular_camera_poses(
#     num_views: int,
#     radius: float,
#     axis: np.ndarray = np.array([0.0, 1.0, 0.0])
# ) -> List[np.ndarray]:
#     # Create a list of poses for a circular camera trajectory
#     # around the given axis with the given radius.
#     # The camera always looks at the origin.
#     # The up vector is always [0, 1, 0].
#     canonical_pose = np.array([
#         [1.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0, radius],
#         [0.0, 0.0, 0.0, 1.0]
#     ])
#     poses = []
#     for i in range(num_views):
#         theta = 2 * np.pi * i / num_views
#         R = rotation_matrix(
#             angle=theta,
#             direction=axis,
#             point=[0, 0, 0]
#         )
#         pose = R @ canonical_pose
#         poses.append(pose)
#     return poses

# def render_views_around_mesh(
#     mesh: Union[trimesh.Trimesh, trimesh.Scene],
#     num_views: int = 36,
#     radius: float = 3.5,
#     axis: np.ndarray = np.array([0.0, 1.0, 0.0]),
#     image_size: tuple = (512, 512),
#     fov: float = 40.0,
#     light_intensity: Optional[float] = 5.0,
#     znear: float = 0.1,
#     zfar: float = 10.0, 
#     normalize_depth: bool = False,
#     flags: int = pyrender.constants.RenderFlags.NONE,
#     return_depth: bool = False, 
#     return_type: Literal['pil', 'ndarray'] = 'pil'
# ) -> Union[
#         List[Image.Image], 
#         List[np.ndarray], 
#         Tuple[List[Image.Image], List[Image.Image]], 
#         Tuple[List[np.ndarray], List[np.ndarray]]
#     ]:

#     meshes = []
#     scenes = []
    
#     if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
#         raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")
#     if isinstance(mesh, trimesh.Trimesh):
#         for i in range(num_views):
#             scenes.append(pyrender.Scene.from_trimesh_scene(trimesh.Scene(mesh)))
#     else:
#         for i in range(num_views):
#             value = math.sin(math.pi * (i - 1) / num_views)
#             scenes.append(pyrender.Scene.from_trimesh_scene(explode_mesh(mesh, 0.2 * value), 
#                                                             ambient_light=[0.02, 0.02, 0.02],
#                                                             bg_color=[0.0, 0.0, 0.0, 1.0]))
            
#     light = pyrender.DirectionalLight(
#         color=np.ones(3), 
#         intensity=light_intensity
#     ) if light_intensity is not None else None
#     camera = pyrender.PerspectiveCamera(
#         yfov=np.deg2rad(fov),
#         aspectRatio=image_size[0]/image_size[1],
#         znear=znear,
#         zfar=zfar
#     )
#     renderer = pyrender.OffscreenRenderer(*image_size)

#     camera_poses = create_circular_camera_poses(
#         num_views, 
#         radius, 
#         axis = axis
#     )

#     images, depths = [], []
#     for i, pose in enumerate(camera_poses):
#         image, depth = render(
#             scenes[i], renderer, camera, pose, light, 
#             normalize_depth=normalize_depth,
#             flags=flags,
#             return_type=return_type
#         )
#         images.append(image)
#         depths.append(depth)

#     renderer.delete()

#     if return_depth:
#         return images, depths
#     return images

# def render_normal_views_around_mesh(
#     mesh: Union[trimesh.Trimesh, trimesh.Scene],
#     num_views: int = 36,
#     radius: float = 3.5,
#     axis: np.ndarray = np.array([0.0, 1.0, 0.0]),
#     image_size: tuple = (512, 512),
#     fov: float = 40.0,
#     light_intensity: Optional[float] = 5.0,
#     znear: float = 0.1,
#     zfar: float = 10.0,
#     normalize_depth: bool = False,
#     flags: int = pyrender.constants.RenderFlags.NONE,
#     return_depth: bool = False, 
#     return_type: Literal['pil', 'ndarray'] = 'pil'
# ) -> Union[
#         List[Image.Image], 
#         List[np.ndarray], 
#         Tuple[List[Image.Image], List[Image.Image]], 
#         Tuple[List[np.ndarray], List[np.ndarray]]
#     ]:
    
#     if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
#         raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")
#     if isinstance(mesh, trimesh.Scene):
#         mesh = mesh.to_geometry()
#     normals = mesh.vertex_normals
#     colors = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
#     mesh.visual = trimesh.visual.ColorVisuals(
#         mesh=mesh,
#         vertex_colors=colors
#     )
#     mesh = trimesh.Scene(mesh)
#     return render_views_around_mesh(
#         mesh, num_views, radius, axis, 
#         image_size, fov, light_intensity, znear, zfar, 
#         normalize_depth, flags,
#         return_depth, return_type
#     )

# def create_camera_pose_on_sphere(
#     azimuth: float = 0.0, # in degrees
#     elevation: float = 0.0, # in degrees
#     radius: float = 3.5,
# ) -> np.ndarray:
#     # Create a camera pose for a given azimuth and elevation
#     # with the given radius.
#     # The camera always looks at the origin.
#     # The up vector is always [0, 1, 0].
#     canonical_pose = np.array([
#         [1.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0, radius],
#         [0.0, 0.0, 0.0, 1.0]
#     ])
#     azimuth = np.deg2rad(azimuth)
#     elevation = np.deg2rad(elevation)
#     position = np.array([
#         np.cos(elevation) * np.sin(azimuth),
#         np.sin(elevation),
#         np.cos(elevation) * np.cos(azimuth),
#     ])
#     R = np.eye(4)
#     R[:3, :3] = rotation_matrix_from_vectors(
#         np.array([0.0, 0.0, 1.0]), 
#         position
#     )
#     pose = R @ canonical_pose
#     return pose

# def render_single_view(
#     mesh: Union[trimesh.Trimesh, trimesh.Scene],
#     azimuth: float = 0.0, # in degrees
#     elevation: float = 0.0, # in degrees
#     radius: float = 3.5,
#     image_size: tuple = (512, 512),
#     fov: float = 40.0,
#     light_intensity: Optional[float] = 5.0,
#     num_env_lights: int = 0, 
#     znear: float = 0.1,
#     zfar: float = 10.0,
#     normalize_depth: bool = False,
#     flags: int = pyrender.constants.RenderFlags.NONE,
#     return_depth: bool = False, 
#     return_type: Literal['pil', 'ndarray'] = 'pil'
# ) -> Union[
#         Image.Image, 
#         np.ndarray, 
#         Tuple[Image.Image, Image.Image], 
#         Tuple[np.ndarray, np.ndarray]
#     ]:
    
#     if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
#         raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")
#     if isinstance(mesh, trimesh.Trimesh):
#         mesh = trimesh.Scene(mesh)

#     scene = pyrender.Scene.from_trimesh_scene(mesh)
#     light = pyrender.DirectionalLight(
#         color=np.ones(3), 
#         intensity=light_intensity
#     ) if light_intensity is not None else None
#     camera = pyrender.PerspectiveCamera(
#         yfov=np.deg2rad(fov),
#         aspectRatio=image_size[0]/image_size[1],
#         znear=znear,
#         zfar=zfar
#     )
#     renderer = pyrender.OffscreenRenderer(*image_size)

#     camera_pose = create_camera_pose_on_sphere(
#         azimuth,
#         elevation,
#         radius
#     )

#     if num_env_lights > 0:
#         env_light_poses = create_circular_camera_poses(
#             num_env_lights,
#             radius,
#             axis = np.array([0.0, 1.0, 0.0])
#         )
#         for pose in env_light_poses:
#             scene.add(pyrender.DirectionalLight(
#                 color=np.ones(3),
#                 intensity=light_intensity
#             ), pose=pose)
#         # set light to None
#         light = None

#     image, depth = render(
#         scene, renderer, camera, camera_pose, light,
#         normalize_depth=normalize_depth,
#         flags=flags,
#         return_type=return_type
#     )
#     renderer.delete()

#     if return_depth:
#         return image, depth
#     return image

# def render_normal_single_view(
#     mesh: Union[trimesh.Trimesh, trimesh.Scene],
#     azimuth: float = 0.0, # in degrees
#     elevation: float = 0.0, # in degrees
#     radius: float = 3.5,
#     image_size: tuple = (512, 512),
#     fov: float = 40.0,
#     light_intensity: Optional[float] = 5.0,
#     znear: float = 0.1,
#     zfar: float = 10.0,
#     normalize_depth: bool = False,
#     flags: int = pyrender.constants.RenderFlags.NONE,
#     return_depth: bool = False,
#     return_type: Literal['pil', 'ndarray'] = 'pil'
# ) -> Union[
#         Image.Image,
#         np.ndarray,
#         Tuple[Image.Image, Image.Image],
#         Tuple[np.ndarray, np.ndarray]
#     ]:

#     if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
#         raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")
#     if isinstance(mesh, trimesh.Scene):
#         mesh = mesh.to_geometry()
#     normals = mesh.vertex_normals
#     colors = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
#     mesh.visual = trimesh.visual.ColorVisuals(
#         mesh=mesh,
#         vertex_colors=colors
#     )
#     mesh = trimesh.Scene(mesh)
#     return render_single_view(
#         mesh, azimuth, elevation, radius, 
#         image_size, fov, light_intensity, znear, zfar,
#         normalize_depth, flags, 
#         return_depth, return_type
#     )

# def export_renderings(
#     images: List[Image.Image],
#     export_path: str,
#     fps: int = 36, 
#     loop: int = 0
# ): 
#     export_type = export_path.split('.')[-1]
#     if export_type == 'mp4':
#         export_to_video(
#             images,
#             export_path,
#             fps=fps,
#         )
#     elif export_type == 'gif':
#         duration = 1000 / fps
#         images[0].save(
#             export_path,
#             save_all=True,
#             append_images=images[1:],
#             duration=duration,
#             loop=loop
#         )
#     else:
#         raise ValueError(f'Unknown export type: {export_type}')
    
# def make_grid_for_images_or_videos(
#     images_or_videos: Union[List[Image.Image], List[List[Image.Image]]],
#     nrow: int = 4, 
#     padding: int = 0, 
#     pad_value: int = 0, 
#     image_size: tuple = (512, 512),
#     return_type: Literal['pil', 'ndarray'] = 'pil'
# ) -> Union[Image.Image, List[Image.Image], np.ndarray]:
#     if isinstance(images_or_videos[0], Image.Image):
#         images = [np.array(image.resize(image_size).convert('RGB')) for image in images_or_videos]
#         images = np.stack(images, axis=0).transpose(0, 3, 1, 2) # [N, C, H, W]
#         images = torch.from_numpy(images)
#         image_grid = make_grid(
#             images,
#             nrow=nrow,
#             padding=padding,
#             pad_value=pad_value,
#             normalize=False
#         ) # [C, H', W']
#         image_grid = image_grid.cpu().numpy()
#         if return_type == 'pil':
#             image_grid = Image.fromarray(image_grid.transpose(1, 2, 0))
#         return image_grid
#     elif isinstance(images_or_videos[0], list) and isinstance(images_or_videos[0][0], Image.Image):
#         image_grids = []
#         for i in range(len(images_or_videos[0])):
#             images = [video[i] for video in images_or_videos]
#             image_grid = make_grid_for_images_or_videos(
#                 images,
#                 nrow=nrow,
#                 padding=padding,
#                 return_type=return_type
#             )
#             image_grids.append(image_grid)
#         if return_type == 'ndarray':
#             image_grids = np.stack(image_grids, axis=0)
#         return image_grids
#     else:
#         raise ValueError(f'Unknown input type: {type(images_or_videos[0])}')