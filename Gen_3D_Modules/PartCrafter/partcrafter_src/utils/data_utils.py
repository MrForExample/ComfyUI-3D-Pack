from partcrafter_src.utils.typing_utils import *

import os
import numpy as np
import trimesh
import torch

def normalize_mesh(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    scale: float = 2.0,
):
    # if not isinstance(mesh, trimesh.Trimesh) and not isinstance(mesh, trimesh.Scene):
    #     raise ValueError("Input mesh is not a trimesh.Trimesh or trimesh.Scene object.")
    bbox = mesh.bounding_box
    translation = -bbox.centroid
    scale = scale / bbox.primitive.extents.max()
    mesh.apply_translation(translation)
    mesh.apply_scale(scale)
    return mesh

def remove_overlapping_vertices(mesh: trimesh.Trimesh, reserve_material: bool = False):
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input mesh is not a trimesh.Trimesh object.")
    vertices = mesh.vertices
    faces = mesh.faces
    unique_vertices, index_map, inverse_map = np.unique(
        vertices, axis=0, return_index=True, return_inverse=True
    )
    clean_faces = inverse_map[faces]
    clean_mesh = trimesh.Trimesh(vertices=unique_vertices, faces=clean_faces, process=True)
    if reserve_material:
        uv = mesh.visual.uv
        material = mesh.visual.material
        clean_uv = uv[index_map]
        clean_visual = trimesh.visual.TextureVisuals(uv=clean_uv, material=material)
        clean_mesh.visual = clean_visual
    return clean_mesh

RGB = [
    (82, 170, 220),
    (215, 91, 78),
    (45, 136, 117), 
    (247, 172, 83),
    (124, 121, 121),
    (127, 171, 209),
    (243, 152, 101),
    (145, 204, 192),
    (150, 59, 121),
    (181, 206, 78),
    (189, 119, 149),
    (199, 193, 222),
    (200, 151, 54),
    (236, 110, 102),
    (238, 182, 212),
]


def get_colored_mesh_composition(
    meshes: Union[List[trimesh.Trimesh], trimesh.Scene],
    is_random: bool = True,
    is_sorted: bool = False, 
    RGB: List[Tuple] = RGB
):
    if isinstance(meshes, trimesh.Scene):
        meshes = meshes.dump()
    if is_sorted:
        volumes = []
        for mesh in meshes:
            try:
                volume = mesh.volume
            except:
                volume = 0.0
            volumes.append(volume)
        # sort by volume from large to small
        meshes = [x for _, x in sorted(zip(volumes, meshes), key=lambda pair: pair[0], reverse=True)]
    colored_scene = trimesh.Scene()
    for idx, mesh in enumerate(meshes):
        if is_random:
            color = (np.random.rand(3) * 256).astype(int)
        else:
            color = np.array(RGB[idx % len(RGB)])
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=color,
        )
        colored_scene.add_geometry(mesh)
    return colored_scene

def mesh_to_surface(
    mesh: trimesh.Trimesh, 
    num_pc: int = 204800, 
    clip_to_num_vertices: bool = False,
    return_dict: bool = False,
):
    # if not isinstance(mesh, trimesh.Trimesh):
    #     raise ValueError("mesh must be a trimesh.Trimesh object")
    if clip_to_num_vertices:
        num_pc = min(num_pc, mesh.vertices.shape[0])
    points, face_indices = mesh.sample(num_pc, return_index=True)
    normals = mesh.face_normals[face_indices]
    if return_dict:
        return {
            "surface_points": points,
            "surface_normals": normals,
        }
    return points, normals

def scene_to_parts(
    mesh: trimesh.Scene,
    normalize: bool = True,
    scale: float = 2.0,
    num_part_pc: int = 204800, 
    clip_to_num_part_vertices: bool = False,
    return_type: Literal["mesh", "point"] = "mesh",
) -> Union[List[trimesh.Geometry], List[Dict[str, np.ndarray]]]:
    if not isinstance(mesh, trimesh.Scene):
        raise ValueError("mesh must be a trimesh.Scene object")
    if normalize:
        mesh = normalize_mesh(mesh, scale=scale)
    parts: List[trimesh.Geometry] = mesh.dump()
    if return_type == "point":
        datas: List[Dict[str, np.ndarray]] = []
        for geom in parts:
            data = mesh_to_surface(
                geom,
                num_pc=num_part_pc,
                clip_to_num_vertices=clip_to_num_part_vertices,
                return_dict=True,
            )
            datas.append(data)
        return datas
    elif return_type == "mesh":
        return parts
    else:
        raise ValueError("return_type must be 'mesh' or 'point'")
    
def get_center(mesh: trimesh.Trimesh, method: Literal['mass', 'bbox']):
    if method == 'mass':
        return mesh.center_mass
    elif method =='bbox':
        return mesh.bounding_box.centroid
    else:
        raise ValueError('type must be mass or bbox')
    
def get_direction(vector: np.ndarray):
    return vector / np.linalg.norm(vector)

def move_mesh_by_center(mesh: trimesh.Trimesh, scale: float, method: Literal['mass', 'bbox'] = 'mass'):
    offset = scale - 1
    center = get_center(mesh, method)
    direction = get_direction(center)
    translation = direction * offset
    mesh = mesh.copy()
    mesh.apply_translation(translation)
    return mesh

def move_meshes_by_center(meshes: Union[List[trimesh.Trimesh], trimesh.Scene], scale: float):
    if isinstance(meshes, trimesh.Scene):
        meshes = meshes.dump()
    moved_meshes = []
    for mesh in meshes:
        moved_mesh = move_mesh_by_center(mesh, scale)
        moved_meshes.append(moved_mesh)
    moved_meshes = trimesh.Scene(moved_meshes)
    return moved_meshes

def get_series_splited_meshes(meshes: List[trimesh.Trimesh], scale: float, num_steps: int) -> List[trimesh.Scene]:
    series_meshes = []
    for i in range(num_steps):
        temp_scale = 1 + (scale - 1) * i / (num_steps - 1)
        temp_meshes = move_meshes_by_center(meshes, temp_scale)
        series_meshes.append(temp_meshes)
    return series_meshes

def load_surface(data, num_pc=204800):

    surface = data["surface_points"]  # Nx3
    normal = data["surface_normals"]  # Nx3

    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], num_pc, replace=False)
    surface = torch.FloatTensor(surface[ind])
    normal = torch.FloatTensor(normal[ind])
    surface = torch.cat([surface, normal], dim=-1)

    return surface

def load_surfaces(surfaces, num_pc=204800):
    surfaces = [load_surface(surface, num_pc) for surface in surfaces]
    surfaces = torch.stack(surfaces, dim=0)
    return surfaces