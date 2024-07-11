import pySDF as SDF
import numpy as np
from CharacterGen.Stage_3D.lrm.models.isosurface import MarchingTetrahedraHelper

def get_tetra_for_mesh(mesh_path, resolution=128):
    isosurface_helper = MarchingTetrahedraHelper(resolution, f"load/{resolution}_tets.npz")
    isosurface_helper.points_range = (-1, 1)
    mesh = trimesh.load(mesh_path)
    dmtet = np.load(f"")
    sdf = SDF(mesh.vertices, mesh.faces)
    sdf_gt = sdf(isosurface_helper.grid_vertices.numpy())
    return sdf_gt

