from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .io_gltf import GltfDocument, load_gltf_or_glb, save_glb
from .mesh_ops import get_all_meshes_triangles, merge_all_meshes_to_single
from .transforms import (
    compose_matrix,
    quaternion_matrix,
    direction_to_quaternion,
    compute_mesh_global_transforms,
    transform_local_vertices_to_world,
    preprocess_transforms_inplace,
)
from .rigging_ops import (
    check_input_file_compatibility,
    clean_rigging_data_from_gltf,
    insert_generated_data_to_gltf,
)
from .validate import validate_before_save


__all__ = [
    "GltfDocument",
    "load_gltf_or_glb",
    "save_glb",
    "get_all_meshes_triangles",
    "merge_all_meshes_to_single",
    "compose_matrix",
    "quaternion_matrix",
    "direction_to_quaternion",
    "compute_mesh_global_transforms",
    "transform_local_vertices_to_world",
    "preprocess_transforms_inplace",
    "check_input_file_compatibility",
    "clean_rigging_data_from_gltf",
    "insert_generated_data_to_gltf",
    "validate_before_save",
]

