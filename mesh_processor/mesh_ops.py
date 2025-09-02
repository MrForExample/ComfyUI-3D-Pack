from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .io_gltf import GltfDocument
from .accessors import access_data, append_accessor_and_bufferview
from .transforms import compute_mesh_global_transforms, transform_local_vertices_to_world, set_node_trs_identity


def get_all_meshes_triangles(doc: GltfDocument, transform_to_global: bool = True):
    if not doc.meshes():
        raise ValueError("No meshes found in the GLB/GLTF.")

    all_vertices_groups: List[Dict[str, Any]] = []
    vertices_count = 0
    all_pos_data: List[np.ndarray] = []
    all_indices_data: List[np.ndarray] = []

    mesh_transforms: Dict[int, np.ndarray] = {}
    if transform_to_global:
        mesh_transforms = compute_mesh_global_transforms(doc)

    for i_m, mesh_json in enumerate(doc.meshes()):
        for i_p, primitive in enumerate(mesh_json.get("primitives", [])):
            if "attributes" in primitive and "indices" in primitive:
                attributes = primitive["attributes"]
                if "POSITION" in attributes:
                    pos_accessor_index = int(attributes["POSITION"])
                    pos_data = access_data(doc, pos_accessor_index)
                    if pos_data.dtype != np.float32:
                        pos_data = pos_data.astype(np.float32)

                    if transform_to_global and i_m in mesh_transforms:
                        transform_matrix = mesh_transforms[i_m]
                        identity = np.eye(4, dtype=np.float32)
                        if not np.allclose(transform_matrix, identity, atol=1e-6):
                            fake_node = {"matrix": transform_matrix.flatten(order="F").tolist()}
                            pos_data = transform_local_vertices_to_world(pos_data, fake_node)

                    all_vertices_groups.append({
                        "mesh_index": i_m,
                        "primitive_index": i_p,
                        "vertices_range": (vertices_count, vertices_count + pos_data.shape[0]),
                        "global_transform_applied": transform_to_global and i_m in mesh_transforms,
                    })
                    all_pos_data.append(pos_data)

                    indices_accessor_index = int(primitive["indices"])
                    indices_data = access_data(doc, indices_accessor_index)
                    if indices_data.dtype != np.int64:
                        indices_data = indices_data.astype(np.int64)
                    all_indices_data.append(indices_data + vertices_count)

                    vertices_count += pos_data.shape[0]

    return (
        np.concatenate(all_pos_data, axis=0),
        np.concatenate(all_indices_data, axis=0).reshape(-1, 3),
        all_vertices_groups,
    )


def _choose_index_component_type(num_vertices: int) -> int:
    # 5123 = UNSIGNED_SHORT, 5125 = UNSIGNED_INT
    return 5123 if num_vertices <= 65535 else 5125


def merge_all_meshes_to_single(doc: GltfDocument) -> bool:
    if len(doc.meshes()) <= 1:
        return True

    mesh_transforms = compute_mesh_global_transforms(doc)

    merged_vertices: List[np.ndarray] = []
    merged_indices: List[np.ndarray] = []
    vertex_offset = 0

    for mesh_index, mesh in enumerate(doc.meshes()):
        for primitive in mesh.get("primitives", []):
            if "attributes" in primitive and "POSITION" in primitive["attributes"]:
                pos_accessor_index = int(primitive["attributes"]["POSITION"])
                vertices = access_data(doc, pos_accessor_index).astype(np.float32)
                if mesh_index in mesh_transforms:
                    transform_matrix = mesh_transforms[mesh_index]
                    identity = np.eye(4, dtype=np.float32)
                    if not np.allclose(transform_matrix, identity, atol=1e-6):
                        pos_h = np.ones((vertices.shape[0], 4), dtype=np.float32)
                        pos_h[:, :3] = vertices
                        vertices = (pos_h @ transform_matrix.T)[:, :3].astype(np.float32)

                if "indices" in primitive:
                    indices_accessor_index = int(primitive["indices"]) 
                    indices = access_data(doc, indices_accessor_index).astype(np.int64)
                    indices = indices + vertex_offset
                else:
                    indices = np.arange(len(vertices), dtype=np.int64) + vertex_offset
                    indices = indices.reshape(-1, 3)

                merged_vertices.append(vertices)
                merged_indices.append(indices)
                vertex_offset += len(vertices)

    if not merged_vertices:
        return False

    final_vertices = np.concatenate(merged_vertices, axis=0)
    final_indices = np.concatenate(merged_indices, axis=0)

    acc_pos, _ = append_accessor_and_bufferview(
        doc,
        final_vertices.astype(np.float32),
        component_type=5126,
        element_type="VEC3",
        target=34962,
    )

    index_component_type = _choose_index_component_type(final_vertices.shape[0])
    index_dtype = np.uint16 if index_component_type == 5123 else np.uint32
    acc_idx, _ = append_accessor_and_bufferview(
        doc,
        final_indices.astype(index_dtype).reshape(-1),
        component_type=index_component_type,
        element_type="SCALAR",
        target=34963,
    )

    merged_mesh = {
        "name": "Merged_Mesh",
        "primitives": [
            {
                "attributes": {"POSITION": acc_pos},
                "indices": acc_idx,
                "mode": 4,
            }
        ],
    }

    doc.json()["meshes"] = [merged_mesh]

    update_nodes_for_merged_mesh(doc)
    return True


def update_nodes_for_merged_mesh(doc: GltfDocument) -> None:
    nodes = doc.nodes()
    meshes = doc.meshes()
    merged_node = {"name": "Merged_Mesh_Node", "mesh": 0}
    nodes.append(merged_node)
    merged_node_index = len(nodes) - 1

    scene_index = doc.json().get("scene", 0)
    scenes = doc.scenes()
    if not scenes:
        doc.json()["scenes"] = [{"nodes": [merged_node_index]}]
        doc.json()["scene"] = 0
    else:
        scenes[scene_index].setdefault("nodes", [])
        scenes[scene_index]["nodes"].append(merged_node_index)

