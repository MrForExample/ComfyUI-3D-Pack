from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .io_gltf import GltfDocument
from .accessors import access_data, update_accessor_binary_data, recompute_accessor_min_max


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
            [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx), 0.0],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def compose_matrix(scale: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    S = np.diag([scale[0], scale[1], scale[2], 1.0])
    R = quaternion_matrix(rotation)
    T = np.eye(4)
    T[:3, 3] = translation
    return T @ (R @ S)


def direction_to_quaternion(direction: np.ndarray) -> np.ndarray:
    if not np.issubdtype(direction.dtype, np.floating):
        direction = direction.astype(np.float32)
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    direction = direction / norm
    forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dot = float(np.clip(np.dot(forward, direction), -1.0, 1.0))
    if dot > 0.9999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if dot < -0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = np.cross(forward, direction)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        if abs(direction[2]) < 0.9:
            axis = np.cross(up, direction)
        else:
            axis = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float32), direction)
        axis = axis / np.linalg.norm(axis)
    else:
        axis = axis / axis_norm
    angle = np.arccos(dot)
    half = angle * 0.5
    s = np.sin(half)
    c = np.cos(half)
    quat = np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float32)
    return quat / np.linalg.norm(quat)


def parse_node_transform(node_json: Dict[str, Any]) -> np.ndarray:
    if "matrix" in node_json:
        m = np.array(node_json["matrix"], dtype=np.float32).reshape(4, 4)
        return m.T
    t = np.array(node_json.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
    r = np.array(node_json.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
    s = np.array(node_json.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
    return compose_matrix(s, r, t)


def compute_global_transform(doc: GltfDocument, node_index: int, parent_transform: Optional[np.ndarray] = None) -> np.ndarray:
    node_json = doc.nodes()[node_index]
    local = parse_node_transform(node_json)
    return parent_transform @ local if parent_transform is not None else local


def compute_mesh_global_transforms(doc: GltfDocument) -> Dict[int, np.ndarray]:
    mesh_to_transform: Dict[int, np.ndarray] = {}
    scene_index = doc.json().get("scene")
    if scene_index is None:
        return mesh_to_transform
    root_nodes = doc.scenes()[scene_index].get("nodes", [])

    def traverse(node_idx: int, parent: Optional[np.ndarray] = None) -> None:
        global_m = compute_global_transform(doc, node_idx, parent)
        node = doc.nodes()[node_idx]
        if "mesh" in node:
            mesh_to_transform[int(node["mesh"])] = global_m
        for child in node.get("children", []) or []:
            traverse(int(child), global_m)

    for root in root_nodes:
        traverse(int(root))
    return mesh_to_transform


def transform_positions_inplace(doc: GltfDocument, mesh_idx: int, matrix: np.ndarray) -> None:
    mesh = doc.meshes()[mesh_idx]
    for primitive in mesh.get("primitives", []):
        attrs = primitive.get("attributes", {})
        if "POSITION" not in attrs:
            continue
        acc_idx = int(attrs["POSITION"])
        data = access_data(doc, acc_idx)
        pos_h = np.ones((data.shape[0], 4), dtype=np.float32)
        pos_h[:, :3] = data.astype(np.float32)
        transformed = (pos_h @ matrix.T)[:, :3].astype(np.float32)
        update_accessor_binary_data(doc, acc_idx, transformed)
        recompute_accessor_min_max(doc, acc_idx)


def transform_local_vertices_to_world(pos_data: np.ndarray, node: Dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        M = np.array(node["matrix"], dtype=np.float32).reshape(4, 4).T
    else:
        t = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
        r = np.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
        s = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
        M = compose_matrix(s, r, t)
    pos_h = np.concatenate([pos_data, np.ones((pos_data.shape[0], 1), dtype=np.float32)], axis=1)
    out = (M @ pos_h.T).T[:, :3]
    return out


def set_node_trs_identity(doc: GltfDocument, node_idx: int) -> None:
    node = doc.nodes()[node_idx]
    for key in ["matrix", "translation", "rotation", "scale"]:
        if key in node:
            node.pop(key)


def preprocess_transforms_inplace(doc: GltfDocument) -> None:
    """Apply node transforms to vertex positions and set TRS for all nodes.

    Idea: for each mesh, we calculate the global matrix, transform the POSITION of all
    primitives of this mesh and remove the transform for the nodes referring to this mesh.
    """
    mesh_transforms = compute_mesh_global_transforms(doc)
    # quick exit if all identity
    any_non_identity = any(
        not np.allclose(M, np.eye(4), atol=1e-6) for M in mesh_transforms.values()
    )
    if not any_non_identity:
        return

    for mesh_idx, M in mesh_transforms.items():
        if np.allclose(M, np.eye(4), atol=1e-6):
            continue
        # transform all positions in the mesh
        mesh = doc.meshes()[mesh_idx]
        for primitive in mesh.get("primitives", []):
            attrs = primitive.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            acc_idx = int(attrs["POSITION"])
            data = access_data(doc, acc_idx)
            pos_h = np.ones((data.shape[0], 4), dtype=np.float32)
            pos_h[:, :3] = data.astype(np.float32)
            transformed = (pos_h @ M.T)[:, :3].astype(np.float32)
            update_accessor_binary_data(doc, acc_idx, transformed)
            recompute_accessor_min_max(doc, acc_idx)

        # reset TRS for all nodes referring to this mesh
        for node_idx, node in enumerate(doc.nodes()):
            if node.get("mesh") == mesh_idx:
                set_node_trs_identity(doc, node_idx)

