from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .io_gltf import GltfDocument
from .transforms import (
    compute_mesh_global_transforms,
    compose_matrix,
    direction_to_quaternion,
)
from .accessors import append_accessor_and_bufferview

_EPS = np.finfo(float).eps * 4.0


def check_input_file_compatibility(doc: GltfDocument) -> Dict[str, Any]:
    info = {
        "has_skins": False,
        "has_animations": False,
        "skin_count": 0,
        "animation_count": 0,
        "issues": [],
        "recommendations": [],
        "mesh_transforms_non_identity": [],
    }

    if "skins" in doc.json():
        info["has_skins"] = True
        info["skin_count"] = len(doc.json()["skins"])
        info["issues"].append(f"File already contains {info['skin_count']} skin(s)")
        info["recommendations"].append("Remove existing skin data before auto-rigging")

    if "animations" in doc.json():
        info["has_animations"] = True
        info["animation_count"] = len(doc.json()["animations"])
        info["issues"].append(
            f"File already contains {info['animation_count']} animation(s)"
        )
        info["recommendations"].append(
            "Existing animations may conflict with generated rig"
        )

    mesh_transforms = compute_mesh_global_transforms(doc)
    for mesh_index, transform in mesh_transforms.items():
        identity = np.eye(4, dtype=np.float32)
        if not np.allclose(transform, identity, atol=1e-6):
            mesh_name = doc.meshes()[mesh_index].get("name", f"Mesh_{mesh_index}")
            info["mesh_transforms_non_identity"].append(
                {"mesh_index": mesh_index, "mesh_name": mesh_name, "transform": transform.tolist()}
            )

    if info["mesh_transforms_non_identity"]:
        info["issues"].append(
            f"{len(info['mesh_transforms_non_identity'])} mesh(es) have non-identity transformations"
        )
        info["recommendations"].append(
            "Mesh positions will be transformed to global space for auto-rigging"
        )
    return info


def clean_rigging_data_from_gltf(doc: GltfDocument) -> Dict[str, Any]:
    cleanup_info = {
        "removed_skins": 0,
        "removed_animations": 0,
        "removed_skin_refs": 0,
        "removed_joint_attrs": 0,
        "removed_weight_attrs": 0,
        "removed_accessors": 0,
        "removed_buffer_views": 0,
        "binary_buffer_rebuilt": False,
        "operations": [],
    }

    # collect accessors to remove
    joint_weight_accessors = set()
    for mesh in doc.meshes():
        for primitive in mesh.get("primitives", []):
            attrs = primitive.get("attributes", {})
            if "JOINTS_0" in attrs:
                joint_weight_accessors.add(int(attrs["JOINTS_0"]))
                cleanup_info["removed_joint_attrs"] += 1
            if "WEIGHTS_0" in attrs:
                joint_weight_accessors.add(int(attrs["WEIGHTS_0"]))
                cleanup_info["removed_weight_attrs"] += 1

    inverse_bind_accessors = set()
    if "skins" in doc.json():
        for skin in doc.json()["skins"]:
            if "inverseBindMatrices" in skin:
                inverse_bind_accessors.add(int(skin["inverseBindMatrices"]))

    all_skin_accessors = joint_weight_accessors | inverse_bind_accessors

    skin_buffer_views = set()
    for accessor_idx in all_skin_accessors:
        if accessor_idx < len(doc.accessors()):
            accessor = doc.accessors()[accessor_idx]
            if "bufferView" in accessor:
                skin_buffer_views.add(int(accessor["bufferView"]))

    # remove animations
    if "animations" in doc.json():
        cleanup_info["removed_animations"] = len(doc.json()["animations"])
        del doc.json()["animations"]
        cleanup_info["operations"].append(
            f"Removed 'animations' section with {cleanup_info['removed_animations']} elements"
        )

    # remove skins
    if "skins" in doc.json():
        cleanup_info["removed_skins"] = len(doc.json()["skins"])
        del doc.json()["skins"]
        cleanup_info["operations"].append(
            f"Removed 'skins' section with {cleanup_info['removed_skins']} elements"
        )

    # remove node skin refs
    for idx, node in enumerate(doc.nodes()):
        if "skin" in node:
            node.pop("skin")
            cleanup_info["removed_skin_refs"] += 1
            cleanup_info["operations"].append(f"Removed skin reference from node index {idx}")

    # remove JOINTS_0/WEIGHTS_0 attributes
    for m_idx, mesh in enumerate(doc.meshes()):
        for p_idx, primitive in enumerate(mesh.get("primitives", [])):
            attrs = primitive.get("attributes", {})
            removed = []
            if "JOINTS_0" in attrs:
                attrs.pop("JOINTS_0")
                removed.append("JOINTS_0")
            if "WEIGHTS_0" in attrs:
                attrs.pop("WEIGHTS_0")
                removed.append("WEIGHTS_0")
            if removed:
                cleanup_info["operations"].append(
                    f"Removed attributes {removed} from mesh {m_idx} primitive {p_idx}"
                )

    # rebuild accessors without skin ones
    if all_skin_accessors:
        old_accessors = doc.accessors()
        new_accessors = []
        old_to_new = {}
        for i, acc in enumerate(old_accessors):
            if i in all_skin_accessors:
                cleanup_info["removed_accessors"] += 1
            else:
                old_to_new[i] = len(new_accessors)
                new_accessors.append(acc)
        doc.json()["accessors"] = new_accessors

        # remap references
        for mesh in doc.meshes():
            for primitive in mesh.get("primitives", []):
                attrs = primitive.get("attributes", {})
                for key, acc_idx in list(attrs.items()):
                    if acc_idx in old_to_new:
                        attrs[key] = old_to_new[acc_idx]
                    else:
                        # invalid -> drop attribute
                        attrs.pop(key)
                if "indices" in primitive:
                    idx = primitive["indices"]
                    if idx in old_to_new:
                        primitive["indices"] = old_to_new[idx]
                    else:
                        primitive.pop("indices", None)

    # bufferViews cleanup requires rebuilding binary; keep it simple for now
    # caller may opt to rebuild or leave holes; we just report counts
    cleanup_info["removed_buffer_views"] = len(skin_buffer_views)
    return cleanup_info


def clean_skin_data_from_gltf(doc: GltfDocument) -> Dict[str, Any]:
    return clean_rigging_data_from_gltf(doc)


@dataclass
class JointData:
    joint_node_index: int
    joint_name: str
    parent_index: Optional[int] = None
    children_indices: Tuple[int, ...] = ()
    forward_bind_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    inverse_bind_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    local_transform_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    original_index: Optional[int] = None


def insert_generated_data_to_gltf(
    doc: GltfDocument,
    joints: np.ndarray,
    conns: np.ndarray,
    skins: np.ndarray,
    all_vertices_groups: List[Dict[str, Any]],
) -> None:
    # A) collect used joints and build remap
    used = set()
    for v_group in all_vertices_groups:
        v0, v1 = v_group["vertices_range"]
        skins_data = skins[v0:v1]
        top_idx = np.argsort(-skins_data, axis=1)[:, :4]
        rows = np.arange(skins_data.shape[0])[:, None]
        top_w = skins_data[rows, top_idx]
        mask = top_w > _EPS
        used.update(top_idx[mask].tolist())
    used_sorted = sorted(used)
    old_to_new = {old: new for new, old in enumerate(used_sorted)}

    # B) build joint nodes and skin
    current_nodes_len = len(doc.nodes())
    joints_nodes_indices: List[int] = []
    all_joints_data: List[JointData] = []
    joints_global_pos: List[np.ndarray] = []
    for new_joint_idx, old_joint_idx in enumerate(used_sorted):
        parent_index = conns[old_joint_idx] if old_joint_idx < len(conns) else old_joint_idx
        joint_node_index = current_nodes_len + new_joint_idx
        joint_name = f"Joint_{new_joint_idx}"
        jd = JointData(joint_node_index, joint_name)
        jd.original_index = int(old_joint_idx)
        joints_nodes_indices.append(joint_node_index)
        all_joints_data.append(jd)
        parent_pick = int(parent_index) if parent_index < len(joints) else int(old_joint_idx)
        joints_global_pos.append(joints[parent_pick])

    # build parent/children links
    for jd in all_joints_data:
        old_idx = int(jd.original_index)
        parent_old = int(conns[old_idx]) if old_idx < len(conns) else old_idx
        if parent_old == old_idx:
            continue
        if parent_old in old_to_new:
            parent_new = old_to_new[parent_old]
            parent_node_index = all_joints_data[parent_new].joint_node_index
            jd.parent_index = parent_node_index
            parent_children = list(all_joints_data[parent_new].children_indices)
            if jd.joint_node_index not in parent_children:
                parent_children.append(jd.joint_node_index)
            all_joints_data[parent_new].children_indices = tuple(parent_children)

    # compute matrices
    inverse_bind_mats: List[np.ndarray] = []
    for i_name, jd in enumerate(all_joints_data):
        global_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        global_translation = np.asarray(joints_global_pos[i_name], dtype=np.float32)
        if jd.children_indices:
            offset = np.zeros(3, dtype=np.float32)
            for child_index in jd.children_indices:
                offset += (
                    joints_global_pos[child_index - current_nodes_len] - global_translation
                )
            global_rotation = direction_to_quaternion(offset)
        elif jd.parent_index is not None:
            offset_from_parent = (
                global_translation
                - joints_global_pos[jd.parent_index - current_nodes_len]
            )
            global_rotation = direction_to_quaternion(offset_from_parent)
        else:
            global_rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        forward = compose_matrix(global_scale, global_rotation, global_translation)
        jd.forward_bind_matrix = forward
        jd.inverse_bind_matrix = np.linalg.inv(forward)
        inverse_bind_mats.append(jd.inverse_bind_matrix)

    inverse_bind_mats = np.stack(inverse_bind_mats, axis=0).astype(np.float32)
    # glTF expects column-major arrays in JSON; we transpose like original code
    inverse_bind_mats = np.transpose(inverse_bind_mats, (0, 2, 1)).copy(order="C")

    # C) add vertex joint/weight attributes per groups
    for v_group in all_vertices_groups:
        v0, v1 = v_group["vertices_range"]
        skins_data = skins[v0:v1]
        top_idx = np.argsort(-skins_data, axis=1)[:, :4]
        rows = np.arange(skins_data.shape[0])[:, None]
        weights_data = skins_data[rows, top_idx].astype(np.float32)

        joints_data = np.zeros_like(top_idx, dtype=np.uint16)
        for i_row in range(top_idx.shape[0]):
            for i_col in range(4):
                old = int(top_idx[i_row, i_col])
                if old in old_to_new:
                    joints_data[i_row, i_col] = np.uint16(old_to_new[old])
                else:
                    joints_data[i_row, i_col] = np.uint16(0)
                    weights_data[i_row, i_col] = 0.0

        # D) normalize weights
        sums = np.sum(weights_data, axis=1, keepdims=True)
        mask_zero = sums < _EPS
        weights_data = np.where(mask_zero, 0.25, weights_data / sums)

        mesh_index = v_group["mesh_index"]
        prim_index = v_group["primitive_index"]
        primitive = doc.meshes()[mesh_index]["primitives"][prim_index]

        acc_j, _ = append_accessor_and_bufferview(
            doc, joints_data.astype(np.uint16), component_type=5123, element_type="VEC4", target=34962
        )
        acc_w, _ = append_accessor_and_bufferview(
            doc, weights_data.astype(np.float32), component_type=5126, element_type="VEC4", target=34962
        )
        primitive.setdefault("attributes", {})
        primitive["attributes"]["JOINTS_0"] = acc_j
        primitive["attributes"]["WEIGHTS_0"] = acc_w

    # skin JSON
    acc_ibm, _ = append_accessor_and_bufferview(
        doc, inverse_bind_mats.astype(np.float32), component_type=5126, element_type="MAT4", target=None
    )
    doc.json()["skins"] = [
        {"joints": joints_nodes_indices, "inverseBindMatrices": acc_ibm, "name": "Generated_Skin"}
    ]

    # add joint nodes
    root_joint_nodes: List[int] = []
    for jd in all_joints_data:
        node_json = {
            "name": jd.joint_name,
            "matrix": np.transpose(jd.local_transform_matrix, (1, 0)).copy(order="C").flatten().tolist(),
        }
        if jd.children_indices:
            node_json["children"] = list(jd.children_indices)
        doc.nodes().append(node_json)
        if jd.parent_index is None:
            root_joint_nodes.append(jd.joint_node_index)

    # joints root
    doc.nodes().append({"name": "Joints_Root", "children": root_joint_nodes})
    joints_root_index = len(doc.nodes()) - 1
    scene_index = doc.json().get("scene", 0)
    doc.scenes()[scene_index].setdefault("nodes", [])
    doc.scenes()[scene_index]["nodes"].append(joints_root_index)

