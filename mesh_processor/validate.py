from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .io_gltf import GltfDocument


def validate_before_save(doc: GltfDocument) -> List[str]:
    errors: List[str] = []
    # buffers length and 4-byte alignment will be handled by writer, but sanity-check
    for i, buf in enumerate(doc.buffers()):
        expect = int(doc.binary(i).nbytes)
        if buf.get("byteLength", -1) != expect:
            errors.append(f"Buffer {i} byteLength mismatch: json={buf.get('byteLength')} actual={expect}")
    # bufferViews within ranges
    for i, bv in enumerate(doc.bufferViews()):
        b = int(bv.get("buffer", 0))
        off = int(bv.get("byteOffset", 0))
        ln = int(bv.get("byteLength", 0))
        if off + ln > int(doc.binary(b).nbytes):
            errors.append(f"bufferView {i} out of range: {off}+{ln}>{doc.binary(b).nbytes}")
        if off % 4 != 0:
            errors.append(f"bufferView {i} is not 4-byte aligned: offset={off}")
    # accessors consistency
    for i, acc in enumerate(doc.accessors()):
        if "bufferView" not in acc:
            continue
        bv = doc.bufferViews()[acc["bufferView"]]
        component_type = acc.get("componentType")
        acc_type = acc.get("type")
        if component_type not in (5120, 5121, 5122, 5123, 5125, 5126):
            errors.append(f"accessor {i} invalid componentType {component_type}")
        if acc_type not in ("SCALAR", "VEC2", "VEC3", "VEC4", "MAT2", "MAT3", "MAT4"):
            errors.append(f"accessor {i} invalid type {acc_type}")
    # position min/max
    for i, acc in enumerate(doc.accessors()):
        if acc.get("type") == "VEC3":
            if "min" not in acc or "max" not in acc:
                errors.append(f"accessor {i} missing min/max for VEC3")
    # primitives mode
    for m_idx, mesh in enumerate(doc.meshes()):
        for p_idx, prim in enumerate(mesh.get("primitives", [])):
            if prim.get("mode", 4) != 4:
                errors.append(f"mesh {m_idx} prim {p_idx} mode must be 4 (TRIANGLES)")
    return errors

