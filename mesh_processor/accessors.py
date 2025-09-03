from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .io_gltf import GltfDocument, get_binary_data
from .utils import ensure_little_endian, as_uint8_buffer


_ELEMENT_SHAPES: Dict[str, Tuple[int, ...]] = {
    "SCALAR": (1,),
    "VEC2": (2,),
    "VEC3": (3,),
    "VEC4": (4,),
    "MAT2": (2, 2),
    "MAT3": (3, 3),
    "MAT4": (4, 4),
}

_ITEM_TYPES: Dict[int, Any] = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}


def _dtype_itemsize(component_type: int) -> int:
    return np.dtype(_ITEM_TYPES[component_type]).itemsize


def _validate_stride(stride: int) -> None:
    if stride != 0 and stride < 4:
        raise ValueError("Stride is too small.")
    if stride > 252:
        raise ValueError("Stride is too big.")


def access_data(doc: GltfDocument, accessor_index: int) -> np.ndarray:
    accessor = doc.accessors()[accessor_index]
    buffer_view_index = accessor.get("bufferView")
    if buffer_view_index is None:
        raise NotImplementedError("Undefined buffer view")

    accessor_byte_offset = int(accessor.get("byteOffset", 0))
    component_type = int(accessor["componentType"])  # glTF enum
    element_count = int(accessor["count"])  # number of elements
    element_type = accessor["type"]

    if accessor.get("sparse") is not None:
        raise NotImplementedError("Sparse accessors are not supported")

    buffer_view = doc.bufferViews()[buffer_view_index]
    buffer_index = int(buffer_view["buffer"])  # which buffer
    buffer_byte_length = int(buffer_view["byteLength"])  # length of this view
    element_byte_offset = int(buffer_view.get("byteOffset", 0))
    element_byte_stride = int(buffer_view.get("byteStride", 0))
    _validate_stride(element_byte_stride)

    element_shape = _ELEMENT_SHAPES[element_type]
    item_dtype = np.dtype(_ITEM_TYPES[component_type])
    item_count = int(np.prod(element_shape))
    item_size = item_dtype.itemsize

    size = element_count * item_count * item_size
    if size > buffer_byte_length:
        raise ValueError("Buffer did not have enough data for the accessor")

    buffers = doc.buffers()
    if buffer_index >= len(buffers):
        raise ValueError("Invalid buffer index")

    binary_data = get_binary_data(doc, buffer_index)
    if len(binary_data) < buffers[buffer_index].get("byteLength", 0):
        raise ValueError("Not enough binary data for the buffer")

    if element_byte_stride == 0:
        element_byte_stride = item_size * item_count
    if element_byte_stride < item_size * item_count:
        raise ValueError("Items should not overlap")

    # force little-endian
    item_dtype_str = np.dtype(item_dtype).newbyteorder("<").str

    dtype = np.dtype(
        {
            "names": ["element"],
            "formats": [str(element_shape) + item_dtype_str],
            "offsets": [0],
            "itemsize": element_byte_stride,
        }
    )

    byte_offset = accessor_byte_offset + element_byte_offset
    if byte_offset % item_size != 0:
        raise ValueError("Misaligned data")
    byte_length = element_count * element_byte_stride

    view = binary_data[byte_offset : byte_offset + byte_length].view(dtype)["element"]
    if element_type in ("MAT2", "MAT3", "MAT4"):
        view = np.transpose(view, (0, 2, 1))
    return view


def update_accessor_binary_data(doc: GltfDocument, accessor_idx: int, new_data: np.ndarray) -> None:
    accessor = doc.accessors()[accessor_idx]
    buffer_view_idx = accessor.get("bufferView")
    if buffer_view_idx is None:
        raise NotImplementedError("Undefined buffer view")

    buffer_view = doc.bufferViews()[buffer_view_idx]
    buffer_idx = int(buffer_view["buffer"])  # which buffer
    byte_offset = int(buffer_view.get("byteOffset", 0))
    accessor_byte_offset = int(accessor.get("byteOffset", 0))

    binary_data = doc.binary(buffer_idx)
    # ensure little-endian float32 or appropriate component
    new_bytes = ensure_little_endian(new_data).tobytes()

    start_pos = byte_offset + accessor_byte_offset
    end_pos = start_pos + len(new_bytes)

    new_binary_data = binary_data.copy()
    new_binary_data[start_pos:end_pos] = as_uint8_buffer(np.frombuffer(new_bytes, dtype=np.uint8))
    doc.set_binary(buffer_idx, new_binary_data)
    doc.update_buffer_length(buffer_idx)


def recompute_accessor_min_max(doc: GltfDocument, accessor_idx: int) -> None:
    accessor = doc.accessors()[accessor_idx]
    if accessor.get("type") != "VEC3":
        return
    data = access_data(doc, accessor_idx)
    accessor["min"] = data.min(axis=0).tolist()
    accessor["max"] = data.max(axis=0).tolist()


def append_accessor_and_bufferview(
    doc: GltfDocument,
    array: np.ndarray,
    component_type: int,
    element_type: str,
    target: Optional[int] = None,
) -> Tuple[int, int]:
    """Append array bytes into buffer 0 and create bufferView + accessor.

    Returns: (accessor_index, buffer_view_index)
    """
    if element_type not in _ELEMENT_SHAPES:
        raise ValueError("Invalid element type")
    if component_type not in _ITEM_TYPES:
        raise ValueError("Invalid component type")

    array_le = ensure_little_endian(array, np.dtype(_ITEM_TYPES[component_type]))
    element_shape = _ELEMENT_SHAPES[element_type]

    # validate last dimension matches element
    num_components = int(np.prod(element_shape))
    if array_le.size % num_components != 0:
        raise ValueError("Array size is not divisible by element components")

    if array_le.ndim == 1:
        count = array_le.size // num_components
    else:
        count = array_le.shape[0]

    # append to binary buffer 0
    buf0 = doc.binary(0)
    byte_offset = int(buf0.nbytes)
    bytes_data = array_le.tobytes()
    buf0_new = np.concatenate([buf0, np.frombuffer(bytes_data, dtype=np.uint8)])
    doc.set_binary(0, buf0_new)

    # create bufferView
    bv_index = len(doc.bufferViews())
    byte_length = len(bytes_data)
    bufferview: Dict[str, Any] = {
        "buffer": 0,
        "byteOffset": byte_offset,
        "byteLength": byte_length,
    }
    if target is not None:
        bufferview["target"] = target
    doc.bufferViews().append(bufferview)

    # create accessor
    acc_index = len(doc.accessors())
    accessor: Dict[str, Any] = {
        "bufferView": bv_index,
        "componentType": int(component_type),
        "count": int(count),
        "type": element_type,
    }
    # if POSITION like, store min/max
    if element_type == "VEC3":
        arr2d = array_le.reshape((-1, 3))
        accessor["min"] = arr2d.min(axis=0).tolist()
        accessor["max"] = arr2d.max(axis=0).tolist()
    doc.accessors().append(accessor)

    doc.update_buffer_length(0)
    return acc_index, bv_index

