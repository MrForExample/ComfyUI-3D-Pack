from __future__ import annotations

import json
import struct
from base64 import b64decode
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import numpy as np

from .utils import NpEncoder, pad_to_4_bytes, warn


GLTF_MAGIC = 0x46546C67
JSON_CHUNK_TYPE = 0x4E4F534A
BINARY_CHUNK_TYPE = 0x004E4942
DATA_URI_PREFIX = "data:application/octet-stream;base64,"


def read_header(stream: BinaryIO) -> Optional[Tuple[int, int]]:
    header = stream.read(12)
    if len(header) != 12:
        return None
    magic, version, length = struct.unpack("<III", header)
    if magic != GLTF_MAGIC:
        return None
    return version, length


def read_chunks(stream: BinaryIO, length: int) -> Optional[Tuple[Dict[str, Any], np.ndarray]]:
    json_data: Optional[Dict[str, Any]] = None
    binary_data: Optional[np.ndarray] = None

    while stream.tell() < length:
        chunk_header = stream.read(8)
        if len(chunk_header) != 8:
            warn(f"Invalid chunk header size: {len(chunk_header)} bytes (expected 8)")
            break
        chunk_length, chunk_type = struct.unpack("<II", chunk_header)
        chunk_data = stream.read(chunk_length)
        if len(chunk_data) != chunk_length:
            warn(
                f"Incomplete chunk data: {len(chunk_data)} bytes (expected {chunk_length})"
            )
            return None

        if chunk_type == JSON_CHUNK_TYPE:
            json_data = json.loads(chunk_data)
        elif chunk_type == BINARY_CHUNK_TYPE:
            binary_data = np.frombuffer(chunk_data, dtype=np.uint8)
        else:
            warn("Unsupported chunk type")
            return None

    if json_data is None:
        raise ValueError("Missing json header")

    if binary_data is None:
        binary_data = np.array([], dtype=np.uint8)

    return json_data, binary_data


class GltfDocument:
    """Minimal GLTF/GLB container with helpers over core collections."""

    def __init__(self, json_data: Dict[str, Any], binary_buffers: Optional[Dict[int, np.ndarray]] = None):
        self._json = json_data
        self._bin: Dict[int, np.ndarray] = binary_buffers or {0: np.array([], dtype=np.uint8)}

    # Accessors to top-level collections
    def json(self) -> Dict[str, Any]:
        return self._json

    def buffers(self) -> List[Dict[str, Any]]:
        return self._json.setdefault("buffers", [])

    def bufferViews(self) -> List[Dict[str, Any]]:
        return self._json.setdefault("bufferViews", [])

    def accessors(self) -> List[Dict[str, Any]]:
        return self._json.setdefault("accessors", [])

    def nodes(self) -> List[Dict[str, Any]]:
        return self._json.setdefault("nodes", [])

    def meshes(self) -> List[Dict[str, Any]]:
        return self._json.setdefault("meshes", [])

    def scenes(self) -> List[Dict[str, Any]]:
        return self._json.setdefault("scenes", [])

    def materials(self) -> List[Dict[str, Any]]:
        return self._json.setdefault("materials", [])

    def images(self) -> List[Dict[str, Any]]:
        return self._json.setdefault("images", [])

    def textures(self) -> List[Dict[str, Any]]:
        return self._json.setdefault("textures", [])

    def samplers(self) -> List[Dict[str, Any]]:
        return self._json.setdefault("samplers", [])

    def binary(self, index: int = 0) -> np.ndarray:
        return self._bin.setdefault(index, np.array([], dtype=np.uint8))

    def set_binary(self, index: int, data: np.ndarray) -> None:
        self._bin[index] = data

    def update_buffer_length(self, buffer_index: int = 0) -> None:
        if not self.buffers():
            self._json["buffers"] = [{"byteLength": int(self.binary(buffer_index).nbytes)}]
        else:
            self.buffers()[buffer_index]["byteLength"] = int(self.binary(buffer_index).nbytes)


def get_binary_data(doc: GltfDocument, buffer_index: int) -> np.ndarray:
    """Return binary data for a buffer index.

    Supports embedded base64 buffers in GLTF as well as GLB's main buffer.
    """
    buffers = doc.buffers()
    if buffer_index >= len(buffers):
        raise IndexError("Buffer index out of range")
    binary = doc._bin.get(buffer_index)
    if binary is not None and binary.size > 0:
        return binary
    uri = buffers[buffer_index].get("uri")
    if uri and uri.startswith(DATA_URI_PREFIX):
        binary_data = b64decode(uri[len(DATA_URI_PREFIX) :])
        arr = np.frombuffer(binary_data, dtype=np.uint8)
        doc.set_binary(buffer_index, arr)
        return arr
    # if no uri, return existing buffer (may be empty for GLB until provided)
    return doc.binary(buffer_index)


def load_gltf_or_glb(path: str) -> GltfDocument:
    with open(path, "rb") as f:
        stream = f
        version_and_length = read_header(stream)
        if version_and_length is None:  # GLTF JSON
            stream.seek(0)
            json_data = json.load(stream)
            # embedded or external buffers will be resolved lazily
            return GltfDocument(json_data=json_data)
        version, length = version_and_length
        if version != 2:
            raise ValueError(f"Only GLB version 2 is supported, found version {version}")
        json_data, binary_data = read_chunks(stream, length)
        return GltfDocument(json_data=json_data, binary_buffers={0: binary_data})


def save_glb(doc: GltfDocument, stream: BinaryIO) -> None:
    """Write GLB v2 with proper 4-byte alignment for JSON and BIN chunks."""
    # ensure buffers length is correct for buffer 0
    doc.update_buffer_length(0)

    json_bytes = json.dumps(doc.json(), cls=NpEncoder, separators=(",", ":")).encode("utf-8")
    json_padded = pad_to_4_bytes(json_bytes, b"\x20")

    bin_bytes = bytes(doc.binary(0))
    bin_padded = pad_to_4_bytes(bin_bytes, b"\x00")

    version = 2
    file_header_size = 12
    chunk_header_size = 8
    file_length = (
        file_header_size
        + chunk_header_size
        + len(json_padded)
        + chunk_header_size
        + len(bin_padded)
    )

    # write GLB header
    stream.write(struct.pack("<III", GLTF_MAGIC, version, file_length))

    # JSON chunk
    stream.write(struct.pack("<II", len(json_padded), JSON_CHUNK_TYPE))
    stream.write(json_padded)

    # BIN chunk
    stream.write(struct.pack("<II", len(bin_padded), BINARY_CHUNK_TYPE))
    stream.write(bin_padded)

