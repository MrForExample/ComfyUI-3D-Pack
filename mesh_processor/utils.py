import json
import warnings
from typing import Any, Optional

import numpy as np


def pad_to_4_bytes(data: bytes, pad_byte: bytes = b"\x20") -> bytes:
    """Pad given bytes to the next 4-byte boundary.

    JSON chunks must be padded with spaces (0x20),
    binary chunks must be padded with zeros (0x00).
    """
    if not data:
        return data
    remainder = len(data) % 4
    if remainder == 0:
        return data
    return data + pad_byte * (4 - remainder)


def ensure_little_endian(array: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Return an array with little-endian dtype (copy if needed)."""
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    if array.dtype.byteorder in (">", "|"):
        return array.byteswap().newbyteorder("<")
    if array.dtype.byteorder == "<":
        return array
    # native without explicit endianness
    return array.astype(array.dtype.newbyteorder("<"), copy=False)


def as_uint8_buffer(array: np.ndarray) -> np.ndarray:
    """View array bytes as uint8 vector."""
    if not isinstance(array, np.ndarray):
        raise TypeError("as_uint8_buffer expects a numpy array")
    return np.frombuffer(array.tobytes(), dtype=np.uint8)


def warn(message: str) -> None:
    warnings.warn(message)


class NpEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars/arrays to native Python types.

    Important: cast numpy integers to int, not to string.
    """

    def default(self, obj: Any) -> Any:
        # scalars
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        # arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

