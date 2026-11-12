"""Owned IOLS payloads, independent of mutable kernel and caller containers."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np


def owned_value(value: Any) -> Any:
    """Copy arrays into immutable bytes and recursively freeze nested containers."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("IOLS records cannot own object arrays")
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({key: owned_value(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(owned_value(item) for item in value)
    return value


def plain_value(value: Any) -> Any:
    """Return detached arrays and plain mutable containers for serialization."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: plain_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain_value(item) for item in value]
    return value
