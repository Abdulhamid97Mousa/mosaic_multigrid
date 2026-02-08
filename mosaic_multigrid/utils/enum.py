"""Indexed enumeration utilities for type-safe constants with integer indexing."""
from __future__ import annotations

import functools
from typing import Any

import aenum as enum
import numpy as np
from numpy.typing import ArrayLike, NDArray as ndarray


@functools.cache
def _enum_array(enum_cls: enum.EnumMeta):
    """Return an array of all values of the given enum."""
    return np.array([item.value for item in enum_cls])


@functools.cache
def _enum_index(enum_item: enum.Enum):
    """Return the index of the given enum item."""
    return list(enum_item.__class__).index(enum_item)


class IndexedEnum(enum.Enum):
    """Enum where each member has a corresponding integer index."""

    def __int__(self):
        return self.to_index()

    @classmethod
    def add_item(cls, name: str, value: Any):
        """Add a new item to the enumeration."""
        enum.extend_enum(cls, name, value)
        _enum_array.cache_clear()
        _enum_index.cache_clear()

    @classmethod
    def from_index(cls, index: int | ArrayLike) -> enum.Enum | ndarray:
        """Return the enum item corresponding to the given index (or array of indices)."""
        out = _enum_array(cls)[index]
        return cls(out) if out.ndim == 0 else out

    def to_index(self) -> int:
        """Return the integer index of this enum item."""
        return _enum_index(self)
