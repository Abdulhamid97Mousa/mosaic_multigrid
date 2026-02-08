"""Constants, enumerations, and color definitions for the MOSAIC multigrid environment."""
from __future__ import annotations

import enum

import numpy as np
from numpy.typing import NDArray as ndarray

from ..utils.enum import IndexedEnum


#: Tile size in pixels for rendering grid cells
TILE_PIXELS = 32

COLORS: dict[str, ndarray] = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey': np.array([100, 100, 100]),
}

DIR_TO_VEC: list[ndarray] = [
    np.array((1, 0)),   # right  (positive X)
    np.array((0, 1)),   # down   (positive Y)
    np.array((-1, 0)),  # left   (negative X)
    np.array((0, -1)),  # up     (negative Y)
]


class Type(str, IndexedEnum):
    """Enumeration of object types."""
    unseen = 'unseen'
    empty = 'empty'
    wall = 'wall'
    floor = 'floor'
    door = 'door'
    key = 'key'
    ball = 'ball'
    box = 'box'
    goal = 'goal'
    lava = 'lava'
    agent = 'agent'
    # MOSAIC extensions
    objgoal = 'objgoal'
    switch = 'switch'


class Color(str, IndexedEnum):
    """Enumeration of object colors."""
    red = 'red'
    green = 'green'
    blue = 'blue'
    purple = 'purple'
    yellow = 'yellow'
    grey = 'grey'

    @classmethod
    def add_color(cls, name: str, rgb: ndarray[np.uint8]) -> None:
        """
        Add a new color to the ``Color`` enumeration.

        Parameters
        ----------
        name : str
            Name of the new color.
        rgb : ndarray[np.uint8] of shape (3,)
            RGB value of the new color.
        """
        cls.add_item(name, name)
        COLORS[name] = np.asarray(rgb, dtype=np.uint8)

    @staticmethod
    def cycle(n: int) -> tuple[Color, ...]:
        """Return a cycle of ``n`` colors."""
        return tuple(Color.from_index(i % len(Color)) for i in range(int(n)))

    def rgb(self) -> ndarray[np.uint8]:
        """Return the RGB value of this ``Color``."""
        return COLORS[self]


class State(str, IndexedEnum):
    """Enumeration of object states."""
    open = 'open'
    closed = 'closed'
    locked = 'locked'


class Direction(enum.IntEnum):
    """Enumeration of agent directions."""
    right = 0
    down = 1
    left = 2
    up = 3

    def to_vec(self) -> ndarray[np.int8]:
        """Return the unit vector corresponding to this ``Direction``."""
        return DIR_TO_VEC[self]


# Backward-compatibility index mappings
OBJECT_TO_IDX: dict[Type, int] = {t: t.to_index() for t in Type}
IDX_TO_OBJECT: dict[int, Type] = {t.to_index(): t for t in Type}
COLOR_TO_IDX: dict[Color, int] = {c: c.to_index() for c in Color}
IDX_TO_COLOR: dict[int, Color] = {c.to_index(): c for c in Color}
STATE_TO_IDX: dict[State, int] = {s: s.to_index() for s in State}
COLOR_NAMES: list[str] = sorted(list(Color))
