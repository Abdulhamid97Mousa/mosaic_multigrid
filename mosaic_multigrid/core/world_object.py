"""World objects for the MOSAIC multigrid environment.

Includes standard grid objects (Goal, Floor, Lava, Wall, Door, Key, Ball, Box)
and MOSAIC-specific objects (ObjectGoal, Switch) for team-based Soccer/Collect
environments.
"""
from __future__ import annotations

import functools

import numpy as np
from numpy.typing import ArrayLike, NDArray as ndarray
from typing import Any, TYPE_CHECKING

from .constants import Color, State, Type
from ..utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)

if TYPE_CHECKING:
    from .agent import Agent


class WorldObjMeta(type):
    """
    Metaclass for world objects.

    Each subclass is associated with a unique :class:`Type` enumeration value.
    By default the type name is the lowercase class name, but this can be
    overridden by setting ``type_name`` in the class definition. Type names
    are dynamically added to the :class:`Type` enumeration if not already present.
    """

    # Registry mapping type index to object class
    _TYPE_IDX_TO_CLASS: dict[int, type] = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        if name != 'WorldObj':
            type_name = class_dict.get('type_name', name.lower())

            # Add the type name to the Type enumeration if not present
            if type_name not in set(Type):
                Type.add_item(type_name, type_name)

            # Register the class with its corresponding type index
            meta._TYPE_IDX_TO_CLASS[Type(type_name).to_index()] = cls

        return cls


class WorldObj(np.ndarray, metaclass=WorldObjMeta):
    """
    Base class for grid world objects.

    Each object is stored as a 3-element integer numpy array encoding
    (type, color, state) indices, enabling efficient vectorized operations
    on the grid.

    Attributes
    ----------
    type : Type
        The object type.
    color : Color
        The object color.
    state : State
        The object state.
    contains : WorldObj or None
        The object contained by this object, if any.
    init_pos : tuple[int, int] or None
        The initial position of the object.
    cur_pos : tuple[int, int] or None
        The current position of the object.
    """

    # Vector indices
    TYPE = 0
    COLOR = 1
    STATE = 2

    # Vector dimension
    dim = 3

    def __new__(cls, type: str | None = None, color: str = Color.from_index(0)):
        """
        Parameters
        ----------
        type : str or None
            Object type (inferred from class name if not provided).
        color : str
            Object color.
        """
        type_name = type or getattr(cls, 'type_name', cls.__name__.lower())
        type_idx = Type(type_name).to_index()

        # Resolve to the correct subclass via the metaclass registry
        cls = WorldObjMeta._TYPE_IDX_TO_CLASS.get(type_idx, cls)

        obj = np.zeros(cls.dim, dtype=int).view(cls)
        obj[WorldObj.TYPE] = type_idx
        obj[WorldObj.COLOR] = Color(color).to_index()
        obj.contains: WorldObj | None = None
        obj.init_pos: tuple[int, int] | None = None
        obj.cur_pos: tuple[int, int] | None = None

        return obj

    def __bool__(self) -> bool:
        return self.type != Type.empty

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(color={self.color})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: Any) -> bool:
        return self is other

    @staticmethod
    @functools.cache
    def empty() -> WorldObj:
        """Return a singleton empty WorldObj instance."""
        return WorldObj(type=Type.empty)

    @staticmethod
    def from_array(arr: ArrayLike) -> WorldObj | None:
        """
        Convert an array to a WorldObj instance.

        Parameters
        ----------
        arr : ArrayLike[int]
            Array encoding the object type, color, and state.
        """
        type_idx = arr[WorldObj.TYPE]

        if type_idx == Type.empty.to_index():
            return None

        if type_idx in WorldObjMeta._TYPE_IDX_TO_CLASS:
            cls = WorldObjMeta._TYPE_IDX_TO_CLASS[type_idx]
            obj = cls.__new__(cls)
            obj[...] = arr
            return obj

        raise ValueError(f'Unknown object type index: {arr[WorldObj.TYPE]}')

    @functools.cached_property
    def type(self) -> Type:
        """Return the object type."""
        return Type.from_index(self[WorldObj.TYPE])

    @property
    def color(self) -> Color:
        """Return the object color."""
        return Color.from_index(self[WorldObj.COLOR])

    @color.setter
    def color(self, value: str) -> None:
        """Set the object color."""
        self[WorldObj.COLOR] = Color(value).to_index()

    @property
    def state(self) -> State:
        """Return the object state."""
        return State.from_index(self[WorldObj.STATE])

    @state.setter
    def state(self, value: str) -> None:
        """Set the object state."""
        self[WorldObj.STATE] = State(value).to_index()

    def can_overlap(self) -> bool:
        """Can an agent overlap with this object?"""
        return self.type == Type.empty

    def can_pickup(self) -> bool:
        """Can an agent pick this up?"""
        return False

    def can_contain(self) -> bool:
        """Can this object contain another object?"""
        return False

    def toggle(self, env, agent: Agent, pos: tuple[int, int]) -> bool:
        """
        Toggle the state of this object or trigger an action.

        Parameters
        ----------
        env : MultiGridEnv
            The environment this object is contained in.
        agent : Agent
            The agent performing the toggle action.
        pos : tuple[int, int]
            The (x, y) position of this object in the grid.

        Returns
        -------
        success : bool
            Whether the toggle action was successful.
        """
        return False

    def encode(self) -> tuple[int, int, int]:
        """
        Encode a 3-tuple description of this object.

        Returns
        -------
        type_idx : int
            Index of the object type.
        color_idx : int
            Index of the object color.
        state_idx : int
            Index of the object state.
        """
        return tuple(self)

    @staticmethod
    def decode(type_idx: int, color_idx: int, state_idx: int) -> WorldObj | None:
        """
        Create an object from a 3-tuple description.

        Parameters
        ----------
        type_idx : int
            Index of the object type.
        color_idx : int
            Index of the object color.
        state_idx : int
            Index of the object state.
        """
        arr = np.array([type_idx, color_idx, state_idx])
        return WorldObj.from_array(arr)

    def render(self, img: ndarray[np.uint8]) -> None:
        """
        Draw the world object.

        Parameters
        ----------
        img : ndarray[np.uint8] of shape (width, height, 3)
            RGB image array to render object on.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Standard objects
# ---------------------------------------------------------------------------


class Goal(WorldObj):
    """Goal object an agent may be searching for."""

    def __new__(cls, color: str = Color.green, index: int = 0, reward: float = 1.0):
        """
        Parameters
        ----------
        color : str
            Object color.
        index : int
            Team or goal index for reward assignment.
        reward : float
            Reward value given when the goal is reached.
        """
        obj = super().__new__(cls, color=color)
        obj.index = index
        obj.reward = reward
        return obj

    def can_overlap(self) -> bool:
        return True

    def render(self, img: ndarray[np.uint8]) -> None:
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color.rgb())


class Floor(WorldObj):
    """Colored floor tile an agent can walk over."""

    def __new__(cls, color: str = Color.blue):
        return super().__new__(cls, color=color)

    def can_overlap(self) -> bool:
        return True

    def render(self, img: ndarray[np.uint8]) -> None:
        # Pale version of the floor color
        color = self.color.rgb() / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    """Lava object an agent can fall onto."""

    def __new__(cls):
        return super().__new__(cls, color=Color.red)

    def can_overlap(self) -> bool:
        return True

    def render(self, img: ndarray[np.uint8]) -> None:
        c = (255, 128, 0)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(WorldObj):
    """Wall object that agents cannot move through."""

    @functools.cache
    def __new__(cls, color: str = Color.grey):
        return super().__new__(cls, color=color)

    def render(self, img: ndarray[np.uint8]) -> None:
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color.rgb())


class Door(WorldObj):
    """
    Door object that may be opened or closed.

    Locked doors require a matching-color key to open.

    Attributes
    ----------
    is_open : bool
        Whether the door is open.
    is_locked : bool
        Whether the door is locked.
    """

    def __new__(
        cls,
        color: str = Color.blue,
        is_open: bool = False,
        is_locked: bool = False,
    ):
        door = super().__new__(cls, color=color)
        door.is_open = is_open
        door.is_locked = is_locked
        return door

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(color={self.color}, state={self.state})"

    @property
    def is_open(self) -> bool:
        return self.state == State.open

    @is_open.setter
    def is_open(self, value: bool) -> None:
        if value:
            self.state = State.open
        elif not self.is_locked:
            self.state = State.closed

    @property
    def is_locked(self) -> bool:
        return self.state == State.locked

    @is_locked.setter
    def is_locked(self, value: bool) -> None:
        if value:
            self.state = State.locked
        elif not self.is_open:
            self.state = State.closed

    def can_overlap(self) -> bool:
        return self.is_open

    def toggle(self, env, agent, pos) -> bool:
        if self.is_locked:
            carried_obj = agent.state.carrying
            if isinstance(carried_obj, Key) and carried_obj.color == self.color:
                self.is_locked = False
                self.is_open = True
                env.grid.update(*pos)
                return True
            return False

        self.is_open = not self.is_open
        env.grid.update(*pos)
        return True

    def render(self, img: ndarray[np.uint8]) -> None:
        c = self.color.rgb()

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * c)
            # Key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))
            # Door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(WorldObj):
    """Key object that can be picked up and used to unlock doors."""

    def __new__(cls, color: str = Color.blue):
        return super().__new__(cls, color=color)

    def can_pickup(self) -> bool:
        return True

    def render(self, img: ndarray[np.uint8]) -> None:
        c = self.color.rgb()

        # Vertical shaft
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):
    """Ball object that can be picked up by agents."""

    def __new__(cls, color: str = Color.blue, index: int = 0, reward: float = 1.0):
        """
        Parameters
        ----------
        color : str
            Object color.
        index : int
            Team or ball index for reward assignment.
        reward : float
            Reward value given when this ball is delivered.
        """
        obj = super().__new__(cls, color=color)
        obj.index = index
        obj.reward = reward
        return obj

    def can_pickup(self) -> bool:
        return True

    def render(self, img: ndarray[np.uint8]) -> None:
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), self.color.rgb())


class Box(WorldObj):
    """Box object that may contain other objects."""

    def __new__(cls, color: str = Color.yellow, contains: WorldObj | None = None):
        box = super().__new__(cls, color=color)
        box.contains = contains
        return box

    def can_pickup(self) -> bool:
        return True

    def can_contain(self) -> bool:
        return True

    def toggle(self, env, agent, pos) -> bool:
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

    def render(self, img: ndarray[np.uint8]) -> None:
        c = self.color.rgb()

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)


# ---------------------------------------------------------------------------
# MOSAIC-specific objects
# ---------------------------------------------------------------------------


class ObjectGoal(WorldObj):
    """
    Drop target for Soccer-style environments.

    Agents receive a reward when they drop a matching object on this tile.

    Attributes
    ----------
    target_type : str
        The type name of the object that should be dropped here (e.g. "ball").
    index : int
        Team or goal index for reward assignment.
    reward : float
        Reward value given on successful drop.
    """
    type_name = 'objgoal'

    def __new__(
        cls,
        color: str = Color.green,
        target_type: str = 'ball',
        index: int = 0,
        reward: float = 1.0,
    ):
        """
        Parameters
        ----------
        color : str
            Object color.
        target_type : str
            Type of object that counts as a valid drop.
        index : int
            Team or goal index.
        reward : float
            Reward given for a successful drop.
        """
        obj = super().__new__(cls, color=color)
        obj.target_type = target_type
        obj.index = index
        obj.reward = reward
        return obj

    def can_overlap(self) -> bool:
        return False

    def render(self, img: ndarray[np.uint8]) -> None:
        c = self.color.rgb()

        # Outer rectangle
        fill_coords(img, point_in_rect(0.0, 1.0, 0.0, 1.0), c)
        # Inner square (darker)
        fill_coords(img, point_in_rect(0.25, 0.75, 0.25, 0.75), c * 0.5)


class Switch(WorldObj):
    """
    Toggleable switch object.

    Agents can walk over switches and toggle them to trigger effects.
    """
    type_name = 'switch'

    def __new__(cls, color: str = Color.yellow):
        return super().__new__(cls, color=color)

    def can_overlap(self) -> bool:
        return True

    def toggle(self, env, agent, pos) -> bool:
        if self.state == State.open:
            self.state = State.closed
        else:
            self.state = State.open
        return True

    def render(self, img: ndarray[np.uint8]) -> None:
        c = self.color.rgb()

        # Small centered square
        fill_coords(img, point_in_rect(0.3, 0.7, 0.3, 0.7), c)
