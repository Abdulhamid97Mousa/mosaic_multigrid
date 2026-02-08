"""Grid class for the MOSAIC multigrid environment.

Stores world objects in a numpy array with a lazy object dict cache,
following the multigrid-ini pattern. Agents are NOT stored on the grid;
their positions are tracked via :attr:`AgentState.pos`.
"""
from __future__ import annotations

import numpy as np

from collections import defaultdict
from functools import cached_property
from numpy.typing import NDArray as ndarray
from typing import Any, Callable, Iterable

from .agent import Agent
from .constants import Type, TILE_PIXELS
from .world_object import Wall, WorldObj

from ..utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
)


class Grid:
    """
    Class representing a grid of :class:`.WorldObj` objects.

    Attributes
    ----------
    width : int
        Width of the grid.
    height : int
        Height of the grid.
    world_objects : dict[tuple[int, int], WorldObj]
        Dictionary of world objects indexed by ``(x, y)`` location.
    state : ndarray[int] of shape (width, height, WorldObj.dim)
        Grid state where each ``(x, y)`` entry is a world-object encoding.
    """

    # Static cache of pre-rendered tiles
    _tile_cache: dict[tuple[Any, ...], Any] = {}

    def __init__(self, width: int, height: int):
        """
        Parameters
        ----------
        width : int
            Width of the grid.
        height : int
            Height of the grid.
        """
        assert width >= 3
        assert height >= 3
        self.world_objects: dict[tuple[int, int], WorldObj] = {}
        self.state: ndarray[np.int_] = np.zeros(
            (width, height, WorldObj.dim), dtype=int)
        self.state[...] = WorldObj.empty()

    @cached_property
    def width(self) -> int:
        """Width of the grid."""
        return self.state.shape[0]

    @cached_property
    def height(self) -> int:
        """Height of the grid."""
        return self.state.shape[1]

    @property
    def grid(self) -> list[WorldObj | None]:
        """Return a flat list of all world objects in the grid."""
        return [
            self.get(i, j)
            for i in range(self.width)
            for j in range(self.height)
        ]

    def set(self, x: int, y: int, obj: WorldObj | None):
        """
        Set a world object at the given coordinates.

        Parameters
        ----------
        x : int
            Grid x-coordinate.
        y : int
            Grid y-coordinate.
        obj : WorldObj or None
            Object to place, or None to clear the cell.
        """
        self.world_objects[x, y] = obj

        if isinstance(obj, WorldObj):
            self.state[x, y] = obj
        elif obj is None:
            self.state[x, y] = WorldObj.empty()
        else:
            raise TypeError(f"cannot set grid value to {type(obj)}")

    def get(self, x: int, y: int) -> WorldObj | None:
        """
        Get the world object at the given coordinates.

        Lazily creates a WorldObj instance from the state array if no
        cached instance exists.

        Parameters
        ----------
        x : int
            Grid x-coordinate.
        y : int
            Grid y-coordinate.
        """
        if (x, y) not in self.world_objects:
            self.world_objects[x, y] = WorldObj.from_array(self.state[x, y])

        return self.world_objects[x, y]

    def update(self, x: int, y: int):
        """
        Sync the state array from the cached world object at ``(x, y)``.

        Call this after mutating a WorldObj in-place (e.g. toggling a Door).

        Parameters
        ----------
        x : int
            Grid x-coordinate.
        y : int
            Grid y-coordinate.
        """
        if (x, y) in self.world_objects:
            self.state[x, y] = self.world_objects[x, y]

    def horz_wall(
        self,
        x: int, y: int,
        length: int | None = None,
        obj_type: Callable[[], WorldObj] = Wall,
    ):
        """
        Create a horizontal wall.

        Parameters
        ----------
        x : int
            Leftmost x-coordinate of the wall.
        y : int
            Y-coordinate of the wall.
        length : int or None
            Length of the wall. If None, extends to the right edge.
        obj_type : Callable[[], WorldObj]
            Factory returning the WorldObj to use for each wall cell.
        """
        length = self.width - x if length is None else length
        self.state[x:x + length, y] = obj_type()

    def vert_wall(
        self,
        x: int, y: int,
        length: int | None = None,
        obj_type: Callable[[], WorldObj] = Wall,
    ):
        """
        Create a vertical wall.

        Parameters
        ----------
        x : int
            X-coordinate of the wall.
        y : int
            Topmost y-coordinate of the wall.
        length : int or None
            Length of the wall. If None, extends to the bottom edge.
        obj_type : Callable[[], WorldObj]
            Factory returning the WorldObj to use for each wall cell.
        """
        length = self.height - y if length is None else length
        self.state[x, y:y + length] = obj_type()

    def wall_rect(self, x: int, y: int, w: int, h: int):
        """
        Create a walled rectangle (border only).

        Parameters
        ----------
        x : int
            X-coordinate of the top-left corner.
        y : int
            Y-coordinate of the top-left corner.
        w : int
            Width of the rectangle.
        h : int
            Height of the rectangle.
        """
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    @classmethod
    def render_tile(
        cls,
        obj: WorldObj | None = None,
        agent: Agent | None = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3,
    ) -> ndarray[np.uint8]:
        """
        Render a tile and cache the result.

        Parameters
        ----------
        obj : WorldObj or None
            Object to render.
        agent : Agent or None
            Agent to render.
        highlight : bool
            Whether to highlight the tile.
        tile_size : int
            Tile size in pixels.
        subdivs : int
            Downsampling factor for supersampling / anti-aliasing.
        """
        # Build a hashable cache key
        key: tuple[Any, ...] = (highlight, tile_size)
        if agent:
            # Include carrying info so the ball overlay is not skipped by cache
            carrying_key = agent.state.carrying.encode() if agent.state.carrying is not None else None
            key += (agent.state.color, agent.state.dir, carrying_key)
        else:
            key += (None, None, None)
        key = obj.encode() + key if obj else key

        if key in cls._tile_cache:
            return cls._tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3),
            dtype=np.uint8,
        )

        # Grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        # Draw the object
        if obj is not None:
            obj.render(img)

        # Draw the agent
        if agent is not None and not agent.state.terminated:
            agent.render(img)

        # Highlight
        if highlight:
            highlight_img(img)

        # Anti-alias via downsampling
        img = downsample(img, subdivs)

        cls._tile_cache[key] = img
        return img

    def render(
        self,
        tile_size: int,
        agents: Iterable[Agent] = (),
        highlight_mask: ndarray[np.bool_] | None = None,
    ) -> ndarray[np.uint8]:
        """
        Render the entire grid at a given scale.

        Parameters
        ----------
        tile_size : int
            Tile size in pixels.
        agents : Iterable[Agent]
            Agents to overlay on the grid.
        highlight_mask : ndarray[bool] of shape (width, height) or None
            Boolean mask indicating which cells to highlight.
        """
        if highlight_mask is None:
            highlight_mask = np.zeros(
                shape=(self.width, self.height), dtype=bool)

        # Build agent-location lookup (non-terminated agents get priority)
        location_to_agent = defaultdict(type(None))
        for agent in sorted(agents, key=lambda a: not a.terminated):
            location_to_agent[tuple(agent.pos)] = agent

        # Initialize pixel array
        width_px = self.width * tile_size
        height_px = self.height * tile_size
        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render each cell
        for j in range(self.height):
            for i in range(self.width):
                cell = self.get(i, j)
                tile_img = Grid.render_tile(
                    cell,
                    agent=location_to_agent[i, j],
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(
        self, vis_mask: ndarray[np.bool_] | None = None,
    ) -> ndarray[np.int_]:
        """
        Produce a compact numpy encoding of the grid.

        Unseen cells are encoded with :attr:`Type.unseen`.

        Parameters
        ----------
        vis_mask : ndarray[bool] of shape (width, height) or None
            Visibility mask. If None, all cells are visible.
        """
        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        encoding = self.state.copy()
        encoding[~vis_mask][..., WorldObj.TYPE] = Type.unseen.to_index()
        return encoding

    @staticmethod
    def decode(
        array: ndarray[np.int_],
    ) -> tuple[Grid, ndarray[np.bool_]]:
        """
        Decode an array grid encoding back into a Grid instance.

        Parameters
        ----------
        array : ndarray[int] of shape (width, height, dim)
            Grid encoding.

        Returns
        -------
        grid : Grid
            Decoded Grid instance.
        vis_mask : ndarray[bool] of shape (width, height)
            Visibility mask.
        """
        width, height, dim = array.shape
        assert dim == WorldObj.dim

        vis_mask = array[..., WorldObj.TYPE] != Type.unseen.to_index()
        grid = Grid(width, height)
        grid.state[vis_mask] = array[vis_mask]
        return grid, vis_mask
