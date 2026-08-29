"""Observation wrappers for MOSAIC multigrid environments.

Provides observation transformations that can be composed around any
:class:`~mosaic_multigrid.base.MultiGridEnv`.
"""
from __future__ import annotations

import gymnasium as gym
import numba as nb
import numpy as np

from gymnasium import spaces
from numpy.typing import NDArray as ndarray

from .core.constants import Color, Direction, Type
from .core.world_object import WorldObj


# -----------------------------------------------------------------------
# FullyObsWrapper
# -----------------------------------------------------------------------

class FullyObsWrapper(gym.ObservationWrapper):
    """
    Replace partial agent views with the full grid observation.

    Each agent receives the entire grid encoded as
    ``(width, height, WorldObj.dim)`` instead of the default
    ``(view_size, view_size, WorldObj.dim)`` partial view.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        new_image_space = spaces.Box(
            low=0, high=255,
            shape=(env.unwrapped.width, env.unwrapped.height, WorldObj.dim),
            dtype=np.uint8,
        )
        self.observation_space = spaces.Dict({
            agent_id: spaces.Dict({
                'image': new_image_space,
                'direction': agent_space['direction'],
                'mission': agent_space['mission'],
            })
            for agent_id, agent_space in env.observation_space.items()
        })

    def observation(self, obs: dict) -> dict:
        env = self.unwrapped
        full_grid = env.grid.encode()

        # Overlay agent positions on the full grid
        for agent in env.agents:
            if not agent.state.terminated:
                i, j = agent.state.pos
                full_grid[i, j] = agent.encode()

        return {
            agent_id: {
                'image': full_grid.copy(),
                'direction': agent_obs['direction'],
                'mission': agent_obs['mission'],
            }
            for agent_id, agent_obs in obs.items()
        }


# -----------------------------------------------------------------------
# ImgObsWrapper
# -----------------------------------------------------------------------

class ImgObsWrapper(gym.ObservationWrapper):
    """
    Extract only the image from the observation dict.

    Drops ``direction`` and ``mission`` keys, returning a uint8 image
    array per agent.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            agent_id: spaces.Box(
                low=0, high=255,
                shape=agent_space['image'].shape,
                dtype=np.uint8,
            )
            for agent_id, agent_space in env.observation_space.items()
        })

    def observation(self, obs: dict) -> dict:
        return {
            agent_id: agent_obs['image'].astype(np.uint8)
            for agent_id, agent_obs in obs.items()
        }


# -----------------------------------------------------------------------
# OneHotObsWrapper
# -----------------------------------------------------------------------

# One-hot dimension sizes: TYPE (13), COLOR (6), DIRECTION (4)
# The +1 accounts for the ball-carrying binary bit appended after direction.
_DIM_SIZES = np.array([len(Type), len(Color), len(Direction)])
_ONE_HOT_DIM = int(_DIM_SIZES.sum()) + 1  # 13 + 6 + 4 + 1 = 24

# Sentinel offset used by Agent.encode() for ball-carrying state.
# STATE values >= 100 encode direction + carrying: e.g. 102 = left + has ball.
_CARRY_SENTINEL = 100


@nb.njit(cache=True)
def _one_hot(image: ndarray, dim_sizes: ndarray) -> ndarray:
    """
    Convert a ``(H, W, 3)`` integer image into one-hot encoding.

    Channels 0 (TYPE) and 1 (COLOR) are standard one-hot.  Channel 2
    (STATE) is **factored** into two independent features:

    - **Direction** (4-way one-hot): extracted as ``value % 100``
    - **Carrying** (1 binary bit): ``1`` if ``value >= 100``, else ``0``

    This handles the sentinel encoding used by
    :meth:`~mosaic_multigrid.core.agent.Agent.encode` where ball-carrying
    agents produce STATE values 100--103 (direction + 100).

    Parameters
    ----------
    image : ndarray of shape (H, W, 3)
        Integer-encoded observation image.
    dim_sizes : ndarray of shape (3,)
        ``[len(Type), len(Color), len(Direction)]``.

    Returns
    -------
    out : ndarray of shape (H, W, sum(dim_sizes) + 1)
        One-hot encoded image with appended carrying bit.
    """
    h, w, _ = image.shape
    total = 0
    for k in range(len(dim_sizes)):
        total += dim_sizes[k]
    total += 1  # carrying bit

    out = np.zeros((h, w, total), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            offset = 0

            # Channels 0, 1: standard one-hot (TYPE, COLOR)
            for k in range(2):
                idx = image[i, j, k]
                if 0 <= idx < dim_sizes[k]:
                    out[i, j, offset + idx] = 1.0
                offset += dim_sizes[k]

            # Channel 2: factored into DIRECTION one-hot + CARRYING bit
            state_val = image[i, j, 2]
            if state_val >= _CARRY_SENTINEL:
                direction = state_val - _CARRY_SENTINEL
                carrying = 1
            else:
                direction = state_val
                carrying = 0

            if 0 <= direction < dim_sizes[2]:
                out[i, j, offset + direction] = 1.0
            offset += dim_sizes[2]

            out[i, j, offset] = carrying

    return out


class OneHotObsWrapper(gym.ObservationWrapper):
    """
    One-hot encode the observation image.

    Converts the ``(H, W, 3)`` integer-encoded image into a
    ``(H, W, 24)`` float32 tensor with factored one-hot encoding:

    ==========  =====  ===========================================
    Feature     Dims   Description
    ==========  =====  ===========================================
    TYPE         13    Object type (unseen, empty, wall, ..., switch)
    COLOR         6    Object color (red, green, blue, ...)
    DIRECTION     4    Agent facing direction (right, down, left, up)
    CARRYING      1    Ball-carrying flag (0 or 1)
    ==========  =====  ===========================================

    The DIRECTION + CARRYING split correctly handles the sentinel
    encoding (100--103) used by ball-carrying agents.

    All non-image observation keys (direction, mission) are passed
    through unchanged.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        sample_space = next(iter(env.observation_space.values()))
        h, w, _ = sample_space['image'].shape

        new_image_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(h, w, _ONE_HOT_DIM),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            agent_id: spaces.Dict({
                **{k: v for k, v in agent_space.items()},
                'image': new_image_space,
            })
            for agent_id, agent_space in env.observation_space.items()
        })

    def observation(self, obs: dict) -> dict:
        return {
            agent_id: {
                **agent_obs,
                'image': _one_hot(agent_obs['image'], _DIM_SIZES),
            }
            for agent_id, agent_obs in obs.items()
        }


# -----------------------------------------------------------------------
# GlobalObsWrapper
# -----------------------------------------------------------------------

class GlobalObsWrapper(gym.ObservationWrapper):
    """
    Add a ``'global_rgb'`` key to every agent's observation dict.

    Following the MeltingPot ``WORLD.RGB`` convention, every agent receives
    an identical top-down RGB render of the **entire grid** — a global map
    of the current environment state.

    Intended for **debugging and visualization**, not as a training
    observation (RGB rendering is expensive and not partial-observable).
    Compose with IndAgObs environments to get both the agent's local view
    and the global map side-by-side.

    Parameters
    ----------
    env : gym.Env
        The wrapped environment (any mosaic_multigrid env).
    tile_size : int
        Pixel size of each grid tile in the RGB output.  Default 32
        gives ``(width×32, height×32, 3)`` — e.g. 512×352 for 16×11.
    highlight : bool
        If ``True``, each agent's field-of-view is shaded on the global
        map (same highlight used by ``render_mode='human'``).  Default
        ``False`` gives a clean overhead map with no overlays.

    Example
    -------
    ::

        import gymnasium as gym
        import mosaic_multigrid
        from mosaic_multigrid.wrappers import GlobalObsWrapper

        env = gym.make('MosaicMultiGrid-Soccer-2v2-IndAgObs-v1')
        env = GlobalObsWrapper(env)

        obs, _ = env.reset()
        frame = obs[0]['global_rgb']   # (352, 512, 3) uint8 — full field
        agent_view = obs[0]['image']   # (3, 3, 3) uint8 — partial view

    Notes
    -----
    The global RGB is generated once per step and shared across all agents
    (O(1) render cost regardless of agent count).  The underlying
    :meth:`~mosaic_multigrid.base.MultiGridEnv.get_frame` call uses the
    environment's sport-specific renderer (basketball court, football field,
    soccer pitch) so the visual matches what ``render_mode='rgb_array'``
    would produce.
    """

    def __init__(
        self,
        env: gym.Env,
        tile_size: int = 32,
        highlight: bool = False,
    ):
        super().__init__(env)
        self._tile_size = tile_size
        self._highlight = highlight

        base = env.unwrapped
        # Shape: (width × tile_size, height × tile_size, 3)
        global_rgb_space = spaces.Box(
            low=0,
            high=255,
            shape=(base.width * tile_size, base.height * tile_size, 3),
            dtype=np.uint8,
        )

        self.observation_space = spaces.Dict({
            agent_id: spaces.Dict({
                **agent_space.spaces,
                'global_rgb': global_rgb_space,
            })
            for agent_id, agent_space in env.observation_space.items()
        })

    def observation(self, obs: dict) -> dict:
        # One render shared by all agents — O(1) cost
        global_rgb = self.env.unwrapped.get_frame(
            highlight=self._highlight,
            tile_size=self._tile_size,
        )
        return {
            agent_id: {**agent_obs, 'global_rgb': global_rgb}
            for agent_id, agent_obs in obs.items()
        }


# -----------------------------------------------------------------------
# SingleAgentWrapper
# -----------------------------------------------------------------------

class SingleAgentWrapper(gym.Wrapper):
    """
    Unwrap a multi-agent environment for single-agent use.

    Extracts observations, rewards, terminations, and truncations for
    agent 0 only. Actions are passed as scalars (not dicts).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = env.observation_space[0]
        self.action_space = env.action_space[0]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[0], info.get(0, {})

    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step({0: action})
        return (
            obs[0],
            rewards[0],
            terminated[0],
            truncated[0],
            info.get(0, {}),
        )

