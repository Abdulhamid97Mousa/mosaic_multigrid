"""Observation wrappers for MOSAIC multigrid environments.

Provides observation transformations that can be composed around any
:class:`~gym_multigrid.base.MultiGridEnv`.
"""
from __future__ import annotations

import gymnasium as gym
import numba as nb
import numpy as np

from gymnasium import spaces
from numpy.typing import NDArray as ndarray

from .core.constants import Color, Direction, State, Type
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

# Compute one-hot dimension sizes from enum lengths
_DIM_SIZES = np.array([len(Type), len(Color), max(len(State), len(Direction))])
_ONE_HOT_DIM = int(_DIM_SIZES.sum())


@nb.njit(cache=True)
def _one_hot(image: ndarray, dim_sizes: ndarray) -> ndarray:
    """
    Convert a ``(H, W, 3)`` integer image into one-hot encoding.

    Parameters
    ----------
    image : ndarray of shape (H, W, 3)
        Integer-encoded observation image.
    dim_sizes : ndarray of shape (3,)
        Number of categories per channel.

    Returns
    -------
    out : ndarray of shape (H, W, sum(dim_sizes))
        One-hot encoded image.
    """
    h, w, c = image.shape
    total = 0
    for k in range(c):
        total += dim_sizes[k]

    out = np.zeros((h, w, total), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            offset = 0
            for k in range(c):
                idx = image[i, j, k]
                if 0 <= idx < dim_sizes[k]:
                    out[i, j, offset + idx] = 1.0
                offset += dim_sizes[k]
    return out


class OneHotObsWrapper(gym.ObservationWrapper):
    """
    One-hot encode the observation image.

    Converts ``(H, W, 3)`` integer encoding into
    ``(H, W, sum(dim_sizes))`` one-hot float32 encoding. Uses Numba JIT
    for performance.
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
                'image': new_image_space,
                'direction': agent_space['direction'],
                'mission': agent_space['mission'],
            })
            for agent_id, agent_space in env.observation_space.items()
        })

    def observation(self, obs: dict) -> dict:
        return {
            agent_id: {
                'image': _one_hot(agent_obs['image'], _DIM_SIZES),
                'direction': agent_obs['direction'],
                'mission': agent_obs['mission'],
            }
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
