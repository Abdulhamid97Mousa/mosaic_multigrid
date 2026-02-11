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

    All non-image observation keys (direction, mission, teammate_*)
    are passed through unchanged, making this wrapper composable
    with :class:`TeamObsWrapper`.
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


# -----------------------------------------------------------------------
# TeamObsWrapper
# -----------------------------------------------------------------------

class TeamObsWrapper(gym.ObservationWrapper):
    """
    Add teammate observation features to each agent's observation dict.

    Follows the standard MARL observation augmentation pattern established
    by SMAC (Samvelyan et al., 2019): each agent receives its own local
    partial view (image) **unchanged**, plus structured features about its
    teammates (relative positions, directions, ball-carrying status).

    This enables team coordination strategies (passing, defensive
    positioning) that are impossible with independent-view observations
    alone, since agents on a 15x10 field with ``view_size=3`` almost
    never see teammates in their 3x3 window.

    New observation keys added per agent:

    - ``teammate_positions``: ``ndarray (N, 2)`` -- relative ``(dx, dy)``
      from this agent to each teammate (in grid coordinates).
    - ``teammate_directions``: ``ndarray (N,)`` -- direction each
      teammate is facing (0=right, 1=down, 2=left, 3=up).
    - ``teammate_has_ball``: ``ndarray (N,)`` -- 1 if teammate is
      carrying a ball, 0 otherwise.

    Where ``N`` is the number of teammates (agents sharing the same
    ``team_index``).

    The original ``image``, ``direction``, and ``mission`` keys are
    preserved without modification.

    Parameters
    ----------
    env : gym.Env
        A MOSAIC multigrid environment (must have ``.agents`` attribute
        with ``team_index``, ``state.pos``, ``state.dir``, ``carrying``).

    References
    ----------
    Samvelyan, M., Rashid, T., de Witt, C. S., et al. (2019).
    "The StarCraft Multi-Agent Challenge." CoRR, abs/1902.04043.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base_env = env.unwrapped

        # Build team membership: {agent_index: [teammate_indices]}
        self._team_map: dict[int, list[int]] = {}
        for agent in base_env.agents:
            teammates = [
                a.index for a in base_env.agents
                if a.team_index == agent.team_index and a.index != agent.index
            ]
            self._team_map[agent.index] = teammates

        # Max number of teammates (for observation space bounds)
        max_teammates = max(len(t) for t in self._team_map.values()) if self._team_map else 0

        # Grid size bounds for relative positions
        max_dim = max(base_env.width, base_env.height)

        # Build new observation space for each agent
        new_obs_space = {}
        for agent_id, agent_space in env.observation_space.items():
            n_teammates = len(self._team_map.get(agent_id, []))
            new_obs_space[agent_id] = spaces.Dict({
                'image': agent_space['image'],
                'direction': agent_space['direction'],
                'mission': agent_space['mission'],
                'teammate_positions': spaces.Box(
                    low=-max_dim, high=max_dim,
                    shape=(n_teammates, 2),
                    dtype=np.int64,
                ),
                'teammate_directions': spaces.Box(
                    low=0, high=3,
                    shape=(n_teammates,),
                    dtype=np.int64,
                ),
                'teammate_has_ball': spaces.Box(
                    low=0, high=1,
                    shape=(n_teammates,),
                    dtype=np.int64,
                ),
            })

        self.observation_space = spaces.Dict(new_obs_space)

    def observation(self, obs: dict) -> dict:
        base_env = self.unwrapped
        agents = base_env.agents

        new_obs = {}
        for agent_id, agent_obs in obs.items():
            agent = agents[agent_id]
            ax, ay = int(agent.state.pos[0]), int(agent.state.pos[1])

            teammate_ids = self._team_map[agent_id]
            n = len(teammate_ids)

            positions = np.zeros((n, 2), dtype=np.int64)
            directions = np.zeros(n, dtype=np.int64)
            has_ball = np.zeros(n, dtype=np.int64)

            for i, tid in enumerate(teammate_ids):
                t = agents[tid]
                tx, ty = int(t.state.pos[0]), int(t.state.pos[1])
                positions[i] = [tx - ax, ty - ay]  # relative (dx, dy)
                directions[i] = int(t.state.dir)
                has_ball[i] = int(
                    t.state.carrying is not None
                    and t.state.carrying.type == Type.ball
                )

            new_obs[agent_id] = {
                'image': agent_obs['image'],
                'direction': agent_obs['direction'],
                'mission': agent_obs['mission'],
                'teammate_positions': positions,
                'teammate_directions': directions,
                'teammate_has_ball': has_ball,
            }

        return new_obs
