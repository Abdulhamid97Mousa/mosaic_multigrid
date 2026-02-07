"""PettingZoo ParallelEnv adapter for MOSAIC multigrid environments.

Usage::

    from gym_multigrid.pettingzoo import PettingZooWrapper, to_pettingzoo_env

    # Wrap an existing env instance
    env = PettingZooWrapper(my_multigrid_env)

    # Or create from an env class
    PZEnvCls = to_pettingzoo_env(SoccerGame4HEnv10x15N2)
    env = PZEnvCls(render_mode='rgb_array')
"""
from __future__ import annotations

import functools
from typing import Any

import gymnasium

try:
    from pettingzoo import ParallelEnv
except ImportError:
    raise ImportError(
        'PettingZoo is required for this adapter. '
        'Install with: pip install "mosaic_multigrid[pettingzoo]"'
    )

from ..base import MultiGridEnv

AgentID = int


class PettingZooWrapper(ParallelEnv):
    """
    Wrap a :class:`~gym_multigrid.base.MultiGridEnv` as a PettingZoo
    :class:`~pettingzoo.ParallelEnv`.

    Parameters
    ----------
    env : MultiGridEnv
        The multigrid environment to wrap.
    """

    metadata: dict[str, Any] = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'mosaic_multigrid_v0',
    }

    def __init__(self, env: MultiGridEnv):
        self.env = env
        self.possible_agents: list[AgentID] = [a.index for a in env.agents]
        self.agents: list[AgentID] = self.possible_agents[:]

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.action_space[agent]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, Any], dict[AgentID, dict]]:
        obs, info = self.env.reset(seed=seed)
        self.agents = self.possible_agents[:]
        return obs, {a: {} for a in self.agents} if not info else info

    def step(
        self,
        actions: dict[AgentID, int],
    ) -> tuple[
        dict[AgentID, Any],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        # Remove terminated/truncated agents from active list
        self.agents = [
            a for a in self.agents
            if not (terminations.get(a, False) or truncations.get(a, False))
        ]

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Rendering / cleanup
    # ------------------------------------------------------------------

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def state(self):
        """Return the full grid state (for centralized training)."""
        return self.env.grid.encode()


# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------

def to_pettingzoo_env(
    env_cls: type[MultiGridEnv],
    *wrappers: type[gymnasium.Wrapper],
    metadata: dict[str, Any] | None = None,
) -> type[PettingZooWrapper]:
    """
    Create a PettingZoo ParallelEnv class from a multigrid env class.

    Parameters
    ----------
    env_cls : type[MultiGridEnv]
        The multigrid environment class to wrap.
    *wrappers : type[gymnasium.Wrapper]
        Optional wrappers to apply (in order) before the PettingZoo wrapper.
    metadata : dict or None
        Override metadata for the PettingZoo environment.

    Returns
    -------
    type[PettingZooWrapper]
        A PettingZoo ParallelEnv subclass.

    Example
    -------
    >>> from gym_multigrid.envs import SoccerGame4HEnv10x15N2
    >>> PZEnv = to_pettingzoo_env(SoccerGame4HEnv10x15N2)
    >>> env = PZEnv(render_mode='rgb_array')
    >>> obs, info = env.reset(seed=42)
    """
    class PettingZooEnv(PettingZooWrapper):
        def __init__(self, *args, **kwargs):
            env = env_cls(*args, **kwargs)
            for wrapper in wrappers:
                env = wrapper(env)
            super().__init__(env)

    PettingZooEnv.__name__ = f'PettingZoo_{env_cls.__name__}'
    PettingZooEnv.__qualname__ = PettingZooEnv.__name__
    if metadata is not None:
        PettingZooEnv.metadata = metadata

    return PettingZooEnv
