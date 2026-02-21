"""PettingZoo adapters for MOSAIC multigrid environments.

Supports both PettingZoo APIs:

- **Parallel API** (simultaneous stepping): All agents submit actions at once
  via a single ``step(actions_dict)`` call.  This is the native mode for
  mosaic_multigrid.

- **AEC API** (Agent-Environment Cycle): Agents take turns sequentially.
  Each call to ``env.step(action)`` advances only the current agent
  (``env.agent_selection``).  Non-acting agents must submit ``Action.noop``
  (index 0) so the environment can advance without moving them.

  ``Action.noop`` (index 0) was added to the action enum specifically for
  AEC compatibility.  Without it, non-acting agents would silently execute
  ``Action.left`` (the previous index 0), corrupting episodes.  This design
  is inspired by MeltingPot (Google DeepMind), which uses ``NOOP=0`` for
  the same reason.

Usage::

    from mosaic_multigrid.pettingzoo import (
        PettingZooWrapper,       # Parallel API wrapper
        to_pettingzoo_env,       # Parallel factory
        to_pettingzoo_aec_env,   # AEC factory
    )
    from mosaic_multigrid.core.actions import Action

    # --- Parallel API (all agents act simultaneously) ---
    PZParallel = to_pettingzoo_env(SoccerGame4HEnv10x15N2)
    env = PZParallel(render_mode='rgb_array')
    obs, infos = env.reset(seed=42)
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

    # --- AEC API (agents take turns) ---
    PZAec = to_pettingzoo_aec_env(SoccerGame4HEnv10x15N2)
    env = PZAec(render_mode='rgb_array')
    env.reset(seed=42)
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        action = None if term or trunc else env.action_space(agent).sample()
        env.step(action)
        # Non-active agents automatically receive Action.noop via the
        # parallel_to_aec converter; you only supply the current agent's action.
"""
from __future__ import annotations

import functools
from typing import Any

import gymnasium

try:
    from pettingzoo import ParallelEnv
    from pettingzoo.utils.conversions import parallel_to_aec
except ImportError:
    raise ImportError(
        'PettingZoo is required for this adapter. '
        'Install with: pip install "mosaic_multigrid[pettingzoo]"'
    )

from ..base import MultiGridEnv

AgentID = int


class PettingZooWrapper(ParallelEnv):
    """
    Wrap a :class:`~mosaic_multigrid.base.MultiGridEnv` as a PettingZoo
    :class:`~pettingzoo.ParallelEnv` (simultaneous stepping).

    All agents submit actions at the same time and receive observations
    after the environment has processed all actions.

    Parameters
    ----------
    env : MultiGridEnv
        The multigrid environment to wrap.
    """

    metadata: dict[str, Any] = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'mosaic_multigrid_v0',
        'is_parallelizable': True,
    }

    def __init__(self, env: MultiGridEnv):
        self.env = env
        # Use unwrapped to reach base MultiGridEnv attributes through any
        # Gymnasium wrappers (e.g. TeamObsWrapper, FullyObsWrapper).
        base = env.unwrapped if hasattr(env, 'unwrapped') else env
        self.possible_agents: list[AgentID] = [a.index for a in base.agents]
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

        # PettingZoo requires native Python types (not numpy.bool_, numpy.float64)
        terminations = {a: bool(v) for a, v in terminations.items()}
        truncations = {a: bool(v) for a, v in truncations.items()}
        rewards = {a: float(v) for a, v in rewards.items()}

        # Ensure infos exist for all active agents (PettingZoo requirement)
        if not infos:
            infos = {a: {} for a in self.agents}
        else:
            for a in self.agents:
                if a not in infos:
                    infos[a] = {}

        # Remove terminated/truncated agents from active list and their infos
        still_alive = [
            a for a in self.agents
            if not (terminations.get(a, False) or truncations.get(a, False))
        ]
        dead_agents = set(self.agents) - set(still_alive)
        for a in dead_agents:
            infos.pop(a, None)
        self.agents = still_alive

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
        return self.env.unwrapped.grid.encode()


# -----------------------------------------------------------------------
# Parallel Factory
# -----------------------------------------------------------------------

def to_pettingzoo_env(
    env_cls: type[MultiGridEnv],
    *wrappers: type[gymnasium.Wrapper],
    metadata: dict[str, Any] | None = None,
) -> type[PettingZooWrapper]:
    """
    Create a PettingZoo **ParallelEnv** class from a multigrid env class.

    All agents act simultaneously each step.

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
    >>> from mosaic_multigrid.envs import SoccerGame4HEnv10x15N2
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


# -----------------------------------------------------------------------
# AEC Factory
# -----------------------------------------------------------------------

def to_pettingzoo_aec_env(
    env_cls: type[MultiGridEnv],
    *wrappers: type[gymnasium.Wrapper],
    metadata: dict[str, Any] | None = None,
):
    """
    Create a PettingZoo **AECEnv** class from a multigrid env class.

    Agents take turns acting sequentially (Agent Environment Cycle).
    Internally, this wraps the ParallelEnv using PettingZoo's
    ``parallel_to_aec`` conversion utility.

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
    type
        A callable that creates an AECEnv instance.

    Example
    -------
    >>> from mosaic_multigrid.envs import SoccerGame4HEnv10x15N2
    >>> AecEnvFactory = to_pettingzoo_aec_env(SoccerGame4HEnv10x15N2)
    >>> env = AecEnvFactory(render_mode='rgb_array')
    >>> env.reset(seed=42)
    >>> for agent in env.agent_iter():
    ...     obs, reward, term, trunc, info = env.last()
    ...     action = None if term or trunc else env.action_space(agent).sample()
    ...     env.step(action)
    """
    parallel_cls = to_pettingzoo_env(env_cls, *wrappers, metadata=metadata)

    def make_aec(*args, **kwargs):
        parallel_env = parallel_cls(*args, **kwargs)
        return parallel_to_aec(parallel_env)

    make_aec.__name__ = f'AEC_{env_cls.__name__}'
    make_aec.__qualname__ = make_aec.__name__

    return make_aec
