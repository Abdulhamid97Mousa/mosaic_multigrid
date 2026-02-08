"""Ray RLlib MultiAgentEnv adapter for MOSAIC multigrid environments.

Usage::

    from mosaic_multigrid.rllib import RLlibWrapper, to_rllib_env

    # Wrap an existing env instance
    env = RLlibWrapper(my_multigrid_env)

    # Or get a class for registration
    env_cls = to_rllib_env(SoccerGame4HEnv10x15N2)

    # Module-level creator for Ray worker picklability
    from ray.tune.registry import register_env
    register_env('MosaicMultiGrid-Soccer-v0', lambda cfg: env_cls(**cfg))
"""
from __future__ import annotations

from typing import Any

try:
    from ray.rllib.env import MultiAgentEnv
    from ray.tune.registry import register_env
except ImportError:
    raise ImportError(
        'Ray RLlib is required for this adapter. '
        'Install with: pip install "mosaic_multigrid[rllib]"'
    )

from ..base import MultiGridEnv
from ..envs import CONFIGURATIONS

AgentID = int


class RLlibWrapper(MultiAgentEnv):
    """
    Wrap a :class:`~gym_multigrid.base.MultiGridEnv` as an RLlib
    :class:`~ray.rllib.env.MultiAgentEnv`.

    The main addition is the mandatory ``__all__`` key in
    ``terminated`` and ``truncated`` dicts that RLlib requires.

    Parameters
    ----------
    env : MultiGridEnv
        The multigrid environment to wrap.
    """

    def __init__(self, env: MultiGridEnv):
        super().__init__()
        self.env = env
        self.agents = [a.index for a in env.agents]
        self.possible_agents = self.agents[:]

        self.observation_spaces = {
            a: env.observation_space[a] for a in self.agents
        }
        self.action_spaces = {
            a: env.action_space[a] for a in self.agents
        }

    def get_observation_space(self, agent_id: AgentID):
        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id: AgentID):
        return self.action_spaces[agent_id]

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, action_dict: dict[AgentID, int]):
        obs, rewards, terminated, truncated, infos = self.env.step(action_dict)

        # RLlib requires __all__ keys
        terminated['__all__'] = all(
            terminated.get(a, False) for a in self.possible_agents
        )
        truncated['__all__'] = all(
            truncated.get(a, False) for a in self.possible_agents
        )

        return obs, rewards, terminated, truncated, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------

def to_rllib_env(
    env_cls: type[MultiGridEnv],
) -> type[RLlibWrapper]:
    """
    Create an RLlib MultiAgentEnv class from a multigrid env class.

    Parameters
    ----------
    env_cls : type[MultiGridEnv]
        The multigrid environment class to wrap.

    Returns
    -------
    type[RLlibWrapper]
        An RLlib MultiAgentEnv subclass.
    """
    class RLlibEnv(RLlibWrapper):
        def __init__(self, config: dict | None = None):
            config = config or {}
            env = env_cls(**config)
            super().__init__(env)

    RLlibEnv.__name__ = f'RLlib_{env_cls.__name__}'
    RLlibEnv.__qualname__ = RLlibEnv.__name__
    return RLlibEnv


# -----------------------------------------------------------------------
# Auto-registration (module-level creators for picklability)
# -----------------------------------------------------------------------

def _make_creator(cls: type[MultiGridEnv]):
    """Return a module-level env creator for Ray worker pickling."""
    def creator(config: dict) -> RLlibWrapper:
        return RLlibWrapper(cls(**config))
    return creator


for _env_id, (_env_cls, _defaults) in CONFIGURATIONS.items():
    register_env(_env_id, _make_creator(_env_cls))
