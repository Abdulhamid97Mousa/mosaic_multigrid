"""MOSAIC multigrid environments -- registration and exports.

All environments are registered with Gymnasium and can be created via
``gymnasium.make('MosaicMultiGrid-Soccer-v0')``, etc.
"""
from __future__ import annotations

import gymnasium as gym

from .soccer_game import (
    SoccerGameEnv,
    SoccerGame4HEnv10x15N2,
    SoccerGameIndAgObsEnv,
    SoccerGame4HIndAgObsEnv16x11N2,
    SoccerGame2HIndAgObsEnv16x11N2,
)
from .collect_game import (
    CollectGameEnv,
    CollectGame3HEnv10x10N3,  # 3 agents, individual competition
    CollectGame4HEnv10x10N2,  # 4 agents, 2v2 teams
    CollectGame2HEnv10x10N2,  # 2 agents, 1v1 teams
    CollectGameIndAgObsEnv,
    CollectGame3HIndAgObsEnv10x10N3,  # 3 agents, IndAgObs with natural termination
    CollectGame4HIndAgObsEnv10x10N2,  # 4 agents 2v2, IndAgObs with natural termination
    CollectGame2HIndAgObsEnv10x10N2,  # 2 agents 1v1, IndAgObs with natural termination
)
from .basketball_game import (
    BasketballGameEnv,
    BasketballGameIndAgObsEnv,
    BasketballGame6HIndAgObsEnv19x11N3,
)
from ..wrappers import TeamObsWrapper


# -----------------------------------------------------------------------
# TeamObs environment classes (SMAC-style teammate awareness)
#
# These thin wrappers compose IndAgObs base envs with TeamObsWrapper,
# adding structured teammate features (positions, directions, has_ball)
# to each agent's observation dict. Follows the observation augmentation
# pattern from SMAC (Samvelyan et al., 2019).
#
# Only defined for team-based environments (2v2+). The 3-agent Collect
# has agents_index=[1,2,3] (each agent = own team, no teammates).
# 1v1 environments also have no teammates, so no TeamObs variants.
# -----------------------------------------------------------------------

class SoccerTeamObsEnv(TeamObsWrapper):
    """Soccer 2v2 (16x11, IndAgObs) with SMAC-style teammate awareness."""
    # Class-level metadata required by gymnasium.make() -- gym.Wrapper defines
    # metadata as a @property (instance proxy), which breaks pre-instantiation
    # checks.  Override with the base env's metadata dict.
    metadata = SoccerGame4HIndAgObsEnv16x11N2.metadata

    def __init__(self, **kwargs):
        super().__init__(SoccerGame4HIndAgObsEnv16x11N2(**kwargs))


class Collect2vs2TeamObsEnv(TeamObsWrapper):
    """Collect 2v2 (10x10, IndAgObs) with SMAC-style teammate awareness."""
    metadata = CollectGame4HIndAgObsEnv10x10N2.metadata

    def __init__(self, **kwargs):
        super().__init__(CollectGame4HIndAgObsEnv10x10N2(**kwargs))


class Basketball3vs3TeamObsEnv(TeamObsWrapper):
    """Basketball 3vs3 (19x11, IndAgObs) with SMAC-style teammate awareness."""
    metadata = BasketballGame6HIndAgObsEnv19x11N3.metadata

    def __init__(self, **kwargs):
        super().__init__(BasketballGame6HIndAgObsEnv19x11N3(**kwargs))


# -----------------------------------------------------------------------
# Configuration registry (env_name -> (env_cls, default_kwargs))
# -----------------------------------------------------------------------

CONFIGURATIONS: dict[str, tuple[type, dict]] = {
    # -----------------------------------------------------------------------
    # Original environments (v1.0.2) - kept for backward compatibility
    # -----------------------------------------------------------------------
    'MosaicMultiGrid-Soccer-v0': (SoccerGame4HEnv10x15N2, {}),
    'MosaicMultiGrid-Collect-v0': (CollectGame3HEnv10x10N3, {}),  # 3-agent individual
    'MosaicMultiGrid-Collect-2vs2-v0': (CollectGame4HEnv10x10N2, {}),  # 4-agent teams
    'MosaicMultiGrid-Collect-1vs1-v0': (CollectGame2HEnv10x10N2, {}),  # 2-agent 1v1 teams

    # -----------------------------------------------------------------------
    # IndAgObs environments (v1.1.0) - RECOMMENDED for RL training
    # -----------------------------------------------------------------------
    'MosaicMultiGrid-Soccer-IndAgObs-v0': (SoccerGame4HIndAgObsEnv16x11N2, {}),
    'MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0': (SoccerGame2HIndAgObsEnv16x11N2, {}),
    'MosaicMultiGrid-Collect-IndAgObs-v0': (CollectGame3HIndAgObsEnv10x10N3, {}),
    'MosaicMultiGrid-Collect-2vs2-IndAgObs-v0': (CollectGame4HIndAgObsEnv10x10N2, {}),
    'MosaicMultiGrid-Collect-1vs1-IndAgObs-v0': (CollectGame2HIndAgObsEnv10x10N2, {}),

    # -----------------------------------------------------------------------
    # TeamObs environments (v2.0.0) - SMAC-style teammate awareness
    #
    # Build on IndAgObs base envs + TeamObsWrapper. Each agent receives
    # its local 3x3 image UNCHANGED, plus structured teammate features:
    #   teammate_positions (N,2), teammate_directions (N,), teammate_has_ball (N,)
    # Only for team-based (2v2+) environments. No 1v1 variants (no teammates).
    # -----------------------------------------------------------------------
    'MosaicMultiGrid-Soccer-TeamObs-v0': (SoccerTeamObsEnv, {}),
    'MosaicMultiGrid-Collect-2vs2-TeamObs-v0': (Collect2vs2TeamObsEnv, {}),

    # -----------------------------------------------------------------------
    # Basketball environments (v3.0.3) - 3vs3 on basketball court
    #
    # 19x11 grid (17x9 playable), 3vs3 teams, basketball-court rendering.
    # Same mechanics as Soccer IndAgObs (teleport pass, steal cooldown,
    # ball respawn, first-to-2-goals termination).
    # -----------------------------------------------------------------------
    'MosaicMultiGrid-Basketball-3vs3-IndAgObs-v0': (BasketballGame6HIndAgObsEnv19x11N3, {}),
    'MosaicMultiGrid-Basketball-3vs3-TeamObs-v0': (Basketball3vs3TeamObsEnv, {}),
}

# -----------------------------------------------------------------------
# Gymnasium registration
# -----------------------------------------------------------------------

for _env_id, (_env_cls, _default_kwargs) in CONFIGURATIONS.items():
    gym.register(
        id=_env_id,
        entry_point=_env_cls,
        kwargs=_default_kwargs,
    )
