"""MOSAIC multigrid environments -- registration and exports.

All environments are registered with Gymnasium and can be created via
``gymnasium.make('MosaicMultiGrid-Soccer-v0')``, etc.
"""
from __future__ import annotations

import gymnasium as gym

from .soccer_game import (
    SoccerGameEnv,
    SoccerGame4HEnv10x15N2,
    SoccerGameEnhancedEnv,
    SoccerGame4HEnhancedEnv16x11N2,
)
from .collect_game import (
    CollectGameEnv,
    CollectGame3HEnv10x10N3,  # 3 agents, individual competition
    CollectGame4HEnv10x10N2,  # 4 agents, 2v2 teams
    CollectGameEnhancedEnv,
    CollectGame3HEnhancedEnv10x10N3,  # 3 agents, enhanced with natural termination
    CollectGame4HEnhancedEnv10x10N2,  # 4 agents 2v2, enhanced with natural termination
)

# -----------------------------------------------------------------------
# Configuration registry (env_name -> (env_cls, default_kwargs))
# -----------------------------------------------------------------------

CONFIGURATIONS: dict[str, tuple[type, dict]] = {
    # -----------------------------------------------------------------------
    # Original environments (v1.0.2) - kept for backward compatibility
    # -----------------------------------------------------------------------
    'MosaicMultiGrid-Soccer-v0': (SoccerGame4HEnv10x15N2, {}),
    'MosaicMultiGrid-Collect-v0': (CollectGame3HEnv10x10N3, {}),  # 3-agent individual
    'MosaicMultiGrid-Collect2vs2-v0': (CollectGame4HEnv10x10N2, {}),  # 4-agent teams

    # -----------------------------------------------------------------------
    # Enhanced environments (v1.1.0) - RECOMMENDED for RL training
    # -----------------------------------------------------------------------
    'MosaicMultiGrid-Soccer-Enhanced-v0': (SoccerGame4HEnhancedEnv16x11N2, {}),
    'MosaicMultiGrid-Collect-Enhanced-v0': (CollectGame3HEnhancedEnv10x10N3, {}),
    'MosaicMultiGrid-Collect2vs2-Enhanced-v0': (CollectGame4HEnhancedEnv10x10N2, {}),
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
