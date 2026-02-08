"""MOSAIC multigrid environments -- registration and exports.

All environments are registered with Gymnasium and can be created via
``gymnasium.make('MosaicMultiGrid-Soccer-v0')``, etc.
"""
from __future__ import annotations

import gymnasium as gym

from .soccer_game import SoccerGameEnv, SoccerGame4HEnv10x15N2
from .collect_game import CollectGameEnv, CollectGame4HEnv10x10N2

# -----------------------------------------------------------------------
# Configuration registry (env_name -> (env_cls, default_kwargs))
# -----------------------------------------------------------------------

CONFIGURATIONS: dict[str, tuple[type, dict]] = {
    'MosaicMultiGrid-Soccer-v0': (SoccerGame4HEnv10x15N2, {}),
    'MosaicMultiGrid-Collect-v0': (CollectGame4HEnv10x10N2, {}),
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
