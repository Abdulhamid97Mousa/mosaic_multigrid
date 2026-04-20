"""MOSAIC multigrid environments — registration and exports.

All environments are registered with Gymnasium. Create via:

    gymnasium.make('MosaicMultiGrid-AF-2v2-IndAgObs-v1')
    gymnasium.make('MosaicMultiGrid-S-2v2-IndAgObs-v1')
    gymnasium.make('MosaicMultiGrid-BB-3v3-IndAgObs-v1')

Naming scheme (v6.5.0+):
    MosaicMultiGrid-<Sport>-[Team-]<Format>-[ObsVariant-]v1

    Sport:      S (Soccer) | BB (Basketball) | AF (AmericanFootball) | Collect
    Team:       G (Green only) | B (Blue only) — omitted for symmetric matchups
    Format:     NvM  (e.g. 1v0, 0v1, 1v1, 2v2, 3v3, 2v0, 0v2, 3v0, 0v3)
    ObsVariant: IndAgObs | TeamObs — omitted for solo (1v0 / 0v1) envs

v6.5.0 defaults applied to all environments:
    scoring reward  +15.0       (was +1.0 in v6.4.0)
    max_steps       300         (was 200 in v6.4.0)
    zero_sum        True        opponent team receives −15.0 on score
    timeout_penalty −1.0        applied when max_steps reached without winner
    ball provenance tracking    blocks same-team pickup-farming
    pass-chain cap              −0.2 on A→B→A bounce pass
    proximity reward            +0.01/step within Manhattan dist 3 of ball
"""
from __future__ import annotations

import gymnasium as gym

from .soccer_game import (
    SoccerGame4HIndAgObsEnv16x11N2,
    SoccerGame2HIndAgObsEnv16x11N2,
    SoccerSoloGreenIndAgObsEnv16x11,
    SoccerSoloBlueIndAgObsEnv16x11,
    SoccerGreen2v0IndAgObsEnv16x11,
    SoccerBlue0v2IndAgObsEnv16x11,
    SoccerGreen3v0IndAgObsEnv16x11,
    SoccerBlue0v3IndAgObsEnv16x11,
    SoccerGame6HIndAgObsEnv16x11N3,
)
from .collect_game import (
    CollectGame3HIndAgObsEnv10x10N3,
    CollectGame4HIndAgObsEnv10x10N2,
    CollectGame2HIndAgObsEnv10x10N2,
)
from .basketball_game import (
    BasketballGame6HIndAgObsEnv19x11N3,
    BasketballGame4HIndAgObsEnv19x11N2,
    BasketballGame2HIndAgObsEnv19x11N1,
    BasketballSoloGreenIndAgObsEnv19x11,
    BasketballSoloBlueIndAgObsEnv19x11,
    BasketballGreen2v0IndAgObsEnv19x11,
    BasketballBlue0v2IndAgObsEnv19x11,
    BasketballGreen3v0IndAgObsEnv19x11,
    BasketballBlue0v3IndAgObsEnv19x11,
)
from .american_football_game import (
    AmericanFootballSoloGreenEnv16x11,
    AmericanFootballSoloBlueEnv16x11,
    AmericanFootball1v1Env16x11,
    AmericanFootball2v2Env16x11,
    AmericanFootball3v3Env16x11,
    AmericanFootballGreen2v0Env16x11,
    AmericanFootballBlue0v2Env16x11,
    AmericanFootballGreen3v0Env16x11,
    AmericanFootballBlue0v3Env16x11,
)
from ..wrappers import TeamObsWrapper, GlobalObsWrapper


# -----------------------------------------------------------------------
# TeamObs wrapper classes
# -----------------------------------------------------------------------

# Soccer
class SoccerTeamObsEnv(TeamObsWrapper):
    metadata = SoccerGame4HIndAgObsEnv16x11N2.metadata
    def __init__(self, **kwargs):
        super().__init__(SoccerGame4HIndAgObsEnv16x11N2(**kwargs))

class SoccerGreen2v0TeamObsEnv(TeamObsWrapper):
    metadata = SoccerGreen2v0IndAgObsEnv16x11.metadata
    def __init__(self, **kwargs):
        super().__init__(SoccerGreen2v0IndAgObsEnv16x11(**kwargs))

class SoccerBlue0v2TeamObsEnv(TeamObsWrapper):
    metadata = SoccerBlue0v2IndAgObsEnv16x11.metadata
    def __init__(self, **kwargs):
        super().__init__(SoccerBlue0v2IndAgObsEnv16x11(**kwargs))

class SoccerGreen3v0TeamObsEnv(TeamObsWrapper):
    metadata = SoccerGreen3v0IndAgObsEnv16x11.metadata
    def __init__(self, **kwargs):
        super().__init__(SoccerGreen3v0IndAgObsEnv16x11(**kwargs))

class SoccerBlue0v3TeamObsEnv(TeamObsWrapper):
    metadata = SoccerBlue0v3IndAgObsEnv16x11.metadata
    def __init__(self, **kwargs):
        super().__init__(SoccerBlue0v3IndAgObsEnv16x11(**kwargs))

class Soccer3v3TeamObsEnv(TeamObsWrapper):
    metadata = SoccerGame6HIndAgObsEnv16x11N3.metadata
    def __init__(self, **kwargs):
        super().__init__(SoccerGame6HIndAgObsEnv16x11N3(**kwargs))

# Basketball
class Basketball1v1TeamObsEnv(TeamObsWrapper):
    metadata = BasketballGame2HIndAgObsEnv19x11N1.metadata
    def __init__(self, **kwargs):
        super().__init__(BasketballGame2HIndAgObsEnv19x11N1(**kwargs))

class Basketball2v2TeamObsEnv(TeamObsWrapper):
    metadata = BasketballGame4HIndAgObsEnv19x11N2.metadata
    def __init__(self, **kwargs):
        super().__init__(BasketballGame4HIndAgObsEnv19x11N2(**kwargs))

class Basketball3v3TeamObsEnv(TeamObsWrapper):
    metadata = BasketballGame6HIndAgObsEnv19x11N3.metadata
    def __init__(self, **kwargs):
        super().__init__(BasketballGame6HIndAgObsEnv19x11N3(**kwargs))

class BasketballGreen2v0TeamObsEnv(TeamObsWrapper):
    metadata = BasketballGreen2v0IndAgObsEnv19x11.metadata
    def __init__(self, **kwargs):
        super().__init__(BasketballGreen2v0IndAgObsEnv19x11(**kwargs))

class BasketballBlue0v2TeamObsEnv(TeamObsWrapper):
    metadata = BasketballBlue0v2IndAgObsEnv19x11.metadata
    def __init__(self, **kwargs):
        super().__init__(BasketballBlue0v2IndAgObsEnv19x11(**kwargs))

class BasketballGreen3v0TeamObsEnv(TeamObsWrapper):
    metadata = BasketballGreen3v0IndAgObsEnv19x11.metadata
    def __init__(self, **kwargs):
        super().__init__(BasketballGreen3v0IndAgObsEnv19x11(**kwargs))

class BasketballBlue0v3TeamObsEnv(TeamObsWrapper):
    metadata = BasketballBlue0v3IndAgObsEnv19x11.metadata
    def __init__(self, **kwargs):
        super().__init__(BasketballBlue0v3IndAgObsEnv19x11(**kwargs))

# Collect
class Collect2v2TeamObsEnv(TeamObsWrapper):
    metadata = CollectGame4HIndAgObsEnv10x10N2.metadata
    def __init__(self, **kwargs):
        super().__init__(CollectGame4HIndAgObsEnv10x10N2(**kwargs))

# American Football
class AmericanFootball2v2TeamObsEnv(TeamObsWrapper):
    metadata = AmericanFootball2v2Env16x11.metadata
    def __init__(self, **kwargs):
        super().__init__(AmericanFootball2v2Env16x11(**kwargs))

class AmericanFootball3v3TeamObsEnv(TeamObsWrapper):
    metadata = AmericanFootball3v3Env16x11.metadata
    def __init__(self, **kwargs):
        super().__init__(AmericanFootball3v3Env16x11(**kwargs))

class AmericanFootballGreen2v0TeamObsEnv(TeamObsWrapper):
    metadata = AmericanFootballGreen2v0Env16x11.metadata
    def __init__(self, **kwargs):
        super().__init__(AmericanFootballGreen2v0Env16x11(**kwargs))

class AmericanFootballBlue0v2TeamObsEnv(TeamObsWrapper):
    metadata = AmericanFootballBlue0v2Env16x11.metadata
    def __init__(self, **kwargs):
        super().__init__(AmericanFootballBlue0v2Env16x11(**kwargs))

class AmericanFootballGreen3v0TeamObsEnv(TeamObsWrapper):
    metadata = AmericanFootballGreen3v0Env16x11.metadata
    def __init__(self, **kwargs):
        super().__init__(AmericanFootballGreen3v0Env16x11(**kwargs))

class AmericanFootballBlue0v3TeamObsEnv(TeamObsWrapper):
    metadata = AmericanFootballBlue0v3Env16x11.metadata
    def __init__(self, **kwargs):
        super().__init__(AmericanFootballBlue0v3Env16x11(**kwargs))


# -----------------------------------------------------------------------
# Environment registry (v6.5.0)
# -----------------------------------------------------------------------

CONFIGURATIONS: dict[str, tuple[type, dict]] = {

    # -------------------------------------------------------------------
    # Soccer (S) — 16x11 grid
    # Solo: no obs suffix (single agent, no cooperation)
    # -------------------------------------------------------------------
    'MosaicMultiGrid-S-G-1v0-v1':           (SoccerSoloGreenIndAgObsEnv16x11,   {}),
    'MosaicMultiGrid-S-B-0v1-v1':           (SoccerSoloBlueIndAgObsEnv16x11,    {}),
    # Symmetric competitive
    'MosaicMultiGrid-S-1v1-IndAgObs-v1':    (SoccerGame2HIndAgObsEnv16x11N2,    {}),
    'MosaicMultiGrid-S-2v2-IndAgObs-v1':    (SoccerGame4HIndAgObsEnv16x11N2,    {}),
    'MosaicMultiGrid-S-2v2-TeamObs-v1':     (SoccerTeamObsEnv,                  {}),
    'MosaicMultiGrid-S-3v3-IndAgObs-v1':    (SoccerGame6HIndAgObsEnv16x11N3,    {}),
    'MosaicMultiGrid-S-3v3-TeamObs-v1':     (Soccer3v3TeamObsEnv,               {}),
    # One-sided Green
    'MosaicMultiGrid-S-G-2v0-IndAgObs-v1':  (SoccerGreen2v0IndAgObsEnv16x11,    {}),
    'MosaicMultiGrid-S-G-2v0-TeamObs-v1':   (SoccerGreen2v0TeamObsEnv,          {}),
    'MosaicMultiGrid-S-G-3v0-IndAgObs-v1':  (SoccerGreen3v0IndAgObsEnv16x11,    {}),
    'MosaicMultiGrid-S-G-3v0-TeamObs-v1':   (SoccerGreen3v0TeamObsEnv,          {}),
    # One-sided Blue
    'MosaicMultiGrid-S-B-0v2-IndAgObs-v1':  (SoccerBlue0v2IndAgObsEnv16x11,     {}),
    'MosaicMultiGrid-S-B-0v2-TeamObs-v1':   (SoccerBlue0v2TeamObsEnv,           {}),
    'MosaicMultiGrid-S-B-0v3-IndAgObs-v1':  (SoccerBlue0v3IndAgObsEnv16x11,     {}),
    'MosaicMultiGrid-S-B-0v3-TeamObs-v1':   (SoccerBlue0v3TeamObsEnv,           {}),

    # -------------------------------------------------------------------
    # Basketball (BB) — 19x11 grid
    # -------------------------------------------------------------------
    'MosaicMultiGrid-BB-G-1v0-v1':          (BasketballSoloGreenIndAgObsEnv19x11, {}),
    'MosaicMultiGrid-BB-B-0v1-v1':          (BasketballSoloBlueIndAgObsEnv19x11,  {}),
    # Symmetric competitive
    'MosaicMultiGrid-BB-1v1-IndAgObs-v1':   (BasketballGame2HIndAgObsEnv19x11N1,  {}),
    'MosaicMultiGrid-BB-1v1-TeamObs-v1':    (Basketball1v1TeamObsEnv,             {}),
    'MosaicMultiGrid-BB-2v2-IndAgObs-v1':   (BasketballGame4HIndAgObsEnv19x11N2,  {}),
    'MosaicMultiGrid-BB-2v2-TeamObs-v1':    (Basketball2v2TeamObsEnv,             {}),
    'MosaicMultiGrid-BB-3v3-IndAgObs-v1':   (BasketballGame6HIndAgObsEnv19x11N3,  {}),
    'MosaicMultiGrid-BB-3v3-TeamObs-v1':    (Basketball3v3TeamObsEnv,             {}),
    # One-sided Green
    'MosaicMultiGrid-BB-G-2v0-IndAgObs-v1': (BasketballGreen2v0IndAgObsEnv19x11,  {}),
    'MosaicMultiGrid-BB-G-2v0-TeamObs-v1':  (BasketballGreen2v0TeamObsEnv,        {}),
    'MosaicMultiGrid-BB-G-3v0-IndAgObs-v1': (BasketballGreen3v0IndAgObsEnv19x11,  {}),
    'MosaicMultiGrid-BB-G-3v0-TeamObs-v1':  (BasketballGreen3v0TeamObsEnv,        {}),
    # One-sided Blue
    'MosaicMultiGrid-BB-B-0v2-IndAgObs-v1': (BasketballBlue0v2IndAgObsEnv19x11,   {}),
    'MosaicMultiGrid-BB-B-0v2-TeamObs-v1':  (BasketballBlue0v2TeamObsEnv,         {}),
    'MosaicMultiGrid-BB-B-0v3-IndAgObs-v1': (BasketballBlue0v3IndAgObsEnv19x11,   {}),
    'MosaicMultiGrid-BB-B-0v3-TeamObs-v1':  (BasketballBlue0v3TeamObsEnv,         {}),

    # -------------------------------------------------------------------
    # American Football (AF) — 16x11 grid
    # -------------------------------------------------------------------
    'MosaicMultiGrid-AF-G-1v0-v1':          (AmericanFootballSoloGreenEnv16x11,   {}),
    'MosaicMultiGrid-AF-B-0v1-v1':          (AmericanFootballSoloBlueEnv16x11,    {}),
    # Symmetric competitive
    'MosaicMultiGrid-AF-1v1-IndAgObs-v1':   (AmericanFootball1v1Env16x11,         {}),
    'MosaicMultiGrid-AF-2v2-IndAgObs-v1':   (AmericanFootball2v2Env16x11,         {}),
    'MosaicMultiGrid-AF-2v2-TeamObs-v1':    (AmericanFootball2v2TeamObsEnv,       {}),
    'MosaicMultiGrid-AF-3v3-IndAgObs-v1':   (AmericanFootball3v3Env16x11,         {}),
    'MosaicMultiGrid-AF-3v3-TeamObs-v1':    (AmericanFootball3v3TeamObsEnv,       {}),
    # One-sided Green
    'MosaicMultiGrid-AF-G-2v0-IndAgObs-v1': (AmericanFootballGreen2v0Env16x11,    {}),
    'MosaicMultiGrid-AF-G-2v0-TeamObs-v1':  (AmericanFootballGreen2v0TeamObsEnv,  {}),
    'MosaicMultiGrid-AF-G-3v0-IndAgObs-v1': (AmericanFootballGreen3v0Env16x11,    {}),
    'MosaicMultiGrid-AF-G-3v0-TeamObs-v1':  (AmericanFootballGreen3v0TeamObsEnv,  {}),
    # One-sided Blue
    'MosaicMultiGrid-AF-B-0v2-IndAgObs-v1': (AmericanFootballBlue0v2Env16x11,     {}),
    'MosaicMultiGrid-AF-B-0v2-TeamObs-v1':  (AmericanFootballBlue0v2TeamObsEnv,   {}),
    'MosaicMultiGrid-AF-B-0v3-IndAgObs-v1': (AmericanFootballBlue0v3Env16x11,     {}),
    'MosaicMultiGrid-AF-B-0v3-TeamObs-v1':  (AmericanFootballBlue0v3TeamObsEnv,   {}),

    # -------------------------------------------------------------------
    # Collect (C) — 10x10 grid
    # -------------------------------------------------------------------
    'MosaicMultiGrid-C-IndAgObs-v1':      (CollectGame3HIndAgObsEnv10x10N3, {}),
    'MosaicMultiGrid-C-1v1-IndAgObs-v1':  (CollectGame2HIndAgObsEnv10x10N2, {}),
    'MosaicMultiGrid-C-2v2-IndAgObs-v1':  (CollectGame4HIndAgObsEnv10x10N2, {}),
    'MosaicMultiGrid-C-2v2-TeamObs-v1':   (Collect2v2TeamObsEnv,            {}),
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
