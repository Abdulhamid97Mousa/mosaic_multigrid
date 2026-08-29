"""MOSAIC multigrid environments — registration and exports.

All environments are registered with Gymnasium. Create via:

    gymnasium.make('MosaicMultiGrid-AF-2v2-IndAgObs-v1')
    gymnasium.make('MosaicMultiGrid-S-2v2-IndAgObs-v1')
    gymnasium.make('MosaicMultiGrid-BB-3v3-IndAgObs-v1')

Naming scheme (v6.8.0+):
    MosaicMultiGrid-<Sport>-[Team-]<Format>-[ObsVariant-]v1

    Sport:      S (Soccer) | BB (Basketball) | AF (AmericanFootball) | Collect
    Team:       G (Green only) | B (Blue only) — omitted for symmetric matchups
    Format:     NvM  (e.g. 1v0, 0v1, 1v1, 2v2, 3v3, 2v0, 0v2, 3v0, 0v3)
    ObsVariant: IndAgObs — omitted for solo (1v0 / 0v1) envs

v6.8.0 defaults applied to all environments:
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
    SoccerGreen4v0IndAgObsEnv16x11,
    SoccerBlue0v4IndAgObsEnv16x11,
    SoccerGreen5v0IndAgObsEnv16x11,
    SoccerBlue0v5IndAgObsEnv16x11,
    SoccerGreen6v0IndAgObsEnv16x11,
    SoccerBlue0v6IndAgObsEnv16x11,
    SoccerGame8HIndAgObsEnv16x11N4,
    SoccerGame1v2IndAgObsEnv16x11,
    SoccerGame2v1IndAgObsEnv16x11,
    SoccerGame1v3IndAgObsEnv16x11,
    SoccerGame3v1IndAgObsEnv16x11,
    SoccerGame1v4IndAgObsEnv16x11,
    SoccerGame4v1IndAgObsEnv16x11,
    SoccerGame2v3IndAgObsEnv16x11,
    SoccerGame3v2IndAgObsEnv16x11,
    SoccerGame2v4IndAgObsEnv16x11,
    SoccerGame4v2IndAgObsEnv16x11,
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
    BasketballGreen4v0IndAgObsEnv19x11,
    BasketballBlue0v4IndAgObsEnv19x11,
    BasketballGreen5v0IndAgObsEnv19x11,
    BasketballBlue0v5IndAgObsEnv19x11,
    BasketballGreen6v0IndAgObsEnv19x11,
    BasketballBlue0v6IndAgObsEnv19x11,
    BasketballGame8HIndAgObsEnv19x11N4,
    BasketballGame1v2IndAgObsEnv19x11,
    BasketballGame2v1IndAgObsEnv19x11,
    BasketballGame1v3IndAgObsEnv19x11,
    BasketballGame3v1IndAgObsEnv19x11,
    BasketballGame1v4IndAgObsEnv19x11,
    BasketballGame4v1IndAgObsEnv19x11,
    BasketballGame2v3IndAgObsEnv19x11,
    BasketballGame3v2IndAgObsEnv19x11,
    BasketballGame2v4IndAgObsEnv19x11,
    BasketballGame4v2IndAgObsEnv19x11,
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
    AmericanFootballGreen4v0Env16x11,
    AmericanFootballBlue0v4Env16x11,
    AmericanFootballGreen5v0Env16x11,
    AmericanFootballBlue0v5Env16x11,
    AmericanFootballGreen6v0Env16x11,
    AmericanFootballBlue0v6Env16x11,
    AmericanFootball4v4Env16x11,
    AmericanFootball1v2Env16x11,
    AmericanFootball2v1Env16x11,
    AmericanFootball1v3Env16x11,
    AmericanFootball3v1Env16x11,
    AmericanFootball1v4Env16x11,
    AmericanFootball4v1Env16x11,
    AmericanFootball2v3Env16x11,
    AmericanFootball3v2Env16x11,
    AmericanFootball2v4Env16x11,
    AmericanFootball4v2Env16x11,
)
from ..wrappers import GlobalObsWrapper


# -----------------------------------------------------------------------
# Environment registry (v6.8.0)
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
    'MosaicMultiGrid-S-3v3-IndAgObs-v1':    (SoccerGame6HIndAgObsEnv16x11N3,    {}),
    # One-sided Green
    'MosaicMultiGrid-S-G-2v0-IndAgObs-v1':  (SoccerGreen2v0IndAgObsEnv16x11,    {}),
    'MosaicMultiGrid-S-G-3v0-IndAgObs-v1':  (SoccerGreen3v0IndAgObsEnv16x11,    {}),
    # One-sided Blue
    'MosaicMultiGrid-S-B-0v2-IndAgObs-v1':  (SoccerBlue0v2IndAgObsEnv16x11,     {}),
    'MosaicMultiGrid-S-B-0v3-IndAgObs-v1':  (SoccerBlue0v3IndAgObsEnv16x11,     {}),
    'MosaicMultiGrid-S-G-4v0-IndAgObs-v1':  (SoccerGreen4v0IndAgObsEnv16x11,    {}),
    'MosaicMultiGrid-S-B-0v4-IndAgObs-v1':  (SoccerBlue0v4IndAgObsEnv16x11,     {}),
    'MosaicMultiGrid-S-G-5v0-IndAgObs-v1':  (SoccerGreen5v0IndAgObsEnv16x11,    {}),
    'MosaicMultiGrid-S-B-0v5-IndAgObs-v1':  (SoccerBlue0v5IndAgObsEnv16x11,     {}),
    'MosaicMultiGrid-S-G-6v0-IndAgObs-v1':  (SoccerGreen6v0IndAgObsEnv16x11,    {}),
    'MosaicMultiGrid-S-B-0v6-IndAgObs-v1':  (SoccerBlue0v6IndAgObsEnv16x11,     {}),
    # 4v4 symmetric competitive
    'MosaicMultiGrid-S-4v4-IndAgObs-v1':    (SoccerGame8HIndAgObsEnv16x11N4,    {}),


    # Asymmetric competitive variants
    'MosaicMultiGrid-S-1v2-IndAgObs-v1': (SoccerGame1v2IndAgObsEnv16x11, {}),
    'MosaicMultiGrid-S-2v1-IndAgObs-v1': (SoccerGame2v1IndAgObsEnv16x11, {}),
    'MosaicMultiGrid-S-1v3-IndAgObs-v1': (SoccerGame1v3IndAgObsEnv16x11, {}),
    'MosaicMultiGrid-S-3v1-IndAgObs-v1': (SoccerGame3v1IndAgObsEnv16x11, {}),
    'MosaicMultiGrid-S-1v4-IndAgObs-v1': (SoccerGame1v4IndAgObsEnv16x11, {}),
    'MosaicMultiGrid-S-4v1-IndAgObs-v1': (SoccerGame4v1IndAgObsEnv16x11, {}),
    'MosaicMultiGrid-S-2v3-IndAgObs-v1': (SoccerGame2v3IndAgObsEnv16x11, {}),
    'MosaicMultiGrid-S-3v2-IndAgObs-v1': (SoccerGame3v2IndAgObsEnv16x11, {}),
    'MosaicMultiGrid-S-2v4-IndAgObs-v1': (SoccerGame2v4IndAgObsEnv16x11, {}),
    'MosaicMultiGrid-S-4v2-IndAgObs-v1': (SoccerGame4v2IndAgObsEnv16x11, {}),
    # -------------------------------------------------------------------
    # Basketball (BB) — 19x11 grid
    # -------------------------------------------------------------------
    'MosaicMultiGrid-BB-G-1v0-v1':          (BasketballSoloGreenIndAgObsEnv19x11, {}),
    'MosaicMultiGrid-BB-B-0v1-v1':          (BasketballSoloBlueIndAgObsEnv19x11,  {}),
    # Symmetric competitive
    'MosaicMultiGrid-BB-1v1-IndAgObs-v1':   (BasketballGame2HIndAgObsEnv19x11N1,  {}),
    'MosaicMultiGrid-BB-2v2-IndAgObs-v1':   (BasketballGame4HIndAgObsEnv19x11N2,  {}),
    'MosaicMultiGrid-BB-3v3-IndAgObs-v1':   (BasketballGame6HIndAgObsEnv19x11N3,  {}),
    # One-sided Green
    'MosaicMultiGrid-BB-G-2v0-IndAgObs-v1': (BasketballGreen2v0IndAgObsEnv19x11,  {}),
    'MosaicMultiGrid-BB-G-3v0-IndAgObs-v1': (BasketballGreen3v0IndAgObsEnv19x11,  {}),
    # One-sided Blue
    'MosaicMultiGrid-BB-B-0v2-IndAgObs-v1': (BasketballBlue0v2IndAgObsEnv19x11,   {}),
    'MosaicMultiGrid-BB-B-0v3-IndAgObs-v1': (BasketballBlue0v3IndAgObsEnv19x11,   {}),
    'MosaicMultiGrid-BB-G-4v0-IndAgObs-v1': (BasketballGreen4v0IndAgObsEnv19x11,  {}),
    'MosaicMultiGrid-BB-B-0v4-IndAgObs-v1': (BasketballBlue0v4IndAgObsEnv19x11,   {}),
    'MosaicMultiGrid-BB-G-5v0-IndAgObs-v1': (BasketballGreen5v0IndAgObsEnv19x11,  {}),
    'MosaicMultiGrid-BB-B-0v5-IndAgObs-v1': (BasketballBlue0v5IndAgObsEnv19x11,   {}),
    'MosaicMultiGrid-BB-G-6v0-IndAgObs-v1': (BasketballGreen6v0IndAgObsEnv19x11,  {}),
    'MosaicMultiGrid-BB-B-0v6-IndAgObs-v1': (BasketballBlue0v6IndAgObsEnv19x11,   {}),
    # 4v4 symmetric competitive
    'MosaicMultiGrid-BB-4v4-IndAgObs-v1':   (BasketballGame8HIndAgObsEnv19x11N4,  {}),


    # Asymmetric competitive variants
    'MosaicMultiGrid-BB-1v2-IndAgObs-v1': (BasketballGame1v2IndAgObsEnv19x11, {}),
    'MosaicMultiGrid-BB-2v1-IndAgObs-v1': (BasketballGame2v1IndAgObsEnv19x11, {}),
    'MosaicMultiGrid-BB-1v3-IndAgObs-v1': (BasketballGame1v3IndAgObsEnv19x11, {}),
    'MosaicMultiGrid-BB-3v1-IndAgObs-v1': (BasketballGame3v1IndAgObsEnv19x11, {}),
    'MosaicMultiGrid-BB-1v4-IndAgObs-v1': (BasketballGame1v4IndAgObsEnv19x11, {}),
    'MosaicMultiGrid-BB-4v1-IndAgObs-v1': (BasketballGame4v1IndAgObsEnv19x11, {}),
    'MosaicMultiGrid-BB-2v3-IndAgObs-v1': (BasketballGame2v3IndAgObsEnv19x11, {}),
    'MosaicMultiGrid-BB-3v2-IndAgObs-v1': (BasketballGame3v2IndAgObsEnv19x11, {}),
    'MosaicMultiGrid-BB-2v4-IndAgObs-v1': (BasketballGame2v4IndAgObsEnv19x11, {}),
    'MosaicMultiGrid-BB-4v2-IndAgObs-v1': (BasketballGame4v2IndAgObsEnv19x11, {}),
    # -------------------------------------------------------------------
    # American Football (AF) — 16x11 grid
    # -------------------------------------------------------------------
    'MosaicMultiGrid-AF-G-1v0-v1':          (AmericanFootballSoloGreenEnv16x11,   {}),
    'MosaicMultiGrid-AF-B-0v1-v1':          (AmericanFootballSoloBlueEnv16x11,    {}),
    # Symmetric competitive
    'MosaicMultiGrid-AF-1v1-IndAgObs-v1':   (AmericanFootball1v1Env16x11,         {}),
    'MosaicMultiGrid-AF-2v2-IndAgObs-v1':   (AmericanFootball2v2Env16x11,         {}),
    'MosaicMultiGrid-AF-3v3-IndAgObs-v1':   (AmericanFootball3v3Env16x11,         {}),
    # One-sided Green
    'MosaicMultiGrid-AF-G-2v0-IndAgObs-v1': (AmericanFootballGreen2v0Env16x11,    {}),
    'MosaicMultiGrid-AF-G-3v0-IndAgObs-v1': (AmericanFootballGreen3v0Env16x11,    {}),
    # One-sided Blue
    'MosaicMultiGrid-AF-B-0v2-IndAgObs-v1': (AmericanFootballBlue0v2Env16x11,     {}),
    'MosaicMultiGrid-AF-B-0v3-IndAgObs-v1': (AmericanFootballBlue0v3Env16x11,     {}),
    'MosaicMultiGrid-AF-G-4v0-IndAgObs-v1': (AmericanFootballGreen4v0Env16x11,    {}),
    'MosaicMultiGrid-AF-B-0v4-IndAgObs-v1': (AmericanFootballBlue0v4Env16x11,     {}),
    'MosaicMultiGrid-AF-G-5v0-IndAgObs-v1': (AmericanFootballGreen5v0Env16x11,    {}),
    'MosaicMultiGrid-AF-B-0v5-IndAgObs-v1': (AmericanFootballBlue0v5Env16x11,     {}),
    'MosaicMultiGrid-AF-G-6v0-IndAgObs-v1': (AmericanFootballGreen6v0Env16x11,    {}),
    'MosaicMultiGrid-AF-B-0v6-IndAgObs-v1': (AmericanFootballBlue0v6Env16x11,     {}),
    # 4v4 symmetric competitive
    'MosaicMultiGrid-AF-4v4-IndAgObs-v1':   (AmericanFootball4v4Env16x11,         {}),


    # Asymmetric competitive variants
    'MosaicMultiGrid-AF-1v2-IndAgObs-v1': (AmericanFootball1v2Env16x11, {}),
    'MosaicMultiGrid-AF-2v1-IndAgObs-v1': (AmericanFootball2v1Env16x11, {}),
    'MosaicMultiGrid-AF-1v3-IndAgObs-v1': (AmericanFootball1v3Env16x11, {}),
    'MosaicMultiGrid-AF-3v1-IndAgObs-v1': (AmericanFootball3v1Env16x11, {}),
    'MosaicMultiGrid-AF-1v4-IndAgObs-v1': (AmericanFootball1v4Env16x11, {}),
    'MosaicMultiGrid-AF-4v1-IndAgObs-v1': (AmericanFootball4v1Env16x11, {}),
    'MosaicMultiGrid-AF-2v3-IndAgObs-v1': (AmericanFootball2v3Env16x11, {}),
    'MosaicMultiGrid-AF-3v2-IndAgObs-v1': (AmericanFootball3v2Env16x11, {}),
    'MosaicMultiGrid-AF-2v4-IndAgObs-v1': (AmericanFootball2v4Env16x11, {}),
    'MosaicMultiGrid-AF-4v2-IndAgObs-v1': (AmericanFootball4v2Env16x11, {}),
    # -------------------------------------------------------------------
    # Collect (C) — 10x10 grid
    # -------------------------------------------------------------------
    'MosaicMultiGrid-C-IndAgObs-v1':      (CollectGame3HIndAgObsEnv10x10N3, {}),
    'MosaicMultiGrid-C-1v1-IndAgObs-v1':  (CollectGame2HIndAgObsEnv10x10N2, {}),
    'MosaicMultiGrid-C-2v2-IndAgObs-v1':  (CollectGame4HIndAgObsEnv10x10N2, {}),
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
