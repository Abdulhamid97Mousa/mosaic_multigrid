"""MOSAIC multigrid environments — registration and exports.

All environments are registered with Gymnasium. Create via:

    gymnasium.make('MosaicMultiGrid-Soccer-2v2-IndAgObs-v1')
    gymnasium.make('MosaicMultiGrid-AmericanFootball-2v2-IndAgObs-v1')
    gymnasium.make('MosaicMultiGrid-Basketball-3v3-IndAgObs-v1')

Naming scheme (v6.5.0+):
    MosaicMultiGrid-<Sport>-<Format>-<ObsVariant>-v1
      Sport:      Soccer | Basketball | AmericanFootball | Collect
      Format:     1v1 | 2v2 | 3v3 | Solo-Green | Solo-Blue
      ObsVariant: IndAgObs (always explicit) | TeamObs

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
)
from .collect_game import (
    CollectGame3HIndAgObsEnv10x10N3,
    CollectGame4HIndAgObsEnv10x10N2,
    CollectGame2HIndAgObsEnv10x10N2,
)
from .basketball_game import (
    BasketballGame6HIndAgObsEnv19x11N3,
    BasketballSoloGreenIndAgObsEnv19x11,
    BasketballSoloBlueIndAgObsEnv19x11,
)
from .american_football_game import (
    AmericanFootballSoloGreenEnv16x11,
    AmericanFootballSoloBlueEnv16x11,
    AmericanFootball1v1Env16x11,
    AmericanFootball2v2Env16x11,
    AmericanFootball3v3Env16x11,
)
from ..wrappers import TeamObsWrapper, GlobalObsWrapper


# -----------------------------------------------------------------------
# TeamObs environment classes — SMAC-style teammate awareness
#
# Thin wrappers that compose an IndAgObs base env with TeamObsWrapper,
# adding structured teammate features (positions, directions, has_ball)
# to each agent's observation dict.
#
# Only defined for team-based (2v2+) environments. 1v1 and Solo variants
# have no teammates, so no TeamObs exists for them.
# -----------------------------------------------------------------------

class SoccerTeamObsEnv(TeamObsWrapper):
    """Soccer 2v2 (16x11, IndAgObs) with SMAC-style teammate awareness."""
    metadata = SoccerGame4HIndAgObsEnv16x11N2.metadata

    def __init__(self, **kwargs):
        super().__init__(SoccerGame4HIndAgObsEnv16x11N2(**kwargs))


class Collect2v2TeamObsEnv(TeamObsWrapper):
    """Collect 2v2 (10x10, IndAgObs) with SMAC-style teammate awareness."""
    metadata = CollectGame4HIndAgObsEnv10x10N2.metadata

    def __init__(self, **kwargs):
        super().__init__(CollectGame4HIndAgObsEnv10x10N2(**kwargs))


class Basketball3v3TeamObsEnv(TeamObsWrapper):
    """Basketball 3v3 (19x11, IndAgObs) with SMAC-style teammate awareness."""
    metadata = BasketballGame6HIndAgObsEnv19x11N3.metadata

    def __init__(self, **kwargs):
        super().__init__(BasketballGame6HIndAgObsEnv19x11N3(**kwargs))


class AmericanFootball2v2TeamObsEnv(TeamObsWrapper):
    """American Football 2v2 (16x11, IndAgObs) with SMAC-style teammate awareness."""
    metadata = AmericanFootball2v2Env16x11.metadata

    def __init__(self, **kwargs):
        super().__init__(AmericanFootball2v2Env16x11(**kwargs))


class AmericanFootball3v3TeamObsEnv(TeamObsWrapper):
    """American Football 3v3 (16x11, IndAgObs) with SMAC-style teammate awareness."""
    metadata = AmericanFootball3v3Env16x11.metadata

    def __init__(self, **kwargs):
        super().__init__(AmericanFootball3v3Env16x11(**kwargs))


# -----------------------------------------------------------------------
# Environment registry (v6.5.0)
# -----------------------------------------------------------------------

CONFIGURATIONS: dict[str, tuple[type, dict]] = {

    # -------------------------------------------------------------------
    # Soccer — 16x11 grid, walk-in scoring at goal square
    # -------------------------------------------------------------------
    'MosaicMultiGrid-Soccer-Solo-Green-IndAgObs-v1': (SoccerSoloGreenIndAgObsEnv16x11, {}),
    'MosaicMultiGrid-Soccer-Solo-Blue-IndAgObs-v1':  (SoccerSoloBlueIndAgObsEnv16x11,  {}),
    'MosaicMultiGrid-Soccer-1v1-IndAgObs-v1':        (SoccerGame2HIndAgObsEnv16x11N2,   {}),
    'MosaicMultiGrid-Soccer-2v2-IndAgObs-v1':        (SoccerGame4HIndAgObsEnv16x11N2,   {}),
    'MosaicMultiGrid-Soccer-2v2-TeamObs-v1':         (SoccerTeamObsEnv,                 {}),

    # -------------------------------------------------------------------
    # Basketball — 19x11 grid, walk-in scoring at basket square
    # -------------------------------------------------------------------
    'MosaicMultiGrid-Basketball-Solo-Green-IndAgObs-v1': (BasketballSoloGreenIndAgObsEnv19x11, {}),
    'MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v1':  (BasketballSoloBlueIndAgObsEnv19x11,  {}),
    'MosaicMultiGrid-Basketball-3v3-IndAgObs-v1':        (BasketballGame6HIndAgObsEnv19x11N3,   {}),
    'MosaicMultiGrid-Basketball-3v3-TeamObs-v1':         (Basketball3v3TeamObsEnv,              {}),

    # -------------------------------------------------------------------
    # American Football — 16x11 grid, walk-in scoring at end zone column
    # -------------------------------------------------------------------
    'MosaicMultiGrid-AmericanFootball-Solo-Green-IndAgObs-v1': (AmericanFootballSoloGreenEnv16x11, {}),
    'MosaicMultiGrid-AmericanFootball-Solo-Blue-IndAgObs-v1':  (AmericanFootballSoloBlueEnv16x11,  {}),
    'MosaicMultiGrid-AmericanFootball-1v1-IndAgObs-v1':        (AmericanFootball1v1Env16x11,        {}),
    'MosaicMultiGrid-AmericanFootball-2v2-IndAgObs-v1':        (AmericanFootball2v2Env16x11,        {}),
    'MosaicMultiGrid-AmericanFootball-2v2-TeamObs-v1':         (AmericanFootball2v2TeamObsEnv,      {}),
    'MosaicMultiGrid-AmericanFootball-3v3-IndAgObs-v1':        (AmericanFootball3v3Env16x11,        {}),
    'MosaicMultiGrid-AmericanFootball-3v3-TeamObs-v1':         (AmericanFootball3v3TeamObsEnv,      {}),

    # -------------------------------------------------------------------
    # Collect — 10x10 grid, collect items competitively
    # -------------------------------------------------------------------
    'MosaicMultiGrid-Collect-IndAgObs-v1':      (CollectGame3HIndAgObsEnv10x10N3, {}),
    'MosaicMultiGrid-Collect-1v1-IndAgObs-v1':  (CollectGame2HIndAgObsEnv10x10N2, {}),
    'MosaicMultiGrid-Collect-2v2-IndAgObs-v1':  (CollectGame4HIndAgObsEnv10x10N2, {}),
    'MosaicMultiGrid-Collect-2v2-TeamObs-v1':   (Collect2v2TeamObsEnv,            {}),
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
