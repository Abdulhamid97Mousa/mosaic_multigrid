"""Action enumeration for the MOSAIC multigrid environment.

Action table (mosaic_multigrid v2, this fork)
----------------------------------------------
  noop    = 0   — No operation (see rationale below)
  left    = 1   — Turn 90° counter-clockwise
  right   = 2   — Turn 90° clockwise
  forward = 3   — Move one cell in the facing direction
  pickup  = 4   — Pick up an object (or steal ball from opponent)
  drop    = 5   — Drop / pass an object
  toggle  = 6   — Toggle / activate an object
  done    = 7   — Signal task completion

Comparison across forks
------------------------
  Upstream Fickinger 2020 : still=0  left=1  …  done=7  (8 actions)
  mosaic_multigrid v1     : left=0   right=1 …  done=6  (7 actions, no no-op)
  mosaic_multigrid v2     : noop=0   left=1  …  done=7  (8 actions, explicit no-op)  ← this file
  multigrid-ini Oguntola  : left=0   right=1 …  done=6  (7 actions)
  MeltingPot DeepMind     : NOOP=0   FORWARD=1 … INTERACT=7  (8 actions)

Why noop=0 was added (AEC + Parallel API compatibility)
---------------------------------------------------------
In AEC (Agent-Environment Cycle) mode, only one agent acts per physics step.
All other agents must still submit a *valid* action so the environment can
advance, but they must not change state.  Without a dedicated no-op:

  - The previous action 0 was ``left`` (turn left).
  - Non-acting agents would silently rotate on every step — corrupting the
    episode and invalidating any research comparing AEC vs. Parallel results.

The fix is a dedicated ``noop`` action that the environment treats as "do
nothing, advance time only."  This design is directly inspired by MeltingPot
(Google DeepMind), which uses ``NOOP=0`` for the same reason.

The ``done`` action (index 7) signals intentional task completion and is
semantically different from ``noop``.  Both cause no physical movement, but
only ``noop`` should be used by non-acting agents in AEC mode.

Migration note
--------------
All action indices shifted **up by 1** compared to mosaic_multigrid v1.
Any pre-trained policy or hardcoded action mapping from v1 will need updating:

  v1 → v2:  left=0→1  right=1→2  forward=2→3  pickup=3→4
             drop=4→5  toggle=5→6  done=6→7
"""
from __future__ import annotations

import enum


class Action(enum.IntEnum):
    """Enumeration of possible agent actions."""
    noop    = 0            #: No operation — AEC compatibility (non-acting agents wait)
    left    = enum.auto()  #: Turn 90° counter-clockwise
    right   = enum.auto()  #: Turn 90° clockwise
    forward = enum.auto()  #: Move one cell forward
    pickup  = enum.auto()  #: Pick up an object
    drop    = enum.auto()  #: Drop an object
    toggle  = enum.auto()  #: Toggle / activate an object
    done    = enum.auto()  #: Done completing task
