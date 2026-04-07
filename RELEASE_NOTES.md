# Release Notes

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/).

## [6.4.0] - 2026-04-08

### New Features

#### SMAC-Style Reward Shaping (All Three Sports)

All sport environments now include dense reward shaping following the SMAC
(Samvelyan et al., 2019) pattern. Controlled by `reward_shaping=True` (default).

Without shaping, agents receive reward only on scoring. Random agents score
~0.6% of episodes, providing no gradient signal for PPO/MAPPO/IPPO. With
shaping, PPO reaches 56% touchdown rate within 150 iterations.

**Reward components (when `reward_shaping=True`):**

| Event | Reward | All Sports |
|-------|--------|------------|
| Pick up ball | +0.1 | Per possession |
| Move toward goal while carrying | +0.01 per step | AF: 1D column, Soccer/Basketball: 2D Manhattan |
| Move away from goal while carrying | -0.01 per step | Same |
| Score (touchdown/goal) | +1.0 | Existing |

**Distance metrics:**
- American Football: 1D column distance via `_target_endzone_x()`
- Soccer: 2D Manhattan distance to goal square via `_target_goal_pos()`
- Basketball: 2D Manhattan distance to goal square via `_target_goal_pos()`

```python
# Training (dense rewards, default)
env = AmericanFootball1v1Env16x11(reward_shaping=True)

# Evaluation (sparse, goal-only rewards)
env = AmericanFootball1v1Env16x11(reward_shaping=False)
```

#### Configurable `view_size` for American Football
All American Football variants now accept `view_size` as a constructor
parameter (default: 3). Use `view_size=7` for 39% field coverage.

#### American Football Infrastructure (matching Soccer/Basketball IndAgObs)
- 10-step dual steal cooldown (prevents ping-pong exploit)
- Teleport passing (ball teleports to random teammate)
- Event tracking: `goal_scored_by`, `pass_completed`, `steal_completed`
- Per-agent telemetry: `position` and `carrying` per step

### Bug Fixes

- **Fixed `ObjectGoal.can_overlap()`**: Changed from `False` to `True`.
  This was a critical v6.3.0 bug: walk-in scoring was dead code because
  agents physically could not step onto goal squares. Soccer and Basketball
  goals now allow overlap, enabling the walk-in scoring mechanic that v6.3.0
  introduced but never actually worked.

- **Fixed `see_through_walls`**: Changed from `True` to `False` for American
  Football, matching Soccer/Basketball behavior.

- **Fixed `goal_scored_by` initialization**: Added missing list initialization
  in `reset()`, fixing `AttributeError` on first touchdown.

### Documentation
- **NEW: `AMERICAN_FOOTBALL.md`**: Grid layout, scoring, reward shaping, SMAC comparison.
- **Renamed: `SOCCER_IMPROVEMENTS.md` -> `FOOTBALL.md`**

### Backward Compatibility
- Default `view_size=3` preserved (no observation shape change unless overridden)
- Default `reward_shaping=True` is new behavior; set `False` for v6.3.0 behavior
- `ObjectGoal.can_overlap()=True` changes walk-in scoring from broken to working
- All Gymnasium environment IDs unchanged
- 303 tests pass (0 failures)

---

Given a breaking change in this release, I've decided to bump the version from 6.2.0 to 6.3.0.

## [6.3.0] - 2025-03-14
### Breaking Changes

#### Walk-In Scoring (All Sports)
**Previous behavior (v6.0-6.2):**
- Agents had to execute the `DROP` action while **facing** the goal square to score
- Required precise positioning and action sequencing (navigate → face ball → pickup → navigate to goal → face goal → drop)
- Complex learning objective (6 steps)

**New behavior (v6.3+):**
- Agents score by **walking into the goal square** while carrying the ball
- Simplified to navigate → pickup → navigate to goal → score (3 steps)
- **DROP action** no longer scores - it only handles teleport passing and ground drops
- More intuitive (like real sports: carry ball into goal area)
- Faster convergence (reduced action space complexity)

#### Goal Representation (Consistent Across All Sports)
| Sport | Goal Type | Size | Scoring Method |
|-------|-----------|------|----------------|
| **Soccer** | Single square | 1x1 tile | Walk into goal square while carrying ball |
| **Basketball** | Single square | 1x1 tile | Walk into goal square while carrying ball |
| **American Football** | End zone | Full vertical column (1x9 tiles) | Walk into end zone column while carrying ball |

#### DROP Action Changes
**v6.0-6.2:**
- DROP = score at goal OR pass to teammate OR drop on ground

- Priority: score > pass > drop

**v6.3+:**
- DROP = teleport pass to teammate OR drop on ground (NO scoring)
- Scoring happens automatically when agent walks into goal while carrying
- Simplifies action logic

### Fixed Issues
- **Legacy multigrid.py removed**: The old `multigrid.py` file that used deprecated Gym API has been renamed to `multigrid_legacy.py.bak`
- **Test updates**: All tests updated to match v6.3.0 walk-in scoring behavior
  - American Football: Game terminates when `goals_to_win` reached (default 2)
  - Soccer/Basketball: Walk-in scoring tests pass
  - Goal tracking (`goal_scored_by`) properly populated in all environments
### Migration Guide
#### For Users with v6.2.x code
1. **Scoring logic**: Remove any DROP-based scoring logic - scoring is now automatic
2. **Action sequences**: Simplify from 6-step to 3-step sequences
   ```python
   # OLD (v6.2)
   # 1. Navigate to ball
   # 2. Face ball
   # 3. Execute PICKUP
   # 4. Navigate to goal
   # 5. Face goal square
   # 6. Execute DROP to score
   
   # NEW (v6.3)
   # 1. Navigate to ball
   # 2. Execute PICKUP
   # 3. Navigate to goal square -> AUTOMATIC SCORE!
   ```

3. **DROP action**: Use DROP only for passing to teammates,4. **Goal detection**: Agents score when they step INTO the goal square (not when facing it)
#### For Trained Models
- **Checkpoints incompatible**: Models trained on v6.2.x will need retraining
- **Reward distribution unchanged**: Team rewards still work the same way
- **Episode termination unchanged**: First team to `goals_to_win` (default 2) still wins
### Documentation Updates
- **SOCCER_IMPROVEMENTS.md**: Added v6.3.0 section with walk-in scoring details
- **BASKETBALL.md**: Added v6.3.0 section with walk-in scoring details
- **OBSERVATION_MODELS.md**: Added v6.3.0 update header
- **RELEASE_NOTES.md**: Created this file for version tracking
### Backward Compatibility
- Old environment names still work (e.g., `SoccerGame4HEnhancedEnv16x11N2` is now an alias for `SoccerGame4HIndAgObsEnv16x11N2`)
- All Gymnasium environment IDs remain unchanged
- Observation space unchanged (still 3-channel uint8 images)
### Next Steps
1. Update `pyproject.toml` version to `6.3.0`
2. Build distribution: `python -m build`
3. Upload to PyPI: `twine upload dist/*`
4. Update documentation with migration guide
