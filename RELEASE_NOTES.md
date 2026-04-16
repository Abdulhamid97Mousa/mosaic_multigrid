# Release Notes

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/).

## [6.5.0] - 2026-04-16

### Overview

This release fixes the reward signal across all three sport environments so that
MAPPO/IPPO agents can actually learn to score. The previous rewards (+1.0 scoring,
no timeout pressure, no proximity signal) were too weak for the sparse-reward
landscape on these grids. v6.5.0 recalibrates all reward components based on
empirical training results with MAPPO.

**Training validation (MAPPO, 1M steps, 8 parallel envs):**
- American Football 2v2: entropy 1.76→1.09, episodes end in avg 80 steps (vs 300 timeout) ✓
- Soccer 2v2: entropy 1.71→1.30, goal discovered at step ~480k, episodes < 300 steps ✓
- Basketball 3v3: ongoing (3M-step run with `n_epochs=5`, `lr=3e-4`)

### Breaking Changes

#### Scoring reward: +1.0 → +15.0 (all sport environments)

The previous +1.0 scoring reward was dominated by the `ent_coef=0.01` entropy
bonus in MAPPO (which contributed ~0.01 × 2.08 ≈ 0.02 per step × 300 steps =
~6 in entropy value). Raising to +15.0 makes the scoring signal unambiguously
the dominant learning gradient.

Affected classes: `SoccerGameIndAgObsEnv`, `BasketballGameIndAgObsEnv`,
`AmericanFootballEnv` (all walk-in scoring paths).

#### max_steps: 200 → 300 (Soccer and Basketball)

All soccer and basketball IndAgObs RL environments now default to 300 steps
instead of 200. This matches the American Football default set in v6.4.0 and
ensures the MAPPO buffer constraint `buffer_size = parallels × max_steps` can
be set consistently to 2400 (8 × 300) across all three sports.

Affected classes:
- `SoccerGame4HIndAgObsEnv16x11N2` (2v2)
- `SoccerGame2HIndAgObsEnv16x11N2` (1v1)
- `BasketballGame6HIndAgObsEnv19x11N3` (3v3)

### New Features

#### Ball provenance tracking (`_ball_last_carrier_team`)

Blocks the same-team pickup farming exploit where agents chain
`PICKUP → DROP → PICKUP → DROP` repeatedly to accumulate +0.1 pickup rewards
without ever advancing toward the goal. Now the `+0.1` pickup reward is only
awarded when the ball's previous carrier was from the **opposing** team (or
nobody), not the agent's own team.

```python
# In BasketballGameIndAgObsEnv / SoccerGameIndAgObsEnv step():
if carrying and not prev_carrying:
    last_team = self._ball_last_carrier_team.get(ball_idx)
    if last_team is None or last_team != agent.team_index:
        rewards[agent.index] += 0.1   # only genuine first possession
```

#### Pass-chain counter (`_ball_pass_count`)

Penalizes A→B→A bounce passes that farm the first-pass +0.1 reward without
tactical purpose. The counter resets on: ground drop, steal, goal, episode reset.

| Pass number | Reward | Rationale |
|-------------|--------|-----------|
| 1st pass | +0.1 | Reward genuine teamwork |
| 2nd consecutive pass | -0.2 | Penalise A→B→A bounce |
| 3rd+ pass | 0 | Silent — prevents stacking negative signals |

#### Timeout penalty: −1.0 (all sport environments)

Episodes that reach `max_steps` without a winner now apply a −1.0 penalty to
all agents. This creates the pressure that prevents agents from learning to stall.

#### Proximity reward: +0.01/step within Manhattan distance 3 of ball

When an agent is not carrying the ball and is within Manhattan distance 3 of
any ball on the grid (or any agent carrying the ball), it receives +0.01 per
step. This is the critical "missing rung" in the reward ladder that allows
random-walk exploration to discover the ball before the first pickup event.

```python
# Reward ladder (from random walk to scoring):
# +0.01/step near ball  → +0.1 on pickup → +0.05/step toward goal → +15.0 score
```

Currently implemented in `BasketballGameIndAgObsEnv`. Pending: Soccer, American Football.

### Configuration Changes

The following MAPPO training hyperparameters were updated based on empirical
training results and the FXP paper (Feng et al., 2023) recommendations for
symmetric zero-sum multi-agent games:

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| `use_parameter_sharing` | False | **True** | FXP: shared policy eliminates local NE traps in symmetric games |
| `buffer_size` | 200 | **2400** | 8 × 300 = one full episode per env per update (GAE requires complete episodes) |
| `n_minibatch` | 1 | **4** | Prevents trivial critic collapse on small minibatches |
| `ent_coef` | 0.05 | **0.01** | FXP paper gridworld recommendation (0.05 kept policy random) |
| `n_epochs` | 10 | **5** (basketball) | Fewer passes per buffer reduces critic non-stationarity with 6 agents |
| `learning_rate` | 0.0007 | **0.0003** (basketball) | Dampens oscillation from 6-agent joint policy updates |

### Bug Fixes

- **`use_global_state`**: Set to `True` in 1v1 MAPPO configs (was `False`).
  The MAPPO centralized critic should see both agents' observations for proper
  credit assignment in competitive play.
- **Soccer/Basketball 1v1 network size**: Corrected `representation_hidden_size`
  from `[64, 64]` to `[128, 128]` to match the `(7,7,3)=147` flattened observation
  when `MOSAIC_VIEW_SIZE=7`.

### New Training Scripts

Direct-launch v6_5 scripts added for all variants:

| Script | Environment |
|--------|-------------|
| `mappo_american_football_1v1_v6_5.sh` | AF 1v1 competitive |
| `mappo_soccer_1vs1_v6_5.sh` | Soccer 1v1 competitive |
| `mappo_american_football_solo_green_v6_5.sh` | AF Solo Green (curriculum) |
| `mappo_american_football_solo_blue_v6_5.sh` | AF Solo Blue (curriculum) |

All scripts follow the same pattern as the existing 2v2/3v3 v6_5 scripts:
no MOSAIC GUI required, pre-flight check for v6.5.0 source, direct background
launch with stdout/stderr logging.

### Backward Compatibility

- Environment Gymnasium IDs unchanged
- `reward_shaping=False` still produces sparse goal-only rewards
- `max_steps` can be overridden at `gym.make()` time
- Observation shapes unchanged

---

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
