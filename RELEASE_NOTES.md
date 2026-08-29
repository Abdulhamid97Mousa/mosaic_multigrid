# Release Notes

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/).

---

## [7.0.0] - 2026-08-28

### Overview

**Breaking release.** `TeamObs` environments and SMAC-style teammate
observations are removed from the package entirely. No deprecation window.
Existing training code that used `TeamObs` variants should migrate to
`IndAgObs` variants combined with algorithm-level agent identity
(EPyMARL `obs_agent_id`).

### Breaking Changes

**All `MosaicMultiGrid-*-TeamObs-v1` env IDs are removed.**
`gymnasium.make('MosaicMultiGrid-BB-2v2-TeamObs-v1')` now raises
`gymnasium.error.NameNotFound`. 42 registered IDs affected across the 4
sport families (Soccer, Basketball, American Football, Collect).

**`TeamObsWrapper` class removed from `mosaic_multigrid.wrappers`.**
`from mosaic_multigrid.wrappers import TeamObsWrapper` now raises
`ImportError`.

**Removed environment classes (41 total):** all `*TeamObsEnv` classes
covering symmetric competitive, one-sided cooperative, and 4v4 formats
across Soccer, Basketball, American Football, and Collect.

**Removed test file:** `tests/test_teamobs.py`.

### Migration

For symmetry-breaking in parameter-sharing MARL, use **EPyMARL-style
`obs_agent_id`** — a one-hot agent identity vector appended inside your
training algorithm, not at the environment level. Reference:

> Papoudakis, G., Christianos, F., Schäfer, L., and Albrecht, S. V. (2021).
> Benchmarking multi-agent deep reinforcement learning algorithms in
> cooperative tasks. NeurIPS Track on Datasets and Benchmarks.

Existing code that used TeamObs should switch to:

1. Use the standard `IndAgObs` env variant.
2. Inside the training loop, append `jnp.eye(n_agents)` (JAX) or
   `torch.eye(n_agents)` (PyTorch) to each agent's observation before
   feeding to the shared policy network.

### Changed

- `README.md`, `FOOTBALL.md`, `BASKETBALL.md`, `AMERICAN_FOOTBALL.md`,
  `COLLECT_IMPROVEMENTS.md`, `OBSERVATION_MODELS.md`, and
  `PARTIAL_OBSERVABILITY.md` no longer mention TeamObs or SMAC.
- `OBSERVATION_MODELS.md` retitled to *Observation Model: Independent Agent
  Observations (IndObs)* and truncated to the IndObs-only content
  (679 → 254 lines).
- New figures `figures/envs_{BB,AF,S}_v2.png` show 9 canonical envs per
  sport in a 3×3 grid (2 solo + 3 symmetric competitive + 4 one-sided
  cooperative). Original `envs_{BB,AF,S}.png` kept for historical
  reference but unlinked from README.
- New `scripts/generate_env_figures.py` for reproducible figure
  regeneration.
- `figures/Gym-MosaicMultiGrid-Soccer-TeamObs-v0.png` removed.

### Compatibility Fallback

Users who cannot migrate immediately can pin to the previous release:

```
pip install mosaic_multigrid==6.9.0
```

### Registered Environment Summary (post-7.0.0)

51 environments remain registered across 4 sports:

- **Soccer (S):** 16 envs — 2 solo, 1v1, 2v2, 3v3, 4v4, and one-sided
  cooperative from 2v0/0v2 up to 6v0/0v6 (all IndAgObs).
- **Basketball (BB):** 16 envs — same structure as Soccer.
- **American Football (AF):** 16 envs — same structure as Soccer.
- **Collect (C):** 3 envs — individual 3-agent, 1v1 team, 2v2 team.

Only the 9-env "canonical" subset per sport is showcased in the README
figures; the higher-agent-count variants (4v0 → 6v0, 4v4) remain
registered for research use.

---

## [6.7.0] - 2026-05-15

### Overview

Enforces correct goal geometry across all JAX training environments and
the Gymnasium sport environments. The central change is a **mandatory
`goal_rows` parameter** in every JAX environment constructor: omitting it
raises an `AssertionError` with a precise error message pointing to the
correct flag, preventing silent misconfigurations that caused the invisible-
goals training regression discovered in the V2 campaign. Observation
encoding is now defined through named constants asserted unique at
import time, eliminating the class of bug where goals were encoded
identically to floor cells.

### Breaking Changes

#### JAX environments: `goal_rows` is now mandatory

`SoccerJAX`, `AmericanFootballJAX`, and `BasketballJAX` no longer fall back
to a silent default when `goal_rows` is not supplied. Passing `goal_rows=None`
(the previous default) now raises:

```
AssertionError: [SoccerJAX] goal_rows must be explicitly provided.
  Expected for soccer: goal_rows=[4, 5, 6]
  In training scripts: --goal-rows 4 5 6
  Use DEFAULT_GOAL_ROWS if you want the canonical value.
```

**Correct canonical values:**

| Environment | `goal_rows` | Description |
|---|---|---|
| `SoccerJAX` | `[4, 5, 6]` | 3-cell goal centred at y=5 |
| `AmericanFootballJAX` | `list(range(1, 10))` | Full end-zone column (rows 1–9) |
| `BasketballJAX` | `[5]` | Single hoop at centre row |

Additional validation fires at construction time:
- Non-empty list required
- No duplicate row values
- All values in `[1, HEIGHT−2]` (playable rows only)

**Migration — any direct instantiation must now pass `goal_rows`:**

```python
# Before (silently used DEFAULT_GOAL_ROWS — now raises AssertionError)
env = SoccerJAX(variant='G-2v0', view_size=7)

# After
from jaxmarl_worker.environments.soccer_jax import DEFAULT_GOAL_ROWS
env = SoccerJAX(variant='G-2v0', view_size=7, goal_rows=[4, 5, 6])
```

### New Features

#### Observation encoding constants with import-time integrity assertions

All three JAX environment modules now define named constants for every
observation encoding value and assert their uniqueness at module load time:

```python
_OBJ_FLOOR      = 1.0   # passable empty cell
_OBJ_WALL       = 2.0   # impassable border
_OBJ_GREEN_GOAL = 5.0   # STATIC_GRID base==5
_OBJ_BLUE_GOAL  = 6.0   # STATIC_GRID base==6
_OBJ_BALL       = 7.0   # loose ball
_OBJ_AGENT      = 10.0  # agent (overrides all)
```

The following assertions fire at `import` time — before any training starts:

```python
assert len(_ALL_OBJ_VALS) == len(set(_ALL_OBJ_VALS)), "collision"
assert _OBJ_GREEN_GOAL != _OBJ_FLOOR, "Invisible goals bug"
assert _OBJ_BLUE_GOAL  != _OBJ_FLOOR, "Invisible goals bug"
```

This prevents the regression from the V2 campaign (May 2026) where
`obj=1.0` was accidentally used for goals, encoding them identically
to empty floor cells and producing invisible-goals checkpoints.

#### `--goal-rows` CLI flag in all algorithm files

All four training algorithm files now accept a `--goal-rows` flag:

```bash
# Soccer
--goal-rows 4 5 6

# American Football (full end-zone column)
--goal-rows 1 2 3 4 5 6 7 8 9

# Basketball
--goal-rows 5
```

Files updated: `mappo_indagobs_scan.py`, `mappo_teamobs_scan.py`,
`ippo_scan.py`, `happo_scan.py`.

All 18 V2 training scripts now pass `--goal-rows` explicitly, making
goal geometry visible in the launch command and auditable from logs.

#### Correct goal geometry restored in Gymnasium Soccer environments

All `SoccerGameIndAgObsEnv` subclasses have been updated from a single-cell
goal at y=5 to a **3-cell goal arc at y=4,5,6** on each side of the pitch,
matching the JAX training environments:

```python
# Before (1-cell goal — inconsistent with JAX)
goal_pos=[[1, 5], [14, 5]],  goal_index=[1, 2]

# After (3-cell goal arc — consistent with JAX)
goal_pos=[[1,4],[1,5],[1,6],[14,4],[14,5],[14,6]],  goal_index=[1,1,1,2,2,2]
```

`_target_goal_pos()` updated to return the **centre cell** (y=5) regardless
of goal arc width, ensuring reward-shaping gradients point to the middle of
the goal rather than its top edge.

Affected classes (all `SoccerGameIndAgObsEnv` subclasses on the 16×11 grid):
`SoccerGame4HIndAgObsEnv16x11N2`, `SoccerGame2HIndAgObsEnv16x11N2`,
`SoccerSoloGreenIndAgObsEnv16x11`, `SoccerSoloBlueIndAgObsEnv16x11`,
`SoccerGreen2v0IndAgObsEnv16x11`, `SoccerBlue0v2IndAgObsEnv16x11`,
`SoccerGreen3v0IndAgObsEnv16x11`, `SoccerBlue0v3IndAgObsEnv16x11`,
`SoccerGame6HIndAgObsEnv16x11N3`.

#### New Gymnasium environment: `MosaicMultiGrid-S-1v1-TeamObs-v1`

`Soccer1v1TeamObsEnv` (TeamObs wrapper over `SoccerGame2HIndAgObsEnv16x11N2`)
added and registered, completing the Soccer environment matrix to 16 registered
IDs (matching Basketball and American Football).

#### `envs_S.png` regenerated — 4×4 grid (16 variants)

`figures/envs_S.png` updated to the same 4×4 layout as `envs_BB.png` and
`envs_AF.png`, showing all 16 registered Soccer Gym IDs with 3-cell goals
visible in every panel.

### Backward Compatibility

- Gymnasium environment IDs unchanged (16 Soccer, same IDs)
- `reward_shaping=False` still produces sparse goal-only rewards
- Observation spaces unchanged
- **JAX environments: `goal_rows=None` is no longer accepted** — see Breaking Changes

---

## [6.6.0] - 2026-05-07

### Overview

Adds the missing "find-the-ball" reward gradient across all three sport
environments. Symmetric ball-approach shaping fires when an agent is not
carrying any ball, mirroring the existing carrying-toward-goal shaping.
This fixes the post-goal collapse (P(2nd pickup | 1st goal) ~52–68%)
by giving the policy a navigation gradient during ball-finding phases.

### New Features

#### Symmetric ball-approach shaping (all sport environments)

Added `+0.01 × Δdist` reward for agents **not carrying** a ball, computed
toward the nearest loose ball on the grid. Mirrors the existing
`+0.01 × Δdist` reward toward the goal while carrying.

| Phase | Before 6.6.0 | 6.6.0+ |
|---|---|---|
| Carrying | gradient toward goal | gradient toward goal (unchanged) |
| Not carrying | no gradient | gradient toward loose ball |

The shaping block is now structurally identical across Soccer, Basketball,
and American Football — one mental model applies to all three.

### Consistency Refactor

#### American Football: 1D → 2D distance shaping

`AmericanFootballEnv` previously used 1D x-only shaping via
`_prev_x: dict[int, int]` and `_target_endzone_x()`. Upgraded to 2D
Manhattan distance matching Soccer and Basketball:

- `_prev_x` → `_prev_pos: dict[int, tuple[int, int]]`
- `_target_endzone_x()` → `_target_endzone_pos()` returning `(target_x, height // 2)`
- Carrying reward: `0.01 × (prev_manhattan − curr_manhattan)`

Agents now get a small reward for converging to the centre row while
carrying, in addition to horizontal progress toward the end zone.
Scoring mechanics are unchanged (triggers on any cell of the end-zone column).

### Bug Fixes

#### Basketball base-class latent crash

`BasketballGameEnv._handle_pickup` (base class) wrote to
`self.steals_completed` which is only initialised in `BasketballGameIndAgObsEnv`.
Direct instantiation of the base class raised `AttributeError` on first steal.
Tracking removed from the base class — now matches `SoccerGameEnv` which
has always been silent in the base steal.

### Cleanup

Removed `_prev_carrying: dict[int, bool]` from all three sport environments.
The variable was written every step but never read — a remnant of a
pickup-bonus term removed in 6.5.0.

### Backward Compatibility

- Environment Gymnasium IDs unchanged
- Observation spaces unchanged
- `reward_shaping=False` still produces sparse goal-only rewards
- Checkpoints from 6.5.0 remain loadable but are expected to underperform
  fresh 6.6.0 training due to the new ball-approach gradient

---

## [6.5.0] - 2026-04-18

### Overview

This release fixes the reward signal across all sport environments (American Football,
Soccer, Basketball) and Collect to eliminate reward hacking and incentivise fast, decisive
play.

**Reward hacking discovered via MAPPO/IPPO evaluation:** Agents learned to spam
`pickup → drop → pickup` earning `+0.10` per cycle indefinitely without advancing toward
the goal. In 1v1 competitive settings IPPO achieved ep_ret=14.37 over 200 steps purely
from pickup/drop cycles, completely ignoring the +15 scoring reward.

**v6.5.0 removes the exploitable components and replaces them with hack-proof alternatives.**

**Training validation (MAPPO-scan, JAX, 131M steps per variant):**
- Solo (G-1v0, B-0v1): ep_ret ≈ 1.1–1.2, entropy ≈ 0.37 (converged)
- 2v2: IPPO scored in 32 steps with emergent role specialisation
- 1v1/3v3 competitive: near-zero ep_ret at 2000 updates — need more training

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

#### Removed: `+0.1` pickup reward and `+0.01` proximity reward (all sport environments)

The pickup reward was the root cause of the hacking loop — removing it eliminates all
pickup/drop cycling. The proximity reward was also removed since it was decoupled from
possession and therefore not tied to actual goal progress.

`_ball_last_carrier_team` tracking in `AmericanFootballEnv` is also removed (no longer
needed since the reward it guarded is gone).

#### Added: Time-efficiency bonus on scoring (all sport and Collect environments)

```
score_reward = base + (max_steps - step) × 0.05
```

Scoring at step 10 yields ≈ +14.5 bonus; at step 290 yields ≈ +0.5. This creates
a continuous pressure to advance quickly without making late-game goals worthless.
Applied to all scoring paths including JAX environments (`MAX_STEPS - new_state.step`).

#### Added: `+0.3` steal reward (all sport environments)

Fires only when an agent takes the ball from an **opponent who is currently carrying it**.
Cannot be farmed since it requires a live opponent in possession.

```python
# JAX: steal detection via ball_carried_by state
prev_carrier = state.ball_carried_by[safe_bidx]
is_steal = picked_up & (prev_carrier >= 0) & (prev_carrier_team != team)
```

Also fixed: Basketball `_handle_pickup` now restricts steals to opponents only.

#### Kept: `+0.01 × Δdistance while carrying` (all sport environments)

Carrying-gated distance shaping remains. Drop resets it — pickup/drop cycling yields zero.

### New Features

#### Ball provenance tracking (`_ball_last_carrier_team`)

Blocks the same-team pickup farming exploit. Retained in Soccer and Basketball
for internal bookkeeping.

```python
# In BasketballGameIndAgObsEnv / SoccerGameIndAgObsEnv step():
if carrying and not prev_carrying:
    last_team = self._ball_last_carrier_team.get(ball_idx)
    if last_team is None or last_team != agent.team_index:
        rewards[agent.index] += 0.1   # only genuine first possession
```

#### Timeout penalty: −1.0 (all sport environments)

Episodes that reach `max_steps` without a winner apply a −1.0 penalty to
all agents, preventing stalling strategies.

### Configuration Changes

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| `use_parameter_sharing` | False | **True** | Eliminates local NE traps in symmetric games |
| `buffer_size` | 200 | **2400** | One full episode per env per update |
| `n_minibatch` | 1 | **4** | Prevents trivial critic collapse |
| `ent_coef` | 0.05 | **0.01** | Reduces entropy dominance |

### Bug Fixes

- **`use_global_state`**: Set to `True` in 1v1 MAPPO configs.
- **Soccer/Basketball 1v1 network size**: Corrected `representation_hidden_size`
  from `[64, 64]` to `[128, 128]` to match the `(7,7,3)=147` flattened
  observation at `view_size=7`.

### Backward Compatibility

- Environment Gymnasium IDs unchanged
- `reward_shaping=False` still produces sparse goal-only rewards
- `max_steps` can be overridden at `gym.make()` time
- Observation shapes unchanged
