# mosaic_multigrid

**Research-grade multi-agent gridworld environments for reproducible RL experiments.**

A maintained fork of [gym-multigrid](https://github.com/ArnaudFickinger/gym-multigrid) by Arnaud Fickinger (2020), modernized to the Gymnasium API with Numba JIT-accelerated observations, reproducible seeding, and multi-agent framework adapters.

This fork is developed as part of the [MOSAIC](https://github.com/Abdulhamid97Mousa/MOSAIC) project (Multi-Agent Orchestration System).

## Design Philosophy: Best of Both Worlds

**mosaic_multigrid = gym-multigrid game design + INI multigrid modern infrastructure**

We kept the **challenging partial observability** (`view_size=3`) that makes Soccer/Collect interesting for competitive multi-agent research, while adopting **modern API and optimizations** from INI multigrid standards.

### What We Kept from gym-multigrid (Fickinger 2020)

1. **Partial observability** - `view_size=3` for Soccer/Collect (challenging team coordination)
2. **Game mechanics** - Ball passing, stealing, scoring, team rewards
3. **Research continuity** - Comparable with original papers

### What We Adopted from INI multigrid (2022+)
1. **Gymnasium 1.0+ API** - Modern 5-tuple dict-keyed observations
2. **3-channel encoding** - `[type, color, state]` format (not 6-channel)
3. **Agent class design** - Separate from WorldObj, cleaner architecture
4. **pygame rendering** - Modern window system (not matplotlib)
5. **Modular structure** - ~20 focused modules (not 1442-line monolith)

### What We Built (Our Contributions)

1. **Reproducibility fix** - Fixed critical global RNG bug
2. **Numba JIT optimization** - 10-100× faster observation generation
3. **Comprehensive tests** - 130 tests covering all functionality
4. **Framework adapters** - RLlib, PettingZoo integration
5. **Observation wrappers** - FullyObs, ImgObs, OneHot, SingleAgent
---

## What Changed from Upstream: The Full Story

Showing how we combined the best of both packages:

| Aspect | gym-multigrid (Fickinger 2020) | INI multigrid (Oguntola 2023) | **mosaic_multigrid (This Fork)** |
|--------|-------------------------------|-------------------------------|----------------------------------|
| **API** | Old Gym 4-tuple, list-based | Gymnasium 5-tuple, dict-keyed |  **Gymnasium 5-tuple, dict-keyed** (from INI) |
| **Actions** | 8 (still=0..done=7) | 7 (left=0..done=6) |  **7 actions, no "still"** (from INI) |
| **Observations** | `(view, view, 6)` list | `(view, view, 3)` dict |  **`(view, view, 3)` dict** (from INI) |
| **view_size** | **3** (Soccer/Collect) | **7** (default) |  **3 (KEPT from gym-multigrid)** for competitive challenge |
| **Game Logic** | **Soccer, Collect, team rewards** | Exploration tasks (no team games) |  **Soccer, Collect** (from gym-multigrid) |
| **`reset()`** | `List[obs]` | `(Dict[obs], Dict[info])` |  **`(Dict[obs], Dict[info])`** (from INI) |
| **`step()`** | `(List[obs], ndarray, bool, dict)` | `(Dict, Dict, Dict, Dict, Dict)` |  **5-tuple per-agent dicts** (from INI) |
| **Render** | `render(mode='human')` param | `render_mode` constructor param |  **`render_mode` constructor** (from INI) |
| **Seeding** | `env.seed(42)` + **broken global RNG** | `reset(seed=42)` + `self.np_random` |  **Fixed seeding** (from INI) + **bug fix** (ours) |
| **Window** | matplotlib | pygame |  **pygame** (from INI) |
| **Performance** | Pure Python loops | Pure Python |  **Numba JIT** (ours, 10-100× faster) |
| **Structure** | 1442-line monolith | Modular package |  **~20 focused modules** (from INI) |
| **Dependencies** | `gym>=0.9.6, numpy` | `gymnasium, numpy, pygame` |  **+ numba, aenum** (optimizations) |
| **Tests** | Basic test script | Unknown |  **130 comprehensive tests** (ours) |
| **Use Case** | Multi-agent team research | Single-agent exploration |  **Multi-agent competitive** with modern API |

**Legend**:
-  = What we adopted/built
- Items from gym-multigrid: view_size=3, Soccer/Collect game mechanics
- Items from INI multigrid: Gymnasium API, 3-channel encoding, pygame, modular structure
- Our contributions: Reproducibility fix, Numba JIT, comprehensive tests, framework adapters

### Bugs Fixed

1. **Reproducibility bug** (critical): `step()` used `np.random.permutation()` (global RNG) for action ordering. Now uses `self.np_random.random(size=N).argsort()` to respect environment seeding.
2. **No `render_mode`**: Constructor now accepts `render_mode='rgb_array'` or `render_mode='human'`, following Gymnasium convention.
3. **Legacy 4-tuple**: `step()` returns Gymnasium 5-tuple `(obs, rewards, terminated, truncated, info)` with per-agent dicts.

## Included Environments

### SoccerGame

<p align="center">
  <img src="https://raw.githubusercontent.com/Abdulhamid97Mousa/mosaic_multigrid/main/figures/soccer.png" width="200">
  <img src="https://raw.githubusercontent.com/Abdulhamid97Mousa/mosaic_multigrid/main/figures/soccer_2.png" width="200">
  <img src="https://raw.githubusercontent.com/Abdulhamid97Mousa/mosaic_multigrid/main/figures/soccer_4.png" width="200">
</p>

Team-based competitive environment. Agents score by dropping the ball at the opposing team's goal. Supports ball passing, stealing, and zero-sum team rewards.

**Default variant:** `SoccerGame4HEnv10x15N2` -- 4 agents (2v2), 15x10 grid, 1 ball.

### CollectGame

<p align="center">
  <img src="https://raw.githubusercontent.com/Abdulhamid97Mousa/mosaic_multigrid/main/figures/collect.png" width="200">
  <img src="https://raw.githubusercontent.com/Abdulhamid97Mousa/mosaic_multigrid/main/figures/collect_2.png" width="200">
</p>

Cooperative/competitive collection. Agents earn rewards for picking up same-color balls and penalties for different-color balls.

**Default variant:** `CollectGame4HEnv10x10N2` -- 4 agents, 10x10 grid, 2 ball colors.

## Installation

### From PyPI (recommended)

```bash
pip install mosaic-multigrid

# With optional framework adapters
pip install mosaic-multigrid[rllib]       # Ray RLlib support
pip install mosaic-multigrid[pettingzoo]  # PettingZoo support
pip install mosaic-multigrid[dev]         # pytest
```

### From source

```bash
git clone https://github.com/Abdulhamid97Mousa/mosaic_multigrid.git
cd mosaic_multigrid
pip install -e .
```

## Quick Start

```python
from mosaic_multigrid.envs import SoccerGame4HEnv10x15N2

env = SoccerGame4HEnv10x15N2(render_mode='rgb_array')
obs, info = env.reset(seed=42)

# obs is a dict keyed by agent index: {0: {...}, 1: {...}, 2: {...}, 3: {...}}
# Each agent's obs has 'image', 'direction', 'mission' keys
print(obs[0]['image'].shape)  # (3, 3, 3) - partial view!

actions = {i: env.action_space[i].sample() for i in range(env.num_agents)}
obs, rewards, terminated, truncated, info = env.step(actions)

# Render RGB frame
frame = env.render()  # Returns numpy array (H, W, 3)
env.close()
```

## Partial Observability

**Agents have limited field of view!** We use **view_size=3** (from gym-multigrid) for competitive team games. This creates challenging coordination problems where agents can't see the entire field.

### Why view_size=3?

We **kept the small view size from gym-multigrid** for research continuity:
-  **Challenging** - Forces team coordination and communication
-  **Realistic** - Agents can't see everything (fog of war)
-  **Research proven** - Comparable with Fickinger et al. (2020)

We **adopted modern infrastructure from INI multigrid**:
-  Gymnasium API, 3-channel encoding, pygame rendering, Numba JIT

### Visual Comparison

#### Agent View Size

Each agent has **limited perception** - they only see a local grid around them, not the entire environment.

#### Default View: 3×3 (mosaic_multigrid - Competitive)

```
Soccer environment (view_size=3):

Full Grid (15×10):                               Agent 0's View (3×3):
┌─────────────────────────────────┐              ┌──────┐
│ W  W  W  W  W  W  W  W  W  W  W │              │ W W W│  ← Top row (walls)
│ W  .  .  .  .  .  .  .  .  .  W │              │ . . .│  ← Middle row
│ W  🔵→ .  .  .  .  ⚽ .  . .  W │              │ .🔵 .│  ← Agent at bottom-center looking up
│ W  .  .  .  .  .  .  .  .  .  W │              └──────┘
│ W  .  .  .  .  .  .  .  .  .  W │
│ W  🟥 .  .  .  .  .  .  .  🟦 W │              Coverage: 9 cells (3×3)
│ W  .  .  .  .  .  .  .  .  .  W │              Forward: 2 tiles
│ W  .  .  .  .  .  .  .  .  .  W │              Sides: 1 tile each
│ W  🔵 .  .  .  .  .  .  .  🔴 W │
│ W  W  W  W  W  W  W  W  W  W  W │              ⚠️ CANNOT see ball! 
└─────────────────────────────────┘              ⚠️ CANNOT see goals!
                                                 ⚠️ CANNOT see teammates!
Legend:
🔵 Blue team  🔴 Red team  ⚽ Ball  🟥🟦 Goals  W=Wall  .=Empty
```

#### View Rotation

**The view rotates with the agent!** The agent is always at the bottom-center, facing "up" in its own reference frame.

```
Agent facing RIGHT (direction=0):     Agent facing DOWN (direction=1):
┌───────┐                              ┌───────┐
│ . . . │ → Agent's forward            │ . A . │
│ A . . │   view is to the right       │ . . . │
│ . . . │   in the global grid         │ . . . │
└───────┘                              └───────┘
                                      ↓ Forward view is downward

Agent facing LEFT (direction=2):      Agent facing UP (direction=3):
┌───────┐                              ┌───────┐
│ . . . │                              │ . . . │
│ . . A │ ← Forward view is            │ . . . │
│ . . . │   to the left                │ . A . |
└───────┘                              └───────┘
                                      ↑ Forward is upward
```

### Configurable View Size

```python
from mosaic_multigrid.envs import SoccerGameEnv

# Default: 3×3 (competitive challenge)
env = SoccerGameEnv(view_size=3, ...)
obs, _ = env.reset()
print(obs[0]['image'].shape)  # (3, 3, 3)

# Match INI multigrid: 7×7 (easier)
env = SoccerGameEnv(view_size=7, ...)
obs, _ = env.reset()
print(obs[0]['image'].shape)  # (7, 7, 3)
```

### Observation Format (Compatible with INI multigrid)

- `obs[agent_id]['image']` shape: `(view_size, view_size, 3)`
  - Channel 0: Object type (wall, ball, goal, agent, etc.)
  - Channel 1: Object color (red, blue, green, etc.)
  - Channel 2: Object state (open/closed door, agent direction)
- `obs[agent_id]['direction']`: int (0=right, 1=down, 2=left, 3=up)
- `obs[agent_id]['mission']`: Mission string

**The agent is always at the bottom-center of its view**, looking forward. The view rotates with the agent's direction.

📖 **See [PARTIAL_OBSERVABILITY.md](PARTIAL_OBSERVABILITY.md) for detailed visual diagrams and comparison with INI multigrid.**

### Reproducibility

```python
# Same seed → identical trajectories (reproducibility bug is fixed)
for trial in range(2):
    env = SoccerGame4HEnv10x15N2(render_mode='rgb_array')
    obs, _ = env.reset(seed=42)
    for step in range(100):
        actions = {i: 2 for i in range(4)}  # all forward
        obs, *_ = env.step(actions)
    # obs will be identical across trials
```

## Architecture

```
gym_multigrid/
├── base.py                  # MultiGridEnv (Gymnasium-compliant base)
├── core/
│   ├── constants.py         # Type, Color, State, Direction enums
│   ├── actions.py           # Action enum (7 actions, no "still")
│   ├── world_object.py      # WorldObj numpy-subclass + all object types
│   ├── agent.py             # Agent + AgentState (vectorized, team_index)
│   ├── grid.py              # Grid (numpy state + world_objects cache)
│   └── mission.py           # Mission + MissionSpace
├── utils/
│   ├── enum.py              # IndexedEnum (aenum-based, dynamically extensible)
│   ├── rendering.py         # Tile rendering (fill_coords, downsample, etc.)
│   ├── random.py            # RandomMixin (seeded RNG utilities)
│   ├── obs.py               # Numba JIT observation generation (hot path)
│   └── misc.py              # front_pos, PropertyAlias
├── envs/
│   ├── soccer_game.py       # SoccerGameEnv + variants
│   └── collect_game.py      # CollectGameEnv + variants
├── wrappers.py              # FullyObs, ImgObs, OneHotObs, SingleAgent
├── pettingzoo/              # PettingZoo ParallelEnv adapter
└── rllib/                   # Ray RLlib MultiAgentEnv adapter
```

### Core Design Decisions

**Agent-not-in-grid**: Agents are NOT stored on the grid (following multigrid-ini). Agent positions are tracked via `AgentState.pos`. The observation generator inserts agents into the observation grid dynamically. This avoids grid corruption when agents overlap.

**numpy subclass pattern**: `WorldObj(np.ndarray)` and `AgentState(np.ndarray)` — domain objects ARE their numerical encoding. No serialization overhead.

**team_index separation**: `agent.index` (unique identity) vs `agent.team_index` (team membership). The original code conflated these — your agent index WAS your team.

**Numba JIT**: All observation generation functions use `@nb.njit(cache=True)`. Enum values are extracted to plain `int` constants at module level because Numba cannot access Python enum attributes.

## Action Space

### Action Enum Comparison

| Action | Upstream (Fickinger 2020) | mosaic_multigrid (this fork) | multigrid-ini (Oguntola 2023) |
|--------|---:|---:|---:|
| still | **0** | -- | -- |
| left | 1 | **0** | **0** |
| right | 2 | **1** | **1** |
| forward | 3 | **2** | **2** |
| pickup | 4 | **3** | **3** |
| drop | 5 | **4** | **4** |
| toggle | 6 | **5** | **5** |
| done | 7 | **6** | **6** |
| **Total** | **8** | **7** | **7** |

The `still` action (do nothing) was removed. Agents that need to skip a turn send `Action.done` instead. This aligns with multigrid-ini and shrinks the action space from 8 to 7.

> **Migration note:** All action indices shifted down by 1 compared to the upstream code. Any pre-trained policy or hardcoded action mapping from the old environment will need updating.

## Observation Space

Each agent receives a partial observation dict:

```python
{
    'image': np.ndarray,     # (view_size, view_size, 3) — [Type, Color, State] per cell
    'direction': int,        # Agent facing direction (0=right, 1=down, 2=left, 3=up)
    'mission': str,          # Mission string
}
```

The default `view_size=7` gives each agent a 7x7 partial view. Each cell encodes 3 values (Type index, Color index, State index), down from 6 in the original.

## Wrappers

| Wrapper | Description |
|---------|-------------|
| `FullyObsWrapper` | Full grid observation instead of partial agent view |
| `ImgObsWrapper` | Returns only the image array (drops direction/mission) |
| `OneHotObsWrapper` | One-hot encodes the observation image (Numba JIT) |
| `SingleAgentWrapper` | Unwraps multi-agent dict for single-agent use |

## Framework Adapters

### PettingZoo

```python
from gym_multigrid.pettingzoo import to_pettingzoo_env

env = to_pettingzoo_env('MosaicMultiGrid-Soccer-v0', render_mode='rgb_array')
# Returns a PettingZoo ParallelEnv
```

### Ray RLlib

```python
from gym_multigrid.rllib import to_rllib_env

env_cls = to_rllib_env('MosaicMultiGrid-Soccer-v0')
# Returns an RLlib MultiAgentEnv class (adds __all__ keys to terminated/truncated)
```

## Requirements

- Python >= 3.9
- gymnasium >= 0.26
- numpy >= 1.18
- numba >= 0.53
- pygame >= 2.2
- aenum >= 1.3

**Optional:**
- `ray[rllib] >= 2.0` (for RLlib adapter)
- `pettingzoo >= 1.22` (for PettingZoo adapter)

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

94 tests covering: Action enum, Type/Color/State/Direction enums, WorldObj encode/decode, Grid operations, AgentState vectorized ops, Agent team_index, Mission/MissionSpace, MultiGridEnv reset/step/render, pickup/drop mechanics, Numba JIT observations, rendering dimensions, and reproducibility.

## Citation

If you use this environment, please cite both the original work and this fork:

```bibtex
@misc{gym_multigrid,
  author = {Fickinger, Arnaud},
  title = {Multi-Agent Gridworld Environment for OpenAI Gym},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ArnaudFickinger/gym-multigrid}},
}

@misc{mosaic_multigrid,
  author = {Mousa, Abdulhamid},
  title = {mosaic\_multigrid: Research-Grade Multi-Agent Gridworld Environments},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Abdulhamid97Mousa/mosaic_multigrid}},
}
```

## License

Apache License 2.0 -- see [LICENSE](LICENSE) for details.

**Original work:** MiniGrid (Copyright 2020 Maxime Chevalier-Boisvert), MultiGrid extension (Copyright 2020 Arnaud Fickinger).
**This fork:** Copyright 2026 Abdulhamid Mousa.
