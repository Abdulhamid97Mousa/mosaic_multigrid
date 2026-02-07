# mosaic_multigrid

**Research-grade multi-agent gridworld environments for reproducible RL experiments.**

A maintained fork of [gym-multigrid](https://github.com/ArnaudFickinger/gym-multigrid) by Arnaud Fickinger (2020), modernized to the Gymnasium API with Numba JIT-accelerated observations, reproducible seeding, and multi-agent framework adapters.

This fork is developed as part of the [MOSAIC](https://github.com/Abdulhamid97Mousa/MOSAIC) project (Multi-Agent Orchestration System).

## What Changed from Upstream

| Aspect | Upstream (Fickinger 2020) | This Fork |
|--------|--------------------------|-----------|
| API | Old Gym 4-tuple, list-based | Gymnasium 5-tuple, dict-keyed per agent |
| Actions | 8 (still=0..done=7) | 7 (left=0..done=6), no "still" |
| Observations | `(view, view, 6)` list | `(view, view, 3)` dict with `image`, `direction`, `mission` |
| `reset()` | `List[obs]` | `(Dict[int, obs], Dict[int, info])` |
| `step()` | `(List[obs], ndarray, bool, dict)` | `(Dict, Dict, Dict, Dict, Dict)` |
| Render | `render(mode='human')` param | `render_mode` constructor param |
| Seeding | `env.seed(42)` + broken global RNG | `reset(seed=42)` + `self.np_random` |
| Window | matplotlib | pygame |
| Performance | Pure Python loops | Numba JIT on observation generation |
| Structure | 1 monolithic 1442-line file | ~20 focused modules |
| Dependencies | `gym>=0.9.6, numpy` | `gymnasium>=0.26, numpy, numba, pygame, aenum` |

### Bugs Fixed

1. **Reproducibility bug** (critical): `step()` used `np.random.permutation()` (global RNG) for action ordering. Now uses `self.np_random.random(size=N).argsort()` to respect environment seeding.
2. **No `render_mode`**: Constructor now accepts `render_mode='rgb_array'` or `render_mode='human'`, following Gymnasium convention.
3. **Legacy 4-tuple**: `step()` returns Gymnasium 5-tuple `(obs, rewards, terminated, truncated, info)` with per-agent dicts.

## Included Environments

### SoccerGame

<p align="center">
  <img src="figures/soccer.png" width="200">
  <img src="figures/soccer_2.png" width="200">
  <img src="figures/soccer_4.png" width="200">
</p>

Team-based competitive environment. Agents score by dropping the ball at the opposing team's goal. Supports ball passing, stealing, and zero-sum team rewards.

**Default variant:** `SoccerGame4HEnv10x15N2` -- 4 agents (2v2), 15x10 grid, 1 ball.

### CollectGame

<p align="center">
  <img src="figures/collect.png" width="200">
  <img src="figures/collect_2.png" width="200">
</p>

Cooperative/competitive collection. Agents earn rewards for picking up same-color balls and penalties for different-color balls.

**Default variant:** `CollectGame4HEnv10x10N2` -- 4 agents, 10x10 grid, 2 ball colors.

## Installation

```bash
git clone https://github.com/Abdulhamid97Mousa/mosaic_multigrid.git
cd mosaic_multigrid
pip install -e .

# With optional framework adapters
pip install -e ".[rllib]"       # Ray RLlib support
pip install -e ".[pettingzoo]"  # PettingZoo support
pip install -e ".[dev]"         # pytest
```

## Quick Start

```python
from gym_multigrid.envs import SoccerGame4HEnv10x15N2

env = SoccerGame4HEnv10x15N2(render_mode='rgb_array')
obs, info = env.reset(seed=42)

# obs is a dict keyed by agent index: {0: {...}, 1: {...}, 2: {...}, 3: {...}}
# Each agent's obs has 'image', 'direction', 'mission' keys

actions = {i: env.action_space[i].sample() for i in range(env.num_agents)}
obs, rewards, terminated, truncated, info = env.step(actions)

# Render RGB frame
frame = env.render()  # Returns numpy array (H, W, 3)
env.close()
```

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
