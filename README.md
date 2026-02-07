# mosaic_multigrid

**Research-grade multi-agent gridworld environments for reproducible RL experiments.**

A maintained fork of [gym-multigrid](https://github.com/ArnaudFickinger/gym-multigrid) by Arnaud Fickinger (2020), with critical bug fixes for reproducibility and modern RL framework compatibility.

This fork is developed as part of the [MOSAIC](https://github.com/Abdulhamid97Mousa/MOSAIC) project (Multi-Agent Orchestration System).

## Why This Fork?

The upstream `gym-multigrid` has not been updated since 2020 and contains several bugs that make it unsuitable for reproducible research. This fork addresses:

1. **Non-deterministic action ordering** -- the `step()` method uses the global NumPy RNG instead of the seeded environment RNG, making experiments non-reproducible even when properly seeded.
2. **No `render_mode` constructor parameter** -- environments use the legacy `render(mode='human')` API, which is incompatible with Gymnasium's `render_mode` constructor protocol and prevents integration with automated frame-capture wrappers.
3. **Old Gym API** -- returns 4-tuple `(obs, reward, done, info)` instead of the Gymnasium 5-tuple `(obs, reward, terminated, truncated, info)`.

See [Known Issues](#known-issues) for full details.

## Included Environments

### SoccerGame

<p align="center">
  <img src="figures/soccer.png" width="200">
  <img src="figures/soccer_2.png" width="200">
  <img src="figures/soccer_4.png" width="200">
</p>

Each agent gets a positive reward when any agent on their team drops the ball in the correct goal, and a negative reward when the opposing team scores. Agents can pass the ball or take it from another agent. The number of teams, players per team, goals, and balls are configurable.

**Default variant:** `SoccerGame4HEnv10x15N2` -- 4 agents (2v2), 15x10 grid, 1 ball, zero-sum rewards.

### CollectGame

<p align="center">
  <img src="figures/collect.png" width="200">
  <img src="figures/collect_2.png" width="200">
</p>

Each agent gets a positive reward for collecting a ball of the same color and a negative reward for collecting a ball of a different color. The number of balls, colors, and players are configurable.

**Default variant:** `CollectGame4HEnv10x10N2` -- 4 agents, 10x10 grid, 2 ball colors.

## Installation

```bash
git clone https://github.com/Abdulhamid97Mousa/mosaic_multigrid.git
cd mosaic_multigrid
pip install -e .
```

## Quick Start

```python
from gym_multigrid.envs import SoccerGame4HEnv10x15N2

env = SoccerGame4HEnv10x15N2()
env.seed(42)
obs = env.reset()

# 4 agents, 7 possible actions each (0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle, 6=done)
actions = [env.action_space.sample() for _ in range(4)]
obs, rewards, done, info = env.step(actions)

# Render RGB frame
img = env.render(mode='rgb_array')  # Returns numpy array (H, W, 3)
```

## Requirements

- Python 3.8+
- OpenAI Gym >= 0.21.0
- NumPy >= 1.15.0

## Known Issues

### Issue #1: Non-Deterministic Action Ordering (Reproducibility Bug)

**Severity:** Critical for research reproducibility
**Location:** `gym_multigrid/multigrid.py:1249`

```python
def step(self, actions):
    self.step_count += 1
    order = np.random.permutation(len(actions))  # BUG: uses global RNG
```

The `step()` method randomizes the order in which agent actions are executed using `np.random.permutation()`, which draws from the **global NumPy RNG**. This completely ignores the seeded `self.np_random` that was set via `env.seed()`.

**Impact:**
- Even with proper seeding (`env.seed(42)`), the initial state is deterministic but subsequent steps are not
- Action execution order varies between runs, causing trajectory divergence
- Makes ablation studies, LLM-vs-RL comparisons, and published results non-reproducible
- This is likely a contributing factor to why gym-multigrid Soccer is rarely seen in published research

**Expected behavior:** `step()` should use `self.np_random.permutation()` instead of `np.random.permutation()` to respect environment seeding.

**Workaround (MOSAIC):** MOSAIC applies a `ReproducibleMultiGridWrapper` that seeds the global `np.random` from the environment's seeded RNG before each `step()` call. See [MOSAIC docs](https://github.com/Abdulhamid97Mousa/MOSAIC) for details.

---

### Issue #2: No `render_mode` Constructor Parameter (Gymnasium Incompatibility)

**Severity:** High for framework integration
**Location:** `gym_multigrid/envs/soccer_game.py:108-118`, `gym_multigrid/multigrid.py:1383`

The environment constructors accept **zero arguments** for the concrete variants:

```python
class SoccerGame4HEnv10x15N2(SoccerGameEnv):
    def __init__(self):  # No render_mode parameter!
        super().__init__(size=None, height=10, width=15, ...)
```

And the `render()` method uses the legacy Gym API:

```python
def render(self, mode='human', close=False, highlight=False, tile_size=TILE_PIXELS):
    # mode is a method parameter, not a constructor attribute
```

**Impact:**
- Cannot set `render_mode='rgb_array'` at construction time (Gymnasium convention)
- Automated frame-capture wrappers (like MOSAIC's FastLane, Gymnasium's `RecordVideo`, or `HumanRendering`) that check `env.render_mode` will find it missing and skip frame capture
- The `sitecustomize.py` gym.make() patching approach (used for BabyAI environments) does not work here because Soccer environments are created via direct class instantiation, not `gym.make()`
- Training visualization is silently broken: no error is raised, but no frames are published

**Expected behavior:** Environment constructors should accept a `render_mode` parameter and store it as `self.render_mode`, following the [Gymnasium API convention](https://gymnasium.farama.org/api/env/#gymnasium.Env.render_mode).

**Workaround (MOSAIC):** The `GymToGymnasiumWrapper` in `xuance_worker/environments/multigrid.py` needs to explicitly set `render_mode` on the wrapped environment and ensure `render()` returns RGB arrays when `render_mode='rgb_array'`.

---

### Issue #3: Legacy Gym 4-Tuple Return Format

**Severity:** Medium (compatibility)
**Location:** `gym_multigrid/multigrid.py` (step method), `gym_multigrid/envs/soccer_game.py:103-105`

```python
def step(self, actions):
    obs, rewards, done, info = MultiGridEnv.step(self, actions)
    return obs, rewards, done, info  # 4-tuple, no truncated signal
```

**Impact:**
- Modern RL libraries (CleanRL, Stable-Baselines3, RLlib) expect the Gymnasium 5-tuple: `(obs, reward, terminated, truncated, info)`
- Requires a wrapper to split `done` into `terminated` and `truncated`
- The `max_steps=10000` truncation is conflated with episode termination

## Observation Space

Each grid cell is encoded as a tuple containing:
- Object type (wall, floor, lava, door, key, ball, box, goal, object goal, agent)
- Object color
- Type of object the other agent is carrying
- Color of object the other agent is carrying
- Direction of the other agent
- Whether the other agent is self (useful for fully observable view)

## Action Space

| Action | ID | Description |
|--------|---:|-------------|
| Turn left | 0 | Rotate 90 degrees left |
| Turn right | 1 | Rotate 90 degrees right |
| Move forward | 2 | Move one cell forward |
| Pick up | 3 | Pick up object in front |
| Drop | 4 | Drop carried object |
| Toggle | 5 | Interact with object |
| Done | 6 | Signal task completion |

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
