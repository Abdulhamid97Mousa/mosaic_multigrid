# American Football Environment

## Overview

American Football is a competitive multi-agent environment on a 16x11 grid
(same dimensions as Soccer). Agents score **touchdowns** by walking into the
opponent's end zone while carrying the ball. Unlike Soccer and Basketball
(which use single-square ObjectGoal targets), American Football uses full
vertical-column **EndZone** objects spanning 9 cells.

## Grid Layout

```
     Col:  0   1   2  ...  13  14  15
Row 0:  [wall wall wall ...     wall wall]
Row 1:  [wall  GZ  ..midfield..  BZ  wall]
  ...   [wall  GZ  ..midfield..  BZ  wall]
Row 9:  [wall  GZ  ..midfield..  BZ  wall]
Row 10: [wall wall wall ...     wall wall]
```

- **Green End Zone (GZ):** Column 1, rows 1-9 (9 cells). Defended by Green team (team_index=0). Blue scores here.
- **Blue End Zone (BZ):** Column 14, rows 1-9 (9 cells). Defended by Blue team (team_index=1). Green scores here.
- **Midfield:** Columns 2-13, rows 1-9 (108 playable cells). Ball and agents spawn here.

## Scoring Mechanics

| Aspect | American Football | Soccer | Basketball |
|--------|-------------------|--------|------------|
| **How to score** | Walk INTO end zone while carrying | Walk INTO goal square while carrying | Walk INTO goal square while carrying |
| **Goal/zone size** | Full column (1x9 cells) | Single square (1x1) | Single square (1x1) |
| **Drop action** | Pass only (no scoring) | Pass only (no scoring) | Pass only (no scoring) |
| **Goals to win** | 2 (default) | 2 (default) | 2 (default) |
| **Ball respawn** | Random midfield position | Random position | Random position |

## Reward Shaping (v6.4.0)

American Football includes SMAC-style dense reward shaping, enabled by default.
This follows the pattern from Samvelyan et al. (2019) where intermediate signals
guide learning in sparse-reward environments.

### Reward Components

| Event | Reward | When |
|-------|--------|------|
| **Pickup ball** | +0.1 | Agent transitions from not-carrying to carrying |
| **Move toward end zone** | +0.01 per column | While carrying, each column closer to target |
| **Move away from end zone** | -0.01 per column | While carrying, each column farther from target |
| **Touchdown** | +1.0 | Walk into opponent's end zone while carrying |
| **Opponent touchdown** | -1.0 | Only when `zero_sum=True` (competitive variants) |

### Why Reward Shaping?

Without shaping, agents receive reward **only** on touchdowns. On a 16x11 grid
with `view_size=3` (7% field coverage), a random agent scores ~0.6% of the time.
The gradient signal is too weak for standard RL algorithms (PPO, MAPPO, IPPO).

With shaping, agents get useful feedback from the first episode where they
accidentally pick up the ball. In testing, PPO reached 56% touchdown rate
within 150 iterations with shaping vs 0.8% without.

### Disabling Reward Shaping

For evaluation or when comparing against sparse baselines:

```python
# Dense rewards (default, for training)
env = AmericanFootball1v1Env16x11(reward_shaping=True)

# Sparse rewards (for evaluation)
env = AmericanFootball1v1Env16x11(reward_shaping=False)
```

## Game Mechanics

### Ball Handling

- **Pickup (action 4):** Pick up ball from ground, or steal from opponent in front
- **Drop (action 5):** Teleport pass to random teammate, or drop on ground
- **Carrying:** Agent holds ball until dropping, scoring, or being stolen from

### Stealing (with cooldown)

When an agent steals the ball from an opponent:
- Both the stealer and victim receive a 10-step cooldown
- During cooldown, neither can attempt another steal
- Prevents the "ping-pong" exploit where agents steal back and forth

### Teleport Passing

When an agent uses the drop action while carrying:
1. If teammates exist and one is not carrying: ball teleports to a random teammate
2. Otherwise: ball drops on the ground in front

### Episode Termination

- **Terminated:** First team to reach `goals_to_win` touchdowns (default: 2)
- **Truncated:** Step count reaches `max_steps` (default: 300)

## Environment Variants

### Solo (Curriculum Pre-training)

| Environment ID | Agents | Purpose |
|----------------|--------|---------|
| `MosaicMultiGrid-AmericanFootball-Solo-Green-v0` | 1 (Green) | Learn scoring chain without opponent |
| `MosaicMultiGrid-AmericanFootball-Solo-Blue-v0` | 1 (Blue) | Learn scoring chain without opponent |

### Competitive

| Environment ID | Agents | Teams | Zero-Sum |
|----------------|--------|-------|----------|
| `MosaicMultiGrid-AmericanFootball-1v1-v0` | 2 | 1v1 | Yes |
| `MosaicMultiGrid-AmericanFootball-2v2-v0` | 4 | 2v2 | Yes |
| `MosaicMultiGrid-AmericanFootball-3v3-v0` | 6 | 3v3 | Yes |

### TeamObs (SMAC-style teammate awareness)

| Environment ID | Agents | Extra Obs |
|----------------|--------|-----------|
| `MosaicMultiGrid-AmericanFootball-2v2-TeamObs-v0` | 4 | teammate positions, directions, has_ball |
| `MosaicMultiGrid-AmericanFootball-3v3-TeamObs-v0` | 6 | teammate positions, directions, has_ball |

### Configurable View Size

All variants accept `view_size` as a constructor parameter:

```python
# Default: 3x3 partial view (7% field coverage, challenging)
env = AmericanFootball1v1Env16x11(view_size=3)

# Wider view: 7x7 (39% field coverage, easier for training)
env = AmericanFootball1v1Env16x11(view_size=7)
```

## Event Tracking

The environment tracks all game events in the info dict:

```python
obs, rewards, terminated, truncated, info = env.step(actions)

# After a touchdown:
info[agent_id]["goal_scored_by"]  # {"step": 42, "scorer": 0, "team": 0}

# After a pass:
info[agent_id]["pass_completed"]  # {"step": 38, "passer": 0, "receiver": 1, "team": 0}

# After a steal:
info[agent_id]["steal_completed"]  # {"step": 55, "stealer": 1, "victim": 0, "team": 1}

# Every step:
info[agent_id]["position"]   # (x, y) tuple
info[agent_id]["carrying"]   # True/False
```

## Comparison with SMAC

| Dimension | SMAC | American Football |
|-----------|------|-------------------|
| **Reward density** | Dense (damage/step) | Dense (pickup + distance shaping) |
| **Reward flag** | `reward_sparse=True` | `reward_shaping=False` |
| **Observation** | Flattened features (~100 dims) | Image grid (3x3x3 or 7x7x3) |
| **Opponent** | Scripted bot (stationary) | Learning agent (non-stationary) |
| **Coordination** | Cooperative (all vs AI) | Competitive (team vs team) |

## Rendering

American Football uses a custom pygame renderer with:
- Brown grass stripes (alternating light/dark)
- White yard lines and hash marks
- Colored end zones (green and blue with transparency)
- Directional agent triangles with team colors
- Oval brown ball with white laces
- Agent FOV highlights and ID labels
