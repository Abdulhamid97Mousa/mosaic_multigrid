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

## Reward Shaping (v6.5.0)

American Football includes SMAC-style dense reward shaping, enabled by default.
The reward structure is a continuous ladder: each rung is reachable from the
one below without requiring luck.

```
+0.01/step near ball  →  +0.1 on pickup  →  +0.05/step toward end zone  →  +15.0 on touchdown
(always reachable)        (rare but reachable)  (follows naturally)            (follows naturally)
```

### Reward Components

| Event | Reward | When |
|-------|--------|------|
| **Near ball (proximity)** | +0.01 / step | Agent within Manhattan dist ≤ 3 of ball, not carrying |
| **Pickup ball** | +0.1 | Agent picks up ball; only if last carrier was opponent or nobody (provenance check) |
| **Move toward end zone** | +0.05 × Δcol | While carrying, each column closer to target end zone |
| **Move away from end zone** | −0.05 × Δcol | While carrying, each column farther from target |
| **First pass to teammate** | +0.1 | First teleport pass per possession chain |
| **Second consecutive pass (A→B→A)** | −0.2 | Penalises bounce passing that farms the pass reward |
| **Touchdown** | +15.0 | Walk into opponent's end zone while carrying |
| **Opponent touchdown** | −15.0 | When `zero_sum=True` (all competitive variants) |
| **Timeout (no winner)** | −1.0 (all agents) | Applied when `max_steps` reached with no winner |

### Why Each Layer Exists

**Layer 1 — Ball provenance (`_ball_last_carrier_team`)**

Blocks the same-team pickup-farming exploit where agents chain
`PICKUP → DROP → PICKUP → DROP` to accumulate +0.1 per cycle in place.
The +0.1 pickup bonus is only paid when the ball last belonged to the
opposing team (or nobody). Implemented via `_ball_last_carrier_team` dict
mapping `ball_index → last_carrier_team_index`.

**Layer 2 — Timeout penalty (−1.0)**

Without a timeout penalty, "do nothing" is a safe Nash equilibrium — nobody
scores, nobody loses. The −1.0 penalty makes stalling actively bad.

**Layer 3 — Touchdown reward (+15.0) and `zero_sum=True`**

At +1.0, the scoring signal was numerically dominated by the PPO entropy bonus.
At +15.0, one touchdown is worth 300 proximity-reward steps — the unambiguous
dominant objective. `zero_sum=True` means the opposing team receives −15.0.

**Layer 4 — Distance shaping (×0.05) and pass-chain cap**

Raised from ×0.01 to ×0.05 so the gradient toward the end zone overcomes
random-walk noise. The pass-chain cap (−0.2 on second consecutive pass)
prevents A→B→A teleport-bounce farming.

### Own-Touchdown-Zone Prevention

Agents **cannot score in their own end zone**. The scoring check in `step()`
includes an explicit guard comparing the goal's team index to the agent's team:

```python
# In AmericanFootballEnv.step():
if end_zone.team_index != agent.team_index:
    # Only opponent's end zone counts — own end zone is inert
    self._team_reward(agent.team_index, rewards, 15.0)
```

Green agents (team_index=0) can only score in the Blue end zone (column 14),
and Blue agents (team_index=1) can only score in the Green end zone (column 1).

### Training Results (MAPPO 2v2, 1M steps, v6.5.0)

AF 2v2 is the best-performing environment across all three sports:
- Entropy: 1.76 → **1.09** (clear policy commitment)
- Episodes ending in avg **80 steps** (vs 300-step timeout)
- Scoring discovered at ~step 100k — earliest of all three sports
- Critic loss stable at ~0.3 (no oscillation)

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
| `MosaicMultiGrid-AmericanFootball-Solo-Green-IndAgObs-v1` | 1 (Green) | Learn scoring chain without opponent |
| `MosaicMultiGrid-AmericanFootball-Solo-Blue-IndAgObs-v1` | 1 (Blue) | Learn scoring chain without opponent |

### Competitive

| Environment ID | Agents | Teams | Zero-Sum |
|----------------|--------|-------|----------|
| `MosaicMultiGrid-AmericanFootball-1v1-IndAgObs-v1` | 2 | 1v1 | Yes |
| `MosaicMultiGrid-AmericanFootball-2v2-IndAgObs-v1` | 4 | 2v2 | Yes |
| `MosaicMultiGrid-AmericanFootball-3v3-IndAgObs-v1` | 6 | 3v3 | Yes |

### TeamObs (SMAC-style teammate awareness)

| Environment ID | Agents | Extra Obs |
|----------------|--------|-----------|
| `MosaicMultiGrid-AmericanFootball-2v2-TeamObs-v1` | 4 | teammate positions, directions, has_ball |
| `MosaicMultiGrid-AmericanFootball-3v3-TeamObs-v1` | 6 | teammate positions, directions, has_ball |

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
