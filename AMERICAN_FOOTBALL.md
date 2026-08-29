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

American Football includes dense reward shaping enabled by default (`reward_shaping=True`).
All components are hack-proof: no reward fires on a bare action — only on meaningful
game-state transitions toward scoring.

### Reward Components

| Event | Reward | When |
|-------|--------|------|
| **Move toward end zone** | +0.01 × Δcol | While **carrying**, each column closer to target end zone |
| **Steal ball from opponent** | +0.3 | Agent takes ball from an opponent **currently carrying** it |
| **Touchdown** | +15 + (max_steps − step) × 0.05 | Walk into opponent's end zone while carrying |
| **Opponent touchdown** | −(15 + time_bonus) | When `zero_sum=True` (all competitive variants) |
| **Timeout (no winner)** | −1.0 (all agents) | Applied when `max_steps` reached with no winner |

Time-efficiency bonus at scoring step 10: +14.5 extra (total ≈ 29.5). At step 290: +0.5 extra.

### Why Each Component Exists

**Carrying-gated distance shaping**

Only fires while the agent holds the ball and moves toward the goal. Drop immediately
resets it — `pickup → drop → pickup` cycling yields zero reward, closing the exploit
where agents earned +0.1/step without advancing.

**Steal reward (+0.3)**

Encourages interception and defensive pressure. Requires an opponent in active possession
— cannot be farmed by picking up ground balls or cycling with teammates.

**Time-efficiency scoring**

Transforms the single +15 terminal reward into a gradient over the whole episode.
Scoring on step 10 is dramatically better than step 290 — agents are incentivised to
find the ball and advance immediately rather than stalling.

**Timeout penalty (−1.0)**

Without it, "do nothing" is a safe Nash equilibrium. The penalty makes stalling actively
costly for both teams.

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

## Rendering

American Football uses a custom pygame renderer with:
- Brown grass stripes (alternating light/dark)
- White yard lines and hash marks
- Colored end zones (green and blue with transparency)
- Directional agent triangles with team colors
- Oval brown ball with white laces
- Agent FOV highlights and ID labels
