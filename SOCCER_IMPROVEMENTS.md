# Soccer Environment - Improvements & Bug Fixes

## Overview

This document explains the improvements made to the **Soccer environment** in MOSAIC multigrid to fix critical bugs and enhance strategic gameplay for multi-agent reinforcement learning research.

---

## Critical Bugs Fixed

### **Bug #1: Ball Disappears After Scoring**

**Problem:**
```python
# Old code in soccer_game.py line 204:
if ball.index in (0, fwd_obj.index):
    self._team_reward(fwd_obj.index, rewards, fwd_obj.reward)
    agent.state.carrying = None  # Ball DELETED, never respawns!
    return
```

**Impact:**
- Team scores a goal -- ball disappears forever
- Episode continues for **9,900+ remaining steps** with no ball
- Both teams search endlessly, finding nothing
- Wasted computation, no learning signal for RL agents

**Fix:**
```python
# New code - Ball respawns after scoring
if ball.index in (0, fwd_obj.index):
    self._team_reward(fwd_obj.index, rewards, fwd_obj.reward)
    agent.state.carrying = None

    # NEW: Respawn ball at random location
    new_ball = Ball(color=Color.red, index=0)
    self.place_obj(new_ball)  # Random position, seeded by self.np_random

    # NEW: Check win condition (first to 2 goals)
    self.team_scores[fwd_obj.index] += 1
    if self.team_scores[fwd_obj.index] >= self.goals_to_win:
        # Team wins! Terminate episode
        for a in self.agents:
            a.state.terminated = True
    return
```

**Result:** [FIXED] Ball respawns after each goal, game continues until first team scores 2 goals

---

### **Bug #2: No Termination Signal**

**Problem:**
- Episodes ran for 10,000 steps regardless of score
- First team to score didn't win - game continued endlessly
- Bad for RL: sparse rewards, wasted computation

**Fix:**
```python
# Added termination condition
self.goals_to_win = 2  # First to 2 goals wins

# In _handle_drop when scoring:
self.team_scores[team] += 1
if self.team_scores[team] >= self.goals_to_win:
    for agent in self.agents:
        agent.state.terminated = True
```

**Result:** [FIXED] Episode terminates when one team scores 2 goals (natural termination)

---

### **Bug #3: Observability - Can't See Who Has Ball**

<p align="center">
  <img src="https://github.com/Abdulhamid97Mousa/mosaic_multigrid/raw/main/figures/Agent_Cant_See_Who_Has_Ball_Bug_3.png" width="600">
</p>

**Problem:**
```
Red Agent's View:
+----------+
| W  W  W  |  W = Wall
|           |  . = Empty cell
| G-> .  . |  G = Green agent (but is Green carrying ball? Unknown!)
|    ^     |
| .  B  .  |
+----------+

# Observation encoding (before fix):
obs[1, 0, 0] = 10  # Type: Agent
obs[1, 0, 1] = 2   # Color: Green
obs[1, 0, 2] = 2   # State: Direction (left) -- NO info about ball!
```

**Why this is critical:**
- Red agent cannot see if Green agent is carrying the ball!
- Stealing mechanic is blind - agents can't decide: "Should I chase this agent?"
- Defense strategies impossible without memory (LSTM/GRU to track ball over time)
- With 3x3 view, agents often see OTHER agents but NOT the ball itself

**Root cause:**
The 3-channel encoding only had space for:
- Channel 0: Object TYPE (agent=10, ball=7, etc.)
- Channel 1: Object COLOR (green=1, red=0, etc.)
- Channel 2: Object STATE (direction 0-3)

**No room to encode "has ball"!**

---

**Fix - Repurpose Unused STATE Channel Space:**

**Key Insight**: Soccer and Collect games have **NO DOORS**!
- Door states use STATE channel values: 0=open, 1=closed, 2=locked
- Agent direction uses STATE channel values: 0=right, 1=down, 2=left, 3=up
- **Since Soccer/Collect have no doors, we can repurpose that space!**

**Solution**: Use STATE channel offset 100 for "agent carrying ball"

**Implementation requires changes in TWO locations:**

**1. Agent Encoding** (`core/agent.py` lines 139-188):
```python
def encode(self) -> tuple[int, int, int]:
    """Encode agent as (type, color, state) with ball carrying flag."""
    state = self.state.dir  # Base direction: 0-3

    # Check if agent is carrying a ball
    if (self.state.carrying is not None and
            self.state.carrying.type == Type.ball):
        state += 100  # Add carrying flag (100-103 range)

    return (Type.agent.to_index(), self.state.color.to_index(), state)
```

**2. Observation Generation** (`utils/obs.py` lines 188-202):
```python
# Insert agent grid encodings at their positions
for agent in range(num_agents):
    if not agent_terminated[agent]:
        i, j = agent_pos[agent]
        encoding = agent_grid_encoding[agent].copy()

        # Check if agent is carrying a ball - add carrying flag to STATE
        carrying = agent_carrying_encoding[agent]
        if carrying[TYPE] == BALL:
            # Add carrying flag to STATE channel
            encoding[STATE] += CARRYING_BALL_OFFSET

        grid_encoding[i, j, GRID_ENCODING_IDX] = encoding
```

**STATE Channel Encoding for Agents:**
```
0-3:     Agent direction (right/down/left/up) when NOT carrying ball
100-103: Agent direction + ball carrying flag
  - 100 = facing right + has ball
  - 101 = facing down + has ball
  - 102 = facing left + has ball
  - 103 = facing up + has ball
```

**Decoding in RL Policy:**
```python
# Extract ball carrying information
state_value = obs[agent_id]['image'][y, x, 2]  # STATE channel

has_ball = (state_value >= 100)
direction = state_value % 100

# Example: STATE=101 means agent facing down (1) with ball (100)
```

**Result:**

Green Agent's View (after fix):

<p align="center">
  <img src="https://github.com/Abdulhamid97Mousa/mosaic_multigrid/raw/main/figures/green_agent_view_has_ball.png" width="200">
</p>

```

# Observation encoding (after fix):
obs[1, 0, 0] = 10  # Type: Agent
obs[1, 0, 1] = 2   # Color: Green
obs[1, 0, 2] = 102 # State: 102 = left + CARRYING_BALL

# Decoding in RL policy:
has_ball = (obs[1, 0, 2] >= 100)  # True!
direction = obs[1, 0, 2] % 100     # 2 (left)
```

**RL agents can now see who has the ball AND their direction!**

**Why this works:**
1. **No conflicts**: Door states (0-2), agent direction (0-3), agent+ball (100-103) never overlap
2. **Zero overhead**: Still 3 channels, still uint8 (0-255 range has plenty of room)
3. **Backward compatible**: Door-based environments unaffected (they don't use 100+)
4. **Preserves direction**: Both direction AND carrying state encoded together

**Impact on gameplay:**
- Agents can identify ball carrier in their view
- Defense can focus on carrier, not random agents
- Stealing becomes strategic, not random
- No LSTM required for basic ball tracking
- **Faster training** -- agents have the information they need!

---

## Agent Observability Examples

### Example 1: Self-Awareness

**Q: Does agent know if IT has the ball?**
```python
# Agent 0's observation at its own position [1, 2] (bottom-center):
obs[0]['image'][1, 2, :] = [Type.ball, Color.red, 0]
#                           ^
#                    "I have the ball!"

# Agent 1's observation (no ball):
obs[1]['image'][1, 2, :] = [Type.empty, Color.red, 0]
#                           ^
#                    "I don't have anything"
# Note: empty uses Color.red (index 0) by default -- only the Type matters here
```

**Answer**: YES - Agents see what they're carrying at their own position.

---

### Example 2: Teammate Awareness (Limitation and Solution)

**Q: Can agent 1 see where its teammate agent 0 is?**

**Answer: Rarely.** With `view_size=3`, each agent sees only a 3x3 window (9 cells)
out of 126 playable cells on a 16x11 field. That is approximately **7% of the grid**.
Teammates are almost never within 1 tile of each other during normal play.

```
Agent 1's 3x3 view on a 16x11 field:

+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
|   |   |   | . | . | . |   |   |   |   |   |   |   |   |   |   |
+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
|   |   |   | . |A1 | . |   |   |   |   |   |A0 |   |   |   |   |
+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
|   |   |   | . | . | . |   |   |   |   |   |   |   |   |   |   |
+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+

Agent 1 sees 9 cells (shaded) -- Agent 0 is 8 tiles away, INVISIBLE.
```

**Passing is still possible but blind.** The `DROP` action uses teleport passing:
the ball warps to a random eligible teammate anywhere on the grid. The passer
does NOT need to see the receiver. RL agents learn WHEN to pass (e.g., when
near an opponent), not WHERE the teammate is.

**For informed passing, use the TeamObs variant** (see below).

---

### Example 3: Opponent Awareness

**Q: Can blue team agent 2 see that green agent 1 has the ball?**
```python
# Agent 2 (blue) sees agent 1 (green) with ball:
obs[2]['image'][1, 1, :] = [Type.agent, Color.green, 101]
#                           ^            ^             ^
#                           Agent     Opponent    STATE=101!
#                                                (down + ball)

# Agent 2 decodes:
has_ball = (101 >= 100)  # True!
direction = 101 % 100     # 1 (down)

# Agent 2 can learn:
# IF see opponent (different color) AND opponent has ball -> steal with PICKUP
```

**Answer**: YES -- agents can see ball carriers and learn to steal (when within view).

---

## Training Impact Summary

| Aspect | Before Fix | After Fix |
|--------|-----------|-----------|
| **Can see self carrying ball?** | YES (always) | YES (unchanged) |
| **Can see OTHER agent carrying?** | NO | **YES (FIXED)** |
| **Required architecture** | LSTM/GRU (memory) | Feedforward OK |
| **Stealing strategy** | Random/blind | Strategic (learned) |
| **Passing strategy** | Hard to learn | Learnable (has info) |
| **Training speed** | Slower (poor info) | **Faster (better info)** |

**Tested and Verified**: 7 comprehensive tests in `tests/test_ball_carrying_observability.py` all pass.

---

## Game Mechanics Explained

### **1. Teleport Passing (Enhanced -- replaces adjacency handoff)**

> **Change:** In `SoccerGameEnhancedEnv`, the `DROP` action uses **teleport
> passing**. The ball instantly transfers to a random eligible teammate
> **anywhere on the grid** -- no adjacency required. The original
> `SoccerGameEnv` keeps the old 1-cell adjacency handoff for backward
> compatibility.

**How it works (priority order):**
```python
def _handle_drop(self, agent_index, agent, rewards):
    fwd_pos = agent.front_pos

    # -- Priority 1: Score at goal -------------------------
    fwd_cell = self.grid.get(*fwd_pos)
    if fwd_cell and fwd_cell.type == Type.goal:
        ...   # scoring logic (unchanged)
        return

    # -- Priority 2: Teleport pass to teammate -------------
    eligible = [
        a for a in self.agents
        if a.team_index == agent.team_index      # same team
        and a.index != agent.index                # not self
        and a.state.carrying is None              # not carrying
        and not a.terminated                      # still alive
    ]
    if eligible:
        target = eligible[self.np_random.integers(len(eligible))]
        target.state.carrying = agent.state.carrying
        agent.state.carrying = None
        return

    # -- Priority 3: Ground drop ---------------------------
    if self.grid.get(*fwd_pos) is None:
        self.grid.set(*fwd_pos, agent.state.carrying)
        agent.state.carrying = None
```

**Why teleport over adjacency?**
- **Old (adjacency):** Agent must face teammate in the adjacent cell -- extremely
  hard for RL to coordinate 1-cell alignment on a 14x9 grid.
- **New (teleport):** Agent presses `DROP` and the ball warps to any free
  teammate. RL only needs to learn *when* to pass, not *where* to stand.

**Why it matters:**
- **Coordination:** Agent being chased passes to open teammate across the field
- **Strategic depth:** Passing becomes a tactical timing decision, not a positioning puzzle
- **Emergent behavior:** MAPPO learns "one agent draws defenders, passes to open teammate near goal"
- **Seeded RNG:** Uses `self.np_random.integers()` so results are reproducible

**Real-world parallel:** Like a long-ball pass in real soccer -- instant ball
transfer to a better-positioned teammate anywhere on the pitch

---

### **2. Stealing Mechanism**

**How it works:**
```python
# Agent without ball faces opponent (with ball) and presses PICKUP
def _handle_pickup(self, agent_index, agent, rewards):
    target_agent = self._agent_at(fwd_pos)

    # Stealing from opponent
    if target_agent and target_agent.state.carrying:
        if target_agent.team_index != agent.team_index:
            # Check cooldown
            if agent.action_cooldown > 0:
                return  # Can't steal yet (recovering from tackle)

            # Steal successful!
            agent.state.carrying = target_agent.state.carrying
            target_agent.state.carrying = None

            # Both agents get cooldown (tackle recovery)
            agent.action_cooldown = 10
            target_agent.action_cooldown = 10
```

**Cooldown mechanic:**
- **Attacker:** Can't steal again for 10 steps (tired from tackle)
- **Victim:** Can't pickup for 10 steps (knocked down)
- **Result:** Prevents ping-pong stealing, creates tactical window

**Why it matters:**
- **Defense pressure:** Opponent can steal ball, forcing offensive urgency
- **Team coordination:** While one agent is in cooldown, teammate can support/recover
- **Emergent behavior:** Defensive specialist learns to intercept opponents

**Real-world parallel:** Like real soccer tackles - both players need recovery time

---

## New Map Layout (FIFA Aspect Ratio)

### **14x9 Playable Area (16x11 Total)**

<p align="center">
  <img src="https://github.com/Abdulhamid97Mousa/mosaic_multigrid/raw/main/figures/14×9_Playable_Area_16×11_Total.png" width="600">
</p>

Inspired by FIFA recommended pitch dimensions (105m x 68m = 1.54 ratio)

```
Legend:
  W = Wall (impassable, auto-generated by grid.wall_rect())
  . = Empty cell (playable)
  G1 = Team 1 (Green) goal -- left side at (1, 5) -- Red team scores here
  G2 = Team 2 (Red) goal -- right side at (14, 5) -- Green team scores here

Game Setup:
  - Ball: Red colored (wildcard, spawns randomly)
  - Green team (index=1): Agents 0 & 1 -- defend left goal G1, score at right goal G2
  - Blue team (index=2): Agents 2 & 3 -- defend right goal G2, score at left goal G1
```

**Dimensions:**
- **Total:** 16x11 (176 cells)
- **Walls:** 50 cells (outer boundary)
- **Goals:** 2 cells (fixed positions)
- **Playable:** 14x9 = **126 empty cells**

**Goal positions:**
- Red (team 1): `(1, 5)` -- left side, vertical center
- Blue (team 2): `(14, 5)` -- right side, vertical center

---

## Reward Structure

| Event | Reward | Rationale |
|-------|--------|-----------|
| **Pickup ball** | 0 | Neutral tactical action (no reward) |
| **Steal ball** | 0 | Neutral tactical action (no reward) |
| **Pass ball** | 0 | Neutral tactical action (no reward) |
| **Score goal** | +1 (shared to scoring team) | Positive-only, ONLY way to win |
| **Win (2 goals)** | Episode terminates | Natural termination signal |

**Positive-only rewards (v4.2.0):** IndAgObs Soccer variants now use positive-only shared team rewards, following the SMAC convention (`reward_only_positive=True`). Opponents receive 0 on conceded goals, not -1.

**Event tracking (v4.2.0):** Three event types are tracked as metadata in the info dict: `goal_scored_by` records `{step, scorer, team}`, `passes_completed` records `{step, passer, receiver, team}`, and `steals_completed` records `{step, stealer, victim, team}`. Together these cover the full action chain (steal, pass, goal) for post-hoc credit attribution without affecting the reward signal.

**Why only scoring gives reward?**
- **Clear objective:** Score 2 goals to win (simple for RL)
- **No exploitation:** Can't win by stealing/passing forever
- **Emergent strategy:** Must balance offense/defense/coordination
- **Sparse but learnable:** Clear reward signal on goals

---

## Strategic Depth

### **Emergent Behaviors (Expected with MAPPO)**

**Phase 1: Random Exploration (0-50k episodes)**
- Agents discover pickup, movement, scoring
- Random actions, accidental goals

**Phase 2: Basic Scoring (50k-200k episodes)**
- Learn sequence: pickup -> navigate -> score
- Solo play, no coordination

**Phase 3: Stealing & Defense (200k-500k episodes)**
- Discover stealing mechanic
- Basic defensive positioning
- Chase opponents with ball

**Phase 4: Role Specialization (500k-1M episodes)**
- **Offensive specialist:** Picks ball, uses teammate as shield, rushes to score
- **Defensive specialist:** Guards own goal, intercepts opponents
- **Passing coordination:** Offensive agent passes to better-positioned teammate
- **Baiting:** Defensive agent baits opponent steal, then counter-steals

**Phase 5: Advanced Tactics (1M+ episodes)**
- Fake passes (threaten pass, keep ball)
- Dynamic role switching (context-dependent)
- Opportunistic offense (defender scores when chance arises)

---

## Environment Registry

### **MosaicMultiGrid-Soccer-v0** (Deprecated)

**Status:** Deprecated -- kept for backward compatibility only

```python
# Old environment (broken, not recommended)
env = gym.make('MosaicMultiGrid-Soccer-v0')
obs, _ = env.reset()

# BUGS:
# - Ball disappears after scoring (no respawn)
# - No termination (always runs 10,000 steps)
# - Can't see who is carrying ball
```

**Issues:** Ball disappears, no natural termination, observability problems

---

### **MosaicMultiGrid-Soccer-Enhanced-v0** (Recommended)

**Status:** [RECOMMENDED] For RL training with independent agent views

```python
# Enhanced environment (fixed, recommended)
env = gym.make(
    'MosaicMultiGrid-Soccer-Enhanced-v0',
    max_steps=200,      # Shorter episodes for RL
    goals_to_win=2,     # First to 2 goals wins
)
obs, _ = env.reset()

for step in range(200):
    actions = {i: policy(obs[i]) for i in range(4)}
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated[0]:  # Team scored 2 goals!
        winner = "Green" if rewards[0] > 0 else "Red"
        print(f"{winner} team wins!")
        break
```

**Features:**
- Ball respawns after each goal
- Terminates when team scores 2 goals
- STATE channel encodes ball carrying (observability fixed)
- Dual cooldown on stealing (prevents ping-pong)
- Teleport passing to any teammate (replaces adjacency handoff)
- FIFA-style 14x9 playable area (16x11 total)

**Observation model:** Independent agent views. Each agent sees only its
3x3 local window. No knowledge of teammate positions outside the window.
Passing is blind (teleport to random eligible teammate).

---

### **MosaicMultiGrid-Soccer-TeamObs-v0** (Recommended for team coordination)

**Status:** [RECOMMENDED] For RL training requiring team coordination

```python
# TeamObs variant -- SMAC-style teammate awareness
env = gym.make(
    'MosaicMultiGrid-Soccer-TeamObs-v0',
    render_mode='rgb_array',
)
obs, _ = env.reset()

# Each agent's observation now includes:
print(obs[0].keys())
# dict_keys(['image', 'direction', 'mission',
#            'teammate_positions', 'teammate_directions', 'teammate_has_ball'])

# Teammate features for agent 0:
print(obs[0]['teammate_positions'])    # [[dx, dy]] relative to self (shape: N x 2)
print(obs[0]['teammate_directions'])   # [dir]  teammate facing direction (shape: N)
print(obs[0]['teammate_has_ball'])     # [0/1]  is teammate carrying ball? (shape: N)
```

**What it adds** (over Soccer-Enhanced-v0):

| Feature | Shape | Description |
|---------|-------|-------------|
| `teammate_positions` | (N, 2) int64 | Relative (dx, dy) from self to each teammate |
| `teammate_directions` | (N,) int64 | Direction each teammate faces (0-3) |
| `teammate_has_ball` | (N,) int64 | 1 if teammate carries ball, 0 otherwise |

Where N = number of teammates (1 in Soccer 2v2).

**What stays the same:** The 3x3 local `image`, `direction`, and `mission`
are preserved unchanged from Soccer-Enhanced-v0. TeamObs only ADDS new keys.

**Why this exists:** On a 16x11 field with `view_size=3`, agents see only 7%
of the grid. Teammates are almost never visible in the 3x3 window. Without
TeamObs, passing is entirely blind. With TeamObs, agents can learn informed
passing strategies (e.g., "pass when teammate is near opponent goal").

**Design rationale -- SMAC observation pattern:**

This follows the standard MARL observation augmentation pattern established
by SMAC (Samvelyan et al., 2019). In SMAC, each agent receives its local
view unchanged, plus structured features about allies (relative positions,
health, unit type). This is the standard approach in cooperative MARL:

> Samvelyan, M., Rashid, T., de Witt, C. S., et al. (2019).
> "The StarCraft Multi-Agent Challenge." CoRR, abs/1902.04043.

The key insight: teammate features are **environment-level** observation
augmentation, not a training-time trick. The environment provides richer
observations; the RL algorithm decides what to do with them. This is
orthogonal to CTDE (Centralized Training, Decentralized Execution), which
is a training architecture choice.

---

### **Environment Comparison**

| Aspect | `Soccer-v0` | `Soccer-Enhanced-v0` | `Soccer-TeamObs-v0` |
|--------|-------------|---------------------|---------------------|
| **Status** | Deprecated | Recommended | Recommended |
| **Ball respawn** | No | Yes | Yes |
| **Termination** | Never | First to 2 goals | First to 2 goals |
| **Observability** | No ball carrier info | STATE channel encoding | STATE + teammate features |
| **Teammate info** | None | None (independent views) | Positions + directions + has_ball |
| **Passing strategy** | N/A (broken) | Blind teleport | Informed (knows teammate location) |
| **Cooldown** | None | 10-step dual | 10-step dual |
| **Map size** | 15x10 | 16x11 (FIFA ratio) | 16x11 (FIFA ratio) |
| **Use case** | Legacy only | Standard RL training | Team coordination research |

---

## Summary of Improvements

| Issue | Before (v1.0.2) | After (v2.0.0) |
|-------|-----------------|----------------|
| Ball respawn | Ball disappears | [FIXED] Respawns at random |
| Termination | 10,000 steps always | [FIXED] First to 2 goals |
| Observability | Can't see ball carrier | [FIXED] STATE channel + visual |
| Cooldown | None (infinite stealing) | [FIXED] 10-step dual cooldown |
| Passing | Adjacency handoff (1 cell) | [FIXED] Teleport to any teammate |
| Map size | 15x10 | [FIXED] 16x11 (FIFA ratio) |
| Teammate awareness | None (independent views) | [NEW] TeamObs variant (SMAC-style) |
| Training time | ~6 weeks | [IMPROVED] ~3 weeks (natural termination) |

---

## Conclusion

These improvements transform Soccer from a broken environment into a **research-grade testbed** for:
- Multi-agent coordination (passing, role specialization)
- Competitive team play (positive-only rewards, offense/defense balance)
- Emergent strategic behavior (MAPPO role discovery)
- Controlled observation ablation studies (Independent vs TeamObs)
- Credit assignment research (goal_scored_by tracking in info dict)
