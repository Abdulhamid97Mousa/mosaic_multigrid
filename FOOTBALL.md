# Soccer Environment - Improvements & Bug Fixes

## Overview

This document explains the improvements made to the **Soccer environment** in MOSAIC multigrid to fix critical bugs and enhance strategic gameplay for multi-agent reinforcement learning research.

---

## v6.3.0 Breaking Change: Walk-In Scoring

### **New Scoring Mechanic (Simplified for Better Learning)**

**Previous behavior (v6.0-6.2):** Agents had to execute `DROP` action while facing the goal square to score. This required:
1. Navigate to ball
2. Face ball
3. Execute `PICKUP` to grab ball
4. Navigate to opponent's goal
5. Face the goal square
6. Execute `DROP` to score

**New behavior (v6.3+):** Agents score by **walking into the goal square while carrying the ball**. Simplified to:
1. Navigate to ball
2. Execute `PICKUP` to grab ball
3. Navigate to opponent's goal and **walk into it** → automatic score!

**Why this change?**
- **Faster learning:** Removes one action step (no need to face goal and press DROP)
- **More intuitive:** Walking into goal while carrying ball = score (like real soccer)
- **Consistent with American Football:** Uses same walk-into-endzone mechanic
- **Emergent behaviors appear sooner:** Passing, stealing, defense strategies develop faster

**Implementation:**
```python
def step(self, actions):
    obs, rewards, terms, truncs, infos = super().step(actions)

    # Walk-in scoring: check if agent is standing on goal square while carrying
    for agent in self.agents:
        if agent.state.carrying is not None and not agent.state.terminated:
            pos = agent.state.pos
            pos_tuple = (int(pos[0]), int(pos[1]))

            # Check if agent is on a goal position
            for goal_pos, goal_team_idx in zip(self.goal_pos, self.goal_index):
                goal_pos_tuple = tuple(goal_pos)

                # Agent scored if standing on opponent's goal while carrying ball
                if pos_tuple == goal_pos_tuple and goal_team_idx != agent.team_index:
                    ball = agent.state.carrying

                    # Ball index 0 is wildcard (can score at any goal)
                    if ball.index in (0, goal_team_idx):
                        # GOAL! Award team reward
                        self._team_reward(agent.team_index, rewards, 1.0)

                        # Track which agent scored
                        self.goal_scored_by.append({...})

                        # Remove ball from agent and respawn
                        agent.state.carrying = None
                        new_ball = Ball(color=ball.color, index=ball.index)
                        self.place_obj(new_ball)

                        # Track team score and check win condition
                        self.team_scores[agent.team_index] += 1
                        if self.team_scores[agent.team_index] >= self.goals_to_win:
                            for a in self.agents:
                                a.state.terminated = True
                        break

    return obs, rewards, terms, truncs, infos
```

**DROP action behavior change:**
- **v6.0-6.2:** DROP = score at goal OR pass OR drop on ground
- **v6.3+:** DROP = teleport pass to teammate OR drop on ground (scoring removed from DROP)

---

## Critical Bugs Fixed (v6.0.0)

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

### **1. Walk-In Scoring (v7.0+)**

**How it works:**
Agents score by **walking into the opponent's goal square while carrying the ball**. No need to face the goal or press DROP - just navigate and step into the goal!

**Soccer goal layout:**
```
Single goal square per team (1x1 tile):
- Green team (team 1) defends goal at (1, 5) - left side
- Blue team (team 2) defends goal at (14, 5) - right side

To score: Agent must step onto opponent's goal square while carrying ball
```

**Detection in step():**
```python
# After each step, check if any agent is standing on a goal while carrying
for agent in self.agents:
    if agent.state.carrying is not None:
        pos = agent.state.pos
        pos_tuple = (int(pos[0]), int(pos[1]))

        # Check all goal positions
        for goal_pos, goal_team_idx in zip(self.goal_pos, self.goal_index):
            if tuple(goal_pos) == pos_tuple:
                # Agent is on a goal! Is it opponent's goal?
                if goal_team_idx != agent.team_index:
                    # GOAL! Score, respawn ball, check win condition
                    self._team_reward(agent.team_index, rewards, 1.0)
                    agent.state.carrying = None
                    self.place_obj(new_ball)
                    self.team_scores[agent.team_index] += 1
                    if self.team_scores[agent.team_index] >= self.goals_to_win:
                        # Team wins - terminate episode
                        for a in self.agents:
                            a.state.terminated = True
```

**Why walk-in over drop-based scoring?**
- **Simpler learning objective:** Navigate → pickup → navigate → score (3 steps)
- **No action sequencing:** Don't need to learn "face goal → press DROP" sequence
- **More intuitive:** Like real soccer - carry ball into goal area
- **Faster convergence:** Reduces action space complexity for RL agents

---

### **2. Teleport Passing (All Environments)**

> **Consistent across Soccer, Basketball, and American Football**

**How it works (priority order):**
```python
def _handle_drop(self, agent_index, agent, rewards):
    if agent.state.carrying is None:
        return

    fwd_pos = agent.front_pos
    fwd_obj = self.grid.get(*fwd_pos)

    # -- Priority 1: Teleport pass to teammate -------------
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

    # -- Priority 2: Ground drop ---------------------------
    if fwd_obj is None and self._agent_at(fwd_pos) is None:
        self.grid.set(*fwd_pos, agent.state.carrying)
        agent.state.carrying.cur_pos = fwd_pos
        agent.state.carrying = None
```

**Why teleport passing?**
- **Old (adjacency):** Agent must face teammate in the adjacent cell -- extremely
  hard for RL to coordinate 1-cell alignment on a 14x9 grid.
- **New (teleport):** Agent presses `DROP` and the ball warps to any free
  teammate. RL only needs to learn *when* to pass, not *where* to stand.

**AEC (Agent-Environment Cycle) support:**
- Agents execute actions in sequential order
- One agent moves → others see updated positions
- Enables coordinated steals: Agent A moves away → Agent B sees opportunity → Agent B steals
- Natural turn-based dynamics for multi-agent coordination

**Real-world parallel:** Like a long-ball pass in real soccer -- instant ball
transfer to a better-positioned teammate anywhere on the pitch

---

### **3. Stealing Mechanism (All Environments)**

> **Consistent across Soccer, Basketball, and American Football**

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

**AEC support for coordinated defense:**
- **Sequential execution:** Agent 0 moves → Agent 1 sees new position → Agent 2 decides to steal
- **Reactive defense:** Teammates can see ball carrier move and react in same timestep
- **Strategic positioning:** Defense agents can block passing lanes based on real-time positions

**Why it matters:**
- **Defense pressure:** Opponent can steal ball, forcing offensive urgency
- **Team coordination:** While one agent is in cooldown, teammate can support/recover
- **Emergent behavior:** Defensive specialist learns to intercept opponents

**Real-world parallel:** Like real soccer tackles - both players need recovery time

---

### **4. Goal Representation Across Sports**

| Sport | Goal Type | Size | Scoring Method |
|-------|-----------|------|----------------|
| **Soccer** | Single square | 1x1 tile | Walk into goal square while carrying ball |
| **Basketball** | Single square | 1x1 tile | Walk into goal square while carrying ball |
| **American Football** | End zone | Full vertical column (1x9 tiles) | Walk into end zone column while carrying ball |

**Soccer/Basketball:**
```
Single goal square:
  W W W W W W W W W W W W W W W W
  W . . . . . . . . . . . . . . W
  W . . . . . . . . . . . . . . W
  W . . . . . . . . . . . . . . W
  W . . . . . . . . . . . . . . W
  W G . . . . . . . . . . . . G W  ← Single goal squares
  W . . . . . . . . . . . . . . W     at (1,5) and (14,5)
  W . . . . . . . . . . . . . . W
  W . . . . . . . . . . . . . . W
  W . . . . . . . . . . . . . . W
  W W W W W W W W W W W W W W W W
```

**American Football:**
```
End zone (full column):
  W W W W W W W W W W W W W W W W
  W E . . . . . . . . . . . . E W  ← Entire column is scoring zone
  W E . . . . . . . . . . . . E W     Agent can score from any row
  W E . . . . . . . . . . . . E W     in the end zone column
  W E . . . . . . . . . . . . E W
  W E . . . . . . . . . . . . E W     at x=1 and x=14
  W E . . . . . . . . . . . . E W
  W E . . . . . . . . . . . . E W
  W E . . . . . . . . . . . . E W
  W E . . . . . . . . . . . . E W
  W W W W W W W W W W W W W W W W
```

**Why different goal sizes?**
- **Soccer/Basketball:** Single goal square = precision scoring, harder target
- **American Football:** End zone column = larger target, easier to reach, promotes running plays

**All three use the same walk-in mechanic:**
- Agent carries ball → steps into goal area → automatic score
- No need to face goal or press DROP
- Consistent learning objective across all sports

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

## Action Space

8 discrete actions per agent (same as all MOSAIC environments):

| Action | Index | Description |
|--------|-------|-------------|
| Noop | 0 | No operation — AEC compatibility (non-acting agents wait without moving) |
| Turn left | 1 | Rotate 90° counter-clockwise |
| Turn right | 2 | Rotate 90° clockwise |
| Move forward | 3 | Move one cell in facing direction |
| Pickup | 4 | Pick up ball from ground, or steal from opponent |
| Drop | 5 | Score at goal / teleport pass / drop on ground |
| Toggle | 6 | Unused in soccer |
| Done | 7 | Signal task completion |

Total action space: `Dict(0: Discrete(8), ..., 3: Discrete(8))` — one entry per agent.

**Why `noop` (index 0) was added — AEC + Parallel API compatibility:**

In AEC (Agent-Environment Cycle) mode, only one agent acts per physics step. All other
agents must still submit a *valid* action so the environment can advance. Without a no-op,
non-acting agents would accidentally execute `left` (turn left), corrupting episodes.

`noop=0` is the fix. This design is directly inspired by **MeltingPot** (Google DeepMind),
which uses `NOOP=0` for the same reason. The `done` action (index 7) signals intentional
task completion and is semantically different from `noop`.

> **Migration note (v1 → v2):** All action indices shifted **up by 1**.
> Any pre-trained policy or hardcoded action index from v1 will need updating:
> `left=0→1, right=1→2, forward=2→3, pickup=3→4, drop=4→5, toggle=5→6, done=6→7`

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

### **MosaicMultiGrid-Soccer-2vs2-TeamObs-v0** (Recommended for team coordination)

**Status:** [RECOMMENDED] For RL training requiring team coordination

```python
# TeamObs variant -- SMAC-style teammate awareness
env = gym.make(
    'MosaicMultiGrid-Soccer-2vs2-TeamObs-v0',
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

### **MosaicMultiGrid-Soccer-Solo-Green-IndAgObs-v0** and **MosaicMultiGrid-Soccer-Solo-Blue-IndAgObs-v0** (New in v6.0.0)

**Status:** [NEW] single agent, no opponent

```python
# Solo Green agent on 16x11 soccer field
env = gym.make('MosaicMultiGrid-Soccer-Solo-Green-IndAgObs-v0')
obs, _ = env.reset(seed=42)

# Only 1 agent, same field and goal layout as 1v1/2v2
print(len(env.unwrapped.agents))           # 1
print(env.unwrapped.agents[0].team_index)  # 1 (Green)

# Override view_size at runtime (no separate gym ID)
env = gym.make('MosaicMultiGrid-Soccer-Solo-Green-IndAgObs-v0', view_size=7)
```

**Why solo training exists:**

Training IPPO on the full 1v1 or 2v2 soccer game suffers from five compounding problems that make the policy fail to converge in reasonable time:

1. **Sparse reward:** scoring requires a precise 6-step causal chain (navigate to ball → face → pickup → navigate to goal → face → drop). On a 14×9 playable field with 8 actions, a random agent scores approximately 0 times in 100 episodes of 200 steps each.

2. **Non-stationarity:** in multi-agent training, the opponent's policy is changing during training. From each agent's perspective, the "environment" is non-stationary. This is the standard cooperative/competitive MARL challenge (Littman, 1994) but is amplified by the sparse reward -- the agent cannot distinguish "I played well but the opponent got lucky" from "my policy is bad."

3. **Observation poverty:** `view_size=3` covers only 7.1% of the 14×9 playable field. The agent spends most of its time seeing empty floor with no gradient-useful features.

4. **Zero-sum curriculum mismatch:** Collect (curriculum phase 1) uses `zero_sum=True` (rewards in [-1, +1]) while Soccer (phase 2) uses `zero_sum=False` (rewards in [0, +1]). Hot-swapping the environment corrupts the critic's baseline and value estimates.
5. **Under-training:** with approximately 26 scoring events in 4M training steps, the gradient signal is far too weak for reliable policy improvement. The policy needs hundreds to thousands of scoring events to learn the full pickup-navigate-score chain.

Solo training addresses problems 1 and 2 directly:
- **No opponent** → higher scoring probability (no one steals the ball or blocks the path)
- **Stationary environment** → the agent's "world" doesn't change during training

**Two variants for team-correct deployment:**

| Variant | Team | Agent index | Scores at | Deploy as |
|---------|------|-------------|-----------|-----------|
| Soccer-Solo-Green | Green (team 1) | `agent_0` | Blue goal (14, 5) | `agent_0` directly |
| Soccer-Solo-Blue | Blue (team 2) | `agent_0` | Green goal (1, 5) | Remap to `agent_1` |

The policy is team-dependent `pi_agent_0` trained as Green has implicitly learned "move right when carrying ball" via reward signal. Deploying it in the Blue slot (where the correct goal is to the left) will cause own-goals. This is why both Green and Blue solo variants exist.

**Inherited mechanics that become inert:**
- Teleport passing → no teammates, drops to ground
- Stealing → no opponents on the field
- Steal cooldown → never triggered
- First-to-2-goals termination → still active (agent scores twice to end)

---

### **Environment Comparison**

| Aspect | `Soccer-v0` | `Soccer-Enhanced-v0` | `Soccer-2vs2-TeamObs-v0` | `Soccer-Solo-*-v0` |
|--------|-------------|---------------------|---------------------|---------------------|
| **Status** | Deprecated | Recommended | Recommended | New (v6.0.0) |
| **Agents** | 4 (2v2) | 4 (2v2) | 4 (2v2) | 1 (solo, no opponent) |
| **Ball respawn** | No | Yes | Yes | Yes |
| **Termination** | Never | First to 2 goals | First to 2 goals | First to 2 goals |
| **Observability** | No ball carrier info | STATE channel encoding | STATE + teammate features | STATE channel encoding |
| **Teammate info** | None | None (independent views) | Positions + directions + has_ball | N/A (no teammates) |
| **Passing strategy** | N/A (broken) | Blind teleport | Informed (knows teammate location) | N/A (drops to ground) |
| **Cooldown** | None | 10-step dual | 10-step dual | N/A (no opponents) |
| **Map size** | 15x10 | 16x11 (FIFA ratio) | 16x11 (FIFA ratio) | 16x11 (FIFA ratio) |
| **Use case** | Legacy only | Standard RL training | Team coordination research | Curriculum pre-training |

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
| Solo pre-training | Not available | [NEW] Solo Green/Blue variants (v6.0.0) |
| Training time | ~6 weeks | [IMPROVED] ~3 weeks (natural termination) |

---

## Conclusion

These improvements transform Soccer from a broken environment into a **research-grade testbed** for:
- Multi-agent coordination (passing, role specialization)
- Competitive team play (positive-only rewards, offense/defense balance)
- Emergent strategic behavior (MAPPO role discovery)
- Controlled observation ablation studies (Independent vs TeamObs)
- Credit assignment research (goal_scored_by tracking in info dict)
