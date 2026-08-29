# Basketball 3v3 Environment

## Overview

A 3-on-3 basketball game environment for multi-agent reinforcement learning research. Teams score by **walking into the opposing team's hoop while carrying the ball**. Built on top of the MOSAIC multigrid engine with the same mechanics as Soccer IndAgObs (walk-in scoring, teleport passing, steal cooldown, ball respawn, first-to-N-goals termination).

<p align="center">
  <img src="https://github.com/Abdulhamid97Mousa/mosaic_multigrid/raw/main/figures/basketball_3vs3_render.png" width="600">
</p>

---

## v6.5.0 Changes

### Reward Recalibration

| Parameter | v6.4.0 | v6.5.0 | Reason |
|-----------|--------|--------|--------|
| Scoring reward | +1.0 | **+15.0** | Dominated by entropy bonus at +1.0; needs to be the unambiguous dominant signal |
| Timeout penalty | 0 | **−1.0** | Without pressure, agents learn to stall |
| Proximity reward | none | **+0.01/step within Manhattan dist 3** | Breaks "do nothing" local NE; continuous gradient toward ball from random init |
| `max_steps` | 200 | **300** | More time for 3v3 coordination; consistent with AF default; enables correct `buffer_size=2400` |
| `zero_sum` (3v3) | False | **True** | Competitive game: scoring team +reward, opposing team −reward |
| Ball provenance (`_ball_last_carrier_team`) | absent | **present** | Blocks same-team pickup-farming exploit |
| Pass-chain counter (`_ball_pass_count`) | absent | **present** | Penalises A→B→A bounce passes |

### Training Results (MAPPO, 1M steps)

Two runs completed under `lr=0.0007` and `lr=0.0003`. Both converged to
~0.01 entropy reduction with oscillating critic loss — structural non-stationarity
from 6 agents sharing one policy. A 3M-step run with `n_epochs=5` and `lr=0.0003`
is underway. American Football 2v2 (4 agents) solved cleanly in 1M steps
as a reference point.

---

## v6.3.0 Breaking Change: Walk-In Scoring

### **New Scoring Mechanic (Simplified for Better Learning)**

**Previous behavior (v6.0-6.2):** Agents had to execute `DROP` action while facing the goal square to score.

**New behavior (v6.3+):** Agents score by **walking into the goal square while carrying the ball**. Simplified to:
1. Navigate to ball
2. Execute `PICKUP` to grab ball
3. Navigate to opponent's goal and **walk into it** → automatic score!

**Why this change?**
- **Faster learning:** Removes one action step (no need to face goal and press DROP)
- **More intuitive:** Walking into hoop while carrying ball = score (like driving to the basket)
- **Consistent with Soccer and American Football:** Uses same walk-into-goal mechanic
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

## Registered Environments

| Environment ID | Observation Model | Description |
|----------------|-------------------|-------------|
| `MosaicMultiGrid-Basketball-3v3-IndAgObs-v1` | Independent agent views | Each agent sees only its 3x3 local window |
| `MosaicMultiGrid-Basketball-Solo-Green-IndAgObs-v1` | Solo (no opponent) | Single Green agent (team 1), scores at Blue goal (17, 5). New in v6.0.0. |
| `MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v1` | Solo (no opponent) | Single Blue agent (team 2), scores at Green goal (1, 5). New in v6.0.0. |

The 3v3 environments share the same game mechanics. The only difference is what information each agent receives in its observation. The solo variants use the same court and goal layout but with a single agent and no opponent — designed for curriculum pre-training.

---

## Quick Start

```python
import gymnasium as gym
import mosaic_multigrid.envs

# IndAgObs: independent 3x3 views only
env = gym.make('MosaicMultiGrid-Basketball-3v3-IndAgObs-v1', render_mode='rgb_array')
obs, info = env.reset(seed=42)

# obs is a dict keyed by agent index: {0: {...}, 1: {...}, ..., 5: {...}}
print(obs[0]['image'].shape)     # (3, 3, 3) -- partial view
print(obs[0]['direction'])       # 0-3 (right/down/left/up)

for step in range(200):
    actions = {i: env.action_space[i].sample() for i in range(6)}
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated[0]:  # A team scored 2 goals
        print(f"Game over in {step} steps!")
        break

env.close()
```

---

## Court Layout

### 17x9 Playable Area (19x11 Total)

<p align="center">
  <img src="https://github.com/Abdulhamid97Mousa/mosaic_multigrid/raw/main/figures/basketball_3vs3_render.png" width="600">
</p>


### Dimensions

| Dimension | Value |
|-----------|-------|
| Total grid | 19 x 11 (209 cells) |
| Walls | 56 cells (outer boundary) |
| Goals | 2 cells (fixed positions) |
| Playable | 17 x 9 = 153 cells |
| Aspect ratio | ~1.89:1 (close to NBA court 94ft x 50ft = 1.88:1) |

---

## Teams and Agents

| Team | Index | Color | Side | Agents | Goal (scores HERE) |
|------|-------|-------|------|--------|---------------------|
| Green | 1 | (30, 160, 50) | Left | 0, 1, 2 | G2 at (17, 5) |
| Blue | 2 | (30, 80, 200) | Right | 3, 4, 5 | G1 at (1, 5) |

- `agents_index = [1, 1, 1, 2, 2, 2]` -- 3 agents per team
- Each team defends the goal on their side and scores at the opposite goal
- Agent colors match team: Green triangles vs Blue triangles

---

## Game Mechanics

### Walk-In Scoring

An agent scores by:
1. Picking up the ball (PICKUP action, face ball and press action 4)
2. Navigating to the opposing team's goal square
3. **Walking into the goal square while carrying the ball** → automatic score!

**Basketball goal layout:**
```
Single goal square per team (1x1 tile):
- Green team (team 1) defends goal at (1, 5) - left baseline
- Blue team (team 2) defends goal at (17, 5) - right baseline

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

When a goal is scored:
- Scoring team receives +1 reward per agent (positive-only, opponents get 0)
- Ball respawns at a random empty cell
- Team scores are tracked (first to `goals_to_win` terminates the episode)
- `goal_scored_by`, `passes_completed`, and `steals_completed` events are recorded in the info dict for post-hoc credit assignment analysis

**Why walk-in over drop-based scoring?**
- **Simpler learning objective:** Navigate → pickup → navigate → score (3 steps)
- **No action sequencing:** Don't need to learn "face goal → press DROP" sequence
- **More intuitive:** Like driving to the basket - carry ball to the hoop
- **Faster convergence:** Reduces action space complexity for RL agents

---

### Teleport Passing (All Environments)

> **Consistent across Soccer, Basketball, and American Football**

The DROP action uses **teleport passing** (identical to Soccer IndAgObs):

```python
# Priority order when an agent presses DROP while carrying the ball:
# 1. Teleport pass to a random eligible teammate (anywhere on the grid)
# 2. Drop ball on ground (if forward cell is empty)
```

Teleport passing means the ball instantly transfers to a teammate anywhere on the court. The passer does NOT need to see the receiver. This makes passing a timing decision ("when to pass") rather than a positioning puzzle ("where to stand").

**AEC (Agent-Environment Cycle) support:**
- Agents execute actions in sequential order
- One agent moves → others see updated positions
- Enables coordinated steals: Agent A moves away → Agent B sees opportunity → Agent B steals
- Natural turn-based dynamics for multi-agent coordination

### Stealing (All Environments)

> **Consistent across Soccer, Basketball, and American Football**

An agent can steal the ball from an opponent by facing them and pressing PICKUP:

```python
# Stealing rules:
# - Must face the opponent (in the forward cell)
# - Opponent must be carrying the ball
# - Stealer must NOT be in cooldown
# - Only works against opponents (cannot steal from teammates)
```

**Dual cooldown**: After a successful steal, BOTH the stealer and victim enter a 10-step cooldown during which neither can steal or be stolen from. This prevents ping-pong stealing.

**AEC support for coordinated defense:**
- **Sequential execution:** Agent 0 moves → Agent 1 sees new position → Agent 2 decides to steal
- **Reactive defense:** Teammates can see ball carrier move and react in same timestep
- **Strategic positioning:** Defense agents can block passing lanes based on real-time positions

### Goal Representation Across Sports

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

### Ball Respawn

After each goal, the ball respawns at a random empty cell (seeded by `env.np_random`). This keeps the game flowing continuously.

---

## Episode Termination

| Criterion | Condition |
|-----------|-----------|
| **Terminated** | When any team scores `goals_to_win` goals (default: 2) |
| **Truncated** | When `max_steps` reached (default: 300) |
| **Winner** | First team to reach the goal threshold |

### Configuration

```python
# Default: first to 2 goals, 200 steps max
env = gym.make('MosaicMultiGrid-Basketball-3v3-IndAgObs-v1')

# Custom: first to 3 goals, 500 steps max
env = gym.make(
    'MosaicMultiGrid-Basketball-3v3-IndAgObs-v1',
    goals_to_win=3,
    max_steps=500,
)
```

---

## Observation Space

### IndAgObs (Independent Agent Observations)

Each agent receives:

```python
{
    'image': np.ndarray,     # (3, 3, 3) -- [Type, Color, State] per cell
    'direction': int,        # 0=right, 1=down, 2=left, 3=up
    'mission': str,          # "maximize reward"
}
```

The 3x3 image is a **directional** partial view: the agent is at the rear-center of its view, looking forward. The view rotates with the agent's direction:

Coverage: 9 cells out of 153 playable = ~6% of the court. Agents almost never see teammates or opponents in their local window, which makes team coordination a genuine challenge.

### Ball Carrying Observability

Agents can see when other agents in their view are carrying the ball, using the STATE channel encoding:

```
STATE channel values for agents:
  0-3:     Direction (right/down/left/up) when NOT carrying ball
  100-103: Direction + ball carrying flag (e.g., 101 = down + has ball)

Decoding:
  has_ball  = (state >= 100)
  direction = state % 100
```

This works because Basketball (like Soccer and Collect) has NO doors, so door state values (0-2) are unused and there is no conflict with the 100+ offset range.

---

## Reward Structure (v6.5.0)

### Full Reward Table (v6.5.0)

| Event | Reward | Condition |
|-------|--------|-----------|
| **Move toward goal** | +0.01 × Δdist | While **carrying**; Manhattan dist to opponent's goal square |
| **Steal ball from opponent** | +0.3 | Agent takes ball from an opponent **currently carrying** it |
| **Score basket** | +15 + (max_steps − step) × 0.05 | Walk into opponent's goal square while carrying |
| **Opponent scores** | −(15 + time_bonus) | When `zero_sum=True` |
| **Timeout (no winner)** | −1.0 (all agents) | Applied when `step_count >= max_steps` |

### Why Each Component Exists

**Carrying-gated distance shaping**

Only fires while the agent holds the ball and moves closer to the goal. Drop resets the
signal — `pickup → drop → pickup` cycling yields zero reward. This removes the exploit
where agents earned +0.1 per cycle by alternating pickup and drop in place.

**Steal reward (+0.3)**

Encourages active defense and interception. Requires a live opponent in possession —
cannot be farmed by picking up ground balls or passing between teammates.

**Time-efficiency scoring**

A gradient over the whole episode: scoring early is dramatically better than scoring late.
At step 10 the bonus is ≈+14.5; at step 290 it is ≈+0.5. Agents are pushed to find the
ball and advance immediately.

**Timeout penalty (−1.0)**

Without it, "do nothing" is a safe Nash equilibrium. The penalty makes stalling actively
costly for both teams.

### Own-Goal Prevention

Agents **cannot score on their own goal**. The scoring check in `step()` includes
an explicit `goal_team_idx != agent.team_index` guard:

```python
# In BasketballGameIndAgObsEnv.step():
for goal_pos, goal_team_idx in zip(self.goal_pos, self.goal_index):
    if pos_tuple == goal_pos_tuple and goal_team_idx != agent.team_index:
        # Only opponent's goal counts — own goal physically impossible
        if ball.index in (0, goal_team_idx):
            self._team_reward(agent.team_index, rewards, 15.0)
```

Green team (team_index=1) can only score at the Blue goal (goal_team_idx=2),
and vice versa. An agent standing on their own goal while carrying the ball
receives zero reward and the ball is not removed.

---

## Basketball Court Rendering

The basketball environment uses a custom pygame-based renderer (`rendering/basketball.py`) instead of the default tile-based grid rendering.

### Visual Elements

| Layer | Description |
|-------|-------------|
| Hardwood floor | Alternating vertical plank strips (light/dark) |
| Paint area | Shaded key areas near each basket (3 tiles deep, 5 tiles tall) |
| Court markings | White lines: boundary, center line, center circle, three-point arcs, free-throw semicircles, restricted area arcs, paint rectangles |
| Goal cells | Team-colored tiles on the baseline with colored border |
| Baskets | Backboard + team-colored hoop + rim + connector + net lines |
| Agent FOV | Semi-transparent team-colored overlay showing each agent's 3x3 directional view |
| Agents | Team-colored triangles pointing in facing direction, with ball indicator glow when carrying |
| Ball | Orange circle with cross-seam lines (basketball appearance) |
| Labels | Small white agent ID numbers |

### Court Configuration

```python
court_cfg = {
    'paint_depth': 3,        # Paint area extends 3 tiles from baseline
    'paint_half_h': 2,       # Paint area half-height (5 tiles total: cy-2 to cy+2)
    'three_pt_radius': 5.0,  # Three-point arc radius in tiles
    'center_radius': 1.5,    # Center circle radius in tiles
    'ft_circle_radius': 2,   # Free-throw semicircle radius in tiles
}
```

### Render Output

```python
env = gym.make('MosaicMultiGrid-Basketball-3v3-IndAgObs-v1', render_mode='rgb_array')
obs, info = env.reset(seed=42)
frame = env.render()
# frame.shape = (352, 608, 3)  -- uint8 RGB at 32px per tile
# height = 11 * 32 = 352, width = 19 * 32 = 608
```

---

## Comparison with Soccer and American Football

| Aspect | Soccer (2v2) | Basketball (3v3) | American Football |
|--------|-------------|------------------|-------------------|
| Grid size | 16 x 11 (14x9 playable) | 19 x 11 (17x9 playable) | 16 x 11 (14x9 playable) |
| Teams | 2v2 (4 agents) | 3v3 (6 agents) | 1v1, 2v2, 3v3 |
| Team colors | Green vs Red | Green vs Blue | Green vs Blue |
| Goal type | Single square (1x1) | Single square (1x1) | End zone column (1x9) |
| Goal positions | (1, 5) and (14, 5) | (1, 5) and (17, 5) | Columns x=1 and x=14 |
| Scoring method | Walk into goal square | Walk into goal square | Walk into end zone column |
| Aspect ratio | 1.56:1 (FIFA) | 1.89:1 (NBA) | 1.56:1 (NFL field) |
| Rendering | FIFA field (grass, lines) | Basketball court (hardwood, arcs, hoops) | Football field (turf, end zones) |
| Teleport passing | Yes | Yes | Yes |
| Steal cooldown | 10 steps (dual) | 10 steps (dual) | 10 steps (dual) |
| Ball respawn | Yes | Yes | Yes |
| Goals to win | 2 | 2 | 2 |
| Max steps | 300 | 300 | 300 |
| Zero-sum | Yes | Yes | Yes |
| Observation (IndAgObs) | 3x3 local view | 3x3 local view | 3x3 local view |

**Consistent mechanics across all three sports:**
- Walk-in scoring (carry ball → step into goal area → automatic score)
- Teleport passing (DROP action transfers ball to random teammate anywhere)
- Stealing with dual cooldown (prevents ping-pong stealing)
- AEC support (sequential agent execution for coordinated defense)
- Ball respawn after each goal

### Key Difference: Team Size

Basketball has 3 agents per team instead of 2. This creates:
- More complex coordination (3 agents to synchronize, not 2)
- Richer passing networks (2 possible pass targets vs 1)
- More defensive coverage needed (3 opponents to track)
- Higher-dimensional action space (6 agents total vs 4)

---


## Action Space

8 discrete actions per agent (same as all MOSAIC environments):

| Action | Index | Description |
|--------|-------|-------------|
| Noop | 0 | No operation — AEC compatibility (non-acting agents wait without moving) |
| Turn left | 1 | Rotate 90 degrees counterclockwise |
| Turn right | 2 | Rotate 90 degrees clockwise |
| Move forward | 3 | Move one cell in facing direction |
| Pickup | 4 | Pick up ball from ground, or steal from opponent |
| Drop | 5 | Teleport pass to teammate / drop on ground (v7.0+: scoring removed) |
| Toggle | 6 | Unused in basketball |
| Done | 7 | Signal task completion |

Total action space: `Dict(0: Discrete(8), ..., 5: Discrete(8))` -- one entry per agent.

`noop` (index 0) was added for AEC (Agent-Environment Cycle) compatibility, inspired by
MeltingPot (Google DeepMind). In AEC mode, non-acting agents submit `noop` so the
environment can advance without moving them. `done` (index 7) signals intentional task
completion and is semantically different from `noop`.

**v6.3.0 change:** The `DROP` action no longer scores at goals. Scoring is now handled
automatically when an agent walks into the goal square while carrying the ball. The DROP
action only handles teleport passing and ground drops.

---

## PettingZoo Integration

Basketball environments work with both PettingZoo APIs:

```python
from mosaic_multigrid.envs import BasketballGame6HIndAgObsEnv19x11N3
from mosaic_multigrid.pettingzoo import to_pettingzoo_env, to_pettingzoo_aec_env

# Parallel API (simultaneous stepping)
PZParallel = to_pettingzoo_env(BasketballGame6HIndAgObsEnv19x11N3)
env = PZParallel(render_mode='rgb_array')

# AEC API (sequential turn-based)
PZAec = to_pettingzoo_aec_env(BasketballGame6HIndAgObsEnv19x11N3)
env = PZAec(render_mode='rgb_array')
```

---

## Solo Variants (v6.0.0)

Single-agent basketball variants with **no opponent on the court**. The agent learns ball pickup, court navigation, and scoring mechanics in isolation before being deployed into a multi-agent game.

### Why Solo Training?

Training IPPO directly on 3v3 basketball is hard because:
- **Sparse reward:** the 6-step scoring chain (navigate → face → pickup → navigate → face → drop) is extremely unlikely to occur by random exploration on a 17x9 playable area
- **Non-stationarity:** 5 other agents are changing their policies during training, so the agent's "environment" keeps shifting
- **Credit assignment:** with 3 agents per team, it is hard to determine which agent's actions contributed to a goal

Solo training removes all three problems. The agent faces a stationary environment (no other policies changing) with higher scoring probability (no one to steal the ball or block the path).

### Registered Solo Environments

| Environment ID | Team | Scores at | Checkpoint key |
|----------------|------|-----------|----------------|
| `MosaicMultiGrid-Basketball-Solo-Green-IndAgObs-v1` | Green (team 1) | Blue goal (17, 5) | `agent_0` (deploy directly as `agent_0`) |
| `MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v1` | Blue (team 2) | Green goal (1, 5) | `agent_0` (remap to `agent_1` at deployment) |

### Quick Start

```python
import gymnasium as gym
from mosaic_multigrid.envs import *

# Solo Green agent on basketball court
env = gym.make('MosaicMultiGrid-Basketball-Solo-Green-IndAgObs-v1')
obs, info = env.reset(seed=42)
print(len(env.unwrapped.agents))  # 1
print(obs[0]['image'].shape)      # (3, 3, 3)

# Override view_size at make time (no separate gym ID needed)
env = gym.make('MosaicMultiGrid-Basketball-Solo-Green-IndAgObs-v1', view_size=7)
obs, info = env.reset(seed=42)
print(obs[0]['image'].shape)      # (7, 7, 3)
```

### What Becomes Inert

The solo classes inherit all IndAgObs mechanics, but several become no-ops:
- **Teleport passing:** no teammates → teammates list is empty → ball drops to ground instead
- **Stealing:** no opponents on the court → never triggered
- **Steal cooldown:** never triggered (no steals)
- **First-to-2-goals:** still active — agent can score twice to end the episode early

---

## Expected RL Training Dynamics

### Phase 1: Random Exploration (0-100k episodes)
- Agents discover movement, pickup, and drop actions
- Accidental goals from random walks
- No coordination

### Phase 2: Ball Seeking (100k-300k episodes)
- Learn to pick up the ball
- Learn to move toward goals
- Solo play, no passing

### Phase 3: Scoring + Stealing (300k-700k episodes)
- Consistent scoring when finding the ball
- Discover stealing mechanic
- Basic defensive positioning

### Phase 4: Passing + Role Specialization (700k-1.5M episodes)
- Learn to pass to open teammates
- Offensive/defensive role emergence
- 2-on-1 fast break plays
- Defensive rotation

### Phase 5: Advanced Team Play (1.5M+ episodes)
- Pick-and-roll style coordination
- Dynamic role switching
- Outlet passing after steals
- Zone defense vs man-to-man defense emergence

The 3v3 format creates richer emergent behaviors than 2v2 Soccer because:
- Passing has 2 targets (tactical choice, not forced)
- Defense requires 3 agents to cover 3 opponents
- Off-ball movement becomes meaningful (creating space for teammates)
