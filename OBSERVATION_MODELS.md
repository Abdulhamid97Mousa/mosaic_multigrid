# Observation Models: IndObs vs TeamObs

## A Detailed Technical Comparison with Empirical Data

---

## 1. Motivation: The Partial Observability Problem

MOSAIC multigrid environments use **partial observability** -- each agent sees only a
small window of the grid, not the full field. This is the standard setup in multi-agent
reinforcement learning (MARL) research, where the environment **state** and the agent
**observation** are fundamentally different objects.

Consider Soccer 2v2 on a 16x11 field with `view_size=3`:

```
Playable area:          14 x 9 = 126 cells
Agent view window:       3 x 3 =   9 cells
Visibility per agent:    9 / 126 = 7.1% of the field
```

Each agent is **blind to 93% of the field** at any given moment. On this field size,
two agents on the same team are almost never within each other's 3x3 view.

### Empirical Teammate Visibility (500 steps, random actions)

The following data was collected from live MOSAIC environments using random policies
over 500 timesteps per environment:

**Soccer 2v2 (16x11 field):**

| Agent | Team | Steps teammate visible in 3x3 | Percentage |
|-------|------|-------------------------------|------------|
| 0 | Green | 45 / 500 | 9.0% |
| 1 | Green | 45 / 500 | 9.0% |
| 2 | Red | 0 / 500 | 0.0% |
| 3 | Red | 0 / 500 | 0.0% |
| **Average** | | | **4.5%** |

**Collect 2v2 (10x10 field):**

| Agent | Team | Steps teammate visible in 3x3 | Percentage |
|-------|------|-------------------------------|------------|
| 0 | Green | 22 / 500 | 4.4% |
| 1 | Green | 22 / 500 | 4.4% |
| 2 | Red | 71 / 500 | 14.2% |
| 3 | Red | 71 / 500 | 14.2% |
| **Average** | | | **9.3%** |

**Over 10,000 agent-steps of search (50 episodes x 200 steps), no agent ever observed
another agent in its 3x3 image at initial reset.** Even during play, teammate
sightings occur less than 10% of the time. Without supplementary information, agents
cannot coordinate.

---

## 2. State vs Observation

These are two distinct concepts in partially observable multi-agent systems.

### State `s_t` (complete, global, hidden from agents)

The **state** is the god's-eye description of the entire environment at timestep `t`.
Nothing is hidden. In MOSAIC multigrid, the state consists of:

```
Global state = grid.encode()

Shape: (width, height, 3) = (16, 11, 3) = 528 numbers

Contents:
  - 123 empty cells  (TYPE=1, COLOR=0, STATE=0)
  -  50 wall cells   (TYPE=2, COLOR=5, STATE=0)
  -   1 ball         (TYPE=6, COLOR=0, STATE=0)
  -   2 goal posts   (TYPE=11, COLOR=1/2, STATE=0)
  +  4 agent encodings embedded at their grid positions
```

Plus the full agent state vectors:

```
Per agent: (pos_x, pos_y, direction, carrying, team_index)
Total for 4 agents: ~20 additional numbers
Grand total: ~548 numbers
```

The state is used **only by the centralized critic during MAPPO training**. Agents
never see it.

### Observation `o_i_t` (partial, local, ego-centric)

The **observation** is what agent `i` actually receives at timestep `t`. It is a
cropped, rotated window centered on the agent. In MOSAIC multigrid, two observation
models are available:

- **IndObs** (Independent Observations) -- the default
- **TeamObs** (Team Observations) -- augmented with teammate features

---

## 3. IndObs: Independent Observations

**Environment:** `MosaicMultiGrid-Soccer-Enhanced-v0`

Each agent receives a dictionary with 3 keys:

```python
obs[agent_id] = {
    'image':     ndarray(3, 3, 3),   # 27 numbers
    'direction': int,                 #  1 number  (0-3)
    'mission':   str,                 #  (not used by neural networks)
}
```

**Total numerical input per agent: 28 numbers** (27 image + 1 direction).

### The Image Tensor: 3 x 3 x 3

The image is a 3D array of shape `(view_size, view_size, 3)`, where the 3 channels
encode each visible cell:

| Channel | Name | Range | What it encodes |
|---------|------|-------|-----------------|
| 0 | TYPE | 0-12 | Object type at this cell |
| 1 | COLOR | 0-5 | Object color (encodes team identity) |
| 2 | STATE | 0-103 | Object state (direction, door status, carrying flag) |

**TYPE values:**

| Index | Type | Description |
|-------|------|-------------|
| 0 | unseen | Outside visibility (fog of war) |
| 1 | empty | Open floor cell |
| 2 | wall | Impassable wall |
| 6 | ball | Collectible ball |
| 10 | agent | Another agent |
| 11 | objgoal | Scoring goal zone |

**COLOR values:**

| Index | Color | Typical meaning |
|-------|-------|-----------------|
| 0 | red | Wildcard ball / default |
| 1 | green | Team 1 (agents 0, 1) |
| 2 | blue | Team 2 (agents 2, 3) |
| 5 | grey | Walls |

**STATE values for agents:**

| Value | Meaning |
|-------|---------|
| 0 | Facing right, NOT carrying ball |
| 1 | Facing down, NOT carrying ball |
| 2 | Facing left, NOT carrying ball |
| 3 | Facing up, NOT carrying ball |
| 100 | Facing right, CARRYING ball |
| 101 | Facing down, CARRYING ball |
| 102 | Facing left, CARRYING ball |
| 103 | Facing up, CARRYING ball |

The 100-offset carrying flag was added in v1.0.0 to solve a critical observability
limitation where agents could not see when other agents were carrying the ball.
See SOCCER_IMPROVEMENTS.md for details.

### Concrete Example: Agent 0 at seed=42

```
Soccer 2v2 Enhanced, 16x11 field, seed=42

Agent 0 (team 1, Green) at position (10, 4), facing down

3x3 image (ego-centric, forward = row 0):

    Row 0:  [1, 0, 0]  [1, 0, 0]  [1, 0, 0]     empty  empty  empty
    Row 1:  [1, 0, 0]  [1, 0, 0]  [1, 0, 0]     empty  empty  empty
    Row 2:  [1, 0, 0]  [1, 0, 0]  [1, 0, 0]     empty  empty  empty

    Direction: 1 (down)
```

Agent 0 sees **9 empty cells**. No ball, no goals, no other agents. It has zero
information about the game state beyond "I am surrounded by open floor and facing
down."

Where are the other agents?

```
Agent 0: (10, 4)  <- this agent
Agent 1: (11, 2)  <- teammate, dx=+1, dy=-2  OUTSIDE 3x3 view
Agent 2: (11, 8)  <- opponent, dx=+1, dy=+4  OUTSIDE 3x3 view
Agent 3: (12, 5)  <- opponent, dx=+2, dy=+1  OUTSIDE 3x3 view
```

The 3x3 view covers `|dx| <= 1 AND |dy| <= 1`. The teammate at `dy=-2` is
invisible. The opponents are even further away.

### Concrete Example: Agent 0 at seed=2 (richer view)

```
Agent 0 (team 1, Green) at position (1, 3), facing down

3x3 image (ego-centric):

    Row 0:  [ 1, 0, 0]  [ 1, 0, 0]  [ 1, 0, 0]     empty    empty    empty
    Row 1:  [11, 1, 0]  [ 1, 0, 0]  [ 1, 0, 0]     GOAL(G)  empty    empty
    Row 2:  [ 2, 5, 0]  [ 2, 5, 0]  [ 2, 5, 0]     wall     wall     wall

    Direction: 1 (down)
```

Here Agent 0 can see a goal post (TYPE=11, COLOR=1=green = its own team's goal) and
the bottom wall. It knows "I am near my team's goal, backing into a wall." But it
still has no idea where its teammate is. The teammate (Agent 1) is at `(13, 4)`,
**12 cells away** -- completely invisible.

### What IndObs Cannot Express

With IndObs, an agent cannot answer any of these questions:

- Where is my teammate?
- Is my teammate carrying the ball?
- Which direction is my teammate facing?
- Is my teammate near the enemy goal (should I defend)?
- Is my teammate near our goal (should I attack)?

The agent must make all decisions based on its 9-cell local window. Coordination
with teammates is limited to the rare moments (< 10% of timesteps) when both
agents happen to occupy adjacent cells.

---

## 4. TeamObs: Team Observations

**Environment:** `MosaicMultiGrid-Soccer-2vs2-TeamObs-v0`

Each agent receives the **full IndObs observation unchanged**, plus 3 additional keys
about teammates:

```python
obs[agent_id] = {
    # --- Original IndObs (unchanged) ---
    'image':                ndarray(3, 3, 3),   # 27 numbers
    'direction':            int,                 #  1 number

    # --- New TeamObs features ---
    'teammate_positions':   ndarray(N, 2),       #  2N numbers (relative dx, dy)
    'teammate_directions':  ndarray(N,),         #  N numbers  (0-3)
    'teammate_has_ball':    ndarray(N,),          #  N numbers  (0 or 1)

    'mission':              str,                 #  (unchanged)
}
```

Where `N` = number of teammates. In 2v2 Soccer, N=1 (one teammate per agent).

**Total numerical input per agent: 32 numbers** (28 IndObs + 4 TeamObs features).

### TeamObs Feature Definitions

**`teammate_positions`** -- shape (N, 2), dtype int64

Relative displacement from this agent to each teammate in grid coordinates.
Computed as `(teammate_x - agent_x, teammate_y - agent_y)`.

```
Example: Agent 0 at (10, 4), Agent 1 (teammate) at (11, 2)
    teammate_positions = [[+1, -2]]
    Meaning: "My teammate is 1 cell to the right and 2 cells up from me"

Antisymmetry property:
    Agent 0 sees teammate at [+1, -2]
    Agent 1 sees teammate at [-1, +2]   (exact negation)
```

The positions are **relative**, not absolute. This is a deliberate design choice:
relative coordinates are translation-invariant, so the policy does not need to
memorize specific grid locations. It only needs to learn spatial relationships
("teammate is nearby" vs "teammate is far away").

Bounds: `[-max_dim, +max_dim]` where `max_dim = max(width, height)`.

**`teammate_directions`** -- shape (N,), dtype int64

The direction each teammate is currently facing.

| Value | Direction |
|-------|-----------|
| 0 | Right (+x) |
| 1 | Down (+y) |
| 2 | Left (-x) |
| 3 | Up (-y) |

This tells the agent which way its teammate is looking, enabling prediction of
the teammate's next move or pass direction.

**`teammate_has_ball`** -- shape (N,), dtype int64

Whether each teammate is currently carrying the ball (1) or not (0).

This is critical for coordination: an agent can decide to run toward the enemy
goal if its teammate has the ball (expecting a pass), or fall back to defend if
neither teammate has possession.

### Concrete Example: Full TeamObs at seed=42

```
Soccer 2v2 TeamObs, 16x11 field, seed=42

Agent 0 (team 1, Green) at (10, 4), facing down:
    image:                [same 3x3x3 as IndObs -- 9 empty cells]
    direction:            1 (down)
    teammate_positions:   [[+1, -2]]       "teammate is 1 right, 2 up"
    teammate_directions:  [0]              "teammate faces right"
    teammate_has_ball:    [0]              "teammate NOT carrying ball"

Agent 1 (team 1, Green) at (11, 2), facing right:
    image:                [3x3x3 -- sees nothing]
    direction:            0 (right)
    teammate_positions:   [[-1, +2]]       "teammate is 1 left, 2 down"
    teammate_directions:  [1]              "teammate faces down"
    teammate_has_ball:    [0]              "teammate NOT carrying ball"

Agent 2 (team 2, Red) at (11, 8), facing left:
    teammate_positions:   [[+1, -3]]       "teammate is 1 right, 3 up"
    teammate_directions:  [0]              "teammate faces right"
    teammate_has_ball:    [0]

Agent 3 (team 2, Red) at (12, 5), facing right:
    teammate_positions:   [[-1, +3]]       "teammate is 1 left, 3 down"
    teammate_directions:  [2]              "teammate faces left"
    teammate_has_ball:    [0]
```

### Concrete Example: Ball Carrying Detection

When Agent 0 picks up the ball, the change propagates through TeamObs:

```
Before pickup:
    Agent 1 sees: teammate_has_ball = [0]   "teammate has no ball"

After Agent 0 picks up ball:
    Agent 1 sees: teammate_has_ball = [1]   "teammate HAS the ball!"
```

This is **instantaneous and global** -- Agent 1 knows its teammate has the ball
regardless of distance. With IndObs, Agent 1 would only know if it happened to be
within the 3x3 window at the exact moment of pickup (< 10% probability).

---

## 5. Side-by-Side Comparison

### Observation Structure

| Key | IndObs | TeamObs | Change |
|-----|--------|---------|--------|
| `image` | (3, 3, 3) int64 | (3, 3, 3) int64 | Unchanged |
| `direction` | Discrete(4) | Discrete(4) | Unchanged |
| `mission` | str | str | Unchanged |
| `teammate_positions` | -- | (N, 2) int64 | NEW |
| `teammate_directions` | -- | (N,) int64 | NEW |
| `teammate_has_ball` | -- | (N,) int64 | NEW |

### Numerical Size (Soccer 2v2, N=1 teammate)

| Metric | IndObs | TeamObs |
|--------|--------|---------|
| Numbers per agent | 28 | 32 |
| Total for 4 agents | 112 | 128 |
| Overhead | -- | +14% |

### Information Content

| Question | IndObs | TeamObs |
|----------|--------|---------|
| What is directly in front of me? | Yes (image) | Yes (image) |
| What direction am I facing? | Yes (direction) | Yes (direction) |
| Where is my teammate? | No (unless in 3x3 window, ~5% of time) | **Always** (relative position) |
| What direction is my teammate facing? | No (unless in view AND decode STATE) | **Always** (explicit integer) |
| Is my teammate carrying the ball? | No (unless in view AND STATE >= 100) | **Always** (explicit 0/1 flag) |
| Where are opponents? | Only if in 3x3 window | Only if in 3x3 window (same as IndObs) |

Note that TeamObs provides information **only about teammates**, not opponents. This
is a deliberate design choice following the SMAC convention: teammate awareness is
considered "radio communication" that a real team would have, while opponent locations
must be discovered through exploration.

---

## 6. How Observations Fit into MAPPO

MAPPO (Multi-Agent Proximal Policy Optimization) uses the **Centralized Training
with Decentralized Execution (CTDE)** paradigm introduced by Lowe et al. (2017) and
refined by Yu et al. (2022). The architecture has two components:

### The Actor (decentralized, per-agent)

Each agent has its own **actor network** (policy) that maps observations to actions:

```
pi_i(a | o_i_t) : observation -> action probabilities

Input:  o_i_t (the agent's local observation)
        IndObs:  28 numbers  [image(27) + direction(1)]
        TeamObs: 32 numbers  [image(27) + direction(1) + positions(2) + dir(1) + ball(1)]

Output: probability distribution over 7 actions
        [left, right, forward, pickup, drop, toggle, done]
```

The actor runs at **both training and execution time**. Each agent selects its
action using only its own observation -- it never sees the global state or other
agents' observations.

In parameter-sharing MAPPO (the most common variant for homogeneous agents), all
agents on the same team share a single actor network. The agent's index or
one-hot ID may be appended to the observation to break symmetry.

### The Critic (centralized, shared)

A single **critic network** (value function) estimates the expected return from
the global state:

```
V(s_t) : state -> value estimate

Input:  s_t (the GLOBAL state -- complete grid + all agent states)
        grid.encode()   = (16, 11, 3) = 528 numbers
        agent_states    = ~20 numbers
        Total           = ~548 numbers

Output: scalar V(s_t)  (estimated cumulative future reward)
```

The critic runs **only during training**. It is discarded at deployment. Its role
is to compute advantage estimates that guide actor updates:

```
A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

The advantage tells the actor: "was this action better or worse than expected
from this state?" The actor then adjusts its policy to favor actions with
positive advantage.

### The CTDE Training Loop

```
Initialize: actor_theta, critic_phi

For each episode:
    s_0, {o_0, o_1, o_2, o_3} = env.reset()

    For each timestep t = 0, 1, 2, ...:

        # DECENTRALIZED execution: each agent uses its OWN observation
        a_0 ~ pi_theta(. | o_0_t)      # Agent 0 picks action from 28 or 32 numbers
        a_1 ~ pi_theta(. | o_1_t)      # Agent 1 picks action from 28 or 32 numbers
        a_2 ~ pi_theta(. | o_2_t)      # Agent 2 picks action
        a_3 ~ pi_theta(. | o_3_t)      # Agent 3 picks action

        # Environment steps with all 4 actions simultaneously
        s_{t+1}, {o_i_{t+1}}, {r_i_t}, done = env.step({a_0, a_1, a_2, a_3})

        # CENTRALIZED training: critic sees full state
        V_t     = V_phi(s_t)           # Value of current state  (from 548 numbers)
        V_{t+1} = V_phi(s_{t+1})       # Value of next state

        # Generalized Advantage Estimation (GAE)
        delta_t = r_t + gamma * V_{t+1} - V_t
        A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}

        # PPO actor update (clipped surrogate objective)
        ratio = pi_theta(a_t | o_i_t) / pi_theta_old(a_t | o_i_t)
        L_actor = -min(ratio * A_t, clip(ratio, 1-eps, 1+eps) * A_t)

        # Critic update (value function regression)
        L_critic = (V_phi(s_t) - returns_t)^2

        # Update both networks
        theta <- theta - alpha * grad(L_actor)
        phi   <- phi   - alpha * grad(L_critic)
```

### Why the Critic Alone Is Not Enough

A common question: "The critic sees everything, so why does the actor need TeamObs?"

The critic provides the **training signal** (advantage estimates), but the actor
must learn a **policy conditioned on its observation**. Consider this scenario:

```
State: teammate has ball, is near enemy goal
Optimal action for Agent 0: run to defend own goal

Critic knows: V(state) is high because teammate is about to score
Advantage for "run to defend": positive (good strategic choice)

But with IndObs:
    Agent 0's observation: [empty, empty, empty, empty, empty, empty, empty, empty, empty]
    direction: 1
    Total information: "I see nothing, I face down"

    The actor receives gradient: "go defend more often"
    But the observation has NO FEATURE that correlates with "teammate has ball"
    The actor cannot learn WHEN to defend vs WHEN to attack
    Because the input is identical in both situations
```

With TeamObs:

```
Agent 0's observation includes:
    teammate_positions: [[+5, -3]]     <- teammate is far top-right
    teammate_has_ball: [1]             <- teammate HAS the ball

    Now the actor CAN learn:
    "When teammate_has_ball=1 AND teammate is near enemy goal -> I should defend"
    "When teammate_has_ball=0 -> I should search for the ball"

    The input features DIFFERENTIATE the two situations
```

This is the fundamental insight: the critic can compute correct advantages, but the
actor needs **input features that are predictive of the optimal action**. TeamObs
provides those features for team coordination scenarios.

---

## 7. Relationship to SMAC (StarCraft Multi-Agent Challenge)

The TeamObs design follows the observation augmentation pattern established by the
StarCraft Multi-Agent Challenge (SMAC), the most widely used MARL benchmark.

### Citation

```
Samvelyan, M., Rashid, T., de Witt, C. S., Farquhar, G., Nardelli, N.,
Rudner, T. G. J., Hung, C.-M., Torr, P. H. S., Foerster, J., and
Whiteson, S. (2019). "The StarCraft Multi-Agent Challenge."
In Proceedings of the 18th International Conference on Autonomous Agents
and MultiAgent Systems (AAMAS 2019), pp. 2186-2188.
arXiv: 1902.04043
```

### How SMAC Structures Observations

In SMAC, each agent (a StarCraft unit) receives:

1. **Local features**: health, shield, weapon cooldown, unit type
2. **Ally features**: relative distance, relative (x, y), health, shield, unit type
   for each visible ally
3. **Enemy features**: relative distance, relative (x, y), health, shield, unit type
   for each visible enemy

The key design choice is that **ally features are provided regardless of sight range**.
An agent always knows where its allies are, even if they are on the other side of the
map. This models the real-world assumption that a team has radio communication.

Enemy features, by contrast, are only available within the agent's sight radius.
Enemies outside the sight range produce zero-valued features.

### MOSAIC TeamObs Follows the Same Pattern

| SMAC Concept | MOSAIC TeamObs Equivalent |
|---|---|
| Ally relative (x, y) | `teammate_positions` (N, 2) |
| Ally unit type | Not needed (homogeneous agents) |
| Ally health/shield | Not applicable (no health) |
| Ally direction | `teammate_directions` (N,) |
| Ally carrying status | `teammate_has_ball` (N,) |
| Enemy features | Not included (must discover through 3x3 view) |
| Local image features | `image` (3, 3, 3) -- unchanged from IndObs |

The design rationale is the same in both systems:

1. **Ally information is "free"** -- models radio communication within a team
2. **Enemy information must be earned** -- requires exploration and positioning
3. **Local view is unchanged** -- the augmentation is additive, not a replacement

### Why SMAC Needed This

Samvelyan et al. (2019) found that without ally features, agents in StarCraft could
not learn basic formations like "focus fire" (all allies attack the same enemy) or
"kiting" (ranged units retreat while attacking). The observation space simply did not
contain enough information to distinguish "ally is nearby and healthy" from "ally is
far away and dying."

The same problem exists in MOSAIC multigrid: without TeamObs, a Soccer agent cannot
distinguish "teammate has ball near enemy goal" from "teammate is on the other side
of the field." Both situations produce identical IndObs observations when the teammate
is outside the 3x3 window (which is 90-95% of the time).

---

## 8. Strategies Enabled by TeamObs

### With IndObs Only

The best a trained policy can learn is **individual ball-chasing**: move toward
visible balls, pick them up, move toward visible goals. There is no mechanism for
team coordination because the observation contains no teammate information.

Emergent "coordination" is accidental -- two agents happen to cover different parts
of the field because they were initialized in different positions, not because they
made a deliberate strategic choice.

### With TeamObs

The policy network receives the 4 extra numbers needed to learn **conditional
coordination strategies**:

**Coverage splitting:**
```
if teammate_positions[0][1] < 0:    # teammate is above me (dy < 0)
    -> I should search the bottom half
else:                                # teammate is below me
    -> I should search the top half
```

**Positional play:**
```
if teammate_has_ball[0] == 1:       # teammate has the ball
    if teammate near enemy goal:     # (inferred from positions)
        -> I run to defend our goal (anticipate counterattack)
    else:
        -> I position near enemy goal (to receive pass and score)
```

**Pass coordination (Soccer):**
```
if I have ball AND teammate near enemy goal:
    -> use teleport pass (drop action near teammate)
if I have ball AND teammate near our goal:
    -> carry ball myself toward enemy goal
```

**Strategic stealing (Soccer):**
```
if teammate_has_ball[0] == 0:       # nobody on our team has ball
    -> aggressively pursue opponent who likely has ball
if teammate_has_ball[0] == 1:       # teammate has ball
    -> play defense, block opponents from stealing
```

These strategies are impossible to learn with IndObs because the policy has no
input features that correlate with teammate state.

---

## 9. Environment Registry

MOSAIC multigrid provides paired variants for direct comparison:

### Soccer Environments

| Environment ID | Observation Model | Use Case |
|---|---|---|
| `MosaicMultiGrid-Soccer-Enhanced-v0` | IndObs | Baseline (individual play) |
| `MosaicMultiGrid-Soccer-2vs2-TeamObs-v0` | TeamObs | Team coordination research |

### Collect Environments

| Environment ID | Observation Model | Use Case |
|---|---|---|
| `MosaicMultiGrid-Collect-2vs2-Enhanced-v0` | IndObs | Baseline (individual play) |
| `MosaicMultiGrid-Collect-2vs2-TeamObs-v0` | TeamObs | Team coordination research |

Note: The individual Collect variant (`MosaicMultiGrid-Collect-Enhanced-v0`) has 3
agents on 3 separate teams (N=0 teammates per agent). TeamObs produces empty arrays
in this case and provides no benefit.

### Usage

```python
import gymnasium as gym

# IndObs -- standard partial observability
env_ind = gym.make('MosaicMultiGrid-Soccer-Enhanced-v0', render_mode='rgb_array')
obs, _ = env_ind.reset(seed=42)
print(obs[0].keys())   # dict_keys(['image', 'direction', 'mission'])

# TeamObs -- with teammate awareness
env_team = gym.make('MosaicMultiGrid-Soccer-2vs2-TeamObs-v0', render_mode='rgb_array')
obs, _ = env_team.reset(seed=42)
print(obs[0].keys())   # dict_keys(['image', 'direction', 'mission',
                        #            'teammate_positions', 'teammate_directions',
                        #            'teammate_has_ball'])
```

---

## 10. Summary

| Aspect | IndObs | TeamObs |
|--------|--------|---------|
| **Keys per observation** | 3 | 6 |
| **Numbers per agent** | 28 | 32 |
| **Overhead** | -- | +14% |
| **Teammate visibility** | ~5% of timesteps (3x3 window) | 100% (explicit features) |
| **Coordination learning** | Accidental only | Deliberate strategies possible |
| **SMAC equivalent** | Enemy-only features | Full ally + enemy features |
| **MAPPO actor input** | 28-dim vector | 32-dim vector |
| **MAPPO critic input** | ~548-dim state (same) | ~548-dim state (same) |
| **Recommended for** | Individual skill evaluation | Team coordination research |

---

## References

1. Samvelyan, M., Rashid, T., de Witt, C. S., Farquhar, G., Nardelli, N.,
   Rudner, T. G. J., Hung, C.-M., Torr, P. H. S., Foerster, J., and
   Whiteson, S. (2019). "The StarCraft Multi-Agent Challenge."
   In *Proceedings of AAMAS 2019*, pp. 2186-2188. arXiv: 1902.04043.

2. Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Baez, A., Liang, J.,
   and Wu, Y. (2022). "The Surprising Effectiveness of PPO in Cooperative
   Multi-Agent Games." In *Advances in Neural Information Processing Systems 35
   (NeurIPS 2022)*.

3. Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., and Mordatch, I.
   (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
   Environments." In *Advances in Neural Information Processing Systems 30
   (NeurIPS 2017)*.

4. Rashid, T., Samvelyan, M., de Witt, C. S., Farquhar, G., Foerster, J.,
   and Whiteson, S. (2018). "QMIX: Monotonic Value Function Factorisation
   for Deep Multi-Agent Reinforcement Learning." In *Proceedings of the 35th
   International Conference on Machine Learning (ICML 2018)*.
