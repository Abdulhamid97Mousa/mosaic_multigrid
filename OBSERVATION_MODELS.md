# Observation Model: Independent Agent Observations (IndObs)

## Technical Reference with Empirical Data

---

## Scoring Mechanic

As of v6.3.0, all sport environments use walk-in scoring: an agent scores by occupying the goal cell while carrying the ball. The DROP action is reserved for teleport passing and ground drops. The observation space, including the STATE channel encoding for ball carrying (values 100--103), is unchanged by this modification. The walk-in mechanic is consistent across all three sports. Soccer and Basketball use a single-cell goal zone (in v6.7.0 a 3-cell arc at $y \in \{4, 5, 6\}$), while American Football uses a full end-zone column spanning all playable rows.

---

## 1. Motivation: The Partial Observability Problem

mosaic_multigrid environments use **partial observability**: each agent $i$ sees only a small window of the grid, not the full field. This is the standard setup in multi-agent reinforcement learning (MARL) research [Bernstein et al., 2002], where the environment **state** $s_t$ and the agent **observation** $o_t^i$ are fundamentally different objects.

Consider Soccer 2v2 on a 16x11 field with `view_size=3`:

```
Playable area:           14 x 9  = 126 cells
Agent view window:        3 x 3  =   9 cells
Visibility per agent:    9 / 126 = 7.1% of the field
```

Each agent is **blind to 93% of the field** at any given moment. On this field size, two agents on the same team are almost never within each other's 3x3 view.

### Empirical Teammate Visibility (500 steps, random actions)

Data collected from live mosaic_multigrid environments using random policies, 500 timesteps per environment.

**Soccer 2v2 (16x11 field):**

| Agent | Team | Steps teammate visible in 3x3 | Percentage |
|---|---|---|---|
| 0 | Green | 45 / 500 | 9.0% |
| 1 | Green | 45 / 500 | 9.0% |
| 2 | Blue | 0 / 500 | 0.0% |
| 3 | Blue | 0 / 500 | 0.0% |
| **Average** | | | **4.5%** |

**Collect 2v2 (10x10 field):**

| Agent | Team | Steps teammate visible in 3x3 | Percentage |
|---|---|---|---|
| 0 | Green | 22 / 500 | 4.4% |
| 1 | Green | 22 / 500 | 4.4% |
| 2 | Blue | 71 / 500 | 14.2% |
| 3 | Blue | 71 / 500 | 14.2% |
| **Average** | | | **9.3%** |

Over $10{,}000$ agent-steps of search (50 episodes $\times$ 200 steps), no agent ever observed another agent in its 3x3 image at initial reset. Even during play, teammate sightings occur less than 10% of the time. Without supplementary information, agents cannot coordinate.

---

## 2. State vs Observation

These are two distinct concepts in partially observable multi-agent systems.

### State $s_t$ (complete, global, hidden from agents)

The **state** is the god's-eye description of the environment at timestep $t$. Nothing is hidden. In mosaic_multigrid, the state consists of

```
Global state = grid.encode()

Shape: (width, height, 3) = (16, 11, 3) = 528 numbers

Contents:
  - 123 empty cells  (TYPE=1, COLOR=0, STATE=0)
  -  50 wall cells   (TYPE=2, COLOR=5, STATE=0)
  -   1 ball         (TYPE=6, COLOR=0, STATE=0)
  -   2 goal posts   (TYPE=11, COLOR=1 or 2, STATE=0)
  +   4 agent encodings embedded at their grid positions
```

plus the full agent state vectors:

```
Per agent: (pos_x, pos_y, direction, carrying, team_index)
Total for 4 agents: ~20 additional numbers
Grand total: ~548 numbers
```

The state is used **only by the centralised critic during MAPPO training** [Yu et al., 2022]. Agents never see it.

### Observation $o_t^i$ (partial, local, egocentric)

The **observation** is what agent $i$ actually receives at timestep $t$. It is a cropped, rotated window centred on the agent. In mosaic_multigrid, all environments use the **IndAgObs** (Independent Agent Observations) model described below.

---

## 3. IndAgObs: Independent Agent Observations

**Environment:** `MosaicMultiGrid-S-2v2-IndAgObs-v1`

Each agent receives a dictionary with three keys:

```python
obs[agent_id] = {
    'image':     ndarray(3, 3, 3),   # 27 numbers
    'direction': int,                 #  1 number  (0--3)
    'mission':   str,                 #  not used by neural networks
}
```

**Total numerical input per agent: 28 numbers** (27 image + 1 direction).

### The Image Tensor: 3 x 3 x 3

The image is a 3D array of shape `(view_size, view_size, 3)`, where the 3 channels encode each visible cell.

| Channel | Name | Range | What it encodes |
|---|---|---|---|
| 0 | TYPE | 0--12 | Object type at this cell |
| 1 | COLOR | 0--5 | Object colour (encodes team identity) |
| 2 | STATE | 0--103 | Object state (direction, door status, carrying flag) |

**TYPE values:**

| Index | Type | Description |
|---|---|---|
| 0 | unseen | Outside visibility (fog of war) |
| 1 | empty | Open floor cell |
| 2 | wall | Impassable wall |
| 6 | ball | Collectible ball |
| 10 | agent | Another agent |
| 11 | objgoal | Scoring goal zone |

**COLOR values:**

| Index | Color | Typical meaning |
|---|---|---|
| 0 | red | Wildcard ball / default |
| 1 | green | Team 1 (agents 0, 1) |
| 2 | blue | Team 2 (agents 2, 3) |
| 5 | grey | Walls |

**STATE values for agents:**

| Value | Meaning |
|---|---|
| 0 | Facing right, NOT carrying ball |
| 1 | Facing down, NOT carrying ball |
| 2 | Facing left, NOT carrying ball |
| 3 | Facing up, NOT carrying ball |
| 100 | Facing right, CARRYING ball |
| 101 | Facing down, CARRYING ball |
| 102 | Facing left, CARRYING ball |
| 103 | Facing up, CARRYING ball |

The 100-offset carrying flag was introduced to resolve a critical observability limitation: without it, an observing agent could not determine whether a visible teammate was in possession of the ball.

> **Note on colour terminology.** COLOR index 0 is labelled "red" because the ball object uses this colour slot. There is no "red team." The two competing teams are always Green (team 1, COLOR=1) and Blue (team 2, COLOR=2).

### Concrete Example: Agent 0 at seed = 42

```
Soccer 2v2 Enhanced, 16x11 field, seed = 42

Agent 0 (team 1, Green) at position (10, 4), facing down

3x3 image (egocentric, forward = row 0):

    Row 0:  [1, 0, 0]  [1, 0, 0]  [1, 0, 0]     empty  empty  empty
    Row 1:  [1, 0, 0]  [1, 0, 0]  [1, 0, 0]     empty  empty  empty
    Row 2:  [1, 0, 0]  [1, 0, 0]  [1, 0, 0]     empty  empty  empty

    Direction: 1 (down)
```

Agent 0 sees **9 empty cells**. No ball, no goals, no other agents. It has zero information about the game state beyond "I am surrounded by open floor and facing down."

Where are the other agents?

```
Agent 0: (10, 4)  <- this agent
Agent 1: (11, 2)  <- teammate, dx = +1, dy = -2  OUTSIDE 3x3 view
Agent 2: (11, 8)  <- opponent, dx = +1, dy = +4  OUTSIDE 3x3 view
Agent 3: (12, 5)  <- opponent, dx = +2, dy = +1  OUTSIDE 3x3 view
```

The 3x3 view covers $|dx| \leq 1$ and $|dy| \leq 1$. The teammate at $dy = -2$ is invisible. The opponents are even further away.

### Concrete Example: Agent 0 at seed = 2 (richer view)

```
Agent 0 (team 1, Green) at position (1, 3), facing down

3x3 image (egocentric):

    Row 0:  [ 1, 0, 0]  [ 1, 0, 0]  [ 1, 0, 0]     empty    empty    empty
    Row 1:  [11, 1, 0]  [ 1, 0, 0]  [ 1, 0, 0]     GOAL(G)  empty    empty
    Row 2:  [ 2, 5, 0]  [ 2, 5, 0]  [ 2, 5, 0]     wall     wall     wall

    Direction: 1 (down)
```

Here Agent 0 can see a goal post (TYPE = 11, COLOR = 1 = green, its own team's goal) and the bottom wall. It knows "I am near my team's goal, backing into a wall." But it still has no idea where its teammate is. The teammate (Agent 1) is at $(13, 4)$, **12 cells away** and completely invisible.

### What IndObs Cannot Express

With IndObs, an agent cannot answer any of these questions:

- Where is my teammate?
- Is my teammate carrying the ball?
- Which direction is my teammate facing?
- Is my teammate near the enemy goal (should I defend)?
- Is my teammate near our goal (should I attack)?

The agent must make all decisions based on its 9-cell local window. Coordination with teammates is limited to the rare moments (under 10% of timesteps) when both agents happen to occupy adjacent cells.

---

## 3b. JAX Observation Encoding (jaxmarl_worker)

The JAX training environments (`SoccerJAX`, `AmericanFootballJAX`, `BasketballJAX`) use a **continuous float encoding** that differs from the Gymnasium integer channels above. Each cell in the egocentric view window contributes three channels.

| Channel | Quantity |
|---|---|
| 0 | Object type |
| 1 | Colour (team or object) |
| 2 | State (direction, carrying) |

**Object type encoding** uses named constants that are asserted unique at module import time (preventing the invisible-goals regression):

| Constant | Value | Semantics |
|---|---|---|
| `_OBJ_FLOOR` | 1.0 | Passable empty cell |
| `_OBJ_WALL` | 2.0 | Impassable border |
| `_OBJ_GREEN_GOAL` | 5.0 | Green team's goal zone (STATIC_GRID base = 5) |
| `_OBJ_BLUE_GOAL` | 6.0 | Blue team's goal zone (STATIC_GRID base = 6) |
| `_OBJ_BALL` | 7.0 | Ball (no agent present) |
| `_OBJ_AGENT` | 10.0 | Agent (overrides all other values) |

The values are strictly increasing and injective: the network can distinguish any two cell types from channel 0 alone. The goal encoding uses distinct values (5.0 for green, 6.0 for blue) so that a shared policy can determine which goal is its scoring target without an explicit team-identity input.

The total per-agent observation is a flattened vector of length $3 \times \texttt{view\_size}^2$. For `view_size = 7` this gives $3 \times 49 = 147$ values per agent.

---

## References

Bernstein, D. S., Givan, R., Immerman, N., and Zilberstein, S. (2002). The complexity of decentralized control of Markov decision processes. *Mathematics of Operations Research*, 27(4), 819--840.

Kim, W. and Sung, Y. (2023). Parameter sharing with network pruning for scalable multi-agent deep reinforcement learning. In *Proceedings of the 22nd International Conference on Autonomous Agents and Multiagent Systems (AAMAS)*, London, UK.

Papoudakis, G., Christianos, F., Schäfer, L., and Albrecht, S. V. (2021). Benchmarking multi-agent deep reinforcement learning algorithms in cooperative tasks. In *Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks*.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint* arXiv:1707.06347.

Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., and Wu, Y. (2022). The surprising effectiveness of PPO in cooperative multi-agent games. In *Advances in Neural Information Processing Systems 35 (NeurIPS)*.
