# Partial Observability in mosaic_multigrid

## Background

The sport environments in mosaic_multigrid are instances of a **Decentralised Partially Observable Markov Decision Process** (Dec-POMDP) [Bernstein et al., 2002], the standard formalism for cooperative multi-agent reinforcement learning under partial information [Oliehoek and Amato, 2016].

A Dec-POMDP is defined by the tuple

$$
\langle\, M,\ \mathcal{S},\ \{\mathcal{A}_i\}_{i=1}^{M},\ \{\Omega_i\}_{i=1}^{M},\ \mathcal{T},\ \mathcal{O},\ r,\ \gamma\, \rangle,
$$

where $M$ is the number of agents, $\mathcal{S}$ is the state space, $\mathcal{A}_i$ is the action space of agent $i$, $\Omega_i$ is its local observation space, $\mathcal{T}: \mathcal{S}\times\boldsymbol{\mathcal{A}} \to \Delta(\mathcal{S})$ is the transition function, $\mathcal{O}: \mathcal{S}\times\boldsymbol{\mathcal{A}} \to \Delta(\boldsymbol{\Omega})$ is the joint observation function, $r: \mathcal{S}\times\boldsymbol{\mathcal{A}} \to \mathbb{R}$ is the shared team reward, and $\gamma\in[0,1]$ is the discount factor. The joint action and observation spaces are $\boldsymbol{\mathcal{A}}=\prod_i \mathcal{A}_i$ and $\boldsymbol{\Omega}=\prod_i \Omega_i$.

The global state $s_t \in \mathcal{S}$ is never directly observed. Instead, each agent $i$ receives a local observation $o_t^i \in \Omega_i$ and conditions its policy

$$
\pi_i: H_i \to \Delta(\mathcal{A}_i)
$$

on its own observation-action history $h_t^i \in H_i = (\Omega_i \times \mathcal{A}_i)^{*}$. In mosaic_multigrid, partial observability is implemented by an **egocentric view window** of configurable radius `view_size`. At each timestep, agent $i$ receives a cropped, rotation-normalised grid image of shape `(view_size, view_size, 3)`, together with a discrete direction scalar.

---

## View Size and Field Coverage

The default view size inherited from gym-multigrid [Fickinger et al., 2020] is `view_size=3`, yielding a 3x3 window. On the 16x11 Soccer field this covers 9 of 126 playable cells, roughly 7%.

| Parameter | view_size = 3 | view_size = 7 |
|---|---|---|
| Visible cells | 9 (3x3) | 49 (7x7) |
| Forward range | 2 cells | 6 cells |
| Lateral range | 1 cell each side | 3 cells each side |
| Field coverage (16x11) | ~7% | ~39% |
| Typical ball visibility | infrequent | frequent |
| Memory architecture | LSTM / GRU recommended | feedforward sufficient |

The 7x7 view is used in the JAX training environments (`jaxmarl_worker`) because it provides sufficient gradient signal for ball discovery without requiring a recurrent policy. The 3x3 window is retained in the Gymnasium environments for backward compatibility with prior work.

---

## Rationale for Partial Observability

### Research Continuity

The `view_size=3` configuration originates from gym-multigrid [Fickinger et al., 2020], which established the Soccer and Collect benchmarks under this observation regime. Retaining this setting ensures that results obtained on mosaic_multigrid are directly comparable to prior experimental work.

### Emergent Coordination Under Information Constraints

Small view windows create a qualitatively different problem from fully observable settings. When agents can see only a small neighbourhood, team-level coordination cannot be achieved through simple greedy local policies. This forces the emergence of spatial role specialisation (for example, attack and defend splits) and, in settings that augment observations with teammate state (TeamObs), explicit conditioning on communicated information.

### Parameter Sharing and Observation Degeneracy

A critical consequence of small view windows in settings that employ **parameter sharing** across agents [Christianos et al., 2021; Kim and Sung, 2023] is **observation degeneracy**. When multiple agents simultaneously occupy regions of the field with similar local structure (open floor, no visible ball or goals), their observations $o_t^i$ and $o_t^j$ become near-identical. Under a shared policy $\pi_\theta$, near-identical observations produce near-identical action distributions, leading to **crowding**: all agents converging to the same cell or direction.

This pathology is the primary motivation for the TeamObs observation variant (see `OBSERVATION_MODELS.md`), which provides each agent with unique teammate-relative features, breaking observation symmetry even when the local grid image is uninformative.

---

## Agent-Centric View Rotation

The observation window is rotation-normalised: agent $i$ is always placed at the bottom-centre of its own view, facing "up" in the egocentric frame. The global grid is rotated accordingly.

```
Global facing direction    Egocentric frame (agent at [view_size-1, view_size//2])
─────────────────────────  ─────────────────────────────────────────────────────
direction = 0 (right)      forward view is +x in global grid
direction = 1 (down)       forward view is +y in global grid
direction = 2 (left)       forward view is -x in global grid
direction = 3 (up)         forward view is -y in global grid
```

This convention ensures that the policy network $\pi_\theta$ learns direction-invariant spatial relationships ("ball is in front of me") without needing to separately represent all four rotations of the same configuration.

---

## Observation Encoding

Each cell in the view window is encoded as a three-channel integer tuple $[\text{type}, \text{color}, \text{state}]$:

| Channel | Name | Range | Semantics |
|---|---|---|---|
| 0 | TYPE | 0--12 | Object category at this cell |
| 1 | COLOR | 0--5 | Object colour (encodes team identity) |
| 2 | STATE | 0--103 | Direction, door status, or carrying flag |

**Selected TYPE values:**

| Value | Category |
|---|---|
| 0 | Unseen (outside view window) |
| 1 | Empty floor |
| 2 | Wall |
| 6 | Ball |
| 10 | Agent |
| 11 | Goal zone (ObjectGoal) |

**STATE encoding for agents.** The STATE channel doubles as a carrying indicator. Values $\{0, 1, 2, 3\}$ encode the agent's facing direction when not carrying a ball; values $\{100, 101, 102, 103\}$ encode direction plus a carrying flag (offset of 100). This representation was introduced to address an observability limitation in the base gym-multigrid encoding, where an observing agent could not determine whether a visible teammate was in possession of the ball.

| Value | Interpretation |
|---|---|
| 0--3 | Facing direction (right / down / left / up), not carrying |
| 100--103 | Facing direction (right / down / left / up), carrying ball |

---

## Field Coverage Comparison

The diagrams below illustrate the difference in field coverage between the two supported view sizes. Agent position is marked $A$, ball position is marked $*$, goal zones as $G_1$ and $G_2$, walls as $W$, and empty cells as a period.

```
view_size = 3 (9 cells)               view_size = 7 (49 cells)
─────────────────────────────         ─────────────────────────────────────
Full 16x11 field (excerpt):           Egocentric window:
W  W  W  W  W  W  W  W ...            W  W  W  W  W  W  W
W  .  .  .  .  .  .  . ...            W  .  .  .  .  .  W
W  A  .  .  .  *  .  . ...            W  .  .  .  .  *  W  <- ball visible
W  .  .  .  .  .  .  . ...            W  .  .  .  .  .  W
W  .  .  .  .  .  .  . ...            W  .  .  .  .  .  W
W G1  .  .  .  .  . G2 ...            W  .  A  .  .  .  W  <- agent
W  .  .  .  .  .  .  . ...            W─────────────────W

Egocentric window (3x3):              Coverage: 49 / 126 = 39%
W  W  W
.  .  .
.  A  .

Coverage: 9 / 126 = 7%
Ball position (6, 2): outside window
```

---

## Inherited Infrastructure vs. Redesigned Components

mosaic_multigrid modernises the gym-multigrid codebase while preserving the partial observability design. The table below separates inherited behaviour from replaced infrastructure.

| Component | Source | Status |
|---|---|---|
| `view_size=3`, view rotation | gym-multigrid [Fickinger et al., 2020] | Retained |
| Soccer and Collect game mechanics | gym-multigrid [Fickinger et al., 2020] | Retained |
| Team-based gameplay (`team_index`) | gym-multigrid [Fickinger et al., 2020] | Retained |
| Gymnasium 1.0+ API (5-tuple, dict-keyed) | INI multigrid | Adopted |
| 3-channel encoding $[\text{type}, \text{color}, \text{state}]$ | INI multigrid | Adopted |
| pygame rendering | INI multigrid | Adopted |
| Modular package structure | INI multigrid | Adopted |
| Reproducibility fix (seeded `np_random`) | mosaic_multigrid | New |
| Numba JIT optimisation | mosaic_multigrid | New |
| IndAgObs / TeamObs observation variants | mosaic_multigrid | New |
| American Football and Basketball environments | mosaic_multigrid | New |

---

## Configuring View Size

The view size may be overridden at environment construction time:

```python
from mosaic_multigrid.envs import SoccerGameEnv

env = SoccerGameEnv(
    view_size=5,
    agents_index=[1, 1, 2, 2],
    goal_pos=[[1, 5], [14, 5]],
    goal_index=[1, 2],
    num_balls=[1],
    balls_index=[0],
)
obs, _ = env.reset()
print(obs[0]['image'].shape)  # (5, 5, 3)
```

A `FullyObsWrapper` is also provided for ablation studies that compare partial and full observability:

```python
from mosaic_multigrid.wrappers import FullyObsWrapper

env_full = FullyObsWrapper(SoccerGameEnv(...))
obs, _ = env_full.reset()
print(obs[0]['image'].shape)  # (width, height, 3) -- entire grid
```

---

## References

Bernstein, D. S., Givan, R., Immerman, N., and Zilberstein, S. (2002). The complexity of decentralized control of Markov decision processes. *Mathematics of Operations Research*, 27(4), 819--840.

Christianos, F., Papoudakis, G., Rahman, A., and Albrecht, S. V. (2021). Scaling multi-agent reinforcement learning with selective parameter sharing. In *Proceedings of the 38th International Conference on Machine Learning (ICML)*, pages 1989--1998.

Fickinger, A., Cohen, S., Russell, S., and Amos, B. (2020). Multi-agent MDP homomorphic networks. *arXiv preprint* arXiv:2004.09676.

Kim, W. and Sung, Y. (2023). Parameter sharing with network pruning for scalable multi-agent deep reinforcement learning. In *Proceedings of the 22nd International Conference on Autonomous Agents and Multiagent Systems (AAMAS)*, London, UK.

Oliehoek, F. A. and Amato, C. (2016). *A Concise Introduction to Decentralized POMDPs*. Springer.

Samvelyan, M., Rashid, T., de Witt, C. S., Farquhar, G., Nardelli, N., Rudner, T. G. J., Hung, C.-M., Torr, P. H. S., Foerster, J., and Whiteson, S. (2019). The StarCraft multi-agent challenge. *arXiv preprint* arXiv:1902.04043.
