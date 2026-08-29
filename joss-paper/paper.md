---
title: 'mosaic_multigrid: A Sport-Themed Gymnasium Multi-Agent Gridworld Environment for Cooperative and Competitive Reinforcement Learning'
tags:
  - Python
  - Gymnasium
  - PettingZoo
  - multi-agent reinforcement learning
  - MARL
  - gridworld
  - partial observability
  - benchmark environment
authors:
  - name: Abdulhamid M. Mousa
    orcid: 0009-0003-8044-3468
    affiliation: 1
  - name: Ming Liu
    corresponding: true
    affiliation: 1
affiliations:
 - name: "School of Optics and Photonics, Beijing Institute of Technology, Beijing 100081, China"
   index: 1
date: 29 August 2026
bibliography: paper.bib
---

# Summary

`mosaic_multigrid` is a Gymnasium [@gymnasium] multi-agent reinforcement
learning (MARL) environment library providing cooperative and competitive
sport-themed gridworld tasks: **Basketball**, **American Football**, and
**Soccer**. The package extends the `gym-multigrid`
lineage [@gym-multigrid; @minigrid] with a modernised Gymnasium 1.0+ API,
Numba-JIT-accelerated observation generation [@numba] that runs 10 to
100 times faster than pure-Python encoding, and reproducible
per-environment seeding. Each sport family exposes matched solo,
one-sided cooperative, symmetric competitive, and asymmetric team
variants sharing a common set of
mechanics: teleport passing, ball stealing, walk-in
scoring, and two goal termination. This method
keeps algorithmic comparisons run across sports on an equal footing. The
library ships with adapters for both the PettingZoo Parallel and Agent
Environment Cycle (AEC) APIs [@pettingzoo], tested with
Gymnasium-compatible training pipelines such as RLlib [@rllib] and
Stable Baselines 3 [@sb3], and is used by the MOSAIC evaluation platform
[@mosaic] as one of its supported environments for cross-paradigm
comparison of MARL, LLM, and VLM decision-makers.

The upstream `gym-multigrid` Soccer environment shipped with several
long-standing bugs. \autoref{fig:soccer-diff} shows the visual
difference between the original upstream rendering and the corrected
`mosaic_multigrid` version, which restores proper goal geometry,
ball-carrier state visualisation, and FIFA-style field aesthetics.

![Left: upstream `gym-multigrid` Soccer rendering. Right: corrected
`mosaic_multigrid` Soccer rendering with proper 3-cell goal arcs,
ball-carrier state overlay, and FIFA-style field
graphics.\label{fig:soccer-diff}](../figures/soccer_comparison.png)

# Statement of need

Cooperative-competitive multi-agent reinforcement learning research
requires environments that are (i) fast enough to iterate on, (ii)
simple enough to reason about, and (iii) *rich enough to expose real
coordination problems*. The most widely-adopted small-scale MARL
benchmarks each solve one of these constraints well but not all three:
the Multi-Particle Environment (MPE) shipped with MADDPG [@maddpg] is
fast and simple but has minimal partial observability; SMAC [@smac] on
the StarCraft II engine is rich but slow, heavy, and tied to a
proprietary game engine; MAgent [@magent] scales to millions of agents
but the environments themselves are simple grid battles; Neural MMO
[@neuralmmo] targets large-scale multi-agent worlds at the cost of
setup complexity.

The `gym-multigrid` family of gridworld environments [@gym-multigrid;
@minigrid] historically filled the small, fast, partially-observable
niche, but the upstream `gym-multigrid` project has been unmaintained
since 2020, uses the legacy Gym 4-tuple API, and provides only two
game families (Soccer and Collect) with a limited set of variants.
There is currently no maintained, Gymnasium-native multi-agent
gridworld package that offers matched cooperative and competitive
team variants across multiple sport-themed games under a single API.
This gap becomes wider when researchers want to compare algorithms
across domains or study transfer between related but distinct
coordination tasks. `mosaic_multigrid` fills the gap.

The intended users are (a) MARL algorithm researchers who need a fast,
reproducible test-bed with real partial observability and team
coordination, (b) authors of cross-paradigm evaluation platforms
(such as MOSAIC [@mosaic]) that need a variety of MARL tasks,
and (c) instructors who need a small, self-contained set of
environments for teaching cooperative and competitive RL.

# State of the field

`mosaic_multigrid` sits at the intersection of two groups. On the
gridworld axis it descends from `gym-multigrid` [@gym-multigrid] and
`MiniGrid` [@minigrid]; on the MARL-API axis it aligns with PettingZoo
[@pettingzoo] and the multi-agent extensions of Gymnasium
[@gymnasium]. Compared to related work:

- **`gym-multigrid`** [@gym-multigrid]: the direct ancestor. Provides
  Soccer and Collect only, uses the legacy Gym 4-tuple API, and is
  unmaintained. `mosaic_multigrid` restores maintenance, modernises to
  the Gymnasium 5-tuple dict-keyed API, adds Basketball and American
  Football, adds JIT-accelerated observations, and fixes a critical
  reproducibility bug in the upstream global-RNG seeding.

- **`MiniGrid`** [@minigrid]: excellent single-agent gridworld
  research library, but not multi-agent by design.

- **`PettingZoo`** [@pettingzoo]: a *framework* for MARL APIs.
  `mosaic_multigrid` implements the PettingZoo Parallel and AEC APIs
  so that any PettingZoo-compatible trainer works out-of-the-box.


# Software design

The library is organised around four sport families implemented as
subclasses of a shared `MultiGridEnv` base class (adopted from the
`INI multigrid` architecture [@ini-multigrid]): `SoccerGame`,
`BasketballGame`, `AmericanFootballGame`, and `CollectGame`.

Each sport uses a distinct goal geometry that mirrors the physical
sport it represents (\autoref{fig:goals}): Basketball scores through a
single-cell hoop, Soccer scores through a three-cell FIFA-style arc,
and American Football scores across an end-zone column spanning all
playable rows. The Collect variant has no goal cells; agents score by
picking up wildcard balls.

![Goal geometry differs by sport: a 1-cell hoop for Basketball, a
3-cell arc for Soccer, and a full end-zone column for American
Football. Solo (`G-1v0`) variants shown to keep the goal geometry
uncluttered.\label{fig:goals}](../figures/goal_geometry.png)

**Observation model.** Each agent receives a partial local view of the
grid encoded as `(view_size, view_size, 3)` in a `[type, color, state]`
factored channel format, following Gymnasium convention. The default
`view_size=3` (retained from `gym-multigrid`) yields 7 to 9 percent
field visibility per agent, forcing the emergence of role
specialisation and communication-through-action rather than global
reasoning.

**Performance.** All observation generation is JIT-compiled with Numba
[@numba], producing 10 to 100 times speed-ups over the pure-Python
upstream encoding. Rendering uses `pygame` (adopted from
`INI multigrid` [@ini-multigrid]) rather than `matplotlib` for
interactive visualisation and headless `rgb_array` mode capture.

**API surface.** The package registers 81 environments with Gymnasium
following the naming scheme
`MosaicMultiGrid-<Sport>-[Team-]<Format>-[ObsVariant-]v1`, giving
users four categories of variant per sport
(\autoref{fig:envs-bb}, \autoref{fig:envs-af}, \autoref{fig:envs-s}):

1. **Solo** (`G-1v0`, `B-0v1`) for curriculum pre-training.
2. **Symmetric competitive** (`1v1`, `2v2`, `3v3`, `4v4`) for balanced
   team-vs-team benchmarks.
3. **One-sided cooperative** (`G-2v0` through `G-6v0`, `B-0v2` through
   `B-0v6`) for pure team-coordination training without an opponent.
4. **Asymmetric competitive** (`1v2`, `2v1`, `1v3`, `3v1`, `1v4`, `4v1`,
   `2v3`, `3v2`, `2v4`, `4v2`) for research on defender-outnumbered,
   attacker-outnumbered, and unequal-team-size scenarios.

All environments share consistent mechanics: teleport passing on the
DROP action, ball stealing with 10-step dual cooldown, walk-in scoring,
ball respawn on goal, and first-to-two-goals termination.

**Symmetry breaking under parameter sharing.** For shared-policy MARL
training, `mosaic_multigrid` is designed to work with the standard
EPyMARL `obs_agent_id` pattern [@epymarl], where a one-hot agent
identity is appended to each observation inside the training loop.
This keeps the environment observation space clean while allowing the
algorithm layer to break observation degeneracy [@obs-degeneracy]
under a fully-shared policy.

**Testing.** The package ships with 289 automated tests covering
environment construction, observation-space contracts, reward
correctness, event tracking (`goal_scored_by`, `passes_completed`,
`steals_completed`), rendering, and PettingZoo Parallel/AEC parity.

![Basketball (BB) environments on a 19×11 grid. Nine canonical
variants shown: 2 solo, 3 symmetric competitive (1v1, 2v2, 3v3), and
4 one-sided cooperative (G-2v0, G-3v0, B-0v2, B-0v3). All variants
share the same walk-in scoring and teleport-passing mechanics.
\label{fig:envs-bb}](../figures/envs_BB.png)

![American Football (AF) environments on a 16×11 grid with full
end-zone column scoring. The 9 canonical variants mirror the
Basketball layout to enable direct cross-sport algorithmic comparison.
\label{fig:envs-af}](../figures/envs_AF.png)

![Soccer (S) environments on a 16×11 grid with 3-cell FIFA-style goal
arcs at $y \in \{4,5,6\}$. The 9 canonical variants use the same
Green/Blue team schema and mechanics as Basketball and American
Football.\label{fig:envs-s}](../figures/envs_S.png)


# Methods

We model each cooperative task as a decentralised partially observable
Markov decision process (Dec-POMDP) [@dec-pomdp]
$\mathcal{M} = \langle \mathcal{N}, S, \mathcal{A}, T, O, \Omega, r, \gamma, \rho_0 \rangle$,
where the agents in $\mathcal{N}$ share a common team reward. At each
timestep the observation function $O$ emits per-agent **egocentric
partial observations**: each local view $o_i^t$ is a $(k, k, 3)$ grid
crop centred on agent $i$ and rotated to align with its facing
direction (default view size $k = 3$), so no agent observes the full
state and coordination must emerge under strict partial observability.
For shared-parameter policies we follow the EPyMARL convention
[@epymarl] and concatenate a one-hot agent-identity vector to each
local observation inside the training loop, breaking the observation
degeneracy [@obs-degeneracy] that would otherwise collapse a
fully-shared policy into identical action distributions across agents
[@shared-experience; @selective-sharing].


# Research impact statement

`mosaic_multigrid` enables a class of controlled, reproducible
multi-agent reinforcement learning experiments on cooperative and
competitive sport-themed gridworlds that were previously fragmented
across unmaintained forks of `gym-multigrid` and one-off in-house
environments. The package has already supported active research on
cross-paradigm multi-agent systems as the environment layer for the
MOSAIC evaluation platform for unified comparison of RL, LLM, VLM,
and human decision-makers [@mosaic], under submission to the Machine
Learning Open Source Software track of JMLR. This constitutes
realized use in an ongoing research program beyond a purely
methodological contribution.


# AI usage disclosure

Generative AI tools, specifically Claude Sonnet 4.5 (Anthropic,
2025) and GPT-5.1 (OpenAI, 2025), were used mainly for language
polishing and documentation drafting of this manuscript, and for
minor code refactoring assistance. All scientific content, software
design decisions, implementation, and validation results were
conceived, written, and verified by the authors; the majority of
the codebase was authored directly by the authors rather than
produced through autonomous or "vibe coding" workflows.


# Acknowledgements

We gratefully acknowledge the upstream authors of `gym-multigrid`
[@gym-multigrid], `MiniGrid` [@minigrid], and the `INI multigrid`
[@ini-multigrid] modernisation on which this work builds.

# References
