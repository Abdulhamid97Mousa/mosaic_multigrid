# Partial Observability in mosaic_multigrid

## Heritage: Why view_size=3?

**mosaic_multigrid inherits partial observability from the original gym-multigrid** (Fickinger 2020), which uses **view_size=3** for Soccer and Collect environments. This is a deliberate design choice for **competitive multi-agent scenarios**.

### Design Philosophy

We **kept view_size=3** from gym-multigrid because:
- ✅ Creates **challenging team coordination** problems
- ✅ Forces agents to **communicate and strategize**
- ✅ More **realistic** - agents can't see the entire field
- ✅ Proven in **research** (Fickinger et al., 2020)

We **updated other parts** to match modern standards (INI multigrid conventions):
- ✅ Gymnasium 1.0+ API (5-tuple dict-keyed)
- ✅ Numba JIT optimization
- ✅ Fixed reproducibility bugs
- ✅ Modern rendering (pygame)

### Comparison: gym-multigrid vs INI multigrid

| Aspect | mosaic_multigrid (our package) | INI multigrid |
|--------|--------------------------------|---------------|
| **Origin** | gym-multigrid (Fickinger 2020) | MiniGrid + modern redesign |
| **Soccer view_size** | **3** (competitive team play) | **7** (single-agent default) |
| **Use case** | Multi-agent coordination | Single-agent + exploration |
| **Difficulty** | ⭐⭐⭐⭐⭐ Very Hard | ⭐⭐⭐ Moderate |
| **API** | Gymnasium 1.0+ (dict-keyed) | Gymnasium 1.0+ (dict-keyed) |
| **Observation format** | Same (compatible!) | Same (compatible!) |

**Both use the same MiniGrid-style partial observability mechanism** - we just kept the smaller view size for competitive games!

### Design Rationale: Why We Kept view_size=3

**mosaic_multigrid is a modernization, not a redesign.** We kept `view_size=3` for Soccer/Collect environments because:

#### 1. **Research Continuity**
- Original gym-multigrid papers used `view_size=3`
- Changing it would break comparability with prior research
- Researchers expect Soccer to be hard, not easy

#### 2. **Competitive Multi-Agent Design**
Small view size creates **emergent coordination behaviors**:
- Agents must **share information** (via actions/positions)
- Teams develop **implicit communication** strategies
- Forces **strategic positioning** (scouts, defenders)
- Creates **fog of war** dynamics like real sports

#### 3. **What We DID Change (Matching INI multigrid)**

We modernized the **infrastructure** while keeping the **game design**:

| Component | Old (gym-multigrid) | New (mosaic_multigrid) | Rationale |
|-----------|---------------------|------------------------|-----------|
| **API** | Old Gym (4-tuple, lists) | Gymnasium 1.0+ (5-tuple, dicts) | Match INI multigrid + modern standards |
| **Seeding** | Broken (`np.random`) | Fixed (`self.np_random`) | Reproducible research |
| **Rendering** | matplotlib window | pygame | Match INI multigrid rendering |
| **Performance** | Pure Python loops | Numba JIT | 10-100× faster |
| **Structure** | 1442-line monolith | Modular package | Match INI multigrid architecture |
| **Observation format** | 6-channel encoding | 3-channel (type,color,state) | Match INI multigrid + MiniGrid |
| **Agent class** | Inherited from WorldObj | Separate Agent class | Match INI multigrid design |
| **view_size** | **3 (KEPT)** | **3 (KEPT)** | **Preserve game difficulty!** |

#### 4. **Research Philosophy**

Different view sizes serve different research questions:

**view_size=3 (mosaic_multigrid Soccer)**
- "How do agents coordinate with limited information?"
- "Can teams develop emergent communication?"
- "What strategies emerge under fog of war?"

**view_size=7 (INI multigrid)**
- "How do agents explore and navigate?"
- "Can agents learn efficient policies?"
- "What happens with more complete information?"

### What We Inherited vs What We Built

#### From gym-multigrid (Fickinger 2020) ✓ KEPT

✅ **Partial observability** (`view_size=3` for Soccer/Collect)
✅ **View rotation** (agent-centric reference frame)
✅ **Soccer and Collect game mechanics**
✅ **Team-based gameplay** (team_index system)
✅ **Ball passing, stealing, scoring** logic

#### From INI multigrid (2022+) ✓ ADOPTED

✅ **Gymnasium 1.0+ API** (5-tuple dict-keyed)
✅ **Agent class design** (separate from WorldObj)
✅ **3-channel encoding** `[type, color, state]`
✅ **pygame rendering** (instead of matplotlib)
✅ **Modular package structure**
✅ **Type/Color/State enums** architecture

#### Our Contributions ✓ ADDED

✅ **Reproducibility fix** (fixed global RNG bug)
✅ **Numba JIT optimization** (10-100× faster)
✅ **Comprehensive tests** (130 tests)
✅ **Framework adapters** (RLlib, PettingZoo)
✅ **Observation wrappers** (FullyObs, ImgObs, OneHot)
✅ **Production-ready** (type hints, docs, CI/CD ready)

---

## Agent View Size

Each agent has **limited perception** - they only see a local grid around them, not the entire environment.

### Default View: 3×3 (mosaic_multigrid - Competitive)

```
Soccer environment (view_size=3):

Full Grid (15×10):               Agent 0's View (3×3):
┌─────────────────────────────────┐              ┌──────┐
│ W  W  W  W  W  W  W  W  W  W  W │              │ W W W│  ← Top row (walls)
│ W  .  .  .  .  .  .  .  .  .  W │              │ . . .│  ← Middle row
│ W  🔵→ .  .  .  .  ⚽ .  . .  W │              │ .🔵 .│  ← Agent at bottom-center looking up
│ W  .  .  .  .  .  .  .  .  .  W │              └──────┘
│ W  .  .  .  .  .  .  .  .  .  W │
│ W  🟥 .  .  .  .  .  .  .  🟦 W │              Coverage: 9 cells (3×3)
│ W  .  .  .  .  .  .  .  .  .  W │              Forward: 2 tiles
│ W  .  .  .  .  .  .  .  .  .  W │              Sides: 1 tile each
│ W  🔵 .  .  .  .  .  .  .  🔴 W │
│ W  W  W  W  W  W  W  W  W  W  W │              ⚠️ CANNOT see ball! 
└─────────────────────────────────┘              ⚠️ CANNOT see goals!
                                                 ⚠️ CANNOT see teammates!
Legend:
🔵 Blue team  🔴 Red team  ⚽ Ball  🟥🟦 Goals  W=Wall  .=Empty
```

### Larger View: 7×7 (INI multigrid - Exploration)

```
Full Grid (15×10):                                Agent 0's View (7×7):
┌─────────────────────────────────┐              ┌─────────────────────┐
│ W  W  W  W  W  W  W  W  W  W  W │              │ W  W  W  W  W  W  W │
│ W  .  .  .  .  .  .  .  .  .  W │              │ W  W  W  W  W  W  W │
│ W  🔵→ .  .  .  .  ⚽ .  .  . W │              │ W  .  .  .  .  .  W │
│ W  .  .  .  .  .  .  .  .  .  W │              │ W  .  .  .  .  ⚽ W │  ← Can see ball!
│ W  .  .  .  .  .  .  .  .  .  W │              │ W  .  .  .  .  .  W │
│ W  🟥 .  .  .  .  .  .  .  🟦 W │              │ W  .  .  .  .  .  W │
│ W  .  .  .  .  .  .  .  .  .  W │              │ W  . 🔵→ .  .  .  W │  ← Agent
│ W  .  .  .  .  .  .  .  .  .  W │              └─────────────────────┘
│ W  🔵 .  .  .  .  .  .  .  🔴 W │
│ W  W  W  W  W  W  W  W  W  W  W │              Coverage: 49 cells (7×7)
└─────────────────────────────────┘              Forward: 6 tiles
                                                  Sides: 3 tiles each
If you changed mosaic_multigrid to view_size=7
                                                  ✅ CAN see ball!
                                                  ✅ CAN see more context!
```

### Side-by-Side Comparison

| Feature | view_size=3 (mosaic_multigrid) | view_size=7 (INI default) |
|---------|--------------------------------|---------------------------|
| **Total cells** | 9 cells (3×3) | 49 cells (7×7) |
| **Forward vision** | 2 tiles | 6 tiles |
| **Side vision** | 1 tile each | 3 tiles each |
| **Grid coverage** | ~5% of 15×10 field | ~30% of field |
| **Can see ball?** | ❌ Often not! | ✅ Usually yes |
| **Difficulty** | ⭐⭐⭐⭐⭐ Very Hard | ⭐⭐⭐ Moderate |
| **Research use** | Team coordination | Exploration/learning |
| **Requires memory** | ✅ Yes (LSTM/GRU) | Maybe (feedforward ok) |

## View Rotation

**The view rotates with the agent!** The agent is always at the bottom-center, facing "up" in its own reference frame.

```
Agent facing RIGHT (direction=0):     Agent facing DOWN (direction=1):
┌───────┐                              ┌───────┐
│ . . . │ → Agent's forward            │ . A . │
│ A . . │   view is to the right       │ . . . │
│ . . . │   in the global grid         │ . . . │
└───────┘                              └───────┘
                                      ↓ Forward view is downward

Agent facing LEFT (direction=2):      Agent facing UP (direction=3):
┌───────┐                              ┌───────┐
│ . . . │                              │ . . . │
│ . . A │ ← Forward view is            │ . . . │
│ . . . │   to the left                │ . A . |
└───────┘                              └───────┘
                                      ↑ Forward is upward
```

## Observation Encoding

Each cell in the view is encoded as a 3-tuple:

```python
obs[agent_id]['image'][y, x] = [type_idx, color_idx, state_idx]

# Example values:
[0, 0, 0]   → Unseen (outside view)
[1, 0, 0]   → Empty floor
[2, 0, 0]   → Wall (gray)
[7, 1, 0]   → Ball (red)
[10, 2, 0]  → Agent (blue)
[12, 1, 0]  → Goal (red)
```

### Complete Observation Dict

```python
obs = {
    0: {
        'image': np.array([...]),      # (view_size, view_size, 3) uint8
        'direction': 0,                # 0=right, 1=down, 2=left, 3=up
        'mission': 'maximize reward'   # Mission string
    },
    1: { ... },  # Agent 1's observation
    2: { ... },  # Agent 2's observation
    3: { ... },  # Agent 3's observation
}
```

## Why Partial Observability?

1. **Realistic**: Agents can't see the entire world
2. **Challenging**: Requires memory and coordination
3. **Scalable**: Observation size doesn't grow with grid size
4. **Research**: Studies emergent communication and teamwork

## Comparison: Partial vs Full Observability

```python
from mosaic_multigrid.envs import SoccerGame4HEnv10x15N2
from mosaic_multigrid.wrappers import FullyObsWrapper

# Partial observability (default)
env = SoccerGame4HEnv10x15N2()
obs, _ = env.reset()
print(obs[0]['image'].shape)  # (3, 3, 3) - small!

# Full observability (see entire grid)
env = FullyObsWrapper(SoccerGame4HEnv10x15N2())
obs, _ = env.reset()
print(obs[0]['image'].shape)  # (15, 10, 3) - entire grid!
```

## Configuring View Size

```python
from mosaic_multigrid.envs import SoccerGameEnv

# Custom view size
env = SoccerGameEnv(
    view_size=5,           # 5×5 view (instead of default 3×3)
    agents_index=[1,1,2,2],
    goal_pos=[[1,5], [13,5]],
    goal_index=[1, 2],
    num_balls=[1],
    balls_index=[0],
)

obs, _ = env.reset()
print(obs[0]['image'].shape)  # (5, 5, 3)
```

## Tips for Training

1. **Start small**: `view_size=3` is challenging but faster
2. **Increase gradually**: Try 5, 7 as agents improve
3. **Use RNNs**: Partial observability benefits from memory
4. **Communication**: Agents may learn to signal teammates
5. **Full obs baseline**: Compare with `FullyObsWrapper` for debugging

## Example: What Does the Agent See?

```python
from mosaic_multigrid.envs import SoccerGame4HEnv10x15N2

env = SoccerGame4HEnv10x15N2()
obs, _ = env.reset(seed=42)

# Agent 0's view
agent_view = obs[0]['image']  # (3, 3, 3)

# Decode cells
for y in range(3):
    for x in range(3):
        type_idx, color_idx, state_idx = agent_view[y, x]
        print(f"Cell ({x},{y}): type={type_idx}, color={color_idx}, state={state_idx}")

# Typical output:
# Cell (0,0): type=2, color=0, state=0  → Wall
# Cell (1,0): type=2, color=0, state=0  → Wall
# Cell (2,0): type=2, color=0, state=0  → Wall
# Cell (0,1): type=1, color=0, state=0  → Empty
# Cell (1,1): type=1, color=0, state=0  → Empty
# Cell (2,1): type=1, color=0, state=0  → Empty
# Cell (0,2): type=1, color=0, state=0  → Empty
# Cell (1,2): type=10, color=1, state=0 → Agent (self)
# Cell (2,2): type=1, color=0, state=0  → Empty
```

The agent sees walls ahead, empty floor, and itself at the bottom-center!
