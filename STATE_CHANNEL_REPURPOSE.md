# STATE Channel Repurposing: Using Door Space for Ball Carrying

## The Opportunity

### Current STATE Channel Usage:

```python
# From constants.py:
class State(str, IndexedEnum):
    open = 'open'      # Index 0 - ONLY used by doors
    closed = 'closed'  # Index 1 - ONLY used by doors
    locked = 'locked'  # Index 2 - ONLY used by doors

class Direction(enum.IntEnum):
    right = 0          # Used by agents
    down = 1           # Used by agents
    left = 2           # Used by agents
    up = 3             # Used by agents
```

### Soccer Environment Objects:

```python
# Objects that exist:
- Wall       → STATE = 0 (unused)
- Ball       → STATE = 0 (unused)
- Goal       → STATE = 0 (unused)
- Agent      → STATE = 0-3 (direction)
- Empty      → STATE = 0 (unused)

# Objects that DON'T exist:
- Door       ❌ Not in Soccer
- Key        ❌ Not in Soccer
```

**Conclusion**: Door state values (0-2) are **completely wasted** in Soccer! We can use higher values for "carrying ball"!

---

## Solution: Overload STATE Channel for Ball Carrying

### Approach 1: **Add Carrying Offset (Recommended)**

Use high values (100+) to indicate "agent carrying ball":

```python
# STATE channel encoding for agents:
STATE = direction                  # 0-3: Agent NOT carrying
STATE = 100 + direction            # 100-103: Agent IS carrying ball

Examples:
[Type.agent, Color.green, 0]   → Green agent facing RIGHT, no ball
[Type.agent, Color.green, 1]   → Green agent facing DOWN, no ball
[Type.agent, Color.green, 100] → Green agent facing RIGHT, HAS BALL ✅
[Type.agent, Color.green, 101] → Green agent facing DOWN, HAS BALL ✅
```

**Advantages:**
- ✅ Preserves both direction AND carrying information
- ✅ Clean separation (0-3 vs 100-103, no conflict)
- ✅ Backward compatible (existing code still works)
- ✅ Easy to decode: `has_ball = (state >= 100)`

**Implementation:**

```python
# In core/agent.py Agent.encode() method:
def encode(self) -> tuple[int, int, int]:
    """Encode agent as (type, color, state)."""
    state = self.state.dir  # 0-3

    # If carrying ball, add 100 to state
    if (self.state.carrying is not None and
        self.state.carrying.type == Type.ball):
        state += 100

    return (Type.agent.to_index(), self.state.color.to_index(), state)
```

---

### Approach 2: **Extend State Enum (Alternative)**

Add new state values to the State enum:

```python
class State(str, IndexedEnum):
    # Existing (for doors)
    open = 'open'           # 0
    closed = 'closed'       # 1
    locked = 'locked'       # 2

    # NEW: For agents carrying objects
    carrying_ball = 'carrying_ball'       # 3
    carrying_key = 'carrying_key'         # 4
    carrying_box = 'carrying_box'         # 5
```

**Disadvantages:**
- ❌ Loses direction information! (can't encode both state=3 and direction)
- ❌ Would need to change agent encoding to use COLOR or TYPE for direction
- ❌ More invasive change

**Not recommended** - Approach 1 is cleaner!

---

## Implementation Plan

### Step 1: Modify Agent Encoding

**File**: `mosaic_multigrid/core/agent.py`

```python
# Current code (line 139-152):
def encode(self) -> tuple[int, int, int]:
    """
    Encode a description of this agent as a 3-tuple of integers.

    Returns
    -------
    type_idx : int
        Index of the agent type.
    color_idx : int
        Index of the agent color.
    agent_dir : int
        The direction of the agent.
    """
    return (Type.agent.to_index(), self.state.color.to_index(), self.state.dir)
```

**New code**:

```python
def encode(self) -> tuple[int, int, int]:
    """
    Encode a description of this agent as a 3-tuple of integers.

    Returns
    -------
    type_idx : int
        Index of the agent type.
    color_idx : int
        Index of the agent color.
    state : int
        Agent state encoding:
        - 0-3: Direction (right, down, left, up) when NOT carrying
        - 100-103: Direction + carrying ball flag

    Notes
    -----
    The carrying ball flag uses offset 100 to avoid conflicting with
    door state values (0=open, 1=closed, 2=locked) used in other
    environments. Soccer and Collect don't have doors, so this space
    is available for repurposing.
    """
    state = self.state.dir  # Base direction: 0-3

    # Check if agent is carrying a ball
    if (self.state.carrying is not None and
        self.state.carrying.type == Type.ball):
        state += 100  # Add carrying flag

    return (Type.agent.to_index(), self.state.color.to_index(), state)
```

---

### Step 2: Add Decoding Constants

**File**: `mosaic_multigrid/utils/obs.py` (at top with other constants)

```python
# Add after line 48:

# Ball carrying flag for STATE channel
CARRYING_BALL_OFFSET = 100

# Helper for checking if agent has ball
def agent_has_ball(state_value: int) -> bool:
    """Check if STATE channel indicates agent is carrying ball."""
    return state_value >= CARRYING_BALL_OFFSET

def get_agent_direction(state_value: int) -> int:
    """Extract direction from STATE channel (removes carrying flag if present)."""
    return state_value % CARRYING_BALL_OFFSET
```

---

### Step 3: Update Documentation

**File**: `mosaic_multigrid/README.md` (line 321-326)

```python
# Current:
### Observation Format (Compatible with INI multigrid)

- `obs[agent_id]['image']` shape: `(view_size, view_size, 3)`
  - Channel 0: Object type (wall, ball, goal, agent, etc.)
  - Channel 1: Object color (red, blue, green, etc.)
  - Channel 2: Object state (open/closed door, agent direction)
```

**New**:

```python
### Observation Format (Enhanced for Multi-Agent)

- `obs[agent_id]['image']` shape: `(view_size, view_size, 3)`
  - Channel 0: Object type (wall, ball, goal, agent, etc.)
  - Channel 1: Object color (red, blue, green, etc.)
  - Channel 2: Object state encoding:
    - **For doors**: 0=open, 1=closed, 2=locked (standard MiniGrid)
    - **For agents**:
      - 0-3 = direction (right/down/left/up) when NOT carrying
      - 100-103 = direction + CARRYING BALL flag ✅
    - **For other objects**: 0 (unused)

**Key Enhancement**: Agents can now see when other agents are carrying the ball!
This solves the partial observability problem in Soccer and Collect environments.
```

---

### Step 4: Add Test

**File**: `mosaic_multigrid/tests/test_carrying_observability.py` (NEW)

```python
"""Test that agents can observe when other agents are carrying balls."""
import numpy as np
import pytest
from mosaic_multigrid.envs import SoccerGame4HEnv10x15N2
from mosaic_multigrid.core.constants import Type


def test_agent_can_see_other_carrying_ball():
    """Test that Agent 0 can see when Agent 1 is carrying a ball."""
    env = SoccerGame4HEnv10x15N2()
    obs, _ = env.reset(seed=42)

    # Give ball to Agent 1
    ball = env.grid.get(5, 5)  # Assume ball spawned here
    if ball and ball.type == Type.ball:
        env.agents[1].state.carrying = ball
        env.grid.set(5, 5, None)  # Remove from grid

    # Position Agent 0 next to Agent 1 so it's visible
    env.agents[0].state.pos = (4, 5)  # Left of Agent 1
    env.agents[0].state.dir = 0  # Facing right (towards Agent 1)

    # Generate observations
    obs = env.gen_obs()

    # Find Agent 1 in Agent 0's view
    agent0_view = obs[0]['image']

    # Agent 1 should be visible at position (2, 1) in view (forward-right)
    # Check the STATE channel
    for y in range(3):
        for x in range(3):
            if agent0_view[y, x, 0] == Type.agent.to_index():
                state_value = agent0_view[y, x, 2]

                # Check if this is Agent 1 (has ball)
                if state_value >= 100:
                    print(f"✅ Agent 0 can see Agent 1 carrying ball!")
                    print(f"   STATE value: {state_value}")
                    print(f"   Direction: {state_value % 100}")
                    assert True
                    return

    pytest.fail("Agent 0 could not see that Agent 1 is carrying ball")


def test_agent_without_ball_encoded_correctly():
    """Test that agents WITHOUT ball have normal direction encoding."""
    env = SoccerGame4HEnv10x15N2()
    obs, _ = env.reset(seed=42)

    # Agent 0 should NOT have ball initially
    assert env.agents[0].state.carrying is None

    # Check encoding
    encoding = env.agents[0].encode()
    type_idx, color_idx, state_value = encoding

    assert type_idx == Type.agent.to_index()
    assert 0 <= state_value <= 3, f"Agent without ball should have state 0-3, got {state_value}"
    print(f"✅ Agent without ball encoded correctly: state={state_value}")


def test_agent_with_ball_encoded_with_flag():
    """Test that agents WITH ball have carrying flag in STATE."""
    env = SoccerGame4HEnv10x15N2()
    obs, _ = env.reset(seed=42)

    # Give ball to Agent 0
    ball = env.grid.get(5, 5)
    if ball and ball.type == Type.ball:
        env.agents[0].state.carrying = ball
        env.grid.set(5, 5, None)

    # Check encoding
    encoding = env.agents[0].encode()
    type_idx, color_idx, state_value = encoding

    assert type_idx == Type.agent.to_index()
    assert state_value >= 100, f"Agent with ball should have state >= 100, got {state_value}"

    direction = state_value % 100
    assert 0 <= direction <= 3, f"Direction should be 0-3, got {direction}"

    print(f"✅ Agent with ball encoded correctly: state={state_value}, direction={direction}")
```

---

## Compatibility Analysis

### ✅ Backward Compatible Environments:

Environments with doors (if any) are **not affected**:
- Door STATE values remain 0-2
- Agent STATE values use 0-3 or 100-103 (no overlap)
- No breaking changes!

### ✅ No Conflict with Existing Code:

```python
# Existing code checking door states still works:
if obj.type == Type.door:
    if obj.state == State.open:  # 0
        pass  # Can walk through
    elif obj.state == State.closed:  # 1
        pass  # Can toggle
    elif obj.state == State.locked:  # 2
        pass  # Need key

# New code for agents:
if obj.type == Type.agent:
    if obj.state >= 100:
        has_ball = True
        direction = obj.state - 100
    else:
        has_ball = False
        direction = obj.state
```

**No conflicts!** Door states (0-2) and agent carrying states (100-103) are completely separate!

---

## Performance Impact

### Memory: **ZERO**
- Still 3 channels
- Still uint8 values (0-255 range, plenty of room)

### Computation: **NEGLIGIBLE**
- One extra check: `if carrying and type == ball: state += 100`
- One extra modulo: `direction = state % 100` (only when needed)

### Training: **HUGE IMPROVEMENT**
- Agents can now see who has the ball
- Better decision making (chase carrier, not random agents)
- Faster convergence expected

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Can see own carrying?** | ✅ Yes (at own position) | ✅ Yes (unchanged) |
| **Can see others carrying?** | ❌ NO! | ✅ YES! |
| **STATE encoding** | 0-3 (direction only) | 0-3 or 100-103 (direction + ball) |
| **Memory overhead** | - | **ZERO** (still 3 channels) |
| **Backward compatible** | - | ✅ Yes (doors still work) |
| **Training difficulty** | Very hard (blind) | Moderate (observable) |

---

## Recommendation

**Implement this immediately!**

This is a **zero-cost enhancement** that:
1. ✅ Fixes the observability bug
2. ✅ Uses wasted STATE channel space (doors don't exist in Soccer)
3. ✅ Maintains 3-channel efficiency
4. ✅ Backward compatible
5. ✅ Aligns with SOCCER_IMPROVEMENTS.md claims

The only question is: **Do you want this as default, or make it configurable?**

```python
# Option A: Always on (recommended)
class SoccerGameEnv(MultiGridEnv):
    # encoding happens automatically in Agent.encode()

# Option B: Configurable
class SoccerGameEnv(MultiGridEnv):
    def __init__(self, encode_ball_carrying: bool = True, ...):
        self.encode_ball_carrying = encode_ball_carrying
```

I recommend **Option A** (always on) - there's no downside!
