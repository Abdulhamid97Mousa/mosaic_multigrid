"""Tests for the MOSAIC multigrid core domain model.

Covers: Action enum, Type/Color/State/Direction enums, WorldObj encode/decode,
Grid set/get/walls/encode/decode, AgentState vectorized ops, Agent with
team_index, Mission/MissionSpace.
"""
import numpy as np
import pytest

from gym_multigrid.core import (
    Action,
    Agent,
    AgentState,
    Grid,
    Mission,
    MissionSpace,
    Ball,
    Box,
    Door,
    Floor,
    Goal,
    Key,
    Lava,
    ObjectGoal,
    Switch,
    Wall,
    WorldObj,
    Color,
    Direction,
    State,
    Type,
    TILE_PIXELS,
)


# ---------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------

class TestAction:
    def test_seven_actions(self):
        assert len(Action) == 7

    def test_action_values(self):
        assert Action.left == 0
        assert Action.right == 1
        assert Action.forward == 2
        assert Action.pickup == 3
        assert Action.drop == 4
        assert Action.toggle == 5
        assert Action.done == 6

    def test_no_still_action(self):
        names = [a.name for a in Action]
        assert 'still' not in names


# ---------------------------------------------------------------
# Type / Color / State / Direction enums
# ---------------------------------------------------------------

class TestType:
    def test_unseen_is_index_zero(self):
        assert Type.unseen.to_index() == 0

    def test_empty_is_index_one(self):
        assert Type.empty.to_index() == 1

    def test_mosaic_types_exist(self):
        """MOSAIC-specific types not in vanilla multigrid."""
        assert Type('objgoal') is not None
        assert Type('switch') is not None

    def test_roundtrip_index(self):
        for t in Type:
            assert Type.from_index(t.to_index()) == t

    def test_at_least_13_types(self):
        assert len(Type) >= 13


class TestColor:
    def test_six_colors(self):
        assert len(Color) == 6

    def test_rgb_returns_array(self):
        rgb = Color.red.rgb()
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (3,)
        assert rgb[0] == 255  # red channel

    def test_cycle(self):
        colors = list(Color.cycle(8))
        assert len(colors) == 8

    def test_from_index_roundtrip(self):
        for c in Color:
            assert Color.from_index(c.to_index()) == c


class TestDirection:
    def test_four_directions(self):
        assert len(Direction) == 4

    def test_values(self):
        assert Direction.right == 0
        assert Direction.down == 1
        assert Direction.left == 2
        assert Direction.up == 3

    def test_to_vec(self):
        vec = Direction.right.to_vec()
        assert tuple(vec) == (1, 0)
        vec = Direction.up.to_vec()
        assert tuple(vec) == (0, -1)


class TestState:
    def test_three_states(self):
        assert len(State) == 3


# ---------------------------------------------------------------
# WorldObj
# ---------------------------------------------------------------

class TestWorldObj:
    def test_dim_is_three(self):
        assert WorldObj.dim == 3

    def test_empty_singleton(self):
        e1 = WorldObj.empty()
        e2 = WorldObj.empty()
        assert e1 is e2  # cached

    def test_empty_type(self):
        e = WorldObj.empty()
        assert e.type == Type.empty

    def test_encode_decode_roundtrip(self):
        ball = Ball(color=Color.red, index=0)
        t, c, s = ball.encode()
        decoded = WorldObj.decode(t, c, s)
        assert decoded is not None
        assert decoded.type == Type.ball
        assert decoded.color == Color.red

    def test_from_array(self):
        wall = Wall()
        arr = np.array(wall)
        recovered = WorldObj.from_array(arr)
        assert recovered.type == Type.wall

    def test_wall_is_cached(self):
        w1 = Wall()
        w2 = Wall()
        assert w1 is w2  # functools.cache

    def test_ball_has_index_and_reward(self):
        b = Ball(color=Color.blue, index=2, reward=0.5)
        assert b.index == 2
        assert b.reward == 0.5

    def test_goal_has_index_and_reward(self):
        g = Goal(color=Color.green, index=1, reward=2.0)
        assert g.index == 1
        assert g.reward == 2.0

    def test_object_goal_mosaic_specific(self):
        og = ObjectGoal(
            color=Color.red, target_type='ball', index=1, reward=1.0)
        assert og.type == Type.objgoal
        assert og.target_type == 'ball'
        assert og.index == 1

    def test_switch_mosaic_specific(self):
        sw = Switch(color=Color.yellow)
        assert sw.type == Type.switch
        assert sw.can_overlap()

    def test_door_states(self):
        d = Door(color=Color.blue, is_open=False, is_locked=True)
        assert d.is_locked
        assert not d.is_open

    def test_can_pickup(self):
        assert Ball().can_pickup()
        assert Key().can_pickup()
        assert Box().can_pickup()
        assert not Wall().can_pickup()
        assert not Goal().can_pickup()

    def test_can_overlap(self):
        assert Goal().can_overlap()
        assert Floor().can_overlap()
        assert Lava().can_overlap()
        assert not Wall().can_overlap()
        assert not ObjectGoal().can_overlap()


# ---------------------------------------------------------------
# Grid
# ---------------------------------------------------------------

class TestGrid:
    def test_construction(self):
        g = Grid(10, 15)
        assert g.width == 10
        assert g.height == 15
        assert g.state.shape == (10, 15, WorldObj.dim)

    def test_set_get(self):
        g = Grid(5, 5)
        b = Ball(color=Color.red)
        g.set(2, 3, b)
        cell = g.get(2, 3)
        assert cell is b

    def test_set_none_clears(self):
        g = Grid(5, 5)
        g.set(2, 3, Ball())
        g.set(2, 3, None)
        cell = g.get(2, 3)
        assert cell is None

    def test_empty_cell_returns_none(self):
        g = Grid(5, 5)
        cell = g.get(2, 2)
        assert cell is None

    def test_horz_wall(self):
        g = Grid(10, 10)
        g.horz_wall(0, 0, length=10)
        # All cells in row 0 should be walls
        for x in range(10):
            assert g.state[x, 0, WorldObj.TYPE] == Type.wall.to_index()

    def test_vert_wall(self):
        g = Grid(10, 10)
        g.vert_wall(0, 0, length=10)
        for y in range(10):
            assert g.state[0, y, WorldObj.TYPE] == Type.wall.to_index()

    def test_wall_rect(self):
        g = Grid(10, 10)
        g.wall_rect(0, 0, 10, 10)
        # Borders should be walls
        for x in range(10):
            assert g.state[x, 0, WorldObj.TYPE] == Type.wall.to_index()
            assert g.state[x, 9, WorldObj.TYPE] == Type.wall.to_index()
        # Interior should be empty
        assert g.state[5, 5, WorldObj.TYPE] == Type.empty.to_index()

    def test_encode_decode_roundtrip(self):
        g = Grid(8, 8)
        g.wall_rect(0, 0, 8, 8)
        g.set(3, 3, Ball(color=Color.red))

        encoded = g.encode()
        g2, vis_mask = Grid.decode(encoded)

        assert g2.width == 8
        assert g2.height == 8
        assert vis_mask.all()  # all visible
        assert np.array_equal(g2.state, g.state)

    def test_render_returns_image(self):
        g = Grid(5, 5)
        g.wall_rect(0, 0, 5, 5)
        img = g.render(tile_size=32)
        assert img.shape == (5 * 32, 5 * 32, 3)
        assert img.dtype == np.uint8


# ---------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------

class TestAgentState:
    def test_single_agent_creation(self):
        s = AgentState()
        assert s.shape == (AgentState.dim,)

    def test_multi_agent_creation(self):
        s = AgentState(4)
        assert s.shape == (4, AgentState.dim)

    def test_pos_default_is_negative(self):
        s = AgentState()
        assert tuple(s.pos) == (-1, -1)

    def test_dir_default_is_negative(self):
        s = AgentState()
        # Uninitialized dir is -1 (not a valid Direction)
        assert s.dir == -1

    def test_set_pos(self):
        s = AgentState()
        s.pos = (5, 7)
        assert tuple(s.pos) == (5, 7)

    def test_set_dir(self):
        s = AgentState()
        s.dir = Direction.right
        assert s.dir == Direction.right

    def test_terminated_default_false(self):
        s = AgentState()
        assert s.terminated is False

    def test_set_terminated(self):
        s = AgentState()
        s.terminated = True
        assert s.terminated is True

    def test_carrying_default_none(self):
        s = AgentState()
        assert s.carrying is None

    def test_set_carrying(self):
        s = AgentState()
        b = Ball(color=Color.red)
        s.carrying = b
        assert s.carrying is b

    def test_clear_carrying(self):
        s = AgentState()
        s.carrying = Ball()
        s.carrying = None
        assert s.carrying is None

    def test_vectorized_operations(self):
        s = AgentState(3)
        s[0].pos = (1, 2)
        s[1].pos = (3, 4)
        s[2].pos = (5, 6)
        # Access all positions
        positions = s.pos
        assert positions.shape == (3, 2)
        assert tuple(positions[0]) == (1, 2)


# ---------------------------------------------------------------
# Agent
# ---------------------------------------------------------------

class TestAgent:
    def test_default_team_index_equals_index(self):
        a = Agent(index=3)
        assert a.team_index == 3

    def test_custom_team_index(self):
        a = Agent(index=0, team_index=2)
        assert a.index == 0
        assert a.team_index == 2

    def test_reset_sets_color_from_team(self):
        a0 = Agent(index=0, team_index=0)
        a1 = Agent(index=1, team_index=0)
        a0.reset()
        a1.reset()
        # Same team → same color
        assert a0.color == a1.color

    def test_reset_different_teams_different_colors(self):
        a0 = Agent(index=0, team_index=0)
        a1 = Agent(index=1, team_index=1)
        a0.reset()
        a1.reset()
        assert a0.color != a1.color

    def test_front_pos(self):
        a = Agent(index=0)
        a.reset()
        a.state.pos = (5, 5)
        a.state.dir = Direction.right
        assert tuple(a.front_pos) == (6, 5)

    def test_observation_space(self):
        a = Agent(index=0, view_size=7)
        obs_space = a.observation_space
        assert 'image' in obs_space.spaces
        assert 'direction' in obs_space.spaces
        assert 'mission' in obs_space.spaces

    def test_action_space(self):
        a = Agent(index=0)
        assert a.action_space.n == len(Action)

    def test_encode(self):
        a = Agent(index=0)
        a.reset()
        a.state.dir = Direction.right
        enc = a.encode()
        assert len(enc) == 3
        assert enc[0] == Type.agent.to_index()


# ---------------------------------------------------------------
# Mission / MissionSpace
# ---------------------------------------------------------------

class TestMission:
    def test_string_representation(self):
        m = Mission('maximize reward')
        assert str(m) == 'maximize reward'

    def test_equality(self):
        m1 = Mission('maximize reward')
        m2 = Mission('maximize reward')
        assert m1 == m2

    def test_hash(self):
        m = Mission('maximize reward')
        assert hash(m) == hash('maximize reward')


class TestMissionSpace:
    def test_from_string(self):
        space = MissionSpace.from_string('get the ball')
        m = space.sample()
        assert str(m) == 'get the ball'

    def test_parameterized(self):
        space = MissionSpace(
            mission_func=lambda color: f'get the {color} ball',
            ordered_placeholders=[['red', 'blue']],
        )
        m = space.sample()
        assert 'ball' in str(m)

    def test_contains(self):
        space = MissionSpace.from_string('hello')
        assert space.contains('hello')
        assert not space.contains('goodbye')
