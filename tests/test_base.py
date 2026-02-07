"""Tests for the MOSAIC multigrid base environment.

Covers: MultiGridEnv construction, reset returns 2-tuple dicts, step returns
5-tuple dicts, observation shapes, reproducibility (seeded reset), Numba JIT
obs generation, rendering (rgb_array mode).
"""
import numpy as np
import pytest

from gym_multigrid.base import MultiGridEnv
from gym_multigrid.core import (
    Action,
    Agent,
    Ball,
    Color,
    Direction,
    Grid,
    TILE_PIXELS,
    Type,
    WorldObj,
)
from gym_multigrid.utils.obs import gen_obs_grid_encoding


# ---------------------------------------------------------------
# Minimal concrete environment for testing
# ---------------------------------------------------------------

class SimpleTestEnv(MultiGridEnv):
    """Minimal environment: walled 8x8 grid, 2 agents at fixed positions."""

    def __init__(self, num_agents=2, **kwargs):
        agents = [
            Agent(index=i, team_index=i % 2)
            for i in range(num_agents)
        ]
        super().__init__(
            agents=agents,
            grid_size=8,
            max_steps=50,
            render_mode='rgb_array',
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Place agents at fixed positions
        positions = [(2, 2), (5, 5), (2, 5), (5, 2)]
        for i, agent in enumerate(self.agents):
            agent.state.pos = positions[i % len(positions)]
            agent.state.dir = Direction.right


class BallTestEnv(MultiGridEnv):
    """Environment with a ball for testing pickup/drop."""

    def __init__(self, **kwargs):
        agents = [Agent(index=0, team_index=0)]
        super().__init__(
            agents=agents,
            grid_size=8,
            max_steps=50,
            render_mode='rgb_array',
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Agent facing right at (3, 3), ball at (4, 3) directly in front
        self.agents[0].state.pos = (3, 3)
        self.agents[0].state.dir = Direction.right
        self.ball = Ball(color=Color.red, index=0, reward=1.0)
        self.grid.set(4, 3, self.ball)


# ---------------------------------------------------------------
# Construction
# ---------------------------------------------------------------

class TestConstruction:
    def test_simple_env_creation(self):
        env = SimpleTestEnv()
        assert env.num_agents == 2
        assert env.width == 8
        assert env.height == 8

    def test_agents_have_unique_indices(self):
        env = SimpleTestEnv(num_agents=4)
        indices = [a.index for a in env.agents]
        assert indices == [0, 1, 2, 3]

    def test_agents_have_team_indices(self):
        env = SimpleTestEnv(num_agents=4)
        teams = [a.team_index for a in env.agents]
        assert teams == [0, 1, 0, 1]

    def test_observation_space_is_dict(self):
        env = SimpleTestEnv()
        obs_space = env.observation_space
        assert 0 in obs_space.spaces
        assert 1 in obs_space.spaces

    def test_action_space_is_dict(self):
        env = SimpleTestEnv()
        act_space = env.action_space
        assert 0 in act_space.spaces
        assert act_space[0].n == 7  # 7 actions


# ---------------------------------------------------------------
# Reset
# ---------------------------------------------------------------

class TestReset:
    def test_reset_returns_2_tuple(self):
        env = SimpleTestEnv()
        result = env.reset(seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_observations_are_dict(self):
        env = SimpleTestEnv()
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert 0 in obs
        assert 1 in obs

    def test_reset_obs_has_image_direction_mission(self):
        env = SimpleTestEnv()
        obs, _ = env.reset(seed=42)
        assert 'image' in obs[0]
        assert 'direction' in obs[0]
        assert 'mission' in obs[0]

    def test_reset_image_shape(self):
        env = SimpleTestEnv()
        obs, _ = env.reset(seed=42)
        # Default view_size=7, WorldObj.dim=3
        assert obs[0]['image'].shape == (7, 7, 3)

    def test_reset_clears_step_count(self):
        env = SimpleTestEnv()
        env.reset(seed=42)
        env.step({0: Action.forward, 1: Action.forward})
        env.reset(seed=42)
        assert env.step_count == 0


# ---------------------------------------------------------------
# Step
# ---------------------------------------------------------------

class TestStep:
    def test_step_returns_5_tuple(self):
        env = SimpleTestEnv()
        env.reset(seed=42)
        result = env.step({0: Action.forward, 1: Action.left})
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_observations_are_dict(self):
        env = SimpleTestEnv()
        env.reset(seed=42)
        obs, rewards, terminated, truncated, info = env.step(
            {0: Action.forward, 1: Action.left})
        assert isinstance(obs, dict)
        assert 0 in obs and 1 in obs

    def test_step_rewards_are_dict(self):
        env = SimpleTestEnv()
        env.reset(seed=42)
        _, rewards, _, _, _ = env.step(
            {0: Action.forward, 1: Action.left})
        assert isinstance(rewards, dict)
        assert 0 in rewards and 1 in rewards

    def test_step_terminated_are_dict(self):
        env = SimpleTestEnv()
        env.reset(seed=42)
        _, _, terminated, _, _ = env.step(
            {0: Action.forward, 1: Action.left})
        assert isinstance(terminated, dict)
        assert 0 in terminated and 1 in terminated

    def test_step_truncated_are_dict(self):
        env = SimpleTestEnv()
        env.reset(seed=42)
        _, _, _, truncated, _ = env.step(
            {0: Action.forward, 1: Action.left})
        assert isinstance(truncated, dict)

    def test_step_increments_count(self):
        env = SimpleTestEnv()
        env.reset(seed=42)
        env.step({0: Action.done, 1: Action.done})
        assert env.step_count == 1

    def test_truncation_at_max_steps(self):
        env = SimpleTestEnv()
        env.reset(seed=42)
        for _ in range(50):
            _, _, _, truncated, _ = env.step(
                {0: Action.done, 1: Action.done})
        assert truncated[0] is True
        assert truncated[1] is True

    def test_rotation_left(self):
        env = SimpleTestEnv(num_agents=1)
        env.reset(seed=42)
        initial_dir = env.agents[0].state.dir
        env.step({0: Action.left})
        new_dir = env.agents[0].state.dir
        assert new_dir == (initial_dir - 1) % 4

    def test_rotation_right(self):
        env = SimpleTestEnv(num_agents=1)
        env.reset(seed=42)
        initial_dir = env.agents[0].state.dir
        env.step({0: Action.right})
        new_dir = env.agents[0].state.dir
        assert new_dir == (initial_dir + 1) % 4

    def test_forward_movement(self):
        env = SimpleTestEnv(num_agents=1)
        env.reset(seed=42)
        # Agent starts at (2,2) facing right
        initial_pos = tuple(env.agents[0].state.pos)
        env.step({0: Action.forward})
        new_pos = tuple(env.agents[0].state.pos)
        assert new_pos == (initial_pos[0] + 1, initial_pos[1])

    def test_wall_blocks_movement(self):
        env = SimpleTestEnv(num_agents=1)
        env.reset(seed=42)
        # Agent at (2,2) facing right → move toward wall at x=7
        for _ in range(10):
            env.step({0: Action.forward})
        pos_x = env.agents[0].state.pos[0]
        assert pos_x < 7  # can't pass through wall


# ---------------------------------------------------------------
# Pickup / Drop
# ---------------------------------------------------------------

class TestPickupDrop:
    def test_pickup_ball(self):
        env = BallTestEnv()
        env.reset(seed=42)
        # Ball is at (4,3), agent at (3,3) facing right
        assert env.agents[0].state.carrying is None
        env.step({0: Action.pickup})
        assert env.agents[0].state.carrying is not None
        assert env.agents[0].state.carrying.type == Type.ball

    def test_pickup_removes_from_grid(self):
        env = BallTestEnv()
        env.reset(seed=42)
        env.step({0: Action.pickup})
        cell = env.grid.get(4, 3)
        assert cell is None

    def test_drop_places_on_grid(self):
        env = BallTestEnv()
        env.reset(seed=42)
        env.step({0: Action.pickup})  # pick up ball
        env.step({0: Action.drop})    # drop in front (now empty)
        assert env.agents[0].state.carrying is None
        cell = env.grid.get(4, 3)
        assert cell is not None
        assert cell.type == Type.ball


# ---------------------------------------------------------------
# Numba JIT observation generation
# ---------------------------------------------------------------

class TestNumbaObs:
    def test_obs_grid_shape(self):
        env = SimpleTestEnv()
        env.reset(seed=42)
        obs = gen_obs_grid_encoding(
            env.grid.state,
            env.agent_states,
            env.agents[0].view_size,
            env.agents[0].see_through_walls,
        )
        assert obs.shape == (2, 7, 7, 3)

    def test_obs_contains_agent_at_bottom_center(self):
        """Agent should see itself at (view_size//2, view_size-1)."""
        env = SimpleTestEnv(num_agents=1)
        env.reset(seed=42)
        obs = gen_obs_grid_encoding(
            env.grid.state,
            env.agent_states,
            env.agents[0].view_size,
            env.agents[0].see_through_walls,
        )
        # Bottom center is where agent's carrying is shown
        # The cell there should not be 'unseen' (type index 0)
        center = obs[0, 3, 6]  # view_size//2=3, view_size-1=6
        assert center[0] != Type.unseen.to_index()


# ---------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------

class TestRendering:
    def test_rgb_array_returns_ndarray(self):
        env = SimpleTestEnv()
        env.reset(seed=42)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB
        assert frame.dtype == np.uint8

    def test_frame_dimensions(self):
        env = SimpleTestEnv()
        env.reset(seed=42)
        frame = env.render()
        # 8x8 grid, 32px tiles → 256x256
        assert frame.shape[0] == 8 * TILE_PIXELS
        assert frame.shape[1] == 8 * TILE_PIXELS


# ---------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_observations(self):
        """Same seed → identical trajectories."""
        results = []
        for _ in range(2):
            env = SimpleTestEnv()
            obs, _ = env.reset(seed=42)
            for step in range(10):
                actions = {i: Action.forward for i in range(2)}
                obs, *_ = env.step(actions)
            results.append(obs[0]['image'].copy())
            env.close()

        assert np.array_equal(results[0], results[1])

    def test_different_seeds_different_behavior(self):
        """Different seeds should produce different random sequences."""
        # This test uses place_agent which is random
        from gym_multigrid.base import MultiGridEnv
        from gym_multigrid.core import Grid, Agent, Direction

        class RandomPlaceEnv(MultiGridEnv):
            def __init__(self):
                agents = [Agent(index=0)]
                super().__init__(agents=agents, grid_size=10, max_steps=10,
                                 render_mode='rgb_array')

            def _gen_grid(self, width, height):
                self.grid = Grid(width, height)
                self.grid.wall_rect(0, 0, width, height)
                self.place_agent(self.agents[0])

        env1 = RandomPlaceEnv()
        env1.reset(seed=1)
        pos1 = tuple(env1.agents[0].state.pos)

        env2 = RandomPlaceEnv()
        env2.reset(seed=999)
        pos2 = tuple(env2.agents[0].state.pos)

        # With very different seeds on a 10x10 grid, positions should differ
        # (vanishingly unlikely to collide)
        assert pos1 != pos2
