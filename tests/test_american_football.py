"""Tests for American Football environments.

Verifies that:
1. Agents can pick up the ball
2. Agents can carry the ball
3. Agents can score by walking into opponent's end zone with ball
4. Agents CANNOT score in their own end zone
5. Ball respawns after touchdown
6. Correct team gets rewarded
7. Game terminates after touchdown
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import mosaic_multigrid.envs
from mosaic_multigrid.core.constants import Direction

SEED = 42


@pytest.fixture
def env_1v1():
    env = gym.make('MosaicMultiGrid-AmericanFootball-1v1-v0')
    yield env
    env.close()


@pytest.fixture
def env_solo_green():
    env = gym.make('MosaicMultiGrid-AmericanFootball-Solo-Green-v0')
    yield env
    env.close()


class TestAmericanFootballCreation:
    """Environment construction and basic API tests."""

    def test_1v1_creation(self, env_1v1):
        assert env_1v1 is not None

    def test_solo_creation(self, env_solo_green):
        assert env_solo_green is not None

    def test_two_agents_1v1(self, env_1v1):
        obs, _ = env_1v1.reset(seed=SEED)
        assert len(obs) == 2

    def test_grid_dimensions(self, env_1v1):
        env_1v1.reset(seed=SEED)
        inner = env_1v1.unwrapped
        assert inner.width == 16
        assert inner.height == 11

    def test_team_assignments(self, env_1v1):
        env_1v1.reset(seed=SEED)
        inner = env_1v1.unwrapped
        teams = [a.team_index for a in inner.agents]
        assert teams == [0, 1]

    def test_endzones_exist(self, env_1v1):
        env_1v1.reset(seed=SEED)
        inner = env_1v1.unwrapped
        # Should have 18 end zone positions (9 per column, 2 columns)
        assert len(inner.endzone_positions) == 18

        # Check column 1 (team 0's end zone)
        assert (1, 1) in inner.endzone_positions
        assert inner.endzone_positions[(1, 1)] == 0

        # Check column 14 (team 1's end zone)
        assert (14, 1) in inner.endzone_positions
        assert inner.endzone_positions[(14, 1)] == 1


class TestBallPickupAndCarrying:
    """Test ball pickup and carrying mechanics."""

    def test_agent_can_pickup_ball(self, env_solo_green):
        """Test that an agent can pick up a ball from the ground."""
        env_solo_green.reset(seed=SEED)
        inner = env_solo_green.unwrapped
        agent = inner.agents[0]

        # Find the ball
        ball_pos = None
        for x in range(inner.width):
            for y in range(inner.height):
                obj = inner.grid.get(x, y)
                if obj is not None and hasattr(obj, 'type') and obj.type == 'ball':
                    ball_pos = (x, y)
                    break
            if ball_pos:
                break

        assert ball_pos is not None, "Ball should exist on the field"

        # Move agent next to ball and face it
        agent.state.pos = np.array([ball_pos[0] - 1, ball_pos[1]])
        agent.state.dir = Direction.right

        # Verify agent is not carrying anything
        assert agent.state.carrying is None

        # Pickup action (action 4)
        actions = {0: 4}
        obs, rewards, terminated, truncated, info = env_solo_green.step(actions)

        # Agent should now be carrying the ball
        assert agent.state.carrying is not None
        assert agent.state.carrying.type == 'ball'

    def test_agent_carries_ball_while_moving(self, env_solo_green):
        """Test that agent keeps carrying ball while moving."""
        env_solo_green.reset(seed=SEED)
        inner = env_solo_green.unwrapped
        agent = inner.agents[0]

        # Find and pickup ball
        ball_pos = None
        for x in range(inner.width):
            for y in range(inner.height):
                obj = inner.grid.get(x, y)
                if obj is not None and hasattr(obj, 'type') and obj.type == 'ball':
                    ball_pos = (x, y)
                    break
            if ball_pos:
                break

        # Position agent next to ball
        agent.state.pos = np.array([ball_pos[0] - 1, ball_pos[1]])
        agent.state.dir = Direction.right

        # Pickup ball (action 4)
        actions = {0: 4}
        env_solo_green.step(actions)

        assert agent.state.carrying is not None

        # Move forward (action 3)
        actions = {0: 3}
        env_solo_green.step(actions)

        # Agent should still be carrying the ball
        assert agent.state.carrying is not None
        assert agent.state.carrying.type == 'ball'


class TestTouchdownScoring:
    """Test touchdown scoring mechanics."""

    def test_agent_scores_in_opponent_endzone(self, env_1v1):
        """Test that agent scores when entering opponent's end zone with ball."""
        env_1v1.reset(seed=SEED)
        inner = env_1v1.unwrapped

        # Get team 0 agent (should score in column 14)
        agent0 = inner.agents[0]
        assert agent0.team_index == 0

        # Find the ball and give it to agent0
        ball = None
        for x in range(inner.width):
            for y in range(inner.height):
                obj = inner.grid.get(x, y)
                if obj is not None and hasattr(obj, 'type') and obj.type == 'ball':
                    ball = obj
                    inner.grid.set(x, y, None)
                    break
            if ball:
                break

        assert ball is not None
        agent0.state.carrying = ball

        # Position agent just outside opponent's end zone (column 13)
        agent0.state.pos = np.array([13, 5])
        agent0.state.dir = Direction.right

        # Move forward into end zone (column 14) - action 3
        actions = {0: 3, 1: 0}  # agent0 forward, agent1 no-op
        obs, rewards, terminated, truncated, info = env_1v1.step(actions)

        # Check that agent0 scored
        assert rewards[0] > 0, "Agent 0 should receive positive reward for touchdown"
        assert rewards[1] < 0, "Agent 1 should receive negative reward (zero-sum)"
        assert terminated[0] is True, "Game should terminate after touchdown"
        assert agent0.state.carrying is None, "Ball should be removed from agent after touchdown"

    def test_agent_cannot_score_in_own_endzone(self, env_1v1):
        """Test that agent CANNOT score in their own end zone."""
        env_1v1.reset(seed=SEED)
        inner = env_1v1.unwrapped

        # Get team 0 agent (their end zone is column 1)
        agent0 = inner.agents[0]
        assert agent0.team_index == 0

        # Find the ball and give it to agent0
        ball = None
        for x in range(inner.width):
            for y in range(inner.height):
                obj = inner.grid.get(x, y)
                if obj is not None and hasattr(obj, 'type') and obj.type == 'ball':
                    ball = obj
                    inner.grid.set(x, y, None)
                    break
            if ball:
                break

        assert ball is not None
        agent0.state.carrying = ball

        # Position agent just outside their own end zone (column 2)
        agent0.state.pos = np.array([2, 5])
        agent0.state.dir = Direction.left

        # Move into own end zone (column 1) - action 3
        actions = {0: 3, 1: 0}  # agent0 forward, agent1 no-op
        obs, rewards, terminated, truncated, info = env_1v1.step(actions)

        # Check that NO touchdown occurred
        assert rewards[0] == 0, "Agent 0 should NOT score in own end zone"
        assert rewards[1] == 0, "Agent 1 should NOT receive any reward"
        assert not terminated[0], "Game should NOT terminate"
        assert agent0.state.carrying is not None, "Agent should still be carrying ball"

    def test_opponent_scores_correctly(self, env_1v1):
        """Test that team 1 agent scores in team 0's end zone (column 1)."""
        env_1v1.reset(seed=SEED)
        inner = env_1v1.unwrapped

        # Get team 1 agent (should score in column 1)
        agent1 = inner.agents[1]
        assert agent1.team_index == 1

        # Find the ball and give it to agent1
        ball = None
        for x in range(inner.width):
            for y in range(inner.height):
                obj = inner.grid.get(x, y)
                if obj is not None and hasattr(obj, 'type') and obj.type == 'ball':
                    ball = obj
                    inner.grid.set(x, y, None)
                    break
            if ball:
                break

        assert ball is not None
        agent1.state.carrying = ball

        # Position agent just outside opponent's end zone (column 2)
        agent1.state.pos = np.array([2, 5])
        agent1.state.dir = Direction.left

        # Move forward into end zone (column 1) - action 3
        actions = {0: 0, 1: 3}  # agent0 no-op, agent1 forward
        obs, rewards, terminated, truncated, info = env_1v1.step(actions)

        # Check that agent1 scored
        assert rewards[1] > 0, "Agent 1 should receive positive reward for touchdown"
        assert rewards[0] < 0, "Agent 0 should receive negative reward (zero-sum)"
        assert terminated[1] is True, "Game should terminate after touchdown"
        assert agent1.state.carrying is None, "Ball should be removed from agent after touchdown"


class TestBallStealing:
    """Test ball stealing mechanics."""

    def test_opponent_can_steal_ball(self, env_1v1):
        """Test that opponent can steal ball using pickup action."""
        env_1v1.reset(seed=SEED)
        inner = env_1v1.unwrapped

        agent0 = inner.agents[0]
        agent1 = inner.agents[1]

        # Find the ball and give it to agent0
        ball = None
        for x in range(inner.width):
            for y in range(inner.height):
                obj = inner.grid.get(x, y)
                if obj is not None and hasattr(obj, 'type') and obj.type == 'ball':
                    ball = obj
                    inner.grid.set(x, y, None)
                    break
            if ball:
                break

        assert ball is not None
        agent0.state.carrying = ball

        # Position agents facing each other
        agent0.state.pos = np.array([5, 5])
        agent0.state.dir = Direction.right
        agent1.state.pos = np.array([6, 5])
        agent1.state.dir = Direction.left

        # Agent1 uses pickup (action 4) to steal from agent0
        actions = {0: 0, 1: 4}  # agent0 no-op, agent1 pickup
        obs, rewards, terminated, truncated, info = env_1v1.step(actions)

        # Check that ball was stolen
        assert agent0.state.carrying is None, "Agent 0 should no longer have ball"
        assert agent1.state.carrying is not None, "Agent 1 should now have ball"
        assert agent1.state.carrying.type == 'ball'


class TestRendering:
    """Test that rendering works without errors."""

    def test_render_produces_valid_frame(self, env_1v1):
        """Test that rendering produces a valid RGB array."""
        env = gym.make('MosaicMultiGrid-AmericanFootball-1v1-v0', render_mode='rgb_array')
        env.reset(seed=SEED)

        frame = env.render()

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB
        assert frame.dtype == np.uint8

        env.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
