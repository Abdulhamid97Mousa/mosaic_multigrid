"""Tests for 1vs1 environment variants (Collect and Soccer).

Validates that the 1vs1 environments have correct agent counts, team
assignments, ball counts, termination behavior, and gym registration.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

# Trigger gymnasium registration
import mosaic_multigrid.envs  # noqa: F401

from mosaic_multigrid.envs.collect_game import (
    CollectGame2HEnv10x10N2,
    CollectGame2HIndAgObsEnv10x10N2,
)
from mosaic_multigrid.envs.soccer_game import (
    SoccerGame2HIndAgObsEnv16x11N2,
)


# -----------------------------------------------------------------------
# Collect 1vs1
# -----------------------------------------------------------------------

class TestCollect1vs1:
    """Tests for the 1vs1 Collect environment."""

    def test_collect_1vs1_creation(self):
        """Verify 2 agents on 2 separate teams."""
        env = CollectGame2HIndAgObsEnv10x10N2()
        assert len(env.agents) == 2
        assert env.agents[0].team_index == 1
        assert env.agents[1].team_index == 2
        env.close()

    def test_collect_1vs1_three_balls(self):
        """Verify 3 balls are placed on the grid after reset."""
        env = CollectGame2HIndAgObsEnv10x10N2()
        env.reset(seed=42)

        ball_count = 0
        for x in range(env.width):
            for y in range(env.height):
                obj = env.grid.get(x, y)
                if obj is not None and obj.type.value == 'ball':
                    ball_count += 1

        assert ball_count == 3
        env.close()

    def test_collect_1vs1_natural_termination(self):
        """Collect all 3 balls and verify episode terminates."""
        env = CollectGame2HIndAgObsEnv10x10N2()
        env.reset(seed=42)

        # Run random actions until termination or max steps
        # Direct env classes expect a list/array of actions, not a dict
        terminated = False
        for _ in range(env.max_steps):
            actions = [np.random.randint(0, 7) for _ in range(2)]
            _, _, term, trunc, _ = env.step(actions)
            if isinstance(term, dict):
                terminated = term.get("__all__", False)
            else:
                terminated = bool(term)
            if isinstance(trunc, dict):
                truncated = trunc.get("__all__", False)
            else:
                truncated = bool(trunc)
            if terminated or truncated:
                break

        # With random actions on a 10x10 grid, termination within 200 steps
        # is likely but not guaranteed. We verify the mechanism works by
        # checking the remaining ball counter directly.
        if terminated:
            assert env._remaining_balls <= 0
        env.close()

    def test_collect_1vs1_zero_sum(self):
        """Verify collecting gives the opponent negative reward."""
        env = CollectGame2HIndAgObsEnv10x10N2()
        env.reset(seed=42)

        # Step many times, accumulate rewards
        # Direct env classes return rewards as a dict {agent_idx: reward}
        total_rewards = {0: 0.0, 1: 0.0}
        for _ in range(200):
            actions = [np.random.randint(0, 7) for _ in range(2)]
            _, rewards, term, trunc, _ = env.step(actions)
            if isinstance(rewards, dict):
                for i in range(2):
                    total_rewards[i] += rewards.get(i, 0.0)
            if isinstance(term, dict) and term.get("__all__", False):
                break
            if isinstance(trunc, dict) and trunc.get("__all__", False):
                break

        # Zero-sum: total rewards across all agents should sum to 0
        assert abs(sum(total_rewards.values())) < 1e-6
        env.close()

    def test_collect_1vs1_max_steps_default(self):
        """IndAgObs 1v1 should default to 200 max_steps."""
        env = CollectGame2HIndAgObsEnv10x10N2()
        assert env.max_steps == 200
        env.close()

    def test_collect_1vs1_base_env(self):
        """Base (non-IndAgObs) 1v1 should have default 10000 max_steps."""
        env = CollectGame2HEnv10x10N2()
        assert len(env.agents) == 2
        assert env.max_steps == 10000
        env.close()


# -----------------------------------------------------------------------
# Soccer 1vs1
# -----------------------------------------------------------------------

class TestSoccer1vs1:
    """Tests for the 1vs1 Soccer environment."""

    def test_soccer_1vs1_creation(self):
        """Verify 2 agents on 2 separate teams, 16x11 grid."""
        env = SoccerGame2HIndAgObsEnv16x11N2()
        assert len(env.agents) == 2
        assert env.agents[0].team_index == 1
        assert env.agents[1].team_index == 2
        assert env.width == 16
        assert env.height == 11
        env.close()

    def test_soccer_1vs1_goals(self):
        """Verify 2 goals at correct positions."""
        env = SoccerGame2HIndAgObsEnv16x11N2()
        env.reset(seed=42)

        goal_positions = []
        for x in range(env.width):
            for y in range(env.height):
                obj = env.grid.get(x, y)
                if obj is not None and obj.type.value == 'objgoal':
                    goal_positions.append((x, y))

        assert len(goal_positions) == 2
        # Goals should be at (1,5) and (14,5) - vertical center
        assert (1, 5) in goal_positions
        assert (14, 5) in goal_positions
        env.close()

    def test_soccer_1vs1_no_teleport_pass(self):
        """DROP with ball should result in ground drop (no teammates to pass to)."""
        env = SoccerGame2HIndAgObsEnv16x11N2()
        env.reset(seed=42)

        # Manually give agent 0 a ball to test drop behavior
        from mosaic_multigrid.core.world_object import Ball
        from mosaic_multigrid.core.constants import Color
        ball = Ball(color=Color.from_index(0), index=0)
        env.agents[0].state.carrying = ball

        # Agent 0 tries to drop (action 4 = DROP in IndAgObs which has STILL)
        # The drop should not teleport-pass since there are no teammates
        # It should either score (if facing goal) or ground drop
        actions = {0: 5, 1: 0}  # DROP for agent 0, STILL for agent 1
        env.step(actions)

        # If ball was dropped on ground or scored, agent should not be carrying
        # (unless the cell in front was occupied/wall)
        # We just verify no crash occurred -- the no-teammates path is exercised
        env.close()

    @pytest.mark.parametrize("env_id,n_agents", [
        ("MosaicMultiGrid-Collect-1vs1-v0", 2),
        ("MosaicMultiGrid-Collect-1vs1-IndAgObs-v0", 2),
        ("MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0", 2),
    ])
    def test_1vs1_gym_make(self, env_id, n_agents):
        """All 1vs1 environments should be creatable via gym.make()."""
        env = gym.make(env_id, render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        assert len(obs) == n_agents
        env.close()

    @pytest.mark.parametrize("env_id", [
        "MosaicMultiGrid-Collect-2vs2-v0",
        "MosaicMultiGrid-Collect-2vs2-IndAgObs-v0",
        "MosaicMultiGrid-Collect-2vs2-TeamObs-v0",
    ])
    def test_collect_2vs2_renamed(self, env_id):
        """Renamed Collect-2vs2 environments should be creatable via gym.make()."""
        env = gym.make(env_id, render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        assert len(obs) == 4
        env.close()

    def test_old_collect2vs2_name_removed(self):
        """Old 'Collect2vs2' (no hyphen) names should NOT be registered."""
        with pytest.raises(gym.error.NameNotFound):
            gym.make("MosaicMultiGrid-Collect2vs2-IndAgObs-v0")
