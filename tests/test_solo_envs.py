"""Tests for Solo environments (Soccer and Basketball).

Verifies that:
1. Solo Green and Blue environments can be created, reset, and stepped
2. Solo environments have exactly 1 agent (no opponent)
3. view_size can be overridden at make time
4. Observation space dimensions match view_size parameter
5. Field coverage analysis for view_size=3 vs view_size=7
6. Episode rollouts complete successfully
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import mosaic_multigrid.envs
from mosaic_multigrid.envs import (
    SoccerSoloGreenIndAgObsEnv16x11,
    SoccerSoloBlueIndAgObsEnv16x11,
    BasketballSoloGreenIndAgObsEnv19x11,
    BasketballSoloBlueIndAgObsEnv19x11,
)

SEED = 42


@pytest.fixture
def soccer_solo_green_v3():
    env = gym.make('MosaicMultiGrid-Soccer-Solo-Green-IndAgObs-v0', view_size=3)
    yield env
    env.close()


@pytest.fixture
def soccer_solo_green_v7():
    env = gym.make('MosaicMultiGrid-Soccer-Solo-Green-IndAgObs-v0', view_size=7)
    yield env
    env.close()


@pytest.fixture
def soccer_solo_blue_v3():
    env = gym.make('MosaicMultiGrid-Soccer-Solo-Blue-IndAgObs-v0', view_size=3)
    yield env
    env.close()


@pytest.fixture
def soccer_solo_blue_v7():
    env = gym.make('MosaicMultiGrid-Soccer-Solo-Blue-IndAgObs-v0', view_size=7)
    yield env
    env.close()


class TestSoccerSoloCreation:
    """Environment construction and basic API tests."""

    def test_green_creation_v3(self, soccer_solo_green_v3):
        assert soccer_solo_green_v3 is not None

    def test_green_creation_v7(self, soccer_solo_green_v7):
        assert soccer_solo_green_v7 is not None

    def test_blue_creation_v3(self, soccer_solo_blue_v3):
        assert soccer_solo_blue_v3 is not None

    def test_blue_creation_v7(self, soccer_solo_blue_v7):
        assert soccer_solo_blue_v7 is not None

    def test_single_agent_green(self, soccer_solo_green_v3):
        obs, _ = soccer_solo_green_v3.reset(seed=SEED)
        assert len(obs) == 1
        assert 0 in obs

    def test_single_agent_blue(self, soccer_solo_blue_v3):
        obs, _ = soccer_solo_blue_v3.reset(seed=SEED)
        assert len(obs) == 1
        assert 0 in obs

    def test_grid_dimensions_green(self, soccer_solo_green_v3):
        soccer_solo_green_v3.reset(seed=SEED)
        inner = soccer_solo_green_v3.unwrapped
        assert inner.width == 16
        assert inner.height == 11

    def test_grid_dimensions_blue(self, soccer_solo_blue_v3):
        soccer_solo_blue_v3.reset(seed=SEED)
        inner = soccer_solo_blue_v3.unwrapped
        assert inner.width == 16
        assert inner.height == 11

    def test_team_assignment_green(self, soccer_solo_green_v3):
        soccer_solo_green_v3.reset(seed=SEED)
        inner = soccer_solo_green_v3.unwrapped
        assert len(inner.agents) == 1
        assert inner.agents[0].team_index == 1

    def test_team_assignment_blue(self, soccer_solo_blue_v3):
        soccer_solo_blue_v3.reset(seed=SEED)
        inner = soccer_solo_blue_v3.unwrapped
        assert len(inner.agents) == 1
        assert inner.agents[0].team_index == 2

    def test_goals_at_correct_positions(self, soccer_solo_green_v3):
        soccer_solo_green_v3.reset(seed=SEED)
        inner = soccer_solo_green_v3.unwrapped
        assert inner.goal_pos == [[1, 5], [14, 5]]
        assert inner.goal_index == [1, 2]


class TestViewSizeOverride:
    """Test that view_size can be overridden at make time."""

    def test_view_size_3_green(self, soccer_solo_green_v3):
        obs, _ = soccer_solo_green_v3.reset(seed=SEED)
        assert obs[0]['image'].shape == (3, 3, 3)

    def test_view_size_7_green(self, soccer_solo_green_v7):
        obs, _ = soccer_solo_green_v7.reset(seed=SEED)
        assert obs[0]['image'].shape == (7, 7, 3)

    def test_view_size_3_blue(self, soccer_solo_blue_v3):
        obs, _ = soccer_solo_blue_v3.reset(seed=SEED)
        assert obs[0]['image'].shape == (3, 3, 3)

    def test_view_size_7_blue(self, soccer_solo_blue_v7):
        obs, _ = soccer_solo_blue_v7.reset(seed=SEED)
        assert obs[0]['image'].shape == (7, 7, 3)

    def test_agent_view_size_attribute_v3(self, soccer_solo_green_v3):
        soccer_solo_green_v3.reset(seed=SEED)
        inner = soccer_solo_green_v3.unwrapped
        assert inner.agents[0].view_size == 3

    def test_agent_view_size_attribute_v7(self, soccer_solo_green_v7):
        soccer_solo_green_v7.reset(seed=SEED)
        inner = soccer_solo_green_v7.unwrapped
        assert inner.agents[0].view_size == 7


class TestFieldCoverageAnalysis:
    """Analyze field coverage for different view sizes."""

    def test_coverage_view_size_3(self, soccer_solo_green_v3):
        obs, _ = soccer_solo_green_v3.reset(seed=SEED)
        inner = soccer_solo_green_v3.unwrapped

        view_cells = 3 * 3
        playable_area = (inner.width - 2) * (inner.height - 2)
        coverage = (view_cells / playable_area) * 100

        assert view_cells == 9
        assert playable_area == 14 * 9  # 126 cells
        assert abs(coverage - 7.14) < 0.1  # Approximately 7.14%

    def test_coverage_view_size_7(self, soccer_solo_green_v7):
        obs, _ = soccer_solo_green_v7.reset(seed=SEED)
        inner = soccer_solo_green_v7.unwrapped

        view_cells = 7 * 7
        playable_area = (inner.width - 2) * (inner.height - 2)
        coverage = (view_cells / playable_area) * 100

        assert view_cells == 49
        assert playable_area == 14 * 9  # 126 cells
        assert abs(coverage - 38.89) < 0.1  # Approximately 38.89%

    def test_coverage_increase_ratio(self):
        view_3_cells = 3 * 3
        view_7_cells = 7 * 7
        ratio = view_7_cells / view_3_cells

        assert abs(ratio - 5.44) < 0.1  # 5.44x more cells visible


class TestObservationStructure:
    """Test observation dictionary structure."""

    def test_observation_keys_v3(self, soccer_solo_green_v3):
        obs, _ = soccer_solo_green_v3.reset(seed=SEED)
        expected_keys = {'image', 'direction', 'mission'}
        assert set(obs[0].keys()) == expected_keys

    def test_observation_keys_v7(self, soccer_solo_green_v7):
        obs, _ = soccer_solo_green_v7.reset(seed=SEED)
        expected_keys = {'image', 'direction', 'mission'}
        assert set(obs[0].keys()) == expected_keys

    def test_image_dtype(self, soccer_solo_green_v3):
        obs, _ = soccer_solo_green_v3.reset(seed=SEED)
        assert obs[0]['image'].dtype == np.int64

    def test_direction_range(self, soccer_solo_green_v3):
        obs, _ = soccer_solo_green_v3.reset(seed=SEED)
        assert 0 <= obs[0]['direction'] <= 3

    def test_mission_string(self, soccer_solo_green_v3):
        obs, _ = soccer_solo_green_v3.reset(seed=SEED)
        assert str(obs[0]['mission']) == 'maximize reward'


class TestEpisodeRollout:
    """Test episode rollouts complete successfully."""

    def test_rollout_green_v3(self, soccer_solo_green_v3):
        obs, _ = soccer_solo_green_v3.reset(seed=SEED)

        for _ in range(50):
            action = soccer_solo_green_v3.action_space[0].sample()
            obs, rewards, terminated, truncated, info = soccer_solo_green_v3.step({0: action})

            if terminated[0] or truncated[0]:
                break

        assert True  # Rollout completed without errors

    def test_rollout_green_v7(self, soccer_solo_green_v7):
        obs, _ = soccer_solo_green_v7.reset(seed=SEED)

        for _ in range(50):
            action = soccer_solo_green_v7.action_space[0].sample()
            obs, rewards, terminated, truncated, info = soccer_solo_green_v7.step({0: action})

            if terminated[0] or truncated[0]:
                break

        assert True  # Rollout completed without errors

    def test_rollout_blue_v3(self, soccer_solo_blue_v3):
        obs, _ = soccer_solo_blue_v3.reset(seed=SEED)

        for _ in range(50):
            action = soccer_solo_blue_v3.action_space[0].sample()
            obs, rewards, terminated, truncated, info = soccer_solo_blue_v3.step({0: action})

            if terminated[0] or truncated[0]:
                break

        assert True  # Rollout completed without errors

    def test_rollout_blue_v7(self, soccer_solo_blue_v7):
        obs, _ = soccer_solo_blue_v7.reset(seed=SEED)

        for _ in range(50):
            action = soccer_solo_blue_v7.action_space[0].sample()
            obs, rewards, terminated, truncated, info = soccer_solo_blue_v7.step({0: action})

            if terminated[0] or truncated[0]:
                break

        assert True  # Rollout completed without errors


class TestRewardStructure:
    """Test reward structure in solo environments."""

    def test_initial_reward_zero(self, soccer_solo_green_v3):
        obs, _ = soccer_solo_green_v3.reset(seed=SEED)
        action = 0  # noop
        obs, rewards, terminated, truncated, info = soccer_solo_green_v3.step({0: action})

        assert rewards[0] == 0  # No reward for noop

    def test_reward_dict_structure(self, soccer_solo_green_v3):
        obs, _ = soccer_solo_green_v3.reset(seed=SEED)
        action = 0
        obs, rewards, terminated, truncated, info = soccer_solo_green_v3.step({0: action})

        assert isinstance(rewards, dict)
        assert 0 in rewards


class TestActionSpace:
    """Test action space structure."""

    def test_action_space_discrete(self, soccer_solo_green_v3):
        assert hasattr(soccer_solo_green_v3.action_space[0], 'n')
        assert soccer_solo_green_v3.action_space[0].n == 8

    def test_action_space_dict_like(self, soccer_solo_green_v3):
        from gymnasium.spaces import Dict
        assert isinstance(soccer_solo_green_v3.action_space, Dict)
        assert 0 in soccer_solo_green_v3.action_space.spaces


class TestSeeding:
    """Test that seeding produces reproducible results."""

    def test_reproducible_reset_green(self):
        env1 = gym.make('MosaicMultiGrid-Soccer-Solo-Green-IndAgObs-v0', view_size=3)
        env2 = gym.make('MosaicMultiGrid-Soccer-Solo-Green-IndAgObs-v0', view_size=3)

        obs1, _ = env1.reset(seed=SEED)
        obs2, _ = env2.reset(seed=SEED)

        np.testing.assert_array_equal(obs1[0]['image'], obs2[0]['image'])
        assert obs1[0]['direction'] == obs2[0]['direction']

        env1.close()
        env2.close()

    def test_reproducible_reset_blue(self):
        env1 = gym.make('MosaicMultiGrid-Soccer-Solo-Blue-IndAgObs-v0', view_size=3)
        env2 = gym.make('MosaicMultiGrid-Soccer-Solo-Blue-IndAgObs-v0', view_size=3)

        obs1, _ = env1.reset(seed=SEED)
        obs2, _ = env2.reset(seed=SEED)

        np.testing.assert_array_equal(obs1[0]['image'], obs2[0]['image'])
        assert obs1[0]['direction'] == obs2[0]['direction']

        env1.close()
        env2.close()


class TestBasketballSolo:
    """Basic tests for Basketball solo environments."""

    def test_basketball_solo_green_creation(self):
        env = gym.make('MosaicMultiGrid-Basketball-Solo-Green-IndAgObs-v0', view_size=3)
        obs, _ = env.reset(seed=SEED)
        assert len(obs) == 1
        assert obs[0]['image'].shape == (3, 3, 3)
        env.close()

    def test_basketball_solo_blue_creation(self):
        env = gym.make('MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v0', view_size=3)
        obs, _ = env.reset(seed=SEED)
        assert len(obs) == 1
        assert obs[0]['image'].shape == (3, 3, 3)
        env.close()

    def test_basketball_solo_green_view_size_7(self):
        env = gym.make('MosaicMultiGrid-Basketball-Solo-Green-IndAgObs-v0', view_size=7)
        obs, _ = env.reset(seed=SEED)
        assert obs[0]['image'].shape == (7, 7, 3)
        env.close()

    def test_basketball_solo_blue_view_size_7(self):
        env = gym.make('MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v0', view_size=7)
        obs, _ = env.reset(seed=SEED)
        assert obs[0]['image'].shape == (7, 7, 3)
        env.close()
