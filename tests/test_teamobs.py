"""Tests for TeamObs environments and TeamObsWrapper.

Validates the SMAC-style teammate awareness observation augmentation
(Samvelyan et al., 2019) for team-based MOSAIC multigrid environments.

TeamObsWrapper adds three structured features per agent:
  - teammate_positions  (N, 2) int64 -- relative (dx, dy) to teammates
  - teammate_directions (N,)   int64 -- direction each teammate faces
  - teammate_has_ball   (N,)   int64 -- 1 if teammate carries the ball

The original image/direction/mission keys are preserved unchanged.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from mosaic_multigrid.envs import (
    SoccerGame4HEnhancedEnv16x11N2,
    CollectGame4HEnhancedEnv10x10N2,
    CollectGame3HEnhancedEnv10x10N3,
    SoccerTeamObsEnv,
    Collect2vs2TeamObsEnv,
)
from mosaic_multigrid.wrappers import TeamObsWrapper
from mosaic_multigrid.core.constants import Type


# -----------------------------------------------------------------------
# TeamObsWrapper unit tests (Soccer)
# -----------------------------------------------------------------------

class TestTeamObsWrapperSoccer:
    """TeamObsWrapper applied to Soccer 2v2 Enhanced."""

    @pytest.fixture
    def env(self):
        base = SoccerGame4HEnhancedEnv16x11N2(render_mode="rgb_array")
        env = TeamObsWrapper(base)
        yield env
        env.close()

    def test_observation_keys(self, env):
        """Each agent observation should have 6 keys: original 3 + 3 teammate features."""
        obs, _ = env.reset(seed=42)
        expected_keys = {
            "image", "direction", "mission",
            "teammate_positions", "teammate_directions", "teammate_has_ball",
        }
        for agent_id, agent_obs in obs.items():
            assert set(agent_obs.keys()) == expected_keys, (
                f"Agent {agent_id} keys: {set(agent_obs.keys())}"
            )

    def test_image_shape_preserved(self, env):
        """TeamObs should preserve the original 3x3x3 image shape."""
        obs, _ = env.reset(seed=42)
        for agent_id, agent_obs in obs.items():
            img = agent_obs["image"]
            assert img.shape == (3, 3, 3), (
                f"Agent {agent_id} image shape: {img.shape}"
            )
            assert isinstance(img, np.ndarray)

    def test_teammate_count_soccer_2v2(self, env):
        """Soccer 2v2: each agent has exactly 1 teammate."""
        obs, _ = env.reset(seed=42)
        for agent_id, agent_obs in obs.items():
            assert agent_obs["teammate_positions"].shape == (1, 2), (
                f"Agent {agent_id} teammate_positions shape: "
                f"{agent_obs['teammate_positions'].shape}"
            )
            assert agent_obs["teammate_directions"].shape == (1,)
            assert agent_obs["teammate_has_ball"].shape == (1,)

    def test_positions_are_antisymmetric(self, env):
        """Agent 0 -> Agent 1 relative pos should be negation of Agent 1 -> Agent 0."""
        obs, _ = env.reset(seed=42)
        # Agents 0 and 1 are on same team (team_index=1 in Soccer)
        pos_0_to_1 = obs[0]["teammate_positions"][0]
        pos_1_to_0 = obs[1]["teammate_positions"][0]
        np.testing.assert_array_equal(
            pos_0_to_1, -pos_1_to_0,
            err_msg="Relative positions should be antisymmetric",
        )

    def test_directions_valid_range(self, env):
        """Teammate directions should be in [0, 3]."""
        obs, _ = env.reset(seed=42)
        for agent_id, agent_obs in obs.items():
            dirs = agent_obs["teammate_directions"]
            assert np.all(dirs >= 0) and np.all(dirs <= 3), (
                f"Agent {agent_id} teammate_directions out of range: {dirs}"
            )

    def test_has_ball_initially_zero(self, env):
        """At reset, no agent carries the ball, so has_ball should be all zeros."""
        obs, _ = env.reset(seed=42)
        for agent_id, agent_obs in obs.items():
            np.testing.assert_array_equal(
                agent_obs["teammate_has_ball"],
                np.zeros(1, dtype=np.int64),
                err_msg=f"Agent {agent_id} teammate_has_ball should be 0 at reset",
            )

    def test_observation_space_matches_observation(self, env):
        """Observation space should match the actual observation structure."""
        obs, _ = env.reset(seed=42)
        for agent_id, agent_obs in obs.items():
            space = env.observation_space[agent_id]
            assert "teammate_positions" in space.spaces
            assert "teammate_directions" in space.spaces
            assert "teammate_has_ball" in space.spaces
            # Check shape matches
            assert space["teammate_positions"].shape == agent_obs["teammate_positions"].shape
            assert space["teammate_directions"].shape == agent_obs["teammate_directions"].shape
            assert space["teammate_has_ball"].shape == agent_obs["teammate_has_ball"].shape

    def test_step_preserves_teammate_features(self, env):
        """Teammate features should be present after stepping."""
        obs, _ = env.reset(seed=42)
        actions = {a: env.action_space[a].sample() for a in range(4)}
        obs, _, _, _, _ = env.step(actions)
        for agent_id, agent_obs in obs.items():
            assert "teammate_positions" in agent_obs
            assert "teammate_directions" in agent_obs
            assert "teammate_has_ball" in agent_obs

    def test_team_mapping_soccer_2v2(self, env):
        """Soccer 2v2 team mapping: {0:[1], 1:[0], 2:[3], 3:[2]}."""
        expected = {0: [1], 1: [0], 2: [3], 3: [2]}
        assert env._team_map == expected, f"Team map: {env._team_map}"

    def test_run_50_steps(self, env):
        """Run 50 steps without error to validate stability."""
        env.reset(seed=42)
        for _ in range(50):
            actions = {a: env.action_space[a].sample() for a in range(4)}
            obs, rewards, terms, truncs, infos = env.step(actions)
            # Verify structure on every step
            for agent_id in obs:
                assert obs[agent_id]["teammate_positions"].shape == (1, 2)


# -----------------------------------------------------------------------
# TeamObsWrapper unit tests (Collect 2v2)
# -----------------------------------------------------------------------

class TestTeamObsWrapperCollect:
    """TeamObsWrapper applied to Collect 2v2 Enhanced."""

    @pytest.fixture
    def env(self):
        base = CollectGame4HEnhancedEnv10x10N2(render_mode="rgb_array")
        env = TeamObsWrapper(base)
        yield env
        env.close()

    def test_observation_keys(self, env):
        obs, _ = env.reset(seed=42)
        expected_keys = {
            "image", "direction", "mission",
            "teammate_positions", "teammate_directions", "teammate_has_ball",
        }
        for agent_id, agent_obs in obs.items():
            assert set(agent_obs.keys()) == expected_keys

    def test_teammate_count_collect_2v2(self, env):
        """Collect 2v2: each agent has exactly 1 teammate."""
        obs, _ = env.reset(seed=42)
        for agent_id, agent_obs in obs.items():
            assert agent_obs["teammate_positions"].shape == (1, 2)

    def test_team_mapping_collect_2v2(self, env):
        """Collect 2v2: agents_index=[1,1,2,2] -> same team map as Soccer."""
        expected = {0: [1], 1: [0], 2: [3], 3: [2]}
        assert env._team_map == expected


# -----------------------------------------------------------------------
# TeamObs on individual-team envs (edge case)
# -----------------------------------------------------------------------

class TestTeamObsNoTeammates:
    """TeamObsWrapper on a 3-agent Collect where each agent is its own team."""

    def test_empty_teammate_arrays(self):
        """With agents_index=[1,2,3], each agent has 0 teammates."""
        base = CollectGame3HEnhancedEnv10x10N3(render_mode="rgb_array")
        env = TeamObsWrapper(base)
        obs, _ = env.reset(seed=42)

        for agent_id, agent_obs in obs.items():
            assert agent_obs["teammate_positions"].shape == (0, 2), (
                f"Agent {agent_id}: expected (0, 2), got "
                f"{agent_obs['teammate_positions'].shape}"
            )
            assert agent_obs["teammate_directions"].shape == (0,)
            assert agent_obs["teammate_has_ball"].shape == (0,)
        env.close()


# -----------------------------------------------------------------------
# Registered environment tests (gym.make)
# -----------------------------------------------------------------------

class TestRegisteredTeamObsEnvs:
    """Test TeamObs environments via gymnasium.make()."""

    @pytest.mark.parametrize("env_id,n_agents", [
        ("MosaicMultiGrid-Soccer-TeamObs-v0", 4),
        ("MosaicMultiGrid-Collect-2vs2-TeamObs-v0", 4),
    ])
    def test_gym_make(self, env_id, n_agents):
        """Environment should be creatable via gym.make()."""
        env = gym.make(env_id, render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        assert len(obs) == n_agents
        for agent_obs in obs.values():
            assert "teammate_positions" in agent_obs
        env.close()

    def test_soccer_teamobs_is_wrapper_subclass(self):
        """SoccerTeamObsEnv should be a TeamObsWrapper subclass."""
        assert issubclass(SoccerTeamObsEnv, TeamObsWrapper)

    def test_collect_teamobs_is_wrapper_subclass(self):
        """Collect2vs2TeamObsEnv should be a TeamObsWrapper subclass."""
        assert issubclass(Collect2vs2TeamObsEnv, TeamObsWrapper)

    def test_soccer_teamobs_render(self):
        """Soccer TeamObs should render RGB frames."""
        env = gym.make("MosaicMultiGrid-Soccer-TeamObs-v0", render_mode="rgb_array")
        env.reset(seed=42)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        env.close()


# -----------------------------------------------------------------------
# Wrapper composition tests
# -----------------------------------------------------------------------

class TestTeamObsComposition:
    """Test TeamObsWrapper with other wrappers."""

    def test_fully_obs_plus_teamobs(self):
        """FullyObsWrapper + TeamObsWrapper should work."""
        from mosaic_multigrid.wrappers import FullyObsWrapper
        base = SoccerGame4HEnhancedEnv16x11N2(render_mode="rgb_array")
        env = FullyObsWrapper(base)
        env = TeamObsWrapper(env)
        obs, _ = env.reset(seed=42)

        # Image should be full grid (16, 11, 3)
        assert obs[0]["image"].shape == (16, 11, 3)
        # Teammate features should be present
        assert obs[0]["teammate_positions"].shape == (1, 2)
        env.close()

    def test_teamobs_plus_single_agent(self):
        """TeamObsWrapper + SingleAgentWrapper should unwrap correctly."""
        from mosaic_multigrid.wrappers import SingleAgentWrapper
        base = SoccerGame4HEnhancedEnv16x11N2(render_mode="rgb_array")
        env = TeamObsWrapper(base)
        env = SingleAgentWrapper(env)
        obs, _ = env.reset(seed=42)

        # Should be agent 0's observation (not dict-keyed by agent)
        assert isinstance(obs, dict)
        assert "teammate_positions" in obs
        assert "image" in obs
        assert obs["teammate_positions"].shape == (1, 2)
        env.close()


# -----------------------------------------------------------------------
# PettingZoo integration with TeamObs
# -----------------------------------------------------------------------

try:
    from pettingzoo import ParallelEnv
    from pettingzoo.test import parallel_api_test
    from mosaic_multigrid.pettingzoo import to_pettingzoo_env
    HAS_PETTINGZOO = True
except ImportError:
    HAS_PETTINGZOO = False


@pytest.mark.skipif(not HAS_PETTINGZOO, reason="pettingzoo required")
class TestTeamObsPettingZoo:
    """Test TeamObs envs wrapped with PettingZoo API."""

    def test_soccer_teamobs_pettingzoo_parallel(self):
        """Soccer TeamObs should work as PettingZoo ParallelEnv."""
        PZCls = to_pettingzoo_env(SoccerGame4HEnhancedEnv16x11N2, TeamObsWrapper)
        env = PZCls(render_mode="rgb_array")
        obs, infos = env.reset(seed=42)

        assert isinstance(env, ParallelEnv)
        assert len(env.agents) == 4

        # Observations should have teammate features
        for agent in env.agents:
            ob = obs[agent]
            assert "teammate_positions" in ob
            assert "teammate_has_ball" in ob
        env.close()

    def test_soccer_teamobs_passes_parallel_api_test(self):
        """Soccer TeamObs should pass PettingZoo's parallel_api_test."""
        PZCls = to_pettingzoo_env(SoccerGame4HEnhancedEnv16x11N2, TeamObsWrapper)
        env = PZCls(render_mode="rgb_array")
        try:
            parallel_api_test(env, num_cycles=5)
        finally:
            env.close()
