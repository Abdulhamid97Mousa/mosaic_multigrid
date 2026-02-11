"""Tests for PettingZoo Parallel API integration.

Validates that mosaic_multigrid environments work correctly with
the PettingZoo ParallelEnv API (simultaneous stepping).

All agents submit actions at the same time via a dict, and
receive observations after the environment processes all actions.
"""
from __future__ import annotations

import numpy as np
import pytest


try:
    from pettingzoo import ParallelEnv
    from pettingzoo.test import parallel_api_test
    HAS_PETTINGZOO = True
except ImportError:
    HAS_PETTINGZOO = False

try:
    from mosaic_multigrid.envs import SoccerGame4HEnv10x15N2, CollectGame4HEnv10x10N2
    from mosaic_multigrid.pettingzoo import PettingZooWrapper, to_pettingzoo_env
    HAS_MOSAIC = True
except ImportError:
    HAS_MOSAIC = False


pytestmark = pytest.mark.skipif(
    not (HAS_PETTINGZOO and HAS_MOSAIC),
    reason="pettingzoo and mosaic_multigrid required",
)


# -----------------------------------------------------------------------
# Soccer (4 agents: 2v2)
# -----------------------------------------------------------------------

class TestSoccerParallel:
    """Parallel API tests for Soccer environment."""

    @pytest.fixture
    def env(self):
        PZCls = to_pettingzoo_env(SoccerGame4HEnv10x15N2)
        env = PZCls(render_mode="rgb_array")
        yield env
        env.close()

    def test_is_parallel_env(self, env):
        """Wrapper should be an instance of PettingZoo ParallelEnv."""
        assert isinstance(env, ParallelEnv)

    def test_reset_returns_obs_and_info(self, env):
        """reset() should return (obs_dict, info_dict) keyed by agent ID."""
        obs, infos = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert isinstance(infos, dict)
        assert len(obs) == 4  # Soccer has 4 agents

    def test_agents_list(self, env):
        """possible_agents and agents should list all 4 agent IDs."""
        env.reset(seed=42)
        assert len(env.possible_agents) == 4
        assert len(env.agents) == 4
        assert env.possible_agents == env.agents

    def test_observation_spaces(self, env):
        """Each agent should have a valid observation space."""
        env.reset(seed=42)
        for agent in env.agents:
            space = env.observation_space(agent)
            assert space is not None
            assert hasattr(space, "shape")

    def test_action_spaces(self, env):
        """Each agent should have a valid action space."""
        env.reset(seed=42)
        for agent in env.agents:
            space = env.action_space(agent)
            assert space is not None

    def test_step_with_random_actions(self, env):
        """step() should accept a dict of actions and return 5-tuple of dicts."""
        env.reset(seed=42)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terms, dict)
        assert isinstance(truncs, dict)
        assert isinstance(infos, dict)

    def test_observations_are_dicts_with_image(self, env):
        """Observations should be dicts containing 'image' (numpy array), 'direction', 'mission'."""
        obs, _ = env.reset(seed=42)
        for agent_id, ob in obs.items():
            assert isinstance(ob, dict), f"Agent {agent_id} obs is {type(ob)}"
            assert "image" in ob, f"Agent {agent_id} obs missing 'image' key"
            assert isinstance(ob["image"], np.ndarray), f"Agent {agent_id} image is {type(ob['image'])}"

    def test_multiple_steps(self, env):
        """Run 10 steps without error."""
        env.reset(seed=42)
        for _ in range(10):
            if not env.agents:
                break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)

    def test_state_returns_grid(self, env):
        """state() should return the full grid encoding."""
        env.reset(seed=42)
        state = env.state()
        assert isinstance(state, np.ndarray)
        assert state.ndim == 3  # (height, width, channels)

    def test_render_returns_rgb(self, env):
        """render() should return an RGB array."""
        env.reset(seed=42)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB

    def test_metadata_is_parallelizable(self, env):
        """Metadata should indicate the env is parallelizable."""
        assert env.metadata.get("is_parallelizable") is True


# -----------------------------------------------------------------------
# Collect (3 agents: cooperative)
# -----------------------------------------------------------------------

class TestCollectParallel:
    """Parallel API tests for Collect environment."""

    @pytest.fixture
    def env(self):
        PZCls = to_pettingzoo_env(CollectGame4HEnv10x10N2)
        env = PZCls(render_mode="rgb_array")
        yield env
        env.close()

    def test_is_parallel_env(self, env):
        assert isinstance(env, ParallelEnv)

    def test_reset_and_step(self, env):
        """Basic reset + step cycle should work."""
        obs, infos = env.reset(seed=42)
        assert len(obs) > 0

        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)

    def test_multiple_steps(self, env):
        """Run 20 steps without error."""
        env.reset(seed=42)
        for _ in range(20):
            if not env.agents:
                break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)


# -----------------------------------------------------------------------
# PettingZoo official API test (comprehensive)
# -----------------------------------------------------------------------

class TestPettingZooOfficialAPITest:
    """Run PettingZoo's official parallel_api_test validator."""

    def test_soccer_passes_api_test(self):
        """Soccer should pass PettingZoo's parallel_api_test."""
        PZCls = to_pettingzoo_env(SoccerGame4HEnv10x15N2)
        env = PZCls(render_mode="rgb_array")
        try:
            parallel_api_test(env, num_cycles=10)
        finally:
            env.close()

    def test_collect_passes_api_test(self):
        """Collect should pass PettingZoo's parallel_api_test."""
        PZCls = to_pettingzoo_env(CollectGame4HEnv10x10N2)
        env = PZCls(render_mode="rgb_array")
        try:
            parallel_api_test(env, num_cycles=10)
        finally:
            env.close()
