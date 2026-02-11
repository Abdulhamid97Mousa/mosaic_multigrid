"""Tests for PettingZoo AEC (Agent Environment Cycle) API integration.

Validates that mosaic_multigrid environments work correctly with
the PettingZoo AEC API (sequential turn-based stepping).

In AEC mode, agents take turns acting one at a time via agent_iter().
The environment processes each action individually, and agents receive
updated observations before their next turn.

Internally, this uses PettingZoo's parallel_to_aec converter to wrap
the native ParallelEnv.
"""
from __future__ import annotations

import numpy as np
import pytest


try:
    from pettingzoo.utils.env import AECEnv
    from pettingzoo.test import api_test as aec_api_test
    HAS_PETTINGZOO = True
except ImportError:
    HAS_PETTINGZOO = False

try:
    from mosaic_multigrid.envs import SoccerGame4HEnv10x15N2, CollectGame4HEnv10x10N2
    from mosaic_multigrid.pettingzoo import to_pettingzoo_aec_env
    HAS_MOSAIC = True
except ImportError:
    HAS_MOSAIC = False


pytestmark = pytest.mark.skipif(
    not (HAS_PETTINGZOO and HAS_MOSAIC),
    reason="pettingzoo and mosaic_multigrid required",
)


# -----------------------------------------------------------------------
# Soccer AEC (4 agents: 2v2, turn-based stepping)
# -----------------------------------------------------------------------

class TestSoccerAEC:
    """AEC API tests for Soccer environment."""

    @pytest.fixture
    def env(self):
        AecFactory = to_pettingzoo_aec_env(SoccerGame4HEnv10x15N2)
        env = AecFactory(render_mode="rgb_array")
        yield env
        env.close()

    def test_is_aec_env(self, env):
        """Wrapper should be an instance of PettingZoo AECEnv."""
        assert isinstance(env, AECEnv)

    def test_has_agent_iter(self, env):
        """AEC env should have agent_iter() method."""
        assert hasattr(env, "agent_iter")
        assert callable(env.agent_iter)

    def test_has_last(self, env):
        """AEC env should have last() method."""
        assert hasattr(env, "last")
        assert callable(env.last)

    def test_has_agent_selection(self, env):
        """AEC env should have agent_selection attribute after reset."""
        env.reset(seed=42)
        assert hasattr(env, "agent_selection")
        assert env.agent_selection in env.agents

    def test_reset(self, env):
        """reset() should initialize agents and agent_selection."""
        env.reset(seed=42)
        assert len(env.agents) == 4  # Soccer has 4 agents
        assert len(env.possible_agents) == 4

    def test_observation_spaces(self, env):
        """Each agent should have a valid observation space."""
        env.reset(seed=42)
        for agent in env.possible_agents:
            space = env.observation_space(agent)
            assert space is not None
            assert hasattr(space, "shape")

    def test_action_spaces(self, env):
        """Each agent should have a valid action space."""
        env.reset(seed=42)
        for agent in env.possible_agents:
            space = env.action_space(agent)
            assert space is not None

    def test_last_returns_5_tuple(self, env):
        """last() should return (obs, reward, termination, truncation, info)."""
        env.reset(seed=42)
        result = env.last()
        assert len(result) == 5
        obs, reward, term, trunc, info = result
        # Observations are dicts with 'image', 'direction', 'mission' keys
        assert isinstance(obs, dict), f"obs is {type(obs)}, expected dict"
        assert "image" in obs
        assert isinstance(obs["image"], np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

    def test_agent_iter_cycles_through_agents(self, env):
        """agent_iter should cycle through all agents."""
        env.reset(seed=42)
        seen_agents = set()
        for i, agent in enumerate(env.agent_iter()):
            obs, reward, term, trunc, info = env.last()
            action = None if term or trunc else env.action_space(agent).sample()
            env.step(action)
            seen_agents.add(agent)
            if i >= 7:  # 2 full cycles of 4 agents
                break

        # Should have seen all 4 agents
        assert len(seen_agents) == 4

    def test_sequential_stepping_10_cycles(self, env):
        """Run 10 full cycles of sequential stepping without error."""
        env.reset(seed=42)
        step_count = 0
        max_steps = 4 * 10  # 4 agents x 10 cycles

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            action = None if term or trunc else env.action_space(agent).sample()
            env.step(action)
            step_count += 1
            if step_count >= max_steps:
                break

        assert step_count == max_steps

    def test_render_returns_rgb(self, env):
        """render() should return an RGB array."""
        env.reset(seed=42)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB


# -----------------------------------------------------------------------
# Collect AEC (cooperative, turn-based stepping)
# -----------------------------------------------------------------------

class TestCollectAEC:
    """AEC API tests for Collect environment."""

    @pytest.fixture
    def env(self):
        AecFactory = to_pettingzoo_aec_env(CollectGame4HEnv10x10N2)
        env = AecFactory(render_mode="rgb_array")
        yield env
        env.close()

    def test_is_aec_env(self, env):
        assert isinstance(env, AECEnv)

    def test_reset_and_sequential_step(self, env):
        """Basic AEC cycle: reset, iterate, step."""
        env.reset(seed=42)
        assert len(env.agents) > 0

        step_count = 0
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            action = None if term or trunc else env.action_space(agent).sample()
            env.step(action)
            step_count += 1
            if step_count >= 20:
                break

        assert step_count == 20

    def test_agent_selection_changes(self, env):
        """agent_selection should change after each step."""
        env.reset(seed=42)
        prev_agent = env.agent_selection

        obs, reward, term, trunc, info = env.last()
        action = env.action_space(prev_agent).sample()
        env.step(action)

        # After stepping, agent_selection should be a different agent
        # (unless the game ended)
        if env.agents:
            assert env.agent_selection != prev_agent


# -----------------------------------------------------------------------
# PettingZoo official AEC API test (comprehensive)
# -----------------------------------------------------------------------

class TestPettingZooOfficialAECAPITest:
    """Run PettingZoo's official aec_api_test validator."""

    def test_soccer_passes_aec_api_test(self):
        """Soccer should pass PettingZoo's aec_api_test."""
        AecFactory = to_pettingzoo_aec_env(SoccerGame4HEnv10x15N2)
        env = AecFactory(render_mode="rgb_array")
        try:
            aec_api_test(env, num_cycles=10)
        finally:
            env.close()

    def test_collect_passes_aec_api_test(self):
        """Collect should pass PettingZoo's aec_api_test."""
        AecFactory = to_pettingzoo_aec_env(CollectGame4HEnv10x10N2)
        env = AecFactory(render_mode="rgb_array")
        try:
            aec_api_test(env, num_cycles=10)
        finally:
            env.close()
