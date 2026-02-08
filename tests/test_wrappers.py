"""Tests for MOSAIC multigrid observation wrappers.

Covers: FullyObsWrapper, ImgObsWrapper, OneHotObsWrapper, SingleAgentWrapper.
"""
import numpy as np
import pytest

from mosaic_multigrid.envs import SoccerGame4HEnv10x15N2
from mosaic_multigrid.wrappers import (
    FullyObsWrapper,
    ImgObsWrapper,
    OneHotObsWrapper,
    SingleAgentWrapper,
)
from mosaic_multigrid.core import Action, WorldObj


# ---------------------------------------------------------------
# FullyObsWrapper
# ---------------------------------------------------------------

class TestFullyObsWrapper:
    def test_observation_shape_is_full_grid(self):
        env = SoccerGame4HEnv10x15N2()
        env = FullyObsWrapper(env)
        obs, _ = env.reset(seed=42)

        # Soccer env is 15x10, observations should be (15, 10, 3)
        assert obs[0]['image'].shape == (15, 10, WorldObj.dim)

    def test_all_agents_see_same_grid(self):
        env = SoccerGame4HEnv10x15N2()
        env = FullyObsWrapper(env)
        obs, _ = env.reset(seed=42)

        # All agents see the same full grid
        for i in range(1, 4):
            assert np.array_equal(obs[0]['image'], obs[i]['image'])

    def test_direction_and_mission_preserved(self):
        env = SoccerGame4HEnv10x15N2()
        env = FullyObsWrapper(env)
        obs, _ = env.reset(seed=42)

        assert 'direction' in obs[0]
        assert 'mission' in obs[0]


# ---------------------------------------------------------------
# ImgObsWrapper
# ---------------------------------------------------------------

class TestImgObsWrapper:
    def test_removes_direction_and_mission(self):
        env = SoccerGame4HEnv10x15N2()
        env = ImgObsWrapper(env)
        obs, _ = env.reset(seed=42)

        # Observation should be just the image array
        assert isinstance(obs[0], np.ndarray)
        assert obs[0].ndim == 3
        assert obs[0].shape[2] == WorldObj.dim

    def test_observation_is_uint8(self):
        env = SoccerGame4HEnv10x15N2()
        env = ImgObsWrapper(env)
        obs, _ = env.reset(seed=42)
        assert obs[0].dtype == np.uint8


# ---------------------------------------------------------------
# OneHotObsWrapper
# ---------------------------------------------------------------

class TestOneHotObsWrapper:
    def test_one_hot_shape(self):
        env = SoccerGame4HEnv10x15N2()
        env = OneHotObsWrapper(env)
        obs, _ = env.reset(seed=42)

        # Default view_size=3, should be (3, 3, sum(dim_sizes))
        # dim_sizes = [len(Type), len(Color), max(len(State), len(Direction))]
        from mosaic_multigrid.core import Type, Color, State, Direction
        expected_depth = len(Type) + len(Color) + max(len(State), len(Direction))

        assert obs[0]['image'].shape[2] == expected_depth

    def test_one_hot_is_float32(self):
        env = SoccerGame4HEnv10x15N2()
        env = OneHotObsWrapper(env)
        obs, _ = env.reset(seed=42)
        assert obs[0]['image'].dtype == np.float32

    def test_one_hot_values_binary(self):
        env = SoccerGame4HEnv10x15N2()
        env = OneHotObsWrapper(env)
        obs, _ = env.reset(seed=42)

        # All values should be 0 or 1
        unique = np.unique(obs[0]['image'])
        assert set(unique).issubset({0.0, 1.0})

    def test_direction_and_mission_preserved(self):
        env = SoccerGame4HEnv10x15N2()
        env = OneHotObsWrapper(env)
        obs, _ = env.reset(seed=42)

        assert 'direction' in obs[0]
        assert 'mission' in obs[0]


# ---------------------------------------------------------------
# SingleAgentWrapper
# ---------------------------------------------------------------

class TestSingleAgentWrapper:
    def test_unwraps_observations(self):
        env = SoccerGame4HEnv10x15N2()
        env = SingleAgentWrapper(env)
        obs, info = env.reset(seed=42)

        # Should return obs for agent 0 directly (not dict)
        assert isinstance(obs, dict)
        assert 'image' in obs
        assert 0 not in obs  # not dict-keyed by agent ID

    def test_unwraps_step_returns(self):
        env = SoccerGame4HEnv10x15N2()
        env = SingleAgentWrapper(env)
        env.reset(seed=42)

        # Action is scalar, not dict
        obs, reward, terminated, truncated, info = env.step(Action.forward)

        assert isinstance(obs, dict)  # still has image/direction/mission
        assert isinstance(reward, (int, float))  # scalar
        assert isinstance(terminated, (bool, np.bool_))  # scalar (numpy or python bool)
        assert isinstance(truncated, (bool, np.bool_))  # scalar

    def test_observation_space_unwrapped(self):
        env = SoccerGame4HEnv10x15N2()
        env = SingleAgentWrapper(env)

        # Should be agent 0's observation space (not dict of spaces)
        assert 'image' in env.observation_space.spaces

    def test_action_space_unwrapped(self):
        env = SoccerGame4HEnv10x15N2()
        env = SingleAgentWrapper(env)

        # Should be scalar Discrete space
        assert env.action_space.n == len(Action)


# ---------------------------------------------------------------
# Wrapper Composition
# ---------------------------------------------------------------

class TestWrapperComposition:
    def test_fully_obs_plus_img(self):
        """FullyObsWrapper + ImgObsWrapper."""
        env = SoccerGame4HEnv10x15N2()
        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        obs, _ = env.reset(seed=42)

        # Should be full grid (15, 10, 3) as uint8 array
        assert obs[0].shape == (15, 10, WorldObj.dim)
        assert obs[0].dtype == np.uint8

    def test_img_plus_one_hot(self):
        """ImgObsWrapper + OneHotObsWrapper won't work (OneHot needs dict)."""
        # This is expected to fail or produce unexpected results
        # because OneHotObsWrapper expects dict obs with 'image' key
        pass  # Document limitation

    def test_single_agent_at_end(self):
        """SingleAgentWrapper should be outermost."""
        env = SoccerGame4HEnv10x15N2()
        env = ImgObsWrapper(env)
        env = SingleAgentWrapper(env)
        obs, _ = env.reset(seed=42)

        # Final obs should be just the image for agent 0
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 3
