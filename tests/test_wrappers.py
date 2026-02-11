"""Tests for MOSAIC multigrid observation wrappers.

Covers: FullyObsWrapper, ImgObsWrapper, OneHotObsWrapper, SingleAgentWrapper,
        TeamObsWrapper, and wrapper composition.
"""
import numpy as np
import pytest

from mosaic_multigrid.envs import SoccerGame4HEnv10x15N2
from mosaic_multigrid.wrappers import (
    FullyObsWrapper,
    ImgObsWrapper,
    OneHotObsWrapper,
    SingleAgentWrapper,
    TeamObsWrapper,
    _one_hot,
    _DIM_SIZES,
    _ONE_HOT_DIM,
)
from mosaic_multigrid.core import Action, WorldObj
from mosaic_multigrid.core.constants import Type, Color, Direction


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

        # dim_sizes = [len(Type), len(Color), len(Direction)] + 1 carrying bit
        expected_depth = len(Type) + len(Color) + len(Direction) + 1  # 24

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

    def test_teamobs_then_onehot_preserves_keys(self):
        """OneHotObsWrapper must pass through TeamObs extra keys."""
        env = SoccerGame4HEnv10x15N2()
        env = TeamObsWrapper(env)
        env = OneHotObsWrapper(env)
        obs, _ = env.reset(seed=42)

        assert 'teammate_positions' in obs[0]
        assert 'teammate_directions' in obs[0]
        assert 'teammate_has_ball' in obs[0]
        assert obs[0]['image'].dtype == np.float32
        assert obs[0]['image'].shape[2] == _ONE_HOT_DIM

    def test_teamobs_then_onehot_obs_space_has_all_keys(self):
        """OneHotObsWrapper observation_space must include TeamObs keys."""
        env = SoccerGame4HEnv10x15N2()
        env = TeamObsWrapper(env)
        env = OneHotObsWrapper(env)

        space = env.observation_space[0]
        assert 'teammate_positions' in space.spaces
        assert 'teammate_directions' in space.spaces
        assert 'teammate_has_ball' in space.spaces
        assert space['image'].shape[2] == _ONE_HOT_DIM


# ---------------------------------------------------------------
# OneHot: Ball-Carrying Encoding (Option B)
# ---------------------------------------------------------------

class TestOneHotBallCarrying:
    """Verify that the factored one-hot encoding correctly handles
    Agent.encode() sentinel values (100-103) for ball-carrying agents."""

    def test_carrying_agent_sets_carry_bit(self):
        """STATE=102 (carrying ball, facing left) should set carry=1."""
        # Type.agent=10, Color.green=1, STATE=102 (left + carrying)
        image = np.array([[[10, 1, 102]]], dtype=np.uint8)
        result = _one_hot(image, _DIM_SIZES)

        assert result.shape == (1, 1, _ONE_HOT_DIM)

        # TYPE: index 10 (agent) should be 1
        assert result[0, 0, 10] == 1.0

        # COLOR: offset=13, index 1 (green) -> position 14
        assert result[0, 0, 14] == 1.0

        # DIRECTION: offset=19, index 2 (left) -> position 21
        assert result[0, 0, 21] == 1.0

        # CARRYING bit: offset=23 -> position 23
        assert result[0, 0, 23] == 1.0

    def test_non_carrying_agent_clear_carry_bit(self):
        """STATE=2 (no ball, facing left) should set carry=0."""
        image = np.array([[[10, 1, 2]]], dtype=np.uint8)
        result = _one_hot(image, _DIM_SIZES)

        # DIRECTION: left (index 2) should be set
        assert result[0, 0, 21] == 1.0

        # CARRYING bit should be 0
        assert result[0, 0, 23] == 0.0

    def test_all_four_carrying_directions(self):
        """Verify STATE 100, 101, 102, 103 each encode correctly."""
        for direction_val in range(4):
            state_val = 100 + direction_val
            image = np.array([[[10, 0, state_val]]], dtype=np.uint8)
            result = _one_hot(image, _DIM_SIZES)

            # Exactly one direction bit set
            dir_slice = result[0, 0, 19:23]
            assert dir_slice.sum() == 1.0
            assert dir_slice[direction_val] == 1.0

            # Carrying bit set
            assert result[0, 0, 23] == 1.0

    def test_all_four_non_carrying_directions(self):
        """Verify STATE 0, 1, 2, 3 each encode correctly without carry."""
        for direction_val in range(4):
            image = np.array([[[10, 0, direction_val]]], dtype=np.uint8)
            result = _one_hot(image, _DIM_SIZES)

            dir_slice = result[0, 0, 19:23]
            assert dir_slice.sum() == 1.0
            assert dir_slice[direction_val] == 1.0

            assert result[0, 0, 23] == 0.0

    def test_non_agent_cells_have_zero_carry_bit(self):
        """Walls, empty cells, etc. should have carry=0."""
        # Type.wall=2, Color.grey=5, STATE=0
        image = np.array([[[2, 5, 0]]], dtype=np.uint8)
        result = _one_hot(image, _DIM_SIZES)

        assert result[0, 0, 23] == 0.0

    def test_one_hot_dim_is_24(self):
        """Confirm the total one-hot dimension is 24."""
        assert _ONE_HOT_DIM == 24
        assert _ONE_HOT_DIM == len(Type) + len(Color) + len(Direction) + 1
