"""Tests for RandomMixin seeding reproducibility.

Validates that after reset(seed=N), the RandomMixin's internal generator
is re-synced with the new np_random created by Gymnasium. Without this
fix, place_obj() and place_agent() use a stale generator, making grid
generation non-reproducible across reset(seed=N) calls.

Bug (before fix):
    __init__:       RandomMixin captures self.np_random (generator A)
    reset(seed=42): gym.Env.reset() replaces self.np_random (generator B)
                    RandomMixin still uses generator A (stale!)
    _gen_grid:      calls self._rand_int() -> uses generator A, NOT B
    Result:         grid layout is NOT reproducible with the same seed

Fix:
    reset() now calls RandomMixin.__init__(self, self.np_random) after
    super().reset(seed=N), re-syncing the mixin with generator B.
"""
from __future__ import annotations

import numpy as np
import pytest

from mosaic_multigrid.envs import (
    SoccerGame4HEnhancedEnv16x11N2,
    CollectGame4HEnhancedEnv10x10N2,
    CollectGame3HEnhancedEnv10x10N3,
)


class TestRandomMixinSeeding:
    """Verify that reset(seed=N) produces reproducible grid layouts."""

    @pytest.mark.parametrize("env_cls", [
        SoccerGame4HEnhancedEnv16x11N2,
        CollectGame4HEnhancedEnv10x10N2,
        CollectGame3HEnhancedEnv10x10N3,
    ])
    def test_same_seed_same_grid(self, env_cls):
        """Two resets with the same seed must produce identical grids."""
        env = env_cls(render_mode="rgb_array")

        env.reset(seed=123)
        grid_a = env.grid.encode().copy()
        positions_a = env.agent_states.pos.copy()

        env.reset(seed=123)
        grid_b = env.grid.encode().copy()
        positions_b = env.agent_states.pos.copy()

        np.testing.assert_array_equal(
            grid_a, grid_b,
            err_msg="Same seed must produce identical grid layout",
        )
        np.testing.assert_array_equal(
            positions_a, positions_b,
            err_msg="Same seed must produce identical agent positions",
        )
        env.close()

    @pytest.mark.parametrize("env_cls", [
        SoccerGame4HEnhancedEnv16x11N2,
        CollectGame4HEnhancedEnv10x10N2,
        CollectGame3HEnhancedEnv10x10N3,
    ])
    def test_different_seed_different_grid(self, env_cls):
        """Different seeds should (almost certainly) produce different grids."""
        env = env_cls(render_mode="rgb_array")

        env.reset(seed=100)
        grid_a = env.grid.encode().copy()

        env.reset(seed=999)
        grid_b = env.grid.encode().copy()

        # Grids should differ (statistically guaranteed for different seeds)
        assert not np.array_equal(grid_a, grid_b), (
            "Different seeds should produce different grids"
        )
        env.close()

    @pytest.mark.parametrize("env_cls", [
        SoccerGame4HEnhancedEnv16x11N2,
        CollectGame4HEnhancedEnv10x10N2,
        CollectGame3HEnhancedEnv10x10N3,
    ])
    def test_seed_reproducible_across_instances(self, env_cls):
        """Two separate instances with the same seed must produce identical grids."""
        env1 = env_cls(render_mode="rgb_array")
        env2 = env_cls(render_mode="rgb_array")

        env1.reset(seed=42)
        env2.reset(seed=42)

        np.testing.assert_array_equal(
            env1.grid.encode(), env2.grid.encode(),
            err_msg="Same seed on different instances must produce identical grids",
        )
        np.testing.assert_array_equal(
            env1.agent_states.pos, env2.agent_states.pos,
            err_msg="Same seed on different instances must produce identical agent positions",
        )
        env1.close()
        env2.close()

    def test_mixin_generator_is_synced_after_reset(self):
        """After reset(seed=N), the mixin's internal generator must match np_random."""
        env = CollectGame3HEnhancedEnv10x10N3(render_mode="rgb_array")
        env.reset(seed=77)

        # Access the name-mangled attribute
        mixin_gen = env._RandomMixin__np_random

        # The mixin's generator should be the same object as self.np_random
        assert mixin_gen is env.np_random, (
            "RandomMixin.__np_random must be re-synced to self.np_random after reset"
        )
        env.close()

    def test_multiple_resets_stay_reproducible(self):
        """Reproducibility should hold even after many reset cycles."""
        env = CollectGame4HEnhancedEnv10x10N2(render_mode="rgb_array")

        # Do several resets with different seeds first
        for s in [1, 2, 3, 4, 5]:
            env.reset(seed=s)

        # Now reset with target seed
        env.reset(seed=42)
        grid_after_many = env.grid.encode().copy()
        pos_after_many = env.agent_states.pos.copy()

        # Fresh env, reset once with same seed
        env2 = CollectGame4HEnhancedEnv10x10N2(render_mode="rgb_array")
        env2.reset(seed=42)

        np.testing.assert_array_equal(
            grid_after_many, env2.grid.encode(),
            err_msg="Seed reproducibility must hold after many prior resets",
        )
        np.testing.assert_array_equal(
            pos_after_many, env2.agent_states.pos,
            err_msg="Agent position reproducibility must hold after many prior resets",
        )
        env.close()
        env2.close()


class TestParameterValidation:
    """Verify that mismatched parameter lengths raise ValueError."""

    def test_collect_mismatched_num_balls_balls_index(self):
        with pytest.raises(ValueError, match="num_balls.*balls_index"):
            CollectGame3HEnhancedEnv10x10N3.__bases__[0](
                num_balls=[5, 3],
                balls_index=[0],
                balls_reward=[1.0, 1.0],
                agents_index=[1, 2, 3],
            )

    def test_collect_mismatched_num_balls_balls_reward(self):
        with pytest.raises(ValueError, match="num_balls.*balls_reward"):
            CollectGame3HEnhancedEnv10x10N3.__bases__[0](
                num_balls=[5],
                balls_index=[0],
                balls_reward=[1.0, 2.0],
                agents_index=[1, 2, 3],
            )

    def test_soccer_mismatched_goal_pos_goal_index(self):
        with pytest.raises(ValueError, match="goal_pos.*goal_index"):
            SoccerGame4HEnhancedEnv16x11N2.__bases__[0](
                goal_pos=[[1, 3], [6, 3]],
                goal_index=[1],
                num_balls=[1],
                balls_index=[0],
                agents_index=[1, 1, 2, 2],
            )

    def test_soccer_mismatched_num_balls_balls_index(self):
        with pytest.raises(ValueError, match="num_balls.*balls_index"):
            SoccerGame4HEnhancedEnv16x11N2.__bases__[0](
                goal_pos=[[1, 3]],
                goal_index=[1],
                num_balls=[1, 2],
                balls_index=[0],
                agents_index=[1, 1, 2, 2],
            )
