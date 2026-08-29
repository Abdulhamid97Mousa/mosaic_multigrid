"""Tests for Basketball 3vs3 environments.

Verifies that:
1. The IndAgObs variant can be created, reset, and stepped
2. Basketball court rendering produces valid RGB frames
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import mosaic_multigrid.envs
from mosaic_multigrid.envs import (
    BasketballGame6HIndAgObsEnv19x11N3,
)

SEED = 42


@pytest.fixture
def env_ind():
    env = gym.make('MosaicMultiGrid-BB-3v3-IndAgObs-v1')
    yield env
    env.close()


class TestBasketballCreation:
    """Environment construction and basic API tests."""

    def test_indagobs_creation(self, env_ind):
        assert env_ind is not None

    def test_six_agents(self, env_ind):
        obs, _ = env_ind.reset(seed=SEED)
        assert len(obs) == 6

    def test_grid_dimensions(self, env_ind):
        env_ind.reset(seed=SEED)
        inner = env_ind.unwrapped
        assert inner.width == 19
        assert inner.height == 11

    def test_team_assignments(self, env_ind):
        env_ind.reset(seed=SEED)
        inner = env_ind.unwrapped
        teams = [a.team_index for a in inner.agents]
        assert teams == [1, 1, 1, 2, 2, 2]

    def test_goals_at_baselines(self, env_ind):
        env_ind.reset(seed=SEED)
        inner = env_ind.unwrapped
        assert inner.goal_pos == [[1, 5], [17, 5]]
        assert inner.goal_index == [1, 2]


class TestBasketballRendering:
    """Court rendering tests."""

    def test_render_returns_rgb_array(self):
        env = gym.make('MosaicMultiGrid-BB-3v3-IndAgObs-v1',
                        render_mode='rgb_array')
        env.reset(seed=SEED)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB
        env.close()

    def test_render_dimensions(self):
        env = gym.make('MosaicMultiGrid-BB-3v3-IndAgObs-v1',
                        render_mode='rgb_array')
        env.reset(seed=SEED)
        frame = env.render()
        # 19x11 grid at 32px per tile
        assert frame.shape == (11 * 32, 19 * 32, 3)
        env.close()


class TestBasketballGameplay:
    """Game mechanics: stepping, rewards, termination."""

    def test_step_returns_5_tuple(self, env_ind):
        env_ind.reset(seed=SEED)
        action = {i: env_ind.action_space[i].sample() for i in range(6)}
        result = env_ind.step(action)
        assert len(result) == 5

    def test_rewards_are_dict(self, env_ind):
        env_ind.reset(seed=SEED)
        action = {i: env_ind.action_space[i].sample() for i in range(6)}
        _, rewards, _, _, _ = env_ind.step(action)
        assert isinstance(rewards, dict)
        assert len(rewards) == 6

    def test_truncation_at_max_steps(self, env_ind):
        env_ind.reset(seed=SEED)
        for _ in range(256):
            action = {i: env_ind.action_space[i].sample() for i in range(6)}
            _, _, terms, truncs, _ = env_ind.step(action)
        # At step 256, should be truncated (if no team scored 2 goals)
        assert any(truncs.values())

    def test_no_zero_sum_penalty(self, env_ind):
        """Opponents must not receive negative goal rewards (zero_sum=False).

        Shaping rewards (ball-approach, carry-toward-goal) can be negative
        when agents move away from their target — that is expected behaviour.
        Only goal rewards must never penalise the non-scoring team.
        """
        inner = env_ind.unwrapped
        assert inner.zero_sum is False, "Basketball IndAgObs should not be zero-sum"


class TestRewardSignCorrectness:
    """Verify scoring team gets POSITIVE reward and opponents get zero.

    Basketball IndAgObs uses positive-only rewards (zero_sum=False).
    Scoring team gets +1 shared, opponents get 0.
    """

    def test_green_team_scores_positive_reward(self):
        """Green (team 1) scores at Blue's goal -> Green gets +reward, Blue gets 0."""
        from mosaic_multigrid.core.actions import Action
        from mosaic_multigrid.core.world_object import Ball

        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        assert env.zero_sum is False
        env.reset(seed=SEED)

        agent = env.agents[0]  # team_index = 1 (green)
        assert agent.team_index == 1

        ball = Ball(color='red', index=0)
        agent.state.carrying = ball
        agent.state.pos = (16, 5)   # one step left of Blue's goal at (17, 5)
        agent.state.dir = 0  # facing right -> walks into (17, 5)

        # IndAgObs: scoring is walk-in (forward into goal), not drop-at-goal
        actions = {0: Action.forward, 1: Action.done, 2: Action.done,
                   3: Action.done, 4: Action.done, 5: Action.done}
        _, rewards, _, _, _ = env.step(actions)

        green_total = rewards[0] + rewards[1] + rewards[2]
        blue_total = rewards[3] + rewards[4] + rewards[5]

        assert green_total > 0, (
            f"Green (scoring team) should get positive reward, got {green_total}. "
            f"Rewards: {rewards}"
        )
        assert blue_total < 1.0, (
            f"Blue (conceding team) should NOT get goal reward, got {blue_total}. "
            f"Rewards: {rewards}"
        )
        env.close()

    def test_blue_team_scores_positive_reward(self):
        """Blue (team 2) scores at Green's goal -> Blue gets +reward, Green gets 0."""
        from mosaic_multigrid.core.actions import Action
        from mosaic_multigrid.core.world_object import Ball

        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)

        agent = env.agents[3]  # team_index = 2 (blue)
        assert agent.team_index == 2

        ball = Ball(color='red', index=0)
        agent.state.carrying = ball
        agent.state.pos = (2, 5)
        agent.state.dir = 2  # facing left -> front_pos = (1, 5)

        actions = {0: Action.done, 1: Action.done, 2: Action.done,
                   3: Action.drop, 4: Action.done, 5: Action.done}
        _, rewards, _, _, _ = env.step(actions)

        green_total = rewards[0] + rewards[1] + rewards[2]
        blue_total = rewards[3] + rewards[4] + rewards[5]

        assert blue_total > 0, (
            f"Blue (scoring team) should get positive reward, got {blue_total}. "
            f"Rewards: {rewards}"
        )
        assert green_total < 1.0, (
            f"Green (conceding team) should NOT get goal reward, got {green_total}. "
            f"Rewards: {rewards}"
        )
        env.close()

    def test_own_goal_blocked(self):
        """An agent cannot score at their own team's goal."""
        from mosaic_multigrid.core.actions import Action
        from mosaic_multigrid.core.world_object import Ball

        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)

        agent = env.agents[0]  # team_index = 1 (green)
        ball = Ball(color='red', index=0)
        agent.state.carrying = ball
        agent.state.pos = (2, 5)
        agent.state.dir = 2  # facing left -> front_pos = (1, 5) = Green's goal

        actions = {0: Action.drop, 1: Action.done, 2: Action.done,
                   3: Action.done, 4: Action.done, 5: Action.done}
        _, rewards, _, _, _ = env.step(actions)

        # No goal reward should be given (shaped rewards may be non-zero)
        assert all(r < 1.0 for r in rewards.values()), (
            f"Own-goal should NOT give goal reward (+1.0), got "
            f"Rewards: {rewards}"
        )
        env.close()


class TestBasketballEventTracking:
    """goal_scored_by and passes_completed tracking in Basketball IndAgObs."""

    def test_goal_scored_by_empty_on_reset(self):
        """goal_scored_by should be empty after reset."""
        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)
        assert env.goal_scored_by == []
        env.close()

    def test_passes_empty_on_reset(self):
        """passes_completed should be empty after reset."""
        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)
        assert env.passes_completed == []
        env.close()

    def test_goal_tracking_records_scorer(self):
        """After a walk-in goal, goal_scored_by should contain the scorer's info."""
        from mosaic_multigrid.core.actions import Action
        from mosaic_multigrid.core.world_object import Ball

        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)

        agent = env.agents[0]  # team 1
        ball = Ball(color='red', index=0)
        agent.state.carrying = ball
        agent.state.pos = (16, 5)
        agent.state.dir = 0  # facing right -> walks into (17, 5) = Blue goal

        actions = {0: Action.forward, 1: Action.done, 2: Action.done,
                   3: Action.done, 4: Action.done, 5: Action.done}
        env.step(actions)

        assert len(env.goal_scored_by) == 1
        goal = env.goal_scored_by[0]
        assert goal["scorer"] == 0
        assert goal["team"] == 1
        assert "step" in goal
        env.close()

    def test_goal_in_info_dict(self):
        """Info dict should contain goal_scored_by on walk-in scoring steps."""
        from mosaic_multigrid.core.actions import Action
        from mosaic_multigrid.core.world_object import Ball

        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)

        agent = env.agents[0]
        ball = Ball(color='red', index=0)
        agent.state.carrying = ball
        agent.state.pos = (16, 5)
        agent.state.dir = 0  # facing right -> walks into (17, 5) = Blue goal

        actions = {0: Action.forward, 1: Action.done, 2: Action.done,
                   3: Action.done, 4: Action.done, 5: Action.done}
        _, _, _, _, infos = env.step(actions)

        for agent_id in infos:
            assert "goal_scored_by" in infos[agent_id]
            assert infos[agent_id]["goal_scored_by"]["scorer"] == 0
        env.close()

    def test_pass_tracking_records_passer(self):
        """Teleport pass should record passer, receiver, and team."""
        from mosaic_multigrid.core.actions import Action
        from mosaic_multigrid.core.world_object import Ball

        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)

        # Agent 0 (team 1) carries ball, drop in middle of field
        agent0 = env.agents[0]
        ball = Ball(color='red', index=0)
        agent0.state.carrying = ball
        agent0.state.pos = (9, 5)
        agent0.state.dir = 0

        # Clear teammates so only one is eligible
        env.agents[1].state.carrying = None
        env.agents[2].state.carrying = None
        env.grid.set(*agent0.front_pos, None)

        actions = {0: Action.drop, 1: Action.done, 2: Action.done,
                   3: Action.done, 4: Action.done, 5: Action.done}
        env.step(actions)

        assert len(env.passes_completed) == 1
        p = env.passes_completed[0]
        assert p["passer"] == 0
        assert p["team"] == 1
        assert p["receiver"] in (1, 2)  # one of the teammates
        assert "step" in p
        env.close()

    def test_pass_in_info_dict(self):
        """Info dict should contain pass_completed on passing steps."""
        from mosaic_multigrid.core.actions import Action
        from mosaic_multigrid.core.world_object import Ball

        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)

        agent0 = env.agents[0]
        ball = Ball(color='red', index=0)
        agent0.state.carrying = ball
        agent0.state.pos = (9, 5)
        agent0.state.dir = 0
        env.agents[1].state.carrying = None
        env.agents[2].state.carrying = None
        env.grid.set(*agent0.front_pos, None)

        actions = {0: Action.drop, 1: Action.done, 2: Action.done,
                   3: Action.done, 4: Action.done, 5: Action.done}
        _, _, _, _, infos = env.step(actions)

        for agent_id in infos:
            assert "pass_completed" in infos[agent_id]
            assert infos[agent_id]["pass_completed"]["passer"] == 0
        env.close()

    def test_tracking_resets_on_new_episode(self):
        """Both tracking lists should clear on reset."""
        from mosaic_multigrid.core.actions import Action
        from mosaic_multigrid.core.world_object import Ball

        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)

        # Score a goal to populate goal_scored_by
        agent = env.agents[0]
        ball = Ball(color='red', index=0)
        agent.state.carrying = ball
        agent.state.pos = (16, 5)
        agent.state.dir = 0

        actions = {0: Action.forward, 1: Action.done, 2: Action.done,
                   3: Action.done, 4: Action.done, 5: Action.done}
        env.step(actions)
        assert len(env.goal_scored_by) >= 1

        env.reset(seed=SEED + 1)
        assert env.goal_scored_by == []
        assert env.passes_completed == []
        assert env.steals_completed == []
        env.close()

    def test_steal_tracking_records_stealer(self):
        """Stealing from opponent should record stealer, victim, and team."""
        from mosaic_multigrid.core.actions import Action
        from mosaic_multigrid.core.world_object import Ball

        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)

        # Agent 3 (team 2) carries ball
        agent3 = env.agents[3]
        ball = Ball(color='red', index=0)
        agent3.state.carrying = ball
        agent3.state.pos = (9, 5)

        # Agent 0 (team 1) faces agent 3 and steals
        agent0 = env.agents[0]
        agent0.state.carrying = None
        agent0.state.pos = (8, 5)
        agent0.state.dir = 0  # facing right -> front_pos = (9, 5)
        agent0.action_cooldown = 0

        actions = {0: Action.pickup, 1: Action.done, 2: Action.done,
                   3: Action.done, 4: Action.done, 5: Action.done}
        env.step(actions)

        assert len(env.steals_completed) == 1
        s = env.steals_completed[0]
        assert s["stealer"] == 0
        assert s["victim"] == 3
        assert s["team"] == 1
        assert "step" in s
        env.close()

    def test_steal_in_info_dict(self):
        """Info dict should contain steal_completed on stealing steps."""
        from mosaic_multigrid.core.actions import Action
        from mosaic_multigrid.core.world_object import Ball

        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)

        agent3 = env.agents[3]
        ball = Ball(color='red', index=0)
        agent3.state.carrying = ball
        agent3.state.pos = (9, 5)

        agent0 = env.agents[0]
        agent0.state.carrying = None
        agent0.state.pos = (8, 5)
        agent0.state.dir = 0
        agent0.action_cooldown = 0

        actions = {0: Action.pickup, 1: Action.done, 2: Action.done,
                   3: Action.done, 4: Action.done, 5: Action.done}
        _, _, _, _, infos = env.step(actions)

        for agent_id in infos:
            assert "steal_completed" in infos[agent_id]
            assert infos[agent_id]["steal_completed"]["stealer"] == 0
        env.close()


# ---------------------------------------------------------------
# Step telemetry tests (position + carrying in infos)
# ---------------------------------------------------------------

class TestBasketballStepTelemetry:
    """Verify 'position' and 'carrying' keys appear in infos after every step.

    These keys are injected by BasketballGameIndAgObsEnv.step() to support
    post-hoc credit assignment and trajectory analysis.
    """

    def test_position_present_in_infos(self):
        from mosaic_multigrid.core import Action
        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)
        _, _, _, _, infos = env.step({i: Action.noop for i in range(6)})
        for agent_id in range(6):
            assert "position" in infos[agent_id], (
                f"'position' missing from infos[{agent_id}]"
            )
        env.close()

    def test_carrying_present_in_infos(self):
        from mosaic_multigrid.core import Action
        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)
        _, _, _, _, infos = env.step({i: Action.noop for i in range(6)})
        for agent_id in range(6):
            assert "carrying" in infos[agent_id], (
                f"'carrying' missing from infos[{agent_id}]"
            )
        env.close()

    def test_position_is_tuple_of_two_ints(self):
        from mosaic_multigrid.core import Action
        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)
        _, _, _, _, infos = env.step({i: Action.noop for i in range(6)})
        for agent_id in range(6):
            pos = infos[agent_id]["position"]
            assert isinstance(pos, tuple), f"position is {type(pos)}, expected tuple"
            assert len(pos) == 2
            assert all(isinstance(c, int) for c in pos)
        env.close()

    def test_carrying_is_bool(self):
        from mosaic_multigrid.core import Action
        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)
        _, _, _, _, infos = env.step({i: Action.noop for i in range(6)})
        for agent_id in range(6):
            assert isinstance(infos[agent_id]["carrying"], bool)
        env.close()

    def test_position_matches_agent_state(self):
        from mosaic_multigrid.core import Action
        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)
        _, _, _, _, infos = env.step({i: Action.noop for i in range(6)})
        for agent in env.agents:
            expected = tuple(int(c) for c in agent.state.pos)
            assert infos[agent.index]["position"] == expected, (
                f"infos position {infos[agent.index]['position']} != "
                f"agent state {expected}"
            )
        env.close()

    def test_carrying_false_when_not_carrying(self):
        from mosaic_multigrid.core import Action
        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)
        for agent in env.agents:
            agent.state.carrying = None
        _, _, _, _, infos = env.step({i: Action.noop for i in range(6)})
        for agent_id in range(6):
            assert infos[agent_id]["carrying"] is False
        env.close()

    def test_carrying_true_when_agent_holds_ball(self):
        from mosaic_multigrid.core import Action
        from mosaic_multigrid.core.world_object import Ball
        from mosaic_multigrid.core.constants import Color
        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)
        env.agents[0].state.carrying = Ball(color=Color.red, index=0)
        _, _, _, _, infos = env.step({i: Action.noop for i in range(6)})
        assert infos[0]["carrying"] is True
        env.close()

    def test_telemetry_present_on_every_step(self):
        """Telemetry must appear on every step, not just scoring steps."""
        from mosaic_multigrid.core import Action
        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)
        for step_num in range(10):
            _, _, _, _, infos = env.step({i: Action.noop for i in range(6)})
            for agent_id in range(6):
                assert "position" in infos[agent_id], (
                    f"step {step_num}: 'position' missing for agent {agent_id}"
                )
                assert "carrying" in infos[agent_id], (
                    f"step {step_num}: 'carrying' missing for agent {agent_id}"
                )
        env.close()
