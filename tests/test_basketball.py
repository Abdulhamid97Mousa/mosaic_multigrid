"""Tests for Basketball 3vs3 environments.

Verifies that:
1. Both IndAgObs and TeamObs variants can be created, reset, and stepped
2. TeamObs has exactly 3 extra observation keys vs IndAgObs
3. Shared keys (image, direction, mission) are identical under same seed
4. TeamObs teammate_positions are verified relative (dx, dy) offsets
5. Observation spaces are structurally different
6. Basketball court rendering produces valid RGB frames
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import mosaic_multigrid.envs
from mosaic_multigrid.envs import (
    BasketballGame6HIndAgObsEnv19x11N3,
    Basketball3vs3TeamObsEnv,
)

SEED = 42
EXPECTED_EXTRA_KEYS = {'teammate_positions', 'teammate_directions', 'teammate_has_ball'}


@pytest.fixture
def env_ind():
    env = gym.make('MosaicMultiGrid-Basketball-3vs3-IndAgObs-v0')
    yield env
    env.close()


@pytest.fixture
def env_team():
    env = gym.make('MosaicMultiGrid-Basketball-3vs3-TeamObs-v0')
    yield env
    env.close()


class TestBasketballCreation:
    """Environment construction and basic API tests."""

    def test_indagobs_creation(self, env_ind):
        assert env_ind is not None

    def test_teamobs_creation(self, env_team):
        assert env_team is not None

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


class TestObservationDifference:
    """Prove IndAgObs and TeamObs observations are structurally different."""

    def test_teamobs_has_extra_keys(self, env_ind, env_team):
        obs_ind, _ = env_ind.reset(seed=SEED)
        obs_team, _ = env_team.reset(seed=SEED)

        ind_keys = set(obs_ind[0].keys())
        team_keys = set(obs_team[0].keys())
        extra = team_keys - ind_keys

        assert extra == EXPECTED_EXTRA_KEYS

    def test_indagobs_never_has_teammate_features(self, env_ind):
        obs, _ = env_ind.reset(seed=SEED)
        for _ in range(20):
            action = {i: env_ind.action_space[i].sample() for i in range(6)}
            obs, _, _, _, _ = env_ind.step(action)

        for agent_id in range(6):
            assert 'teammate_positions' not in obs[agent_id]
            assert 'teammate_directions' not in obs[agent_id]
            assert 'teammate_has_ball' not in obs[agent_id]

    def test_shared_keys_identical_under_same_seed(self, env_ind, env_team):
        obs_ind, _ = env_ind.reset(seed=SEED)
        obs_team, _ = env_team.reset(seed=SEED)

        shared_keys = set(obs_ind[0].keys())
        for agent_id in range(6):
            for key in shared_keys:
                val_ind = obs_ind[agent_id][key]
                val_team = obs_team[agent_id][key]
                if isinstance(val_ind, np.ndarray):
                    assert np.array_equal(val_ind, val_team), \
                        f"Agent {agent_id} key '{key}' mismatch"
                else:
                    assert val_ind == val_team, \
                        f"Agent {agent_id} key '{key}' mismatch"

    def test_teammate_positions_shape(self, env_team):
        obs, _ = env_team.reset(seed=SEED)
        for agent_id in range(6):
            # 3vs3: each agent has 2 teammates
            assert obs[agent_id]['teammate_positions'].shape == (2, 2)

    def test_teammate_directions_shape(self, env_team):
        obs, _ = env_team.reset(seed=SEED)
        for agent_id in range(6):
            assert obs[agent_id]['teammate_directions'].shape == (2,)

    def test_teammate_has_ball_shape(self, env_team):
        obs, _ = env_team.reset(seed=SEED)
        for agent_id in range(6):
            assert obs[agent_id]['teammate_has_ball'].shape == (2,)

    def test_teammate_positions_are_relative(self, env_team):
        obs, _ = env_team.reset(seed=SEED)
        inner = env_team.unwrapped

        # Agent 0 (Green): teammates are agents 1 and 2
        a0 = np.array(inner.agents[0].state.pos)
        a1 = np.array(inner.agents[1].state.pos)
        a2 = np.array(inner.agents[2].state.pos)

        expected = {tuple(a1 - a0), tuple(a2 - a0)}
        actual = {tuple(obs[0]['teammate_positions'][i]) for i in range(2)}

        assert actual == expected, \
            f"Expected relative offsets {expected}, got {actual}"

    def test_observation_spaces_differ(self, env_ind, env_team):
        ind_space = env_ind.observation_space[0]
        team_space = env_team.observation_space[0]

        assert len(team_space.spaces) == len(ind_space.spaces) + 3

        for key in EXPECTED_EXTRA_KEYS:
            assert key in team_space.spaces
            assert key not in ind_space.spaces


class TestBasketballRendering:
    """Court rendering tests."""

    def test_render_returns_rgb_array(self):
        env = gym.make('MosaicMultiGrid-Basketball-3vs3-IndAgObs-v0',
                        render_mode='rgb_array')
        env.reset(seed=SEED)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB
        env.close()

    def test_render_dimensions(self):
        env = gym.make('MosaicMultiGrid-Basketball-3vs3-IndAgObs-v0',
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
        for _ in range(200):
            action = {i: env_ind.action_space[i].sample() for i in range(6)}
            _, _, terms, truncs, _ = env_ind.step(action)
        # At step 200, should be truncated (if no team scored 2 goals)
        assert any(truncs.values())

    def test_positive_only_rewards(self, env_ind):
        """Rewards should be positive-only (no negative rewards for opponents)."""
        env_ind.reset(seed=SEED)
        for _ in range(200):
            action = {i: env_ind.action_space[i].sample() for i in range(6)}
            _, rewards, terms, _, _ = env_ind.step(action)
            for r in rewards.values():
                assert r >= 0, f"Negative reward found: {rewards}"
            if any(terms.values()):
                break


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
        agent.state.pos = (16, 5)
        agent.state.dir = 0  # facing right -> front_pos = (17, 5)

        actions = {0: Action.drop, 1: Action.done, 2: Action.done,
                   3: Action.done, 4: Action.done, 5: Action.done}
        _, rewards, _, _, _ = env.step(actions)

        green_total = rewards[0] + rewards[1] + rewards[2]
        blue_total = rewards[3] + rewards[4] + rewards[5]

        assert green_total > 0, (
            f"Green (scoring team) should get positive reward, got {green_total}. "
            f"Rewards: {rewards}"
        )
        assert blue_total == 0, (
            f"Blue (conceding team) should get 0 reward, got {blue_total}. "
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
        assert green_total == 0, (
            f"Green (conceding team) should get 0 reward, got {green_total}. "
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

        total = sum(rewards.values())
        assert total == 0, (
            f"Own-goal should give zero reward, got total={total}. "
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
        """After a goal, goal_scored_by should contain the scorer's info."""
        from mosaic_multigrid.core.actions import Action
        from mosaic_multigrid.core.world_object import Ball

        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)

        agent = env.agents[0]  # team 1
        ball = Ball(color='red', index=0)
        agent.state.carrying = ball
        agent.state.pos = (16, 5)
        agent.state.dir = 0

        actions = {0: Action.drop, 1: Action.done, 2: Action.done,
                   3: Action.done, 4: Action.done, 5: Action.done}
        env.step(actions)

        assert len(env.goal_scored_by) == 1
        goal = env.goal_scored_by[0]
        assert goal["scorer"] == 0
        assert goal["team"] == 1
        assert "step" in goal
        env.close()

    def test_goal_in_info_dict(self):
        """Info dict should contain goal_scored_by on scoring steps."""
        from mosaic_multigrid.core.actions import Action
        from mosaic_multigrid.core.world_object import Ball

        env = BasketballGame6HIndAgObsEnv19x11N3(render_mode=None)
        env.reset(seed=SEED)

        agent = env.agents[0]
        ball = Ball(color='red', index=0)
        agent.state.carrying = ball
        agent.state.pos = (16, 5)
        agent.state.dir = 0

        actions = {0: Action.drop, 1: Action.done, 2: Action.done,
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

        actions = {0: Action.drop, 1: Action.done, 2: Action.done,
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
