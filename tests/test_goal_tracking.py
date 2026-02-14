"""Tests for event tracking and positive-only rewards in Soccer IndAgObs.

Changes tested:
- zero_sum=False in IndAgObs Soccer variants (no negative rewards to opponents)
- goal_scored_by list tracks which agent scored each goal
- goal_scored_by appears in info dict on scoring steps only
- passes_completed list tracks teleport passes between teammates
- passes_completed appears in info dict on passing steps only
"""
import numpy as np

from mosaic_multigrid.core.actions import Action
from mosaic_multigrid.core.world_object import Ball
from mosaic_multigrid.envs import (
    SoccerGame4HIndAgObsEnv16x11N2,
    SoccerGame2HIndAgObsEnv16x11N2,
)

SEED = 42


class TestZeroSumRemoved:
    """IndAgObs Soccer variants no longer use zero-sum rewards."""

    def test_2v2_no_negative_rewards(self):
        """When team 1 scores, team 2 should get 0 reward (not -1)."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        assert env.zero_sum is False

        env.reset(seed=SEED)

        # Place agent 0 (team 1) facing team 2's goal at (14, 5)
        agent = env.agents[0]
        assert agent.team_index == 1

        ball = Ball(color='red', index=0)
        agent.state.carrying = ball
        agent.state.pos = (13, 5)
        agent.state.dir = 0  # facing right -> front_pos = (14, 5)

        actions = {0: Action.drop, 1: Action.done, 2: Action.done, 3: Action.done}
        _, rewards, _, _, _ = env.step(actions)

        # Scoring team gets +1 each (shared reward)
        assert rewards[0] == 1.0, f"Scorer should get +1, got {rewards[0]}"
        assert rewards[1] == 1.0, f"Teammate should get +1, got {rewards[1]}"

        # Opponents get 0 (not -1)
        assert rewards[2] == 0.0, f"Opponent should get 0, got {rewards[2]}"
        assert rewards[3] == 0.0, f"Opponent should get 0, got {rewards[3]}"
        env.close()

    def test_1v1_no_negative_rewards(self):
        """1v1 variant also has zero_sum=False."""
        env = SoccerGame2HIndAgObsEnv16x11N2(render_mode=None)
        assert env.zero_sum is False
        env.close()

    def test_positive_only_total(self):
        """Total reward across all agents should be positive (not zero)."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)

        agent = env.agents[0]
        ball = Ball(color='red', index=0)
        agent.state.carrying = ball
        agent.state.pos = (13, 5)
        agent.state.dir = 0

        actions = {0: Action.drop, 1: Action.done, 2: Action.done, 3: Action.done}
        _, rewards, _, _, _ = env.step(actions)

        total = sum(rewards.values())
        assert total > 0, f"Total reward should be positive, got {total}"
        assert total == 2.0, f"Total should be 2.0 (shared to 2 teammates), got {total}"
        env.close()


class TestGoalScoredByTracking:
    """goal_scored_by list tracks which agent scored each goal."""

    def test_goal_scored_by_empty_on_reset(self):
        """goal_scored_by should be empty after reset."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)
        assert env.goal_scored_by == []
        env.close()

    def test_goal_scored_by_records_scorer(self):
        """After a goal, goal_scored_by should contain the scorer's info."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)

        # Agent 0 (team 1) scores
        agent = env.agents[0]
        ball = Ball(color='red', index=0)
        agent.state.carrying = ball
        agent.state.pos = (13, 5)
        agent.state.dir = 0

        actions = {0: Action.drop, 1: Action.done, 2: Action.done, 3: Action.done}
        env.step(actions)

        assert len(env.goal_scored_by) == 1
        goal = env.goal_scored_by[0]
        assert goal["scorer"] == 0, f"Expected scorer=0, got {goal['scorer']}"
        assert goal["team"] == 1, f"Expected team=1, got {goal['team']}"
        assert "step" in goal
        env.close()

    def test_goal_scored_by_in_info_dict(self):
        """Info dict should contain goal_scored_by on scoring steps."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)

        # Agent 0 scores
        agent = env.agents[0]
        ball = Ball(color='red', index=0)
        agent.state.carrying = ball
        agent.state.pos = (13, 5)
        agent.state.dir = 0

        actions = {0: Action.drop, 1: Action.done, 2: Action.done, 3: Action.done}
        _, _, _, _, infos = env.step(actions)

        # All agents should see the goal event in their info
        for agent_id in infos:
            assert "goal_scored_by" in infos[agent_id], (
                f"Agent {agent_id} info missing goal_scored_by"
            )
            assert infos[agent_id]["goal_scored_by"]["scorer"] == 0
        env.close()

    def test_no_goal_scored_by_on_normal_step(self):
        """Info dict should NOT contain goal_scored_by on non-scoring steps."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)

        # Normal step (no scoring)
        actions = {0: Action.done, 1: Action.done, 2: Action.done, 3: Action.done}
        _, _, _, _, infos = env.step(actions)

        for agent_id in infos:
            assert "goal_scored_by" not in infos[agent_id], (
                f"Agent {agent_id} should not have goal_scored_by on non-scoring step"
            )
        env.close()

    def test_multiple_goals_tracked(self):
        """Multiple goals should all be tracked in goal_scored_by list."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)

        # Agent 0 (team 1) scores first goal
        agent0 = env.agents[0]
        ball = Ball(color='red', index=0)
        agent0.state.carrying = ball
        agent0.state.pos = (13, 5)
        agent0.state.dir = 0

        actions = {0: Action.drop, 1: Action.done, 2: Action.done, 3: Action.done}
        env.step(actions)

        assert len(env.goal_scored_by) == 1
        assert env.goal_scored_by[0]["scorer"] == 0

        # Agent 2 (team 2) scores second goal at team 1's goal (1, 5)
        agent2 = env.agents[2]
        ball2 = Ball(color='red', index=0)
        agent2.state.carrying = ball2
        agent2.state.pos = (2, 5)
        agent2.state.dir = 2  # facing left -> front_pos = (1, 5)

        actions = {0: Action.done, 1: Action.done, 2: Action.drop, 3: Action.done}
        env.step(actions)

        assert len(env.goal_scored_by) == 2
        assert env.goal_scored_by[1]["scorer"] == 2
        assert env.goal_scored_by[1]["team"] == 2
        env.close()

    def test_goal_scored_by_resets_on_new_episode(self):
        """goal_scored_by should clear on reset."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)

        # Score a goal
        agent = env.agents[0]
        ball = Ball(color='red', index=0)
        agent.state.carrying = ball
        agent.state.pos = (13, 5)
        agent.state.dir = 0

        actions = {0: Action.drop, 1: Action.done, 2: Action.done, 3: Action.done}
        env.step(actions)
        assert len(env.goal_scored_by) == 1

        # Reset should clear it
        env.reset(seed=SEED + 1)
        assert env.goal_scored_by == []
        env.close()


class TestPassTracking:
    """passes_completed list tracks teleport passes between teammates."""

    def test_passes_empty_on_reset(self):
        """passes_completed should be empty after reset."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)
        assert env.passes_completed == []
        env.close()

    def test_pass_records_passer_and_receiver(self):
        """Teleport pass should record passer, receiver, and team."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)

        # Agent 0 (team 1) carries ball.
        # Place agent 0 NOT facing a goal or another agent in front,
        # so drop triggers teleport pass to teammate (agent 1).
        agent0 = env.agents[0]
        ball = Ball(color='red', index=0)
        agent0.state.carrying = ball

        # Face an empty cell (not a goal) so Priority 1 fails,
        # teleport pass (Priority 2) kicks in.
        agent0.state.pos = (8, 5)
        agent0.state.dir = 0  # facing right -> (9, 5)

        # Make sure agent 1 (teammate) is not carrying anything
        env.agents[1].state.carrying = None

        # Clear the cell in front so it's not a goal
        fwd = agent0.front_pos
        env.grid.set(*fwd, None)

        actions = {0: Action.drop, 1: Action.done, 2: Action.done, 3: Action.done}
        env.step(actions)

        assert len(env.passes_completed) == 1
        p = env.passes_completed[0]
        assert p["passer"] == 0
        assert p["team"] == 1
        assert p["receiver"] == 1  # only teammate on team 1
        assert "step" in p
        env.close()

    def test_pass_in_info_dict(self):
        """Info dict should contain pass_completed on passing steps."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)

        agent0 = env.agents[0]
        ball = Ball(color='red', index=0)
        agent0.state.carrying = ball
        agent0.state.pos = (8, 5)
        agent0.state.dir = 0
        env.agents[1].state.carrying = None
        env.grid.set(*agent0.front_pos, None)

        actions = {0: Action.drop, 1: Action.done, 2: Action.done, 3: Action.done}
        _, _, _, _, infos = env.step(actions)

        for agent_id in infos:
            assert "pass_completed" in infos[agent_id], (
                f"Agent {agent_id} info missing pass_completed"
            )
            assert infos[agent_id]["pass_completed"]["passer"] == 0
        env.close()

    def test_no_pass_on_normal_step(self):
        """Info dict should NOT contain pass_completed on non-passing steps."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)

        actions = {0: Action.done, 1: Action.done, 2: Action.done, 3: Action.done}
        _, _, _, _, infos = env.step(actions)

        for agent_id in infos:
            assert "pass_completed" not in infos[agent_id]
        env.close()

    def test_passes_reset_on_new_episode(self):
        """passes_completed should clear on reset."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)

        # Force a pass
        agent0 = env.agents[0]
        ball = Ball(color='red', index=0)
        agent0.state.carrying = ball
        agent0.state.pos = (8, 5)
        agent0.state.dir = 0
        env.agents[1].state.carrying = None
        env.grid.set(*agent0.front_pos, None)

        actions = {0: Action.drop, 1: Action.done, 2: Action.done, 3: Action.done}
        env.step(actions)
        assert len(env.passes_completed) >= 1

        env.reset(seed=SEED + 1)
        assert env.passes_completed == []
        env.close()


class TestStealTracking:
    """steals_completed list tracks ball steals from opponents."""

    def test_steals_empty_on_reset(self):
        """steals_completed should be empty after reset."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)
        assert env.steals_completed == []
        env.close()

    def test_steal_records_stealer_and_victim(self):
        """Stealing from opponent should record stealer, victim, and team."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)

        # Agent 2 (team 2) carries ball
        agent2 = env.agents[2]
        ball = Ball(color='red', index=0)
        agent2.state.carrying = ball
        agent2.state.pos = (8, 5)

        # Agent 0 (team 1) faces agent 2 and steals
        agent0 = env.agents[0]
        agent0.state.carrying = None
        agent0.state.pos = (7, 5)
        agent0.state.dir = 0  # facing right -> front_pos = (8, 5)
        agent0.action_cooldown = 0

        actions = {0: Action.pickup, 1: Action.done, 2: Action.done, 3: Action.done}
        env.step(actions)

        assert len(env.steals_completed) == 1
        s = env.steals_completed[0]
        assert s["stealer"] == 0
        assert s["victim"] == 2
        assert s["team"] == 1
        assert "step" in s
        env.close()

    def test_steal_in_info_dict(self):
        """Info dict should contain steal_completed on stealing steps."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)

        agent2 = env.agents[2]
        ball = Ball(color='red', index=0)
        agent2.state.carrying = ball
        agent2.state.pos = (8, 5)

        agent0 = env.agents[0]
        agent0.state.carrying = None
        agent0.state.pos = (7, 5)
        agent0.state.dir = 0
        agent0.action_cooldown = 0

        actions = {0: Action.pickup, 1: Action.done, 2: Action.done, 3: Action.done}
        _, _, _, _, infos = env.step(actions)

        for agent_id in infos:
            assert "steal_completed" in infos[agent_id]
            assert infos[agent_id]["steal_completed"]["stealer"] == 0
        env.close()

    def test_no_steal_on_normal_step(self):
        """Info dict should NOT contain steal_completed on non-stealing steps."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)

        actions = {0: Action.done, 1: Action.done, 2: Action.done, 3: Action.done}
        _, _, _, _, infos = env.step(actions)

        for agent_id in infos:
            assert "steal_completed" not in infos[agent_id]
        env.close()

    def test_steals_reset_on_new_episode(self):
        """steals_completed should clear on reset."""
        env = SoccerGame4HIndAgObsEnv16x11N2(render_mode=None)
        env.reset(seed=SEED)

        # Force a steal
        agent2 = env.agents[2]
        ball = Ball(color='red', index=0)
        agent2.state.carrying = ball
        agent2.state.pos = (8, 5)

        agent0 = env.agents[0]
        agent0.state.carrying = None
        agent0.state.pos = (7, 5)
        agent0.state.dir = 0
        agent0.action_cooldown = 0

        actions = {0: Action.pickup, 1: Action.done, 2: Action.done, 3: Action.done}
        env.step(actions)
        assert len(env.steals_completed) >= 1

        env.reset(seed=SEED + 1)
        assert env.steals_completed == []
        env.close()
