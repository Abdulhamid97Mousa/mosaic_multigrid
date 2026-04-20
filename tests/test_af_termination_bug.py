"""Regression test for American Football termination bug.

Bug report:
    Green scores 1 goal, Blue scores 1 goal (total = 2 goals, 1 per team).
    Agents freeze (stop moving). Expected: game continues until a SINGLE
    team reaches goals_to_win (2). Actual: game appeared to end when total
    goals across both teams == goals_to_win.

Root cause under investigation:
    1. Incorrect termination trigger (total goals vs per-team goals).
    2. Ball respawn bug: after a touchdown self.place_obj(ball) is called
       with NO bounding box, so the ball can land anywhere — including
       adjacent to the end zone where the scoring agent still stands.
       If the agent picks up the ball while still in the end zone the
       touchdown check fires again immediately, causing a team to appear
       to score 2 goals before the opponent has a chance to respond.
"""
from __future__ import annotations

import numpy as np
import pytest

from mosaic_multigrid.envs.american_football_game import AmericanFootball2v2Env16x11
from mosaic_multigrid.core.world_object import Ball
from mosaic_multigrid.core.constants import Color


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(seed: int = 0) -> AmericanFootball2v2Env16x11:
    env = AmericanFootball2v2Env16x11()
    env.reset(seed=seed)
    return env


def _find_ball(env) -> Ball | None:
    """Return the first Ball object found on the grid (not carried)."""
    for x in range(env.width):
        for y in range(env.height):
            obj = env.grid.get(x, y)
            if isinstance(obj, Ball):
                return obj
    return None


def _teleport_agent_to_endzone_with_ball(env, agent_index: int):
    """
    Place *agent_index* inside the opponent's end zone while carrying the
    ball.  The agent is guaranteed to be carrying exactly one ball.

    Returns the ball object that was placed.
    """
    agent = env.agents[agent_index]
    team = agent.team_index

    # Green (team 0) scores at column width-2 = 14.
    # Blue  (team 1) scores at column 1.
    scoring_col = env.width - 2 if team == 0 else 1

    # Pick a row in the middle of the end zone.
    row = env.height // 2

    # --- Remove agent from its current position in the grid view ---
    # (AgentState is a shared-memory numpy array so just overwrite pos.)
    agent.state.pos = np.array([scoring_col, row])

    # --- Give the agent the ball ---
    ball = _find_ball(env)
    if ball is None:
        # Ball is already being carried; steal it.
        for a in env.agents:
            if a.state.carrying is not None:
                ball = a.state.carrying
                a.state.carrying = None
                env.grid.set(*ball.cur_pos, None)
                break
    else:
        env.grid.set(*ball.cur_pos, None)

    agent.state.carrying = ball
    return ball


# ---------------------------------------------------------------------------
# Test 1 — termination is NOT triggered after 1 goal per team
# ---------------------------------------------------------------------------

class TestOneGoalPerTeam:
    """After green scores 1 and blue scores 1 the game must continue."""

    def _score_once(self, env, agent_index: int):
        """Teleport *agent_index* into the end zone with ball, then step."""
        _teleport_agent_to_endzone_with_ball(env, agent_index)
        actions = {i: 0 for i in range(env.num_agents)}  # noop
        obs, rewards, terminated, truncated, info = env.step(actions)
        return terminated, truncated

    def test_no_termination_after_one_goal_each(self):
        """Core regression: 1 goal each must NOT end the episode."""
        env = _make_env(seed=42)

        # Green (agent 0) scores goal 1.
        terminated, truncated = self._score_once(env, agent_index=0)
        assert env.team_scores[0] == 1, "Green should have 1 goal"
        assert env.team_scores[1] == 0, "Blue should have 0 goals"
        assert not any(terminated.values()), (
            f"BUG: Episode terminated after green's first goal. "
            f"terminated={terminated}, team_scores={env.team_scores}"
        )

        # Blue (agent 2) scores goal 1.
        terminated, truncated = self._score_once(env, agent_index=2)
        assert env.team_scores[1] == 1, "Blue should have 1 goal"
        assert not any(terminated.values()), (
            f"BUG: Episode terminated after blue's first goal. "
            f"team_scores={env.team_scores}, terminated={terminated}"
        )

    def test_agents_can_still_act_after_one_goal_each(self):
        """Agents must be able to take actions after 1 goal each."""
        env = _make_env(seed=0)
        self._score_once(env, agent_index=0)
        self._score_once(env, agent_index=2)

        # All agents should have terminated=False in their state.
        for agent in env.agents:
            assert not agent.state.terminated, (
                f"BUG: agent {agent.index} has state.terminated=True "
                f"with only 1 goal per team."
            )

        # Stepping again should not raise and should return valid obs.
        actions = {i: env.action_space[i].sample() for i in range(env.num_agents)}
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert len(obs) == env.num_agents
        assert not any(terminated.values())

    def test_terminated_dict_keys_match_agent_count(self):
        """Terminated dict must cover all agents, not just scoring agent."""
        env = _make_env(seed=1)
        self._score_once(env, agent_index=0)
        self._score_once(env, agent_index=2)

        actions = {i: 0 for i in range(env.num_agents)}
        _, _, terminated, _, _ = env.step(actions)
        assert set(terminated.keys()) == set(range(env.num_agents))


# ---------------------------------------------------------------------------
# Test 2 — termination IS triggered when one team reaches goals_to_win
# ---------------------------------------------------------------------------

class TestWinCondition:
    """Game ends only when a single team reaches goals_to_win (2)."""

    def test_green_wins_after_two_goals(self):
        """Green winning (2 goals) must terminate the episode."""
        env = _make_env(seed=10)

        for _ in range(env.goals_to_win):
            _teleport_agent_to_endzone_with_ball(env, agent_index=0)
            actions = {i: 0 for i in range(env.num_agents)}
            _, _, terminated, _, _ = env.step(actions)

        assert env.team_scores[0] == env.goals_to_win, (
            f"Green should have {env.goals_to_win} goals, got {env.team_scores[0]}"
        )
        assert all(terminated.values()), (
            f"BUG: Episode did NOT terminate after green won. "
            f"terminated={terminated}"
        )

    def test_blue_wins_after_two_goals(self):
        """Blue winning (2 goals) must terminate the episode."""
        env = _make_env(seed=20)

        for _ in range(env.goals_to_win):
            _teleport_agent_to_endzone_with_ball(env, agent_index=2)
            actions = {i: 0 for i in range(env.num_agents)}
            _, _, terminated, _, _ = env.step(actions)

        assert env.team_scores[1] == env.goals_to_win
        assert all(terminated.values()), (
            f"BUG: Episode did NOT terminate after blue won. "
            f"terminated={terminated}"
        )

    def test_termination_requires_two_goals_not_one(self):
        """Scoring exactly one goal should NOT terminate the episode."""
        env = _make_env(seed=30)
        _teleport_agent_to_endzone_with_ball(env, agent_index=0)
        actions = {i: 0 for i in range(env.num_agents)}
        _, _, terminated, _, _ = env.step(actions)

        assert env.team_scores[0] == 1
        assert not any(terminated.values()), (
            f"BUG: Episode terminated after only 1 goal. "
            f"goals_to_win={env.goals_to_win}, terminated={terminated}"
        )


# ---------------------------------------------------------------------------
# Test 3 — ball respawn is confined to midfield after a touchdown
# ---------------------------------------------------------------------------

class TestBallRespawnAfterTouchdown:
    """
    Bug: after a touchdown, place_obj(ball) is called with no bounding box.
    The ball can land in the end zone or adjacent to it, allowing the scoring
    agent to immediately score again on the next step (pickup while in end zone).
    The ball must be respawned in midfield (columns 2 .. width-3).
    """

    def test_ball_not_in_endzone_after_touchdown(self):
        """Ball must not respawn in an end zone column after a goal."""
        env = _make_env(seed=99)
        endzone_cols = {1, env.width - 2}

        for trial in range(30):
            env.reset(seed=trial)
            _teleport_agent_to_endzone_with_ball(env, agent_index=0)
            actions = {i: 0 for i in range(env.num_agents)}
            env.step(actions)

            ball = _find_ball(env)
            if ball is not None:
                col = int(ball.cur_pos[0])
                assert col not in endzone_cols, (
                    f"BUG (trial {trial}): Ball respawned at column {col} "
                    f"which is an end zone. ball.cur_pos={ball.cur_pos}"
                )

    def test_ball_in_midfield_after_touchdown(self):
        """Ball must respawn inside midfield (columns 2..width-3)."""
        env = _make_env(seed=50)
        midfield_min = 2
        midfield_max = env.width - 3  # inclusive

        for trial in range(30):
            env.reset(seed=trial * 7)
            _teleport_agent_to_endzone_with_ball(env, agent_index=0)
            actions = {i: 0 for i in range(env.num_agents)}
            env.step(actions)

            ball = _find_ball(env)
            if ball is not None:
                col = int(ball.cur_pos[0])
                assert midfield_min <= col <= midfield_max, (
                    f"BUG (trial {trial}): Ball respawned at column {col}, "
                    f"expected midfield [{midfield_min}, {midfield_max}]. "
                    f"ball.cur_pos={ball.cur_pos}"
                )


# ---------------------------------------------------------------------------
# Test 4 — team_scores per-team semantics (not total)
# ---------------------------------------------------------------------------

class TestTeamScoresSemantics:
    """team_scores must track per-team totals, not aggregate."""

    def test_each_team_score_tracked_independently(self):
        env = _make_env(seed=77)

        # Green scores once.
        _teleport_agent_to_endzone_with_ball(env, agent_index=0)
        env.step({i: 0 for i in range(env.num_agents)})
        assert env.team_scores == {0: 1, 1: 0}, env.team_scores

        # Blue scores once.
        _teleport_agent_to_endzone_with_ball(env, agent_index=2)
        env.step({i: 0 for i in range(env.num_agents)})
        assert env.team_scores == {0: 1, 1: 1}, env.team_scores

        # Total = 2, but neither team has reached goals_to_win=2.
        # The sum must NOT be compared against goals_to_win.
        total = sum(env.team_scores.values())
        assert total == 2
        assert total < env.goals_to_win * 2  # sanity
        for agent in env.agents:
            assert not agent.state.terminated, (
                f"BUG: agent {agent.index} terminated with team_scores="
                f"{env.team_scores} (neither team won)"
            )
