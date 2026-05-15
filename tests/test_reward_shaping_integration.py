"""Integration tests for reward shaping through the REAL action pipeline.

Why these tests exist
---------------------
The tests in test_pickup_shaping_no_exploit.py verify the shaping *rule* by
setting agent.state.carrying directly.  But the actual exploit that MAPPO
policies discovered goes through the REAL action pipeline:

  agent A uses DROP action -> _handle_drop teleports ball to teammate B
  -> B now has carrying=True -> shaping block sees False->True transition

These tests exercise the FULL pipeline using real game actions and prove:
  1. A teleport pass (DROP action) to a teammate earns ZERO pickup bonus
  2. A pass-loop of N drops accumulates only the initial ground-pickup bonus
  3. A scoring sequence earns the full sparse reward + shaping
  4. Scoring total >> pass-loop total
  5. Own-goals are impossible (game-logic guard)

Key mechanic: PICKUP (action 4) picks up from agent.front_pos, not the
agent's cell.  DROP (action 5) triggers teleport pass to a random free
teammate.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import mosaic_multigrid.envs  # noqa: F401

RIGHT = 0
LEFT = 2

NOOP = 0
FORWARD = 3
PICKUP = 4
DROP = 5

EPS = 1e-6


def _find_ball_on_grid(inner):
    """Find the first Ball on the grid, remove it, return (ball, pos)."""
    for x in range(inner.width):
        for y in range(inner.height):
            obj = inner.grid.get(x, y)
            if obj is not None and getattr(obj, "type", None) == "ball":
                inner.grid.set(x, y, None)
                return obj, (x, y)
    raise AssertionError("No ball found on the grid")


def _place(inner, idx, pos, direction, carrying=None):
    """Position agent and sync all shaping bookkeeping.

    When carrying is set (simulating a ground pickup), _prev_carrying is
    forced to False so the next step() sees a False->True transition and
    fires the +0.1 pickup bonus.  This matches what happens in a real game:
    the agent picks up the ball in the SAME step that transitions carrying.
    """
    a = inner.agents[idx]
    a.state.pos = np.array(list(pos), dtype=np.int64)
    a.state.dir = direction
    a.state.carrying = carrying
    cx, cy = int(pos[0]), int(pos[1])
    # Always set prev_carrying=False so the first step fires the pickup bonus
    # when carrying is set (simulates a pickup that happened "just now").
    if hasattr(inner, "_prev_carrying"):
        inner._prev_carrying[idx] = False
    if hasattr(inner, "_prev_x"):
        inner._prev_x[idx] = cx
    if hasattr(inner, "_prev_pos"):
        inner._prev_pos[idx] = (cx, cy)


def _reset_provenance(inner):
    """Clear ball provenance so ground pickups earn the bonus."""
    if hasattr(inner, "_ball_last_carrier_team"):
        inner._ball_last_carrier_team = {}


def _park_others(inner, exclude, base_x=7, base_y=1):
    """Park agents not in `exclude` at non-overlapping mid-field positions."""
    slot = 0
    for i in range(len(inner.agents)):
        if i not in exclude:
            _place(inner, i, (base_x + slot, base_y + slot), RIGHT)
            slot += 1


def _find_holder(inner, agent_ids):
    """Return the agent_id in agent_ids who is carrying, or None."""
    for j in agent_ids:
        if inner.agents[j].state.carrying is not None:
            return j
    return None


def _team_agents(inner, team_idx):
    """Return list of agent indices on the given team."""
    return [i for i, a in enumerate(inner.agents) if a.team_index == team_idx]


# =====================================================================
# American Football 2v2
# =====================================================================

class TestAFLiveActions:
    ENV_ID = "MosaicMultiGrid-AF-2v2-IndAgObs-v1"
    N = 4

    @pytest.fixture
    def env(self):
        e = gym.make(self.ENV_ID)
        e.reset(seed=42)
        yield e
        e.close()

    def _noop(self):
        return {i: NOOP for i in range(self.N)}

    @pytest.mark.skip(reason="+0.1 pickup bonus removed in v6.5.0")
    def test_ground_pickup_via_action_earns_bonus(self, env):
        """PICKUP action on a ground ball -> +0.1 shaping bonus."""
        inner = env.unwrapped
        ball, bpos = _find_ball_on_grid(inner)

        # Place ball at (6,5), agent 0 at (5,5) facing right so front_pos=(6,5)
        inner.grid.set(6, 5, ball)
        _place(inner, 0, (5, 5), RIGHT)
        _reset_provenance(inner)
        _park_others(inner, {0})

        actions = self._noop()
        actions[0] = PICKUP
        _, rewards, _, _, _ = env.step(actions)

        assert rewards[0] == pytest.approx(0.1, abs=EPS), (
            f"Ground pickup should give +0.1, got {rewards[0]}"
        )
        assert inner.agents[0].state.carrying is not None

    def test_teleport_pass_receiver_gets_zero(self, env):
        """DROP action -> teleport to teammate. Receiver gets 0 bonus.

        This is the actual exploit path through real actions.
        """
        inner = env.unwrapped
        ball, _ = _find_ball_on_grid(inner)

        # Give ball directly to agent 0 (simulate a ground pickup that
        # already happened). Then DROP triggers teleport pass.
        _place(inner, 0, (7, 5), RIGHT, carrying=ball)
        _reset_provenance(inner)
        _park_others(inner, {0})

        # Step to sync carrying state
        env.step(self._noop())

        # Agent 0 drops -> teleport to random free teammate
        actions = self._noop()
        actions[0] = DROP
        _, r, _, _, _ = env.step(actions)

        assert inner.agents[0].state.carrying is None, "Agent 0 should have dropped"

        team0 = _team_agents(inner, 0)
        receiver = _find_holder(inner, team0)
        assert receiver is not None, "Teammate should have received ball"

        # The receiver's reward must be 0 (same-team provenance check)
        assert abs(r[receiver]) < EPS, (
            f"Same-team teleport receive should get 0, agent {receiver} got {r[receiver]}"
        )

    @pytest.mark.skip(reason="+0.1 pickup bonus removed in v6.5.0")
    def test_pass_loop_10_hops_single_bonus(self, env):
        """10 teleport-pass cycles: total team reward = 0.1 (initial only).

        The exploit under v6.4.0 would have produced 1.1.
        """
        inner = env.unwrapped
        ball, _ = _find_ball_on_grid(inner)

        _place(inner, 0, (7, 5), RIGHT, carrying=ball)
        _reset_provenance(inner)
        _park_others(inner, {0})

        team0 = _team_agents(inner, 0)
        total = 0.0

        # First step: agent 0 has ball from ground -> +0.1 bonus
        _, r, _, _, _ = env.step(self._noop())
        total += sum(r[j] for j in team0)

        # 10 drop cycles
        for cycle in range(10):
            holder = _find_holder(inner, team0)
            assert holder is not None, f"Cycle {cycle}: nobody has ball"
            actions = self._noop()
            actions[holder] = DROP
            _, r, _, _, _ = env.step(actions)
            total += sum(r[j] for j in team0)

        assert total == pytest.approx(0.1, abs=EPS), (
            f"10-hop pass-loop should total 0.1, got {total}. "
            "If >0.1 the exploit is still present."
        )

    def test_scoring_earns_touchdown(self, env):
        """Pickup + 2 forward steps into end zone -> +0.1 + 0.02 + 1.0 = 1.12."""
        inner = env.unwrapped
        ball, _ = _find_ball_on_grid(inner)

        # Agent 0 (team 0) scores at column 14.
        # Place agent at (12,5), ball at (13,5). Pickup, fwd to 13, fwd to 14.
        inner.grid.set(13, 5, ball)
        _place(inner, 0, (12, 5), RIGHT)
        _reset_provenance(inner)
        _park_others(inner, {0})

        total = 0.0

        # Pickup
        actions = self._noop()
        actions[0] = PICKUP
        _, r, _, _, _ = env.step(actions)
        total += r[0]
        assert inner.agents[0].state.carrying is not None

        # Forward to 13
        actions = self._noop()
        actions[0] = FORWARD
        _, r, _, _, _ = env.step(actions)
        total += r[0]

        # Forward to 14 (endzone -> touchdown)
        actions = self._noop()
        actions[0] = FORWARD
        _, r, _, _, _ = env.step(actions)
        total += r[0]

        assert total > 1.0, f"Scoring should earn >1.0, got {total:.3f}"
        assert len(inner.goal_scored_by) > 0

    def test_scoring_vs_passloop_gradient(self, env):
        """Scoring earns >= 10x pass-loop. Gradient points at scoring."""
        # --- Pass-loop ---
        env_l = gym.make(self.ENV_ID)
        env_l.reset(seed=42)
        il = env_l.unwrapped
        ball_l, _ = _find_ball_on_grid(il)
        _place(il, 0, (7, 5), RIGHT, carrying=ball_l)
        _reset_provenance(il)
        _park_others(il, {0})

        team0 = _team_agents(il, 0)
        loop_total = 0.0
        _, r, _, _, _ = env_l.step(self._noop())
        loop_total += sum(r[j] for j in team0)
        for _ in range(10):
            holder = _find_holder(il, team0)
            actions = {i: NOOP for i in range(self.N)}
            actions[holder] = DROP
            _, r, _, _, _ = env_l.step(actions)
            loop_total += sum(r[j] for j in team0)
        env_l.close()

        # --- Scoring ---
        env_s = gym.make(self.ENV_ID)
        env_s.reset(seed=42)
        is_ = env_s.unwrapped
        ball_s, _ = _find_ball_on_grid(is_)
        is_.grid.set(13, 5, ball_s)
        _place(is_, 0, (12, 5), RIGHT)
        _reset_provenance(is_)
        _park_others(is_, {0})

        score_total = 0.0
        actions = {i: NOOP for i in range(self.N)}
        actions[0] = PICKUP
        _, r, _, _, _ = env_s.step(actions)
        score_total += sum(r[j] for j in team0)
        for _ in range(2):
            actions = {i: NOOP for i in range(self.N)}
            actions[0] = FORWARD
            _, r, _, _, _ = env_s.step(actions)
            score_total += sum(r[j] for j in team0)
        env_s.close()

        print(f"\n[AF 2v2] loop={loop_total:.3f}  score={score_total:.3f}")
        # v6.6.0: pickup bonus removed — pass-loop earns 0 (no pickup shaping)
        assert loop_total == pytest.approx(0.0, abs=0.05), (
            f"Pass-loop should earn ~0 without pickup bonus, got {loop_total:.3f}"
        )
        assert score_total >= 1.0, f"Score should be >=1.0, got {score_total:.3f}"
        assert score_total > loop_total  # scoring still beats passlooping

    def test_own_goal_scores_zero(self, env):
        """Walking into own end zone with ball -> no touchdown."""
        inner = env.unwrapped
        ball, _ = _find_ball_on_grid(inner)

        # Agent 0 (team 0). Own end zone = column 1.
        # Place at (3,5), ball at (2,5). Pickup, fwd to 2, fwd to 1.
        inner.grid.set(2, 5, ball)
        _place(inner, 0, (3, 5), LEFT)
        _reset_provenance(inner)
        _park_others(inner, {0})

        # Pickup
        actions = self._noop()
        actions[0] = PICKUP
        env.step(actions)
        assert inner.agents[0].state.carrying is not None

        # Forward to 2
        actions = self._noop()
        actions[0] = FORWARD
        env.step(actions)

        # Forward to 1 (own endzone)
        actions = self._noop()
        actions[0] = FORWARD
        _, r, _, _, _ = env.step(actions)

        assert r[0] < 0.5, f"Own-goal should not give TD, got {r[0]}"
        assert inner.agents[0].state.carrying is not None, (
            "Ball should NOT be consumed on own-goal"
        )


# =====================================================================
# Soccer 2vs2
# =====================================================================

class TestSoccerLiveActions:
    ENV_ID = "MosaicMultiGrid-S-2v2-IndAgObs-v1"
    N = 4

    @pytest.fixture
    def env(self):
        e = gym.make(self.ENV_ID)
        e.reset(seed=42)
        yield e
        e.close()

    def _noop(self):
        return {i: NOOP for i in range(self.N)}

    @pytest.mark.skip(reason="+0.1 pickup bonus removed in v6.5.0")
    def test_ground_pickup_earns_bonus(self, env):
        inner = env.unwrapped
        ball, _ = _find_ball_on_grid(inner)
        inner.grid.set(6, 5, ball)
        _place(inner, 0, (5, 5), RIGHT)
        _reset_provenance(inner)
        _park_others(inner, {0})

        actions = self._noop()
        actions[0] = PICKUP
        _, rewards, _, _, _ = env.step(actions)
        assert rewards[0] == pytest.approx(0.1, abs=EPS)

    def test_teleport_pass_receiver_gets_zero(self, env):
        inner = env.unwrapped
        ball, _ = _find_ball_on_grid(inner)
        _place(inner, 0, (7, 5), RIGHT, carrying=ball)
        _reset_provenance(inner)
        _park_others(inner, {0})

        env.step(self._noop())  # sync carrying

        actions = self._noop()
        actions[0] = DROP
        _, r, _, _, _ = env.step(actions)

        team1 = _team_agents(inner, 1)
        receiver = _find_holder(inner, team1)
        assert receiver is not None, "Teammate should have received ball"
        assert abs(r[receiver]) < EPS, (
            f"Same-team receive should get 0, agent {receiver} got {r[receiver]}"
        )

    @pytest.mark.skip(reason="+0.1 pickup bonus removed in v6.5.0")
    def test_pass_loop_10_hops_single_bonus(self, env):
        inner = env.unwrapped
        ball, _ = _find_ball_on_grid(inner)
        _place(inner, 0, (7, 5), RIGHT, carrying=ball)
        _reset_provenance(inner)
        _park_others(inner, {0})

        team1 = _team_agents(inner, 1)
        total = 0.0

        _, r, _, _, _ = env.step(self._noop())
        total += sum(r[j] for j in team1)

        for _ in range(10):
            holder = _find_holder(inner, team1)
            actions = self._noop()
            actions[holder] = DROP
            _, r, _, _, _ = env.step(actions)
            total += sum(r[j] for j in team1)

        assert total == pytest.approx(0.1, abs=EPS), (
            f"Soccer pass-loop should total 0.1, got {total}"
        )

    def test_walk_in_goal_earns_sparse(self, env):
        """Pickup + walk onto opponent goal -> +1.0 sparse for team."""
        inner = env.unwrapped
        ball, _ = _find_ball_on_grid(inner)

        # Agent 0 (team 1) scores at (14,5). Ball at (13,5), agent at (12,5).
        inner.grid.set(13, 5, ball)
        _place(inner, 0, (12, 5), RIGHT)
        _reset_provenance(inner)
        _park_others(inner, {0})

        team1 = _team_agents(inner, 1)
        total = 0.0

        # Pickup
        actions = self._noop()
        actions[0] = PICKUP
        _, r, _, _, _ = env.step(actions)
        total += sum(r[j] for j in team1)

        # Fwd to 13
        actions = self._noop()
        actions[0] = FORWARD
        _, r, _, _, _ = env.step(actions)
        total += sum(r[j] for j in team1)

        # Fwd to 14 (opponent goal) -> scores
        actions = self._noop()
        actions[0] = FORWARD
        _, r, _, _, _ = env.step(actions)
        total += sum(r[j] for j in team1)

        print(f"\n[Soccer 2v2] team1 scoring total = {total:.3f}")
        assert total > 1.0, f"Scoring should yield >1.0, got {total:.3f}"
        assert len(inner.goal_scored_by) > 0

    def test_own_goal_walk_in_zero(self, env):
        """Walking onto own goal with ball -> no score."""
        inner = env.unwrapped
        ball, _ = _find_ball_on_grid(inner)

        # Agent 0 (team 1). Own goal = (1,5), goal_index=1 (belongs to team 1).
        inner.grid.set(2, 5, ball)
        _place(inner, 0, (3, 5), LEFT)
        _reset_provenance(inner)
        _park_others(inner, {0})

        actions = self._noop()
        actions[0] = PICKUP
        env.step(actions)

        actions = self._noop()
        actions[0] = FORWARD
        env.step(actions)

        actions = self._noop()
        actions[0] = FORWARD
        _, r, _, _, _ = env.step(actions)

        team1 = _team_agents(inner, 1)
        t1r = sum(r[j] for j in team1)
        assert t1r < 0.5, f"Own-goal should not give sparse reward, got {t1r}"
        assert inner.agents[0].state.carrying is not None


# =====================================================================
# Basketball 3vs3
# =====================================================================

class TestBasketballLiveActions:
    ENV_ID = "MosaicMultiGrid-BB-3v3-IndAgObs-v1"
    N = 6

    @pytest.fixture
    def env(self):
        e = gym.make(self.ENV_ID)
        e.reset(seed=42)
        yield e
        e.close()

    def _noop(self):
        return {i: NOOP for i in range(self.N)}

    @pytest.mark.skip(reason="+0.1 pickup bonus removed in v6.5.0")
    def test_ground_pickup_earns_bonus(self, env):
        inner = env.unwrapped
        ball, _ = _find_ball_on_grid(inner)
        inner.grid.set(6, 5, ball)
        _place(inner, 0, (5, 5), RIGHT)
        _reset_provenance(inner)
        _park_others(inner, {0})

        actions = self._noop()
        actions[0] = PICKUP
        _, rewards, _, _, _ = env.step(actions)
        assert rewards[0] == pytest.approx(0.1, abs=EPS)

    def test_teleport_pass_receiver_gets_zero(self, env):
        inner = env.unwrapped
        ball, _ = _find_ball_on_grid(inner)
        _place(inner, 0, (9, 5), RIGHT, carrying=ball)
        _reset_provenance(inner)
        _park_others(inner, {0})

        env.step(self._noop())  # sync carrying

        actions = self._noop()
        actions[0] = DROP
        _, r, _, _, _ = env.step(actions)

        team1 = _team_agents(inner, 1)
        receiver = _find_holder(inner, team1)
        assert receiver is not None, "Teammate should have received ball"
        assert abs(r[receiver]) < EPS, (
            f"Same-team receive should get 0, agent {receiver} got {r[receiver]}"
        )

    @pytest.mark.skip(reason="+0.1 pickup bonus removed in v6.5.0")
    def test_pass_loop_10_hops_single_bonus(self, env):
        inner = env.unwrapped
        ball, _ = _find_ball_on_grid(inner)
        _place(inner, 0, (9, 5), RIGHT, carrying=ball)
        _reset_provenance(inner)
        _park_others(inner, {0})

        team1 = _team_agents(inner, 1)
        total = 0.0

        _, r, _, _, _ = env.step(self._noop())
        total += sum(r[j] for j in team1)

        for _ in range(10):
            holder = _find_holder(inner, team1)
            actions = self._noop()
            actions[holder] = DROP
            _, r, _, _, _ = env.step(actions)
            total += sum(r[j] for j in team1)

        assert total == pytest.approx(0.1, abs=EPS), (
            f"Basketball pass-loop should total 0.1, got {total}"
        )

    def test_scoring_earns_touchdown(self, env):
        """Pickup + walk onto opponent goal -> team reward."""
        inner = env.unwrapped
        ball, _ = _find_ball_on_grid(inner)

        # Agent 0 (team 1) scores at (17,5). Ball at (16,5), agent at (15,5).
        inner.grid.set(16, 5, ball)
        _place(inner, 0, (15, 5), RIGHT)
        _reset_provenance(inner)
        _park_others(inner, {0})

        team1 = _team_agents(inner, 1)
        total = 0.0

        # Pickup
        actions = self._noop()
        actions[0] = PICKUP
        _, r, _, _, _ = env.step(actions)
        total += sum(r[j] for j in team1)

        # Fwd to 16
        actions = self._noop()
        actions[0] = FORWARD
        _, r, _, _, _ = env.step(actions)
        total += sum(r[j] for j in team1)

        # Fwd to 17 (goal)
        actions = self._noop()
        actions[0] = FORWARD
        _, r, _, _, _ = env.step(actions)
        total += sum(r[j] for j in team1)

        print(f"\n[Basketball 3v3] team1 scoring total = {total:.3f}")
        assert total > 1.0, f"Scoring should yield >1.0, got {total:.3f}"
        assert len(inner.goal_scored_by) > 0

    def test_own_goal_walk_in_zero(self, env):
        """Walking onto own goal -> no score, ball not consumed."""
        inner = env.unwrapped
        ball, _ = _find_ball_on_grid(inner)

        # Agent 0 (team 1). Own goal = (1,5), goal_index=1.
        inner.grid.set(2, 5, ball)
        _place(inner, 0, (3, 5), LEFT)
        _reset_provenance(inner)
        _park_others(inner, {0})

        actions = self._noop()
        actions[0] = PICKUP
        env.step(actions)

        actions = self._noop()
        actions[0] = FORWARD
        env.step(actions)

        actions = self._noop()
        actions[0] = FORWARD
        _, r, _, _, _ = env.step(actions)

        team1 = _team_agents(inner, 1)
        t1r = sum(r[j] for j in team1)
        assert t1r < 0.5, f"Own-goal should not give sparse reward, got {t1r}"
        assert inner.agents[0].state.carrying is not None
