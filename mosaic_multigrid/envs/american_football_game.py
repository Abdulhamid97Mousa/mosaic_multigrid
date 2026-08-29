"""
American Football environment for MOSAIC MultiGrid.

v6.4.0: Added reward shaping (SMAC-style dense rewards), steal cooldown,
teleport passing, and event tracking.

Scoring mechanics:
- Agents score by walking INTO the opponent's end zone while carrying the ball
- No need to use 'drop' action to score
- Opponents can steal the ball using 'pickup' action (with cooldown)
- Teammates can pass using 'drop' action (teleport pass)
- Agents cannot score on their own end zone

Reward shaping (reward_shaping=True, default):
- +0.01 per column moved toward opponent's end zone while carrying
- +0.3 for stealing the ball from a carrying opponent
- Time-efficiency bonus: score_reward + (max_steps - step) × 0.05
- Set reward_shaping=False for sparse evaluation (touchdown only).

Grid: 16x11 (same as Soccer)
- End zones at columns 1 (Green's, Blue scores here) and 14 (Blue's, Green scores here)
- Playable midfield: columns 2-13, rows 1-9
"""

from __future__ import annotations

import numpy as np

from mosaic_multigrid.base import MultiGridEnv
from mosaic_multigrid.core import Agent, Ball, EndZone, Grid
from mosaic_multigrid.core.constants import Color, Type
from mosaic_multigrid.rendering import render_american_football


class AmericanFootballEnv(MultiGridEnv):
    """
    Base American Football environment with SMAC-style reward shaping.

    Scoring: Walk into opponent's end zone while carrying ball (touchdown).
    End zones span full column height at each end of the field.
    Episode terminates when a team reaches goals_to_win touchdowns.

    Reward shaping follows SMAC (Samvelyan et al., 2019):
    - Dense rewards by default for trainability
    - Sparse mode available for evaluation (reward_shaping=False)
    """

    def __init__(
        self,
        size: int | None = 16,
        width: int | None = None,
        height: int | None = None,
        view_size: int = 3,
        num_balls: int = 1,
        agents_index: list[int] | None = None,
        balls_index: list[int] | None = None,
        balls_reward: list[float] | None = None,
        zero_sum: bool = False,
        render_mode: str | None = None,
        max_steps: int = 300,
        goals_to_win: int = 2,
        reward_shaping: bool = True,
        steal_cooldown: int = 10,
    ):
        self.num_balls = num_balls
        self.balls_index = balls_index or []
        self.balls_reward = balls_reward or []
        self.zero_sum = zero_sum
        self.goals_to_win = goals_to_win
        self.reward_shaping = reward_shaping
        self.steal_cooldown = steal_cooldown

        # Store end zone positions and team ownership
        # Will be populated in _gen_grid
        self.endzone_positions: dict[tuple[int, int], int] = {}

        # Track team scores for termination
        self.team_scores: dict[int, int] = {}

        # Reward shaping state (per-agent tracking)
        self._prev_pos: dict[int, tuple[int, int]] = {}

        agents_index = agents_index or []
        agents = [
            Agent(
                index=i,
                team_index=team,
                view_size=view_size,
                see_through_walls=False,
            )
            for i, team in enumerate(agents_index)
        ]

        super().__init__(
            agents=agents,
            width=width if width is not None else size,
            height=height if height is not None else size,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=view_size,
            render_mode=render_mode,
        )

    def _gen_grid(self, width: int, height: int):
        """Generate American Football field with end zones."""
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Create end zones (full column height)
        # Green's end zone (column 1) - Blue scores here (team_index=0)
        for row in range(1, height - 1):
            endzone = EndZone(Color.green, team_index=0)
            self.put_obj(endzone, 1, row)
            self.endzone_positions[(1, row)] = 0

        # Blue's end zone (column 14) - Green scores here (team_index=1)
        for row in range(1, height - 1):
            endzone = EndZone(Color.blue, team_index=1)
            self.put_obj(endzone, width - 2, row)
            self.endzone_positions[(width - 2, row)] = 1

        # Place balls in midfield (columns 2-13)
        for ball_index in range(self.num_balls):
            ball = Ball(Color.grey, ball_index)
            self.place_obj(
                ball,
                top=(2, 1),
                size=(width - 4, height - 2),
                max_tries=100,
            )

        # Place agents in midfield
        for agent in self.agents:
            self.place_agent(
                agent,
                top=(2, 1),
                size=(width - 4, height - 2),
                max_tries=100,
            )

    def get_full_render(self, highlight: bool, tile_size: int):
        """Override to use American Football-style rendering."""
        return render_american_football(self, tile_size)

    def _target_endzone_pos(self, agent: Agent) -> tuple[int, int]:
        """Return (x, y) target for shaping rewards — centre of the scoring end zone.

        The end zone spans the full column height; (target_x, height // 2) is the
        geometric centre. Y movement toward the middle gets a small reward while
        carrying, but scoring still triggers on any cell of the end-zone column.
        """
        target_x = self.width - 2 if agent.team_index == 0 else 1
        return target_x, self.height // 2

    def _find_loose_ball_pos(self) -> tuple[int, int] | None:
        """Return (x, y) of the first loose ball on the grid, or None if all balls are carried."""
        for y in range(self.height):
            for x in range(self.width):
                obj = self.grid.get(x, y)
                if obj is not None and obj.type.value == 'ball':
                    return (x, y)
        return None

    def reset(self, **kwargs):
        """Reset with team score and reward shaping tracking."""
        obs, info = super().reset(**kwargs)

        # Initialize team scores
        unique_teams = set(agent.team_index for agent in self.agents)
        self.team_scores = {team: 0 for team in unique_teams}
        self.goal_scored_by: list[dict] = []
        self.passes_completed: list[dict] = []
        self.steals_completed: list[dict] = []

        # Initialize steal cooldowns
        for agent in self.agents:
            agent.action_cooldown = 0

        # Initialize reward shaping state
        for agent in self.agents:
            self._prev_pos[agent.index] = (int(agent.state.pos[0]), int(agent.state.pos[1]))

        # Ball provenance is empty at reset. Ground balls have no last
        # carrier, so the first pickup of any ball earns the +0.1 bonus
        # via the `last_team is None` branch in the shaping block below.
        self._ball_last_carrier_team = {}

        return obs, info

    def _team_reward(
        self,
        scoring_team: int,
        rewards: dict[int, float],
        reward: float = 1.0,
    ):
        """Distribute reward to all agents on scoring_team.
        If zero_sum, agents on other teams receive -reward."""
        for agent in self.agents:
            if agent.team_index == scoring_team:
                rewards[agent.index] += reward
            elif self.zero_sum:
                rewards[agent.index] -= reward

    def _handle_pickup(
        self,
        agent_index: int,
        agent: Agent,
        rewards: dict[int, float],
    ):
        """Pickup with ball stealing and dual cooldown (like Soccer IndAgObs)."""
        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(*fwd_pos)

        # Normal pickup from ground (no cooldown check)
        if fwd_obj is not None and fwd_obj.can_pickup():
            if agent.state.carrying is None:
                agent.state.carrying = fwd_obj
                self.grid.set(*fwd_pos, None)
                return

        # Steal from opponent (with cooldown)
        target = self._agent_at(fwd_pos)
        if target is not None and target.state.carrying is not None:
            if agent.state.carrying is None:
                # Check cooldown
                if hasattr(agent, 'action_cooldown') and agent.action_cooldown > 0:
                    return
                # Only steal from opponents
                if target.team_index != agent.team_index:
                    agent.state.carrying = target.state.carrying
                    target.state.carrying = None
                    # Dual cooldown
                    agent.action_cooldown = self.steal_cooldown
                    target.action_cooldown = self.steal_cooldown
                    # Track steal
                    self.steals_completed.append({
                        "step": self.step_count,
                        "stealer": agent.index,
                        "victim": target.index,
                        "team": agent.team_index,
                    })

    def _handle_drop(
        self,
        agent_index: int,
        agent: Agent,
        rewards: dict[int, float],
    ):
        """Drop with teleport passing (like Soccer/Basketball IndAgObs)."""
        if agent.state.carrying is None:
            return

        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(*fwd_pos)

        # Priority 1: Teleport pass to teammate
        teammates = [
            a for a in self.agents
            if a.team_index == agent.team_index
            and a.index != agent.index
            and a.state.carrying is None
            and not a.state.terminated
        ]
        if teammates:
            target = teammates[self.np_random.integers(len(teammates))]
            target.state.carrying = agent.state.carrying
            agent.state.carrying = None
            self.passes_completed.append({
                "step": self.step_count,
                "passer": agent.index,
                "receiver": target.index,
                "team": agent.team_index,
            })
            return

        # Priority 2: Drop on empty ground
        if fwd_obj is None and self._agent_at(fwd_pos) is None:
            self.grid.set(*fwd_pos, agent.state.carrying)
            agent.state.carrying.cur_pos = fwd_pos
            agent.state.carrying = None

    def step(self, actions):
        """Step with cooldowns, touchdown detection, reward shaping, and telemetry."""
        # Decrement cooldowns
        for agent in self.agents:
            if hasattr(agent, 'action_cooldown') and agent.action_cooldown > 0:
                agent.action_cooldown -= 1

        goals_before = len(self.goal_scored_by)
        passes_before = len(self.passes_completed)
        steals_before = len(self.steals_completed)

        obs, rewards, terminated, truncated, info = super().step(actions)

        # Check for touchdowns: agent in opponent's end zone while carrying
        for agent in self.agents:
            if agent.state.carrying is not None:
                pos = agent.state.pos
                pos_tuple = (int(pos[0]), int(pos[1]))

                if pos_tuple in self.endzone_positions:
                    endzone_team = self.endzone_positions[pos_tuple]

                    if endzone_team != agent.team_index:
                        # TOUCHDOWN!
                        ball = agent.state.carrying
                        ball_index = ball.index

                        reward = self.balls_reward[ball_index] if ball_index < len(self.balls_reward) else 1.0
                        time_bonus = (self.max_steps - self.step_count) * 0.05
                        self._team_reward(agent.team_index, rewards, reward + time_bonus)

                        self.goal_scored_by.append({
                            "step": self.step_count,
                            "scorer": agent.index,
                            "team": agent.team_index,
                        })

                        agent.state.carrying = None
                        self.place_obj(ball)

                        self.team_scores[agent.team_index] += 1
                        if self.team_scores[agent.team_index] >= self.goals_to_win:
                            for a in self.agents:
                                a.state.terminated = True
                        break

        # Propagate walk-in termination into the returned terminated dict so
        # callers see the correct value in the same step it is triggered.
        if any(a.state.terminated for a in self.agents):
            terminated = {a.index: bool(a.state.terminated) for a in self.agents}

        # ---- Reward shaping ----
        if self.reward_shaping:
            # +0.3 steal bonus — agents that took the ball from a carrying opponent
            steal_set = {s["stealer"] for s in self.steals_completed[steals_before:]}
            loose_ball = self._find_loose_ball_pos()

            for agent in self.agents:
                carrying = agent.state.carrying is not None
                cx, cy = int(agent.state.pos[0]), int(agent.state.pos[1])
                px, py = self._prev_pos.get(agent.index, (cx, cy))
                gx, gy = self._target_endzone_pos(agent)

                if agent.index in steal_set:
                    rewards[agent.index] += 0.3

                # +0.01 × Δdist toward the next objective:
                # carrying → end-zone centre;  not carrying → loose ball
                if carrying:
                    prev_dist = abs(px - gx) + abs(py - gy)
                    curr_dist = abs(cx - gx) + abs(cy - gy)
                    rewards[agent.index] += 0.01 * (prev_dist - curr_dist)
                elif loose_ball is not None:
                    bx, by = loose_ball
                    prev_dist = abs(px - bx) + abs(py - by)
                    curr_dist = abs(cx - bx) + abs(cy - by)
                    rewards[agent.index] += 0.01 * (prev_dist - curr_dist)

                self._prev_pos[agent.index] = (cx, cy)

        # ---- Telemetry (per-agent info injection) ----
        for agent in self.agents:
            info[agent.index]["position"] = tuple(int(c) for c in agent.state.pos)
            info[agent.index]["carrying"] = agent.state.carrying is not None

        if len(self.goal_scored_by) > goals_before:
            latest = self.goal_scored_by[-1]
            for aid in info:
                info[aid]["goal_scored_by"] = latest

        if len(self.passes_completed) > passes_before:
            latest = self.passes_completed[-1]
            for aid in info:
                info[aid]["pass_completed"] = latest

        if len(self.steals_completed) > steals_before:
            latest = self.steals_completed[-1]
            for aid in info:
                info[aid]["steal_completed"] = latest

        return obs, rewards, terminated, truncated, info


# ============================================================================
# Solo Variants (Single agent, no opponents)
# ============================================================================

class AmericanFootballSoloGreenEnv16x11(AmericanFootballEnv):
    """Solo Green agent (curriculum pre-training)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0],
            balls_index=[0], balls_reward=[1.0],
            goals_to_win=2, render_mode=render_mode,
            reward_shaping=reward_shaping,
        )


class AmericanFootballSoloBlueEnv16x11(AmericanFootballEnv):
    """Solo Blue agent (curriculum pre-training)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[1],
            balls_index=[0], balls_reward=[1.0],
            goals_to_win=2, render_mode=render_mode,
            reward_shaping=reward_shaping,
        )


# ============================================================================
# 1v1 Variants
# ============================================================================

class AmericanFootball1v1Env16x11(AmericanFootballEnv):
    """1v1 American Football (Green vs Blue)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode,
            reward_shaping=reward_shaping,
        )


# ============================================================================
# 2v2 Variants
# ============================================================================

class AmericanFootball2v2Env16x11(AmericanFootballEnv):
    """2v2 American Football (2 Green vs 2 Blue)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode,
            reward_shaping=reward_shaping,
        )


# ============================================================================
# 3v3 Variants
# ============================================================================

class AmericanFootball3v3Env16x11(AmericanFootballEnv):
    """3v3 American Football (3 Green vs 3 Blue)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 0, 1, 1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode,
            reward_shaping=reward_shaping,
        )


# ============================================================================
# One-sided multi-agent variants (cooperative, no opponent)
# ============================================================================

class AmericanFootballGreen2v0Env16x11(AmericanFootballEnv):
    """2 Green agents, 0 Blue — cooperative AF practice without opposition."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootballBlue0v2Env16x11(AmericanFootballEnv):
    """0 Green agents, 2 Blue — cooperative AF practice without opposition."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootballGreen3v0Env16x11(AmericanFootballEnv):
    """3 Green agents, 0 Blue — cooperative AF practice without opposition."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 0],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootballBlue0v3Env16x11(AmericanFootballEnv):
    """0 Green agents, 3 Blue — cooperative AF practice without opposition."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[1, 1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


# ============================================================================
# 4v0 / 0v4 / 5v0 / 0v5 / 6v0 / 0v6 one-sided variants (cooperative)
# ============================================================================

class AmericanFootballGreen4v0Env16x11(AmericanFootballEnv):
    """4 Green agents, 0 Blue — cooperative AF practice without opposition."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 0, 0],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootballBlue0v4Env16x11(AmericanFootballEnv):
    """0 Green agents, 4 Blue — cooperative AF practice without opposition."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[1, 1, 1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootballGreen5v0Env16x11(AmericanFootballEnv):
    """5 Green agents, 0 Blue — cooperative AF practice without opposition."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 0, 0, 0],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootballBlue0v5Env16x11(AmericanFootballEnv):
    """0 Green agents, 5 Blue — cooperative AF practice without opposition."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[1, 1, 1, 1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootballGreen6v0Env16x11(AmericanFootballEnv):
    """6 Green agents, 0 Blue — cooperative AF practice without opposition."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 0, 0, 0, 0],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootballBlue0v6Env16x11(AmericanFootballEnv):
    """0 Green agents, 6 Blue — cooperative AF practice without opposition."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[1, 1, 1, 1, 1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


# ============================================================================
# 4v4 Variant
# ============================================================================

class AmericanFootball4v4Env16x11(AmericanFootballEnv):
    """4v4 American Football (4 Green vs 4 Blue)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 0, 0, 1, 1, 1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode,
            reward_shaping=reward_shaping,
        )


class AmericanFootball1v2Env16x11(AmericanFootballEnv):
    """1 Green vs 2 Blue asymmetric competitive American Football (16x11)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootball2v1Env16x11(AmericanFootballEnv):
    """2 Green vs 1 Blue asymmetric competitive American Football (16x11)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootball1v3Env16x11(AmericanFootballEnv):
    """1 Green vs 3 Blue asymmetric competitive American Football (16x11)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 1, 1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootball3v1Env16x11(AmericanFootballEnv):
    """3 Green vs 1 Blue asymmetric competitive American Football (16x11)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 0, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootball1v4Env16x11(AmericanFootballEnv):
    """1 Green vs 4 Blue asymmetric competitive American Football (16x11)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 1, 1, 1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootball4v1Env16x11(AmericanFootballEnv):
    """4 Green vs 1 Blue asymmetric competitive American Football (16x11)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 0, 0, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootball2v3Env16x11(AmericanFootballEnv):
    """2 Green vs 3 Blue asymmetric competitive American Football (16x11)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 1, 1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootball3v2Env16x11(AmericanFootballEnv):
    """3 Green vs 2 Blue asymmetric competitive American Football (16x11)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 0, 1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootball2v4Env16x11(AmericanFootballEnv):
    """2 Green vs 4 Blue asymmetric competitive American Football (16x11)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 1, 1, 1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )


class AmericanFootball4v2Env16x11(AmericanFootballEnv):
    """4 Green vs 2 Blue asymmetric competitive American Football (16x11)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 0, 0, 1, 1],
            balls_index=[0], balls_reward=[1.0],
            zero_sum=False, goals_to_win=2,
            render_mode=render_mode, reward_shaping=reward_shaping,
        )
