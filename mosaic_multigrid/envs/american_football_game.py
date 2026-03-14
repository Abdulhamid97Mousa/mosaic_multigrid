"""
American Football environment for MOSAIC MultiGrid.

Simplified scoring mechanics:
- Agents score by walking INTO the opponent's end zone while carrying the ball (touchdown)
- No need to use 'drop' action to score
- Opponents can steal the ball using 'pickup' action
- Teammates can pass using 'drop' action (teleport pass)
- Agents cannot score on their own end zone

Grid: 16×11 (same as Soccer)
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
    Base American Football environment.

    Scoring: Walk into opponent's end zone while carrying ball (touchdown).
    End zones span full column height at each end of the field.
    Episode terminates when a team reaches goals_to_win touchdowns.
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
    ):
        self.num_balls = num_balls
        self.balls_index = balls_index or []
        self.balls_reward = balls_reward or []
        self.zero_sum = zero_sum
        self.goals_to_win = goals_to_win

        # Store end zone positions and team ownership
        # Will be populated in _gen_grid
        self.endzone_positions: dict[tuple[int, int], int] = {}
        
        # Track team scores for termination
        self.team_scores: dict[int, int] = {}

        agents_index = agents_index or []
        agents = [
            Agent(
                index=i,
                team_index=team,
                view_size=view_size,
                see_through_walls=True,
            )
            for i, team in enumerate(agents_index)
        ]

        super().__init__(
            agents=agents,
            width=width if width is not None else size,
            height=height if height is not None else size,
            max_steps=max_steps,
            see_through_walls=True,
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
            # Use grey for American Football (more consistent with sport)
            ball = Ball(Color.grey, ball_index)

            # Place in midfield area only (not in end zones)
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
        """Override to use American Football-style rendering instead of default grid tiles."""
        return render_american_football(self, tile_size)

    def reset(self, **kwargs):
        """Reset with team score tracking."""
        obs, info = super().reset(**kwargs)

        # Initialize team scores (one entry per unique team index)
        unique_teams = set(agent.team_index for agent in self.agents)
        self.team_scores = {team: 0 for team in unique_teams}

        return obs, info

    def _team_reward(
        self,
        scoring_team: int,
        rewards: dict[int, float],
        reward: float = 1.0,
    ):
        """
        Distribute reward to all agents on *scoring_team*.

        If ``self.zero_sum`` is ``True``, agents on other teams receive
        ``-reward``.
        """
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
        """
        Pickup with ball stealing.

        - Normal pickup: pick up a ball from the ground.
        - Steal: if an opponent in front is carrying a ball, take it.
        """
        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(*fwd_pos)

        # Normal pickup from ground
        if fwd_obj is not None and fwd_obj.can_pickup():
            if agent.state.carrying is None:
                agent.state.carrying = fwd_obj
                self.grid.set(*fwd_pos, None)
                return

        # Steal from another agent (only from opponents)
        target = self._agent_at(fwd_pos)
        if target is not None and target.state.carrying is not None:
            if agent.state.carrying is None and target.team_index != agent.team_index:
                agent.state.carrying = target.state.carrying
                target.state.carrying = None

    def _handle_drop(
        self,
        agent_index: int,
        agent: Agent,
        rewards: dict[int, float],
    ):
        """
        Drop with passing (no scoring via drop in American Football).

        - Pass: drop toward a teammate -> transfer ball.
        - Ground drop: drop onto empty cell.
        """
        if agent.state.carrying is None:
            return

        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(*fwd_pos)

        # Try passing to another agent (teammate only)
        target = self._agent_at(fwd_pos)
        if target is not None:
            if target.state.carrying is None and target.team_index == agent.team_index:
                target.state.carrying = agent.state.carrying
                agent.state.carrying = None
                return

        # Drop on empty ground
        if fwd_obj is None and self._agent_at(fwd_pos) is None:
            self.grid.set(*fwd_pos, agent.state.carrying)
            agent.state.carrying.cur_pos = fwd_pos
            agent.state.carrying = None

    def step(self, actions):
        """Override step to check for touchdowns after movement."""
        obs, rewards, terminated, truncated, info = super().step(actions)

        # Check for touchdowns: agent in opponent's end zone while carrying ball
        for agent in self.agents:
            if agent.state.carrying is not None:
                pos = agent.state.pos
                pos_tuple = (int(pos[0]), int(pos[1]))

                # Check if agent is in an end zone
                if pos_tuple in self.endzone_positions:
                    endzone_team = self.endzone_positions[pos_tuple]

                    # Check if it's the opponent's end zone (not own team's)
                    if endzone_team != agent.team_index:
                        # TOUCHDOWN!
                        ball = agent.state.carrying
                        ball_index = ball.index

                        # Award points to the scoring team
                        reward = self.balls_reward[ball_index] if ball_index < len(self.balls_reward) else 1.0
                        self._team_reward(agent.team_index, rewards, reward)

                        # Remove ball from agent
                        agent.state.carrying = None

                        # Respawn ball randomly on map (like Soccer)
                        self.place_obj(ball)

                        # Track team score and check win condition (goals_to_win)
                        self.team_scores[agent.team_index] += 1
                        
                        # Check if team reached goals_to_win (e.g., 2 goals)
                        if self.team_scores[agent.team_index] >= self.goals_to_win:
                            # Terminate episode for all agents
                            for a in self.agents:
                                a.state.terminated = True
                        break

        return obs, rewards, terminated, truncated, info


# ============================================================================
# Solo Variants (Single agent, no opponents)
# ============================================================================

class AmericanFootballSoloGreenEnv16x11(AmericanFootballEnv):
    """Solo Green agent (curriculum pre-training)."""
    def __init__(self, render_mode: str | None = None):
        super().__init__(
            width=16,
            height=11,
            num_balls=1,
            agents_index=[0],
            balls_index=[0],
            balls_reward=[1.0],
            goals_to_win=2,
            render_mode=render_mode,
        )


class AmericanFootballSoloBlueEnv16x11(AmericanFootballEnv):
    """Solo Blue agent (curriculum pre-training)."""
    def __init__(self, render_mode: str | None = None):
        super().__init__(
            width=16,
            height=11,
            num_balls=1,
            agents_index=[1],
            balls_index=[0],
            balls_reward=[1.0],
            goals_to_win=2,
            render_mode=render_mode,
        )


# ============================================================================
# 1v1 Variants
# ============================================================================

class AmericanFootball1v1Env16x11(AmericanFootballEnv):
    """1v1 American Football (Green vs Blue)."""
    def __init__(self, render_mode: str | None = None):
        super().__init__(
            width=16,
            height=11,
            num_balls=1,
            agents_index=[0, 1],
            balls_index=[0],
            balls_reward=[1.0],
            zero_sum=True,
            goals_to_win=2,
            render_mode=render_mode,
        )


# ============================================================================
# 2v2 Variants
# ============================================================================

class AmericanFootball2v2Env16x11(AmericanFootballEnv):
    """2v2 American Football (2 Green vs 2 Blue)."""
    def __init__(self, render_mode: str | None = None):
        super().__init__(
            width=16,
            height=11,
            num_balls=1,
            agents_index=[0, 0, 1, 1],
            balls_index=[0],
            balls_reward=[1.0],
            zero_sum=True,
            goals_to_win=2,
            render_mode=render_mode,
        )


# ============================================================================
# 3v3 Variants
# ============================================================================

class AmericanFootball3v3Env16x11(AmericanFootballEnv):
    """3v3 American Football (3 Green vs 3 Blue)."""
    def __init__(self, render_mode: str | None = None):
        super().__init__(
            width=16,
            height=11,
            num_balls=1,
            agents_index=[0, 0, 0, 1, 1, 1],
            balls_index=[0],
            balls_reward=[1.0],
            zero_sum=True,
            goals_to_win=2,
            render_mode=render_mode,
        )
