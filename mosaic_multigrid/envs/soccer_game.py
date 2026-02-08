"""Soccer game environment for the MOSAIC multigrid package.

Teams score by dropping a ball at the opposing team's goal (an
:class:`~gym_multigrid.core.world_object.ObjectGoal`).  Agents can
**pass** the ball to a teammate by dropping toward an agent, and
**steal** by picking up from an opponent who is carrying.
"""
from __future__ import annotations

from ..base import MultiGridEnv
from ..core.agent import Agent
from ..core.constants import Color
from ..core.grid import Grid
from ..core.world_object import Ball, ObjectGoal


class SoccerGameEnv(MultiGridEnv):
    """
    Multi-agent soccer game on a walled grid.

    Parameters
    ----------
    size : int or None
        Grid size (if square).
    width : int or None
        Grid width.
    height : int or None
        Grid height.
    view_size : int
        Agent partial observation size.
    goal_pos : list[list[int]]
        ``[x, y]`` positions for each goal.
    goal_index : list[int]
        Team index that each goal belongs to.
    num_balls : list[int]
        Number of balls to spawn per ball-type entry.
    agents_index : list[int]
        Team assignment for each agent (e.g. ``[1, 1, 2, 2]``).
    balls_index : list[int]
        Team index per ball-type entry (0 = wildcard, any team can use).
    zero_sum : bool
        If ``True``, scoring gives the opposing team a negative reward.
    render_mode : str or None
        ``'human'`` or ``'rgb_array'``.
    max_steps : int
        Maximum steps per episode.
    """

    def __init__(
        self,
        size: int | None = 10,
        width: int | None = None,
        height: int | None = None,
        view_size: int = 3,
        goal_pos: list[list[int]] | None = None,
        goal_index: list[int] | None = None,
        num_balls: list[int] | None = None,
        agents_index: list[int] | None = None,
        balls_index: list[int] | None = None,
        zero_sum: bool = False,
        render_mode: str | None = None,
        max_steps: int = 10000,
    ):
        self.goal_pos = goal_pos or []
        self.goal_index = goal_index or []
        self.num_balls = num_balls or []
        self.balls_index = balls_index or []
        self.zero_sum = zero_sum

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
            grid_size=size,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=view_size,
            render_mode=render_mode,
        )

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Place goals at fixed positions
        for pos, team_idx in zip(self.goal_pos, self.goal_index):
            goal = ObjectGoal(
                color=Color.from_index(team_idx % len(Color)),
                target_type='ball',
                index=team_idx,
                reward=1.0,
            )
            self.place_obj(goal, top=pos, size=[1, 1])

        # Place balls at random positions
        for number, ball_idx in zip(self.num_balls, self.balls_index):
            for _ in range(number):
                ball = Ball(
                    color=Color.from_index(ball_idx % len(Color)),
                    index=ball_idx,
                )
                self.place_obj(ball)

        # Randomize agent positions
        for agent in self.agents:
            self.place_agent(agent)

    # ------------------------------------------------------------------
    # Team reward
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Action overrides
    # ------------------------------------------------------------------

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

        # Steal from another agent
        target = self._agent_at(fwd_pos)
        if target is not None and target.state.carrying is not None:
            if agent.state.carrying is None:
                agent.state.carrying = target.state.carrying
                target.state.carrying = None

    def _handle_drop(
        self,
        agent_index: int,
        agent: Agent,
        rewards: dict[int, float],
    ):
        """
        Drop with passing and scoring.

        - Score: drop ball on matching ObjectGoal -> team reward.
        - Pass: drop toward a teammate -> transfer ball.
        - Ground drop: drop onto empty cell.
        """
        if agent.state.carrying is None:
            return

        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(*fwd_pos)

        # Try scoring on an ObjectGoal
        if fwd_obj is not None and fwd_obj.type.value == 'objgoal':
            ball = agent.state.carrying
            if fwd_obj.target_type == ball.type.value:
                # Ball index 0 is wildcard (can score at any goal)
                if ball.index in (0, fwd_obj.index):
                    self._team_reward(fwd_obj.index, rewards, fwd_obj.reward)
                    agent.state.carrying = None
                    return

        # Try passing to another agent
        target = self._agent_at(fwd_pos)
        if target is not None:
            if target.state.carrying is None:
                target.state.carrying = agent.state.carrying
                agent.state.carrying = None
                return

        # Drop on empty ground
        if fwd_obj is None and self._agent_at(fwd_pos) is None:
            self.grid.set(*fwd_pos, agent.state.carrying)
            agent.state.carrying.cur_pos = fwd_pos
            agent.state.carrying = None


# -----------------------------------------------------------------------
# Concrete variants
# -----------------------------------------------------------------------

class SoccerGame4HEnv10x15N2(SoccerGameEnv):
    """4 agents (2v2), 15x10 grid, 1 wildcard ball, zero-sum."""

    def __init__(self, **kwargs):
        super().__init__(
            size=None,
            height=10,
            width=15,
            goal_pos=[[1, 5], [13, 5]],
            goal_index=[1, 2],
            num_balls=[1],
            agents_index=[1, 1, 2, 2],
            balls_index=[0],
            zero_sum=True,
            **kwargs,
        )
