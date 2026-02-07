"""Collect game environment for the MOSAIC multigrid package.

Agents earn rewards for picking up balls that match their team index.
Balls with index ``0`` are wildcards and can be collected by any agent.
"""
from __future__ import annotations

from ..base import MultiGridEnv
from ..core.agent import Agent
from ..core.constants import Color
from ..core.grid import Grid
from ..core.world_object import Ball


class CollectGameEnv(MultiGridEnv):
    """
    Multi-agent ball collection game on a walled grid.

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
    num_balls : list[int]
        Number of balls to spawn per ball-type entry.
    agents_index : list[int]
        Team assignment for each agent (e.g. ``[1, 2, 3]``).
    balls_index : list[int]
        Team index per ball-type entry (0 = wildcard).
    balls_reward : list[float]
        Reward value per ball-type entry.
    zero_sum : bool
        If ``True``, collecting gives other teams a negative reward.
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
        view_size: int = 7,
        num_balls: list[int] | None = None,
        agents_index: list[int] | None = None,
        balls_index: list[int] | None = None,
        balls_reward: list[float] | None = None,
        zero_sum: bool = False,
        render_mode: str | None = None,
        max_steps: int = 10000,
    ):
        self.num_balls = num_balls or []
        self.balls_index = balls_index or []
        self.balls_reward = balls_reward or [1.0]
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

        # Place balls at random positions
        for number, ball_idx, reward in zip(
            self.num_balls, self.balls_index, self.balls_reward,
        ):
            for _ in range(number):
                ball = Ball(
                    color=Color.from_index(ball_idx % len(Color)),
                    index=ball_idx,
                    reward=reward,
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

        If ``self.zero_sum``, agents on other teams receive ``-reward``.
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
        Pickup with team-based ball matching.

        Balls with index ``0`` (wildcard) can be collected by anyone.
        Other balls can only be collected by agents on the matching team.
        The ball is consumed on pickup (not carried).
        """
        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(*fwd_pos)

        if fwd_obj is not None and fwd_obj.can_pickup():
            # Ball index 0 is wildcard, otherwise must match agent team
            if fwd_obj.index in (0, agent.team_index):
                self.grid.set(*fwd_pos, None)
                self._team_reward(agent.team_index, rewards, fwd_obj.reward)

    def _handle_drop(
        self,
        agent_index: int,
        agent: Agent,
        rewards: dict[int, float],
    ):
        """No drop action in collect game."""
        pass


# -----------------------------------------------------------------------
# Concrete variants
# -----------------------------------------------------------------------

class CollectGame4HEnv10x10N2(CollectGameEnv):
    """3 agents on separate teams, 10x10 grid, 5 wildcard balls, zero-sum."""

    def __init__(self, **kwargs):
        super().__init__(
            size=10,
            num_balls=[5],
            agents_index=[1, 2, 3],
            balls_index=[0],
            balls_reward=[1],
            zero_sum=True,
            **kwargs,
        )
