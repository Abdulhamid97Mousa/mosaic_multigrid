"""Basketball game environment for the MOSAIC multigrid package.

Teams score by dropping a ball at the opposing team's goal (an
:class:`~gym_multigrid.core.world_object.ObjectGoal`) placed on the
baseline.  In the base environment agents can **pass** the ball to an
adjacent teammate.  The IndAgObs variant upgrades this to **teleport
passing** -- the ball instantly transfers to a teammate anywhere on the
grid.  Agents can **steal** by picking up from an opponent who is
carrying.

Basketball 3vs3:
  - 19x11 total grid (17x9 playable area)
  - 3 agents per team (Green team 1 = left, Blue team 2 = right)
  - Goals on baseline cells: (1, 5) and (17, 5)
  - Same mechanics as Soccer IndAgObs (teleport pass, steal cooldown,
    ball respawn, first-to-N-goals termination)
"""
from __future__ import annotations

from ..base import MultiGridEnv
from ..core.agent import Agent
from ..core.constants import Color
from ..core.grid import Grid
from ..core.world_object import Ball, ObjectGoal
from ..rendering import render_basketball


class BasketballGameEnv(MultiGridEnv):
    """
    Multi-agent basketball game on a walled grid.

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
        Team assignment for each agent (e.g. ``[1, 1, 1, 2, 2, 2]``).
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

        if len(self.goal_pos) != len(self.goal_index):
            raise ValueError(
                f"goal_pos ({len(self.goal_pos)}) and goal_index "
                f"({len(self.goal_index)}) must have the same length"
            )
        if len(self.num_balls) != len(self.balls_index):
            raise ValueError(
                f"num_balls ({len(self.num_balls)}) and balls_index "
                f"({len(self.balls_index)}) must have the same length"
            )

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
        if agent.state.carrying is None:
            return

        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(*fwd_pos)

        # Try scoring on an ObjectGoal (must be opposing team's goal)
        if fwd_obj is not None and fwd_obj.type.value == 'objgoal':
            ball = agent.state.carrying
            if fwd_obj.target_type == ball.type.value:
                if ball.index in (0, fwd_obj.index):
                    if fwd_obj.index != agent.team_index:  # no own-goals
                        self._team_reward(agent.team_index, rewards, fwd_obj.reward)
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
# IndAgObs variant (Individual Agent Observations)
# -----------------------------------------------------------------------

class BasketballGameIndAgObsEnv(BasketballGameEnv):
    """IndAgObs Basketball with teleport passing, ball respawn, termination,
    and stealing cooldown.

    Key improvements over BasketballGameEnv:
    - Teleport passing: ball teleports to teammate anywhere on the grid
    - Ball respawns after each goal
    - Episode terminates when team scores N goals (first to win)
    - Dual cooldown on stealing (both stealer and victim)
    - STATE channel encoding for ball carrying (observability)
    """

    # Court rendering configuration (used by render_basketball)
    court_cfg = {
        'paint_depth': 3,
        'paint_half_h': 2,
        'three_pt_radius': 5.0,
        'center_radius': 1.5,
        'ft_circle_radius': 2,
    }

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
        goals_to_win: int = 2,
        steal_cooldown: int = 10,
    ):
        super().__init__(
            size=size,
            width=width,
            height=height,
            view_size=view_size,
            goal_pos=goal_pos,
            goal_index=goal_index,
            num_balls=num_balls,
            agents_index=agents_index,
            balls_index=balls_index,
            zero_sum=zero_sum,
            render_mode=render_mode,
            max_steps=max_steps,
        )
        self.goals_to_win = goals_to_win
        self.steal_cooldown = steal_cooldown
        self.team_scores: dict[int, int] = {}

    def get_full_render(self, highlight: bool, tile_size: int):
        """Override to use basketball-court rendering."""
        return render_basketball(self, tile_size)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        unique_teams = set(agent.team_index for agent in self.agents)
        self.team_scores = {team: 0 for team in unique_teams}
        for agent in self.agents:
            agent.action_cooldown = 0
        return obs, info

    def step(self, actions):
        for agent in self.agents:
            if hasattr(agent, 'action_cooldown') and agent.action_cooldown > 0:
                agent.action_cooldown -= 1
        return super().step(actions)

    def _handle_pickup(
        self,
        agent_index: int,
        agent: Agent,
        rewards: dict[int, float],
    ):
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
                if hasattr(agent, 'action_cooldown') and agent.action_cooldown > 0:
                    return
                if target.team_index != agent.team_index:
                    agent.state.carrying = target.state.carrying
                    target.state.carrying = None
                    agent.action_cooldown = self.steal_cooldown
                    target.action_cooldown = self.steal_cooldown

    def _handle_drop(
        self,
        agent_index: int,
        agent: Agent,
        rewards: dict[int, float],
    ):
        if agent.state.carrying is None:
            return

        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(*fwd_pos)

        # Priority 1: Score at opposing team's goal
        if fwd_obj is not None and fwd_obj.type.value == 'objgoal':
            ball = agent.state.carrying
            if fwd_obj.target_type == ball.type.value:
                if ball.index in (0, fwd_obj.index):
                    if fwd_obj.index != agent.team_index:  # no own-goals
                        self._team_reward(agent.team_index, rewards, fwd_obj.reward)
                        agent.state.carrying = None

                        # Respawn ball
                        new_ball = Ball(color=ball.color, index=ball.index)
                        self.place_obj(new_ball)

                        # Check win condition
                        self.team_scores[agent.team_index] += 1
                        if self.team_scores[agent.team_index] >= self.goals_to_win:
                            for a in self.agents:
                                a.state.terminated = True
                        return

        # Priority 2: Teleport pass to teammate
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
            return

        # Priority 3: Ground drop
        if fwd_obj is None and self._agent_at(fwd_pos) is None:
            self.grid.set(*fwd_pos, agent.state.carrying)
            agent.state.carrying.cur_pos = fwd_pos
            agent.state.carrying = None


# -----------------------------------------------------------------------
# Concrete 3vs3 variant
# -----------------------------------------------------------------------

class BasketballGame6HIndAgObsEnv19x11N3(BasketballGameIndAgObsEnv):
    """IndAgObs 3vs3 Basketball (19x11 total, 17x9 playable).

    Key features:
    - 19x11 grid (17x9 playable area)
    - 3vs3 teams: Green (team 1, left) vs Blue (team 2, right)
    - Goals at (1, 5) and (17, 5) -- baseline center
    - First to 2 goals wins
    - Ball respawns after each goal
    - 10-step dual cooldown on stealing
    - Teleport passing to any teammate
    - 200 max_steps (enough for 2-3 scoring attempts)
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('max_steps', 200)
        kwargs.setdefault('goals_to_win', 2)
        super().__init__(
            size=None,
            width=19,
            height=11,
            goal_pos=[[1, 5], [17, 5]],
            goal_index=[1, 2],
            num_balls=[1],
            agents_index=[1, 1, 1, 2, 2, 2],  # 3vs3
            balls_index=[0],  # Wildcard ball
            zero_sum=True,
            **kwargs,
        )
