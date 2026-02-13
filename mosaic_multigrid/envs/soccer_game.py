"""Soccer game environment for the MOSAIC multigrid package.

Teams score by dropping a ball at the opposing team's goal (an
:class:`~gym_multigrid.core.world_object.ObjectGoal`).  In the base
environment agents can **pass** the ball to an adjacent teammate.  The
IndAgObs variant upgrades this to **teleport passing** -- the ball
instantly transfers to a teammate anywhere on the grid.  Agents can
**steal** by picking up from an opponent who is carrying.
"""
from __future__ import annotations

from ..base import MultiGridEnv
from ..core.agent import Agent
from ..core.constants import Color
from ..core.grid import Grid
from ..core.world_object import Ball, ObjectGoal
from ..rendering import render_fifa


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

        # Try scoring on an ObjectGoal (must be opposing team's goal)
        if fwd_obj is not None and fwd_obj.type.value == 'objgoal':
            ball = agent.state.carrying
            if fwd_obj.target_type == ball.type.value:
                # Ball index 0 is wildcard (can score at any goal)
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


# -----------------------------------------------------------------------
# IndAgObs variants (Individual Agent Observations) - Fixed for RL training
# -----------------------------------------------------------------------

class SoccerGameIndAgObsEnv(SoccerGameEnv):
    """IndAgObs Soccer game with teleport passing, ball respawn, termination,
    and stealing cooldown.

    This is the RECOMMENDED version for RL training. Fixes critical bugs:

    Key improvements over SoccerGameEnv:
    - Teleport passing: ball teleports to teammate anywhere on the grid
    - Ball respawns after each goal (no disappearing ball bug)
    - Episode terminates when team scores 2 goals (first to win)
    - Dual cooldown on stealing (10 steps for both stealer and victim)
    - STATE channel encoding for ball carrying (observability fixed)
    - ~50x faster training (200 vs 10,000 steps average)
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
        goals_to_win: int = 2,
        steal_cooldown: int = 10,
    ):
        """
        Parameters
        ----------
        goals_to_win : int
            First team to score this many goals wins (default: 2).
        steal_cooldown : int
            Steps both stealer and victim wait after steal (default: 10).
        """
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
        self.goal_scored_by: list[dict] = []
        self.passes_completed: list[dict] = []
        self.steals_completed: list[dict] = []

    def get_full_render(self, highlight: bool, tile_size: int):
        """Override to use FIFA-style rendering instead of default grid tiles."""
        return render_fifa(self, tile_size)

    def reset(self, **kwargs):
        """Reset with team score and cooldown tracking."""
        # Call parent reset first
        obs, info = super().reset(**kwargs)

        # Initialize team scores (one entry per unique team index)
        unique_teams = set(agent.team_index for agent in self.agents)
        self.team_scores = {team: 0 for team in unique_teams}
        self.goal_scored_by = []
        self.passes_completed = []
        self.steals_completed = []

        # Initialize action cooldowns for all agents
        for agent in self.agents:
            agent.action_cooldown = 0

        return obs, info

    def step(self, actions):
        """Step with cooldown decrements and event tracking in info."""
        # Decrement action cooldowns before processing actions
        for agent in self.agents:
            if hasattr(agent, 'action_cooldown') and agent.action_cooldown > 0:
                agent.action_cooldown -= 1

        goals_before = len(self.goal_scored_by)
        passes_before = len(self.passes_completed)
        steals_before = len(self.steals_completed)
        obs, rewards, terms, truncs, infos = super().step(actions)

        # Inject goal event into info dict when a goal was scored this step
        if len(self.goal_scored_by) > goals_before:
            latest_goal = self.goal_scored_by[-1]
            for agent_id in infos:
                infos[agent_id]["goal_scored_by"] = latest_goal

        # Inject pass event into info dict when a pass was completed this step
        if len(self.passes_completed) > passes_before:
            latest_pass = self.passes_completed[-1]
            for agent_id in infos:
                infos[agent_id]["pass_completed"] = latest_pass

        # Inject steal event into info dict when a steal occurred this step
        if len(self.steals_completed) > steals_before:
            latest_steal = self.steals_completed[-1]
            for agent_id in infos:
                infos[agent_id]["steal_completed"] = latest_steal

        return obs, rewards, terms, truncs, infos

    def _handle_pickup(
        self,
        agent_index: int,
        agent: Agent,
        rewards: dict[int, float],
    ):
        """
        Pickup with ball stealing and dual cooldown.

        When stealing from opponent:
        - Both stealer and victim get cooldown (cannot pickup for N steps)
        - Prevents ping-pong stealing exploit
        """
        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(*fwd_pos)

        # Normal pickup from ground (no cooldown check)
        if fwd_obj is not None and fwd_obj.can_pickup():
            if agent.state.carrying is None:
                agent.state.carrying = fwd_obj
                self.grid.set(*fwd_pos, None)
                return

        # Steal from another agent (with cooldown check)
        target = self._agent_at(fwd_pos)
        if target is not None and target.state.carrying is not None:
            if agent.state.carrying is None:
                # Check if agent is in cooldown
                if hasattr(agent, 'action_cooldown') and agent.action_cooldown > 0:
                    return  # Cannot steal yet (recovering from previous tackle)

                # Check if teams are different (can only steal from opponent)
                if target.team_index != agent.team_index:
                    # Steal successful!
                    agent.state.carrying = target.state.carrying
                    target.state.carrying = None

                    # Track the steal
                    self.steals_completed.append({
                        "step": self.step_count,
                        "stealer": agent.index,
                        "victim": target.index,
                        "team": agent.team_index,
                    })

                    # NEW: Apply dual cooldown (both stealer and victim)
                    agent.action_cooldown = self.steal_cooldown
                    target.action_cooldown = self.steal_cooldown

    def _handle_drop(
        self,
        agent_index: int,
        agent: Agent,
        rewards: dict[int, float],
    ):
        """
        Drop with teleport passing, scoring, ball respawn, and termination.

        Priority chain:
        1. **Score** -- drop ball on matching ObjectGoal -> team reward + respawn.
        2. **Teleport pass** -- ball teleports to a random teammate anywhere
           on the grid (teammate must not already be carrying).
        3. **Ground drop** -- drop onto empty cell in front (fallback).

        Teleport passing replaces the old 1-cell adjacency handoff. The ball
        transfers instantly regardless of distance, creating attack/defense
        dynamics when combined with the stealing mechanic.
        """
        if agent.state.carrying is None:
            return

        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(*fwd_pos)

        # Priority 1: Try scoring on opposing team's ObjectGoal
        if fwd_obj is not None and fwd_obj.type.value == 'objgoal':
            ball = agent.state.carrying
            if fwd_obj.target_type == ball.type.value:
                # Ball index 0 is wildcard (can score at any goal)
                if ball.index in (0, fwd_obj.index):
                    if fwd_obj.index != agent.team_index:  # no own-goals
                        self._team_reward(agent.team_index, rewards, fwd_obj.reward)
                        agent.state.carrying = None

                        # Track which agent scored
                        self.goal_scored_by.append({
                            "step": self.step_count,
                            "scorer": agent.index,
                            "team": agent.team_index,
                        })

                        # Respawn ball with same color and team index
                        new_ball = Ball(
                            color=ball.color,
                            index=ball.index,
                        )
                        self.place_obj(new_ball)

                        # Track team score and check win condition
                        self.team_scores[agent.team_index] += 1
                        if self.team_scores[agent.team_index] >= self.goals_to_win:
                            for a in self.agents:
                                a.state.terminated = True
                        return

        # Priority 2: Teleport pass to a teammate (anywhere on the grid)
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

            # Track the completed pass
            self.passes_completed.append({
                "step": self.step_count,
                "passer": agent.index,
                "receiver": target.index,
                "team": agent.team_index,
            })
            return

        # Priority 3: Drop on empty ground (fallback)
        if fwd_obj is None and self._agent_at(fwd_pos) is None:
            self.grid.set(*fwd_pos, agent.state.carrying)
            agent.state.carrying.cur_pos = fwd_pos
            agent.state.carrying = None


class SoccerGame4HIndAgObsEnv16x11N2(SoccerGameIndAgObsEnv):
    """IndAgObs 2v2 Soccer with FIFA aspect ratio (16x11 total, 14x9 playable).

    RECOMMENDED for RL training. Key features:
    - 16x11 grid (FIFA 105m x 68m = 1.54 ratio)
    - 14x9 playable area (126 cells)
    - Goals at (1,5) and (14,5) - vertical center
    - First to 2 goals wins
    - Ball respawns after each goal
    - 10-step dual cooldown on stealing
    - 200 max_steps (enough for 2-3 scoring attempts)
    """

    def __init__(self, **kwargs):
        # Set default max_steps for RL training (can be overridden)
        kwargs.setdefault('max_steps', 200)
        kwargs.setdefault('goals_to_win', 2)
        super().__init__(
            size=None,
            width=16,  # FIFA-style width
            height=11,  # FIFA-style height
            goal_pos=[[1, 5], [14, 5]],  # Goals at vertical center
            goal_index=[1, 2],
            num_balls=[1],
            agents_index=[1, 1, 2, 2],  # 2v2 teams
            balls_index=[0],  # Wildcard ball
            zero_sum=False,
            **kwargs,
        )


class SoccerGame2HIndAgObsEnv16x11N2(SoccerGameIndAgObsEnv):
    """IndAgObs 1v1 Soccer with FIFA aspect ratio (16x11 total, 14x9 playable).

    Simplified 1v1 variant for faster training iteration. Same grid and
    mechanics as 2v2 but with 1 agent per team. Teleport passing becomes
    a no-op (no teammates), making this pure individual play.

    Key features:
    - 16x11 grid (same FIFA ratio as 2v2)
    - 1 agent per team (Green vs Red)
    - Goals at (1,5) and (14,5) - vertical center
    - First to 2 goals wins
    - Ball respawns after each goal
    - 10-step dual cooldown on stealing
    - 200 max_steps
    """

    def __init__(self, **kwargs):
        # Set default max_steps for RL training (can be overridden)
        kwargs.setdefault('max_steps', 200)
        kwargs.setdefault('goals_to_win', 2)
        super().__init__(
            size=None,
            width=16,  # FIFA-style width
            height=11,  # FIFA-style height
            goal_pos=[[1, 5], [14, 5]],  # Goals at vertical center
            goal_index=[1, 2],
            num_balls=[1],
            agents_index=[1, 2],  # 1v1 teams (Green vs Red)
            balls_index=[0],  # Wildcard ball
            zero_sum=False,
            **kwargs,
        )
