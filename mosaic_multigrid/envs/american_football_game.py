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
- +0.1 for picking up the ball (first time per possession)
- +0.01 per column moved toward opponent's end zone while carrying
- -0.01 per column moved away from opponent's end zone while carrying
- Follows SMAC's dense reward pattern for trainability.
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
        timeout_penalty: float = -1.0,
        global_obs: bool = False,
    ):
        self.num_balls = num_balls
        self.balls_index = balls_index or []
        self.balls_reward = balls_reward or []
        self.zero_sum = zero_sum
        self.goals_to_win = goals_to_win
        self.reward_shaping = reward_shaping
        self.steal_cooldown = steal_cooldown
        self.timeout_penalty = timeout_penalty

        # Store end zone positions and team ownership
        # Will be populated in _gen_grid
        self.endzone_positions: dict[tuple[int, int], int] = {}

        # Track team scores for termination
        self.team_scores: dict[int, int] = {}

        # Reward shaping state (per-agent tracking)
        self._prev_carrying: dict[int, bool] = {}
        self._prev_x: dict[int, int] = {}
        # Ball provenance tracking (v6.5.0: blocks pass-loop pickup exploit).
        # Maps ball_index -> team_index of the last agent who carried it.
        # Used by the pickup-shaping rule: a pickup only earns the +0.1 bonus
        # when the ball's last carrier was NOT on the picker's team (steal
        # from opponent, or ground ball that was never carried or was last
        # carried by the opposing team). Same-team pass-to-pickup cycles
        # earn zero bonus, closing the reward-hacking loop documented in
        # RELEASE_NOTES.md and tests/test_pickup_shaping_no_exploit.py.
        self._ball_last_carrier_team: dict[int, int] = {}
        # Consecutive pass counter per ball (v6.5.0).
        # First pass earns +0.1; second+ consecutive pass earns -0.2 penalty.
        # Resets when: ball hits ground, ball stolen, goal scored, episode reset.
        self._ball_pass_count: dict[int, int] = {}

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
            global_obs=global_obs,
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

    def _target_endzone_x(self, agent: Agent) -> int:
        """Return the x-column of the end zone where this agent scores."""
        # team_index=0 (Green) scores at column width-2 (Blue's end zone)
        # team_index=1 (Blue) scores at column 1 (Green's end zone)
        if agent.team_index == 0:
            return self.width - 2
        else:
            return 1

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
            self._prev_carrying[agent.index] = agent.state.carrying is not None
            self._prev_x[agent.index] = int(agent.state.pos[0])

        # Ball provenance is empty at reset. Ground balls have no last
        # carrier, so the first pickup of any ball earns the +0.1 bonus
        # via the `last_team is None` branch in the shaping block below.
        self._ball_last_carrier_team = {}
        self._ball_pass_count = {}

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
                    ball = target.state.carrying
                    agent.state.carrying = ball
                    target.state.carrying = None
                    # Dual cooldown
                    agent.action_cooldown = self.steal_cooldown
                    target.action_cooldown = self.steal_cooldown
                    # Steal resets the pass chain — new possession starts fresh.
                    self._ball_pass_count[ball.index] = 0
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
            ball = agent.state.carrying
            ball_idx = ball.index
            target = teammates[self.np_random.integers(len(teammates))]
            target.state.carrying = ball
            agent.state.carrying = None

            # Pass reward / double-pass penalty.
            # First consecutive pass: +0.1 (reward teamwork).
            # Second+ consecutive pass without a score or possession reset:
            # -0.2 (net negative for A→B→A chain: +0.1 - 0.2 = -0.1).
            count = self._ball_pass_count.get(ball_idx, 0) + 1
            self._ball_pass_count[ball_idx] = count
            if count == 1:
                rewards[agent.index] += 0.1   # first pass: reward teamwork
            elif count == 2:
                rewards[agent.index] -= 0.2   # second pass: punish A→B→A bounce
            # count >= 3: silence (no reward, no penalty). Prevents random-policy
            # teleport cascades from stacking unlimited -0.2 per hop and drowning
            # all positive shaping signals during early exploration.

            self.passes_completed.append({
                "step": self.step_count,
                "passer": agent.index,
                "receiver": target.index,
                "team": agent.team_index,
                "pass_count": count,
            })
            return

        # Priority 2: Drop on empty ground — resets pass chain.
        if fwd_obj is None and self._agent_at(fwd_pos) is None:
            ball = agent.state.carrying
            self._ball_pass_count[ball.index] = 0
            self.grid.set(*fwd_pos, ball)
            ball.cur_pos = fwd_pos
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
                        self._team_reward(agent.team_index, rewards, reward)

                        self.goal_scored_by.append({
                            "step": self.step_count,
                            "scorer": agent.index,
                            "team": agent.team_index,
                        })

                        agent.state.carrying = None
                        self._ball_pass_count[ball_index] = 0  # new possession after reset
                        self.place_obj(ball)

                        self.team_scores[agent.team_index] += 1
                        if self.team_scores[agent.team_index] >= self.goals_to_win:
                            for a in self.agents:
                                a.state.terminated = True
                        break

        # ---- Reward shaping (SMAC-style dense rewards) ----
        if self.reward_shaping:
            # Proximity reward: build ball positions once per step.
            # Carried balls are at the carrier's position; ground balls
            # are found by scanning the (small) grid. Used below to give
            # non-carrying agents +0.01/step when within Manhattan dist 3
            # of any ball — the standard SMAC "move toward objective" signal
            # that bootstraps exploration before the first pickup event.
            _ball_positions = []
            for _a in self.agents:
                if _a.state.carrying is not None:
                    _ball_positions.append(
                        (int(_a.state.pos[0]), int(_a.state.pos[1]))
                    )
            if not _ball_positions:
                from mosaic_multigrid.core.world_object import Ball as _Ball
                for _x in range(self.width):
                    for _y in range(self.height):
                        if isinstance(self.grid.get(_x, _y), _Ball):
                            _ball_positions.append((_x, _y))

            for agent in self.agents:
                carrying = agent.state.carrying is not None
                curr_x = int(agent.state.pos[0])
                prev_carrying = self._prev_carrying.get(agent.index, False)
                prev_x = self._prev_x.get(agent.index, curr_x)
                target_x = self._target_endzone_x(agent)

                # +0.1 for picking up ball, ONLY when the ball was last
                # held by an opposing team (steal) or by no one (ground
                # ball). Same-team pickup (teammate pass) earns zero so
                # agents cannot farm reward via pass-loop cycles.
                # See RELEASE_NOTES.md v6.5.0 for the exploit analysis.
                if carrying and not prev_carrying:
                    ball = agent.state.carrying
                    ball_idx = ball.index
                    last_team = self._ball_last_carrier_team.get(ball_idx)
                    if last_team is None or last_team != agent.team_index:
                        rewards[agent.index] += 0.1

                # Distance shaping while carrying
                if carrying:
                    if target_x > prev_x:
                        # Green scores right: positive dx = progress
                        dx = curr_x - prev_x
                    else:
                        # Blue scores left: negative dx = progress
                        dx = prev_x - curr_x
                    rewards[agent.index] += 0.05 * dx

                # Proximity reward: approach ball before pickup.
                # Fires only when not carrying and a ball is within
                # Manhattan distance 3. Creates gradient toward the ball
                # from the start of the episode so agents learn to seek it
                # before any scoring signal is ever observed.
                if not carrying and _ball_positions:
                    ax, ay = curr_x, int(agent.state.pos[1])
                    min_dist = min(
                        abs(ax - bx) + abs(ay - by)
                        for bx, by in _ball_positions
                    )
                    if min_dist <= 3:
                        rewards[agent.index] += 0.01

                # Update ball provenance: whichever team currently holds
                # each ball is recorded as its last carrier. A subsequent
                # same-team pickup after a teammate pass will see
                # last_team == agent.team_index and skip the +0.1 bonus.
                if carrying:
                    ball_idx = agent.state.carrying.index
                    self._ball_last_carrier_team[ball_idx] = agent.team_index

                # Update tracking state
                self._prev_carrying[agent.index] = carrying
                self._prev_x[agent.index] = curr_x

        # ---- Timeout penalty: punish all agents if episode ends with no goal ----
        # Applied when the time limit is reached (step_count >= max_steps) and
        # neither team scored a single touchdown. This removes the local optimum
        # where agents trickle +0.01*dx shaping rewards without ever committing
        # to score. Uses step_count directly (scalar) because the base class
        # returns truncated as a per-agent dict, which is always truthy.
        # Penalty fires when the episode times out WITHOUT a winner.
        # Checks team_scores directly so scoring 1 goal (but not 2) still
        # triggers the penalty — agents must win (2 goals), not just score once.
        _winner = next(
            (t for t, s in self.team_scores.items() if s >= self.goals_to_win),
            None,
        )
        if self.step_count >= self.max_steps and _winner is None:
            for agent in self.agents:
                rewards[agent.index] += self.timeout_penalty

        # Re-sync terminated dict: base class builds it before our scoring check
        # runs, so any a.state.terminated=True set by the win condition above
        # is invisible to the caller unless we propagate it here.
        for a in self.agents:
            if a.state.terminated:
                terminated[a.index] = True

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
                 reward_shaping: bool = True, global_obs: bool = False):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0],
            balls_index=[0], balls_reward=[15.0],
            goals_to_win=2, render_mode=render_mode,
            reward_shaping=reward_shaping,
            global_obs=global_obs,
        )


class AmericanFootballSoloBlueEnv16x11(AmericanFootballEnv):
    """Solo Blue agent (curriculum pre-training)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True, global_obs: bool = False):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[1],
            balls_index=[0], balls_reward=[15.0],
            goals_to_win=2, render_mode=render_mode,
            reward_shaping=reward_shaping,
            global_obs=global_obs,
        )


# ============================================================================
# 1v1 Variants
# ============================================================================

class AmericanFootball1v1Env16x11(AmericanFootballEnv):
    """1v1 American Football (Green vs Blue)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True, global_obs: bool = False):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 1],
            balls_index=[0], balls_reward=[15.0],
            zero_sum=True, goals_to_win=2,
            render_mode=render_mode,
            reward_shaping=reward_shaping,
            global_obs=global_obs,
        )


# ============================================================================
# 2v2 Variants
# ============================================================================

class AmericanFootball2v2Env16x11(AmericanFootballEnv):
    """2v2 American Football (2 Green vs 2 Blue)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True, global_obs: bool = False):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 1, 1],
            balls_index=[0], balls_reward=[15.0],
            zero_sum=True, goals_to_win=2,
            render_mode=render_mode,
            reward_shaping=reward_shaping,
            global_obs=global_obs,
        )


# ============================================================================
# 3v3 Variants
# ============================================================================

class AmericanFootball3v3Env16x11(AmericanFootballEnv):
    """3v3 American Football (3 Green vs 3 Blue)."""
    def __init__(self, render_mode: str | None = None, view_size: int = 3,
                 reward_shaping: bool = True, global_obs: bool = False):
        super().__init__(
            width=16, height=11, view_size=view_size,
            num_balls=1, agents_index=[0, 0, 0, 1, 1, 1],
            balls_index=[0], balls_reward=[15.0],
            zero_sum=True, goals_to_win=2,
            render_mode=render_mode,
            reward_shaping=reward_shaping,
            global_obs=global_obs,
        )
