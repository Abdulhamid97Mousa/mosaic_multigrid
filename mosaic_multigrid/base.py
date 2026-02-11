"""Base multi-agent gridworld environment for the MOSAIC multigrid package.

Follows the Gymnasium API (5-tuple returns, dict-keyed per agent) with
reproducible seeding via ``self.np_random``. Rendering is handled via
pygame when ``render_mode='human'``.

Subclasses must implement :meth:`_gen_grid` and may override the action
hooks (``_handle_forward``, ``_handle_pickup``, ``_handle_drop``,
``_handle_toggle``) to customize game logic (see Soccer / Collect).
"""
from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import pygame
import pygame.freetype

from abc import ABC, abstractmethod
from collections import defaultdict
from gymnasium import spaces
from itertools import repeat
from numpy.typing import NDArray as ndarray
from typing import Any, Callable, Iterable, Literal, SupportsFloat

from .core.actions import Action
from .core.agent import Agent, AgentState
from .core.constants import Type, TILE_PIXELS
from .core.grid import Grid
from .core.mission import MissionSpace
from .core.world_object import WorldObj
from .utils.obs import gen_obs_grid_encoding
from .utils.random import RandomMixin


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AgentID = int
ObsType = dict[str, Any]


# ---------------------------------------------------------------------------
# Base Environment
# ---------------------------------------------------------------------------

class MultiGridEnv(gym.Env, RandomMixin, ABC):
    """
    Base class for multi-agent 2D gridworld environments.

    :Agents:

        The environment can be configured with any fixed number of agents.
        Agents are represented by :class:`.Agent` instances and identified
        by their index, from ``0`` to ``len(env.agents) - 1``.

    :Observation Space:

        A Dict mapping agent index to an observation dict containing:

            * ``image`` : ndarray of shape (view_size, view_size, WorldObj.dim)
            * ``direction`` : int  (0=right, 1=down, 2=left, 3=up)
            * ``mission`` : Mission

    :Action Space:

        A Dict mapping agent index to a Discrete action space.

    Attributes
    ----------
    agents : list[Agent]
        List of agents in the environment.
    grid : Grid
        Environment grid.
    observation_space : spaces.Dict
        Joint observation space of all agents.
    action_space : spaces.Dict
        Joint action space of all agents.
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 20,
    }

    def __init__(
        self,
        mission_space: MissionSpace | str = 'maximize reward',
        agents: Iterable[Agent] | int = 1,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        allow_agent_overlap: bool = True,
        joint_reward: bool = False,
        success_termination_mode: Literal['any', 'all'] = 'any',
        failure_termination_mode: Literal['any', 'all'] = 'all',
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """
        Parameters
        ----------
        mission_space : MissionSpace or str
            Space of mission strings (agent instructions).
        agents : int or Iterable[Agent]
            Number of agents, or pre-constructed Agent instances.
        grid_size : int or None
            Size of the grid (sets both width and height).
        width : int or None
            Width of the grid (if grid_size is not provided).
        height : int or None
            Height of the grid (if grid_size is not provided).
        max_steps : int
            Maximum number of steps per episode.
        see_through_walls : bool
            Whether agents can see through walls.
        agent_view_size : int
            Size of each agent's partial view (must be odd).
        allow_agent_overlap : bool
            Whether agents can occupy the same cell.
        joint_reward : bool
            Whether all agents receive the same reward.
        success_termination_mode : 'any' or 'all'
            Terminate when any/all agents succeed.
        failure_termination_mode : 'any' or 'all'
            Terminate when any/all agents fail.
        render_mode : str or None
            'human' for pygame window, 'rgb_array' for frame array.
        screen_size : int or None
            Screen size in pixels for human rendering.
        highlight : bool
            Whether to highlight agent view areas when rendering.
        tile_size : int
            Tile size in pixels.
        agent_pov : bool
            Whether to render agent's POV instead of full environment.
        """
        gym.Env.__init__(self)
        RandomMixin.__init__(self, self.np_random)

        # Mission space
        if isinstance(mission_space, str):
            self.mission_space = MissionSpace.from_string(mission_space)
        else:
            self.mission_space = mission_space

        # Grid dimensions
        width, height = (grid_size, grid_size) if grid_size else (width, height)
        assert width is not None and height is not None
        self.width, self.height = width, height
        self.grid: Grid = Grid(width, height)

        # Agents
        if isinstance(agents, int):
            self.num_agents = agents
            self.agent_states = AgentState(agents)
            self.agents: list[Agent] = []
            for i in range(agents):
                agent = Agent(
                    index=i,
                    mission_space=self.mission_space,
                    view_size=agent_view_size,
                    see_through_walls=see_through_walls,
                )
                agent.state = self.agent_states[i]
                self.agents.append(agent)
        elif isinstance(agents, Iterable):
            agents = list(agents)
            assert {a.index for a in agents} == set(range(len(agents)))
            self.num_agents = len(agents)
            self.agent_states = AgentState(self.num_agents)
            self.agents: list[Agent] = sorted(agents, key=lambda a: a.index)
            for agent in self.agents:
                self.agent_states[agent.index] = agent.state
                agent.state = self.agent_states[agent.index]
        else:
            raise ValueError(f'Invalid argument for agents: {agents}')

        # Action enumeration
        self.actions = Action

        # Reward range
        self.reward_range = (0, 1)

        assert isinstance(max_steps, int)
        self.max_steps = max_steps

        # Rendering
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov
        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None

        # Behavior flags
        self.allow_agent_overlap = allow_agent_overlap
        self.joint_reward = joint_reward
        self.success_termination_mode = success_termination_mode
        self.failure_termination_mode = failure_termination_mode

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    @property
    def observation_space(self) -> spaces.Dict:
        """Joint observation space of all agents."""
        return spaces.Dict({
            agent.index: agent.observation_space
            for agent in self.agents
        })

    @property
    def action_space(self) -> spaces.Dict:
        """Joint action space of all agents."""
        return spaces.Dict({
            agent.index: agent.action_space
            for agent in self.agents
        })

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @abstractmethod
    def _gen_grid(self, width: int, height: int):
        """
        Generate the grid for a new episode.

        Must set ``self.grid`` and place agents with valid positions
        and directions.

        Parameters
        ----------
        width : int
            Width of the grid.
        height : int
            Height of the grid.
        """

    def reset(
        self,
        seed: int | None = None,
        **kwargs,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict[str, Any]]]:
        """
        Reset the environment.

        Parameters
        ----------
        seed : int or None
            Seed for the random number generator.

        Returns
        -------
        observations : dict[AgentID, ObsType]
            Observation for each agent.
        infos : dict[AgentID, dict]
            Additional information for each agent.
        """
        super().reset(seed=seed, **kwargs)

        # Re-sync RandomMixin with the (possibly new) np_random generator.
        # gym.Env.reset(seed=N) replaces self.np_random, but the mixin
        # still holds the old generator captured during __init__.
        RandomMixin.__init__(self, self.np_random)

        # Reset agents
        self.mission_space.seed(seed)
        self.mission = self.mission_space.sample()
        self.agent_states = AgentState(self.num_agents)
        for agent in self.agents:
            agent.state = self.agent_states[agent.index]
            agent.reset(mission=self.mission)

        # Generate a new random grid
        self._gen_grid(self.width, self.height)

        # Verify _gen_grid set valid positions/directions
        assert np.all(self.agent_states.pos >= 0)
        assert np.all(self.agent_states.dir >= 0)

        # Check agents don't overlap with solid objects
        for agent in self.agents:
            start_cell = self.grid.get(*agent.state.pos)
            assert start_cell is None or start_cell.can_overlap()

        self.step_count = 0

        observations = self.gen_obs()

        if self.render_mode == 'human':
            self.render()

        return observations, defaultdict(dict)

    def step(
        self,
        actions: dict[AgentID, Action],
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, SupportsFloat],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        """
        Run one timestep of the environment's dynamics.

        Parameters
        ----------
        actions : dict[AgentID, Action]
            Action for each agent acting at this timestep.

        Returns
        -------
        observations : dict[AgentID, ObsType]
            Observation for each agent.
        rewards : dict[AgentID, SupportsFloat]
            Reward for each agent.
        terminations : dict[AgentID, bool]
            Whether the episode has terminated for each agent.
        truncations : dict[AgentID, bool]
            Whether the episode has been truncated for each agent.
        infos : dict[AgentID, dict]
            Additional information for each agent.
        """
        self.step_count += 1
        rewards = self.handle_actions(actions)

        observations = self.gen_obs()
        terminations = dict(enumerate(self.agent_states.terminated))
        truncated = self.step_count >= self.max_steps
        truncations = dict(enumerate(repeat(truncated, self.num_agents)))

        if self.render_mode == 'human':
            self.render()

        return observations, rewards, terminations, truncations, defaultdict(dict)

    # ------------------------------------------------------------------
    # Observation generation
    # ------------------------------------------------------------------

    def gen_obs(self) -> dict[AgentID, ObsType]:
        """
        Generate partially-observable observations for each agent.

        Returns
        -------
        observations : dict[AgentID, ObsType]
            Mapping from agent ID to observation dict.
        """
        direction = self.agent_states.dir
        image = gen_obs_grid_encoding(
            self.grid.state,
            self.agent_states,
            self.agents[0].view_size,
            self.agents[0].see_through_walls,
        )

        observations = {}
        for i in range(self.num_agents):
            observations[i] = {
                'image': image[i],
                'direction': direction[i],
                'mission': self.agents[i].mission,
            }

        return observations

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------

    def handle_actions(
        self,
        actions: dict[AgentID, Action],
    ) -> dict[AgentID, SupportsFloat]:
        """
        Handle actions taken by agents.

        Uses ``self.np_random`` for action order randomization (fixes
        the reproducibility bug in the original gym-multigrid).

        Parameters
        ----------
        actions : dict[AgentID, Action]
            Action for each agent.

        Returns
        -------
        rewards : dict[AgentID, SupportsFloat]
            Reward for each agent.
        """
        rewards = {i: 0 for i in range(self.num_agents)}

        # Randomize agent action order (reproducible)
        if self.num_agents == 1:
            order = (0,)
        else:
            order = self.np_random.random(size=self.num_agents).argsort()

        for i in order:
            if i not in actions:
                continue

            agent, action = self.agents[i], actions[i]

            if agent.state.terminated:
                continue

            # Rotate left
            if action == Action.left:
                agent.state.dir = (agent.state.dir - 1) % 4

            # Rotate right
            elif action == Action.right:
                agent.state.dir = (agent.state.dir + 1) % 4

            # Move forward
            elif action == Action.forward:
                self._handle_forward(i, agent, rewards)

            # Pick up an object
            elif action == Action.pickup:
                self._handle_pickup(i, agent, rewards)

            # Drop an object
            elif action == Action.drop:
                self._handle_drop(i, agent, rewards)

            # Toggle/activate an object
            elif action == Action.toggle:
                self._handle_toggle(i, agent, rewards)

            # Done (no-op)
            elif action == Action.done:
                pass

            else:
                raise ValueError(f'Unknown action: {action}')

        return rewards

    # ------------------------------------------------------------------
    # Action hooks (override in subclasses for custom game logic)
    # ------------------------------------------------------------------

    def _handle_forward(
        self,
        agent_index: int,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
    ):
        """
        Handle forward movement for an agent.

        Override in subclasses for custom movement logic (e.g. Soccer
        switch interactions).

        Parameters
        ----------
        agent_index : int
            Index of the acting agent.
        agent : Agent
            The acting agent.
        rewards : dict
            Reward dictionary to update.
        """
        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(*fwd_pos)

        if fwd_obj is None or fwd_obj.can_overlap():
            if not self.allow_agent_overlap:
                agent_present = np.bitwise_and.reduce(
                    self.agent_states.pos == fwd_pos, axis=1).any()
                if agent_present:
                    return

            agent.state.pos = fwd_pos
            if fwd_obj is not None:
                if fwd_obj.type == Type.goal:
                    self.on_success(agent, rewards, {})
                if fwd_obj.type == Type.lava:
                    self.on_failure(agent, rewards, {})

    def _handle_pickup(
        self,
        agent_index: int,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
    ):
        """
        Handle pickup action for an agent.

        Override in subclasses for custom pickup logic (e.g. Soccer ball
        stealing, Collect ball pickup with team rewards).

        Parameters
        ----------
        agent_index : int
            Index of the acting agent.
        agent : Agent
            The acting agent.
        rewards : dict
            Reward dictionary to update.
        """
        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(*fwd_pos)

        if fwd_obj is not None and fwd_obj.can_pickup():
            if agent.state.carrying is None:
                agent.state.carrying = fwd_obj
                self.grid.set(*fwd_pos, None)

    def _handle_drop(
        self,
        agent_index: int,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
    ):
        """
        Handle drop action for an agent.

        Override in subclasses for custom drop logic (e.g. Soccer passing
        and scoring).

        Parameters
        ----------
        agent_index : int
            Index of the acting agent.
        agent : Agent
            The acting agent.
        rewards : dict
            Reward dictionary to update.
        """
        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(*fwd_pos)

        if agent.state.carrying and fwd_obj is None:
            agent_present = np.bitwise_and.reduce(
                self.agent_states.pos == fwd_pos, axis=1).any()
            if not agent_present:
                self.grid.set(*fwd_pos, agent.state.carrying)
                agent.state.carrying.cur_pos = fwd_pos
                agent.state.carrying = None

    def _handle_toggle(
        self,
        agent_index: int,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
    ):
        """
        Handle toggle action for an agent.

        Parameters
        ----------
        agent_index : int
            Index of the acting agent.
        agent : Agent
            The acting agent.
        rewards : dict
            Reward dictionary to update.
        """
        fwd_pos = agent.front_pos
        fwd_obj = self.grid.get(*fwd_pos)

        if fwd_obj is not None:
            fwd_obj.toggle(self, agent, fwd_pos)

    # ------------------------------------------------------------------
    # Termination / reward callbacks
    # ------------------------------------------------------------------

    def on_success(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool],
    ):
        """
        Callback for when an agent completes its mission.

        Parameters
        ----------
        agent : Agent
            Agent that succeeded.
        rewards : dict
            Reward dictionary to update.
        terminations : dict
            Termination dictionary to update.
        """
        if self.success_termination_mode == 'any':
            self.agent_states.terminated = True
            for i in range(self.num_agents):
                terminations[i] = True
        else:
            agent.state.terminated = True
            terminations[agent.index] = True

        if self.joint_reward:
            for i in range(self.num_agents):
                rewards[i] = self._reward()
        else:
            rewards[agent.index] = self._reward()

    def on_failure(
        self,
        agent: Agent,
        rewards: dict[AgentID, SupportsFloat],
        terminations: dict[AgentID, bool],
    ):
        """
        Callback for when an agent fails its mission.

        Parameters
        ----------
        agent : Agent
            Agent that failed.
        rewards : dict
            Reward dictionary to update.
        terminations : dict
            Termination dictionary to update.
        """
        if self.failure_termination_mode == 'any':
            self.agent_states.terminated = True
            for i in range(self.num_agents):
                terminations[i] = True
        else:
            agent.state.terminated = True
            terminations[agent.index] = True

    def _reward(self) -> float:
        """Compute the reward to be given upon success."""
        return 1 - 0.9 * (self.step_count / self.max_steps)

    def is_done(self) -> bool:
        """Return whether the current episode is finished."""
        truncated = self.step_count >= self.max_steps
        return truncated or all(self.agent_states.terminated)

    # ------------------------------------------------------------------
    # Agent query helpers
    # ------------------------------------------------------------------

    def _agent_at(self, pos) -> Agent | None:
        """
        Return the agent occupying *pos*, or ``None``.

        Parameters
        ----------
        pos : array-like of shape (2,)
            (x, y) grid coordinate.
        """
        pos = tuple(pos)
        for agent in self.agents:
            if not agent.state.terminated and tuple(agent.state.pos) == pos:
                return agent
        return None

    # ------------------------------------------------------------------
    # Object / agent placement utilities
    # ------------------------------------------------------------------

    def place_obj(
        self,
        obj: WorldObj | None,
        top: tuple[int, int] | None = None,
        size: tuple[int, int] | None = None,
        reject_fn: Callable[[MultiGridEnv, tuple[int, int]], bool] | None = None,
        max_tries=math.inf,
    ) -> tuple[int, int]:
        """
        Place an object at a random empty position in the grid.

        Parameters
        ----------
        obj : WorldObj or None
            Object to place.
        top : tuple[int, int] or None
            Top-left corner of the placement rectangle.
        size : tuple[int, int] or None
            Width and height of the placement rectangle.
        reject_fn : Callable or None
            Function to filter out positions.
        max_tries : int or float
            Maximum placement attempts.
        """
        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0
        while True:
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')
            num_tries += 1

            pos = (
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
            )

            # Don't place on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place where agents are
            if np.bitwise_and.reduce(
                    self.agent_states.pos == pos, axis=1).any():
                continue

            # Custom rejection
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj: WorldObj, i: int, j: int):
        """Put an object at a specific position in the grid."""
        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
        self,
        agent: Agent,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=math.inf,
    ) -> tuple[int, int]:
        """Set an agent's starting point at an empty position."""
        agent.state.pos = (-1, -1)
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        agent.state.pos = pos

        if rand_dir:
            agent.state.dir = self._rand_int(0, 4)

        return pos

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def get_full_render(
        self, highlight: bool, tile_size: int,
    ) -> ndarray[np.uint8]:
        """
        Render the full grid with optional agent-view highlighting.

        Parameters
        ----------
        highlight : bool
            Whether to highlight agents' fields of view.
        tile_size : int
            Tile size in pixels.
        """
        # Compute agent visibility masks
        obs_shape = self.agents[0].observation_space['image'].shape[:-1]
        vis_masks = np.zeros((self.num_agents, *obs_shape), dtype=bool)
        for i, agent_obs in self.gen_obs().items():
            vis_masks[i] = (
                agent_obs['image'][..., 0] != Type.unseen.to_index())

        # Build highlight mask
        highlight_mask = np.zeros((self.width, self.height), dtype=bool)

        for agent in self.agents:
            f_vec = agent.state.dir.to_vec()
            r_vec = np.array((-f_vec[1], f_vec[0]))
            top_left = (
                agent.state.pos
                + f_vec * (agent.view_size - 1)
                - r_vec * (agent.view_size // 2)
            )

            for vis_j in range(agent.view_size):
                for vis_i in range(agent.view_size):
                    if not vis_masks[agent.index][vis_i, vis_j]:
                        continue

                    abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                    if abs_i < 0 or abs_i >= self.width:
                        continue
                    if abs_j < 0 or abs_j >= self.height:
                        continue

                    highlight_mask[int(abs_i), int(abs_j)] = True

        return self.grid.render(
            tile_size,
            agents=self.agents,
            highlight_mask=highlight_mask if highlight else None,
        )

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ) -> ndarray[np.uint8]:
        """
        Return an RGB image of the whole environment.

        Parameters
        ----------
        highlight : bool
            Whether to highlight agents' fields of view.
        tile_size : int
            Tile size in pixels.
        agent_pov : bool
            Whether to render agent's POV.
        """
        return self.get_full_render(highlight, tile_size)

    def render(self):
        """Render the environment based on ``self.render_mode``."""
        img = self.get_frame(self.highlight, self.tile_size)

        if self.render_mode == 'human':
            img = np.transpose(img, axes=(1, 0, 2))
            screen_size = (
                self.screen_size * min(img.shape[0] / img.shape[1], 1.0),
                self.screen_size * min(img.shape[1] / img.shape[0], 1.0),
            )
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                pygame.display.set_caption(
                    f'mosaic_multigrid - {self.__class__.__name__}')
                self.window = pygame.display.set_mode(screen_size)
            if self.clock is None:
                self.clock = pygame.time.Clock()

            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset),
                 int(surf.get_size()[1] + offset)))
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))
            bg = pygame.transform.smoothscale(bg, screen_size)

            font_size = 22
            text = str(self.mission)
            font = pygame.freetype.SysFont(
                pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata['render_fps'])
            pygame.display.flip()

        elif self.render_mode == 'rgb_array':
            return img

    def close(self):
        """Close the rendering window."""
        if self.window:
            pygame.quit()

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __str__(self):
        OBJECT_TO_STR = {
            'wall': 'W', 'floor': 'F', 'door': 'D', 'key': 'K',
            'ball': 'A', 'box': 'B', 'goal': 'G', 'lava': 'V',
        }
        AGENT_DIR_TO_STR = {0: '>', 1: 'V', 2: '<', 3: '^'}

        location_to_agent = {
            tuple(agent.pos): agent for agent in self.agents}

        output = ''
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                if (i, j) in location_to_agent:
                    output += 2 * AGENT_DIR_TO_STR[
                        location_to_agent[i, j].dir]
                    continue

                tile = self.grid.get(i, j)
                if tile is None:
                    output += '  '
                elif tile.type == 'door':
                    if tile.is_open:
                        output += '__'
                    elif tile.is_locked:
                        output += 'L' + tile.color[0].upper()
                    else:
                        output += 'D' + tile.color[0].upper()
                else:
                    output += OBJECT_TO_STR.get(
                        str(tile.type), '??') + str(tile.color)[0].upper()

            if j < self.grid.height - 1:
                output += '\n'

        return output
