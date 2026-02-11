"""FIFA-style rendering backend for Soccer environments.

Replaces the default tile-based grid rendering with a soccer-pitch overlay:
alternating grass stripes, white field markings, directional agent triangles,
and per-agent field-of-view highlights.

All drawing uses pygame primitives. The public entry point ``render_fifa()``
returns an ``ndarray[uint8]`` of shape ``(height_px, width_px, 3)`` that
plugs directly into the ``get_full_render()`` pipeline in ``base.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pygame
import pygame.freetype

from ..core.constants import Type


# -----------------------------------------------------------------------
# Color palette
# -----------------------------------------------------------------------

GRASS_LIGHT = (76, 153, 60)
GRASS_DARK = (58, 128, 45)
GRASS_WALL = (45, 100, 35)
FIELD_LINE = (255, 255, 255)
GOAL_NET = (220, 220, 220)

TEAM_COLORS = {
    1: (30, 200, 60),
    2: (60, 80, 220),
}
TEAM_OUTLINE = {
    1: (15, 120, 30),
    2: (30, 40, 140),
}

BALL_COLOR = (255, 60, 60)
BALL_OUTLINE = (180, 30, 30)


# -----------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------

def _draw_grass(surface: pygame.Surface, tile: int, gw: int, gh: int):
    """Alternating vertical grass stripes with darkened wall border."""
    for col in range(gw):
        color = GRASS_LIGHT if col % 2 == 0 else GRASS_DARK
        pygame.draw.rect(surface, color, (col * tile, 0, tile, gh * tile))

    # Thin black grid lines inside playable area only
    for col in range(1, gw):
        lx = col * tile
        pygame.draw.line(surface, (0, 0, 0),
                         (lx, 1 * tile), (lx, (gh - 1) * tile), 1)
    for row in range(1, gh):
        ly = row * tile
        pygame.draw.line(surface, (0, 0, 0),
                         (1 * tile, ly), ((gw - 1) * tile, ly), 1)

    # Darken wall border cells
    for col in range(gw):
        for row in range(gh):
            if col == 0 or col == gw - 1 or row == 0 or row == gh - 1:
                pygame.draw.rect(
                    surface, GRASS_WALL,
                    (col * tile, row * tile, tile, tile))


def _draw_field_markings(surface: pygame.Surface, tile: int,
                         gw: int, gh: int):
    """White FIFA-style field markings (boundary, center, penalty areas)."""
    lw = max(2, tile // 16)

    left = 1 * tile
    right = (gw - 1) * tile
    top = 1 * tile
    bottom = (gh - 1) * tile
    field_w = right - left
    field_h = bottom - top
    cx = left + field_w // 2
    cy = top + field_h // 2

    # Outer boundary
    pygame.draw.rect(surface, FIELD_LINE,
                     (left, top, field_w, field_h), lw)

    # Center line (vertical)
    pygame.draw.line(surface, FIELD_LINE, (cx, top), (cx, bottom), lw)

    # Center circle + dot
    center_r = int(field_h * 0.22)
    pygame.draw.circle(surface, FIELD_LINE, (cx, cy), center_r, lw)
    pygame.draw.circle(surface, FIELD_LINE, (cx, cy), max(3, tile // 8))

    # Penalty areas
    pen_depth = int(field_w * 0.12)
    pen_half_h = int(field_h * 0.35)
    pygame.draw.rect(surface, FIELD_LINE,
                     (left, cy - pen_half_h, pen_depth, pen_half_h * 2), lw)
    pygame.draw.rect(surface, FIELD_LINE,
                     (right - pen_depth, cy - pen_half_h,
                      pen_depth, pen_half_h * 2), lw)

    # Goal areas (smaller boxes)
    goal_depth = int(field_w * 0.05)
    goal_half_h = int(field_h * 0.18)
    pygame.draw.rect(surface, FIELD_LINE,
                     (left, cy - goal_half_h, goal_depth, goal_half_h * 2), lw)
    pygame.draw.rect(surface, FIELD_LINE,
                     (right - goal_depth, cy - goal_half_h,
                      goal_depth, goal_half_h * 2), lw)

    # Penalty spots
    pen_spot_dist = int(field_w * 0.08)
    spot_r = max(2, tile // 10)
    pygame.draw.circle(surface, FIELD_LINE,
                       (left + pen_spot_dist, cy), spot_r)
    pygame.draw.circle(surface, FIELD_LINE,
                       (right - pen_spot_dist, cy), spot_r)


def _draw_goals(surface: pygame.Surface, tile: int, env):
    """Goal posts as team-colored rectangles with net pattern."""
    for pos, idx in zip(env.goal_pos, env.goal_index):
        gx, gy = pos
        color = TEAM_COLORS.get(idx, (200, 200, 200))
        rect = pygame.Rect(gx * tile + 2, gy * tile + 2, tile - 4, tile - 4)

        # Semi-transparent background
        goal_surf = pygame.Surface((tile - 4, tile - 4), pygame.SRCALPHA)
        goal_surf.fill((*color, 80))
        surface.blit(goal_surf, (gx * tile + 2, gy * tile + 2))

        # Net lines
        for ly in range(4, tile - 4, max(3, tile // 8)):
            pygame.draw.line(
                surface, GOAL_NET,
                (gx * tile + 4, gy * tile + ly),
                (gx * tile + tile - 4, gy * tile + ly), 1)

        # Border
        pygame.draw.rect(surface, color, rect, max(2, tile // 12))


def _draw_fov(surface: pygame.Surface, tile: int, env):
    """Directional field-of-view highlight for each agent."""
    gw, gh = env.width, env.height
    vs = env.agents[0].view_size

    for agent in env.agents:
        ax, ay = int(agent.state.pos[0]), int(agent.state.pos[1])
        direction = int(agent.state.dir)
        team = agent.team_index
        color = TEAM_COLORS.get(team, (200, 200, 200))

        # Top-left of view rectangle (matches get_view_exts in obs.py)
        if direction == 0:    # RIGHT
            tx, ty = ax, ay - vs // 2
        elif direction == 1:  # DOWN
            tx, ty = ax - vs // 2, ay
        elif direction == 2:  # LEFT
            tx, ty = ax - vs + 1, ay - vs // 2
        else:                 # UP
            tx, ty = ax - vs // 2, ay - vs + 1

        # Clip to grid bounds
        x0 = max(0, tx)
        y0 = max(0, ty)
        x1 = min(gw - 1, tx + vs - 1)
        y1 = min(gh - 1, ty + vs - 1)

        px = x0 * tile
        py = y0 * tile
        pw = (x1 - x0 + 1) * tile
        ph = (y1 - y0 + 1) * tile

        fov_surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
        fov_surf.fill((*color, 35))
        surface.blit(fov_surf, (px, py))
        pygame.draw.rect(surface, (*color, 160), (px, py, pw, ph), 2)


def _draw_agent(surface: pygame.Surface, tile: int,
                x: int, y: int, direction: int,
                team_index: int, carrying: bool):
    """Agent as a colored triangle pointing in facing direction."""
    cx = x * tile + tile // 2
    cy = y * tile + tile // 2
    size = tile * 0.40

    color = TEAM_COLORS.get(team_index, (200, 200, 200))
    outline = TEAM_OUTLINE.get(team_index, (100, 100, 100))

    # Base triangle facing RIGHT, then rotate
    tip = (cx + size, cy)
    rear_top = (cx - size * 0.6, cy - size * 0.7)
    rear_bot = (cx - size * 0.6, cy + size * 0.7)

    angle = direction * math.pi / 2  # 0=right, 1=down, 2=left, 3=up
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def rotate(px, py):
        dx, dy = px - cx, py - cy
        return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)

    points = [rotate(*tip), rotate(*rear_top), rotate(*rear_bot)]

    # Shadow
    shadow_offset = 2
    shadow_pts = [(px + shadow_offset, py + shadow_offset) for px, py in points]
    shadow_surf = pygame.Surface((tile, tile), pygame.SRCALPHA)
    shifted = [(px - x * tile, py - y * tile) for px, py in shadow_pts]
    pygame.draw.polygon(shadow_surf, (0, 0, 0, 50), shifted)
    surface.blit(shadow_surf, (x * tile, y * tile))

    # Carrying glow
    if carrying:
        glow_surf = pygame.Surface((tile, tile), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (255, 255, 100, 60),
                           (tile // 2, tile // 2), int(size + 4))
        surface.blit(glow_surf, (x * tile, y * tile))

    # Body
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, outline, points, max(2, tile // 14))

    # Ball indicator if carrying
    if carrying:
        ball_r = max(3, tile // 8)
        pygame.draw.circle(surface, BALL_COLOR, (cx, cy), ball_r)
        pygame.draw.circle(surface, BALL_OUTLINE, (cx, cy), ball_r, 1)


def _draw_ball(surface: pygame.Surface, tile: int, x: int, y: int):
    """Ball as a circle with shadow and highlight."""
    cx = x * tile + tile // 2
    cy = y * tile + tile // 2
    radius = int(tile * 0.25)

    # Shadow
    shadow_surf = pygame.Surface((tile, tile), pygame.SRCALPHA)
    pygame.draw.circle(shadow_surf, (0, 0, 0, 50),
                       (tile // 2 + 1, tile // 2 + 1), radius)
    surface.blit(shadow_surf, (x * tile, y * tile))

    # Body
    pygame.draw.circle(surface, BALL_COLOR, (cx, cy), radius)
    pygame.draw.circle(surface, BALL_OUTLINE, (cx, cy), radius,
                       max(1, tile // 16))

    # Highlight
    hl_r = max(1, radius // 3)
    pygame.draw.circle(surface, (255, 180, 180),
                       (cx - radius // 3, cy - radius // 3), hl_r)


def _draw_agent_labels(surface: pygame.Surface, env, font, tile: int):
    """Small agent ID labels."""
    for agent in env.agents:
        x, y = int(agent.state.pos[0]), int(agent.state.pos[1])
        cx = x * tile + tile // 2 - 3
        cy = y * tile + tile // 2 - 5
        font.render_to(
            surface, (cx, cy),
            str(agent.index), fgcolor=(255, 255, 255),
            size=max(9, tile // 3))


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------

def render_fifa(env, tile_size: int = 32) -> np.ndarray:
    """Render the environment with FIFA-style graphics.

    Parameters
    ----------
    env : MultiGridEnv
        The environment instance (must have .width, .height, .agents,
        .grid, .goal_pos, .goal_index attributes).
    tile_size : int
        Pixel size of each grid tile.

    Returns
    -------
    img : ndarray[uint8] of shape (height_px, width_px, 3)
        RGB image in HWC format (matches ``Grid.render()`` output).
    """
    if not pygame.get_init():
        pygame.init()
    if not pygame.freetype.get_init():
        pygame.freetype.init()

    gw, gh = env.width, env.height
    surface = pygame.Surface((gw * tile_size, gh * tile_size))

    # 1. Grass background
    _draw_grass(surface, tile_size, gw, gh)

    # 2. Field markings
    _draw_field_markings(surface, tile_size, gw, gh)

    # 3. Agent FOV overlays
    _draw_fov(surface, tile_size, env)

    # 4. Goals
    _draw_goals(surface, tile_size, env)

    # 5. Ball on ground
    for x in range(gw):
        for y in range(gh):
            obj = env.grid.get(x, y)
            if obj is not None and obj.type == Type.ball:
                _draw_ball(surface, tile_size, x, y)

    # 6. Agents
    for agent in env.agents:
        ax, ay = int(agent.state.pos[0]), int(agent.state.pos[1])
        carrying = agent.state.carrying is not None
        _draw_agent(surface, tile_size, ax, ay,
                    int(agent.state.dir), agent.team_index, carrying)

    # 7. Agent ID labels
    font = pygame.freetype.SysFont('monospace', 14)
    _draw_agent_labels(surface, env, font, tile_size)

    # Convert pygame Surface -> numpy HWC array
    # surfarray.array3d returns (W, H, 3), transpose to (H, W, 3)
    arr = pygame.surfarray.array3d(surface)
    return arr.transpose(1, 0, 2)
