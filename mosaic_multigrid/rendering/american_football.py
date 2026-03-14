"""American Football-style rendering backend for American Football environments.

Replaces the default tile-based grid rendering with a football field overlay:
alternating grass stripes, white field markings, colored end zones, yard lines,
directional agent triangles, and per-agent field-of-view highlights.

All drawing uses pygame primitives. The public entry point ``render_american_football()``
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

GRASS_LIGHT = (139, 90, 43)   # Light brown
GRASS_DARK = (115, 74, 36)    # Dark brown
GRASS_WALL = (90, 60, 30)     # Darker brown for walls
FIELD_LINE = (255, 255, 255)

# End zone colors (semi-transparent overlays)
ENDZONE_GREEN = (30, 200, 60)
ENDZONE_BLUE = (60, 80, 220)

TEAM_COLORS = {
    0: (30, 200, 60),   # Green team
    1: (60, 80, 220),   # Blue team
}
TEAM_OUTLINE = {
    0: (15, 120, 30),
    1: (30, 40, 140),
}

BALL_COLOR = (139, 69, 19)  # Brown for American football
BALL_OUTLINE = (90, 45, 10)


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
    """White American Football field markings (boundary, yard lines)."""
    lw = max(2, tile // 16)

    left = 1 * tile
    right = (gw - 1) * tile
    top = 1 * tile
    bottom = (gh - 1) * tile
    field_w = right - left
    field_h = bottom - top

    # Outer boundary
    pygame.draw.rect(surface, FIELD_LINE,
                     (left, top, field_w, field_h), lw)

    # Yard lines (vertical lines across the field)
    # Draw lines in the midfield area (columns 2 to width-3)
    # Draw yard lines every 2 columns in midfield
    for col in range(2, gw - 2):
        if col % 2 == 0:  # Every other column
            x = col * tile
            pygame.draw.line(surface, FIELD_LINE,
                           (x, top), (x, bottom), lw)

    # Draw hash marks (small horizontal lines)
    hash_length = tile // 3
    for col in range(2, gw - 2):
        x = col * tile
        # Upper hash marks
        y_upper = top + field_h // 3
        pygame.draw.line(surface, FIELD_LINE,
                        (x, y_upper - hash_length // 2),
                        (x, y_upper + hash_length // 2), max(1, lw // 2))
        # Lower hash marks
        y_lower = top + 2 * field_h // 3
        pygame.draw.line(surface, FIELD_LINE,
                        (x, y_lower - hash_length // 2),
                        (x, y_lower + hash_length // 2), max(1, lw // 2))


def _draw_endzones(surface: pygame.Surface, tile: int, env):
    """Draw colored end zones at columns 1 and width-2."""
    gw, gh = env.width, env.height

    # Find end zones from the grid
    for x in range(gw):
        for y in range(gh):
            obj = env.grid.get(x, y)
            if obj is not None and obj.type == Type.endzone:
                # Determine color based on the endzone's color attribute
                if hasattr(obj, 'color'):
                    color_name = str(obj.color)
                    if 'green' in color_name.lower():
                        color = ENDZONE_GREEN
                    elif 'blue' in color_name.lower():
                        color = ENDZONE_BLUE
                    else:
                        color = (150, 150, 150)  # Default gray
                else:
                    color = (150, 150, 150)

                # Draw semi-transparent end zone cell
                endzone_surf = pygame.Surface((tile, tile), pygame.SRCALPHA)
                endzone_surf.fill((*color, 100))
                surface.blit(endzone_surf, (x * tile, y * tile))

                # Draw border
                pygame.draw.rect(surface, color,
                               (x * tile, y * tile, tile, tile),
                               max(2, tile // 12))


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
    """Ball as an oval (American football shape) with shadow and highlight."""
    cx = x * tile + tile // 2
    cy = y * tile + tile // 2
    width = int(tile * 0.35)
    height = int(tile * 0.22)

    # Shadow
    shadow_surf = pygame.Surface((tile, tile), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow_surf, (0, 0, 0, 50),
                       (tile // 2 - width // 2 + 1, tile // 2 - height // 2 + 1,
                        width, height))
    surface.blit(shadow_surf, (x * tile, y * tile))

    # Body (oval shape)
    pygame.draw.ellipse(surface, BALL_COLOR,
                       (cx - width // 2, cy - height // 2, width, height))
    pygame.draw.ellipse(surface, BALL_OUTLINE,
                       (cx - width // 2, cy - height // 2, width, height),
                       max(1, tile // 16))

    # Laces (white lines)
    lace_spacing = max(2, height // 4)
    for i in range(-1, 2):
        lace_y = cy + i * lace_spacing
        pygame.draw.line(surface, (255, 255, 255),
                        (cx - width // 6, lace_y),
                        (cx + width // 6, lace_y), 1)


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

def render_american_football(env, tile_size: int = 32, show_hud: bool = True) -> np.ndarray:
    """Render the environment with American Football-style graphics.

    Parameters
    ----------
    env : MultiGridEnv
        The environment instance (must have .width, .height, .agents,
        .grid, .endzone_positions attributes).
    tile_size : int
        Pixel size of each grid tile.
    show_hud : bool
        Whether to show HUD elements (FOV highlights, agent labels).
        Default True for backward compatibility.

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

    # 3. End zones
    _draw_endzones(surface, tile_size, env)

    # 4. Agent FOV overlays (only if HUD enabled)
    if show_hud:
        _draw_fov(surface, tile_size, env)

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

    # 7. Agent ID labels (only if HUD enabled)
    if show_hud:
        font = pygame.freetype.SysFont('monospace', 14)
        _draw_agent_labels(surface, env, font, tile_size)

    # Convert pygame Surface -> numpy HWC array
    # surfarray.array3d returns (W, H, 3), transpose to (H, W, 3)
    arr = pygame.surfarray.array3d(surface)
    return arr.transpose(1, 0, 2)
