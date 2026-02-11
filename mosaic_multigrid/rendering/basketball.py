"""Basketball-court rendering backend for Basketball environments.

Replaces the default tile-based grid rendering with a basketball-court overlay:
alternating hardwood plank strips, white court markings (three-point arc, paint
area, center circle), team-colored hoops at baselines, directional agent
triangles, and per-agent field-of-view highlights.

All drawing uses pygame primitives.  The public entry point
``render_basketball()`` returns an ``ndarray[uint8]`` of shape
``(height_px, width_px, 3)`` that plugs directly into the
``get_full_render()`` pipeline in ``base.py``.
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

HARDWOOD_LIGHT = (222, 184, 135)
HARDWOOD_DARK = (200, 165, 118)
PAINT_FILL = (190, 155, 110)
OUT_OF_BOUNDS = (55, 45, 35)
COURT_LINE = (255, 255, 255)
GRID_LINE_COLOR = (90, 70, 45)

# Team colors: Green (team 1, left) vs Blue (team 2, right)
TEAM_COLORS = {
    1: (30, 160, 50),
    2: (30, 80, 200),
}
TEAM_OUTLINE = {
    1: (15, 100, 30),
    2: (15, 40, 130),
}
HOOP_COLORS = {
    1: (30, 180, 60),
    2: (40, 100, 220),
}

BALL_COLOR = (255, 140, 0)
BALL_OUTLINE = (160, 80, 0)


# -----------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------

def _draw_hardwood(surface: pygame.Surface, tile: int, gw: int, gh: int):
    """Alternating vertical hardwood plank strips with dark wall border."""
    # Wall cells (dark border)
    surface.fill(OUT_OF_BOUNDS)

    # Hardwood planks on playable area
    for col in range(1, gw - 1):
        color = HARDWOOD_LIGHT if col % 2 == 0 else HARDWOOD_DARK
        for row in range(1, gh - 1):
            pygame.draw.rect(surface, color,
                             (col * tile, row * tile, tile, tile))


def _draw_grid_lines(surface: pygame.Surface, tile: int, gw: int, gh: int):
    """Thin grid lines inside playable area (drawn after paint so they show)."""
    for col in range(1, gw):
        lx = col * tile
        pygame.draw.line(surface, GRID_LINE_COLOR,
                         (lx, 1 * tile), (lx, (gh - 1) * tile), 1)
    for row in range(1, gh):
        ly = row * tile
        pygame.draw.line(surface, GRID_LINE_COLOR,
                         (1 * tile, ly), ((gw - 1) * tile, ly), 1)


def _draw_paint_area(surface: pygame.Surface, tile: int,
                     gw: int, gh: int, paint_depth: int, paint_half_h: int):
    """Fill the painted key area with a distinct shade on both sides."""
    cy = gh // 2
    # Left side
    for dx in range(paint_depth):
        gx = 1 + dx
        for dy in range(-paint_half_h, paint_half_h + 1):
            gy = cy + dy
            if 1 <= gy <= gh - 2:
                pygame.draw.rect(surface, PAINT_FILL,
                                 (gx * tile, gy * tile, tile, tile))
    # Right side
    for dx in range(paint_depth):
        gx = (gw - 2) - dx
        for dy in range(-paint_half_h, paint_half_h + 1):
            gy = cy + dy
            if 1 <= gy <= gh - 2:
                pygame.draw.rect(surface, PAINT_FILL,
                                 (gx * tile, gy * tile, tile, tile))


def _draw_court_markings(surface: pygame.Surface, tile: int,
                         gw: int, gh: int, cfg: dict):
    """White basketball court markings."""
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
    pygame.draw.rect(surface, COURT_LINE,
                     (left, top, field_w, field_h), lw)

    # Center line (vertical)
    pygame.draw.line(surface, COURT_LINE, (cx, top), (cx, bottom), lw)

    # Center circle
    center_r = int(cfg.get('center_radius', 1.5) * tile)
    pygame.draw.circle(surface, COURT_LINE, (cx, cy), center_r, lw)

    paint_depth = cfg.get('paint_depth', 3)
    paint_half_h = cfg.get('paint_half_h', 2)
    three_pt_r = cfg.get('three_pt_radius', 5.0)
    ft_r = cfg.get('ft_circle_radius', 2)

    ph_px = paint_half_h * tile
    pd_px = paint_depth * tile

    # Both sides
    for baseline_px, d in [(left, 1), (right, -1)]:
        by_px = cy

        # Paint rectangle
        if d == 1:
            pr = pygame.Rect(baseline_px, by_px - ph_px, pd_px, ph_px * 2)
        else:
            pr = pygame.Rect(baseline_px - pd_px, by_px - ph_px,
                             pd_px, ph_px * 2)
        pygame.draw.rect(surface, COURT_LINE, pr, lw)

        # Free-throw semicircle (outside paint)
        ft_x = baseline_px + d * pd_px
        if d == 1:
            _arc(surface, (ft_x, by_px), ph_px,
                 -math.pi / 2, math.pi / 2, COURT_LINE, lw)
        else:
            _arc(surface, (ft_x, by_px), ph_px,
                 math.pi / 2, 3 * math.pi / 2, COURT_LINE, lw)

        # Dashed free-throw semicircle (inside paint)
        if d == 1:
            _arc(surface, (ft_x, by_px), ph_px,
                 math.pi / 2, 3 * math.pi / 2, COURT_LINE, 1, dashed=True)
        else:
            _arc(surface, (ft_x, by_px), ph_px,
                 -math.pi / 2, math.pi / 2, COURT_LINE, 1, dashed=True)

        # Three-point arc (centered on baseline)
        tp_r_px = int(three_pt_r * tile)
        court_half_h = field_h / 2.0
        sin_max = min(1.0, court_half_h / tp_r_px) if tp_r_px > 0 else 1.0
        max_angle = math.asin(sin_max) * 0.92
        if d == 1:
            sa, ea = -max_angle, max_angle
        else:
            sa, ea = math.pi - max_angle, math.pi + max_angle
        _arc(surface, (baseline_px, by_px), tp_r_px, sa, ea,
             COURT_LINE, lw)

        # Corner three-point lines
        for angle in [sa, ea]:
            ey = by_px + int(tp_r_px * math.sin(angle))
            ey = max(top, min(bottom, ey))
            ex = baseline_px + d * int(tp_r_px * abs(math.cos(angle)))
            pygame.draw.line(surface, COURT_LINE,
                             (baseline_px, ey), (ex, ey), lw)

        # Restricted area arc
        ra_r = int(0.8 * tile)
        if d == 1:
            _arc(surface, (baseline_px, by_px), ra_r,
                 -math.pi / 2, math.pi / 2, COURT_LINE, max(1, lw - 1))
        else:
            _arc(surface, (baseline_px, by_px), ra_r,
                 math.pi / 2, 3 * math.pi / 2, COURT_LINE, max(1, lw - 1))


def _arc(surface, center, radius, start, end, color, width, dashed=False):
    """Draw an arc via line segments."""
    steps = 64
    pts = []
    for i in range(steps + 1):
        a = start + (end - start) * i / steps
        x = center[0] + int(radius * math.cos(a))
        y = center[1] + int(radius * math.sin(a))
        pts.append((x, y))
    if dashed:
        seg_len = 4
        for i in range(0, len(pts) - 1, seg_len * 2):
            chunk = pts[i:i + seg_len + 1]
            if len(chunk) > 1:
                pygame.draw.lines(surface, color, False, chunk, width)
    else:
        if len(pts) > 1:
            pygame.draw.lines(surface, color, False, pts, width)


def _draw_goals(surface: pygame.Surface, tile: int, env):
    """Goal cells as team-colored tiles on the baseline."""
    for pos, idx in zip(env.goal_pos, env.goal_index):
        gx, gy = pos
        color = HOOP_COLORS.get(idx, (200, 200, 200))
        # Darker fill
        dark = tuple(max(0, c - 80) for c in color)
        rect = pygame.Rect(gx * tile, gy * tile, tile, tile)
        pygame.draw.rect(surface, dark, rect)
        pygame.draw.rect(surface, color, rect, 3)


def _draw_baskets(surface: pygame.Surface, tile: int,
                  gw: int, gh: int, env):
    """Draw backboard + hoop at each baseline, colored per team."""
    left_px = 1 * tile
    right_px = (gw - 1) * tile
    cy_px = (gh // 2) * tile + tile // 2

    for baseline_px, d, goal_idx in [
        (left_px, 1, env.goal_index[0]),
        (right_px, -1, env.goal_index[1]),
    ]:
        hoop_color = HOOP_COLORS.get(goal_idx, (200, 200, 200))
        by_px = cy_px

        # Backboard
        bb_half = int(tile * 0.7)
        pygame.draw.line(surface, (240, 240, 240),
                         (baseline_px, by_px - bb_half),
                         (baseline_px, by_px + bb_half), 5)

        # Target square
        sq = int(tile * 0.16)
        pygame.draw.rect(surface, hoop_color,
                         (baseline_px - sq, by_px - sq, sq * 2, sq * 2), 2)

        # Rim circle
        rim_r = int(tile * 0.22)
        rim_cx = baseline_px + d * int(tile * 0.32)
        pygame.draw.circle(surface, hoop_color, (rim_cx, by_px), rim_r, 3)

        # Connector
        pygame.draw.line(surface, hoop_color,
                         (baseline_px, by_px),
                         (rim_cx - d * rim_r, by_px), 2)

        # Net lines
        net_top = by_px + rim_r
        net_len = int(tile * 0.25)
        for nx in [rim_cx - rim_r // 2, rim_cx, rim_cx + rim_r // 2]:
            pygame.draw.line(surface, (200, 200, 200),
                             (nx, net_top), (rim_cx, net_top + net_len), 1)


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

    tip = (cx + size, cy)
    rear_top = (cx - size * 0.6, cy - size * 0.7)
    rear_bot = (cx - size * 0.6, cy + size * 0.7)

    angle = direction * math.pi / 2
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def rotate(px, py):
        dx, dy = px - cx, py - cy
        return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)

    points = [rotate(*tip), rotate(*rear_top), rotate(*rear_bot)]

    # Shadow
    shadow_offset = 2
    shadow_pts = [(px + shadow_offset, py + shadow_offset)
                  for px, py in points]
    shadow_surf = pygame.Surface((tile, tile), pygame.SRCALPHA)
    shifted = [(px - x * tile, py - y * tile) for px, py in shadow_pts]
    pygame.draw.polygon(shadow_surf, (0, 0, 0, 50), shifted)
    surface.blit(shadow_surf, (x * tile, y * tile))

    # Carrying glow
    if carrying:
        glow_surf = pygame.Surface((tile, tile), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (255, 200, 50, 60),
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
    """Basketball as an orange circle with seam lines."""
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
    # Seam lines
    pygame.draw.line(surface, BALL_OUTLINE,
                     (cx - radius, cy), (cx + radius, cy), 2)
    pygame.draw.line(surface, BALL_OUTLINE,
                     (cx, cy - radius), (cx, cy + radius), 2)


def _draw_agent_labels(surface: pygame.Surface, env, font, tile: int):
    """White agent ID number centered on the triangle."""
    font_size = max(10, tile * 3 // 8)

    for agent in env.agents:
        x, y = int(agent.state.pos[0]), int(agent.state.pos[1])
        label = str(agent.index)

        # Measure text to center it on tile
        text_rect = font.get_rect(label, size=font_size)
        cx = x * tile + (tile - text_rect.width) // 2
        cy = y * tile + (tile - text_rect.height) // 2

        font.render_to(surface, (cx, cy), label,
                        fgcolor=(255, 255, 255), size=font_size)


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------

# Default court config for 3vs3 basketball (17x9 playable, 19x11 total)
_COURT_CFG_3V3 = {
    'paint_depth': 3,
    'paint_half_h': 2,
    'three_pt_radius': 5.0,
    'center_radius': 1.5,
    'ft_circle_radius': 2,
}


def render_basketball(env, tile_size: int = 32) -> np.ndarray:
    """Render the environment with basketball-court graphics.

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
        RGB image in HWC format.
    """
    if not pygame.get_init():
        pygame.init()
    if not pygame.freetype.get_init():
        pygame.freetype.init()

    gw, gh = env.width, env.height
    surface = pygame.Surface((gw * tile_size, gh * tile_size))

    # Court configuration (use env attribute if available, else default)
    cfg = getattr(env, 'court_cfg', _COURT_CFG_3V3)

    # 1. Hardwood floor background
    _draw_hardwood(surface, tile_size, gw, gh)

    # 2. Paint area
    _draw_paint_area(surface, tile_size, gw, gh,
                     cfg['paint_depth'], cfg['paint_half_h'])

    # 3. Grid lines (after paint so lines show through paint area)
    _draw_grid_lines(surface, tile_size, gw, gh)

    # 4. Court markings
    _draw_court_markings(surface, tile_size, gw, gh, cfg)

    # 4. Agent FOV overlays
    _draw_fov(surface, tile_size, env)

    # 5. Goal cells
    _draw_goals(surface, tile_size, env)

    # 6. Baskets (backboard + hoop)
    _draw_baskets(surface, tile_size, gw, gh, env)

    # 7. Ball on ground
    for x in range(gw):
        for y in range(gh):
            obj = env.grid.get(x, y)
            if obj is not None and obj.type == Type.ball:
                _draw_ball(surface, tile_size, x, y)

    # 8. Agents
    for agent in env.agents:
        ax, ay = int(agent.state.pos[0]), int(agent.state.pos[1])
        carrying = agent.state.carrying is not None
        _draw_agent(surface, tile_size, ax, ay,
                    int(agent.state.dir), agent.team_index, carrying)

    # 9. Agent ID labels
    font = pygame.freetype.SysFont('monospace', 14)
    _draw_agent_labels(surface, env, font, tile_size)

    # Convert pygame Surface -> numpy HWC array
    arr = pygame.surfarray.array3d(surface)
    return arr.transpose(1, 0, 2)
