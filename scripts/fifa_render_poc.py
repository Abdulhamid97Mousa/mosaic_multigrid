"""Proof-of-concept: FIFA-style rendering for MOSAIC Soccer environment.

This is a STANDALONE script -- it does NOT modify any existing code.
It reads the environment state and draws a soccer-pitch overlay using pygame.

Usage:
    python scripts/fifa_render_poc.py
    python scripts/fifa_render_poc.py --seed 99
    python scripts/fifa_render_poc.py --steps 50
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pygame
import pygame.freetype

from mosaic_multigrid.envs import SoccerGame4HEnhancedEnv16x11N2
from mosaic_multigrid.core.constants import Type


# -----------------------------------------------------------------------
# Color palette (FIFA-inspired)
# -----------------------------------------------------------------------

GRASS_LIGHT = (76, 153, 60)     # Light grass stripe
GRASS_DARK = (58, 128, 45)      # Dark grass stripe
GRASS_WALL = (45, 100, 35)      # Border / wall area
FIELD_LINE = (255, 255, 255)    # White field markings
GOAL_NET = (220, 220, 220)      # Light grey goal net

TEAM_COLORS = {
    1: (30, 200, 60),           # Green team (slightly darker for contrast on grass)
    2: (60, 80, 220),           # Blue team
}
TEAM_OUTLINE = {
    1: (15, 120, 30),
    2: (30, 40, 140),
}

BALL_COLOR = (255, 60, 60)      # Red ball (matches environment)
BALL_OUTLINE = (180, 30, 30)

SHADOW_COLOR = (0, 0, 0, 40)   # Transparent shadow

# Agent carrying ball indicator
CARRY_GLOW = (255, 255, 100, 120)


# -----------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------

def draw_grass(surface: pygame.Surface, tile: int, gw: int, gh: int):
    """Draw alternating vertical grass stripes across the full field."""
    stripe_width = tile  # Each stripe = 1 column wide
    for col in range(gw):
        color = GRASS_LIGHT if col % 2 == 0 else GRASS_DARK
        pygame.draw.rect(surface, color, (col * tile, 0, tile, gh * tile))

    # Thin black grid lines inside playable area only (skip wall border)
    for col in range(1, gw):
        line_x = col * tile
        pygame.draw.line(surface, (0, 0, 0), (line_x, 1 * tile), (line_x, (gh - 1) * tile), 1)
    for row in range(1, gh):
        line_y = row * tile
        pygame.draw.line(surface, (0, 0, 0), (1 * tile, line_y), ((gw - 1) * tile, line_y), 1)

    # Darken the wall border cells
    for col in range(gw):
        for row in range(gh):
            if col == 0 or col == gw - 1 or row == 0 or row == gh - 1:
                pygame.draw.rect(
                    surface, GRASS_WALL,
                    (col * tile, row * tile, tile, tile),
                )


def draw_field_markings(surface: pygame.Surface, tile: int, gw: int, gh: int):
    """Draw FIFA-style field markings (lines, circles, boxes)."""
    lw = max(2, tile // 16)  # Line width scales with tile size

    # Playable area bounds (inside walls)
    left = 1 * tile
    right = (gw - 1) * tile
    top = 1 * tile
    bottom = (gh - 1) * tile
    field_w = right - left
    field_h = bottom - top
    cx = left + field_w // 2    # Center x
    cy = top + field_h // 2     # Center y

    # Outer boundary
    pygame.draw.rect(surface, FIELD_LINE, (left, top, field_w, field_h), lw)

    # Center line (vertical)
    pygame.draw.line(surface, FIELD_LINE, (cx, top), (cx, bottom), lw)

    # Center circle
    center_r = int(field_h * 0.22)
    pygame.draw.circle(surface, FIELD_LINE, (cx, cy), center_r, lw)

    # Center dot
    pygame.draw.circle(surface, FIELD_LINE, (cx, cy), max(3, tile // 8))

    # --- Penalty areas (left and right) ---
    pen_depth = int(field_w * 0.12)
    pen_half_h = int(field_h * 0.35)

    # Left penalty area
    pygame.draw.rect(
        surface, FIELD_LINE,
        (left, cy - pen_half_h, pen_depth, pen_half_h * 2),
        lw,
    )

    # Right penalty area
    pygame.draw.rect(
        surface, FIELD_LINE,
        (right - pen_depth, cy - pen_half_h, pen_depth, pen_half_h * 2),
        lw,
    )

    # --- Goal areas (smaller boxes inside penalty areas) ---
    goal_depth = int(field_w * 0.05)
    goal_half_h = int(field_h * 0.18)

    # Left goal area
    pygame.draw.rect(
        surface, FIELD_LINE,
        (left, cy - goal_half_h, goal_depth, goal_half_h * 2),
        lw,
    )

    # Right goal area
    pygame.draw.rect(
        surface, FIELD_LINE,
        (right - goal_depth, cy - goal_half_h, goal_depth, goal_half_h * 2),
        lw,
    )

    # --- Penalty spots ---
    pen_spot_dist = int(field_w * 0.08)
    spot_r = max(2, tile // 10)
    pygame.draw.circle(surface, FIELD_LINE, (left + pen_spot_dist, cy), spot_r)
    pygame.draw.circle(surface, FIELD_LINE, (right - pen_spot_dist, cy), spot_r)



def draw_goals(surface: pygame.Surface, tile: int, env):
    """Draw goal posts as colored rectangles at goal positions."""
    for i, (pos, idx) in enumerate(zip(env.goal_pos, env.goal_index)):
        gx, gy = pos
        # Goal color matches team
        color = TEAM_COLORS.get(idx, (200, 200, 200))
        # Draw goal post as a thicker outlined rectangle
        rect = pygame.Rect(gx * tile + 2, gy * tile + 2, tile - 4, tile - 4)
        # Filled semi-transparent background
        goal_surf = pygame.Surface((tile - 4, tile - 4), pygame.SRCALPHA)
        goal_surf.fill((*color, 80))
        surface.blit(goal_surf, (gx * tile + 2, gy * tile + 2))
        # Net pattern (horizontal lines)
        for line_y in range(4, tile - 4, max(3, tile // 8)):
            pygame.draw.line(
                surface, GOAL_NET,
                (gx * tile + 4, gy * tile + line_y),
                (gx * tile + tile - 4, gy * tile + line_y),
                1,
            )
        # Border
        pygame.draw.rect(surface, color, rect, max(2, tile // 12))


def draw_agent(
    surface: pygame.Surface, tile: int,
    x: int, y: int, direction: int,
    team_index: int, carrying: bool,
):
    """Draw an agent as a colored triangle pointing in facing direction."""
    cx = x * tile + tile // 2
    cy = y * tile + tile // 2
    size = tile * 0.40  # Half-extent of the triangle

    color = TEAM_COLORS.get(team_index, (200, 200, 200))
    outline = TEAM_OUTLINE.get(team_index, (100, 100, 100))

    # Base triangle points facing RIGHT, then rotate by direction
    # Tip at front, two rear corners
    tip = (cx + size, cy)
    rear_top = (cx - size * 0.6, cy - size * 0.7)
    rear_bot = (cx - size * 0.6, cy + size * 0.7)

    # Rotate all points around (cx, cy) by direction * 90 degrees
    angle = direction * math.pi / 2  # 0=right, 1=down, 2=left, 3=up
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def rotate(px, py):
        dx, dy = px - cx, py - cy
        return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)

    points = [rotate(*tip), rotate(*rear_top), rotate(*rear_bot)]

    # Shadow
    shadow_offset = 2
    shadow_points = [(px + shadow_offset, py + shadow_offset) for px, py in points]
    shadow_surf = pygame.Surface((tile, tile), pygame.SRCALPHA)
    shifted = [(px - x * tile, py - y * tile) for px, py in shadow_points]
    pygame.draw.polygon(shadow_surf, (0, 0, 0, 50), shifted)
    surface.blit(shadow_surf, (x * tile, y * tile))

    # Carrying glow
    if carrying:
        glow_surf = pygame.Surface((tile, tile), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (255, 255, 100, 60), (tile // 2, tile // 2), int(size + 4))
        surface.blit(glow_surf, (x * tile, y * tile))

    # Agent body (filled triangle)
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, outline, points, max(2, tile // 14))

    # Small ball indicator if carrying
    if carrying:
        ball_r = max(3, tile // 8)
        pygame.draw.circle(surface, BALL_COLOR, (cx, cy), ball_r)
        pygame.draw.circle(surface, BALL_OUTLINE, (cx, cy), ball_r, 1)


def draw_ball(surface: pygame.Surface, tile: int, x: int, y: int):
    """Draw the ball as a circle on the field."""
    cx = x * tile + tile // 2
    cy = y * tile + tile // 2
    radius = int(tile * 0.25)

    # Shadow
    shadow_surf = pygame.Surface((tile, tile), pygame.SRCALPHA)
    pygame.draw.circle(shadow_surf, (0, 0, 0, 50), (tile // 2 + 1, tile // 2 + 1), radius)
    surface.blit(shadow_surf, (x * tile, y * tile))

    # Ball body
    pygame.draw.circle(surface, BALL_COLOR, (cx, cy), radius)
    pygame.draw.circle(surface, BALL_OUTLINE, (cx, cy), radius, max(1, tile // 16))

    # Simple highlight
    highlight_r = max(1, radius // 3)
    pygame.draw.circle(
        surface, (255, 180, 180),
        (cx - radius // 3, cy - radius // 3),
        highlight_r,
    )


def draw_scoreboard(surface: pygame.Surface, env, font, tile: int, gw: int):
    """Draw a small scoreboard at the top."""
    scores = getattr(env, 'team_scores', {})
    t1 = scores.get(1, 0)
    t2 = scores.get(2, 0)
    text = f"Green {t1} - {t2} Blue"
    font.render_to(
        surface, (gw * tile // 2 - 50, 4),
        text, fgcolor=(255, 255, 255), size=max(12, tile // 2),
    )


def draw_fov(surface: pygame.Surface, tile: int, env):
    """Draw each agent's directional field-of-view as a semi-transparent overlay.

    The FOV is a view_size x view_size rectangle extending FORWARD from the
    agent. The agent sits at the back edge (not center) of the view window.
    This matches the actual get_view_exts() geometry in obs.py.
    """
    gw, gh = env.width, env.height
    vs = env.agents[0].view_size  # typically 3

    for agent in env.agents:
        ax, ay = int(agent.state.pos[0]), int(agent.state.pos[1])
        direction = int(agent.state.dir)
        team = agent.team_index
        color = TEAM_COLORS.get(team, (200, 200, 200))

        # Compute top-left of view rectangle (matches get_view_exts)
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

        # Semi-transparent fill
        fov_surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
        fov_surf.fill((*color, 35))
        surface.blit(fov_surf, (px, py))

        # Border outline
        pygame.draw.rect(surface, (*color, 160), (px, py, pw, ph), 2)


def draw_agent_labels(surface: pygame.Surface, env, font, tile: int):
    """Draw small agent ID labels."""
    for agent in env.agents:
        x, y = int(agent.state.pos[0]), int(agent.state.pos[1])
        cx = x * tile + tile // 2 - 3
        cy = y * tile + tile // 2 - 5
        font.render_to(
            surface, (cx, cy),
            str(agent.index), fgcolor=(255, 255, 255), size=max(9, tile // 3),
        )


# -----------------------------------------------------------------------
# Main render function
# -----------------------------------------------------------------------

def render_fifa(env, tile: int = 48) -> pygame.Surface:
    """Render the environment with FIFA-style graphics. Returns a pygame Surface."""
    gw, gh = env.width, env.height
    surface = pygame.Surface((gw * tile, gh * tile))

    # 1. Grass background
    draw_grass(surface, tile, gw, gh)

    # 2. Field markings
    draw_field_markings(surface, tile, gw, gh)

    # 3. Agent field-of-view overlays (drawn early so everything else sits on top)
    draw_fov(surface, tile, env)

    # 4. Goals
    draw_goals(surface, tile, env)

    # 5. Ball on ground (if not carried by anyone)
    for x in range(gw):
        for y in range(gh):
            obj = env.grid.get(x, y)
            if obj is not None and obj.type == Type.ball:
                draw_ball(surface, tile, x, y)

    # 6. Agents
    for agent in env.agents:
        ax, ay = int(agent.state.pos[0]), int(agent.state.pos[1])
        carrying = agent.state.carrying is not None
        draw_agent(surface, tile, ax, ay, int(agent.state.dir), agent.team_index, carrying)

    # 7. Agent ID labels
    pygame.freetype.init()
    font = pygame.freetype.SysFont('monospace', 14)
    draw_agent_labels(surface, env, font, tile)

    return surface


# -----------------------------------------------------------------------
# Interactive viewer
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='FIFA-style Soccer render POC')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tile', type=int, default=48, help='Tile size in pixels')
    parser.add_argument('--steps', type=int, default=0, help='Auto-step N times then pause')
    parser.add_argument('--save', type=str, default=None, help='Save PNG and exit')
    args = parser.parse_args()

    env = SoccerGame4HEnhancedEnv16x11N2(render_mode='rgb_array')
    env.reset(seed=args.seed)

    # Auto-step if requested
    for _ in range(args.steps):
        actions = {i: env.action_space[i].sample() for i in range(4)}
        env.step(actions)

    pygame.init()
    pygame.display.init()

    gw, gh = env.width, env.height
    tile = args.tile
    screen_w, screen_h = gw * tile, gh * tile

    if args.save:
        surface = render_fifa(env, tile)
        pygame.image.save(surface, args.save)
        print(f'Saved to {args.save} ({screen_w}x{screen_h})')
        pygame.quit()
        env.close()
        return

    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption('MOSAIC Soccer - FIFA Render POC')
    clock = pygame.time.Clock()

    running = True
    paused = True
    step_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_n:
                    # Single step
                    actions = {i: env.action_space[i].sample() for i in range(4)}
                    obs, rew, terms, truncs, info = env.step(actions)
                    step_count += 1
                    if any(terms.values()):
                        print(f'Game ended at step {step_count}. Press R to reset.')
                    if any(truncs.values()):
                        print(f'Truncated at step {step_count}. Press R to reset.')
                elif event.key == pygame.K_r:
                    env.reset(seed=args.seed)
                    step_count = 0
                    print('Reset.')

        if not paused:
            actions = {i: env.action_space[i].sample() for i in range(4)}
            obs, rew, terms, truncs, info = env.step(actions)
            step_count += 1
            if any(terms.values()) or any(truncs.values()):
                env.reset(seed=args.seed + step_count)
                step_count = 0

        # Render
        surface = render_fifa(env, tile)
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(10 if not paused else 30)

    pygame.quit()
    env.close()


if __name__ == '__main__':
    main()
