"""Render a 1x3 comparison of the three sport goal geometries.

Left    Basketball   1-cell goal
Middle  Soccer       3-cell FIFA-style arc
Right   American FB  end-zone column (line zone)

Uses the solo (G-1v0) variants so the goal geometry is uncluttered by
multi-agent activity.

Usage:
    cd 3rd_party/environments/mosaic_multigrid
    python scripts/generate_goal_geometry_figure.py

Output:
    figures/goal_geometry.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import mosaic_multigrid  # triggers gymnasium registration
import gymnasium


PANELS = [
    ("MosaicMultiGrid-BB-G-1v0-v1",
     "Basketball (1-cell goal)"),
    ("MosaicMultiGrid-S-G-1v0-v1",
     "Soccer (3-cell FIFA arc goal)"),
    ("MosaicMultiGrid-AF-G-1v0-v1",
     "American Football (end-zone column)"),
]


def render_env(env_id: str, seed: int) -> np.ndarray:
    env = gymnasium.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    frame = env.render()
    env.close()
    return frame


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent.parent
                   / "figures" / "goal_geometry.png")
    args = p.parse_args()

    # Render all frames first so we can size panels by their true pixel width.
    # This is what makes all three envs display at the same HEIGHT — matplotlib
    # subplots would otherwise scale the wider Basketball grid (19x11) down to
    # fit an equal-width panel, making it look smaller than Soccer/AF (16x11).
    frames = [render_env(env_id, args.seed) for env_id, _ in PANELS]
    widths = [f.shape[1] for f in frames]
    heights = [f.shape[0] for f in frames]
    # Sanity: all envs render at the same pixel height per row of grid cells,
    # so heights should already match; if they don't, use the tallest.
    target_h = max(heights)

    # figsize width proportional to total pixel width; height matches tallest
    total_w_pixels = sum(widths)
    fig_w = 18.0  # inches, total figure width
    fig_h = fig_w * target_h / total_w_pixels

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=widths, wspace=0.05)
    for i, (frame, (_, subtitle)) in enumerate(zip(frames, PANELS)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frame)
        ax.set_title(subtitle, fontsize=12)
        ax.axis("off")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
