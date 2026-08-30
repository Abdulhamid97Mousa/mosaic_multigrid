"""Shared render helpers for the category-figure scripts.

Each per-category script (generate_solo_figure.py, generate_cooperative_figure.py,
generate_competitive_symmetric_figure.py, generate_competitive_asymmetric_figure.py)
imports render_grid() from this module and provides only the env list + grid shape.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

import mosaic_multigrid  # noqa: F401 (triggers gymnasium registration)
import gymnasium


def render_env(env_id: str, seed: int = 42) -> np.ndarray:
    """Return a single RGB frame after env.reset(seed)."""
    env = gymnasium.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    frame = env.render()
    env.close()
    return frame


def render_grid(
    env_ids: Sequence[str],
    rows: int,
    cols: int,
    title: str,
    out_path: Path,
    seed: int = 42,
    subplot_size: float = 3.0,
    title_fontsize: int = 16,
    panel_title_fontsize: int = 8,
    col_width_ratios: Sequence[int] | None = None,
) -> None:
    """Render a grid of env previews and save to disk.

    Args:
        env_ids: gym env IDs, in row-major order. len must equal rows*cols
                 or fewer (extra cells become blank).
        rows, cols: grid shape
        title: figure suptitle
        out_path: PNG output path
        seed: reset seed for each env
        subplot_size: inches per subplot cell
        title_fontsize: suptitle size
        panel_title_fontsize: per-env caption size
        col_width_ratios: optional list of length `cols` giving relative
            widths (e.g. [19, 16, 16] for BB/AF/S columns). When supplied,
            all cells in a column get the same width, keeping all rendered
            envs at the same height. Without it, cells are equal-sized and
            wider envs (BB 19x11) appear larger than narrower ones (AF/S 16x11).
    """
    figsize = (cols * subplot_size, rows * subplot_size)
    if col_width_ratios is not None:
        assert len(col_width_ratios) == cols, (
            f"col_width_ratios must have length {cols}, got {len(col_width_ratios)}"
        )
        # Rescale figure width so avg subplot width stays ~subplot_size
        avg = sum(col_width_ratios) / len(col_width_ratios)
        figsize = (subplot_size * cols * (max(col_width_ratios) / avg), rows * subplot_size)
        fig, axes = plt.subplots(
            rows, cols,
            figsize=figsize, dpi=150,
            gridspec_kw={"width_ratios": list(col_width_ratios)},
        )
    else:
        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=150)

    fig.suptitle(title, fontsize=title_fontsize, y=1.00)

    axes_flat = axes.flat if rows * cols > 1 else [axes]

    for idx, ax in enumerate(axes_flat):
        if idx < len(env_ids):
            env_id = env_ids[idx]
            try:
                frame = render_env(env_id, seed)
                ax.imshow(frame)
            except Exception as exc:
                ax.text(
                    0.5, 0.5, f"render failed:\n{exc}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="red",
                )
            ax.set_title(env_id, fontsize=panel_title_fontsize, family="monospace")
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


DEFAULT_FIGURES_DIR = (
    Path(__file__).resolve().parent.parent
    / "docs" / "source" / "_static" / "figures"
)
