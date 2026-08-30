"""Regenerate the Solo env preview PNG.

Renders the 6 solo variants (Green + Blue for each sport family):
BB, AF, S. Collect has no solo variant.

Layout: 2 rows (Green, Blue) x 3 cols (BB, AF, S) = 6 panels.

Usage:
    python scripts/generate_solo_figure.py
    python scripts/generate_solo_figure.py --seed 123

Output:
    docs/source/_static/figures/envs_solo.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _render_utils import DEFAULT_FIGURES_DIR, render_grid


# Row-major: Green row first, Blue row second; columns are BB, AF, S
ENV_IDS = [
    # Green solo
    "MosaicMultiGrid-BB-G-1v0-v1",
    "MosaicMultiGrid-AF-G-1v0-v1",
    "MosaicMultiGrid-S-G-1v0-v1",
    # Blue solo
    "MosaicMultiGrid-BB-B-0v1-v1",
    "MosaicMultiGrid-AF-B-0v1-v1",
    "MosaicMultiGrid-S-B-0v1-v1",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=DEFAULT_FIGURES_DIR / "envs_solo.png")
    args = p.parse_args()

    print(f"[Solo] rendering {len(ENV_IDS)} envs -> {args.out}")
    render_grid(
        env_ids=ENV_IDS,
        rows=2, cols=3,
        title="Solo (1 agent, no opponent)",
        out_path=args.out,
        seed=args.seed,
        # Column widths match the grid widths of each sport's court:
        # BB uses a 19-wide grid, AF and S use 16-wide grids. Without this,
        # BB visually looks larger in an equal-cell layout.
        col_width_ratios=[19, 16, 16],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
