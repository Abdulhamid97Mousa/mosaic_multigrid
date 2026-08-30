"""Regenerate the Competitive (asymmetric) env preview PNG.

Renders the 30 asymmetric competitive variants: 10 per sport
(1v2, 2v1, 1v3, 3v1, 1v4, 4v1, 2v3, 3v2, 2v4, 4v2) for BB, AF, S.
Introduced in mosaic_multigrid v7.0.0.

Layout: 10 rows x 3 cols (portrait/vertical).
Rows: one per matchup (1v2, 2v1, 1v3, 3v1, 1v4, 4v1, 2v3, 3v2, 2v4, 4v2).
Cols: BB, AF, S. Column widths match court sizes (19, 16, 16).

Usage:
    python scripts/generate_competitive_asymmetric_figure.py

Output:
    docs/source/_static/figures/envs_competitive_asymmetric.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _render_utils import DEFAULT_FIGURES_DIR, render_grid


SPORTS = ["BB", "AF", "S"]
MATCHUPS = ["1v2", "2v1", "1v3", "3v1", "1v4", "4v1", "2v3", "3v2", "2v4", "4v2"]

# Row-major, matchup-major: 10 rows (matchups) x 3 cols (sports).
# Reading horizontally across a row lets you compare the same matchup
# across all three sports side-by-side.
ENV_IDS: list[str] = []
for matchup in MATCHUPS:
    for sport in SPORTS:
        ENV_IDS.append(f"MosaicMultiGrid-{sport}-{matchup}-IndAgObs-v1")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate asymmetric competitive env preview grid"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=DEFAULT_FIGURES_DIR / "envs_competitive_asymmetric.png")
    args = p.parse_args()

    print(f"[Competitive asymmetric] rendering {len(ENV_IDS)} envs -> {args.out}")
    render_grid(
        env_ids=ENV_IDS,
        rows=10, cols=3,
        title="Competitive: asymmetric (unequal team sizes) - NEW in v7.0.0",
        out_path=args.out,
        seed=args.seed,
        subplot_size=3.0,
        panel_title_fontsize=8,
        col_width_ratios=[19, 16, 16],   # BB, AF, S grid widths
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
