"""Regenerate the Competitive (symmetric) env preview PNG.

Renders the 15 symmetric competitive variants:
- BB, AF, S: 1v1, 2v2, 3v3, 4v4 (4 each = 12)
- C: base (3-agent individual), 1v1, 2v2 (3)

Layout: 4 rows x 4 cols = 16 cells, last cell blank.
Rows: BB, AF, S, C.

Usage:
    python scripts/generate_competitive_symmetric_figure.py

Output:
    docs/source/_static/figures/envs_competitive_symmetric.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _render_utils import DEFAULT_FIGURES_DIR, render_grid


# Row-major, 4 rows x 4 cols. Last cell in Collect row is blank (only 3 envs).
ENV_IDS = [
    # Basketball row
    "MosaicMultiGrid-BB-1v1-IndAgObs-v1",
    "MosaicMultiGrid-BB-2v2-IndAgObs-v1",
    "MosaicMultiGrid-BB-3v3-IndAgObs-v1",
    "MosaicMultiGrid-BB-4v4-IndAgObs-v1",
    # American Football row
    "MosaicMultiGrid-AF-1v1-IndAgObs-v1",
    "MosaicMultiGrid-AF-2v2-IndAgObs-v1",
    "MosaicMultiGrid-AF-3v3-IndAgObs-v1",
    "MosaicMultiGrid-AF-4v4-IndAgObs-v1",
    # Soccer row
    "MosaicMultiGrid-S-1v1-IndAgObs-v1",
    "MosaicMultiGrid-S-2v2-IndAgObs-v1",
    "MosaicMultiGrid-S-3v3-IndAgObs-v1",
    "MosaicMultiGrid-S-4v4-IndAgObs-v1",
    # Collect row (only 3 envs, 4th cell will be blank)
    "MosaicMultiGrid-C-IndAgObs-v1",
    "MosaicMultiGrid-C-1v1-IndAgObs-v1",
    "MosaicMultiGrid-C-2v2-IndAgObs-v1",
]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate symmetric competitive env preview grid"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=DEFAULT_FIGURES_DIR / "envs_competitive_symmetric.png")
    args = p.parse_args()

    print(f"[Competitive symmetric] rendering {len(ENV_IDS)} envs -> {args.out}")
    render_grid(
        env_ids=ENV_IDS,
        rows=4, cols=4,
        title="Competitive: symmetric (equal team sizes)",
        out_path=args.out,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
