"""Regenerate the Cooperative env preview PNG.

Renders the 30 cooperative variants: 10 per sport (5 Green Nv0 + 5 Blue 0vN)
for BB, AF, S. Team sizes: 2, 3, 4, 5, 6 agents.

Layout: 6 rows x 5 cols = 30 panels.
Rows: BB-Green, BB-Blue, AF-Green, AF-Blue, S-Green, S-Blue.
Cols: N = 2, 3, 4, 5, 6.

Usage:
    python scripts/generate_cooperative_figure.py

Output:
    docs/source/_static/figures/envs_cooperative.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _render_utils import DEFAULT_FIGURES_DIR, render_grid


SPORTS = ["BB", "AF", "S"]
SIZES = [2, 3, 4, 5, 6]

# Row-major:
#   Row 0: BB Green 2v0..6v0
#   Row 1: BB Blue  0v2..0v6
#   Row 2: AF Green 2v0..6v0
#   Row 3: AF Blue  0v2..0v6
#   Row 4: S  Green 2v0..6v0
#   Row 5: S  Blue  0v2..0v6
ENV_IDS: list[str] = []
for sport in SPORTS:
    for n in SIZES:
        ENV_IDS.append(f"MosaicMultiGrid-{sport}-G-{n}v0-IndAgObs-v1")
    for n in SIZES:
        ENV_IDS.append(f"MosaicMultiGrid-{sport}-B-0v{n}-IndAgObs-v1")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate cooperative env preview grid"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=DEFAULT_FIGURES_DIR / "envs_cooperative.png")
    args = p.parse_args()

    print(f"[Cooperative] rendering {len(ENV_IDS)} envs -> {args.out}")
    render_grid(
        env_ids=ENV_IDS,
        rows=6, cols=5,
        title="Cooperative (same-team, no opponent)",
        out_path=args.out,
        seed=args.seed,
        subplot_size=2.5,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
