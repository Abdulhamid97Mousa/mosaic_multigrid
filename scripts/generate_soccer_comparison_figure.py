"""Composite the original upstream vs corrected Soccer renderings into
a single 1x2 side-by-side PNG for use in the JOSS paper.

Source images:
    figures/original-soccer.png    (upstream gym-multigrid rendering)
    figures/extended-soccer.png    (corrected mosaic_multigrid rendering)

Output:
    figures/soccer_comparison.png  (1 row, 2 columns, composite)

Usage:
    cd 3rd_party/environments/mosaic_multigrid
    python scripts/generate_soccer_comparison_figure.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


PANELS = [
    ("original-soccer.png", "Upstream gym-multigrid (Fickinger 2020)"),
    ("extended-soccer.png", "Corrected mosaic_multigrid (this work)"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--figures-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "figures")
    p.add_argument("--out-name", type=str, default="soccer_comparison.png")
    args = p.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for ax, (fname, subtitle) in zip(axes, PANELS):
        src = args.figures_dir / fname
        if not src.exists():
            ax.text(0.5, 0.5, f"missing:\n{src.name}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="red")
        else:
            ax.imshow(mpimg.imread(src))
        ax.set_title(subtitle, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    out = args.figures_dir / args.out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
