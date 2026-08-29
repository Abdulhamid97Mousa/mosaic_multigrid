"""Regenerate the sport-family env preview PNGs used in README.md.

For each sport family (BB, AF, S) renders the 9 canonical public envs
(2 solo + 3 symmetric competitive + 4 one-sided cooperative) in a 3×3
grid, one panel per env. TeamObs was removed in v7.0.0.

Note: the higher-agent-count envs (4v0/5v0/6v0/0v4/0v5/0v6 and 4v4)
are registered and usable but intentionally omitted here — they are
experimental / not part of the canonical showcase.

Usage:
    cd 3rd_party/environments/mosaic_multigrid
    python scripts/generate_env_figures.py                 # all 3 sports
    python scripts/generate_env_figures.py --sport BB      # single sport
    python scripts/generate_env_figures.py --seed 123      # different seed

Outputs:
    figures/envs_BB_v2.png
    figures/envs_AF_v2.png
    figures/envs_S_v2.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import mosaic_multigrid  # triggers gymnasium registration
import gymnasium


ENV_LISTS: dict[str, list[str]] = {
    "BB": [
        "MosaicMultiGrid-BB-G-1v0-v1",
        "MosaicMultiGrid-BB-B-0v1-v1",
        "MosaicMultiGrid-BB-1v1-IndAgObs-v1",
        "MosaicMultiGrid-BB-2v2-IndAgObs-v1",
        "MosaicMultiGrid-BB-3v3-IndAgObs-v1",
        "MosaicMultiGrid-BB-G-2v0-IndAgObs-v1",
        "MosaicMultiGrid-BB-G-3v0-IndAgObs-v1",
        "MosaicMultiGrid-BB-B-0v2-IndAgObs-v1",
        "MosaicMultiGrid-BB-B-0v3-IndAgObs-v1",
    ],
    "AF": [
        "MosaicMultiGrid-AF-G-1v0-v1",
        "MosaicMultiGrid-AF-B-0v1-v1",
        "MosaicMultiGrid-AF-1v1-IndAgObs-v1",
        "MosaicMultiGrid-AF-2v2-IndAgObs-v1",
        "MosaicMultiGrid-AF-3v3-IndAgObs-v1",
        "MosaicMultiGrid-AF-G-2v0-IndAgObs-v1",
        "MosaicMultiGrid-AF-G-3v0-IndAgObs-v1",
        "MosaicMultiGrid-AF-B-0v2-IndAgObs-v1",
        "MosaicMultiGrid-AF-B-0v3-IndAgObs-v1",
    ],
    "S": [
        "MosaicMultiGrid-S-G-1v0-v1",
        "MosaicMultiGrid-S-B-0v1-v1",
        "MosaicMultiGrid-S-1v1-IndAgObs-v1",
        "MosaicMultiGrid-S-2v2-IndAgObs-v1",
        "MosaicMultiGrid-S-3v3-IndAgObs-v1",
        "MosaicMultiGrid-S-G-2v0-IndAgObs-v1",
        "MosaicMultiGrid-S-G-3v0-IndAgObs-v1",
        "MosaicMultiGrid-S-B-0v2-IndAgObs-v1",
        "MosaicMultiGrid-S-B-0v3-IndAgObs-v1",
    ],
}

SPORT_TITLE = {
    "BB": "Basketball (BB) — 19×11 grid",
    "AF": "American Football (AF) — 16×11 grid",
    "S":  "Soccer (S) — 16×11 grid",
}


def render_env(env_id: str, seed: int) -> np.ndarray:
    env = gymnasium.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    frame = env.render()
    env.close()
    return frame


def make_figure(sport: str, seed: int, out_path: Path) -> None:
    env_ids = ENV_LISTS[sport]
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), dpi=150)
    fig.suptitle(SPORT_TITLE[sport], fontsize=16, y=1.00)

    for ax, env_id in zip(axes.flat, env_ids):
        try:
            frame = render_env(env_id, seed)
            ax.imshow(frame)
        except Exception as exc:
            ax.text(0.5, 0.5, f"render failed:\n{exc}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="red")
        ax.set_title(env_id, fontsize=9, family="monospace")
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sport", choices=["BB", "AF", "S", "all"], default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--figures-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "figures")
    args = p.parse_args()

    sports = ["BB", "AF", "S"] if args.sport == "all" else [args.sport]
    for sport in sports:
        out = args.figures_dir / f"envs_{sport}_v2.png"
        n = len(ENV_LISTS[sport])
        print(f"[{sport}] rendering {n} envs → {out}")
        make_figure(sport, args.seed, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
