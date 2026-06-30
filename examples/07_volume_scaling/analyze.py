#!/usr/bin/env python3
"""Plot kernel wall-time vs. dof count from a bench_volume_scaling CSV.

Layout: rows = ndim, cols = kernel (s_full, compute_force). One line per
action. Axes are log-log; a least-squares slope on log(wall_s) vs.
log(dofs) is fitted per (action, kernel, ndim) and reported in the
legend — perfectly-local kernels land on slope ≈ 1, deviations expose
cache cliffs or super-linear cost (e.g. non-local stencils, allocator
warmup).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ACTION_COLORS = {
    "phi4":       "tab:blue",
    "compact_u1": "tab:orange",
    "wilson_su2": "tab:green",
    "wilson_su3": "tab:red",
}
KERNELS = ("s_full", "compute_force")


def fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    lx, ly = np.log(x), np.log(y)
    slope, _ = np.polyfit(lx, ly, 1)
    return float(slope)


def main(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    ndims = sorted(df["ndim"].unique())
    actions = [a for a in ACTION_COLORS if a in set(df["action"])]
    # NB: df.ndim is a pandas property (= 2 for any DataFrame), so always
    # subscript the column by name to filter by lattice dimension.

    fig, axes = plt.subplots(
        len(ndims), len(KERNELS),
        figsize=(5.0 * len(KERNELS), 3.4 * len(ndims)),
        sharex=False, sharey=False, squeeze=False,
    )

    for i, ndim in enumerate(ndims):
        for j, kernel in enumerate(KERNELS):
            ax = axes[i][j]
            xs_all: list[float] = []
            ys_all: list[float] = []
            for action in actions:
                sub = (df[(df["ndim"] == ndim) & (df["kernel"] == kernel) &
                          (df["action"] == action)]
                       .sort_values(by=["dofs"]))
                if sub.empty:
                    continue
                x = sub["dofs"].to_numpy(dtype=float)
                y = sub["wall_s"].to_numpy(dtype=float)
                xs_all.extend(x.tolist())
                ys_all.extend(y.tolist())
                slope = fit_slope(x, y)
                ax.plot(x, y,
                        marker="o", linestyle="-",
                        color=ACTION_COLORS[action],
                        label=f"{action}  (α≈{slope:.2f})")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"ndim={ndim}  ·  {kernel}")
            ax.set_xlabel("dofs")
            ax.set_ylabel("wall time per call [s]")
            ax.grid(True, which="both", alpha=0.3)
            # Reference slope-1 guide line anchored at the bottom-left data
            # point. Ideal local kernels (cost ∝ dofs) should lie parallel
            # to this; rising above it = super-linear cost (cache cliffs).
            if xs_all:
                x_arr = np.asarray(xs_all, dtype=float)
                y_arr = np.asarray(ys_all, dtype=float)
                x0 = float(x_arr.min())
                # Anchor on the lowest *y* among smallest-x points so the
                # guide sits at the floor of the panel.
                y0 = float(y_arr[x_arr == x0].min())
                xs_guide = np.array([x0, float(x_arr.max())])
                ys_guide = y0 * (xs_guide / x0)
                ax.plot(xs_guide, ys_guide,
                        color="0.55", linestyle="--", linewidth=1.0,
                        label="α = 1 (ideal)", zorder=0)
            if ax.has_data():
                ax.legend(fontsize=8, frameon=False, loc="best")

    fig.suptitle("Kernel wall-time vs. lattice dof count "
                 "(slope α from log-log fit; ideal local kernel: α ≈ 1)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: analyze.py <scaling.csv> <out.pdf>", file=sys.stderr)
        sys.exit(2)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
