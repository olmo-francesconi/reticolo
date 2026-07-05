#!/usr/bin/env python3
"""Render the README performance charts as light/dark SVG pairs.

Two charts, each answering one question a lattice practitioner asks:

  * integrator efficiency — energy violation σ(ΔH) vs cost for the three MD
    integrators, from `apps/bench_integrator_efficiency` (φ⁴ and Wilson<SU3>);
  * GPU scaling — double-precision HMC trajectory throughput vs lattice volume
    on FP64 datacenter cards, from the Modal sweeps.

Edit the data tables below when a benchmark is re-run, then:

    python scripts/make_readme_charts.py     # writes docs/bench_*_{light,dark}.svg

Palette from the validated categorical set (blue / aqua / yellow), colourblind
safe in both modes; light-mode aqua/yellow carry direct labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

OUT = Path(__file__).resolve().parent.parent / "docs"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", series=["#2a78d6", "#1baf7a", "#eda100"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", series=["#3987e5", "#199e70", "#c98500"]),
}


def base_rc(t):
    return {
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "savefig.transparent": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.titlecolor": t["ink"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.edgecolor": t["axis"],
        "axes.linewidth": 1.0, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.grid": True, "axes.grid.axis": "both",
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "figure.dpi": 100,
    }


def style_ax(ax, t):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["axis"])


# ---- integrator efficiency (apps/bench_integrator_efficiency) ---------------
# energy violation sigma(dH) vs force evaluations per trajectory (thermalised);
# log-log slope = integrator order (Leapfrog/Omelyan2 ~ -2, Omelyan4 ~ -4)
INTEG = {
    "phi4": {
        "Leapfrog": ([13, 17, 25, 33, 49, 65], [0.689, 0.408, 0.164, 0.0912, 0.0406, 0.0229]),
        "Omelyan2": ([13, 17, 25, 33, 49, 65], [0.322, 0.175, 0.0743, 0.0432, 0.0195, 0.0113]),
        "Omelyan4": ([17, 25, 33, 49, 65], [0.251, 0.0327, 0.0100, 0.00197, 0.000604]),
    },
    "su3": {
        "Leapfrog": ([17, 25, 33, 49, 65], [1.549, 0.694, 0.389, 0.179, 0.0941]),
        "Omelyan2": ([13, 17, 25, 33, 49, 65], [0.982, 0.483, 0.227, 0.129, 0.0581, 0.0309]),
        "Omelyan4": ([17, 25, 33, 49, 65], [0.893, 0.123, 0.0377, 0.00692, 0.00203]),
    },
}
INTEG_LABEL = {"phi4": "φ⁴ (scalar)", "su3": "SU(3) (gauge)"}
INTEG_ORDER = ["Leapfrog", "Omelyan2", "Omelyan4"]

# ---- GPU scaling (Modal sweeps): HMC trajectory throughput, G dof-updates/s --
_n = np.nan
GPU = {
    "phi4": (["16", "32", "64", "128", "192"], {
        "A100": [0.32, 0.74, 0.88, 0.85, _n], "H100": [0.61, 1.55, 1.62, 1.53, 1.51],
        "B200": [0.67, 2.68, 2.93, 2.93, 2.87]}),
    "su3": (["8", "16", "32", "56", "72"], {
        "A100": [0.66, 0.84, 0.92, 0.86, _n], "H100": [1.15, 2.12, 2.09, 1.89, _n],
        "B200": [1.13, 3.09, 2.90, 2.75, 2.75]}),
}
COL_LABEL = {"phi4": "φ⁴ (scalar)", "su3": "SU(3) (gauge)"}


def chart_integrator(mode):
    t = THEMES[mode]
    colour = dict(zip(INTEG_ORDER, t["series"]))
    with mpl.rc_context(base_rc(t)):
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0), sharey=True)
        for c, theory in enumerate(("phi4", "su3")):
            ax = axes[c]
            for name in INTEG_ORDER:
                x, y = INTEG[theory][name]
                ax.plot(x, y, marker="o", ms=5.5, lw=2, color=colour[name], label=name, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([13, 25, 49, 65])
            ax.set_xticklabels(["13", "25", "49", "65"])
            ax.get_xaxis().set_minor_formatter(NullFormatter())
            ax.set_xlim(11, 76)
            ax.set_title(INTEG_LABEL[theory], fontsize=11, fontweight="bold", color=t["ink"], pad=8)
            ax.set_xlabel("force evaluations / trajectory", fontsize=9.5)
            if c == 0:
                ax.set_ylabel("energy violation  σ(ΔH)   ·   lower = better", fontsize=9.5)
            style_ax(ax, t)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center",
                   fontsize=10, labelcolor=t["ink2"], bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("Integrator efficiency  ·  slope = order (2nd vs 4th); Omelyan4 falls off fastest",
                     x=0.02, ha="left", fontsize=12, fontweight="bold", color=t["ink"])
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(OUT / f"bench_integrator_{mode}.svg")
        plt.close(fig)


def chart_gpu(mode):
    t = THEMES[mode]
    gc = dict(zip(("A100", "H100", "B200"), t["series"]))
    with mpl.rc_context(base_rc(t)):
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7))
        for c, theory in enumerate(("phi4", "su3")):
            ax = axes[c]
            labels, series = GPU[theory]
            xi = np.arange(len(labels))
            for gpu, ys in series.items():
                ax.plot(xi, np.array(ys, float), marker="o", ms=5.5, lw=2,
                        color=gc[gpu], label=gpu, zorder=3)
            ax.set_xticks(xi, [f"{L}⁴" for L in labels])
            ax.set_title(COL_LABEL[theory], fontsize=11, fontweight="bold", color=t["ink"], pad=8)
            if c == 0:
                ax.set_ylabel("HMC throughput\nG dof-updates / s", fontsize=9.5)
            ax.margins(y=0.18)
            ax.set_ylim(bottom=0)
            style_ax(ax, t)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center",
                   fontsize=10, labelcolor=t["ink2"], bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("GPU scaling  ·  double precision, by lattice volume",
                     x=0.02, ha="left", fontsize=12, fontweight="bold", color=t["ink"])
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(OUT / f"bench_gpu_{mode}.svg")
        plt.close(fig)


def main():
    for mode in ("light", "dark"):
        chart_integrator(mode)
        chart_gpu(mode)
    print("wrote", *sorted(p.name for p in OUT.glob("bench_*.svg")))


if __name__ == "__main__":
    main()
