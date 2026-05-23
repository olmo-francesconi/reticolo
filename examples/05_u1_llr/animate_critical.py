#!/usr/bin/env python3
"""Animate the DoS evolution during the NR + RM iterations of a u1_llr run.

Same 2x2 layout as analyze_critical.py, but redrawn frame-by-frame from the
per-iteration `a` slices in /replica_NNN/a. Each frame `k` shows ln rho,
rho, and a_n vs E_n reconstructed from a_hist[:, k]; the bottom-left panel
plots the a_n trace up to that iteration with a vertical cursor.

Inputs come from run_critical.sh:
    results/llr_critical_beta<beta>.h5
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE.parent / "_common"))
from llr_reconstruct import (  # noqa: E402
    load_a_history,
    piecewise_log_rho,
    reconstruct_log_rho,
)


def load_llr(path):
    data = load_a_history(path)
    with h5py.File(path, "r") as f:
        size = int(f["/vars"].attrs["size"])
        ndim = int(f["/vars"].attrs["ndim"])
        beta = float(f["/vars"].attrs["beta"])
    n_plaq = (ndim * (ndim - 1) // 2) * (size ** ndim)
    data["scale"] = float(n_plaq)
    data["beta"] = beta
    data["size"] = size
    data["ndim"] = ndim
    return data


def main():
    paths = sorted(RESULTS.glob("llr_critical_beta*.h5"))
    if not paths:
        print(f"no llr_critical_beta*.h5 in {RESULTS}; run run_critical.sh first",
              file=sys.stderr)
        return 1
    path = paths[-1]

    llr   = load_llr(path)
    e_n   = llr["E_n"]
    a_hist = llr["a_hist"]
    delta = llr["delta"]
    scale = llr["scale"]
    beta  = llr["beta"]
    size  = llr["size"]
    ndim  = llr["ndim"]
    n_nr  = llr["n_nr"]

    n_rep, n_iter = a_hist.shape
    e_n_p = e_n / scale

    # Precompute every frame so the animation loop is just blits.
    pw_x = None
    pw_logs   = []
    log_centres_list = []
    rhos      = []
    for k in range(n_iter):
        a_k = a_hist[:, k]
        log_rho = reconstruct_log_rho(e_n, a_k)
        pw_x_now, pw_log = piecewise_log_rho(e_n, a_k, log_rho, delta)
        if pw_x is None:
            pw_x = pw_x_now
        offset = float(pw_log.max())
        pw_log  -= offset
        log_rho -= offset
        rho_un = np.exp(pw_log)
        z = float(np.trapezoid(rho_un, pw_x / scale))
        rho = rho_un / z if z > 0 else rho_un
        pw_logs.append(pw_log)
        log_centres_list.append(log_rho)
        rhos.append(rho)

    pw_x_p = pw_x / scale

    # Fixed axis limits — anchor to the final frame (with padding) so the
    # eye tracks the shape converging onto the published curve rather than
    # being yanked around by early wild NR steps.
    y_log_min = float(pw_logs[-1].min()) - 1.5
    y_rho_max = max(float(r.max()) for r in rhos[-max(1, n_iter // 10):]) * 1.2
    a_min = float(np.nanmin(a_hist))
    a_max = float(np.nanmax(a_hist))
    a_pad = 0.1 * (a_max - a_min) if a_max > a_min else 0.1
    iters_x = np.arange(n_iter)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), constrained_layout=True)

    # ---- ln rho ----
    ax = axes[0, 0]
    ln_pw, = ax.plot([], [], "-", color="C3", lw=1.0, label="LLR (piecewise exp)")
    pt_c, = ax.plot([], [], "x", color="C3", ms=5, mew=1.0,
                    label="LLR window centres")
    ax.set_xlim(e_n_p.min(), e_n_p.max())
    ax.set_ylim(y_log_min, 1.0)
    ax.set_xlabel(r"$S / (6 V)$")
    ax.set_ylabel(r"$\ln \rho$  (peak = 0)")
    ax.legend(loc="lower center", fontsize=8)

    # ---- rho ----
    ax = axes[0, 1]
    ln_rho, = ax.plot([], [], "-", color="C3", lw=1.0)
    ax.set_xlim(e_n_p.min(), e_n_p.max())
    ax.set_ylim(0.0, y_rho_max)
    ax.set_xlabel(r"$S / (6 V)$")
    ax.set_ylabel(r"$\rho$  (unit integral)")

    # ---- a_n convergence (traces grow with k) ----
    ax = axes[1, 0]
    cmap = plt.get_cmap("viridis", n_rep)
    trace_lines = [ax.plot([], [], lw=1.0, color=cmap(n))[0] for n in range(n_rep)]
    ax.axvline(n_nr - 0.5, color="0.4", lw=0.8, ls="--", label="NR → RM")
    ax.axhline(0.0, color="0.8", lw=0.8)
    cursor = ax.axvline(0, color="C3", lw=1.2, alpha=0.6)
    ax.set_xlim(-0.5, n_iter - 0.5)
    ax.set_ylim(a_min - a_pad, a_max + a_pad)
    ax.set_xlabel("iteration (NR then RM)")
    ax.set_ylabel(r"$a_n$")
    ax.set_title(r"$a_n$ convergence per replica")
    ax.legend(loc="best", fontsize=8)

    # ---- a_n vs E_n ----
    ax = axes[1, 1]
    ln_aE, = ax.plot([], [], "o-", color="C2", ms=4, lw=0.7,
                     label=r"$a_n$ at frame $k$")
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--", label=r"$a = 0$ (natural peak)")
    ax.set_xlim(e_n_p.min(), e_n_p.max())
    ax.set_ylim(a_min - a_pad, a_max + a_pad)
    ax.set_xlabel(r"$S / (6 V)$")
    ax.set_ylabel(r"$a_n = d \ln \rho_{\rm natural} / dS$")
    ax.legend(loc="best", fontsize=8)

    title = fig.suptitle("")

    def update(k):
        ln_pw.set_data(pw_x_p, pw_logs[k])
        pt_c.set_data(e_n_p, log_centres_list[k])
        ln_rho.set_data(pw_x_p, rhos[k])
        for n in range(n_rep):
            trace_lines[n].set_data(iters_x[:k + 1], a_hist[n, :k + 1])
        cursor.set_xdata([k, k])
        ln_aE.set_data(e_n_p, a_hist[:, k])
        if k < n_nr:
            phase, idx = "NR", k + 1
        else:
            phase, idx = "RM", k - n_nr + 1
        title.set_text(
            rf"$L={size}^{{{ndim}}}$, $\beta = {beta:.4f}$  —  "
            rf"{phase} iter {idx}  ({k + 1}/{n_iter})"
        )
        return (ln_pw, pt_c, ln_rho, *trace_lines, cursor, ln_aE, title)

    anim = FuncAnimation(fig, update, frames=n_iter, interval=200, blit=False)

    out = HERE / "rho_critical_evolution.gif"
    anim.save(out, writer=PillowWriter(fps=6))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
