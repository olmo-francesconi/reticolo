#!/usr/bin/env python3
"""Plot the LLR-reconstructed ln rho(S) / rho(S) at the 4D compact U(1) bulk
transition (beta_c ~ 1.0106), plus the per-replica a_n diagnostics.

Inputs come from run_critical.sh:
    results/llr_critical_beta<beta>.h5
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

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

    # Median of the last 20% of the RM tail kills the 1/(k+1) wobble.
    a_final = np.median(a_hist[:, -max(1, a_hist.shape[1] // 5):], axis=1)

    # log_rho at window centres (trapezoid of a_n) -> piecewise-exp segments.
    # These are continuous by construction at the window boundaries.
    log_rho_centres = reconstruct_log_rho(e_n, a_final)
    pw_x, pw_log = piecewise_log_rho(e_n, a_final, log_rho_centres, delta)

    # Single offset for centres and piecewise so the `x` markers sit on the
    # line. Use pw_log.max() because the absolute max can lie inside a
    # window (between two centres) where a_n changes sign.
    offset = float(pw_log.max())
    pw_log         -= offset
    log_rho_centres -= offset

    # Intensive plaquette variable e_p = S / (beta * 6V). The piecewise
    # density transforms by the Jacobian `scale`; trapezoid in the rescaled
    # variable picks that up via the normalisation step.
    e_n_p  = e_n  / scale
    pw_x_p = pw_x / scale

    rho_un = np.exp(pw_log)
    z = float(np.trapezoid(rho_un, pw_x_p))
    rho = rho_un / z if z > 0 else rho_un
    rho_centres = np.exp(log_rho_centres) / z if z > 0 else np.exp(log_rho_centres)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), constrained_layout=True)

    # ---- ln rho ----------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(pw_x_p, pw_log, "-", color="C3", linewidth=1.0,
            label="LLR (piecewise exp)")
    ax.plot(e_n_p, log_rho_centres, "x", color="C3", markersize=5,
            markeredgewidth=1.0, label="LLR window centres")
    ax.set_xlabel(r"$S / (6 V)$")
    ax.set_ylabel(r"$\ln \rho$  (peak = 0)")
    ax.set_title(rf"$L={size}^{{{ndim}}}$, $\beta = {beta:.4f}$")
    ax.legend(loc="best", fontsize=9)

    # ---- rho (linear, unit integral) -------------------------------------
    ax = axes[0, 1]
    ax.plot(pw_x_p, rho, "-", color="C3", linewidth=1.0,
            label=r"$\rho(S)$  (unit integral)")
    ax.plot(e_n_p, rho_centres, "x", color="C3", markersize=5,
            markeredgewidth=1.0)
    ax.set_xlabel(r"$S / (6 V)$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title(r"density of states — double peak at $\beta_c$")
    ax.legend(loc="best", fontsize=9)

    # ---- a_n convergence per replica -------------------------------------
    ax = axes[1, 0]
    n_rep, n_iter = a_hist.shape
    iters = np.arange(n_iter)
    cmap = plt.get_cmap("viridis", n_rep)
    for n in range(n_rep):
        show_label = (n_rep <= 8 or n % max(1, n_rep // 6) == 0)
        ax.plot(iters, a_hist[n], linewidth=1.0, color=cmap(n),
                label=f"$e_p$={e_n_p[n]:.3f}" if show_label else None)
    ax.axvline(n_nr - 0.5, color="0.4", linewidth=0.8, linestyle="--",
               label="NR → RM")
    ax.axhline(0.0, color="0.8", linewidth=0.8)
    ax.set_xlabel("iteration (NR then RM)")
    ax.set_ylabel(r"$a_n$")
    ax.set_title(r"$a_n$ convergence per replica")
    ax.legend(loc="best", fontsize=7, ncol=2)

    # ---- a_n vs E_n: the LLR slope = d ln rho_natural / dS ---------------
    # Zero crossings mark natural peaks; the stretch between zero crossings
    # is the metastable region between the two phases.
    ax = axes[1, 1]
    ax.plot(e_n_p, a_final, "o-", color="C2", markersize=4,
            linewidth=0.7, label=r"$a_n$ (LLR final)")
    ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--",
               label=r"$a = 0$ (natural peak)")
    ax.set_xlabel(r"$S / (6 V)$")
    ax.set_ylabel(r"$a_n = d \ln \rho_{\rm natural} / dS$")
    ax.legend(loc="best", fontsize=9)

    out = HERE / "rho_critical_doublepeak.pdf"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
