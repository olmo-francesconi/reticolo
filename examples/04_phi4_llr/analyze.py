#!/usr/bin/env python3
"""Compare HMC histogram of S with the LLR-reconstructed ln rho(S), and
plot the a_n convergence history per replica.

Top row, one column per kappa:
  ln rho(S) — HMC histogram (markers) vs LLR reconstruction (line),
  aligned at the HMC peak.

Bottom row, one column per kappa:
  a_n vs iteration for each replica through NR warm-up + RM phase.
  Vertical dashed line marks the NR -> RM transition.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE.parent / "_common"))
from llr_reconstruct import (  # noqa: E402
    load_a_history as load_llr,
    load_hmc_action as load_hmc,
    piecewise_log_rho,
    reconstruct_log_rho,
)


def kappa_of(path):
    return float(path.stem.split("kappa")[-1])


def main():
    hmc_files = sorted(RESULTS.glob("hmc_kappa*.h5"))
    if not hmc_files:
        print(f"no inputs in {RESULTS}; run run.sh first", file=sys.stderr)
        return 1

    pairs = []
    for h in hmc_files:
        suffix = h.stem.split("kappa")[-1]
        l = RESULTS / f"llr_kappa{suffix}.h5"
        if not l.exists():
            continue
        pairs.append((kappa_of(h), h, l))
    pairs.sort(key=lambda p: p[0])

    n_cols = len(pairs)
    fig, axes = plt.subplots(
        4, n_cols, figsize=(4.5 * n_cols, 14.0), constrained_layout=True, squeeze=False
    )

    for col, (kappa, hmc_path, llr_path) in enumerate(pairs):
        s_chain = load_hmc(hmc_path)
        llr = load_llr(llr_path)
        e_n = llr["E_n"]
        a_hist = llr["a_hist"]
        a_final = np.median(a_hist[:, -max(1, a_hist.shape[1] // 5):], axis=1)
        log_rho_llr = reconstruct_log_rho(e_n, a_final)

        # One histogram bin per LLR window, each bin centred on E_n with width
        # delta. Makes the HMC dos directly comparable to the per-window LLR
        # estimate.
        delta = llr["delta"]
        bin_edges = np.concatenate([[e_n[0] - delta / 2.0], e_n + delta / 2.0])
        raw_counts, _ = np.histogram(s_chain, bins=bin_edges)
        n_total = len(s_chain)
        counts = raw_counts / (n_total * delta)
        # Naive Poisson sigma(N) ~ sqrt(N) per bin. Under-estimates the true
        # uncertainty because HMC samples are autocorrelated.
        counts_err = np.sqrt(raw_counts) / (n_total * delta)
        centres = e_n
        mask = raw_counts > 0
        log_rho_hmc = np.full_like(counts, np.nan, dtype=float)
        log_rho_hmc[mask] = np.log(counts[mask])
        log_rho_hmc_err = np.full_like(counts, np.nan, dtype=float)
        log_rho_hmc_err[mask] = 1.0 / np.sqrt(raw_counts[mask])

        peak_idx = int(np.nanargmax(log_rho_hmc))
        peak_s = centres[peak_idx]
        peak_log_hmc = log_rho_hmc[peak_idx]
        # Same grid → no interp; align at the HMC peak index directly.
        log_rho_llr_shifted = log_rho_llr + (peak_log_hmc - log_rho_llr[peak_idx])

        # Piecewise-exponential reconstruction: one exp per LLR window.
        pw_x, pw_log = piecewise_log_rho(e_n, a_final, log_rho_llr_shifted, delta)

        # ---------- top: ln rho HMC vs LLR ----------
        ax = axes[0, col]
        ax.errorbar(centres[mask], log_rho_hmc[mask], yerr=log_rho_hmc_err[mask],
                    fmt="x", markersize=4, markeredgewidth=0.8,
                    elinewidth=0.8, capsize=2,
                    color="C0", label="HMC histogram")
        ax.plot(pw_x, pw_log, "-", color="C3", linewidth=0.7,
                label="LLR (piecewise exp)")
        ax.plot(e_n, log_rho_llr_shifted, "x", color="C3", markersize=4,
                markeredgewidth=0.8)
        ax.axvline(peak_s, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_title(rf"$\kappa = {kappa:.3f}$")
        ax.set_xlabel(r"$S$")
        ax.set_ylabel(r"$\ln \rho(S)$ + const.")
        ax.legend(loc="best", fontsize=8)

        # ---------- middle: linear-scale rho(S), unit integral ----------
        ax = axes[1, col]
        pw_log_shifted = pw_log - pw_log.max()
        rho_pw_un = np.exp(pw_log_shifted)
        z_llr = float(np.trapezoid(rho_pw_un, pw_x))
        rho_pw = rho_pw_un / z_llr if z_llr > 0 else rho_pw_un
        rho_centre = np.exp(log_rho_llr_shifted - pw_log.max())
        rho_centre = rho_centre / z_llr if z_llr > 0 else rho_centre

        ax.errorbar(centres, counts, yerr=counts_err, fmt="x", markersize=4,
                    markeredgewidth=0.8, elinewidth=0.8, capsize=2,
                    color="C0", label="HMC histogram")
        ax.plot(pw_x, rho_pw, "-", color="C3", linewidth=0.7,
                label="LLR (piecewise exp)")
        ax.plot(e_n, rho_centre, "x", color="C3", markersize=4,
                markeredgewidth=0.8)
        ax.axvline(peak_s, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$S$")
        ax.set_ylabel(r"$\rho(S)$  ($\int \rho\, dS = 1$)")
        ax.legend(loc="best", fontsize=8)

        # ---------- bottom: a_n history per replica ----------
        ax = axes[2, col]
        n_rep, n_iter = a_hist.shape
        iters = np.arange(n_iter)
        cmap = plt.get_cmap("viridis", n_rep)
        for n in range(n_rep):
            ax.plot(iters, a_hist[n], linewidth=1.0, color=cmap(n),
                    label=f"E_n={e_n[n]:.1f}" if (n_rep <= 8 or n % max(1, n_rep // 6) == 0) else None)
        ax.axvline(llr["n_nr"] - 0.5, color="0.4", linewidth=0.8, linestyle="--",
                   label="NR → RM")
        ax.axhline(0.0, color="0.8", linewidth=0.8)
        ax.set_xlabel("iteration (NR then RM)")
        ax.set_ylabel(r"$a_n$")
        ax.set_title(r"$a_n$ convergence per replica")
        ax.legend(loc="best", fontsize=7, ncol=2)

        # a_n vs E_n: the converged LLR slope as a function of the window
        # centre. Energy-convention `a* = d ln g/dS - 1`, so at any natural
        # peak a* = 0; broken-symmetry / metastable regions show up as a
        # plateau or sign change as a sweeps through the corresponding S.
        ax = axes[3, col]
        ax.plot(e_n, a_final, "o-", color="C2", markersize=4,
                linewidth=0.7, label=r"$a_n$ (LLR final)")
        ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--",
                   label=r"$a = 0$ (natural peak)")
        ax.axvline(peak_s, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$S$")
        ax.set_ylabel(r"$a_n = d \ln g/dS - 1$")
        ax.legend(loc="best", fontsize=8)

    out = HERE / "rho_hmc_vs_llr.pdf"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
