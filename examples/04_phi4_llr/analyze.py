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

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def load_hmc(path):
    with h5py.File(path, "r") as f:
        return np.asarray(f["/prod/obs/s"][:], dtype=float)


def load_llr(path):
    with h5py.File(path, "r") as f:
        n_rep = int(f["/cfg"].attrs["n_rep"])
        n_nr = int(f["/cfg"].attrs["n_nr"])
        n_rm = int(f["/cfg"].attrs["n_rm"])
        delta = float(f["/cfg"].attrs["delta"])
        e_n = np.asarray(f["/cfg/E_n"][:], dtype=float)
        a_hist = np.empty((n_rep, n_nr + n_rm), dtype=float)
        for n in range(n_rep):
            a = np.asarray(f[f"/replica_{n:03d}/a"][:], dtype=float)
            # The series length should equal n_nr + n_rm; be defensive anyway.
            m = min(len(a), n_nr + n_rm)
            a_hist[n, :m] = a[:m]
            if m < n_nr + n_rm:
                a_hist[n, m:] = a[-1]
    return {"E_n": e_n, "a_hist": a_hist, "n_nr": n_nr, "n_rm": n_rm, "delta": delta}


def reconstruct_log_rho(e_n, a_final):
    log_rho = np.zeros_like(e_n)
    for i in range(1, len(e_n)):
        log_rho[i] = log_rho[i - 1] + 0.5 * (a_final[i - 1] + a_final[i]) * (e_n[i] - e_n[i - 1])
    return log_rho


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
        2, n_cols, figsize=(4.5 * n_cols, 8.0), constrained_layout=True, squeeze=False
    )

    for col, (kappa, hmc_path, llr_path) in enumerate(pairs):
        s_chain = load_hmc(hmc_path)
        llr = load_llr(llr_path)
        e_n = llr["E_n"]
        a_hist = llr["a_hist"]
        a_final = np.median(a_hist[:, -max(1, a_hist.shape[1] // 5):], axis=1)
        log_rho_llr = reconstruct_log_rho(e_n, a_final)

        # ---------- top: ln rho HMC vs LLR ----------
        ax = axes[0, col]
        n_bins = max(20, int(np.sqrt(len(s_chain))))
        counts, edges = np.histogram(s_chain, bins=n_bins, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        mask = counts > 0
        log_rho_hmc = np.full_like(counts, np.nan, dtype=float)
        log_rho_hmc[mask] = np.log(counts[mask])

        peak_idx = int(np.nanargmax(log_rho_hmc))
        peak_s = centres[peak_idx]
        peak_log_hmc = log_rho_hmc[peak_idx]
        peak_log_llr = float(np.interp(peak_s, e_n, log_rho_llr))
        log_rho_llr_shifted = log_rho_llr + (peak_log_hmc - peak_log_llr)

        ax.plot(centres, log_rho_hmc, marker="o", linewidth=0, markersize=3,
                color="C0", label="HMC histogram")
        ax.plot(e_n, log_rho_llr_shifted, "-", color="C3", linewidth=1.6,
                label="LLR (aligned)")
        ax.axvline(peak_s, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_title(rf"$\kappa = {kappa:.3f}$")
        ax.set_xlabel(r"$S$")
        ax.set_ylabel(r"$\ln \rho(S)$ + const.")
        ax.legend(loc="best", fontsize=8)

        # ---------- bottom: a_n history per replica ----------
        ax = axes[1, col]
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

    out = HERE / "rho_hmc_vs_llr.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
