#!/usr/bin/env python3
"""Compare HMC histogram of S with the LLR-reconstructed ln rho(S) for
4D SU(2) Wilson gauge theory, plus a_n convergence panel.

Mirrors the per-beta panel layout of example 04 (U(1)). LLR math is
identical — Mode A (real LLR), so trapezoidal integration of a(E) along
the replica grid reconstructs ln rho_natural directly.
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
    load_hmc_action as load_hmc,
    piecewise_log_rho,
    reconstruct_log_rho,
)


def load_llr(path):
    # Wrap the shared loader to also stash the plaquette-normalisation scale
    # `beta * n_plaq` so plotting code can render the intensive variable
    # ⟨1 − P⟩ = S / (β · n_plaq).
    data = load_a_history(path)
    with h5py.File(path, "r") as f:
        size = int(f["/vars"].attrs["size"])
        ndim = int(f["/vars"].attrs["ndim"])
        beta = float(f["/vars"].attrs["beta"])
    n_plaq = (ndim * (ndim - 1) // 2) * (size ** ndim)
    data["scale"] = beta * n_plaq
    return data


def beta_of(path):
    return float(path.stem.split("beta")[-1])


def main():
    hmc_files = sorted(RESULTS.glob("hmc_beta*.h5"))
    if not hmc_files:
        print(f"no inputs in {RESULTS}; run run.sh first", file=sys.stderr)
        return 1

    pairs = []
    for h in hmc_files:
        suffix = h.stem.split("beta")[-1]
        l = RESULTS / f"llr_beta{suffix}.h5"
        if not l.exists():
            continue
        pairs.append((beta_of(h), h, l))
    pairs.sort(key=lambda p: p[0])

    n_cols = len(pairs)
    fig, axes = plt.subplots(
        4, n_cols, figsize=(4.5 * n_cols, 14.0), constrained_layout=True, squeeze=False
    )

    for col, (beta, hmc_path, llr_path) in enumerate(pairs):
        s_chain = load_hmc(hmc_path)
        llr = load_llr(llr_path)
        e_n = llr["E_n"]
        a_hist = llr["a_hist"]
        a_final = np.median(a_hist[:, -max(1, a_hist.shape[1] // 5):], axis=1)
        log_rho_llr = reconstruct_log_rho(e_n, a_final)

        delta = llr["delta"]
        bin_edges = np.concatenate([[e_n[0] - delta / 2.0], e_n + delta / 2.0])
        raw_counts, _ = np.histogram(s_chain, bins=bin_edges)
        n_total = len(s_chain)
        counts = raw_counts / (n_total * delta)
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
        log_rho_llr_shifted = log_rho_llr + (peak_log_hmc - log_rho_llr[peak_idx])

        pw_x, pw_log = piecewise_log_rho(e_n, a_final, log_rho_llr_shifted, delta)

        # Rescale x to the intensive variable ⟨1 − P⟩ = S / (β·n_plaq).
        scale = llr["scale"]
        centres = centres / scale
        e_n = e_n / scale
        pw_x = pw_x / scale
        peak_s = peak_s / scale
        counts = counts * scale
        counts_err = counts_err * scale

        ax = axes[0, col]
        ax.errorbar(centres[mask], log_rho_hmc[mask],
                    yerr=log_rho_hmc_err[mask],
                    fmt="x", markersize=4, markeredgewidth=0.8,
                    elinewidth=0.8, capsize=2,
                    color="C0", label="HMC histogram")
        ax.plot(pw_x, pw_log, "-", color="C3", linewidth=0.7,
                label="LLR (piecewise exp)")
        ax.plot(e_n, log_rho_llr_shifted, "x", color="C3", markersize=4,
                markeredgewidth=0.8)
        ax.axvline(peak_s, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_title(rf"$\beta = {beta:.3f}$")
        ax.set_xlabel(r"$\langle 1 - (1/N)\,\mathrm{Re\,Tr}\,U_p\rangle = S / (\beta\, n_p)$")
        ax.set_ylabel(r"$\ln \rho$ + const.")
        ax.legend(loc="best", fontsize=8)

        # Linear-scale rho(S), both normalised to unit integral.
        ax = axes[1, col]
        pw_log_shifted = pw_log - pw_log.max()
        rho_pw_un = np.exp(pw_log_shifted)
        z_llr = float(np.trapezoid(rho_pw_un, pw_x))
        rho_pw = rho_pw_un / z_llr if z_llr > 0 else rho_pw_un
        rho_centre = np.exp(log_rho_llr_shifted - pw_log.max())
        rho_centre = rho_centre / z_llr if z_llr > 0 else rho_centre

        ax.errorbar(centres, counts, yerr=counts_err,
                    fmt="x", markersize=4, markeredgewidth=0.8,
                    elinewidth=0.8, capsize=2,
                    color="C0", label="HMC histogram")
        ax.plot(pw_x, rho_pw, "-", color="C3", linewidth=0.7,
                label="LLR (piecewise exp)")
        ax.plot(e_n, rho_centre, "x", color="C3", markersize=4,
                markeredgewidth=0.8)
        ax.axvline(peak_s, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$\langle 1 - (1/N)\,\mathrm{Re\,Tr}\,U_p\rangle$")
        ax.set_ylabel(r"$\rho$  (unit integral)")
        ax.legend(loc="best", fontsize=8)

        ax = axes[2, col]
        n_rep, n_iter = a_hist.shape
        iters = np.arange(n_iter)
        cmap = plt.get_cmap("viridis", n_rep)
        for n in range(n_rep):
            show_label = (n_rep <= 8 or n % max(1, n_rep // 6) == 0)
            ax.plot(iters, a_hist[n], linewidth=1.0, color=cmap(n),
                    label=f"$E_n/\\beta n_p$={e_n[n]:.3f}" if show_label else None)
        ax.axvline(llr["n_nr"] - 0.5, color="0.4", linewidth=0.8, linestyle="--",
                   label="NR → RM")
        ax.axhline(0.0, color="0.8", linewidth=0.8)
        ax.set_xlabel("iteration (NR then RM)")
        ax.set_ylabel(r"$a_n$")
        ax.set_title(r"$a_n$ convergence per replica")
        ax.legend(loc="best", fontsize=7, ncol=2)

        # a_n vs intensive E: standard convention a* = d ln rho_natural/dS,
        # so at any natural peak a* = 0. For SU(2) in 4D (no first-order
        # transition) this is a single monotone curve through zero at the
        # natural peak.
        ax = axes[3, col]
        ax.plot(e_n, a_final, "o-", color="C2", markersize=4,
                linewidth=0.7, label=r"$a_n$ (LLR final)")
        ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--",
                   label=r"$a = 0$ (natural peak)")
        ax.axvline(peak_s, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$\langle 1 - (1/N)\,\mathrm{Re\,Tr}\,U_p\rangle$")
        ax.set_ylabel(r"$a_n = d \ln \rho_{\rm natural} / dS$")
        ax.legend(loc="best", fontsize=8)

    out = HERE / "rho_hmc_vs_llr.pdf"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
