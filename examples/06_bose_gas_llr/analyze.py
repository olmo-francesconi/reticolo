#!/usr/bin/env python3
"""LLR-vs-HMC diagnostic plot for the 4D Bose gas at finite chemical
potential. Per-µ panel layout mirrors examples 04 and 05:

  Row 0:  ln ρ(s) — HMC histogram (markers) vs LLR piecewise-exp (line).
  Row 1:  ρ(s)    — linear, unit integral, same overlay.
  Row 2:  a_n     — convergence per replica across NR + RM iterations.
  Row 3:  a_n vs s — converged LLR slope = d ln ρ_pq/dS_I.

S_I distribution is symmetric (ρ(S_I) = ρ(−S_I)) — we run the LLR only on
[0, S_I_max), so all per-µ panels use the intensive variable s = S_I / V on
the x-axis with a vertical line marking s = 0 (the natural peak by symmetry).
The HMC chain is folded via |S_I| so both halves contribute to the
histogram; the first bin (the s = 0 boundary) is half-width and uses the
appropriately scaled density estimator.
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
    reconstruct_log_rho,
)


def load_hmc(path):
    # S_I (imaginary action) — bose gas is the only mode-B LLR app, so
    # the HMC twin records `/prod/obs/s_i` rather than `/prod/obs/s`.
    with h5py.File(path, "r") as f:
        return np.asarray(f["/prod/obs/s_i"][:], dtype=float)


def load_llr(path):
    # Wrap the shared loader to also stash mu / volume / size / ndim — the
    # plot code uses `volume` to render the intensive variable s = S_I / V.
    data = load_a_history(path)
    with h5py.File(path, "r") as f:
        size = int(f["/vars"].attrs["size"])
        ndim = int(f["/vars"].attrs["ndim"])
        mu   = float(f["/vars"].attrs["mu"])
    data["size"]   = size
    data["ndim"]   = ndim
    data["mu"]     = mu
    data["volume"] = size ** ndim
    return data


def piecewise_log_rho_symmetric(e_n, a_final, log_rho, delta, n_per_window=16):
    """Finely-sampled piecewise-exponential reconstruction on [−E_max, +E_max].

    LLR ran on [0, S_max] with windows centred at `E_n = δ/2, 3δ/2, ..., S_max
    − δ/2` (the half-step shift means no window sits on S_I = 0). By the
    symmetry ρ(S_I) = ρ(−S_I) the negative-side reconstruction has the same
    `log_rho[n]` value at `−E_n` and slope `−a_final[n]`. Every window is
    mirrored — no skip, no duplicates.
    """
    xs, ys = [], []
    half = 0.5 * delta
    for n in range(len(e_n) - 1, -1, -1):
        ctr   = -e_n[n]
        seg_x = np.linspace(ctr - half, ctr + half, n_per_window)
        seg_y = log_rho[n] + (-a_final[n]) * (seg_x - ctr)
        xs.append(seg_x)
        ys.append(seg_y)
    for n in range(len(e_n)):
        seg_x = np.linspace(e_n[n] - half, e_n[n] + half, n_per_window)
        seg_y = log_rho[n] + a_final[n] * (seg_x - e_n[n])
        xs.append(seg_x)
        ys.append(seg_y)
    return np.concatenate(xs), np.concatenate(ys)


def mu_of(path):
    return float(path.stem.split("mu")[-1])


def main():
    hmc_files = sorted(RESULTS.glob("hmc_mu*.h5"))
    if not hmc_files:
        print(f"no inputs in {RESULTS}; run run.sh first", file=sys.stderr)
        return 1

    pairs = []
    for h in hmc_files:
        suffix = h.stem.split("mu")[-1]
        l = RESULTS / f"llr_mu{suffix}.h5"
        if not l.exists():
            continue
        pairs.append((mu_of(h), h, l))
    pairs.sort(key=lambda p: p[0])

    n_cols = len(pairs)
    fig, axes = plt.subplots(
        4, n_cols, figsize=(4.5 * n_cols, 14.0), constrained_layout=True, squeeze=False
    )

    for col, (mu, hmc_path, llr_path) in enumerate(pairs):
        s_chain = load_hmc(hmc_path)
        llr     = load_llr(llr_path)
        e_n     = llr["E_n"]
        delta   = llr["delta"]
        volume  = llr["volume"]
        a_hist  = llr["a_hist"]
        a_final = np.median(a_hist[:, -max(1, a_hist.shape[1] // 5):], axis=1)
        log_rho_llr = reconstruct_log_rho(e_n, a_final)

        # Symmetric grid for plotting: LLR ran on [0, S_max] but ρ is
        # symmetric so we mirror to display the full bell curve. With the
        # half-step shift the innermost window centre is at S_I = ±δ/2; no
        # element on either side coincides at S_I = 0, so the mirror is
        # full (no skip).
        e_n_sym       = np.concatenate([-e_n[::-1], e_n])
        log_rho_sym   = np.concatenate([log_rho_llr[::-1], log_rho_llr])
        a_final_sym   = np.concatenate([-a_final[::-1], a_final])

        # HMC histogram on the full symmetric grid — no folding; bin centres
        # at e_n_sym, bin width = δ throughout (including the bin centred on
        # zero, which then naturally absorbs samples on both sides of 0).
        bin_edges = np.concatenate([[e_n_sym[0] - delta / 2.0],
                                    e_n_sym + delta / 2.0])
        n_total   = len(s_chain)
        raw_counts, _ = np.histogram(s_chain, bins=bin_edges)
        density_hmc   = raw_counts / (n_total * delta)
        density_err   = np.sqrt(raw_counts) / (n_total * delta)
        mask          = raw_counts > 0
        log_density_hmc       = np.full_like(density_hmc, np.nan, dtype=float)
        log_density_hmc[mask] = np.log(density_hmc[mask])
        log_density_hmc_err   = np.full_like(density_hmc, np.nan, dtype=float)
        log_density_hmc_err[mask] = 1.0 / np.sqrt(raw_counts[mask])

        # Align LLR to HMC at the innermost positive window (S_I = +δ/2);
        # no window sits exactly at S_I = 0 with the half-step shift.
        inner_idx_hmc = len(e_n)  # in e_n_sym, +δ/2 sits here
        inner_idx_llr = 0         # log_rho_llr[0] is the +δ/2 value
        log_rho_llr_shifted = log_rho_llr + (
            log_density_hmc[inner_idx_hmc] - log_rho_llr[inner_idx_llr]
        )
        # Rebuild the symmetric copy with the shift baked in (full mirror).
        log_rho_sym_shifted = np.concatenate(
            [log_rho_llr_shifted[::-1], log_rho_llr_shifted]
        )
        pw_x, pw_log = piecewise_log_rho_symmetric(
            e_n, a_final, log_rho_llr_shifted, delta
        )

        # Intensive x-axis: s = S_I / V.
        e_n_iv  = e_n_sym / volume
        pw_x_iv = pw_x / volume

        # Row 0: ln ρ overlay.
        ax = axes[0, col]
        ax.errorbar(e_n_iv[mask], log_density_hmc[mask], yerr=log_density_hmc_err[mask],
                    fmt="x", markersize=4, markeredgewidth=0.8,
                    elinewidth=0.8, capsize=2, color="C0", label="HMC histogram")
        ax.plot(pw_x_iv, pw_log, "-", color="C3", linewidth=0.7,
                label="LLR (piecewise exp)")
        ax.plot(e_n_iv, log_rho_sym_shifted, "x", color="C3", markersize=4,
                markeredgewidth=0.8)
        ax.axvline(0.0, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_title(rf"$\mu = {mu:.2f}$")
        ax.set_xlabel(r"$S_I / V$")
        ax.set_ylabel(r"$\ln \rho(S_I)$ + const.")
        ax.legend(loc="best", fontsize=8)

        # Row 1: ρ(S_I) linear, unit integral over the plotted x-range.
        ax = axes[1, col]
        pw_log_shifted = pw_log - pw_log.max()
        rho_pw_un = np.exp(pw_log_shifted)
        z_llr     = float(np.trapezoid(rho_pw_un, pw_x_iv))
        rho_pw    = rho_pw_un / z_llr if z_llr > 0 else rho_pw_un
        rho_centre_un = np.exp(log_rho_sym_shifted - pw_log.max())
        rho_centre    = rho_centre_un / z_llr if z_llr > 0 else rho_centre_un

        z_hmc            = float(np.trapezoid(density_hmc, e_n_iv))
        density_hmc_norm = density_hmc / z_hmc if z_hmc > 0 else density_hmc
        density_err_norm = density_err / z_hmc if z_hmc > 0 else density_err

        ax.errorbar(e_n_iv, density_hmc_norm, yerr=density_err_norm,
                    fmt="x", markersize=4, markeredgewidth=0.8,
                    elinewidth=0.8, capsize=2, color="C0", label="HMC histogram")
        ax.plot(pw_x_iv, rho_pw, "-", color="C3", linewidth=0.7,
                label="LLR (piecewise exp)")
        ax.plot(e_n_iv, rho_centre, "x", color="C3", markersize=4,
                markeredgewidth=0.8)
        ax.axvline(0.0, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$S_I / V$")
        ax.set_ylabel(r"$\rho(S_I)$  (unit integral)")
        ax.legend(loc="best", fontsize=8)

        # Row 2: a_n history per replica.
        ax = axes[2, col]
        n_rep, n_iter = a_hist.shape
        iters = np.arange(n_iter)
        cmap = plt.get_cmap("viridis", n_rep)
        for n in range(n_rep):
            show_label = (n_rep <= 8 or n % max(1, n_rep // 6) == 0)
            ax.plot(iters, a_hist[n], linewidth=1.0, color=cmap(n),
                    label=f"$S_I/V$={e_n[n] / volume:.3f}" if show_label else None)
        ax.axvline(llr["n_nr"] - 0.5, color="0.4", linewidth=0.8, linestyle="--",
                   label="NR → RM")
        ax.axhline(0.0, color="0.8", linewidth=0.8)
        ax.set_xlabel("iteration (NR then RM)")
        ax.set_ylabel(r"$a_n$")
        ax.set_title(r"$a_n$ convergence per replica")
        ax.legend(loc="best", fontsize=7, ncol=2)

        # Row 3: converged a_n vs S_I/V. By ρ(−s) = ρ(s) the slope is
        # antisymmetric — we mirror the LLR-sampled positive half onto the
        # negative axis so the symmetry is visible.
        ax = axes[3, col]
        ax.plot(e_n_iv, a_final_sym, "o-", color="C2", markersize=4,
                linewidth=0.7, label=r"$a_n$ (LLR final, mirrored)")
        ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--",
                   label=r"$a = 0$ (natural peak)")
        ax.axvline(0.0, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$S_I / V$")
        ax.set_ylabel(r"$a_n = d \ln \rho_{pq}/dS_I$")
        ax.legend(loc="best", fontsize=8)

    out = HERE / "rho_hmc_vs_llr.pdf"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
