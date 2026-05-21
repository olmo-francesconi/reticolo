#!/usr/bin/env python3
"""Compare HMC histogram of S with the LLR-reconstructed ln rho(S) for
compact U(1), plus a_n convergence panel.

Per-beta panel layout mirrors example 04 (phi^4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

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
            m = min(len(a), n_nr + n_rm)
            a_hist[n, :m] = a[:m]
            if m < n_nr + n_rm:
                a_hist[n, m:] = a[-1]
        size = int(f["/vars"].attrs["size"])
        ndim = int(f["/vars"].attrs["ndim"])
        beta = float(f["/vars"].attrs["beta"])
    # Per-plaquette intensive variable, beta-free:
    #     e_p = (1 - cos theta_p)_avg = S / (beta * 6 V)
    # where V = L^ndim is the lattice volume (sites) and the 6 comes from
    # ndim*(ndim-1)/2 plaquette planes in 4D. Our internal S already has beta
    # baked in (S = beta * sum_p (1 - cos theta_p)), so we divide by both.
    n_plaq = (ndim * (ndim - 1) // 2) * (size ** ndim)   # = 6V in 4D
    scale  = beta * n_plaq
    return {"E_n": e_n, "a_hist": a_hist, "n_nr": n_nr, "n_rm": n_rm,
            "delta": delta, "scale": scale}


def reconstruct_log_rho(e_n, a_final):
    """Reconstruct ln rho_natural at each window centre E_n via the trapezoidal
    rule.

    Standard Wilson convention, mode-A LLR: S_LLR = (1+a)*S + window, so
    weight ∝ exp(-S) * exp(-a*S - window) = natural * tilt * window. The
    saddle of <S - E_n> = 0 puts a* = d ln Omega/dE - 1, equivalent to
    d ln rho_natural/dE. Trapezoidal integration of a(E_n) therefore
    reconstructs ln rho_natural directly — same formula as the scalar twin
    (examples/04_phi4_llr).
    """
    log_rho = np.zeros_like(e_n)
    for i in range(1, len(e_n)):
        slope = 0.5 * (a_final[i - 1] + a_final[i])
        log_rho[i] = log_rho[i - 1] + slope * (e_n[i] - e_n[i - 1])
    return log_rho


def piecewise_log_rho(e_n, a_final, log_rho, delta, n_per_window=16):
    """Finely-sampled piecewise-exponential reconstruction of ln rho_natural.

    Inside each window the local model is
        rho_natural(S) = rho_natural(E_n) * exp(a_n * (S - E_n))
    (slope a_n in log space — same convention as `reconstruct_log_rho`).
    """
    xs, ys = [], []
    half = 0.5 * delta
    for n in range(len(e_n)):
        seg_x = np.linspace(e_n[n] - half, e_n[n] + half, n_per_window)
        seg_y = log_rho[n] + a_final[n] * (seg_x - e_n[n])
        xs.append(seg_x)
        ys.append(seg_y)
    return np.concatenate(xs), np.concatenate(ys)


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

        # One histogram bin per LLR window, each bin centred on E_n with width
        # delta. Makes the HMC dos directly comparable to the per-window LLR
        # estimate.
        delta = llr["delta"]
        bin_edges = np.concatenate([[e_n[0] - delta / 2.0], e_n + delta / 2.0])
        raw_counts, _ = np.histogram(s_chain, bins=bin_edges)
        n_total = len(s_chain)
        # bin width = delta everywhere (uniform LLR grid).
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
        # Both HMC and LLR are now evaluated on the same e_n grid — align at
        # the same index (HMC peak) directly, no interpolation needed.
        log_rho_llr_shifted = log_rho_llr + (peak_log_hmc - log_rho_llr[peak_idx])

        # Piecewise-exponential reconstruction: one exp per LLR window.
        pw_x, pw_log = piecewise_log_rho(e_n, a_final, log_rho_llr_shifted, delta)

        # ---- Rescale x to the mean plaquette (paper convention,
        #      S = beta * sum cos theta_p):
        #          <cos theta_p> = S / (beta * 6 V)
        #      Density picks up the Jacobian factor `scale`; unit-integral
        #      is preserved by the trapezoid normalisation below.
        scale = llr["scale"]   # = beta * 6V
        centres = centres / scale
        e_n = e_n / scale
        pw_x = pw_x / scale
        peak_s = peak_s / scale
        counts = counts * scale
        counts_err = counts_err * scale
        # No reordering needed — the original action grid is monotone.
        e_n_plot = e_n
        centres_plot = centres
        log_rho_llr_plot = log_rho_llr_shifted
        counts_plot = counts
        counts_err_plot = counts_err
        log_rho_hmc_plot = log_rho_hmc
        log_rho_hmc_err_plot = log_rho_hmc_err
        mask_plot = mask

        ax = axes[0, col]
        ax.errorbar(centres_plot[mask_plot], log_rho_hmc_plot[mask_plot],
                    yerr=log_rho_hmc_err_plot[mask_plot],
                    fmt="x", markersize=4, markeredgewidth=0.8,
                    elinewidth=0.8, capsize=2,
                    color="C0", label="HMC histogram")
        ax.plot(pw_x, pw_log, "-", color="C3", linewidth=0.7,
                label="LLR (piecewise exp)")
        ax.plot(e_n_plot, log_rho_llr_plot, "x", color="C3", markersize=4,
                markeredgewidth=0.8)
        ax.axvline(peak_s, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_title(rf"$\beta = {beta:.3f}$")
        ax.set_xlabel(r"$\langle \cos\theta_p\rangle = S / (\beta\, 6 V)$")
        ax.set_ylabel(r"$\ln \rho$ + const.")
        ax.legend(loc="best", fontsize=8)

        # Linear-scale rho(S), both normalised to unit integral.
        ax = axes[1, col]
        pw_log_shifted = pw_log - pw_log.max()
        rho_pw_un = np.exp(pw_log_shifted)
        z_llr = float(np.trapezoid(rho_pw_un, pw_x))
        rho_pw = rho_pw_un / z_llr if z_llr > 0 else rho_pw_un
        # Marker values at the window centres on the same normalisation.
        rho_centre = np.exp(log_rho_llr_shifted - pw_log.max())
        rho_centre = rho_centre / z_llr if z_llr > 0 else rho_centre

        # No re-sort needed.
        rho_centre_plot = rho_centre
        ax.errorbar(centres_plot, counts_plot, yerr=counts_err_plot,
                    fmt="x", markersize=4, markeredgewidth=0.8,
                    elinewidth=0.8, capsize=2,
                    color="C0", label="HMC histogram")
        ax.plot(pw_x, rho_pw, "-", color="C3", linewidth=0.7,
                label="LLR (piecewise exp)")
        ax.plot(e_n_plot, rho_centre_plot, "x", color="C3", markersize=4,
                markeredgewidth=0.8)
        ax.axvline(peak_s, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$\langle \cos\theta_p\rangle = S / (\beta\, 6 V)$")
        ax.set_ylabel(r"$\rho$  (unit integral)")
        ax.legend(loc="best", fontsize=8)

        ax = axes[2, col]
        n_rep, n_iter = a_hist.shape
        iters = np.arange(n_iter)
        cmap = plt.get_cmap("viridis", n_rep)
        for n in range(n_rep):
            show_label = (n_rep <= 8 or n % max(1, n_rep // 6) == 0)
            ax.plot(iters, a_hist[n], linewidth=1.0, color=cmap(n),
                    label=f"$e_p$={e_n[n]:.3f}" if show_label else None)
        ax.axvline(llr["n_nr"] - 0.5, color="0.4", linewidth=0.8, linestyle="--",
                   label="NR → RM")
        ax.axhline(0.0, color="0.8", linewidth=0.8)
        ax.set_xlabel("iteration (NR then RM)")
        ax.set_ylabel(r"$a_n$")
        ax.set_title(r"$a_n$ convergence per replica")
        ax.legend(loc="best", fontsize=7, ncol=2)

        # a_n vs E_n/(6V): the converged LLR slope as a function of the
        # intensive plaquette variable. Standard convention: a* = d ln
        # rho_natural/dS, so at any natural peak a* = 0; bulk first-order
        # transitions show up as a stretch where d ln rho_natural/dS
        # sweeps through the metastable region between the two phases.
        ax = axes[3, col]
        ax.plot(e_n_plot, a_final, "o-", color="C2", markersize=4,
                linewidth=0.7, label=r"$a_n$ (LLR final)")
        ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--",
                   label=r"$a = 0$ (natural peak)")
        ax.axvline(peak_s, color="0.7", linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$\langle 1 - \cos\theta_p\rangle = S / (\beta\, 6 V)$")
        ax.set_ylabel(r"$a_n = d \ln \rho_{\rm natural} / dS$")
        ax.legend(loc="best", fontsize=8)

    out = HERE / "rho_hmc_vs_llr.pdf"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
