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
    """Reconstruct ln rho at each window centre E_n via the trapezoidal rule.
    Continuity at the boundary E_n + delta/2 between adjacent piecewise
    exponentials gives ln rho(E_{n+1}) = ln rho(E_n) + (a_n + a_{n+1}) * delta / 2.
    """
    log_rho = np.zeros_like(e_n)
    for i in range(1, len(e_n)):
        log_rho[i] = log_rho[i - 1] + 0.5 * (a_final[i - 1] + a_final[i]) * (e_n[i] - e_n[i - 1])
    return log_rho


def piecewise_log_rho(e_n, a_final, log_rho, delta, n_per_window=16):
    """Build a finely-sampled piecewise-exponential reconstruction.

    Within each window [E_n - delta/2, E_n + delta/2] the DoS is
        rho(E) = rho(E_n) * exp(a_n * (E - E_n))
    i.e. a straight line of slope a_n in log space. Adjacent segments share
    their endpoint by construction of `reconstruct_log_rho` (continuity).
    """
    xs, ys = [], []
    half = 0.5 * delta
    for n in range(len(e_n)):
        seg_x = np.linspace(e_n[n] - half, e_n[n] + half, n_per_window)
        seg_y = log_rho[n] + a_final[n] * (seg_x - e_n[n])
        xs.append(seg_x)
        ys.append(seg_y)
    return np.concatenate(xs), np.concatenate(ys)


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
