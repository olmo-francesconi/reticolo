#!/usr/bin/env python3
"""Compare vanilla LLR with smoothed LLR at the 4D compact U(1) bulk
transition. Produces three figures:

    compare_vanilla_vs_smoothed.pdf   convergence + reconstruction overlay
    doublepeak_vanilla.pdf          full 2x2 panel for the vanilla run
                                    (same layout as example 04)
    doublepeak_smoothed.pdf           full 2x2 panel for the smoothed run
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


def load_run(path):
    data = load_a_history(path)
    n_rep = data["a_hist"].shape[0]
    n_total = data["a_hist"].shape[1]
    de_hist = np.empty((n_rep, n_total), dtype=float)
    drm_hist = np.full((n_rep, n_total), np.nan, dtype=float)
    dsm_hist = np.full((n_rep, n_total), np.nan, dtype=float)
    smooth_attrs = {}
    with h5py.File(path, "r") as f:
        size = int(f["/vars"].attrs["size"])
        ndim = int(f["/vars"].attrs["ndim"])
        beta = float(f["/vars"].attrs["beta"])
        cfg_attrs = dict(f["/cfg"].attrs.items())
        has_drm = f"/replica_{0:03d}/da_rm" in f
        for n in range(n_rep):
            dE = np.asarray(f[f"/replica_{n:03d}/dE"][:], dtype=float)
            m = min(len(dE), n_total)
            de_hist[n, :m] = dE[:m]
            if m < n_total:
                de_hist[n, m:] = dE[-1]
            if has_drm:
                drm = np.asarray(f[f"/replica_{n:03d}/da_rm"][:], dtype=float)
                dsm = np.asarray(f[f"/replica_{n:03d}/da_sm"][:], dtype=float)
                m = min(len(drm), n_total)
                drm_hist[n, :m] = drm[:m]
                dsm_hist[n, :m] = dsm[:m]
        for k in ("smooth_K", "smooth_degree", "smooth_lambda0", "smooth_lambda_exp"):
            if k in cfg_attrs:
                smooth_attrs[k] = cfg_attrs[k]
    n_plaq = (ndim * (ndim - 1) // 2) * (size ** ndim)
    data["scale"] = float(n_plaq)
    data["beta"] = beta
    data["size"] = size
    data["ndim"] = ndim
    data["de_hist"] = de_hist
    data["drm_hist"] = drm_hist
    data["dsm_hist"] = dsm_hist
    data["has_amplitudes"] = bool(has_drm)
    data["smooth_attrs"] = smooth_attrs
    return data


def plot_doublepeak(data, outpath, title_suffix=""):
    e_n = data["E_n"]
    a_hist = data["a_hist"]
    delta = data["delta"]
    scale = data["scale"]
    beta = data["beta"]
    size = data["size"]
    ndim = data["ndim"]
    n_nr = data["n_nr"]

    a_final = np.median(a_hist[:, -max(1, a_hist.shape[1] // 5):], axis=1)
    log_rho_centres = reconstruct_log_rho(e_n, a_final)
    pw_x, pw_log = piecewise_log_rho(e_n, a_final, log_rho_centres, delta)

    offset = float(pw_log.max())
    pw_log -= offset
    log_rho_centres -= offset

    e_n_p = e_n / scale
    pw_x_p = pw_x / scale

    rho_un = np.exp(pw_log)
    z = float(np.trapezoid(rho_un, pw_x_p))
    rho = rho_un / z if z > 0 else rho_un
    rho_centres = np.exp(log_rho_centres) / z if z > 0 else np.exp(log_rho_centres)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(pw_x_p, pw_log, "-", color="C3", linewidth=1.0,
            label="LLR (piecewise exp)")
    ax.plot(e_n_p, log_rho_centres, "x", color="C3", markersize=5,
            markeredgewidth=1.0, label="LLR window centres")
    ax.set_xlabel(r"$S / (6 V)$")
    ax.set_ylabel(r"$\ln \rho$  (peak = 0)")
    ax.set_title(rf"$L={size}^{{{ndim}}}$, $\beta = {beta:.4f}$  {title_suffix}")
    ax.legend(loc="best", fontsize=9)

    ax = axes[0, 1]
    ax.plot(pw_x_p, rho, "-", color="C3", linewidth=1.0,
            label=r"$\rho(S)$  (unit integral)")
    ax.plot(e_n_p, rho_centres, "x", color="C3", markersize=5,
            markeredgewidth=1.0)
    ax.set_xlabel(r"$S / (6 V)$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title(r"density of states — double peak at $\beta_c$")
    ax.legend(loc="best", fontsize=9)

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

    ax = axes[1, 1]
    ax.plot(e_n_p, a_final, "o-", color="C2", markersize=4,
            linewidth=0.7, label=r"$a_n$ (LLR final)")
    ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--",
               label=r"$a = 0$ (natural peak)")
    ax.set_xlabel(r"$S / (6 V)$")
    ax.set_ylabel(r"$a_n = d \ln \rho_{\rm natural} / dS$")
    ax.legend(loc="best", fontsize=9)

    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"wrote {outpath}")


def plot_compare(van, pol, outpath):
    delta = van["delta"]
    e_n = van["E_n"]
    n_nr = van["n_nr"]
    n_rm = van["n_rm"]

    van_dE_rm = van["de_hist"][:, n_nr:]
    pol_dE_rm = pol["de_hist"][:, n_nr:]
    van_max = np.max(np.abs(van_dE_rm) / delta, axis=0)
    pol_max = np.max(np.abs(pol_dE_rm) / delta, axis=0)

    tail = max(1, n_rm // 5)
    van_a = np.median(van["a_hist"][:, -tail:], axis=1)
    pol_a = np.median(pol["a_hist"][:, -tail:], axis=1)
    van_log = reconstruct_log_rho(e_n, van_a)
    pol_log = reconstruct_log_rho(e_n, pol_a)
    van_x, van_lr = piecewise_log_rho(e_n, van_a, van_log, delta)
    pol_x, pol_lr = piecewise_log_rho(e_n, pol_a, pol_log, delta)
    van_lr -= van_lr.max()
    pol_lr -= pol_lr.max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    steps = np.arange(1, n_rm + 1)
    ax1.semilogy(steps, van_max, lw=1.4, label="vanilla", color="C0")
    ax1.semilogy(steps, pol_max, lw=1.4, label="smoothed (deg=2, K=4)", color="C1")
    ax1.set_xlabel("RM iteration")
    ax1.set_ylabel(r"$\max_n |\langle\Delta E\rangle_n| / \delta$")
    ax1.set_title("convergence (lower is better)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.plot(van_x, van_lr, lw=1.6, label="vanilla", color="C0", alpha=0.8)
    ax2.plot(pol_x, pol_lr, lw=1.6, label="smoothed", color="C1", alpha=0.8, ls="--")
    ax2.set_xlabel("S")
    ax2.set_ylabel(r"$\ln \rho(S)$ (offset to peak)")
    ax2.set_title("reconstruction (must agree)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(
        f"U(1)  L=6  β=1.002   n_rm={n_rm}   tail-median over last {tail} steps",
        fontsize=10,
    )
    fig.savefig(outpath)
    plt.close(fig)
    print(f"wrote {outpath}")

    common = np.linspace(e_n.min(), e_n.max(), 400)
    delta_log = np.abs(np.interp(common, van_x, van_lr) - np.interp(common, pol_x, pol_lr))
    print()
    print("convergence at last step:")
    print(f"  vanilla  max|<dE>/δ| = {van_max[-1]:.3f}")
    print(f"  smoothed   max|<dE>/δ| = {pol_max[-1]:.3f}")
    if van_max[-1] > 0:
        print(f"  ratio smoothed/vanilla = {pol_max[-1]/van_max[-1]:.3f}")
    print()
    print("reconstruction agreement (|ln rho_vanilla − ln rho_smoothed|):")
    print(f"  median = {np.median(delta_log):.3f}")
    print(f"  max    = {np.max(delta_log):.3f}")


def plot_amplitudes(pol, outpath):
    """Per-step update breakdown: how big is the RM contribution vs the
    smoothing contribution at each RM iteration?

      |Δa_rm|[n,s] = |a_rm[n,s] − a_old[n,s]|        (pure RM step)
      |Δa_sm|[n,s] = |λ_s · (â[n,s] − a_rm[n,s])|    (shrinkage step)

    Plotted on log y so the two scales coexist; the smoothing curve
    collapsing fast tells you the algorithm reverts to vanilla RM after
    just a handful of iterations.
    """
    n_nr = pol["n_nr"]
    n_rm = pol["n_rm"]
    drm = pol["drm_hist"][:, n_nr:]
    dsm = pol["dsm_hist"][:, n_nr:]
    if np.all(np.isnan(drm)):
        print("no amplitude series in HDF5; skipping smoothed_amplitudes.pdf")
        return
    steps = np.arange(1, n_rm + 1)

    drm_mean = np.nanmean(drm, axis=0)
    drm_max = np.nanmax(drm, axis=0)
    drm_min = np.nanmin(drm, axis=0)
    dsm_mean = np.nanmean(dsm, axis=0)
    dsm_max = np.nanmax(dsm, axis=0)
    dsm_min = np.nanmin(dsm, axis=0)

    pa = pol["smooth_attrs"]
    lam0 = float(pa.get("smooth_lambda0", 0))
    lam_exp = float(pa.get("smooth_lambda_exp", 0))
    lam_curve = lam0 / np.power(steps, lam_exp) if lam_exp > 0 else np.full_like(steps, lam0)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10.0, 9.5), constrained_layout=True, sharex=True
    )

    ax1.fill_between(steps, drm_min, drm_max, color="C0", alpha=0.15, label="RM min..max")
    ax1.plot(steps, drm_mean, color="C0", lw=1.6, label=r"$|\Delta a_{\rm rm}|$ mean")
    ax1.fill_between(steps, dsm_min, dsm_max, color="C1", alpha=0.15, label="smoothing min..max")
    ax1.plot(steps, dsm_mean, color="C1", lw=1.6, label=r"$|\Delta a_{\rm sm}|$ mean")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"per-replica step magnitude")
    ax1.set_title("RM vs smoothing update size per RM iteration")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    ratio_mean = np.where(drm_mean > 0, dsm_mean / drm_mean, np.nan)
    ratio_max = np.where(drm_max > 0, dsm_max / drm_max, np.nan)
    ax2.plot(steps, ratio_mean, color="C2", lw=1.6, label=r"mean ratio")
    ax2.plot(steps, ratio_max, color="C3", lw=1.0, ls="--", label=r"max ratio")
    ax2.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$|\Delta a_{\rm sm}| / |\Delta a_{\rm rm}|$")
    ax2.set_title("smoothing weight relative to RM step")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    ax3.plot(steps, lam_curve, color="C4", lw=1.6, label=rf"$\lambda_s = {lam0:g}/(s+1)^{{{lam_exp:g}}}$")
    ax3.set_yscale("log")
    ax3.set_xlabel("RM iteration s")
    ax3.set_ylabel(r"$\lambda_s$")
    ax3.set_title("shrinkage weight schedule")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"smoothed LLR update breakdown  (K={int(pa.get('smooth_K', 0))}, "
        f"deg={int(pa.get('smooth_degree', 0))}, λ0={lam0:g}, exp={lam_exp:g})",
        fontsize=11,
    )
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"wrote {outpath}")


def main():
    vpath = RESULTS / "llr_vanilla.h5"
    ppath = RESULTS / "llr_smoothed.h5"
    if not vpath.exists() or not ppath.exists():
        print(f"missing {vpath} or {ppath}; run run.sh first", file=sys.stderr)
        return 1

    van = load_run(vpath)
    pol = load_run(ppath)
    assert np.allclose(van["E_n"], pol["E_n"]), "E_n grids differ"
    assert van["delta"] == pol["delta"], "delta differs"

    plot_compare(van, pol, HERE / "compare_vanilla_vs_smoothed.pdf")

    plot_doublepeak(van, HERE / "doublepeak_vanilla.pdf",
                    title_suffix="— vanilla LLR")
    pa = pol["smooth_attrs"]
    pool_suffix = (
        f"— smoothed (K={int(pa.get('smooth_K', 0))}, "
        f"deg={int(pa.get('smooth_degree', 0))}, "
        f"λ0={float(pa.get('smooth_lambda0', 0)):.2g}, "
        f"exp={float(pa.get('smooth_lambda_exp', 0)):.2g})"
    )
    plot_doublepeak(pol, HERE / "doublepeak_smoothed.pdf",
                    title_suffix=pool_suffix)

    plot_amplitudes(pol, HERE / "smoothed_amplitudes.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
