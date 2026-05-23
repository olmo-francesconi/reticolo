#!/usr/bin/env python3
"""Synthetic benchmark of vanilla LLR vs smoothed LLR.

No MC, no HDF5. We pick a ground-truth profile a*(E), generate fake
<dE> samples under the linear-response model

    <dE>(a, E_n) = δ² · (a*(E_n) − a)  +  ε,   ε ~ N(0, σ²)

(σ ≈ δ/√n_meas in the real algorithm) and drive both algorithms on
the synthetic data. The two algorithms share the same noise sequence
per seed, so any difference is purely due to the smoothing layer.

Three profiles:
  linear       a*(E) = α·(E − E_c)             — polynomial-exact
  sigmoid      a*(E) = α·tanh((E − E_c)/w)    — smooth saturation
  double_peak  a*(E) = −α·u·(u² − w²)/w²       — two-peak DoS analogue

Diagnostics per profile (one PDF each):
  - RMSE vs iteration, vanilla vs smoothed, ±1σ envelope over seeds
  - true a*(E) vs final reconstruction, both algorithms
  - final-step bias per replica (SEM errorbars; unbiasedness check)
  - example trajectories for a handful of replicas
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


# ---------- ground-truth profiles ---------------------------------------

# Normalised: E grid spans [-1, 1], w = 1, max |a*| ≈ 0.5.
def a_true(profile: str, e_grid: np.ndarray, e_c: float = 0.0, w: float = 1.0) -> np.ndarray:
    u = e_grid - e_c
    if profile == "linear":
        return 0.5 * u / w
    if profile == "sigmoid":
        return 0.5 * np.tanh(3.0 * u / w)
    if profile == "double_peak":
        # Extrema at u = ±w/√3, value ±α·2w/(3√3). Set α so |extremum| = 0.5.
        alpha = 0.5 * 3.0 * np.sqrt(3.0) / 2.0
        return -alpha * u * (u * u - w * w) / (w * w)
    raise ValueError(f"unknown profile: {profile}")


# ---------- synthetic <dE> generator ------------------------------------

def sample_dE(a: np.ndarray, a_star: np.ndarray, delta: float, sigma: float,
              rng: np.random.Generator) -> np.ndarray:
    return delta * delta * (a_star - a) + sigma * rng.standard_normal(size=a.shape)


# ---------- local-polynomial smoother (mirrors smoothed_driver.hpp) -----

def local_poly_fit(e: np.ndarray, a: np.ndarray, K: int, deg: int) -> np.ndarray:
    n = len(e)
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - K)
        hi = min(n - 1, i + K)
        x = e[lo:hi + 1] - e[i]
        y = a[lo:hi + 1]
        if len(x) < deg + 1:
            out[i] = a[i]
            continue
        coeffs = np.polyfit(x, y, deg)
        out[i] = coeffs[-1]  # constant term = value at x=0
    return out


# ---------- algorithms --------------------------------------------------

def run_vanilla(a_star, delta, sigma, n_nr, n_rm, rng):
    n_rep = len(a_star)
    a = np.zeros(n_rep)
    hist = np.empty((n_rep, n_nr + n_rm + 1))
    hist[:, 0] = a
    for k in range(n_nr):
        dE = sample_dE(a, a_star, delta, sigma, rng)
        a = a + dE / (delta * delta)
        hist[:, k + 1] = a
    for s in range(n_rm):
        dE = sample_dE(a, a_star, delta, sigma, rng)
        a = a + dE / (delta * delta * (s + 1))
        hist[:, n_nr + s + 1] = a
    return hist


def run_smoothed(a_star, e_grid, delta, sigma, n_nr, n_rm,
               K, deg, lambda0, lambda_exp, rng):
    n_rep = len(a_star)
    a = np.zeros(n_rep)
    hist = np.empty((n_rep, n_nr + n_rm + 1))
    pre = np.empty((n_rep, n_nr + n_rm + 1))
    hist[:, 0] = a
    pre[:, 0] = a
    for k in range(n_nr):
        dE = sample_dE(a, a_star, delta, sigma, rng)
        a = a + dE / (delta * delta)
        hist[:, k + 1] = a
        pre[:, k + 1] = a  # no shrinkage in NR
    for s in range(n_rm):
        dE = sample_dE(a, a_star, delta, sigma, rng)
        a_rm = a + dE / (delta * delta * (s + 1))
        a_hat = local_poly_fit(e_grid, a_rm, K, deg)
        lam = lambda0 / ((s + 1) ** lambda_exp)
        a = (1.0 - lam) * a_rm + lam * a_hat
        pre[:, n_nr + s + 1] = a_rm
        hist[:, n_nr + s + 1] = a
    return hist, pre


# ---------- multi-seed orchestration -----------------------------------

def run_many(profile: str, args, seed_base: int):
    e_grid = np.linspace(-1.0, 1.0, args.n_rep)
    a_star = a_true(profile, e_grid)
    n_total = args.n_nr + args.n_rm + 1
    van_all = np.empty((args.n_seeds, args.n_rep, n_total))
    pol_all = np.empty((args.n_seeds, args.n_rep, n_total))
    for i in range(args.n_seeds):
        # Same seed for both algos → identical noise sequence, so any
        # difference is the smoother, not the random draw.
        rng_v = np.random.default_rng(seed_base + i)
        rng_p = np.random.default_rng(seed_base + i)
        van_all[i] = run_vanilla(a_star, args.delta, args.sigma,
                                 args.n_nr, args.n_rm, rng_v)
        pol_all[i], _ = run_smoothed(a_star, e_grid, args.delta, args.sigma,
                                   args.n_nr, args.n_rm,
                                   args.smooth_K, args.smooth_degree,
                                   args.smooth_lambda0, args.smooth_lambda_exp,
                                   rng_p)
    return e_grid, a_star, van_all, pol_all


# ---------- plot --------------------------------------------------------

def plot_profile(profile, e_grid, a_star, van_all, pol_all, args, outpath):
    n_seeds, n_rep, n_total = van_all.shape
    n_nr = args.n_nr
    n_rm = args.n_rm

    err_v = van_all - a_star[None, :, None]
    err_p = pol_all - a_star[None, :, None]
    rmse_v_seed = np.sqrt(np.mean(err_v * err_v, axis=1))   # (seeds, step)
    rmse_p_seed = np.sqrt(np.mean(err_p * err_p, axis=1))
    rmse_v = np.median(rmse_v_seed, axis=0)
    rmse_p = np.median(rmse_p_seed, axis=0)
    rmse_v_lo = np.percentile(rmse_v_seed, 16, axis=0)
    rmse_v_hi = np.percentile(rmse_v_seed, 84, axis=0)
    rmse_p_lo = np.percentile(rmse_p_seed, 16, axis=0)
    rmse_p_hi = np.percentile(rmse_p_seed, 84, axis=0)

    final_v = van_all[:, :, -1]
    final_p = pol_all[:, :, -1]
    bias_v  = np.mean(final_v - a_star[None, :], axis=0)
    bias_p  = np.mean(final_p - a_star[None, :], axis=0)
    sem_v   = np.std(final_v, axis=0) / np.sqrt(n_seeds)
    sem_p   = np.std(final_p, axis=0) / np.sqrt(n_seeds)
    std_v   = np.std(final_v, axis=0)
    std_p   = np.std(final_p, axis=0)

    steps = np.arange(n_total)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)

    # --- RMSE vs iteration (+ smoothing share overlay) ---
    ax = axes[0, 0]
    ax.fill_between(steps, rmse_v_lo, rmse_v_hi, color="C0", alpha=0.20)
    ax.fill_between(steps, rmse_p_lo, rmse_p_hi, color="C1", alpha=0.20)
    ax.plot(steps, rmse_v, color="C0", lw=1.6, label="vanilla RMSE")
    ax.plot(steps, rmse_p, color="C1", lw=1.6, label="smoothed RMSE")
    ax.axvline(n_nr - 0.5, color="0.5", ls="--", lw=0.8, label="NR→RM")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"RMSE$_s = \sqrt{\langle (a_n - a^*(E_n))^2 \rangle_n}$")
    ax.set_title(f"convergence (median, 16–84% over {n_seeds} seeds)")
    ax.grid(True, which="both", alpha=0.3)

    # Convex-combination weights from the smoothed recursion:
    #   a_new = (1 − λ_s) · a_rm + λ_s · a_hat,    λ_s = λ₀ / (s+1)^α
    # Purely a function of s — same for every replica and every seed.
    s_idx = np.arange(1, n_rm + 1)
    lam = args.smooth_lambda0 / (s_idx ** args.smooth_lambda_exp)
    rm_steps = s_idx + n_nr  # align with the RMSE x-axis

    ax2 = ax.twinx()
    ax2.plot(rm_steps, 1.0 - lam, color="C0", lw=1.0, ls=":",
             label=r"RM weight $1 - \lambda_s$")
    ax2.plot(rm_steps, lam, color="C1", lw=1.0, ls=":",
             label=r"smoothing weight $\lambda_s$")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("convex-combination weight")
    ax2.tick_params(axis="y")

    # Merge legends from both axes into one.
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax2.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r, loc="best", fontsize=8)

    # --- true vs final reconstruction ---
    ax = axes[0, 1]
    ax.plot(e_grid, a_star, "k-", lw=1.4, label=r"true $a^*(E)$")
    ax.errorbar(e_grid, np.mean(final_v, axis=0), yerr=std_v,
                fmt="o", color="C0", ms=3, alpha=0.8, label="vanilla final (±σ)")
    ax.errorbar(e_grid, np.mean(final_p, axis=0), yerr=std_p,
                fmt="s", color="C1", ms=3, alpha=0.8, label="smoothed final (±σ)")
    ax.set_xlabel("E")
    ax.set_ylabel(r"$a(E)$")
    ax.set_title("final reconstruction (seed-averaged)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- bias per replica ---
    ax = axes[1, 0]
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.errorbar(e_grid, bias_v, yerr=sem_v, fmt="o-", color="C0",
                ms=3, lw=0.8, label="vanilla")
    ax.errorbar(e_grid, bias_p, yerr=sem_p, fmt="s-", color="C1",
                ms=3, lw=0.8, label="smoothed")
    ax.set_xlabel("E")
    ax.set_ylabel(r"$\langle \hat a - a^* \rangle$ at last step")
    ax.set_title(f"final-step bias (errorbars = SEM over {n_seeds} seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- example trajectories (seed 0) ---
    ax = axes[1, 1]
    repl_picks = np.linspace(0, n_rep - 1, 5, dtype=int)
    cmap = plt.get_cmap("viridis", len(repl_picks))
    for i, n in enumerate(repl_picks):
        col = cmap(i)
        ax.plot(steps, van_all[0, n, :], color=col, lw=0.8, ls=":",
                label=f"E={e_grid[n]:+.2f}  vanilla" if i == 0 else None)
        ax.plot(steps, pol_all[0, n, :], color=col, lw=1.2, ls="-",
                label=f"E={e_grid[n]:+.2f}  smoothed" if i == 0 else None)
        ax.axhline(a_star[n], color=col, lw=0.5, ls="--", alpha=0.5)
    ax.axvline(n_nr - 0.5, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$a_n$")
    ax.set_title("trajectories — dashed=vanilla, solid=smoothed (seed 0)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"profile: {profile}   "
        f"n_rep={args.n_rep}, δ={args.delta}, σ={args.sigma}, "
        f"n_nr={args.n_nr}, n_rm={args.n_rm}   "
        f"pool: K={args.smooth_K}, deg={args.smooth_degree}, "
        f"λ0={args.smooth_lambda0}, exp={args.smooth_lambda_exp}",
        fontsize=10,
    )
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"wrote {outpath}")
    print(f"  vanilla final RMSE = {rmse_v[-1]:.4e}    (max final bias = {np.max(np.abs(bias_v)):.3e})")
    print(f"  smoothed  final RMSE = {rmse_p[-1]:.4e}    (max final bias = {np.max(np.abs(bias_p)):.3e})")
    if rmse_v[-1] > 0:
        print(f"  ratio pool/vanilla = {rmse_p[-1] / rmse_v[-1]:.3f}")
    print_speedup(rmse_v_seed, rmse_p_seed, args.n_nr)


def steps_to_threshold(rmse_per_seed: np.ndarray, thr: float, n_nr: int) -> float:
    """Median RM step (1-indexed within the RM phase) at which RMSE first
    drops below `thr`. Inf for seeds that never reach it."""
    rm_part = rmse_per_seed[:, n_nr + 1:]  # skip NR phase + initial state
    hits = np.full(rm_part.shape[0], np.inf)
    for i in range(rm_part.shape[0]):
        below = np.where(rm_part[i] < thr)[0]
        if len(below) > 0:
            hits[i] = below[0] + 1  # 1-indexed
    finite = hits[np.isfinite(hits)]
    return float(np.median(finite)) if len(finite) > 0 else float("inf")


def print_speedup(rmse_v_seed: np.ndarray, rmse_p_seed: np.ndarray, n_nr: int) -> None:
    """Iterations-to-threshold speedup at five RMSE targets. Threshold set
    by vanilla's median RMSE at fixed iteration fractions, so the
    comparison is on an equal-RMSE basis."""
    n_total = rmse_v_seed.shape[1]
    n_rm = n_total - n_nr - 1
    median_v = np.median(rmse_v_seed, axis=0)
    # Target RMSE at five points spread across the RM phase.
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    print("  iterations-to-threshold (RM phase, median over seeds):")
    print(f"    {'target RMSE':>14}  {'frac':>5}  {'vanilla s':>10}  {'smoothed s':>10}  {'speedup':>8}")
    for frac in fractions:
        idx = n_nr + max(1, int(round(frac * n_rm)))
        thr = float(median_v[idx])
        sv = steps_to_threshold(rmse_v_seed, thr, n_nr)
        sp = steps_to_threshold(rmse_p_seed, thr, n_nr)
        speedup_str = f"{sv / sp:>7.2f}×" if np.isfinite(sp) and sp > 0 else "    n/a"
        sv_str = f"{sv:>10.1f}" if np.isfinite(sv) else f"{'>n_rm':>10}"
        sp_str = f"{sp:>10.1f}" if np.isfinite(sp) else f"{'>n_rm':>10}"
        print(f"    {thr:>14.3e}  {frac:>5.2f}  {sv_str}  {sp_str}  {speedup_str}")


# ---------- CLI ---------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="all",
                   choices=["linear", "sigmoid", "double_peak", "all"])
    p.add_argument("--n_rep", type=int, default=60)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=0.1,
                   help="noise std on <dE>; in real LLR ~ δ/√n_meas")
    p.add_argument("--n_nr", type=int, default=10)
    p.add_argument("--n_rm", type=int, default=100)
    p.add_argument("--smooth_K", type=int, default=4)
    p.add_argument("--smooth_degree", type=int, default=2)
    p.add_argument("--smooth_lambda0", type=float, default=1.0)
    p.add_argument("--smooth_lambda_exp", type=float, default=1.0)
    p.add_argument("--n_seeds", type=int, default=200)
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--out", default=str(HERE),
                   help="output directory (default: examples/10_.../)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = (["linear", "sigmoid", "double_peak"]
                if args.profile == "all" else [args.profile])

    for prof in profiles:
        e_grid, a_star, van_all, pol_all = run_many(prof, args, args.seed_base)
        plot_profile(prof, e_grid, a_star, van_all, pol_all,
                     args, out_dir / f"benchmark_{prof}.pdf")


if __name__ == "__main__":
    main()
