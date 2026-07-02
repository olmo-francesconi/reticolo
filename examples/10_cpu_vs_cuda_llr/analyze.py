#!/usr/bin/env python3
"""CPU-vs-CUDA LLR agreement plot — distributional comparison.

The LLR slope a(E_n) ≈ d ln g/dE − 1 is a physical property of the density of
states, so the CPU (xoshiro) and CUDA (Philox) backends — the SAME ensemble, not
the same Markov chain — must produce the same a(E_n). Rather than reduce each
backend to a mean and compare point estimates, we compare the DISTRIBUTIONS:

  * Per-seed estimator = the LAST Robbins-Monro a-value (the converged RM
    iterate), NOT the mean over the tail — a tail-mean smears in the RM transient
    and the within-run autocorrelation.
  * At the final iteration the replica exchange is a permutation, so each seed
    contributes exactly one final a per window. Per window we then have a
    5-sample CPU set and a 5-sample GPU set, and we test whether they are drawn
    from the same distribution (two-sample Kolmogorov-Smirnov).

Overlays the per-seed a(E_n) point-clouds of both backends, the ln g(E)
reconstructed from the cross-seed means, and the per-window KS p-value.

Reuses the raw loader from tools/validate/compare_llr.py (handles both exchange
conventions: fixed /cfg/E_n and param-swap per-slot /replica_NNN/E_n).

Usage:
    uv run --with h5py --with numpy --with scipy --with matplotlib \
        examples/10_cpu_vs_cuda_llr/analyze.py \
        --cpu results/<model>/cpu/*.h5 --gpu results/<model>/gpu/*.h5 \
        --out results/comparison_<model>.pdf
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "tools" / "validate"))
from compare_llr import load_replicas  # noqa: E402


def last_values(paths):
    """{window centre: np.array of per-seed FINAL a-values}. One value per seed
    per window: at the last iteration the exchange is a permutation, so each of
    a run's replica slots ends at a distinct window and contributes its final a."""
    per_window = defaultdict(list)
    for p in paths:
        for a, en in load_replicas(p).values():
            per_window[round(float(en[-1]), 6)].append(float(a[-1]))
    return {w: np.array(v) for w, v in sorted(per_window.items())}


def reconstruct_log_g(e, a):
    """ln g(E) up to a constant: cumulative trapezoid of a(E) over the window
    centres (d ln g/dE ≈ a), mean-subtracted so the two backends overlay on shape."""
    lg = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(e))])
    return lg - lg.mean()


def main():
    ap = argparse.ArgumentParser(description="CPU-vs-CUDA LLR distributional agreement")
    ap.add_argument("--cpu", nargs="+", required=True, metavar="H5")
    ap.add_argument("--gpu", nargs="+", required=True, metavar="H5")
    ap.add_argument("--title", default="phi4 LLR — CPU vs CUDA")
    ap.add_argument("--xlabel", default=r"$E_n$  (action window centre)")
    ap.add_argument("--wmin", type=float, default=-np.inf,
                    help="drop windows below this centre (e.g. underconverged rare tails)")
    ap.add_argument("--wmax", type=float, default=np.inf)
    ap.add_argument("--out", default="comparison.pdf")
    args = ap.parse_args()

    cpu = last_values(args.cpu)
    gpu = last_values(args.gpu)
    wins = np.array([w for w in cpu if w in gpu and args.wmin <= w <= args.wmax])

    cpu_mean = np.array([cpu[w].mean() for w in wins])
    cpu_std = np.array([cpu[w].std(ddof=1) for w in wins])
    gpu_mean = np.array([gpu[w].mean() for w in wins])
    gpu_std = np.array([gpu[w].std(ddof=1) for w in wins])
    ks_p = np.array([stats.ks_2samp(cpu[w], gpu[w]).pvalue for w in wins])
    n_reject = int(np.sum(ks_p < 0.05))

    print(f"CPU seeds: {len(args.cpu)}   GPU seeds: {len(args.gpu)}")
    print(f"{'E_n':>8}  {'cpu μ':>9} {'cpu s':>7}  {'gpu μ':>9} {'gpu s':>7}  {'KS p':>7}")
    for i, w in enumerate(wins):
        print(f"{w:8.2f}  {cpu_mean[i]:9.4f} {cpu_std[i]:7.4f}  "
              f"{gpu_mean[i]:9.4f} {gpu_std[i]:7.4f}  {ks_p[i]:7.3f}")
    print(f"\nwindows: {len(wins)}   KS rejections at p<0.05: {n_reject} "
          f"(≈{0.05*len(wins):.1f} expected under the null)   min p = {ks_p.min():.3f}")

    # Per-window reference = pooled (both-backend) mean, for the residual panel.
    ref = np.array([np.concatenate([cpu[w], gpu[w]]).mean() for w in wins])

    # Shared thin-line / open-marker / small-cap-errorbar style.
    ek = dict(lw=0.9, ms=5, mew=0.9, mfc="none", capsize=2.5, capthick=0.8, elinewidth=0.8)
    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(7.5, 9.0), height_ratios=[3, 3, 2.4], constrained_layout=True)

    # (0) a(E_n): cross-seed mean ± std at the true window centre (no x-shift).
    ax0.errorbar(wins, cpu_mean, yerr=cpu_std, fmt="o-", color="C0",
                 label=f"CPU ({len(args.cpu)} seeds)", **ek)
    ax0.errorbar(wins, gpu_mean, yerr=gpu_std, fmt="s--", color="C1",
                 label=f"CUDA ({len(args.gpu)} seeds)", **ek)
    ax0.set_ylabel(r"$a(E_n)$  (mean $\pm$ seed std)")
    ax0.set_title(args.title)
    ax0.legend()
    ax0.grid(alpha=0.3)

    # (1) reconstructed ln g(E) from the cross-seed means.
    ax1.plot(wins, reconstruct_log_g(wins, cpu_mean), "o-", color="C0",
             lw=0.9, ms=5, mew=0.9, mfc="none", label="CPU")
    ax1.plot(wins, reconstruct_log_g(wins, gpu_mean), "s--", color="C1",
             lw=0.9, ms=5, mew=0.9, mfc="none", label="CUDA")
    ax1.set_ylabel(r"$\ln g(E)$  (mean-subtracted)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # (2) residual: each backend's a distribution (mean ± std) about the per-window
    # pooled mean — the fine-scale comparison the full-range panel can't resolve.
    ax2.axhline(0.0, color="k", lw=0.6)
    ax2.errorbar(wins, cpu_mean - ref, yerr=cpu_std, fmt="o-", color="C0", **ek)
    ax2.errorbar(wins, gpu_mean - ref, yerr=gpu_std, fmt="s--", color="C1", **ek)
    ax2.set_ylabel(r"$a - \langle a\rangle_{\mathrm{interval}}$")
    ax2.set_xlabel(args.xlabel)
    ax2.grid(alpha=0.3)

    fig.savefig(args.out)
    print(f"wrote {args.out}")
    verdict = "PASS" if n_reject <= max(1, round(0.05 * len(wins) + 1)) else "CHECK"
    print(f"{verdict}: distributions consistent ({n_reject} window(s) reject at p<0.05)")


if __name__ == "__main__":
    main()
