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
    ap.add_argument("--out", default="comparison.pdf")
    args = ap.parse_args()

    cpu = last_values(args.cpu)
    gpu = last_values(args.gpu)
    wins = np.array([w for w in cpu if w in gpu])
    dx = np.median(np.diff(wins)) if len(wins) > 1 else 1.0
    jit = 0.14 * dx

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

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(7.5, 9.0), height_ratios=[3, 3, 1.6], constrained_layout=True)

    # (0) per-seed a(E_n) point-clouds — the distributions themselves.
    for i, w in enumerate(wins):
        ax0.plot(np.full_like(cpu[w], w - jit), cpu[w], ".", color="C0", ms=6, alpha=0.6)
        ax0.plot(np.full_like(gpu[w], w + jit), gpu[w], ".", color="C1", ms=6, alpha=0.6)
    ax0.plot(wins - jit, cpu_mean, "_", color="C0", ms=14, mew=2)
    ax0.plot(wins + jit, gpu_mean, "_", color="C1", ms=14, mew=2)
    ax0.plot([], [], "o", color="C0", label=f"CPU ({len(args.cpu)} seeds)")
    ax0.plot([], [], "o", color="C1", label=f"CUDA ({len(args.gpu)} seeds)")
    ax0.set_ylabel(r"$a(E_n)$  (per-seed final RM value)")
    ax0.set_title(args.title)
    ax0.legend()
    ax0.grid(alpha=0.3)

    # (1) reconstructed ln g(E) from the cross-seed means.
    lgc, lgg = reconstruct_log_g(wins, cpu_mean), reconstruct_log_g(wins, gpu_mean)
    ax1.plot(wins, lgc, "o-", color="C0", ms=4, label="CPU")
    ax1.plot(wins, lgg, "s--", color="C1", ms=4, label="CUDA")
    ax1.set_ylabel(r"$\ln g(E)$  (mean-subtracted)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # (2) two-sample KS p-value per window (distribution match).
    ax2.axhline(0.05, color="r", ls=":", lw=0.9, label="p = 0.05")
    ax2.stem(wins, ks_p, basefmt=" ", linefmt="k-", markerfmt="kD")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-2, 1.5)
    ax2.set_ylabel("KS two-sample $p$")
    ax2.set_xlabel(args.xlabel)
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    ax2.text(0.02, 0.1, f"rejections p<0.05: {n_reject}/{len(wins)}",
             transform=ax2.transAxes)

    fig.savefig(args.out)
    print(f"wrote {args.out}")
    verdict = "PASS" if n_reject <= max(1, round(0.05 * len(wins) + 1)) else "CHECK"
    print(f"{verdict}: distributions consistent ({n_reject} window(s) reject at p<0.05)")


if __name__ == "__main__":
    main()
