#!/usr/bin/env python3
"""CPU-vs-CUDA LLR agreement plot.

The LLR slope a(E_n) ≈ d ln g/dE − 1 is a physical property of the density of
states, so the CPU (xoshiro) and CUDA (Philox) backends — the SAME ensemble, not
the same Markov chain — must converge to the same a(E_n). This overlays a(E_n)
from both backends with a STATISTICALLY VALID cross-seed error band, reconstructs
ln g(E) from each, and shows the per-window pull (a_gpu − a_cpu)/σ.

Reuses the loader from tools/validate/compare_llr.py (both exchange conventions:
fixed /cfg/E_n and param-swap per-slot /replica_NNN/E_n).

Usage:
    uv run --with h5py --with numpy --with matplotlib \
        examples/10_cpu_vs_cuda_llr/analyze.py \
        --cpu results/cpu/*.h5 --gpu results/gpu/*.h5 --out results/comparison.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the validated multi-seed loader/aggregator (one source of truth).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "tools" / "validate"))
from compare_llr import aggregate  # noqa: E402


def curve(paths, tail):
    """{E_n:(mean,sem,n)} -> sorted (E, a, sem) arrays."""
    agg = aggregate(paths, tail)
    e = np.array(sorted(agg))
    a = np.array([agg[k][0] for k in e])
    sem = np.array([agg[k][1] for k in e])
    return e, a, sem


def reconstruct_log_g(e, a):
    """ln g(E) up to a constant: cumulative trapezoid of a(E) over the window
    centres (energy convention d ln g/dE ≈ a). Returned mean-subtracted so the two
    backends overlay on shape, which is what the comparison tests."""
    lg = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(e))])
    return lg - lg.mean()


def log_g_band(e, sem):
    """1σ band on the cumulative-trapezoid ln g from independent per-window SEMs."""
    w = np.diff(e)
    step_var = np.concatenate([[0.0], (0.5 * w) ** 2 * (sem[1:] ** 2 + sem[:-1] ** 2)])
    return np.sqrt(np.cumsum(step_var))


def main():
    ap = argparse.ArgumentParser(description="CPU-vs-CUDA LLR agreement plot")
    ap.add_argument("--cpu", nargs="+", required=True, metavar="H5")
    ap.add_argument("--gpu", nargs="+", required=True, metavar="H5")
    ap.add_argument("--tail", type=int, default=10,
                    help="average the last N appended a-values (RM-converged region)")
    ap.add_argument("--title", default="phi4 LLR — CPU vs CUDA")
    ap.add_argument("--out", default="comparison.pdf")
    args = ap.parse_args()

    ec, ac, sc = curve(args.cpu, args.tail)
    eg, ag, sg = curve(args.gpu, args.tail)
    have_err = len(args.cpu) > 1 and len(args.gpu) > 1

    # Shared windows for the pull.
    keys = np.array(sorted(set(np.round(ec, 6)) & set(np.round(eg, 6))))
    ac_k = np.array([ac[np.argmin(np.abs(ec - k))] for k in keys])
    ag_k = np.array([ag[np.argmin(np.abs(eg - k))] for k in keys])
    sc_k = np.array([sc[np.argmin(np.abs(ec - k))] for k in keys])
    sg_k = np.array([sg[np.argmin(np.abs(eg - k))] for k in keys])
    d = ag_k - ac_k
    sig = np.hypot(sc_k, sg_k)
    pull = np.where(sig > 0, d / sig, np.nan)
    max_pull = np.nanmax(np.abs(pull)) if have_err else float("nan")

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(7.5, 9.0), height_ratios=[3, 3, 1.6], constrained_layout=True)

    # (0) a(E_n) overlay
    if have_err:
        ax0.fill_between(ec, ac - sc, ac + sc, color="C0", alpha=0.25, lw=0)
        ax0.fill_between(eg, ag - sg, ag + sg, color="C1", alpha=0.25, lw=0)
    ax0.plot(ec, ac, "o-", color="C0", ms=4, label=f"CPU ({len(args.cpu)} seeds)")
    ax0.plot(eg, ag, "s--", color="C1", ms=4, label=f"CUDA ({len(args.gpu)} seeds)")
    ax0.set_ylabel(r"$a(E_n)\ \approx\ d\ln g/dE - 1$")
    ax0.set_title(args.title)
    ax0.legend()
    ax0.grid(alpha=0.3)

    # (1) reconstructed ln g(E)
    lgc, lgg = reconstruct_log_g(ec, ac), reconstruct_log_g(eg, ag)
    if have_err:
        bc, bg = log_g_band(ec, sc), log_g_band(eg, sg)
        ax1.fill_between(ec, lgc - bc, lgc + bc, color="C0", alpha=0.25, lw=0)
        ax1.fill_between(eg, lgg - bg, lgg + bg, color="C1", alpha=0.25, lw=0)
    ax1.plot(ec, lgc, "o-", color="C0", ms=4, label="CPU")
    ax1.plot(eg, lgg, "s--", color="C1", ms=4, label="CUDA")
    ax1.set_ylabel(r"$\ln g(E)$  (mean-subtracted)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # (2) per-window pull
    ax2.axhline(0, color="k", lw=0.8)
    for y in (-3, 3):
        ax2.axhline(y, color="r", ls=":", lw=0.8)
    ax2.axhspan(-1, 1, color="gray", alpha=0.15)
    ax2.plot(keys, pull, "kD", ms=4)
    ax2.set_ylabel(r"$(a_{\rm GPU}-a_{\rm CPU})/\sigma$")
    ax2.set_xlabel(r"$E_n$  (action window centre)")
    ax2.set_ylim(-5, 5)
    ax2.grid(alpha=0.3)
    if have_err:
        ax2.text(0.02, 0.9, rf"max $|{{\rm pull}}|$ = {max_pull:.2f}$\sigma$",
                 transform=ax2.transAxes, va="top")

    fig.savefig(args.out)
    print(f"wrote {args.out}")
    if have_err:
        verdict = "PASS" if max_pull < 3.0 else "CHECK"
        print(f"{verdict}: max |pull| = {max_pull:.2f}σ over {len(keys)} windows")
    else:
        print("single seed per backend — no valid error band (pass >1 seed each)")


if __name__ == "__main__":
    main()
