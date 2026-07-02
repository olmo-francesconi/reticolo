#!/usr/bin/env python3
"""Compare reconstructed LLR a(E_n) between two backends (e.g. CPU vs CUDA), with
a STATISTICALLY VALID error bar from multiple independent seed runs.

The LLR slope a(E_n) ≈ d ln g/dE − 1 is a physical property of the density of
states, so two correct backends must converge to the same a(E_n) — even though
they are different Markov chains (CPU xoshiro vs GPU Philox: same ensemble, not
the same chain).

Why multiple seeds: a single run's a(E_n) comes from ONE Robbins-Monro
trajectory. Its RM tail is strongly autocorrelated, so the naive std/√N over the
tail is NOT a valid error — it treats correlated steps as independent, under-
estimates the error by roughly the autocorrelation time, and thereby inflates any
Δ/σ. The only honest error here is the scatter of the converged a across
INDEPENDENT seeds. So pass several runs per backend; the error bar is the
cross-seed SEM. One file per side still works but prints Δ only (no valid error).

Handles both exchange conventions: fixed-window (/cfg/E_n + /replica_NNN/a) and
param-swap (per-slot time-varying /replica_NNN/E_n paired with /a). Windows are
grouped by centre either way.

Usage:
    uv run --with h5py --with numpy tools/validate/compare_llr.py \
        --a cpu_seed1.h5 cpu_seed2.h5 ... --b gpu_seed1.h5 gpu_seed2.h5 ... \
        [--tail 10] [--sigma 3.0]
"""
import argparse
import sys

import h5py
import numpy as np


def load_replicas(path):
    out = {}
    with h5py.File(path, "r") as f:
        n_rep = int(f["/cfg"].attrs["n_rep"])
        cfg_en = f["/cfg/E_n"][:] if "/cfg/E_n" in f else None
        for n in range(n_rep):
            a = np.asarray(f[f"/replica_{n:03d}/a"][:]).ravel()
            en_key = f"/replica_{n:03d}/E_n"
            en = np.asarray(f[en_key][:]).ravel() if en_key in f \
                else np.full_like(a, cfg_en[n])
            out[n] = (a, en)
    return out


def per_run(path, tail):
    """One converged a per window for a single run: the mean of that run's last
    `tail` a-values in each window centre. -> {E_n: a}."""
    buckets = {}
    for a, en in load_replicas(path).values():
        k = min(tail, len(a))
        for ai, ei in zip(a[-k:], en[-k:]):
            buckets.setdefault(round(float(ei), 6), []).append(float(ai))
    return {w: float(np.mean(v)) for w, v in buckets.items()}


def aggregate(paths, tail):
    """Across independent seed runs -> {E_n: (mean, sem, n_runs)}, where sem is the
    cross-seed standard error (nan for a single run — no valid error)."""
    runs = [per_run(p, tail) for p in paths]
    windows = sorted(set().union(*(set(r) for r in runs)))
    out = {}
    for w in windows:
        vals = np.array([r[w] for r in runs if w in r])
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else float("nan")
        out[w] = (float(vals.mean()), float(sem), len(vals))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Compare LLR a(E_n) between two backends, multi-seed error bars.")
    ap.add_argument("--a", nargs="+", required=True, metavar="H5",
                    help="backend-A seed runs (e.g. CPU); >1 file for a valid error bar")
    ap.add_argument("--b", nargs="+", required=True, metavar="H5",
                    help="backend-B seed runs (e.g. CUDA)")
    ap.add_argument("--tail", type=int, default=10,
                    help="average the last N appended a-values (RM-converged region)")
    ap.add_argument("--sigma", type=float, default=3.0, help="pass threshold in sigma")
    args = ap.parse_args()

    a = aggregate(args.a, args.tail)
    b = aggregate(args.b, args.tail)
    keys = sorted(set(a) & set(b))
    if not keys:
        print("no shared window centres between the two backends", file=sys.stderr)
        return 2

    na, nb = len(args.a), len(args.b)
    have_err = na > 1 and nb > 1
    print(f"backend A: {na} seed(s)   backend B: {nb} seed(s)")
    if not have_err:
        print("WARNING: need >1 seed per backend for a valid error — showing Δ only "
              "(a single run's tail SEM is autocorrelated, not a real error).")
    print(f"{'E_n':>8}  {'a_A':>10} {'sem':>8}  {'a_B':>10} {'sem':>8}  "
          f"{'Δ=B-A':>10} {'Δ/σ':>7}")
    max_nsig = 0.0
    for k in keys:
        aa, sa, _ = a[k]
        bb, sb, _ = b[k]
        d = bb - aa
        sig = np.hypot(sa, sb)
        nsig = abs(d) / sig if (sig and sig == sig) else float("nan")
        if nsig == nsig:
            max_nsig = max(max_nsig, nsig)
        print(f"{k:8.2f}  {aa:10.4f} {sa:8.4f}  {bb:10.4f} {sb:8.4f}  "
              f"{d:10.4f} {nsig:7.2f}")

    if not have_err:
        print(f"\nwindows compared: {len(keys)}   (no σ — single seed per backend)")
        return 0
    print(f"\nwindows compared: {len(keys)}   max |Δ|/σ = {max_nsig:.2f}")
    ok = max_nsig < args.sigma
    print(f"PASS (a(E_n) agrees within ~{args.sigma:g}σ)" if ok
          else f"CHECK (some windows exceed {args.sigma:g}σ — inspect)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
