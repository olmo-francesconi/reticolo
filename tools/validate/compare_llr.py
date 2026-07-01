#!/usr/bin/env python3
"""Compare reconstructed LLR a(E_n) between two runs (e.g. CPU vs CUDA).

The LLR slope a(E_n) ≈ d ln g/dE − 1 is a physical property of the density of
states, so two correct backends must converge to the same a(E_n) within
statistics — even though they are different Markov chains (CPU xoshiro vs GPU
Philox: same ensemble, not the same chain). This is the physics gate for the
CUDA LLR port; the app smoke test only checks the HDF5 schema.

Handles the two exchange conventions:
  - fixed-window (CPU config-swap): window centre is /cfg/E_n[N], slope series
    is /replica_NNN/a.
  - param-swap (CUDA phi4_llr_cuda): windows migrate across slots, so each slot
    carries a time-varying /replica_NNN/E_n series paired with /replica_NNN/a.
Either way we group the tail (E_n, a) pairs by window centre and average.

Usage:
    uv run --with h5py --with numpy \
        tools/validate/compare_llr.py A.h5 B.h5 [--tail 10] [--sigma 3.0]
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


def a_by_window(reps, tail):
    """Group the last `tail` (E_n, a) pairs by window centre -> {E_n: (mean, sem, n)}."""
    buckets = {}
    for a, en in reps.values():
        k = min(tail, len(a))
        for ai, ei in zip(a[-k:], en[-k:]):
            buckets.setdefault(round(float(ei), 6), []).append(float(ai))
    res = {}
    for key, vals in buckets.items():
        v = np.array(vals)
        sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        res[key] = (float(v.mean()), float(sem), len(v))
    return res


def main():
    ap = argparse.ArgumentParser(description="Compare LLR a(E_n) between two runs.")
    ap.add_argument("a_h5", help="first run (e.g. CPU phi4_llr output)")
    ap.add_argument("b_h5", help="second run (e.g. CUDA phi4_llr_cuda output)")
    ap.add_argument("--tail", type=int, default=10,
                    help="average the last N appended a-values (RM-converged region)")
    ap.add_argument("--sigma", type=float, default=3.0, help="pass threshold in sigma")
    args = ap.parse_args()

    a = a_by_window(load_replicas(args.a_h5), args.tail)
    b = a_by_window(load_replicas(args.b_h5), args.tail)
    keys = sorted(set(a) & set(b))
    if not keys:
        print("no shared window centres between the two runs", file=sys.stderr)
        return 2

    print(f"{'E_n':>8}  {'a_A':>10} {'sem':>8}  {'a_B':>10} {'sem':>8}  "
          f"{'Δ=B-A':>10} {'Δ/σ':>7}")
    max_nsig = 0.0
    for k in keys:
        aa, sa, _ = a[k]
        bb, sb, _ = b[k]
        d = bb - aa
        sig = np.hypot(sa, sb)
        nsig = abs(d) / sig if sig and sig == sig else float("nan")
        if nsig == nsig:
            max_nsig = max(max_nsig, nsig)
        print(f"{k:8.2f}  {aa:10.4f} {sa:8.4f}  {bb:10.4f} {sb:8.4f}  "
              f"{d:10.4f} {nsig:7.2f}")

    print(f"\nwindows compared: {len(keys)}   max |Δ|/σ = {max_nsig:.2f}")
    ok = max_nsig < args.sigma
    print(f"PASS (a(E_n) agrees within ~{args.sigma:g}σ)" if ok
          else f"CHECK (some windows exceed {args.sigma:g}σ — inspect)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
