#!/usr/bin/env python3
"""Read every on_sigma_L*_beta*.h5 in ./results, compute the Binder cumulant
per (L, beta), and plot one curve per L on a single axis. Curves crossing
near a single beta locate the critical point of the 3D O(3) sigma model.

Binder cumulant (vector-magnetization form):
    U(L, beta) = 1 - <m4> / (3 <m2>^2)
with  m2 = |M|^2 / V^2  (recorded per config)
and   m4 = m2^2         (squared on the Python side).
"""

from __future__ import annotations

import collections
import pathlib
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
FNAME = re.compile(r"on_sigma_L(\d+)_beta([0-9.]+)\.h5$")


def _load(f: h5py.File, path: str) -> np.ndarray:
    return np.asarray(f[path])  # type: ignore[arg-type]


def binder(path: pathlib.Path) -> tuple[int, float, float]:
    """Return (L, beta, U) for one HDF5 file."""
    m = FNAME.search(path.name)
    if not m:
        raise ValueError(f"unrecognised filename {path.name}")
    L = int(m.group(1))
    beta = float(m.group(2))
    with h5py.File(path, "r") as f:
        m2 = _load(f, "/prod/obs/m2")
    m4 = m2**2
    u = 1.0 - float(np.mean(m4)) / (3.0 * float(np.mean(m2)) ** 2)
    return L, beta, u


def main() -> int:
    files = sorted(RESULTS.glob("on_sigma_L*_beta*.h5"))
    if not files:
        print(f"no results in {RESULTS}; run ./run.sh first", file=sys.stderr)
        return 1

    grouped: dict[int, list[tuple[float, float]]] = collections.defaultdict(list)
    for p in files:
        L, beta, u = binder(p)
        grouped[L].append((beta, u))

    fig, ax = plt.subplots(figsize=(7, 5))
    for L in sorted(grouped):
        rows = sorted(grouped[L])
        betas, us = zip(*rows)
        ax.plot(betas, us, "o-", label=f"L = {L}")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$U = 1 - \langle m^4 \rangle / (3 \langle m^2 \rangle^2)$")
    ax.set_title("Binder cumulant crossings — 3D O(3) sigma model")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = HERE / "binder.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
