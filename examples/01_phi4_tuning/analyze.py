#!/usr/bin/env python3
"""Read every phi4_kappa_*.h5 in ./results, compute the susceptibility per
kappa, and plot the curve. Susceptibility:  chi = V * (<m^2> - <|m|>^2).
"""

from __future__ import annotations

import pathlib
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
FNAME = re.compile(r"phi4_kappa_([0-9.]+)\.h5$")


def _load(f: h5py.File, path: str) -> np.ndarray:
    return np.asarray(f[path])  # type: ignore[arg-type]


def susceptibility(path: pathlib.Path) -> tuple[float, float, float]:
    """Return (kappa, chi, |m|_mean) for one HDF5 file."""
    m = FNAME.search(path.name)
    if not m:
        raise ValueError(f"unrecognised filename {path.name}")
    kappa = float(m.group(1))
    with h5py.File(path, "r") as f:
        mag = _load(f, "/prod/obs/mag")
        m2 = _load(f, "/prod/obs/m2")
        L = int(f["/vars"].attrs["size"])  # type: ignore[arg-type]
    volume = L**4
    chi = volume * (float(np.mean(m2)) - float(np.mean(mag)) ** 2)
    return kappa, chi, float(np.mean(mag))


def main() -> int:
    files = sorted(RESULTS.glob("phi4_kappa_*.h5"))
    if not files:
        print(f"no results in {RESULTS}; run ./run.sh first", file=sys.stderr)
        return 1
    rows = [susceptibility(p) for p in files]
    rows.sort(key=lambda r: r[0])
    kappas, chis, mags = zip(*rows)

    fig, (ax_chi, ax_mag) = plt.subplots(1, 2, figsize=(10, 4))
    ax_chi.plot(kappas, chis, "o-", color="#005f87")
    ax_chi.set_xlabel(r"$\kappa$")
    ax_chi.set_ylabel(r"$\chi = V(\langle m^2\rangle - \langle |m|\rangle^2)$")
    ax_chi.set_title("Susceptibility")
    ax_chi.grid(True, alpha=0.3)

    ax_mag.plot(kappas, mags, "s-", color="#bf3030")
    ax_mag.set_xlabel(r"$\kappa$")
    ax_mag.set_ylabel(r"$\langle |m|\rangle$")
    ax_mag.set_title("Magnetisation")
    ax_mag.grid(True, alpha=0.3)

    fig.tight_layout()
    out = HERE / "susceptibility.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
