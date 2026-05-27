#!/usr/bin/env python3
"""Read an HDF5 file written by a reticolo app and summarise it.

Every app writes a self-describing HDF5 file: reproducibility metadata under
`/run@*`, the resolved command-line flags under `/vars@*`, and one dataset per
time series (e.g. `/prod/obs/s`, `/prod/stats/accepted`). This script opens such
a file, prints that metadata, walks the dataset tree, and reports mean ±
standard error for each production observable plus the HMC acceptance rate.

Usage:
    python read_output.py [path.h5]        # default: phi4.h5

Produce an input first, e.g.:
    ./build/macos-appleclang/apps/phi4_hmc --out phi4.h5 --n_prod 2000

Dependencies: h5py, numpy. The naive standard error below ignores
autocorrelation; for publication-quality errors use the block jackknife /
integrated-autocorrelation helpers in examples/_common/.
"""

from __future__ import annotations

import sys

import h5py
import numpy as np


def print_attrs(f: h5py.File, group: str) -> None:
    """Print the attributes hanging off an HDF5 group (the `/run@*`, `/vars@*`)."""
    if group not in f:
        return
    attrs = dict(f[group].attrs)
    if not attrs:
        return
    print(f"{group}@")
    for key in sorted(attrs):
        val = attrs[key]
        if isinstance(val, bytes):
            val = val.decode()
        print(f"    {key:<22} {val}")
    print()


def summarise_series(f: h5py.File) -> None:
    """Walk every dataset; print shape, then mean ± stderr for 1-D numeric ones."""
    datasets: list[str] = []
    f.visititems(lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None)

    print(f"{'dataset':<24} {'rows':>8} {'mean':>14} {'std err':>12}")
    print("-" * 62)
    for name in datasets:
        data = np.asarray(f[name])
        path = "/" + name
        if data.ndim != 1 or not np.issubdtype(data.dtype, np.number):
            print(f"{path:<24} {data.shape!s:>8}   (non-scalar series)")
            continue
        n = data.size
        mean = float(np.mean(data))
        stderr = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        print(f"{path:<24} {n:>8} {mean:>14.6g} {stderr:>12.3g}")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "phi4.h5"
    try:
        f = h5py.File(path, "r")
    except OSError:
        print(f"cannot open {path!r} — run an app with --out {path} first", file=sys.stderr)
        return 1

    with f:
        print(f"=== {path} ===\n")
        print_attrs(f, "/run")
        print_attrs(f, "/vars")
        summarise_series(f)

        if "/prod/stats/accepted" in f:
            acc = np.asarray(f["/prod/stats/accepted"])
            print(f"\nHMC acceptance: {acc.mean():.3f} over {acc.size} trajectories")

    return 0


if __name__ == "__main__":
    sys.exit(main())
