"""Load every results/*.h5, compute τ_int (Madras-Sokal) for S and Σφ² in both
update units and seconds, dump a flat summary CSV that plot.py consumes.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_common"))
from autocorr import tau_int  # noqa: E402

RESULTS_ROOT = HERE / "results"


def parse_filename(name: str) -> dict:
    """Recover the run parameters from the file name set by run.sh."""
    stem = Path(name).stem
    if stem.startswith("metropolis_"):
        match = re.match(r"metropolis_sigma([0-9.]+)$", stem)
        if not match:
            raise ValueError(f"unparseable metropolis filename: {name}")
        return {"algo": "metropolis", "sigma": float(match.group(1))}
    if stem.startswith("hmc_"):
        match = re.match(r"(hmc_\w+)_tau([0-9.]+)_nmd([0-9]+)$", stem)
        if not match:
            raise ValueError(f"unparseable hmc filename: {name}")
        return {
            "algo": match.group(1),
            "tau": float(match.group(2)),
            "n_md": int(match.group(3)),
        }
    raise ValueError(f"unknown filename pattern: {name}")


def analyze_one(path: Path) -> dict:
    """Load one run, compute τ_int for S and mean_sq in update and wall-time units.

    Wall time is broken into:
      algo_seconds   — pure trajectory/sweep cost, what we use for autocorr
                       conversion (the cost of producing one Markov-chain step).
      obs_seconds    — Σφ² and S_full measurements, irrelevant to the algorithm
                       comparison.
      wall_seconds   — algo + obs + a tiny bookkeeping overhead.

    Both algo_sec and total_sec autocorrelation are reported."""
    meta = parse_filename(path.name)
    with h5py.File(path, "r") as f:
        prod = f["/prod"]
        wall_s   = float(prod.attrs["wall_seconds"])
        algo_s   = float(prod.attrs.get("algo_seconds", wall_s))
        obs_s    = float(prod.attrs.get("obs_seconds", 0.0))
        n_meas   = int(prod.attrs["n_meas"])
        accept   = float(prod.attrs["accept_rate"])
        s   = prod["obs/s"][...]
        msq = prod["obs/mean_sq"][...]

    algo_per_update = algo_s / n_meas
    total_per_update = wall_s / n_meas

    r_s_algo  = tau_int(s,   s_per_update=algo_per_update)
    r_msq_algo = tau_int(msq, s_per_update=algo_per_update)

    return {
        **meta,
        "wall_s": wall_s,
        "algo_s": algo_s,
        "obs_s": obs_s,
        "algo_per_update": algo_per_update,
        "total_per_update": total_per_update,
        "n_meas": n_meas,
        "accept": accept,
        # τ_int in update units (algorithm-agnostic)
        "tau_S": r_s_algo.tau_int,
        "tau_S_err": r_s_algo.tau_int_err,
        "tau_msq": r_msq_algo.tau_int,
        "tau_msq_err": r_msq_algo.tau_int_err,
        # τ_int converted to algorithm-only wall seconds (the fair comparison)
        "tau_S_sec":   r_s_algo.tau_int_seconds,
        "tau_S_sec_err":   r_s_algo.tau_int_seconds_err,
        "tau_msq_sec": r_msq_algo.tau_int_seconds,
        "tau_msq_sec_err": r_msq_algo.tau_int_seconds_err,
        "window_S":   r_s_algo.window,
        "window_msq": r_msq_algo.window,
    }


def process_scenario(scenario_dir: Path) -> None:
    files = sorted(scenario_dir.glob("*.h5"))
    if not files:
        print(f"  (no HDF5 in {scenario_dir.name}, skipping)")
        return

    rows = [analyze_one(p) for p in files]
    columns = [
        "algo", "sigma", "tau", "n_md",
        "accept", "wall_s", "algo_s", "obs_s",
        "algo_per_update", "total_per_update", "n_meas",
        "tau_S", "tau_S_err", "tau_S_sec", "tau_S_sec_err", "window_S",
        "tau_msq", "tau_msq_err", "tau_msq_sec", "tau_msq_sec_err", "window_msq",
    ]
    summary = scenario_dir / "summary.csv"
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"  wrote {summary.relative_to(HERE)} ({len(rows)} runs)")


def main() -> None:
    scenarios = sorted([d for d in RESULTS_ROOT.iterdir()
                        if d.is_dir() and any(d.glob("*.h5"))])
    if not scenarios:
        sys.exit(f"no scenario subdirectories found under {RESULTS_ROOT}; "
                 "run run.sh first (with SCENARIO=easy and SCENARIO=critical)")
    print(f"found {len(scenarios)} scenario(s): {[d.name for d in scenarios]}")
    for d in scenarios:
        print(f"\n--- {d.name} ---")
        process_scenario(d)


if __name__ == "__main__":
    main()
