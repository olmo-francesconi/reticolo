#!/usr/bin/env python3
"""Finite-size-scaling collapse analysis for 2D phi^4 (Ising universality).

Loads every phi4_L*_kappa*.h5 in ./results, computes
    chi(kappa; L) = V * (<m^2> - <|m|>^2)
with block-jackknife errors, then fits the scaling exponents (gamma/nu, 1/nu)
and the critical coupling kappa_c by minimising a Bhattacharjee-Seno
collapse residual: for each L, the rescaled point (x = (kappa-kappa_c) L^{1/nu},
y = chi L^{-gamma/nu}) is compared to the master curve drawn by the other
sizes (linear interpolation in x); the residual is summed and minimised by
Nelder-Mead over (kappa_c, gamma/nu, 1/nu) jointly.

Plots:
  (a) raw chi(kappa) for each L,
  (b) fully collapsed chi L^{-gamma/nu} vs (kappa - kappa_c) L^{1/nu}
      using the fitted exponents.

Reports the fitted (kappa_c, gamma/nu, 1/nu) and compares against the EXACT
2D Ising values (Onsager 1944, exact transfer-matrix solution):
  nu = 1   =>   1/nu = 1
  eta = 1/4   =>   gamma/nu = 2 - eta = 7/4 = 1.75
"""

from __future__ import annotations

import collections
import pathlib
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
FNAME = re.compile(r"phi4_L(\d+)_kappa([0-9.]+)\.h5$")

sys.path.insert(0, str(HERE.parent / "_common"))
from jackknife import jackknife  # noqa: E402


def _load(f: h5py.File, path: str) -> np.ndarray:
    return np.asarray(f[path])  # type: ignore[arg-type]


def chi_with_error(path: pathlib.Path) -> tuple[int, float, float, float]:
    """Return (L, kappa, chi, chi_err) for one HDF5 file."""
    m = FNAME.search(path.name)
    if not m:
        raise ValueError(f"unrecognised filename {path.name}")
    L = int(m.group(1))
    kappa = float(m.group(2))
    with h5py.File(path, "r") as f:
        mag = _load(f, "/prod/obs/mag")
        mag_sq = _load(f, "/prod/obs/mag_sq")
        ndim = int(f["/vars"].attrs.get("ndim", 3))  # type: ignore[arg-type]
    volume = L**ndim

    def chi(means: tuple[float, ...]) -> float:
        mag_mean, mag_sq_mean = means
        return volume * (mag_sq_mean - mag_mean**2)

    chi_val, chi_err = jackknife((mag, mag_sq), chi, n_blocks=40)
    return L, kappa, chi_val, chi_err


def collapse_residual(
    params: np.ndarray, per_L: dict[int, dict[str, np.ndarray]], Ls: list[int]
) -> float:
    """Bhattacharjee-Seno collapse residual: for each L, predict its rescaled
    y from the master curve formed by the other sizes' rescaled points (via
    linear interpolation in x), and sum the chi-squared deviations.
    """
    kappa_c, gamma_over_nu, one_over_nu = params
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    es: list[np.ndarray] = []
    src: list[int] = []
    for L in Ls:
        d = per_L[L]
        x = (d["kappa"] - kappa_c) * (L**one_over_nu)
        y = d["chi"] * (L ** (-gamma_over_nu))
        e = d["err"] * (L ** (-gamma_over_nu))
        xs.append(x)
        ys.append(y)
        es.append(e)
        src.extend([L] * len(x))
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    all_e = np.concatenate(es)
    all_src = np.array(src)

    total = 0.0
    for L in Ls:
        mask = all_src == L
        x_L, y_L, e_L = all_x[mask], all_y[mask], all_e[mask]
        oth_x, oth_y = all_x[~mask], all_y[~mask]
        order = np.argsort(oth_x)
        oth_x_s, oth_y_s = oth_x[order], oth_y[order]
        in_range = (x_L >= oth_x_s[0]) & (x_L <= oth_x_s[-1])
        if not in_range.any():
            continue
        y_pred = np.interp(x_L[in_range], oth_x_s, oth_y_s)
        sigma = np.maximum(e_L[in_range], 1e-12)
        total += float(np.sum(((y_L[in_range] - y_pred) / sigma) ** 2))
    return total


def main() -> int:
    files = sorted(RESULTS.glob("phi4_L*_kappa*.h5"))
    if not files:
        print(f"no results in {RESULTS}; run ./run.sh first", file=sys.stderr)
        return 1

    grouped: dict[int, list[tuple[float, float, float]]] = collections.defaultdict(list)
    for p in files:
        L, kappa, chi, err = chi_with_error(p)
        grouped[L].append((kappa, chi, err))

    Ls = sorted(grouped)
    per_L: dict[int, dict[str, np.ndarray]] = {}
    for L in Ls:
        rows = sorted(grouped[L])
        per_L[L] = {
            "kappa": np.array([r[0] for r in rows]),
            "chi": np.array([r[1] for r in rows]),
            "err": np.array([r[2] for r in rows]),
        }

    if len(Ls) < 2:
        print("need at least two L values to do a collapse", file=sys.stderr)
        return 1

    # Global search inside physical bounds, then a Nelder-Mead polish.
    # Bounds:
    #   kappa_c   — within the scanned kappa window
    #   gamma/nu  — generous physical range
    #   1/nu      — generous physical range
    kappa_lo = float(min(per_L[L]["kappa"].min() for L in Ls))
    kappa_hi = float(max(per_L[L]["kappa"].max() for L in Ls))
    bounds = [(kappa_lo, kappa_hi), (0.5, 6.0), (0.5, 4.0)]
    de = differential_evolution(
        collapse_residual, bounds, args=(per_L, Ls),
        rng=0, polish=False, maxiter=400, tol=1e-7,
    )
    res = minimize(
        collapse_residual, de.x, args=(per_L, Ls), method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-4, "maxiter": 5000},
    )
    kappa_c, gamma_over_nu, one_over_nu = res.x

    # Exact 2D Ising (Onsager): nu = 1, eta = 1/4.
    lit_gamma_over_nu = 7.0 / 4.0
    lit_one_over_nu = 1.0
    print("Fitted FSS parameters (Bhattacharjee-Seno collapse, weighted chi^2):")
    print(f"  kappa_c     = {kappa_c:.5f}")
    print(f"  gamma / nu  = {gamma_over_nu:.3f}   (2D Ising, exact: {lit_gamma_over_nu:.4f})")
    print(f"  1 / nu      = {one_over_nu:.3f}   (2D Ising, exact: {lit_one_over_nu:.4f})")
    print(f"  residual    = {res.fun:.2f}   (sum of (data - master)^2 / sigma^2)")

    fig, (ax_raw, ax_col) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.get_cmap("viridis", len(Ls) + 1)

    for i, L in enumerate(Ls):
        d = per_L[L]
        ax_raw.errorbar(
            d["kappa"], d["chi"], yerr=d["err"], fmt="x-",
            lw=0.7, markersize=5, markeredgewidth=1.0, capsize=2,
            color=colors(i), label=f"L = {L}",
        )
    ax_raw.axvline(kappa_c, color="k", linestyle="--", lw=0.8, alpha=0.6,
                   label=rf"fitted $\kappa_c = {kappa_c:.4f}$")
    ax_raw.set_xlabel(r"$\kappa$")
    ax_raw.set_ylabel(r"$\chi = V(\langle m^2\rangle - \langle |m|\rangle^2)$")
    ax_raw.set_title("Raw susceptibility")
    ax_raw.grid(True, alpha=0.25, lw=0.5)
    ax_raw.legend(fontsize=9)

    for i, L in enumerate(Ls):
        d = per_L[L]
        x_scaled = (d["kappa"] - kappa_c) * (L**one_over_nu)
        y_scaled = d["chi"] * (L ** (-gamma_over_nu))
        yerr_scaled = d["err"] * (L ** (-gamma_over_nu))
        ax_col.errorbar(
            x_scaled, y_scaled, yerr=yerr_scaled, fmt="x-",
            lw=0.7, markersize=5, markeredgewidth=1.0, capsize=2,
            color=colors(i), label=f"L = {L}",
        )
    ax_col.set_xlabel(rf"$(\kappa - \kappa_c)\, L^{{1/\nu}}$")
    ax_col.set_ylabel(rf"$\chi\, L^{{-\gamma/\nu}}$")
    ax_col.set_title(
        rf"Collapse with fitted $\gamma/\nu = {gamma_over_nu:.2f}$, "
        rf"$1/\nu = {one_over_nu:.2f}$"
    )
    ax_col.grid(True, alpha=0.25, lw=0.5)
    ax_col.legend(fontsize=9)

    fig.tight_layout()
    out = HERE / "susceptibility.pdf"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
