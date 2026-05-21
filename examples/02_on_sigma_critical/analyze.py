#!/usr/bin/env python3
"""Binder cumulant FSS collapse for the 3D O(3) sigma model.

Loads every on_sigma_L*_beta*.h5 in ./results, computes
    U(L, beta) = 1 - <m4> / (3 <m2>^2)
with block-jackknife errors (m4 = m2^2 squared on the Python side), and
extracts (beta_c, 1/nu) by minimising a Bhattacharjee-Seno collapse residual:
for each L the points (x = (beta - beta_c) L^{1/nu}, y = U(L, beta)) are
compared to the master curve drawn by the remaining sizes (linear
interpolation in x), and the residual is summed and minimised by
differential_evolution + a Nelder-Mead polish over (beta_c, 1/nu).

Two panels:
  (a) raw U(beta) per L with the fitted beta_c vertical line,
  (b) U vs (beta - beta_c) L^{1/nu} — should collapse onto a single curve.

Literature target (3D Heisenberg, Campostrini et al. 2002):
    nu = 0.7112(5)  =>  1/nu = 1.406
    beta_c ≈ 0.693002
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
FNAME = re.compile(r"on_sigma_L(\d+)_beta([0-9.]+)\.h5$")

sys.path.insert(0, str(HERE.parent / "_common"))
from jackknife import jackknife  # noqa: E402

LIT_ONE_OVER_NU = 1.406
LIT_BETA_C = 0.693002


def _load(f: h5py.File, path: str) -> np.ndarray:
    return np.asarray(f[path])  # type: ignore[arg-type]


def binder_with_error(path: pathlib.Path) -> tuple[int, float, float, float]:
    """Return (L, beta, U, U_err) for one HDF5 file."""
    m = FNAME.search(path.name)
    if not m:
        raise ValueError(f"unrecognised filename {path.name}")
    L = int(m.group(1))
    beta = float(m.group(2))
    with h5py.File(path, "r") as f:
        m2 = _load(f, "/prod/obs/m2")
    m4 = m2**2

    def u(means: tuple[float, ...]) -> float:
        m2_mean, m4_mean = means
        return 1.0 - m4_mean / (3.0 * m2_mean**2)

    u_val, u_err = jackknife((m2, m4), u, n_blocks=40)
    return L, beta, u_val, u_err


def collapse_residual(
    params: np.ndarray, per_L: dict[int, dict[str, np.ndarray]], Ls: list[int]
) -> float:
    """Collapse residual on (x = (beta - beta_c) * L^{1/nu}, y = U)."""
    beta_c, one_over_nu = params
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    es: list[np.ndarray] = []
    src: list[int] = []
    for L in Ls:
        d = per_L[L]
        x = (d["beta"] - beta_c) * (L**one_over_nu)
        xs.append(x)
        ys.append(d["U"])
        es.append(d["err"])
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


def main():
    files = sorted(RESULTS.glob("on_sigma_L*_beta*.h5"))
    if not files:
        print(f"no results in {RESULTS}; run ./run.sh first", file=sys.stderr)
        return 1

    grouped: dict[int, list[tuple[float, float, float]]] = collections.defaultdict(list)
    for p in files:
        L, beta, u, err = binder_with_error(p)
        grouped[L].append((beta, u, err))

    Ls = sorted(grouped)
    per_L: dict[int, dict[str, np.ndarray]] = {}
    for L in Ls:
        rows = sorted(grouped[L])
        per_L[L] = {
            "beta": np.array([r[0] for r in rows]),
            "U": np.array([r[1] for r in rows]),
            "err": np.array([r[2] for r in rows]),
        }

    if len(Ls) < 2:
        print("need at least two L values to do a collapse", file=sys.stderr)
        return 1

    beta_lo = float(min(per_L[L]["beta"].min() for L in Ls))
    beta_hi = float(max(per_L[L]["beta"].max() for L in Ls))
    bounds = [(beta_lo, beta_hi), (0.5, 3.0)]
    de = differential_evolution(
        collapse_residual, bounds, args=(per_L, Ls),
        rng=0, polish=False, maxiter=400, tol=1e-7,
    )
    res = minimize(
        collapse_residual, de.x, args=(per_L, Ls), method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-4, "maxiter": 5000},
    )
    beta_c, one_over_nu = res.x

    print("Fitted FSS parameters (Bhattacharjee-Seno collapse, weighted chi^2):")
    print(f"  beta_c      = {beta_c:.5f}   (3D O(3), MC: {LIT_BETA_C:.5f})")
    print(f"  1 / nu      = {one_over_nu:.3f}   (3D O(3), MC: {LIT_ONE_OVER_NU:.3f})")
    print(f"  residual    = {res.fun:.2f}   (sum of (data - master)^2 / sigma^2)")

    fig, (ax_raw, ax_col) = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.get_cmap("plasma", len(Ls) + 1)

    for i, L in enumerate(Ls):
        d = per_L[L]
        ax_raw.errorbar(
            d["beta"], d["U"], yerr=d["err"], fmt="x-",
            lw=0.7, markersize=5, markeredgewidth=1.0, capsize=2,
            color=colors(i), label=f"L = {L}",
        )
    ax_raw.axvline(beta_c, color="k", linestyle="--", lw=0.8, alpha=0.6,
                   label=rf"fitted $\beta_c = {beta_c:.4f}$")
    ax_raw.axvline(LIT_BETA_C, color="grey", linestyle=":", lw=0.8, alpha=0.6,
                   label=rf"lit. $\beta_c \approx {LIT_BETA_C:.4f}$")
    ax_raw.set_xlabel(r"$\beta$")
    ax_raw.set_ylabel(r"$U = 1 - \langle m^4\rangle / (3 \langle m^2\rangle^2)$")
    ax_raw.set_title("Binder cumulant crossings")
    ax_raw.grid(True, alpha=0.25, lw=0.5)
    ax_raw.legend(loc="lower right", fontsize=9)

    for i, L in enumerate(Ls):
        d = per_L[L]
        x_scaled = (d["beta"] - beta_c) * (L**one_over_nu)
        ax_col.errorbar(
            x_scaled, d["U"], yerr=d["err"], fmt="x-",
            lw=0.7, markersize=5, markeredgewidth=1.0, capsize=2,
            color=colors(i), label=f"L = {L}",
        )
    ax_col.set_xlabel(rf"$(\beta - \beta_c)\, L^{{1/\nu}}$")
    ax_col.set_ylabel(r"$U$")
    ax_col.set_title(rf"Collapse with fitted $1/\nu = {one_over_nu:.2f}$")
    ax_col.grid(True, alpha=0.25, lw=0.5)
    ax_col.legend(fontsize=9)

    fig.tight_layout()
    out = HERE / "binder.pdf"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
