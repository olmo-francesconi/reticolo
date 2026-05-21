"""Shared LLR reconstruction helpers used by examples/04..07 analyze scripts.

The LLR fixed point of <S - E_n> = 0 places a_n = d ln rho/dS at the window
centre, where rho is the natural DoS for real-action mode A and the
phase-quenched DoS for complex mode B (Bose gas). The trapezoidal sum of
a_n then reconstructs ln rho(S) directly. Each analyzer wraps these helpers
to load its own HMC series and parameter-of-filename extractor.
"""
from __future__ import annotations

import h5py
import numpy as np


def load_a_history(path):
    """Return {E_n, a_hist, n_nr, n_rm, delta} from an LLR HDF5 output.

    `a_hist` has shape (n_rep, n_nr + n_rm) — one column per NR iter then
    one per RM sweep. Trailing values are forward-filled if a series is
    shorter than expected (defensive against truncated runs).
    """
    with h5py.File(path, "r") as f:
        n_rep = int(f["/cfg"].attrs["n_rep"])
        n_nr = int(f["/cfg"].attrs["n_nr"])
        n_rm = int(f["/cfg"].attrs["n_rm"])
        delta = float(f["/cfg"].attrs["delta"])
        e_n = np.asarray(f["/cfg/E_n"][:], dtype=float)
        a_hist = np.empty((n_rep, n_nr + n_rm), dtype=float)
        for n in range(n_rep):
            a = np.asarray(f[f"/replica_{n:03d}/a"][:], dtype=float)
            m = min(len(a), n_nr + n_rm)
            a_hist[n, :m] = a[:m]
            if m < n_nr + n_rm:
                a_hist[n, m:] = a[-1]
    return {"E_n": e_n, "a_hist": a_hist, "n_nr": n_nr, "n_rm": n_rm, "delta": delta}


def load_hmc_action(path, dataset="/prod/obs/s"):
    """Read an HMC-twin S (or S_I) series from `path` at `dataset`."""
    with h5py.File(path, "r") as f:
        return np.asarray(f[dataset][:], dtype=float)


def reconstruct_log_rho(e_n, a_final):
    """ln rho at window centres via trapezoidal integration of a_n.

    Continuity at the segment boundary E_n + delta/2 between adjacent
    piecewise-exponential pieces yields
        ln rho(E_{n+1}) = ln rho(E_n) + (a_n + a_{n+1}) * (E_{n+1} - E_n) / 2.
    For mode-A (real) LLR this is ln rho_natural; for mode-B (complex
    Bose gas) it is ln rho_pq, the phase-quenched DoS of S_I.
    """
    log_rho = np.zeros_like(e_n)
    for i in range(1, len(e_n)):
        slope = 0.5 * (a_final[i - 1] + a_final[i])
        log_rho[i] = log_rho[i - 1] + slope * (e_n[i] - e_n[i - 1])
    return log_rho


def piecewise_log_rho(e_n, a_final, log_rho, delta, n_per_window=16):
    """Finely-sampled piecewise-exponential reconstruction of ln rho.

    Inside each window [E_n - delta/2, E_n + delta/2] the local model is
        rho(S) = rho(E_n) * exp(a_n * (S - E_n))
    (linear in log space, slope a_n). Adjacent segments share their
    endpoint by construction of `reconstruct_log_rho`.
    """
    xs, ys = [], []
    half = 0.5 * delta
    for n in range(len(e_n)):
        seg_x = np.linspace(e_n[n] - half, e_n[n] + half, n_per_window)
        seg_y = log_rho[n] + a_final[n] * (seg_x - e_n[n])
        xs.append(seg_x)
        ys.append(seg_y)
    return np.concatenate(xs), np.concatenate(ys)
