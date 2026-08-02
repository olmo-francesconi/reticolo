"""Shared LLR reconstruction helpers used by the examples' analyze scripts.

The DoS slope is NOT a_n in general — it depends on how the window tilts the
action, which the C++ records as /cfg@constraint_self:

  self-constraint (mode A, window on S itself, e.g. phi4 / U(1) / SU(2) LLR)
      S_LLR = (1 + a)*S + (S - E_n)^2 / 2 delta^2
      so the fixed point <S - E_n> = 0 sits at a_n = dln(rho)/dS - 1
      and the DoS slope is (1 + a_n).

  any other constraint (mode B: imag part for the sign problem, or an
  arbitrary observable Q)
      S_LLR = S_base + a*Q + (Q - E_n)^2 / 2 delta^2
      so a_n = dln(rho)/dQ directly and the slope is a_n.

Integrating a_n in mode A yields ln rho(S) - S, not ln rho(S) — an error
linear in S. It is invisible when overlaying against a raw HMC histogram
(which is itself Boltzmann-weighted, so both sides carry the same -S) and
becomes a real error the moment the DoS is reweighted to another coupling.
Use `log_sampled_density` for the histogram overlay rather than reintroducing
the bias into the reconstruction.
"""
from __future__ import annotations

import h5py
import numpy as np


def load_a_history(path):
    """Return {E_n, a_hist, n_nr, n_rm, delta, self_constraint} from an LLR file.

    `a_hist` has shape (n_rep, n_nr + n_rm) — one column per NR iter then
    one per RM sweep. Trailing values are forward-filled if a series is
    shorter than expected (defensive against truncated runs).

    `self_constraint` comes from /cfg@constraint_self and is None for files
    written before that attribute existed; `reconstruct_log_rho` refuses to
    guess, so such files must be re-run or the flag passed by hand.
    """
    with h5py.File(path, "r") as f:
        n_rep = int(f["/cfg"].attrs["n_rep"])
        n_nr = int(f["/cfg"].attrs["n_nr"])
        n_rm = int(f["/cfg"].attrs["n_rm"])
        delta = float(f["/cfg"].attrs["delta"])
        raw_self = f["/cfg"].attrs.get("constraint_self")
        self_constraint = None if raw_self is None else bool(int(raw_self))
        e_n = np.asarray(f["/cfg/E_n"][:], dtype=float)
        a_hist = np.empty((n_rep, n_nr + n_rm), dtype=float)
        for n in range(n_rep):
            a = np.asarray(f[f"/replica_{n:03d}/a"][:], dtype=float)
            m = min(len(a), n_nr + n_rm)
            a_hist[n, :m] = a[:m]
            if m < n_nr + n_rm:
                a_hist[n, m:] = a[-1]
    return {
        "E_n": e_n,
        "a_hist": a_hist,
        "n_nr": n_nr,
        "n_rm": n_rm,
        "delta": delta,
        "self_constraint": self_constraint,
    }


def dos_slope(a_final, *, self_constraint):
    """dln(rho)/dE from the LLR slope a_n — (1 + a) for mode A, a otherwise."""
    if self_constraint is None:
        raise ValueError(
            "self_constraint is unknown: this LLR file predates /cfg@constraint_self. "
            "Re-run the sweep, or pass self_constraint=True for a window on the action "
            "itself and False for an imag-part / observable window."
        )
    return (1.0 + np.asarray(a_final)) if self_constraint else np.asarray(a_final)


def load_hmc_action(path, dataset="/prod/obs/s"):
    """Read an HMC-twin S (or S_I) series from `path` at `dataset`."""
    with h5py.File(path, "r") as f:
        return np.asarray(f[dataset][:], dtype=float)


def reconstruct_log_rho(e_n, a_final, *, self_constraint):
    """ln rho at window centres via trapezoidal integration of the DoS slope.

    Continuity at the segment boundary E_n + delta/2 between adjacent
    piecewise-exponential pieces yields
        ln rho(E_{n+1}) = ln rho(E_n) + (m_n + m_{n+1}) * (E_{n+1} - E_n) / 2
    with m_n = dos_slope(a_n) — see the module docstring for why m != a in
    the self-constrained case. Returns the bare DoS: ln rho_natural for a
    real action, ln rho_pq (phase-quenched, in S_I) for the complex mode.
    """
    slope_n = dos_slope(a_final, self_constraint=self_constraint)
    log_rho = np.zeros_like(e_n)
    for i in range(1, len(e_n)):
        slope = 0.5 * (slope_n[i - 1] + slope_n[i])
        log_rho[i] = log_rho[i - 1] + slope * (e_n[i] - e_n[i - 1])
    return log_rho


def piecewise_log_rho(e_n, a_final, log_rho, delta, n_per_window=16, *, self_constraint):
    """Finely-sampled piecewise-exponential reconstruction of ln rho.

    Inside each window [E_n - delta/2, E_n + delta/2] the local model is
        rho(S) = rho(E_n) * exp(m_n * (S - E_n))
    (linear in log space, slope m_n = dos_slope(a_n)). Adjacent segments
    share their endpoint by construction of `reconstruct_log_rho`.
    """
    slope_n = dos_slope(a_final, self_constraint=self_constraint)
    xs, ys = [], []
    half = 0.5 * delta
    for n in range(len(e_n)):
        seg_x = np.linspace(e_n[n] - half, e_n[n] + half, n_per_window)
        seg_y = log_rho[n] + slope_n[n] * (seg_x - e_n[n])
        xs.append(seg_x)
        ys.append(seg_y)
    return np.concatenate(xs), np.concatenate(ys)


def log_sampled_density(x, log_rho, *, self_constraint):
    """ln of the distribution the HMC twin actually samples, for histogram overlay.

    Mode A samples P(S) ∝ rho(S) e^{-S}, so ln P = ln rho - S. Mode B leaves
    the base weight untouched and histograms the constrained observable, so
    ln P = ln rho. Overlay THIS on a raw HMC histogram — not the bare DoS.
    """
    return np.asarray(log_rho) - np.asarray(x) if self_constraint else np.asarray(log_rho)
