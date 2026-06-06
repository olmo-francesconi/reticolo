"""Polynomial fit of LLR a_n data for complex-action reconstruction.

The piecewise-exponential reconstruction carries the per-window noise of
each a_n straight into the oscillating integral, where cancellation over
many orders of magnitude amplifies it catastrophically. Instead a_n =
d ln rho/dS is fitted globally, as a function of the intensive variable
s = S/V, with an odd polynomial a(s) = sum_{k odd} c_k s^k (rho even =>
a odd). ln rho(sV) = V * P(s) with P the even antiderivative, P(0) = 0.

Fits are unweighted: the RM-tail spread is not a faithful per-window error
(the RM iterates are an autocorrelated converging sequence), so chi2/dof is
relative, not absolute — order selection uses its plateau only. Statistical
errors come from independent-seed replica runs, each fitted separately.
"""
from __future__ import annotations

import numpy as np


def rm_tail(a_hist, tail_frac=0.2):
    """Per-window tail slice of the RM history, shape (n_rep, n_tail)."""
    n_tail = max(1, int(a_hist.shape[1] * tail_frac))
    return a_hist[:, -n_tail:]


def fit_odd_poly(s, a, order):
    """Unweighted LSQ of a(s) onto the odd basis s, s^3, ..., s^order.

    Fitted in the rescaled variable x = s/max|s| so the design matrix stays
    well-conditioned at high order (s ~ 1e-2 makes raw s^11 columns fall
    below the lstsq rank cutoff and silently truncates the basis).

    Returns (coeffs indexed by power 0..order, even entries zero; chi2/dof
    with unit errors — meaningful only relative to other orders).
    """
    s_ref = float(np.abs(s).max())
    x = s / s_ref
    powers = np.arange(1, order + 1, 2)
    design = x[:, None] ** powers[None, :]
    c, *_ = np.linalg.lstsq(design, a, rcond=None)
    resid = design @ c - a
    dof = len(s) - len(powers)
    chi2_dof = float(resid @ resid) / dof if dof > 0 else np.inf
    coeffs = np.zeros(order + 1)
    coeffs[powers] = c / s_ref**powers
    return coeffs, chi2_dof


def select_order(s, a, orders=(1, 3, 5, 7, 9, 11)):
    """Increase the odd order until chi2/dof stops improving by >20%.

    Returns ((order, coeffs, chi2_dof), table) with table = [(order,
    chi2_dof), ...] for logging.
    """
    fits = []
    for o in orders:
        if o >= len(s):
            break
        coeffs, chi2_dof = fit_odd_poly(s, a, o)
        fits.append((o, coeffs, chi2_dof))
    best = fits[0]
    for cur in fits[1:]:
        if cur[2] > 0.8 * best[2]:
            break
        best = cur
    return best, [(o, x2) for o, _, x2 in fits]


def log_rho_coeffs(a_coeffs):
    """Even antiderivative P with P(0) = 0: ln rho(sV) = V * P(s)."""
    out = np.zeros(len(a_coeffs) + 1)
    out[1:] = a_coeffs / (np.arange(len(a_coeffs)) + 1.0)
    return out
