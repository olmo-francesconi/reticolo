"""Integrated autocorrelation analysis for MCMC time series.

Computes the autocorrelation function via FFT (O(N log N)) and the integrated
autocorrelation time via the Madras-Sokal automatic-windowing procedure.

Madras-Sokal (Madras & Sokal 1988, "The pivot algorithm: a highly efficient
Monte Carlo method for the self-avoiding walk", appendix): given the
autocorrelation function ρ(t), τ_int(W) = 1/2 + Σ_{t=1..W} ρ(t). The window W
is grown until W ≥ c·τ_int(W); the smallest such W is the cutoff. c ≈ 4-6 is
standard (Wolff "Monte Carlo errors with less errors" arXiv:hep-lat/0306017).

We expose the same quantities in *wall-time* units. Given the per-update wall
cost s_per_update, τ_int_seconds = τ_int_units · s_per_update. This is the
cross-algorithm comparison the example wants — autocorrelation per CPU second,
not per algorithm-internal step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation function ρ(t) of a 1D real series.

    ρ(0) == 1 by construction; ρ(t) computed via FFT of the de-meaned series
    on the zero-padded length-2N window. Returns the first N lags (ρ(0)..ρ(N-1)).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        raise ValueError("autocorrelation needs at least 2 samples")
    y = x - x.mean()
    # Zero-pad to next power of two ≥ 2N so the cyclic FFT computes the linear
    # correlation without wrap-around.
    m = 1
    while m < 2 * n:
        m *= 2
    f = np.fft.rfft(y, m)
    acf = np.fft.irfft(f * np.conjugate(f), m)[:n]
    return acf / acf[0]


@dataclass
class TauIntResult:
    """Integrated autocorrelation time + windowing diagnostics.

    `tau_int` and `tau_int_err` are in update units. `window` is the
    Madras-Sokal cutoff. `tau_int_seconds` and `tau_int_seconds_err` carry the
    same numbers rescaled by `s_per_update`.
    """

    tau_int: float
    tau_int_err: float
    window: int
    rho: np.ndarray
    s_per_update: float
    tau_int_seconds: float
    tau_int_seconds_err: float
    n_samples: int


def tau_int(
    x: np.ndarray,
    *,
    s_per_update: float = 1.0,
    c: float = 6.0,
) -> TauIntResult:
    """Madras-Sokal integrated autocorrelation time, with wall-time conversion.

    Parameters
    ----------
    x : 1D array
        Time series of one observable; one entry per MC update.
    s_per_update : float
        Seconds of wall-clock per update. The result then includes
        `tau_int_seconds = tau_int · s_per_update`.
    c : float
        Madras-Sokal window factor. The window grows until `W ≥ c · τ_int(W)`.
        6 is the standard "safe" choice; smaller values bias τ_int down.

    Returns
    -------
    TauIntResult
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    rho = autocorrelation(x)

    # Grow W until W ≥ c · τ_int(W). cumsum runs over t = 1..n-1 so τ at
    # window W is 1/2 + sum_{t=1..W} ρ(t).
    cumsum = np.cumsum(rho[1:])
    tau_at = 0.5 + cumsum  # tau_at[W-1] is τ_int with window W (W ≥ 1)
    windows = np.arange(1, n)
    found = np.where(windows >= c * tau_at)[0]
    if found.size == 0:
        # Never converged within the series length — return the largest window
        # we have, but warn the caller via window = n - 1.
        w = n - 1
    else:
        w = int(found[0]) + 1
    tau = float(tau_at[w - 1])

    # Standard Madras-Sokal error estimate: σ(τ_int) ≈ τ_int · √(2(2W+1)/N).
    tau_err = tau * float(np.sqrt(2.0 * (2 * w + 1) / n))

    return TauIntResult(
        tau_int=tau,
        tau_int_err=tau_err,
        window=w,
        rho=rho,
        s_per_update=s_per_update,
        tau_int_seconds=tau * s_per_update,
        tau_int_seconds_err=tau_err * s_per_update,
        n_samples=n,
    )
