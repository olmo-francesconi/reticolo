"""Block-jackknife error analysis for autocorrelated MCMC time series.

Both reference examples bin their per-configuration measurements into ~n_blocks
blocks, then jackknife on the *derived* quantity (susceptibility, Binder
cumulant) so the error includes the non-linearity of the estimator. Binning
absorbs any residual autocorrelation as long as the block size is larger than
the integrated autocorrelation time τ_int — for the examples we use n_blocks
around 40-50, which is enough for the modest run lengths shipped.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def _bin(samples: np.ndarray, n_blocks: int) -> np.ndarray:
    """Return one mean per block. Drops trailing samples that don't fit."""
    n = len(samples)
    if n_blocks > n:
        raise ValueError(f"n_blocks={n_blocks} > n_samples={n}")
    block_size = n // n_blocks
    trimmed = samples[: block_size * n_blocks].reshape(n_blocks, block_size)
    return trimmed.mean(axis=1)


Estimator = Callable[[tuple[float, ...]], float]


def jackknife(
    series: Sequence[np.ndarray],
    estimator: Estimator,
    n_blocks: int = 50,
) -> tuple[float, float]:
    """Block-jackknife (mean, stderr) of `estimator(series)`.

    `series` is a tuple of equal-length per-configuration arrays. `estimator`
    takes a tuple of scalars — the means of each input observable over the
    remaining blocks — and returns a scalar (susceptibility, Binder cumulant,
    ...). The jackknife error captures both autocorrelation (via binning) and
    estimator non-linearity (via leave-one-out reweighting).
    """
    binned = [_bin(np.asarray(s), n_blocks) for s in series]

    def means_over(mask: np.ndarray) -> tuple[float, ...]:
        return tuple(float(b[mask].mean()) for b in binned)

    full = estimator(means_over(np.ones(n_blocks, dtype=bool)))
    loo = np.empty(n_blocks)
    for i in range(n_blocks):
        mask = np.ones(n_blocks, dtype=bool)
        mask[i] = False
        loo[i] = estimator(means_over(mask))
    err = float(np.sqrt((n_blocks - 1) / n_blocks * np.sum((loo - full) ** 2)))
    return float(full), err
