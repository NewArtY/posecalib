"""Robust scale estimate and IRLS weights."""
from __future__ import annotations

import numpy as np

LOSSES = ('linear', 'huber', 'cauchy')


def robust_sigma(residuals, *, floor=np.sqrt(np.finfo(float).eps)) -> float:
    """``1.4826 * median(|r|)``, bounded below by numerical resolution.

    A zero MAD is possible for exact fits or a majority of exact inliers; it
    must not produce a zero IRLS threshold. ``floor`` has the same units as the
    residual components (including confidence weights), not noise variance.
    """
    v = np.abs(np.asarray(residuals).ravel())
    return float(max(1.4826 * np.median(v), floor))


def robust_weight(u, loss: str):
    """IRLS weight for a normalised residual ``u = |r| / k``.

    ``huber``: 1 for u <= 1, 1/u otherwise; ``cauchy``: 1 / (1 + u^2);
    ``linear``: 1 (ordinary least squares).
    """
    if loss == 'huber':
        return np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
    if loss == 'cauchy':
        return 1.0 / (1.0 + u ** 2)
    if loss == 'linear':
        return np.ones_like(u)
    raise ValueError(f'unknown loss {loss!r}; expected one of {LOSSES}')
