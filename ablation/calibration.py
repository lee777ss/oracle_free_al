"""
Temperature scaling for post-hoc probability calibration.

Implements the single-scalar temperature-scaling method of
Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.

Given a trained classifier's validation logits `z ∈ R^{N,C}` and labels
`y ∈ {0..C-1}^N`, fit a single scalar T > 0 that minimises the
negative log-likelihood of `softmax(z / T)`. The resulting T is then
used at deployment to produce calibrated probabilities without
retraining or modifying the model weights.

Temperature scaling:
    - Preserves argmax (does NOT change top-1 predictions)
    - Only rescales confidence
    - Is a monotone transformation — reliability diagrams change,
      but ROC / PR curves are invariant

Why this matters for Topic 7 (02b):
    - The PR-curve threshold sweep picks a cutoff on P(minority).
      If probabilities are systematically over/under-confident, the
      chosen threshold will generalise poorly.
    - Temperature-scaling the logits before threshold fitting gives
      a cleaner P(minority) distribution to sweep, and produces
      deployment probabilities that match empirical frequencies.

Usage
-----
>>> from ablation.calibration import fit_temperature, apply_temperature
>>> T = fit_temperature(val_logits, val_labels)
>>> val_probs = apply_temperature(val_logits, T)
>>> assert np.allclose(val_probs.sum(axis=1), 1.0)

Numerical implementation
------------------------
All work done in numpy via log-sum-exp for stability.
`scipy.optimize.minimize_scalar` with bounded method; T clipped to
[0.05, 20] to guard against degenerate fits.
"""

from typing import Union

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


__all__ = ['fit_temperature', 'apply_temperature']

# Clip range for T. Below 0.05 → softmax is essentially one-hot
# (numerical overflow risk); above 20 → softmax is essentially uniform
# (no information). These bounds also prevent the optimiser from
# wandering into flat regions of the NLL surface.
_T_MIN = 0.05
_T_MAX = 20.0


def _nll_at_T(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood of softmax(logits / T) at the given labels.

    Uses log-sum-exp for numerical stability:
        log_softmax(z / T)[y] = (z[y] / T) - logsumexp(z / T)
        NLL = -mean_i log_softmax(logits_i / T)[labels_i]
    """
    scaled = logits / T                                   # [N, C]
    log_Z = logsumexp(scaled, axis=1)                     # [N]
    # row-wise gather of the true-class scaled logit
    true_class_scaled = scaled[np.arange(scaled.shape[0]), labels]  # [N]
    log_probs = true_class_scaled - log_Z                 # [N]
    return float(-np.mean(log_probs))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 50,
) -> float:
    """Fit a single scalar temperature T > 0 on validation logits.

    Minimises the NLL of ``softmax(logits / T)`` via
    ``scipy.optimize.minimize_scalar`` with ``method='bounded'``.

    Parameters
    ----------
    logits : np.ndarray, shape (N, C), float
        Raw pre-softmax logits on the validation set.
    labels : np.ndarray, shape (N,), int
        True class labels in contiguous space {0..C-1} (already remapped).
    max_iter : int, default 50
        Maximum iterations for the bounded scalar optimiser.

    Returns
    -------
    T : float
        Fitted temperature, clipped to ``[0.05, 20]``.
        Guards:
          * ``len(labels) == 0`` → returns 1.0 (no data)
          * Only one class present in ``labels`` → returns 1.0
            (NLL minimisation is ill-posed; skip scaling)

    Notes
    -----
    Reference: Guo, Pleiss, Sun, Weinberger, ICML 2017,
    "On Calibration of Modern Neural Networks", §4.2 Temperature Scaling.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D, got shape {logits.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits/labels length mismatch: "
            f"{logits.shape[0]} vs {labels.shape[0]}"
        )

    # Degenerate cases — skip fit, return identity temperature.
    if labels.shape[0] == 0:
        return 1.0
    if np.unique(labels).size < 2:
        # Only one class present → NLL is monotone in T (no well-defined min)
        return 1.0

    result = minimize_scalar(
        _nll_at_T,
        args=(logits, labels),
        bounds=(_T_MIN, _T_MAX),
        method='bounded',
        options={'maxiter': max_iter, 'xatol': 1e-4},
    )

    T = float(result.x)
    # Extra safety: clip in case the optimiser returned a boundary value
    # very slightly outside [_T_MIN, _T_MAX] due to floating point.
    return float(np.clip(T, _T_MIN, _T_MAX))


def apply_temperature(
    logits: np.ndarray,
    T: Union[float, int],
) -> np.ndarray:
    """Return ``softmax(logits / T)`` computed in a numerically stable way.

    Parameters
    ----------
    logits : np.ndarray, shape (N, C)
        Raw pre-softmax logits.
    T : float
        Temperature. Must be > 0. Typically the output of
        :func:`fit_temperature`.

    Returns
    -------
    probs : np.ndarray, shape (N, C)
        Calibrated probabilities. Each row sums to 1.
    """
    if T <= 0:
        raise ValueError(f"Temperature must be > 0, got T={T}")
    logits = np.asarray(logits, dtype=np.float64)
    scaled = logits / float(T)
    # Stable softmax via subtraction of row-max
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


print("calibration.py loaded")
