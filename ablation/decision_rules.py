"""Decision rules under prior shift (Topic 7 — Step E of 02a plan).

Three deterministic, numpy-only prediction rules used by
``02b_threshold_tuning.ipynb`` to compare the consequences of

1. plain ``argmax`` on the classifier's logits,
2. Menon et al. (ICLR 2021) *logit adjustment* — the closed-form
   Bayes-optimal correction under pure prior shift, and
3. a PR-curve-tuned scalar threshold on the minority-gate probability.

Why this lives in a separate module
-----------------------------------
``metrics_binary.py`` scores predictions. ``calibration.py`` rescales
logits. Neither of them should own the semantics of "how do we turn a
logit vector into a class label given an assumed deployment prior" —
that is a modelling choice, not a metric and not a calibration.

Effective ``pi_train`` per ablation mode (C=3, Fashion-MNIST {0,4,6})
--------------------------------------------------------------------

    Mode  |  pi_train (effective training prior)
    ------+--------------------------------------
    ab1   |  pool prior ≈ [0.952, 0.024, 0.024]   (no rebalancing)
    ab5   |  [1/3, 1/3, 1/3]                      (BAGAN dynamic balance)
    ab7   |  [1/3, 1/3, 1/3]
    ab10  |  [1/3, 1/3, 1/3]
    ab11  |  [1/3, 1/3, 1/3]

``pi_test`` is the **pool prior** (not the held-out test prior — that
would be an oracle). We are simulating deployment where the only prior
we know is our own training-pool class frequencies.

Ab-1's logit adjustment is by construction the identity map
(``pi_train ≈ pi_test`` ⇒ ``offset ≈ 0``) and is included as a sanity
check — :func:`predict_with_logit_adjustment` should agree with plain
argmax up to floating-point noise.
"""
print("decision_rules.py loaded")

import numpy as np


def logit_adjustment_offset(
    pi_train: np.ndarray,
    pi_test: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Closed-form Bayes-optimal logit offset under prior shift.

    Menon et al., "Long-tail learning via logit adjustment", ICLR 2021.
    Under the assumption that only the class prior differs between
    training and deployment (``P_train(x|y) = P_test(x|y)``), the
    Bayes-optimal classifier for 0-1 loss on the test distribution is

        ŷ = argmax_c ( z_c + log(pi_test_c / pi_train_c) )

    where ``z`` are the training-distribution logits.

    Args:
        pi_train: [C] effective training prior; each entry in (0, 1],
                  sums to 1 up to float tolerance. Zero entries are
                  clipped to ``eps`` to avoid ``-inf``.
        pi_test:  [C] assumed deployment prior; same shape/semantics.
        eps:      lower clip for numerical safety.

    Returns:
        [C] float64 offset vector ``b`` — add to logits before argmax.

    Raises:
        ValueError: shape mismatch or non-1-D input.
    """
    pi_train = np.asarray(pi_train, dtype=np.float64)
    pi_test = np.asarray(pi_test, dtype=np.float64)
    if pi_train.ndim != 1 or pi_test.ndim != 1:
        raise ValueError(
            f"pi_train and pi_test must be 1-D; got shapes "
            f"{pi_train.shape} and {pi_test.shape}"
        )
    if pi_train.shape != pi_test.shape:
        raise ValueError(
            f"pi_train and pi_test must have the same length; got "
            f"{pi_train.shape[0]} vs {pi_test.shape[0]}"
        )
    return np.log(np.clip(pi_test, eps, None) / np.clip(pi_train, eps, None))


def predict_with_logit_adjustment(
    logits: np.ndarray,
    pi_train: np.ndarray,
    pi_test: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Predict class labels with logit adjustment applied.

    Convenience wrapper: adds :func:`logit_adjustment_offset` to each
    row of ``logits`` then takes argmax. Softmax is not needed because
    argmax is invariant to any monotone transform applied element-wise.

    Args:
        logits:   [N, C] raw pre-softmax scores.
        pi_train: [C] effective training prior.
        pi_test:  [C] assumed deployment prior.
        eps:      lower clip for numerical safety in the offset.

    Returns:
        [N] int64 array of predicted class indices in ``{0..C-1}``.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D [N, C]; got shape {logits.shape}")
    offset = logit_adjustment_offset(pi_train, pi_test, eps=eps)
    if offset.shape[0] != logits.shape[1]:
        raise ValueError(
            f"pi_train/pi_test length ({offset.shape[0]}) must match "
            f"num_classes ({logits.shape[1]})"
        )
    # Broadcasting: [N, C] + [C] — row-wise offset addition.
    return (logits + offset).argmax(axis=1).astype(np.int64)


def predict_with_threshold(
    probs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Minority-gate threshold rule.

    Decision: compute ``s(x) = sum_{c>=1} probs[x, c]``.
    If ``s(x) >= threshold`` predict ``argmax_{c>=1} probs[x, c]``
    (i.e. the most likely minority class); else predict class 0
    (the majority class).

    This is the 1-D gate described in Step D of the plan — a single
    scalar threshold on "how minority-like is this sample", with ties
    among minority classes broken by their relative softmax mass.

    Args:
        probs:     [N, C] softmax probabilities. Each row should sum
                   to 1 (not enforced — caller's responsibility).
        threshold: scalar in ``[0, 1]`` — the minority gate.

    Returns:
        [N] int64 predicted class labels in ``{0..C-1}``.

    Raises:
        ValueError: if ``probs`` is not 2-D or has fewer than 2 classes.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D [N, C]; got shape {probs.shape}")
    if probs.shape[1] < 2:
        raise ValueError(
            f"probs must have at least 2 classes; got C={probs.shape[1]}"
        )
    num_classes = probs.shape[1]
    p_minority = probs[:, 1:].sum(axis=1)                    # [N]
    minority_argmax = probs[:, 1:].argmax(axis=1) + 1        # [N] in {1..C-1}
    pred = np.where(p_minority >= threshold, minority_argmax, 0)
    return pred.astype(np.int64)
