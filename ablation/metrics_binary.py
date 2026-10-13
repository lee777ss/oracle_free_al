"""
Binary (minority-focused) metrics for imbalanced classification — 515J.

All metrics are computed after binarizing labels:
    Original class   {0}    → negative (0)  [majority / normal]
    Original classes {1, 2} → positive (1)  [minority / anomaly]

This collapses the multi-class problem into a binary detection task:
"Can the classifier detect ANY minority class vs. the majority?"

Aligns with Chung et al. (2022) who compute F1/Precision/Recall
in a binary anomaly-detection framework (anomaly = positive class).

Metrics returned (8 total, all with 'binary_' prefix):
    binary_f1, binary_precision, binary_recall,
    binary_specificity, binary_gmean, binary_pr_auc, binary_auc, binary_accuracy

Usage:
    from ablation.metrics_binary import compute_metrics_binary, compare_folds_binary
"""

print("metrics_binary.py loaded")

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix, accuracy_score,
    fbeta_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit


def _binarize(y):
    """Binarize labels: class 0 → 0 (majority/negative), classes 1..N-1 → 1 (minority/positive).

    Args:
        y: array-like of integer class labels in {0, 1, ..., num_classes-1}.

    Returns:
        np.ndarray of shape [N] with values in {0, 1}.
    """
    y = np.asarray(y)
    return (y > 0).astype(int)


def _binarize_probs(y_prob):
    """Binarize probability matrix: sum P(class>=1) into positive probability.

    Args:
        y_prob: array-like of shape [N, num_classes] — softmax probabilities.

    Returns:
        np.ndarray of shape [N] — P(minority) = sum of columns 1..N-1.
        Returns None if y_prob is None.
    """
    if y_prob is None:
        return None
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        return y_prob  # already binary probabilities
    # P(positive) = 1 - P(class 0) = sum of P(class 1) + P(class 2) + ...
    return y_prob[:, 1:].sum(axis=1)


def compute_metrics_binary(
    y_true,
    y_pred,
    y_prob=None,
    num_classes: int = 3,
) -> Dict[str, float]:
    """Compute 7 binary metrics after binarizing multi-class labels.

    Binarization: class 0 → negative (majority), classes {1,...,N-1} → positive (minority).

    Args:
        y_true: Ground truth labels in {0, 1, ..., num_classes-1}.
        y_pred: Predicted labels in {0, 1, ..., num_classes-1}.
        y_prob: Softmax probabilities [N, num_classes] or None.
        num_classes: Number of original classes (default 3).

    Returns:
        Dict with keys:
            binary_f1, binary_precision, binary_recall,
            binary_specificity, binary_gmean, binary_auc, binary_accuracy
    """
    # Binarize labels and predictions
    y_true_bin = _binarize(y_true)
    y_pred_bin = _binarize(y_pred)
    y_prob_bin = _binarize_probs(y_prob)

    # pos_label=1 means minority is the "positive" class
    binary_f1 = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    binary_precision = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    binary_recall = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)

    # Specificity = TN / (TN + FP) — recall of the negative (majority) class
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    binary_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # G-mean = sqrt(recall * specificity)
    binary_gmean = float(np.sqrt(binary_recall * binary_specificity))

    # PR-AUC (Average Precision) and ROC-AUC — require probability scores
    if y_prob_bin is not None:
        try:
            binary_pr_auc = average_precision_score(y_true_bin, y_prob_bin)
        except ValueError:
            binary_pr_auc = 0.0
        try:
            binary_auc = roc_auc_score(y_true_bin, y_prob_bin)
        except ValueError:
            binary_auc = 0.5
    else:
        binary_pr_auc = 0.0
        binary_auc = 0.5

    # Accuracy on binarized labels
    binary_accuracy = accuracy_score(y_true_bin, y_pred_bin)

    return {
        'binary_f1':          float(binary_f1),
        'binary_precision':   float(binary_precision),
        'binary_recall':      float(binary_recall),
        'binary_specificity': float(binary_specificity),
        'binary_gmean':       float(binary_gmean),
        'binary_pr_auc':      float(binary_pr_auc),
        'binary_auc':         float(binary_auc),
        'binary_accuracy':    float(binary_accuracy),
    }


def compare_folds_binary(
    fold_results: List[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    """Compute mean ± std across k folds for binary metrics.

    Args:
        fold_results: List of k dicts, each from compute_metrics_binary().

    Returns:
        Dict mapping metric_name → (mean, std).
        Example: {'binary_f1': (0.820, 0.014), ...}
    """
    if not fold_results:
        return {}

    all_keys = fold_results[0].keys()
    summary = {}
    for key in all_keys:
        values = [fr[key] for fr in fold_results if key in fr]
        # Skip non-numeric keys (e.g. Ab-11a's `fold_history` dict).
        if values and any(not isinstance(v, (int, float, np.integer, np.floating))
                          for v in values):
            continue
        if values:
            summary[key] = (float(np.mean(values)), float(np.std(values)))
        else:
            summary[key] = (0.0, 0.0)
    return summary


def format_results_table_binary(
    all_results: Dict[str, Dict],
    modes: List[str] = None,
) -> pd.DataFrame:
    """Format binary CV summary results into a display DataFrame.

    Args:
        all_results: Dict mapping mode → {cv_summary_binary: {metric: (mean, std)}}.
        modes: List of mode keys to include (default: all).

    Returns:
        pd.DataFrame with rows=modes, columns=metrics, values="0.820 ± 0.014".
    """
    if modes is None:
        modes = list(all_results.keys())

    metric_order = [
        'binary_f1', 'binary_gmean', 'binary_auc',
        'binary_precision', 'binary_recall', 'binary_specificity',
        'binary_accuracy',
    ]
    metric_display = {
        'binary_f1':          'F1-Score',
        'binary_gmean':       'G-Mean',
        'binary_auc':         'ROC-AUC',
        'binary_precision':   'Precision',
        'binary_recall':      'Recall',
        'binary_specificity': 'Specificity',
        'binary_accuracy':    'Accuracy',
    }

    rows = []
    for mode in modes:
        if mode not in all_results:
            continue
        cv = all_results[mode].get('cv_summary_binary', {})
        row = {}
        for m in metric_order:
            mean, std = cv.get(m, (0, 0))
            row[metric_display.get(m, m)] = f"{mean:.3f} ± {std:.3f}"
        rows.append(row)

    return pd.DataFrame(rows, index=modes)


# ================================================================
# Topic 7 — PR-curve threshold tuning (Step D of 02a plan)
# ================================================================
#
# These helpers are used by 02b_threshold_tuning.ipynb to pick a
# single deployment threshold τ on the positive-class probability
#
#     p_minority = probs[:, 1:].sum(axis=1)
#
# for two objectives:
#   * 'binary_f1' — F1 of the {0} vs {1..C-1} detection task (515J view)
#   * 'macro_f1'  — full C-class macro F1 with a 1-D minority gate;
#                   if p_minority < τ predict 0 (majority), else
#                   predict argmax over the minority columns 1..C-1.
#
# The 1-D gate is a deliberate simplification: per-class 2-D thresholds
# are left as future work. The 1-D version already isolates the effect
# of correcting for prior shift without multiplying hyperparameters.


def find_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    objective: str = 'binary_f1',
) -> Dict[str, Any]:
    """Sweep candidate thresholds and return the best one for the objective.

    Uses the canonical PR-curve grid from ``sklearn.precision_recall_curve``
    on ``p_minority = probs[:, 1:].sum(axis=1)``. The same grid is used for
    both objectives — the only thing that changes is how the score at each
    candidate is computed.

    Args:
        y_true: [N_total] contiguous labels in ``{0..C-1}`` (already remapped).
        probs:  [N_total, C] softmax probabilities (after temperature scaling).
        objective: ``'binary_f1'``, ``'macro_f1'`` or ``'macro_f2'``.

    Returns:
        Dict with keys:
            best_threshold: float — τ* that maximises the objective.
            best_score:     float — value of the objective at τ*.
            precisions:     np.ndarray [M+1] — from ``precision_recall_curve``.
            recalls:        np.ndarray [M+1] — from ``precision_recall_curve``.
            thresholds:     np.ndarray [M]   — candidate thresholds (sorted).
            fscores:        np.ndarray [M]   — objective value at each
                                               candidate (NOT always F1;
                                               for 'macro_f1' this is
                                               macro_f1 per candidate,
                                               for 'macro_f2' this is
                                               macro_f2 per candidate).
            objective:      str — echo of the input.

    Guards:
        * ``y_true`` has no positives → returns
          ``{'best_threshold': 0.5, 'best_score': 0.0, ... empty arrays}``.
        * Unknown objective → ``ValueError``.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)

    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D, got shape {probs.shape}")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"probs/y_true length mismatch: "
            f"{probs.shape[0]} vs {y_true.shape[0]}"
        )
    if objective not in ('binary_f1', 'macro_f1', 'macro_f2'):
        raise ValueError(
            f"objective must be 'binary_f1', 'macro_f1' or 'macro_f2', got {objective!r}"
        )

    num_classes = probs.shape[1]
    p_minority = probs[:, 1:].sum(axis=1)       # [N]
    y_bin = (y_true > 0).astype(int)            # [N]

    # Degenerate: no positives in this split → F1 is always 0.
    # Return a safe placeholder so the caller can skip the fold cleanly.
    if y_bin.sum() == 0:
        return {
            'best_threshold': 0.5,
            'best_score':     0.0,
            'precisions':     np.array([]),
            'recalls':        np.array([]),
            'thresholds':     np.array([]),
            'fscores':        np.array([]),
            'objective':      objective,
        }

    # sklearn returns (precisions, recalls, thresholds) with
    #   len(precisions) == len(recalls) == len(thresholds) + 1
    # The trailing entry is "predict all negative" (threshold=+inf).
    precisions, recalls, thresholds = precision_recall_curve(y_bin, p_minority)

    if objective == 'binary_f1':
        # F1 at each grid point (vectorised, aligned with thresholds[:])
        p_trim = precisions[:-1]
        r_trim = recalls[:-1]
        denom = p_trim + r_trim
        fscores = np.where(denom > 0, 2.0 * p_trim * r_trim / np.maximum(denom, 1e-12), 0.0)
    else:
        # macro_f1/macro_f2 — needs to re-derive predictions at each threshold.
        # For each τ: pred = 0 if p_minority < τ else argmax(probs[:, 1:]) + 1.
        # This is O(|grid| · N) but |grid| ≤ N in the worst case,
        # so for N ≈ 6.5k total (pooled across 5 folds) this is ~40 M ops —
        # fine. If it ever becomes a bottleneck, switch to an incremental
        # confusion-matrix update along the sorted p_minority axis.
        minority_argmax = probs[:, 1:].argmax(axis=1) + 1   # [N] in {1..C-1}
        fscores = np.empty(len(thresholds), dtype=np.float64)
        class_labels = np.arange(num_classes)
        for i, tau in enumerate(thresholds):
            pred = np.where(p_minority >= tau, minority_argmax, 0)
            if objective == 'macro_f1':
                fscores[i] = f1_score(
                    y_true, pred,
                    labels=class_labels,
                    average='macro',
                    zero_division=0,
                )
            else:
                fscores[i] = fbeta_score(
                    y_true, pred,
                    labels=class_labels,
                    beta=2.0,
                    average='macro',
                    zero_division=0,
                )

    # argmax picks the earliest tie — fine for a deterministic deployment τ.
    best_idx = int(np.argmax(fscores))
    return {
        'best_threshold': float(thresholds[best_idx]),
        'best_score':     float(fscores[best_idx]),
        'precisions':     precisions,
        'recalls':        recalls,
        'thresholds':     thresholds,
        'fscores':        fscores,
        'objective':      objective,
    }


def pooled_threshold_with_holdout(
    fold_logits: List[np.ndarray],
    fold_labels: List[np.ndarray],
    fold_T: List[float],
    objective: str = 'binary_f1',
    calib_frac: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Honest pooled threshold + held-out assessment (unbiased CV score).

    Per fold, stratified-split the validation split into calib/assess halves.
    Pool all calib halves → tune one threshold on pooled calib.
    Pool all assess halves → evaluate the chosen threshold. This gives a
    CV number that is NOT biased by threshold-selection overfitting.

    The **deployment** threshold should still be tuned on all val data
    via :func:`find_best_threshold` — this routine only exists to report
    an unbiased value alongside.

    Args:
        fold_logits: length-K list, element k is ``np.ndarray [N_val_k, C]``
                     of raw pre-softmax logits (from ``run_kfold_ablation``
                     return key ``fold_val_logits``).
        fold_labels: length-K list, element k is ``np.ndarray [N_val_k]``
                     of remapped labels in ``{0..C-1}``.
        fold_T:      length-K list of per-fold temperatures (from
                     :func:`ablation.calibration.fit_temperature`).
        objective:   ``'binary_f1'``, ``'macro_f1'`` or ``'macro_f2'``.
        calib_frac:  fraction of each fold's val split used for threshold
                     fitting; the complement is the held-out assess set.
        seed:        RNG seed for the stratified split.

    Returns:
        Dict with keys:
            holdout_threshold: float — τ* picked on pooled calib.
            holdout_score:     float — objective value on pooled assess
                                       (this is the honest CV number).
            calib_score:       float — objective value on pooled calib
                                       at τ* (optimistic, for reference).
            objective:         str.
            calib_size:        int — total pooled calib samples.
            assess_size:       int — total pooled assess samples.
    """
    # Local import avoids a circular dep (calibration imports from metrics
    # nowhere, but keeping the top-level import list minimal is still nice).
    from .calibration import apply_temperature

    if not (0.0 < calib_frac < 1.0):
        raise ValueError(f"calib_frac must be in (0, 1), got {calib_frac}")
    if len(fold_logits) != len(fold_labels) or len(fold_logits) != len(fold_T):
        raise ValueError(
            "fold_logits, fold_labels, fold_T must have the same length "
            f"(got {len(fold_logits)}, {len(fold_labels)}, {len(fold_T)})"
        )

    calib_probs_parts:  List[np.ndarray] = []
    calib_labels_parts: List[np.ndarray] = []
    assess_probs_parts:  List[np.ndarray] = []
    assess_labels_parts: List[np.ndarray] = []

    for k, (logits, labels, T) in enumerate(zip(fold_logits, fold_labels, fold_T)):
        logits = np.asarray(logits)
        labels = np.asarray(labels)
        probs = apply_temperature(logits, T)

        # Stratified split of this fold's val set.
        # Guard: if the rarer class has only 1 sample, StratifiedShuffleSplit
        # would fail; fall back to a plain index split in that edge case.
        y_bin = (labels > 0).astype(int)
        min_count = int(min(np.bincount(y_bin, minlength=2)))
        if min_count < 2:
            # Not enough to stratify — use deterministic slice.
            n = len(labels)
            n_calib = int(round(calib_frac * n))
            rng = np.random.default_rng(seed + k)
            perm = rng.permutation(n)
            calib_idx = perm[:n_calib]
            assess_idx = perm[n_calib:]
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1.0 - calib_frac,
                random_state=seed + k,
            )
            calib_idx, assess_idx = next(splitter.split(np.zeros_like(y_bin), y_bin))

        calib_probs_parts.append(probs[calib_idx])
        calib_labels_parts.append(labels[calib_idx])
        assess_probs_parts.append(probs[assess_idx])
        assess_labels_parts.append(labels[assess_idx])

    calib_probs = np.concatenate(calib_probs_parts, axis=0)
    calib_labels = np.concatenate(calib_labels_parts, axis=0)
    assess_probs = np.concatenate(assess_probs_parts, axis=0)
    assess_labels = np.concatenate(assess_labels_parts, axis=0)

    # Tune on pooled calib
    tune = find_best_threshold(calib_labels, calib_probs, objective=objective)
    tau_star = tune['best_threshold']
    calib_score = tune['best_score']

    # Evaluate chosen τ on pooled assess
    holdout_score = _score_at_threshold(
        assess_labels, assess_probs, tau_star, objective
    )

    return {
        'holdout_threshold': float(tau_star),
        'holdout_score':     float(holdout_score),
        'calib_score':       float(calib_score),
        'objective':         objective,
        'calib_size':        int(len(calib_labels)),
        'assess_size':       int(len(assess_labels)),
    }


def _score_at_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    objective: str,
) -> float:
    """Compute the objective at a fixed threshold.

    Shared by :func:`pooled_threshold_with_holdout` and by diagnostic
    code in 02b that needs "score this specific τ on this specific split".
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    num_classes = probs.shape[1]
    p_minority = probs[:, 1:].sum(axis=1)

    if objective == 'binary_f1':
        y_bin = (y_true > 0).astype(int)
        pred_bin = (p_minority >= threshold).astype(int)
        return float(f1_score(y_bin, pred_bin, pos_label=1, zero_division=0))
    elif objective == 'macro_f1':
        minority_argmax = probs[:, 1:].argmax(axis=1) + 1
        pred = np.where(p_minority >= threshold, minority_argmax, 0)
        return float(f1_score(
            y_true, pred,
            labels=np.arange(num_classes),
            average='macro',
            zero_division=0,
        ))
    elif objective == 'macro_f2':
        minority_argmax = probs[:, 1:].argmax(axis=1) + 1
        pred = np.where(p_minority >= threshold, minority_argmax, 0)
        return float(fbeta_score(
            y_true, pred,
            labels=np.arange(num_classes),
            beta=2.0,
            average='macro',
            zero_division=0,
        ))
    else:
        raise ValueError(
            f"objective must be 'binary_f1', 'macro_f1' or 'macro_f2', got {objective!r}"
        )
