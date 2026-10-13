"""casestudy.eval.metrics — minority-sensitive multi-class metrics (plan §7).

Standalone re-implementation of the macro-averaged metric battery used in
130J's ``ablation/metrics.py``; this package never imports ``ablation``.

All metrics assume **contiguous** labels ``{0, ..., K-1}`` (the case-study data
layer remaps every dataset before any model sees it) and probability columns
ordered ``0..K-1``.

Headline keys returned (match ``CaseStudyConfig.metrics``):
    macro_recall, gmean, macro_f2, macro_specificity
Plus selection / diagnostic extras:
    macro_f1, macro_precision, macro_auc, accuracy
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

print("casestudy/eval/metrics.py loaded")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    """Compute macro-averaged classification metrics for one evaluation.

    Args:
        y_true: True labels, shape ``[N]``, contiguous ``{0..K-1}``.
        y_pred: Predicted labels, shape ``[N]``, same label space.
        y_prob: Class probabilities, shape ``[N, num_classes]`` (columns 0..K-1).
        num_classes: Number of classes ``K`` (e.g. 3).

    Returns:
        Dict with headline keys ``macro_recall``, ``gmean``, ``macro_f2``,
        ``macro_specificity`` and extras ``macro_f1``, ``macro_precision``,
        ``macro_auc``, ``accuracy``.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        fbeta_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_prob = np.asarray(y_prob)

    labels = list(range(num_classes))
    eps = 1e-9

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_f2 = fbeta_score(
        y_true, y_pred, beta=2.0, labels=labels, average="macro", zero_division=0
    )
    macro_precision = precision_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    # Per-class specificity + recall (for G-mean) from the confusion matrix.
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    specificities: List[float] = []
    recalls: List[float] = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specificities.append(tn / (tn + fp + eps))
        recalls.append(tp / (tp + fn + eps))

    macro_specificity = float(np.mean(specificities))

    recalls_arr = np.asarray(recalls)
    if np.any(recalls_arr <= 0):
        gmean = 0.0
    else:
        gmean = float(np.prod(recalls_arr) ** (1.0 / num_classes))

    try:
        macro_auc = float(
            roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro", labels=labels
            )
        )
    except ValueError:
        macro_auc = 0.5

    return {
        "macro_recall": float(macro_recall),
        "gmean": gmean,
        "macro_f2": float(macro_f2),
        "macro_specificity": macro_specificity,
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_auc": macro_auc,
        "accuracy": float(accuracy),
    }


def per_class_recall(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[int, float]:
    """Per-class recall (for the §9.2 "hardest minority class" diagnostics)."""
    from sklearn.metrics import recall_score

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = list(range(num_classes))
    values = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {c: float(v) for c, v in zip(labels, values)}


def aggregate_folds(
    fold_results: List[Dict[str, float]]
) -> Dict[str, Tuple[float, float]]:
    """Summarize per-fold metric dicts as ``{metric: (mean, std)}`` (ddof=0).

    Non-numeric values are skipped, so heterogeneous dicts are tolerated.
    """
    if not fold_results:
        return {}

    summary: Dict[str, Tuple[float, float]] = {}
    for metric in fold_results[0].keys():
        values = [fold[metric] for fold in fold_results if metric in fold]
        if any(
            not isinstance(v, (int, float, np.integer, np.floating)) for v in values
        ):
            continue
        summary[metric] = (float(np.mean(values)), float(np.std(values)))
    return summary
