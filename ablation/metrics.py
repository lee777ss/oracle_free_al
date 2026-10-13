"""
Macro-averaged metrics for multi-class imbalanced classification.

Extracted and generalized from GAAL_Tunnel_427J's calculate_binary_metrics().

Key change from 427J → 511J:
    427J: Binary classification — metrics computed directly from TP/FP/TN/FN scalars.
    511J: Multi-class — metrics computed PER-CLASS from confusion matrix rows,
          then MACRO-AVERAGED (unweighted mean across classes).

Why macro-averaging?
--------------------
With imbalanced data (e.g., 90% T-shirt, 5% Pullover, 5% Dress),
accuracy is misleading — a model predicting "T-shirt" for everything
gets 90% accuracy but 0% on minority classes.

Macro-averaging treats ALL classes equally:
    macro_metric = (metric_class0 + metric_class1 + ... + metric_classN) / N

This ensures minority class performance has equal weight in the final score.

Metric definitions (per-class, then averaged):
-----------------------------------------------
For each class c, treating it as a binary "c vs not-c" problem:
    TP_c = correctly predicted as class c
    FP_c = incorrectly predicted as class c (actually another class)
    TN_c = correctly NOT predicted as class c
    FN_c = incorrectly NOT predicted as class c (actually class c, predicted other)

    Precision_c  = TP_c / (TP_c + FP_c)     — "of all predicted c, how many are truly c?"
    Recall_c     = TP_c / (TP_c + FN_c)     — "of all truly c, how many did we find?"
    Specificity_c = TN_c / (TN_c + FP_c)   — "of all NOT-c, how many correctly rejected?"
    F1_c = 2 × Precision_c × Recall_c / (Precision_c + Recall_c)

G-mean (geometric mean of per-class recalls):
    G-mean = (Recall_0 × Recall_1 × ... × Recall_N) ^ (1/N)
    If ANY class has recall=0, G-mean=0 — this strongly penalizes
    models that completely ignore a minority class.

AUC (Area Under ROC Curve):
    Uses one-vs-rest (OvR): compute AUC for each class independently
    (class c vs all others), then macro-average.
    Requires PROBABILITY outputs (softmax), not just hard predictions.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

print("metrics.py loaded")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """Compute macro-averaged classification metrics.
    
    This is the 511J generalization of 427J's calculate_binary_metrics().
    
    427J approach (binary):
        TP = (preds==1) & (labels==1)  →  precision = TP/(TP+FP)  etc.
        
    511J approach (multi-class):
        Build confusion matrix → extract per-class TP/FP/TN/FN
        → compute per-class metrics → macro-average
    
    Args:
        y_true: True labels, shape [N]. Values are original class labels
                (e.g., [0, 2, 3] not [0, 1, 2])
        y_pred: Predicted labels, shape [N]. Same label space as y_true.
        y_prob: Predicted probabilities, shape [N, num_classes].
                Row i = softmax output for sample i.
                Column order matches sorted unique classes in y_true.
        num_classes: Number of distinct classes (e.g., 3)
    
    Returns:
        Dictionary with 7+ metrics:
            accuracy          — overall correct / total
            macro_f2          — macro-averaged F2 (recall-weighted)
            macro_precision   — macro-averaged precision
            macro_recall      — macro-averaged recall (= sensitivity)
            macro_specificity — macro-averaged specificity
            macro_gmean       — geometric mean of per-class recalls
            macro_auc         — macro-averaged AUC (one-vs-rest)
    
    Example:
        >>> y_true = np.array([0, 0, 0, 2, 2, 3, 3, 3])
        >>> y_pred = np.array([0, 0, 2, 2, 3, 3, 0, 3])
        >>> y_prob = np.random.rand(8, 3)  # softmax probabilities
        >>> y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        >>> m = compute_metrics(y_true, y_pred, y_prob, num_classes=3)
        >>> print(f"Macro F2: {m['macro_f2']:.3f}")
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, fbeta_score, precision_score, recall_score,
        confusion_matrix, roc_auc_score
    )
    
    # ── 1. Overall accuracy ─────────────────────────────────────────────
    # Simple: correct predictions / total predictions
    accuracy = accuracy_score(y_true, y_pred)
    
    # ── 2. Macro-averaged F1, Precision, Recall via sklearn ─────────────
    # sklearn's average='macro' does exactly what we need:
    #   compute metric per class, then unweighted mean
    #
    # zero_division=0: If a class has no predictions (precision denominator=0)
    # or no true samples (recall denominator=0), treat as 0 instead of warning.
    # This can happen with very small folds or poorly-performing models.
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # Macro F2 (β=2): weights recall 4× more than precision.
    # Aligned with the imbalanced-classification thesis that recall on
    # minority classes is the metric of interest.
    macro_f2 = fbeta_score(y_true, y_pred, beta=2.0, average='macro', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # ── 3. Macro-averaged Specificity (from confusion matrix) ───────────
    #
    # sklearn doesn't have a specificity function, so we compute it manually.
    #
    # How confusion matrix → per-class specificity works:
    #
    # For 3-class confusion matrix C:
    #              Predicted
    #              C0   C1   C2
    #   True C0 [[ 90    5    5 ]    ← row 0
    #   True C1 [  3   40    7 ]    ← row 1
    #   True C2 [  2    8   40 ]]   ← row 2
    #
    # For class c=1:
    #   TP_1 = C[1,1] = 40          (predicted 1, actually 1)
    #   FN_1 = sum(row 1) - C[1,1] = 50 - 40 = 10  (actually 1, predicted other)
    #   FP_1 = sum(col 1) - C[1,1] = 53 - 40 = 13  (predicted 1, actually other)
    #   TN_1 = total - TP - FN - FP = 200 - 40 - 10 - 13 = 137  (not-1, predicted not-1)
    #   Specificity_1 = TN_1 / (TN_1 + FP_1) = 137/150 = 0.913
    #
    # Then: macro_specificity = mean(Spec_0, Spec_1, Spec_2)
    
    # Get sorted unique labels for consistent confusion matrix ordering
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    total = cm.sum()                    # Total number of samples
    per_class_specificity = []          # Will hold specificity for each class
    per_class_recall_for_gmean = []     # Will hold recall for each class (for G-mean)
    
    epsilon = 1e-9  # Small constant to prevent division by zero (same as 427J)
    
    for i in range(len(labels)):
        tp_i = cm[i, i]                    # Diagonal: correct predictions for class i
        fn_i = cm[i, :].sum() - tp_i       # Row sum minus diagonal: missed class i
        fp_i = cm[:, i].sum() - tp_i       # Column sum minus diagonal: false alarms for class i
        tn_i = total - tp_i - fn_i - fp_i  # Everything else
        
        # Specificity: "of all samples NOT in class i, how many correctly classified as not-i?"
        spec_i = tn_i / (tn_i + fp_i + epsilon)
        per_class_specificity.append(spec_i)
        
        # Recall for G-mean computation
        recall_i = tp_i / (tp_i + fn_i + epsilon)
        per_class_recall_for_gmean.append(recall_i)
    
    macro_specificity = float(np.mean(per_class_specificity))
    
    # ── 4. G-mean (geometric mean of per-class recalls) ─────────────────
    #
    # 427J (binary): G-mean = √(recall × specificity)
    #   This is the geometric mean of 2 values: sensitivity and specificity.
    #   For binary case, recall_class0 = specificity and recall_class1 = sensitivity,
    #   so √(recall_0 × recall_1) = √(specificity × sensitivity).
    #   Same thing either way!
    #
    # 511J (multi-class): G-mean = (recall_0 × recall_1 × ... × recall_N) ^ (1/N)
    #   The natural N-class generalization.
    #   Key property: if ANY class recall = 0, G-mean = 0.
    #   This strongly penalizes models that completely ignore a minority class.
    #
    # Implementation note: We use np.prod() ^ (1/N) instead of scipy.stats.gmean
    # to keep dependencies minimal and make the computation transparent.
    
    recalls = np.array(per_class_recall_for_gmean)
    if np.any(recalls <= 0):
        # If any class has zero recall, G-mean is 0
        # (can't take root of product with zero)
        macro_gmean = 0.0
    else:
        # (r0 × r1 × r2) ^ (1/3) for 3 classes
        macro_gmean = float(np.prod(recalls) ** (1.0 / len(recalls)))
    
    # ── 5. Macro-averaged AUC (one-vs-rest) ─────────────────────────────
    #
    # 427J (binary): roc_auc_score(labels, preds) — used hard predictions
    # 511J (multi-class): roc_auc_score(y_true, y_prob, multi_class='ovr')
    #   - Uses PROBABILITIES (softmax output), not hard predictions
    #   - 'ovr' = One-vs-Rest: for each class c, compute AUC of
    #     "probability of class c" vs "is actually class c"
    #   - average='macro': unweighted mean of per-class AUCs
    #
    # Why probabilities instead of hard predictions?
    #   AUC measures ranking quality across ALL thresholds.
    #   Hard predictions only tell you performance at ONE threshold.
    #   Using probabilities gives a more informative metric.
    
    try:
        macro_auc = roc_auc_score(
            y_true, y_prob,
            multi_class='ovr',   # One-vs-Rest decomposition
            average='macro',     # Unweighted mean across classes
            labels=labels        # Ensure consistent label ordering
        )
    except ValueError:
        # roc_auc_score fails if a class has no samples in y_true
        # (can happen with very small folds)
        macro_auc = 0.5  # Fallback to random-chance AUC (same as 427J)
    
    return {
        'macro_gmean': macro_gmean,
        'macro_f1': float(macro_f1),
        'macro_f2': float(macro_f2),
        'macro_auc': float(macro_auc),
        'macro_recall': float(macro_recall),
        'macro_precision': float(macro_precision),
        'macro_specificity': macro_specificity,
        'accuracy': float(accuracy),
    }


def compare_folds(fold_results: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """Compute mean ± std for each metric across k folds.
    
    After running k-fold CV, each fold produces a metrics dictionary.
    This function summarizes them into (mean, std) pairs.
    
    Why mean ± std?
        - Mean: Best estimate of true model performance
        - Std: How much performance varies across folds
        - Low std = model is stable/reliable
        - High std = model performance is sensitive to data split
    
    Args:
        fold_results: List of k metric dictionaries, one per fold.
                      Each dict has the same keys (from compute_metrics).
    
    Returns:
        Dictionary mapping metric_name -> (mean, std)
    
    Example:
        >>> fold_results = [
        ...     {'macro_f1': 0.85, 'accuracy': 0.88, 'macro_auc': 0.92},
        ...     {'macro_f1': 0.87, 'accuracy': 0.89, 'macro_auc': 0.93},
        ...     {'macro_f1': 0.84, 'accuracy': 0.87, 'macro_auc': 0.91},
        ... ]
        >>> summary = compare_folds(fold_results)
        >>> summary['macro_f1']
        (0.8533, 0.0125)
        >>> # Interpretation: F1 = 0.853 ± 0.012 across 3 folds
    
    How it works:
        1. Collect all unique metric keys from the first fold's dict
        2. For each metric, gather all k values into a list
        3. Compute np.mean() and np.std() of that list
        4. Return as (mean, std) tuples
    """
    if not fold_results:
        return {}
    
    # Get metric names from the first fold (all folds have same keys)
    metric_names = fold_results[0].keys()
    
    summary = {}
    for metric in metric_names:
        # Gather this metric's value from each fold
        # e.g., [0.85, 0.87, 0.84, 0.86, 0.83] for 5-fold CV
        values = [fold[metric] for fold in fold_results]

        # Skip non-numeric keys (e.g. Ab-11a's `fold_history` dict).
        if any(not isinstance(v, (int, float, np.integer, np.floating))
               for v in values):
            continue

        # np.std() uses ddof=0 by default (population std)
        # This is standard for k-fold CV reporting
        summary[metric] = (float(np.mean(values)), float(np.std(values)))
    
    return summary


def format_results_table(
    all_results: Dict[str, Dict],
    include_test: bool = True
) -> pd.DataFrame:
    """Format ablation study results as a comparison table.
    
    Creates a pandas DataFrame for easy display in Jupyter notebooks.
    
    Table structure:
        Rows: One per ablation study (Ab-1, Ab-1a, Ab-4, Ab-5, Ab-6)
        Columns: One per metric, showing "mean ± std" for CV results
                 and raw value for test results
    
    Args:
        all_results: Nested dictionary structured as:
            {
                'Ab-1': {
                    'cv_summary': {  # Output of compare_folds()
                        'macro_f1': (0.75, 0.02),
                        'accuracy': (0.85, 0.01),
                        ...
                    },
                    'test_metrics': {  # Output of compute_metrics() on test set
                        'macro_f1': 0.74,
                        'accuracy': 0.84,
                        ...
                    }
                },
                'Ab-4': { ... },
                ...
            }
        include_test: If True, add test set columns alongside CV columns.
    
    Returns:
        DataFrame with formatted strings like "0.750 ± 0.020"
    
    Example output:
                    | F1 (CV)       | F1 (Test) | AUC (CV)      | AUC (Test) |
        Ab-1        | 0.750 ± 0.020 | 0.740     | 0.820 ± 0.015 | 0.810      |
        Ab-4        | 0.830 ± 0.010 | 0.825     | 0.910 ± 0.008 | 0.905      |
    
    How it works:
        1. Iterate over each ablation study in all_results
        2. For each metric in cv_summary, format as "mean ± std" (3 decimal places)
        3. Optionally add test_metrics columns with raw values
        4. Build a DataFrame with ablation names as the index
    """
    if not all_results:
        return pd.DataFrame()
    
    # Metric display names — shorter labels for table headers
    # Maps internal key → display column name
    metric_display = {
        'accuracy': 'Acc',
        'macro_f1': 'F1',
        'macro_precision': 'Prec',
        'macro_recall': 'Recall',
        'macro_specificity': 'Spec',
        'macro_gmean': 'G-mean',
        'macro_auc': 'AUC',
    }
    
    rows = []
    for ablation_name, result in all_results.items():
        row = {}
        cv_summary = result.get('cv_summary', {})
        test_metrics = result.get('test_metrics', {})
        
        for metric_key, display_name in metric_display.items():
            # ── CV column: "mean ± std" format ──────────────────────────
            if metric_key in cv_summary:
                mean, std = cv_summary[metric_key]
                row[f'{display_name} (CV)'] = f'{mean:.3f} ± {std:.3f}'
            
            # ── Test column: raw value ──────────────────────────────────
            if include_test and metric_key in test_metrics:
                row[f'{display_name} (Test)'] = f'{test_metrics[metric_key]:.3f}'
        
        rows.append(row)
    
    # Create DataFrame with ablation names as index
    df = pd.DataFrame(rows, index=list(all_results.keys()))
    df.index.name = 'Ablation'
    
    return df


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: List[int]
) -> Dict[int, Dict[str, float]]:
    """Compute per-class F1, precision, recall for detailed analysis.
    
    While macro-averaged metrics give a single summary number,
    sometimes you need to see which SPECIFIC classes are struggling.
    
    For example, if macro_f2 = 0.70, is that because:
    (a) All classes have F1 ≈ 0.70?  (uniform difficulty)
    (b) Majority class F1=0.95, minority classes F1=0.30?  (minority failure)
    
    This function breaks it down per class to answer that question.
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        class_labels: List of class labels to compute metrics for
                      (e.g., [0, 2, 3] for Fashion-MNIST)
    
    Returns:
        Dictionary mapping class_id -> {'f1': ..., 'precision': ..., 'recall': ...}
    
    Example:
        >>> per_class = compute_per_class_metrics(y_true, y_pred, [0, 2, 3])
        >>> for cls, m in per_class.items():
        ...     print(f"Class {cls}: F1={m['f1']:.3f}, Prec={m['precision']:.3f}, "
        ...           f"Recall={m['recall']:.3f}")
        Class 0: F1=0.950, Prec=0.940, Recall=0.960
        Class 2: F1=0.700, Prec=0.720, Recall=0.680
        Class 3: F1=0.680, Prec=0.650, Recall=0.710
    
    How it works:
        Uses sklearn's per-class classification_report logic:
        For each class c, treats it as binary "c vs not-c":
          - TP = predicted c AND actually c
          - FP = predicted c AND actually NOT c
          - FN = predicted NOT c AND actually c
        Then precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = harmonic mean
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    # precision_recall_fscore_support with average=None returns arrays
    # with one value per class, in the order of `labels` parameter
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=class_labels,  # Compute for these specific classes
        average=None,         # Per-class (not averaged)
        zero_division=0       # Handle classes with no predictions
    )
    # precision = [prec_class0, prec_class2, prec_class3]
    # recall    = [rec_class0,  rec_class2,  rec_class3]
    # f1        = [f1_class0,   f1_class2,   f1_class3]
    # support   = [count_class0, count_class2, count_class3]  (# true samples)
    
    result = {}
    for idx, cls in enumerate(class_labels):
        result[cls] = {
            'f1': float(f1[idx]),
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'support': int(support[idx]),  # Number of true samples for this class
        }
    
    return result
