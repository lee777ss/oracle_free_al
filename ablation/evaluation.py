"""
Held-out test set evaluation and result comparison.

Evaluates trained classifiers on the imbalanced Fashion-MNIST test set
and formats comparison tables across ablation studies.

This module is Step 8 of the 511J ablation pipeline. It runs AFTER
k-fold CV training (Step 9) is complete and produces the final
test-set numbers and comparison tables used in the paper.

427J did evaluation inline in the notebook with threshold tuning.
511J eliminates threshold tuning (uses argmax) and centralizes
all evaluation logic here for reproducibility.
"""

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .metrics import compute_metrics, format_results_table
from .metrics_binary import compute_metrics_binary
from .models import CNNClassifier, DeviceDataLoader, to_device
print("evaluation.py loaded")


def evaluate_on_test(
    model,
    test_loader: DataLoader,
    config,
    device: str
) -> Dict[str, float]:
    """Evaluate single model on held-out test set.
    
    No threshold tuning — uses argmax for multi-class predictions.
    This eliminates the overfitting source from 427J.
    
    Args:
        model: Trained CNNClassifier
        test_loader: Test data loader
        config: AblationConfig instance
        device: Device for inference
    
    Returns:
        metrics: Dictionary of test metrics
    
    Example:
        >>> test_metrics = evaluate_on_test(
        ...     trained_model, test_loader, config, 'cuda'
        ... )
        >>> print(f"Test Macro F1: {test_metrics['macro_f2']:.3f}")
        Test Macro F1: 0.834
    
    How it works:
        1. Put model in eval mode (disable dropout, use running batch norm stats)
        2. Loop over test_loader, collecting logits and true labels
        3. Convert logits → softmax probabilities → argmax predictions
        4. Compare pred_indices directly against labels (both in {0,1,2})
        5. Call compute_metrics() with contiguous labels and predictions
    
    Label space convention:
        All labels are remapped to contiguous {0, 1, ..., N-1} early in
        the pipeline (notebook Step 17b), and config.selected_classes is
        updated to match. Both model predictions (argmax) and DataLoader
        labels are in the same contiguous space — no mapping needed.
        Original labels are only stored in inverse_label_map for display.
    
    Key difference from 427J:
        427J used threshold tuning on the test set, which leaks information.
        511J uses simple argmax — no tuning, no data leakage risk.
    """
    model.eval()    # Disable dropout, use batch norm running averages
    
    all_logits = []   # Raw model outputs (before softmax)
    all_labels = []   # True labels from the test set
    
    with torch.no_grad():   # No gradient computation needed for evaluation
        for images, labels in test_loader:
            # Move data to device if not already there
            # (DeviceDataLoader handles this, but raw DataLoader may not)
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass — get raw logits
            outputs = model(images)   # [B, num_classes]
            
            all_logits.append(outputs.cpu())
            all_labels.append(labels.cpu())
    
    # ── Concatenate all batch outputs ───────────────────────────────
    all_logits = torch.cat(all_logits, dim=0)   # [N_test, num_classes]
    all_labels = torch.cat(all_labels, dim=0)   # [N_test]
    
    # ── Convert logits → probabilities → predictions ────────────────
    #
    # Softmax: logits → probabilities that sum to 1 per sample
    #   e.g., [2.1, -0.3, 0.5] → [0.72, 0.07, 0.21]
    #
    # Argmax: pick the class index with highest probability
    #   e.g., [0.72, 0.07, 0.21] → index 0
    probs = F.softmax(all_logits, dim=1)        # [N_test, num_classes]
    pred_indices = all_logits.argmax(dim=1)      # [N_test]
    
    # ── Compute metrics ─────────────────────────────────────────────
    # Both all_labels (from test_loader) and pred_indices (from argmax)
    # are in remapped contiguous space {0, 1, 2}. No mapping needed.
    metrics = compute_metrics(
        y_true=all_labels.numpy(),        # [N_test] remapped labels {0,1,2}
        y_pred=pred_indices.numpy(),      # [N_test] predictions {0,1,2}
        y_prob=probs.numpy(),             # [N_test, num_classes] softmax probs
        num_classes=config.num_classes    # 3 for 511J
    )
    
    # Binary metrics (515J)
    binary_metrics = compute_metrics_binary(
        y_true=all_labels.numpy(),
        y_pred=pred_indices.numpy(),
        y_prob=probs.numpy(),
        num_classes=config.num_classes,
    )
    metrics.update(binary_metrics)
    
    return metrics


def evaluate_fold_ensemble(
    fold_models: List,
    test_loader: DataLoader,
    config,
    device: str,
    method: str = "majority_vote"
) -> Dict[str, float]:
    """Evaluate ensemble of k-fold models on test set.
    
    Combines predictions from all fold models using ensemble method.
    
    Args:
        fold_models: List of trained CNNClassifier models (one per fold)
        test_loader: Test data loader
        config: AblationConfig instance
        device: Device for inference
        method: Ensemble method ('majority_vote' or 'avg_prob')
    
    Returns:
        metrics: Dictionary of ensemble test metrics
    
    Example:
        >>> # Evaluate ensemble of 5 fold models
        >>> ensemble_metrics = evaluate_fold_ensemble(
        ...     all_fold_models, test_loader, config, 'cuda', method='majority_vote'
        ... )
        >>> print(f"Ensemble Test F1: {ensemble_metrics['macro_f2']:.3f}")
        Ensemble Test F1: 0.845
    
    How it works:
        1. Run each of the k fold models on the SAME test set
        2. Combine their predictions using the chosen ensemble method:
        
           'majority_vote':
             Each model votes for a class (argmax). The class with the
             most votes wins. Ties broken by lowest class index.
             Good when models are diverse (different folds see different data).
             
           'avg_prob':
             Average the softmax probabilities across all k models, then
             argmax the averaged probabilities. This is smoother than
             majority vote because it considers prediction confidence.
             Generally preferred when models produce well-calibrated probs.
        
        3. Compute metrics on the ensemble predictions vs true labels.
    
    Why ensemble?
        K-fold CV trains k separate models. Rather than picking just one
        (which wastes k-1 models), ensembling combines all of them.
        Ensembles typically outperform individual models because errors
        from different folds tend to cancel out ("wisdom of crowds").
    """
    if method not in ('majority_vote', 'avg_prob'):
        raise ValueError(
            f"Unknown ensemble method '{method}'. "
            f"Must be 'majority_vote' or 'avg_prob'."
        )
    
    # ── Collect predictions from all k models ───────────────────────
    #
    # all_model_probs[m] = [N_test, num_classes] softmax probs from model m
    # all_model_preds[m] = [N_test] argmax predictions from model m
    all_model_probs = []
    all_model_preds = []
    all_labels = None    # True labels — same for all models
    
    for model in fold_models:
        model.eval()
        
        batch_logits = []
        batch_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                batch_logits.append(outputs.cpu())
                batch_labels.append(labels.cpu())
        
        logits = torch.cat(batch_logits, dim=0)    # [N_test, num_classes]
        probs = F.softmax(logits, dim=1)            # [N_test, num_classes]
        preds = logits.argmax(dim=1)                # [N_test]
        
        all_model_probs.append(probs)
        all_model_preds.append(preds)
        
        # Store labels from the first model (all models see same test data)
        if all_labels is None:
            all_labels = torch.cat(batch_labels, dim=0)  # [N_test]
    
    # ── Combine predictions using ensemble method ───────────────────
    if method == 'majority_vote':
        # ── Majority Vote ───────────────────────────────────────────
        #
        # Stack all model predictions: [k, N_test]
        # For each test sample, count votes for each class.
        # The class with the most votes wins.
        #
        # Example with k=5, 3 classes, for one sample:
        #   Model votes: [0, 2, 0, 0, 2] → class 0 gets 3 votes → predict 0
        stacked_preds = torch.stack(all_model_preds, dim=0)  # [k, N_test]
        
        # Count votes per class for each sample
        # Use one-hot encoding trick: convert each prediction to one-hot,
        # sum across models, then argmax to find the winning class.
        n_test = stacked_preds.shape[1]
        num_classes = config.num_classes
        
        # vote_counts[i, c] = number of models that predicted class c for sample i
        vote_counts = torch.zeros(n_test, num_classes)
        for m in range(len(fold_models)):
            for c in range(num_classes):
                vote_counts[:, c] += (stacked_preds[m] == c).float()
        
        # Argmax to get the final ensemble prediction (ties: lowest index wins)
        ensemble_pred_indices = vote_counts.argmax(dim=1)   # [N_test]
        
        # For probability-based metrics (AUC), average the softmax probs
        # even though predictions come from voting
        ensemble_probs = torch.stack(all_model_probs, dim=0).mean(dim=0)
    
    else:  # method == 'avg_prob'
        # ── Average Probability ─────────────────────────────────────
        #
        # Average softmax probabilities across all k models:
        #   avg_prob[i, c] = mean of prob[m][i, c] for m in 0..k-1
        #
        # Then argmax the averaged probabilities.
        # This is smoother than majority vote because it considers
        # each model's confidence, not just its top prediction.
        #
        # Example with k=3 for one sample:
        #   Model 0: [0.8, 0.1, 0.1] → votes class 0 (confident)
        #   Model 1: [0.3, 0.4, 0.3] → votes class 1 (uncertain)
        #   Model 2: [0.7, 0.2, 0.1] → votes class 0 (confident)
        #   Average:  [0.6, 0.23, 0.17] → predict class 0
        #
        # In majority_vote: class 0 gets 2 votes, class 1 gets 1 → class 0
        # Same result here, but avg_prob better handles close calls.
        ensemble_probs = torch.stack(all_model_probs, dim=0).mean(dim=0)  # [N_test, C]
        ensemble_pred_indices = ensemble_probs.argmax(dim=1)               # [N_test]
    
    # ── Compute metrics ─────────────────────────────────────────────
    # Both all_labels (from test_loader) and ensemble predictions
    # are in remapped contiguous space {0, 1, 2}. No mapping needed.
    metrics = compute_metrics(
        y_true=all_labels.numpy(),
        y_pred=ensemble_pred_indices.numpy(),
        y_prob=ensemble_probs.numpy(),
        num_classes=config.num_classes
    )
    
    return metrics


def compare_classifiers(
    all_results: Dict[str, Dict],
    config
) -> pd.DataFrame:
    """Compare all ablation studies in formatted table.
    
    Creates comparison table with:
    - Rows: Ablation studies (Ab-1, Ab-1a, Ab-4, Ab-5, Ab-6)
    - Columns: CV metrics (mean±std) and test metrics side-by-side
    
    Args:
        all_results: Dictionary mapping ablation_id -> {
            'cv_summary': {metric: (mean, std)},
            'test_metrics': {metric: value}
        }
        config: AblationConfig instance
    
    Returns:
        comparison_df: Formatted comparison DataFrame
    
    Example:
        >>> comparison_df = compare_classifiers(all_results, config)
        >>> display(comparison_df)
        
        |       | CV Macro F1   | Test Macro F1 | CV Accuracy   | Test Accuracy |
        |-------|---------------|---------------|---------------|---------------|
        | Ab-1  | 0.752 ± 0.018 | 0.745         | 0.854 ± 0.012 | 0.849         |
        | Ab-1a | 0.831 ± 0.015 | 0.827         | 0.891 ± 0.010 | 0.886         |
        | Ab-4  | 0.847 ± 0.011 | 0.842         | 0.902 ± 0.008 | 0.898         |
        | Ab-5  | 0.853 ± 0.013 | 0.849         | 0.906 ± 0.009 | 0.902         |
        | Ab-6  | 0.859 ± 0.010 | 0.855         | 0.910 ± 0.007 | 0.907         |
    
    How it works:
        This is a thin wrapper around metrics.format_results_table().
        It delegates the actual formatting to that utility function,
        which handles the "mean ± std" formatting for CV results and
        raw value columns for test results.
        
        The separation exists because:
        - format_results_table() is a pure formatting utility (metrics.py)
        - compare_classifiers() is the user-facing API (evaluation.py)
          that could add additional logic (filtering, sorting, etc.)
    """
    # Delegate to format_results_table in metrics.py, which builds
    # a DataFrame with "metric (CV)" and "metric (Test)" columns.
    #
    # all_results is expected to have the structure:
    #   {
    #     'Ab-1':  {'cv_summary': {metric: (mean, std)}, 'test_metrics': {metric: val}},
    #     'Ab-1a': {'cv_summary': {metric: (mean, std)}, 'test_metrics': {metric: val}},
    #     ...
    #   }
    #
    # format_results_table() iterates over each ablation, formats CV metrics
    # as "0.750 ± 0.020" strings, and test metrics as "0.740" strings.
    return format_results_table(all_results, include_test=True)


def get_best_fold_model(
    fold_models: List,
    fold_metrics: List[Dict[str, float]],
    metric: str = 'macro_f2'
):
    """Select best model from k-fold based on validation metric.
    
    Args:
        fold_models: List of trained models
        fold_metrics: List of metric dictionaries per fold
        metric: Metric to use for selection (default: 'macro_f2')
    
    Returns:
        best_model: Model with highest validation metric
        best_fold_idx: Index of best fold
        best_metric_value: Best metric value
    
    Example:
        >>> best_model, idx, val = get_best_fold_model(
        ...     fold_models, fold_metrics, metric='macro_f2'
        ... )
        >>> print(f"Best fold: {idx}, Val F1: {val:.3f}")
        Best fold: 2, Val F1: 0.867
    
    How it works:
        1. Extract the target metric value from each fold's metrics dict
        2. Find the fold index with the HIGHEST value (argmax)
        3. Return that fold's model, index, and metric value
    
    Why select best fold?
        When you need a SINGLE model (not an ensemble), pick the fold
        that performed best on validation. This is common for deployment
        or for evaluating single-model test performance.
    """
    if not fold_models or not fold_metrics:
        raise ValueError("fold_models and fold_metrics must be non-empty")
    
    if len(fold_models) != len(fold_metrics):
        raise ValueError(
            f"Mismatch: {len(fold_models)} models but {len(fold_metrics)} metrics dicts"
        )
    
    # Extract the target metric from each fold
    # e.g., for metric='macro_f2': [0.85, 0.87, 0.84, 0.86, 0.83]
    metric_values = [fold[metric] for fold in fold_metrics]
    
    # Find the fold with the highest metric value
    best_fold_idx = int(np.argmax(metric_values))
    best_metric_value = metric_values[best_fold_idx]
    best_model = fold_models[best_fold_idx]
    
    return best_model, best_fold_idx, best_metric_value
