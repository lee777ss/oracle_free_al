"""
Ablation study module for multi-class imbalanced classification with BAGAN.

This module provides a complete framework for conducting stratified k-fold
cross-validation experiments comparing different data augmentation strategies.
"""

from .config import AblationConfig
from .models import CNNClassifier, DeviceDataLoader, EarlyStopping, to_device
from .data import (
    filter_by_classes,
    create_imbalanced_pool,
    build_test_set,
    get_class_distribution,
    split_by_class,
    dataset_to_tensors,
    print_class_distribution
)
from .metrics import compute_metrics, compare_folds, format_results_table
from .metrics_binary import (
    compute_metrics_binary,
    compare_folds_binary,
    format_results_table_binary,
    find_best_threshold,
    pooled_threshold_with_holdout,
)
from .augmentation import (
    oversample_minority_classes,
    generate_bagan_augmented_dataset,
)
from .trainer import ClassifierTrainer
from .evaluation import evaluate_on_test, evaluate_fold_ensemble, compare_classifiers
from .kfold import run_kfold_ablation
from .calibration import fit_temperature, apply_temperature
from .decision_rules import (
    logit_adjustment_offset,
    predict_with_logit_adjustment,
    predict_with_threshold,
)
from .shift_metrics import compute_fid
from .fid import build_real_reference_cache, fid_per_class, ae_fid_per_class
from .gan_backend import get_gan_modules, GANModules
from .reproducibility import (
    set_seed,
    seed_worker,
    make_loader_generator,
    tensor_hash,
    mode_seed,
    collect_env_info,
)

__all__ = [
    # Config
    'AblationConfig',
    # Models
    'CNNClassifier',
    'DeviceDataLoader',
    'EarlyStopping',
    'to_device',
    # Data
    'filter_by_classes',
    'create_imbalanced_pool',
    'build_test_set',
    'get_class_distribution',
    'split_by_class',
    'dataset_to_tensors',
    'print_class_distribution',
    # Metrics
    'compute_metrics',
    'compare_folds',
    'format_results_table',
    # Binary metrics (515J)
    'compute_metrics_binary',
    'compare_folds_binary',
    'format_results_table_binary',
    # Threshold tuning (Topic 7 / Step D of 02a plan)
    'find_best_threshold',
    'pooled_threshold_with_holdout',
    # Augmentation
    'oversample_minority_classes',
    'generate_bagan_augmented_dataset',
    # Training
    'ClassifierTrainer',
    # Evaluation
    'evaluate_on_test',
    'evaluate_fold_ensemble',
    'compare_classifiers',

    # K-fold orchestrator
    'run_kfold_ablation',
    # Calibration (Topic 7)
    'fit_temperature',
    'apply_temperature',
    # Decision rules (Topic 7 / Step E of 02a plan)
    'logit_adjustment_offset',
    'predict_with_logit_adjustment',
    'predict_with_threshold',
    # Shift / fake-quality diagnostics (Phase 1)
    'compute_fid',
    'build_real_reference_cache',
    'fid_per_class',
    'ae_fid_per_class',
    # GAN backend selector (110J)
    'get_gan_modules',
    'GANModules',
    # Reproducibility (119J)
    'set_seed',
    'seed_worker',
    'make_loader_generator',
    'tensor_hash',
    'collect_env_info',
]

__version__ = '1.0.0'
__author__ = 'GAAL Research'
