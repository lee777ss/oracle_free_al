"""
K-fold cross-validation orchestrator for ablation studies.

Main entry point that coordinates stratified k-fold CV, training,
and result aggregation for one ablation mode.

This is Step 9 of the 511J pipeline ->it ties together all previous
modules (config, data, models, metrics, augmentation, trainer) into
a single function call per ablation study.

How it works at a high level:
--
1. StratifiedKFold splits pool_data into k train/val folds
   ("stratified" = each fold preserves the class distribution)
2. For each fold:
   a. Extract train/val tensors from indices
   b. Create a ClassifierTrainer with the specified mode
   c. Call trainer.train_fold() ->(model, metrics, val_logits, val_labels)
3. Collect all k models + metrics
4. Aggregate metrics via compare_folds() ->mean +/- std

427J didn't have this centralized orchestrator ->each ablation study
had its own training loop scattered across notebook cells. 511J
consolidates everything here so running an ablation is a single call.
"""

from typing import Dict, List, Optional, Tuple
import contextlib
import copy
import io
import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import numpy as np

from .trainer import ClassifierTrainer, VALID_MODES, BAGAN_MODES, _ab8c_seed_latents
from .selection import select_indices, describe as describe_selection
from .models import CNNClassifier, DeviceDataLoader, EarlyStopping
from .metrics import compare_folds, compute_metrics
from .metrics_binary import compute_metrics_binary, compare_folds_binary
from .augmentation import _compute_target_counts, generate_bagan_augmented_dataset

# BAGAN imports ->used by _train_bagan_for_fold() to train a fresh
# BAGAN pipeline (AE ->latent fit ->GAN) on each fold's training data.
# These are the vanilla-BAGAN names; when ``bagan_config`` is a
# ``bagan_gp.BAGANGPConfig`` instance the same names get re-bound to the
# BAGAN-GP package classes inside ``_train_bagan_for_fold`` via
# :func:`ablation.gan_backend.get_gan_modules` (110J backend toggle).
from bagan import (
    BAGANAutoencoder, BAGANGenerator, BAGANDiscriminator,
    AutoencoderTrainer, GANTrainer, ClassConditionalLatentGenerator,
    WeightTransfer,
)

from .gan_backend import get_gan_modules

print("kfold.py loaded")


def run_kfold_ablation(
    config,
    mode: str,
    pool_data: torch.Tensor,
    pool_labels: torch.Tensor,
    bagan_config=None,
    pool_indices=None,
    device: str = 'cuda',
    verbose: Optional[bool] = None,
    show_bagan_plots: Optional[bool] = None,
) -> Dict:
    """Run stratified k-fold cross-validation for one ablation study.
    
    This is the main entry point for the ablation pipeline. You call it
    once per ablation mode (ab1, ab1a, ab4, ab5, ab6, ab7, ab8, ab9) and it handles
    everything: splitting data, training k models, collecting metrics.
    
    Complete k-fold CV workflow:
    
        Step 1 ->VALIDATE INPUTS
            Check mode is valid, bagan_config provided if needed.
        
        Step 2 ->CREATE STRATIFIED SPLITS
            StratifiedKFold(n_folds) generates k pairs of (train_idx, val_idx).
            "Stratified" means each fold preserves the original class
            proportions. Without stratification, a fold could randomly
            get zero minority samples ->meaningless training.
            
            Example with n_folds=5, 6600 samples:
              Fold 0: train=5280, val=1320
              Fold 1: train=5280, val=1320
              ...
              Each val fold has ~90.9% class 0, ~4.5% class 2, ~4.5% class 3
        
        Step 3 ->TRAIN EACH FOLD
            For each of k folds:
              a. Slice pool_data/pool_labels by train_idx and val_idx
              b. Create a fresh ClassifierTrainer(config, mode, ...)
              c. trainer.train_fold(train_data, train_labels, val_data, val_labels)
                 ->returns (best_model, validation_metrics,
                             best_val_logits, best_val_labels)
              d. Store model + metrics + raw logits/labels
              e. Print progress (fold N/K, key metrics)
        
        Step 4 ->AGGREGATE RESULTS
            compare_folds(fold_metrics_list) ->{metric: (mean, std)}
            This gives the final CV summary: e.g., "macro_f1: 0.847 +/- 0.011"
    
    Args:
        config: AblationConfig instance ->contains n_folds, random_seed,
                and all training hyperparameters.
        mode: Ablation mode identifier ->one of
              'ab1','ab1a','ab1b','ab4','ab5','ab6','ab7','ab8','ab9'.
              Determines which augmentation strategy is used:
                'ab1'  ->No augmentation (imbalanced baseline)
                'ab1a' ->Classic oversampling (duplicate real minority samples)
                'ab1b' ->Pre-balanced real data (caller provides balanced pool)
                'ab4'  ->BAGAN augmentation (fixed synthetic dataset)
                'ab5'  ->Dynamic BAGAN (fresh synthetic samples per epoch)
                'ab6'  ->BAGAN + Quality Gate (per-batch discriminator filtering)
                'ab7'  ->BAGAN + Margin Sampling (per-batch classifier margin filtering)
                'ab8'  ->Iterative BAGAN acquisition with margin sampling (GAAL)
                'ab9'  ->Pool-drain active learning with patience-based stopping
        pool_data: Full training pool images [N, 1, 64, 64].
                   This is the entire imbalanced training set (before any
                   train/val splitting). StratifiedKFold splits this.
        pool_labels: Full training pool labels [N].
                     Original class labels (e.g., values in {0, 2, 3}).
        bagan_config: BAGANConfig instance ->required for BAGAN modes
                      (ab4, ab5, ab6, ab7, ab8, ab9). Contains BAGAN architecture
                      and training hyperparameters (ae_epochs, gan_epochs,
                      latent_dim, etc.) used by _train_bagan_for_fold() to
                      train a fresh BAGAN on each fold's training data.
                      None for non-BAGAN modes (ab1, ab1a, ab1b).
        pool_indices: Original indices of pool samples in Fashion-MNIST
                      train set. Required for 'ab1b' mode to compute the
                      disjoint reserve pool on-the-fly. None for other modes.
        device: Device for training ('cuda' or 'cpu').
    
    Returns:
        Dictionary with keys:
            'fold_metrics': List[Dict[str, float]]
                Per-fold validation metrics. Length = n_folds.
                Each dict has 7 metrics from compute_metrics():
                {accuracy, macro_f1, macro_precision, macro_recall,
                 macro_specificity, macro_gmean, macro_auc}
                Plus 'stopping_epoch' (int) ->the epoch where
                EarlyStopping restored the best model.
                
            'fold_models': List[CNNClassifier]
                Trained model for each fold. Length = n_folds.
                Each model has best weights (restored by EarlyStopping).
            
            'fold_val_logits': List[np.ndarray]
                Raw pre-softmax logits at the best epoch on each fold's
                validation split. Length = n_folds. Element k has shape
                [N_val_k, num_classes], dtype float32. Consumed by Topic 7
                (temperature scaling, prior-shift correction, PR-curve
                threshold tuning) ->NOT by standard metric aggregation.
            
            'fold_val_labels': List[np.ndarray]
                Remapped contiguous labels {0..num_classes-1} paired with
                fold_val_logits. Length = n_folds. Element k has shape
                [N_val_k], dtype int64.
                
            'cv_summary': Dict[str, Tuple[float, float]]
                Aggregated metrics: {metric_name: (mean, std)}.
                Computed by compare_folds() from metrics.py.
            
            'cv_summary_binary': Dict[str, Tuple[float, float]]
                Binary-metric aggregates (binary_f1, etc.).
            
            'avg_stopping_epoch': int
                Ceiling of mean stopping_epoch across folds.
                Used as n_epochs for retrain_on_full_pool().
    
    Example:
        >>> results = run_kfold_ablation(
        ...     config=config,
        ...     mode='ab4',
        ...     pool_data=pool_data,
        ...     pool_labels=pool_labels,
        ...     bagan_config=bagan_config,
        ...     device='cuda'
        ... )
        >>> print("="*60)
        >>> print(f"Ab-4 Results (5-fold CV)")
        >>> print("="*60)
        >>> for metric, (mean, std) in results['cv_summary'].items():
        ...     print(f"{metric:20s}: {mean:.4f} +/- {std:.4f}")
        ============================================================
        Ab-4 Results (5-fold CV)
        ============================================================
        macro_f1            : 0.8472 +/- 0.0113
        macro_precision     : 0.8519 +/- 0.0098
        macro_recall        : 0.8431 +/- 0.0127
        accuracy            : 0.9021 +/- 0.0082
        >>> print(f"Avg stopping epoch: {results['avg_stopping_epoch']}")
        Avg stopping epoch: 38
    """
    # -> Step 1: Validate inputs --
    #
    # _validate_mode() checks:
    #   - Is mode one of the 5 valid ablation identifiers?
    #   - Is bagan_config provided when required?
    # This catches configuration errors early, before any training starts.
    # The ClassifierTrainer also validates in __init__, but checking here
    # gives clearer error messages with the kfold context.
    _validate_mode(mode, bagan_config)

    if verbose is None:
        verbose = getattr(config, 'cv_verbose', True)
    if show_bagan_plots is None:
        show_bagan_plots = getattr(config, 'cv_show_bagan_plots', True)
    
    # -> Step 1b: Compute reserve pool for Ab-1b --
    #
    # Ab-1b augments each training fold with UNIQUE real images from a
    # disjoint reserve pool (Fashion-MNIST images not in pool_data).
    # We compute the reserve once here and pass it to each fold's trainer.
    reserve_data = None
    reserve_labels = None
    if mode == 'ab1b':
        if pool_indices is None:
            raise ValueError(
                "Mode 'ab1b' requires pool_indices to compute the "
                "disjoint reserve pool. Pass pool_indices=... to "
                "run_kfold_ablation()."
            )
        
        from .data import create_reserve_pool
        reserve_data, reserve_labels = create_reserve_pool(pool_indices)
        # Print reserve pool stats
        if verbose:
            for cls in sorted(torch.unique(reserve_labels).tolist()):
                n = (reserve_labels == cls).sum().item()
                print(f"    Reserve class {cls}: {n} images")
            print(f"    Reserve total: {len(reserve_data)} images")
    
    # -> Step 2: Create stratified k-fold splits --
    #
    # StratifiedKFold from sklearn guarantees that each fold preserves
    # the class distribution of the original dataset.
    #
    # WHY STRATIFIED?
    # With regular KFold on imbalanced data (90% class 0, 5% class 2, 5% class 3),
    # a random split could produce a fold with ZERO minority samples.
    # StratifiedKFold ensures each fold gets proportional minority samples.
    #
    # shuffle=True + random_state: randomize which samples go into which fold,
    # but reproducibly (same seed ->same splits every run).
    #
    # n_splits = config.n_folds (default: 5)
    # Each fold: (N - N/k) training samples + N/k validation samples
    skf = StratifiedKFold(
        n_splits=config.n_folds, # k = 5 by default
        shuffle=True,
        random_state=config.random_seed # random_seed = 42 by default
    )
    
    # Convert labels to numpy for sklearn (StratifiedKFold expects numpy)
    # pool_labels is a torch.Tensor; sklearn needs a numpy array.
    labels_np = pool_labels.numpy()
    
    # Storage for results across all folds
    fold_metrics_list = []    # List of metric dicts, one per fold
    fold_models_list = []     # List of trained CNNClassifier models
    fold_val_logits_list = [] # List of np.ndarray [N_val_k, C]  ->raw logits at best epoch
    fold_val_labels_list = [] # List of np.ndarray [N_val_k]     ->remapped {0..C-1} labels
    
    # -> Step 3: Train each fold --
    #
    # skf.split() yields (train_indices, val_indices) for each fold.
    # It takes (X, y) but only uses y for stratification (X can be anything
    # with the right length ->we pass labels_np for both since X isn't used).
    #
    # enumerate gives us fold_idx for progress printing.
    if verbose:
        print(f"\n{'='*60}")
        print(f"  {mode.upper()} - {config.n_folds}-fold Stratified CV")
        print(f"{'='*60}")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(labels_np, labels_np)):
        # -> 3a. Extract train/val tensors using fold indices --
        #
        # e.g.) Fold 0: Train indices (528,), Val indices (132,)
        # train_idx: numpy array of ~80% of indices (for 5-fold)
        # val_idx:   numpy array of ~20% of indices
        #
        # We index pool_data/pool_labels directly with these indices.
        # PyTorch supports numpy-array indexing on tensors.
        train_data = pool_data[train_idx]       # [N_train, 1, 64, 64]
        train_labels = pool_labels[train_idx]   # [N_train]
        val_data = pool_data[val_idx]           # [N_val, 1, 64, 64]
        val_labels = pool_labels[val_idx]       # [N_val]
        
        # Print fold split info
        if verbose:
            print(f"\n  Fold {fold_idx + 1}/{config.n_folds}")
            print(f"    Train: {len(train_data)} samples | Val: {len(val_data)} samples")
        
        # -> 3b. Train per-fold BAGAN (if needed) --
        #
        # For BAGAN modes, train a fresh BAGAN pipeline exclusively on
        # this fold's training data.  This is the core V1 fix: the BAGAN
        # never sees validation data, preventing data leakage.
        #
        # For non-BAGAN modes, the generator/latent/discriminator are None.
        if mode in BAGAN_MODES:
            fold_generator, fold_latent_gen, fold_discriminator = _train_bagan_for_fold(
                train_data, train_labels, bagan_config, device,
                verbose=verbose, show_bagan_plots=show_bagan_plots,
            )
        else:
            fold_generator = fold_latent_gen = fold_discriminator = None
        
        # -> 3c. Create fresh ClassifierTrainer --
        #
        # A NEW trainer is created each fold. This ensures:
        #   - Fresh model initialization (no weight leakage between folds)
        #   - Fresh EarlyStopping state
        #   - Fresh label mappings (though these are the same for all folds)
        #
        # The trainer's __init__ also validates mode + BAGAN dependencies,
        # providing a second layer of safety.
        #
        # IMPORTANT: fold-local BAGAN components are used ->NOT the
        # caller-provided ones.  Each fold's classifier trains only on
        # synthetic data from that fold's freshly-trained BAGAN.
        trainer = ClassifierTrainer(
            config=config,
            mode=mode,
            generator=fold_generator,
            latent_generator=fold_latent_gen,
            discriminator=fold_discriminator,
            reserve_data=reserve_data,
            reserve_labels=reserve_labels,
            device=device
        )
        # -> 3d. Train this fold --
        #
        # train_fold() runs the complete training pipeline:
        #   1. Augment/prepare training data (based on mode)
        #   2. Create model + optimizer + early stopping
        #   3. Training loop with validation
        #   4. Return (best_model, best_metrics, best_val_logits, best_val_labels)
        #
        # The model returned has its best weights (restored by EarlyStopping
        # to the epoch with best binary_f1).
        #
        # best_val_logits/best_val_labels are the raw pre-softmax logits +
        # remapped labels captured at the best epoch on this fold's
        # validation split. They are used downstream by Topic 7 (temperature
        # scaling, prior-shift correction, PR-curve threshold tuning) and
        # are NOT consumed by standard metric aggregation.
        model, metrics, val_logits, val_lbls = trainer.train_fold(
            train_data, train_labels,
            val_data, val_labels
        )
        
        # -> 3e. Store results --
        fold_models_list.append(model)
        fold_metrics_list.append(metrics) # binary metrics are included in this dict (e.g. 'binary_f1')
        fold_val_logits_list.append(val_logits)   # np.ndarray [N_val_k, C]
        fold_val_labels_list.append(val_lbls)     # np.ndarray [N_val_k]
        
        # -> 3f. Print progress --
        if verbose:
            _print_fold_progress(fold_idx, config.n_folds, mode, metrics)
    
    # -> Step 4: Aggregate results across folds --
    #
    # compare_folds() from metrics.py computes mean +/- std for each metric.
    # Input:  [{'macro_f1': 0.85, ...}, {'macro_f1': 0.87, ...}, ...]
    # Output: {'macro_f1': (0.86, 0.01), 'accuracy': (0.90, 0.005), ...}
    #
    # This gives the final CV summary that goes into the comparison table.
    cv_summary = compare_folds(fold_metrics_list)
    
    # Binary CV summary (515J): extract binary_ keys from fold metrics
    binary_fold_metrics = [
        {k: v for k, v in fm.items() if k.startswith('binary_')}
        for fm in fold_metrics_list
    ]
    cv_summary_binary = compare_folds_binary(binary_fold_metrics)
    
    # -> Print CV summary --
    # Display order: F1 (primary) first, then G-Mean, AUC, Precision,
    # Recall, Specificity, Accuracy.
    _cv_display = [
        ('macro_f1',          'F1-Score     '),
        ('macro_gmean',       'G-Mean       '),
        ('macro_auc',         'ROC-AUC      '),
        ('macro_precision',   'Precision    '),
        ('macro_recall',      'Recall       '),
        ('macro_specificity', 'Specificity  '),
        ('accuracy',          'Accuracy     '),
    ]
    if verbose:
        print(f"\n  {'-'*56}")
        print(f"  {mode.upper()} - CV Summary ({config.n_folds}-fold)")
        print(f"  {'-'*56}")
        for key, label in _cv_display:
            if key in cv_summary:
                mean, std = cv_summary[key]
                print(f"    {label}: {mean:.4f} +/- {std:.4f}")
        print(f"{'='*60}\n")
    
    # -> Step 5: Compute average stopping epoch across folds --
    #
    # Each fold records the epoch where EarlyStopping restored the best
    # model (stopping_epoch).  The ceiling of the mean across folds gives
    # a robust estimate of how many epochs are needed.  This value is
    # used as n_epochs for retrain_on_full_pool() (Step 6 / V2 fix).
    stopping_epochs = [fm['stopping_epoch'] for fm in fold_metrics_list]
    avg_stopping_epoch = math.ceil(sum(stopping_epochs) / len(stopping_epochs))
    if verbose:
        print(f"  Per-fold stopping epochs: {stopping_epochs}")
        print(f"  avg_stopping_epoch (ceil): {avg_stopping_epoch}")
    
    return {
        'fold_metrics': fold_metrics_list,    # List[Dict] ->per-fold raw metrics
        'fold_models': fold_models_list,      # List[CNNClassifier] ->trained models
        'fold_val_logits': fold_val_logits_list,  # List[np.ndarray [N_val_k, C]]
        'fold_val_labels': fold_val_labels_list,  # List[np.ndarray [N_val_k]] remapped {0..C-1}
        'cv_summary': cv_summary,             # Dict[str, (mean, std)] ->raw
        'cv_summary_binary': cv_summary_binary,   # Dict[str, (mean, std)] ->binary (515J)
        'avg_stopping_epoch': avg_stopping_epoch,  # int ->ceil(mean(stopping_epochs))
    }


def _train_bagan_for_fold(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    bagan_config,
    device: str,
    verbose: bool = True,
    show_bagan_plots: bool = True,
):
    """Train a complete BAGAN pipeline on one fold's training data.
    
    This is the core fix for the V1 data-leakage violation: each CV fold
    gets its own BAGAN trained exclusively on the fold's training split.
    The validation data is never seen by the generative model.
    
    Pipeline (mirrors Steps 17d-17f of the notebook):
        1. Autoencoder pre-training  (MSE reconstruction)
        2. Latent-space fitting      (per-class Gaussian N(_c, ?_cI))
        3. Weight transfer           (decoder ->generator, encoder ->discriminator)
        4. GAN adversarial training   ((n+1)-class discriminator)
    
    Epoch counts come from bagan_config (ae_epochs, gan_epochs), which
    should be set to reduced values for CV (e.g. 30 AE + 300 GAN) to
    keep wall-clock time reasonable.  Full epochs are only used for
    the final retrain step.
    
    Args:
        train_data:   Fold training images [N_train, 1, 64, 64] in [-1, 1].
        train_labels: Fold training labels [N_train] ->contiguous {0, 1, 2}.
        bagan_config: BAGANConfig instance with architecture + training params.
        device:       'cuda' or 'cpu'.
    
    Returns:
        generator:        Trained BAGANGenerator (eval-ready).
        latent_generator: Fitted ClassConditionalLatentGenerator.
        discriminator:    Trained BAGANDiscriminator.
    """
    if verbose:
        print(f"    BAGAN fold training: {len(train_data)} samples "
              f"(AE={bagan_config.ae_epochs} ep, GAN={bagan_config.gan_epochs} ep)")
    
    # -> 0. Dispatch on BAGAN variant (110J backend toggle) --
    # If ``bagan_config`` comes from the ``bagan_gp`` package, swap in the
    # WGAN-GP critic + matching trainers/weight-transfer. Otherwise use
    # the vanilla BAGAN classes already imported at module level. This is
    # how 110J achieves the *exact same* fold-training behaviour as 106J
    # whenever ``gan_backend='bagan'`` is selected on the AblationConfig.
    if type(bagan_config).__module__.startswith("bagan_gp"):
        gan = get_gan_modules("bagan_gp")
        AECls = gan.BAGANAutoencoder
        AETrainerCls = gan.AutoencoderTrainer
        GenCls = gan.BAGANGenerator
        DiscCls = gan.BAGANDiscriminator
        GANTrainerCls = gan.GANTrainer
        LatentGenCls = gan.ClassConditionalLatentGenerator
        WT = gan.WeightTransfer
        if verbose:
            print("    [variant] BAGAN-GP (WGAN-GP critic, aux classifier head)")
    else:
        AECls = BAGANAutoencoder
        AETrainerCls = AutoencoderTrainer
        GenCls = BAGANGenerator
        DiscCls = BAGANDiscriminator
        GANTrainerCls = GANTrainer
        LatentGenCls = ClassConditionalLatentGenerator
        WT = WeightTransfer
    
    # -> 1. Build DataLoader from fold training data --
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=bagan_config.batch_size,
        shuffle=True,
    )

    # When verbose=False, silence the BAGAN sub-trainers' unconditional
    # banner prints (AutoencoderTrainer.__init__/.fit, GANTrainer.__init__/.fit,
    # ClassConditionalLatentGenerator.__init__/.fit, WeightTransfer) by
    # redirecting stdout for the rest of this function.
    _stdout_ctx = (
        contextlib.redirect_stdout(io.StringIO())
        if not verbose
        else contextlib.nullcontext()
    )
    with _stdout_ctx:
        # -> 2. Autoencoder pre-training --
        # Train an autoencoder to learn a compressed representation of the
        # fold's training images.  The encoder will later provide the latent
        # space for class-conditional sampling, and the decoder's weights are
        # transferred to the generator for stable GAN initialisation.
        ae_kwargs = dict(
            in_channels=bagan_config.img_channels,
            latent_dim=bagan_config.latent_dim,
            base_channels=bagan_config.encoder_channels[0],
        )
        autoencoder = AECls(**ae_kwargs).to(device)
        ae_trainer = AETrainerCls(
            config=bagan_config,
            autoencoder=autoencoder,
            train_loader=train_loader,
        )
        ae_trainer.fit(
            epochs=bagan_config.ae_epochs,
            verbose=verbose,
            save_every=0,
            visualize_every=10 if show_bagan_plots else 0,
        )

        # -> 3. Fit class-conditional latent distributions --
        # Encode all training images ->group by class ->fit N(_c, ?_cI).
        # These per-class Gaussians allow us to sample latent vectors z that
        # will generate images of a specific class.
        latent_gen = LatentGenCls(bagan_config)
        latent_gen.fit(autoencoder.encoder, train_loader)

        # -> 4. Build GAN models + transfer AE weights --
        # Generator  ->decoder weights  (all layers)
        # Discriminator ->encoder weights (conv layers only)
        # This is the key BAGAN innovation: AE pre-training stabilises GAN
        # training by providing a good starting point.
        generator = GenCls(
            out_channels=bagan_config.img_channels,
            latent_dim=bagan_config.latent_dim,
            base_channels=bagan_config.encoder_channels[0],
        ).to(device)
        discriminator = DiscCls(
            in_channels=bagan_config.img_channels,
            num_classes=bagan_config.num_classes,
            base_channels=bagan_config.encoder_channels[0],
        ).to(device)
        WT.initialize_gan_from_autoencoder(
            autoencoder, generator, discriminator
        )

        # -> 5. GAN adversarial training --
        # Train generator vs. (n+1)-class discriminator (vanilla BAGAN) or
        # WGAN-GP critic + auxiliary classifier (BAGAN-GP). Both expose the
        # same ``GANTrainer(config, generator, discriminator, latent_generator,
        # train_loader).fit(epochs, save_every, verbose)`` API.
        gan_trainer = GANTrainerCls(
            config=bagan_config,
            generator=generator,
            discriminator=discriminator,
            latent_generator=latent_gen,
            train_loader=train_loader,
        )
        if type(bagan_config).__module__.startswith("bagan_gp"):
            gan_trainer.fit(epochs=bagan_config.gan_epochs, verbose=verbose, save_every=0)
        else:
            gan_trainer.fit(
                epochs=bagan_config.gan_epochs,
                verbose=verbose,
                save_every=0,
                visualize_every=50 if show_bagan_plots else 0,
                show_plots=show_bagan_plots,
            )

    return generator, latent_gen, discriminator


def _validate_mode(
    mode: str,
    bagan_config=None,
):
    """Validate that mode is recognized and bagan_config is provided when needed.
    
    This provides early failure with clear error messages BEFORE
    training begins. Without this, you'd get a cryptic error from
    ClassifierTrainer halfway through fold 1.
    
    Validation rules:
        - mode must be in VALID_MODES
        - BAGAN modes (ab4, ab5, ab6, ab7, ab8, ab9) require bagan_config
          so that _train_bagan_for_fold() can train a per-fold BAGAN
        - Non-BAGAN modes (ab1, ab1a, ab1b) don't need bagan_config
    
    Note:
        Generator/latent_generator/discriminator are NO LONGER validated
        here. BAGAN modes now train their own per-fold BAGAN internally
        via _train_bagan_for_fold(), so the caller no longer needs to
        provide pre-trained BAGAN components.
    
    Args:
        mode: Ablation mode string
        bagan_config: BAGANConfig instance (required for BAGAN modes)
    
    Raises:
        ValueError: If mode is invalid or bagan_config is missing for BAGAN modes.
    """
    # -> Check mode is recognized --
    if mode not in VALID_MODES:
        raise ValueError(
            f"Unknown ablation mode '{mode}'. "
            f"Must be one of {sorted(VALID_MODES)}"
        )
    
    # -> Check bagan_config for BAGAN modes --
    # BAGAN modes train a per-fold BAGAN pipeline internally. They need
    # bagan_config to know the architecture + training hyperparameters
    # (latent_dim, ae_epochs, gan_epochs, etc.).
    if mode in BAGAN_MODES and bagan_config is None:
        raise ValueError(
            f"Mode '{mode}' requires bagan_config (BAGANConfig instance) "
            f"for per-fold BAGAN training. "
            f"Pass bagan_config=... to run_kfold_ablation()."
        )


def _split_retrain_holdout(
    pool_data: torch.Tensor,
    pool_labels: torch.Tensor,
    val_fraction: float,
    random_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split pool into train/val for retrain checkpoint monitoring.

    Uses StratifiedShuffleSplit to preserve class proportions.
    With pool 800:80:80 and val_fraction=0.1, the split yields:
        Train: 720 : 72 : 72 = 864 samples
        Val:    80 :  8 :  8 =  96 samples

    Args:
        pool_data: Full pool images [N, 1, 64, 64].
        pool_labels: Full pool labels [N] ->original class labels.
        val_fraction: Fraction held out (e.g., 0.1 for 10%).
        random_seed: For reproducibility.

    Returns:
        (train_data, train_labels, val_data, val_labels)
    """
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=random_seed,
    )
    labels_np = pool_labels.numpy()
    train_idx, val_idx = next(sss.split(labels_np, labels_np))

    return (
        pool_data[train_idx], pool_labels[train_idx],
        pool_data[val_idx], pool_labels[val_idx],
    )


def retrain_on_full_pool(
    config,
    mode: str,
    pool_data: torch.Tensor,
    pool_labels: torch.Tensor,
    avg_stopping_epoch: int,
    device: str,
    generator=None,
    latent_generator=None,
    discriminator=None,
    reserve_data=None,
    reserve_labels=None,
    val_data=None,
    val_labels=None,
    verbose=None,
) -> 'CNNClassifier':
    """Retrain a classifier on the training pool with early stopping.

    After cross-validation selects the best method and determines
    avg_stopping_epoch, we retrain on the training data before
    evaluating on the held-out test set.

    525J design: a stratified hold-out (config.retrain_val_fraction)
    is carved from the pool for monitoring-only validation.  Early
    stopping (patience from config.get_retrain_patience(mode))
    terminates training when binary_f1 on the hold-out stalls.  The
    max epoch cap comes from config.get_retrain_max_epochs(mode,
    avg_stopping_epoch) ->typically 2x the CV estimate.

    The training pipeline mirrors train_fold() with consistent
    regularisation:
        1. Training uses ~90% of pool; ~10% is monitoring hold-out.
           (Set retrain_val_fraction=0.0 to disable hold-out and
           train for avg_stopping_epoch epochs with no ES.)
        2. Early stopping monitors binary_f1 on the hold-out.
        3. Best-checkpoint selection via EarlyStopping (restore_best_weights).

    527J fix: When val_data/val_labels are provided, the internal
    _split_retrain_holdout() is skipped and the pre-split holdout is
    used directly.  This prevents data leakage when the BAGAN was
    trained on pool_data ->the caller splits BEFORE BAGAN training
    and passes the clean holdout here.

    For BAGAN modes, the caller provides pre-trained BAGAN components
    (generator, latent_generator, discriminator) trained on the
    training portion of the pool ->NOT per-fold BAGANs.

    Args:
        config: AblationConfig instance.
        mode: Ablation mode ('ab1','ab1a','ab1b','ab4','ab5','ab6',
              'ab7','ab8','ab9','ab10').
        pool_data: Training pool images [N, 1, 64, 64].  When
                   val_data is provided, this should be the training
                   portion only (BAGAN training set).
        pool_labels: Training pool labels [N] ->original class
                 labels (non-contiguous, e.g. {0, 2, 3}).
        avg_stopping_epoch: CV-derived average stopping epoch.  Used
                  to compute max_epochs via config.get_retrain_max_epochs().
                  When monitoring is disabled, trains for exactly this many.
        device: 'cuda' or 'cpu'.
        generator: Pre-trained BAGANGenerator ->required for BAGAN modes.
        latent_generator: Pre-trained ClassConditionalLatentGenerator.
        discriminator: Pre-trained BAGANDiscriminator (needed for ab6).
        reserve_data: Reserve pool images for ab1b mode.
        reserve_labels: Reserve pool labels for ab1b mode.
        val_data: Pre-split validation holdout images [M, 1, 64, 64].
                  When provided, skips internal _split_retrain_holdout().
                  Should contain samples the BAGAN never saw (527J fix).
        val_labels: Pre-split validation holdout labels [M].

    Returns:
        CNNClassifier with best weights (by binary_f1 on the monitoring
        hold-out via ES), or last-epoch weights if monitoring is disabled.
    """
    # -> Resolve verbose (None ->config.cv_verbose) and gated logger --
    if verbose is None:
        verbose = getattr(config, 'cv_verbose', True)
    _log = print if verbose else (lambda *a, **k: None)
    final_epoch_mode = (mode == 'ab1f')

    # -> Compute retrain hyperparameters from config --
    use_monitoring = (config.retrain_val_fraction > 0
                      or (val_data is not None and val_labels is not None))
    if use_monitoring:
        max_epochs = config.get_retrain_max_epochs(mode, avg_stopping_epoch)
        retrain_pat = config.get_retrain_patience(mode)
    else:
        max_epochs = avg_stopping_epoch
        retrain_pat = None  # no ES when no monitoring

    _log(f"\n{'='*60}")
    _log(f"  {mode.upper()} - Retrain on full pool "
          f"(max_epochs={max_epochs}, patience={retrain_pat})")
    _log(f"{'='*60}")
    _log(f"    CV avg_stopping_epoch={avg_stopping_epoch}")

    # -> Split monitoring hold-out if configured --
    # 527J fix: prefer pre-split holdout (val_data/val_labels) to prevent
    # data leakage when BAGAN was trained on pool_data.
    if val_data is not None and val_labels is not None:
        # Pre-split holdout provided ->BAGAN never saw these samples
        train_data, train_labels = pool_data, pool_labels
        use_monitoring = True
        _log(f"    Retrain monitoring: {len(val_data)} pre-split hold-out samples")
    elif config.retrain_val_fraction > 0:
        train_data, train_labels, val_data, val_labels = _split_retrain_holdout(
            pool_data, pool_labels,
            config.retrain_val_fraction, config.random_seed,
        )
        use_monitoring = True
        _log(f"    Retrain monitoring: {len(val_data)} hold-out samples "
              f"({config.retrain_val_fraction:.0%})")
    else:
        train_data, train_labels = pool_data, pool_labels
        use_monitoring = False

    # -> Create ClassifierTrainer --
    # This gives us access to _prepare_training_data, _remap_labels,
    # _train_epoch, _train_epoch_ab5/6/7, and stored BAGAN components.
    trainer = ClassifierTrainer(
        config=config,
        mode=mode,
        generator=generator,
        latent_generator=latent_generator,
        discriminator=discriminator,
        reserve_data=reserve_data,
        reserve_labels=reserve_labels,
        device=device,
    )

    # -> Create fresh model + optimizer + criterion --
    model = CNNClassifier(
        input_channels=1,
        num_classes=config.num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # -> Create val_loader + early stopping --
    val_loader = None
    early_stopping = None

    if use_monitoring:
        val_labels_remapped = trainer._remap_labels(val_labels)
        val_dataset = TensorDataset(val_data, val_labels_remapped)
        val_loader = DeviceDataLoader(
            DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False),
            device,
        )
        early_stopping = EarlyStopping(
            patience=retrain_pat,
            min_delta=0,
            restore_best_weights=not final_epoch_mode,
            mode='max',  # maximize config.selection_metric (e.g. macro_f2)
        )

    # -> Ab-8: 3-phase AL loop --
    if mode == 'ab8':
        return _retrain_ab8(
            trainer, model, optimizer, criterion,
            train_data, train_labels, max_epochs, config, device,
            val_loader=val_loader,
            retrain_patience=retrain_pat,
            verbose=verbose,
        )

    # -> Ab-8a: 3-phase AL loop with aux-head label gate --
    if mode == 'ab8a':
        return _retrain_ab8a(
            trainer, model, optimizer, criterion,
            train_data, train_labels, max_epochs, config, device,
            val_loader=val_loader,
            retrain_patience=retrain_pat,
            verbose=verbose,
        )

    # -> Ab-8b / Ab-8d: 3-phase AL loop, selector resolved per mode --
    # ab8b defaults to BADGE; ab8d uses the confidence-gated diverse rule.
    if mode in ('ab8b', 'ab8d', 'ab8b_without_gp'):
        return _retrain_ab8b(
            trainer, model, optimizer, criterion,
            train_data, train_labels, max_epochs, config, device,
            val_loader=val_loader,
            retrain_patience=retrain_pat,
            verbose=verbose,
            mode=mode,
        )

    # -> Ab-8c: BADGE + 3-player co-training (D/G/C) --
    # Faithful retrain ->mirrors ``ClassifierTrainer._run_al_loop_ab8c``
    # (per-cycle D-step + G-step + C-step on the full pool) so that the
    # generator returned to E.1 reflects the co-training updates. This
    # is required for F.3 FID to be meaningfully different from the
    # base BAGAN-GP / ab8b snapshot.
    if mode == 'ab8c':
        return _retrain_ab8c(
            trainer, model, optimizer, criterion,
            train_data, train_labels, max_epochs, config, device,
            val_data=val_data, val_labels=val_labels,
            val_loader=val_loader,
            retrain_patience=retrain_pat,
            verbose=verbose,
        )

    # -> Ab-8b-on-BAGAN: Ab-8b BADGE + BAGAN-D acquisition filter --
    # Vanilla-BAGAN counterpart of Ab-8c. Frozen generator, BADGE
    # selection, plus a read-only BAGAN-discriminator (softmax) acquisition
    # filter. No WGAN-GP / G-step.
    if mode == 'ab8b_bagan':
        return _retrain_ab8b_bagan(
            trainer, model, optimizer, criterion,
            train_data, train_labels, max_epochs, config, device,
            val_loader=val_loader,
            retrain_patience=retrain_pat,
            verbose=verbose,
        )

    # -> Ab-9: pool-drain AL loop --
    if mode == 'ab9':
        return _retrain_ab9(
            trainer, model, optimizer, criterion,
            train_data, train_labels, max_epochs, config, device,
            val_loader=val_loader,
            retrain_patience=retrain_pat,
            verbose=verbose,
        )

    # -> Static modes (ab1, ab1a, ab1b, ab4): prepare data once --
    if not trainer._is_dynamic_mode():
        aug_data, aug_labels = trainer._prepare_training_data(
            train_data, train_labels
        )
        aug_labels = trainer._remap_labels(aug_labels)

        train_dataset = TensorDataset(aug_data, aug_labels)
        train_loader = DeviceDataLoader(
            DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
            ),
            device,
        )

        for epoch in range(max_epochs):
            train_loss = trainer._train_epoch(
                model, train_loader, optimizer, criterion
            )

            # -> Monitor on hold-out + early stopping --
            val_msg = ""
            if val_loader is not None:
                _, metrics, _, _ = trainer._validate(
                    model, val_loader, criterion
                )
                _sel = config.selection_metric
                mf1 = metrics[_sel]
                if (not final_epoch_mode) and early_stopping(
                    model, mf1, epoch=epoch
                ):
                    _log(f"    Early stopping at epoch {epoch+1} "
                          f"(best {_sel}={early_stopping.best_loss:.4f} "
                          f"at epoch {early_stopping.best_epoch+1})")
                    break
                star = " *" if early_stopping.improved else ""
                val_msg = f"  val_{_sel}={mf1:.4f}{star}"

            if (epoch + 1) % 10 == 0 or epoch == max_epochs - 1:
                _log(f"    Epoch {epoch+1}/{max_epochs}  loss={train_loss:.4f}{val_msg}")

    # -> Dynamic modes (ab5, ab6, ab7): per-epoch augmentation --
    else:
        epoch_fn = {
            'ab5': trainer._train_epoch_ab5,
            'ab6': trainer._train_epoch_ab6,
            'ab7': trainer._train_epoch_ab7,
            'ab10': trainer._train_epoch_ab10,
            'ab11': trainer._train_epoch_ab11,
            'ab11a': trainer._train_epoch_ab11a,
            # Ab-20 (Commit A): supervised-AE BAGAN-GP variant of Ab-5.
            # Reuses the Ab-5 epoch fn verbatim; the BAGAN-GP generator /
            # latent_generator / discriminator are injected into `trainer`
            # by the caller from saved/bagan_gp_*_107J.pth checkpoints.
            'ab20': trainer._train_epoch_ab5,
        }[mode]

        for epoch in range(max_epochs):
            trainer._current_epoch = epoch  # ab11 warmup gating
            train_loss = epoch_fn(
                model, train_data, train_labels, optimizer, criterion
            )

            # -> Monitor on hold-out + early stopping --
            val_msg = ""
            if val_loader is not None:
                _, metrics, _, _ = trainer._validate(
                    model, val_loader, criterion
                )
                _sel = config.selection_metric
                mf1 = metrics[_sel]
                if (not final_epoch_mode) and early_stopping(
                    model, mf1, epoch=epoch
                ):
                    _log(f"    Early stopping at epoch {epoch+1} "
                          f"(best {_sel}={early_stopping.best_loss:.4f} "
                          f"at epoch {early_stopping.best_epoch+1})")
                    break
                star = " *" if early_stopping.improved else ""
                val_msg = f"  val_{_sel}={mf1:.4f}{star}"

            if (epoch + 1) % 10 == 0 or epoch == max_epochs - 1:
                _log(f"    Epoch {epoch+1}/{max_epochs}  loss={train_loss:.4f}{val_msg}")

    # -> Restore best weights --
    if (
        (not final_epoch_mode)
        and early_stopping is not None
        and early_stopping.best_model is not None
    ):
        model.load_state_dict(early_stopping.best_model)
        _log(f"    Best checkpoint restored from epoch {early_stopping.best_epoch + 1}"
              f" ({config.selection_metric}={early_stopping.best_loss:.4f})")

    _log(f"  Retrain complete.\n")
    return model


def _retrain_ab8(
    trainer: ClassifierTrainer,
    model: CNNClassifier,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    pool_data: torch.Tensor,
    pool_labels: torch.Tensor,
    n_epochs: int,
    config,
    device: str,
    val_loader=None,
    retrain_patience=None,
    verbose: bool = True,
) -> CNNClassifier:
    """Ab-8 retrain: 3-phase AL loop on full pool.

    525J enhancement: if val_loader is provided, validates after each
    phase (warmup, every AL cycle, consolidation) and keeps the best
    checkpoint by binary_f1.  Cycle-level early stopping terminates the
    AL loop when binary_f1 stalls for retrain_patience consecutive cycles.

    Mirrors _run_al_loop_ab8() but without early stopping.
    The epoch budget n_epochs is decomposed into:
        warmup + (num_al_cycles x epochs_per_cycle) + consolidation
    using the same config parameters as the CV training.

    Args:
        trainer: ClassifierTrainer with BAGAN components stored.
        model: Fresh CNNClassifier to train.
        optimizer: Adam optimizer for model.
        criterion: CrossEntropyLoss.
        pool_data: Full pool images [N, 1, 64, 64].
        pool_labels: Full pool labels [N] ->original class labels.
        n_epochs: Total epoch budget.
        config: AblationConfig.
        device: 'cuda' or 'cpu'.
        val_loader: Optional DeviceDataLoader for monitoring hold-out.
                    If None, no validation or checkpointing is performed.

    Returns:
        CNNClassifier with best checkpoint weights (by binary_f1),
        or last-epoch weights if val_loader is None.
    """
    _log = print if verbose else (lambda *a, **k: None)
    # -> Compute deficit per minority class --
    target_counts = _compute_target_counts(pool_labels, config)
    deficits = {}
    for cls in config.minority_classes:
        current = (pool_labels == cls).sum().item()
        deficits[cls] = max(0, target_counts[cls] - current)

    max_deficit = max(deficits.values()) if deficits else 0
    num_al_cycles = max(1, math.ceil(max_deficit / config.ab8_samples_per_cycle))

    # Decompose n_epochs into 3 phases
    warmup = config.ab8_warmup_epochs
    consolidation = config.ab8_consolidation_epochs
    remaining = max(1, n_epochs - warmup - consolidation)
    epochs_per_cycle = max(1, remaining // num_al_cycles)

    _log(f"    Ab-8 retrain plan: warmup={warmup}, "
          f"{num_al_cycles} AL cycles x {epochs_per_cycle} ep, "
          f"consolidation={consolidation}")

    # -> Checkpoint tracking (525J) --
    best_state_dict = None
    best_metric = -1.0
    checkpoint_label = "init"

    # -> Cycle-level early stopping --
    es_counter = 0
    es_best_metric = -1.0

    def _maybe_checkpoint(label):
        """Validate on hold-out and checkpoint if selection_metric improves."""
        nonlocal best_state_dict, best_metric, checkpoint_label
        nonlocal es_counter, es_best_metric
        if val_loader is None:
            return False
        _, metrics, _, _ = trainer._validate(
            model, val_loader, nn.CrossEntropyLoss()
        )
        _sel = config.selection_metric
        mf1 = metrics[_sel]
        star = ""
        if mf1 > best_metric:
            best_metric = mf1
            best_state_dict = copy.deepcopy(model.state_dict())
            checkpoint_label = label
            star = " *"
        # Track cycle-level patience
        if mf1 > es_best_metric:
            es_best_metric = mf1
            es_counter = 0
        else:
            es_counter += 1
        _log(f"      val_{_sel}={mf1:.4f} @ {label}{star}")
        return False

    # Remap labels once
    real_labels_remapped = trainer._remap_labels(pool_labels)

    # Put generator in eval mode
    gen_was_training = trainer.generator.training
    trainer.generator.eval()

    # Cumulative synthetic pool
    synth_data_list = []
    synth_label_list = []

    try:
        # -> Phase 1: Warm-up on original imbalanced data --
        train_ds = TensorDataset(pool_data, real_labels_remapped)
        train_loader = DeviceDataLoader(
            DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
            device,
        )
        for epoch in range(warmup):
            trainer._train_epoch(model, train_loader, optimizer, criterion)
        _log(f"    Warm-up done ({warmup} epochs)")
        _maybe_checkpoint("warmup")

        # -> Phase 2: Active-learning acquisition loop --
        for cycle in range(num_al_cycles):
            # Generate + margin-select per minority class
            for cls in config.minority_classes:
                remapped_c = trainer.label_to_index[cls]
                already_added = sum(
                    (lbl == remapped_c).sum().item()
                    for lbl in synth_label_list
                )
                n_add = min(
                    config.ab8_samples_per_cycle,
                    deficits[cls] - already_added,
                )
                if n_add <= 0:
                    continue

                n_candidates = n_add * config.ab8_pool_size_factor
                z = trainer.latent_generator.sample_for_class_oversampling(
                    target_class=cls, n_samples=n_candidates
                )
                z = z.to(device)

                with torch.no_grad():
                    candidates = trainer.generator(z)
                    model.eval()
                    logits = model(candidates)
                    model.train()

                    probs = torch.softmax(logits, dim=1)
                    sorted_probs, _ = probs.sort(dim=1, descending=True)
                    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                    _, best_idx = torch.topk(margin, k=n_add, largest=False)
                    selected = candidates[best_idx].cpu()

                synth_data_list.append(selected)
                synth_label_list.append(
                    torch.full((n_add,), remapped_c, dtype=torch.long)
                )

            # Build grown dataset (real + cumulative synthetic)
            if synth_data_list:
                all_synth_data = torch.cat(synth_data_list, dim=0)
                all_synth_labels = torch.cat(synth_label_list, dim=0)
                grown_data = torch.cat([pool_data, all_synth_data], dim=0)
                grown_labels = torch.cat(
                    [real_labels_remapped, all_synth_labels], dim=0
                )
            else:
                grown_data = pool_data
                grown_labels = real_labels_remapped

            grown_ds = TensorDataset(grown_data, grown_labels)
            grown_loader = DeviceDataLoader(
                DataLoader(grown_ds, batch_size=config.batch_size, shuffle=True),
                device,
            )

            for epoch in range(epochs_per_cycle):
                trainer._train_epoch(model, grown_loader, optimizer, criterion)

            total_synth = sum(len(t) for t in synth_data_list)
            if (cycle + 1) % 5 == 0 or cycle == num_al_cycles - 1:
                _log(f"    AL cycle {cycle+1}/{num_al_cycles}: "
                      f"synth_pool={total_synth}")

            _maybe_checkpoint(f"cycle-{cycle+1}")

            # -> Cycle-level early stopping --
            if retrain_patience is not None and es_counter >= retrain_patience:
                _log(f"    Ab-8 AL early stopping at cycle {cycle+1} "
                      f"(patience={retrain_patience})")
                break

        # -> Phase 3: Consolidation on final balanced dataset --
        if synth_data_list:
            all_synth_data = torch.cat(synth_data_list, dim=0)
            all_synth_labels = torch.cat(synth_label_list, dim=0)
            final_data = torch.cat([pool_data, all_synth_data], dim=0)
            final_labels = torch.cat(
                [real_labels_remapped, all_synth_labels], dim=0
            )
        else:
            final_data = pool_data
            final_labels = real_labels_remapped

        final_ds = TensorDataset(final_data, final_labels)
        final_loader = DeviceDataLoader(
            DataLoader(final_ds, batch_size=config.batch_size, shuffle=True),
            device,
        )
        for epoch in range(consolidation):
            trainer._train_epoch(model, final_loader, optimizer, criterion)

        _log(f"    Consolidation done ({consolidation} epochs)")
        _maybe_checkpoint("consolidation")

    finally:
        trainer.generator.train(mode=gen_was_training)

    # -> Restore best checkpoint (525J) --
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        _log(f"    Best checkpoint: {checkpoint_label} "
              f"(macro_f1={best_metric:.4f})")

    _log(f"  Retrain complete.\n")
    return model


def _retrain_ab8a(
    trainer: ClassifierTrainer,
    model: CNNClassifier,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    pool_data: torch.Tensor,
    pool_labels: torch.Tensor,
    n_epochs: int,
    config,
    device: str,
    val_loader=None,
    retrain_patience=None,
    verbose: bool = True,
) -> CNNClassifier:
    """Ab-8a retrain: Ab-8 3-phase loop with BAGAN-GP aux-head label gate.

    Mirrors ``_retrain_ab8`` exactly except for Phase-2 step 2a, where
    candidates are filtered by the discriminator's aux head
    (argmax == intended AND softmax-max >= tau) before margin selection.
    Uses ``trainer._adv_class_logits(...)`` so discriminator output format is
    backend-agnostic (BAGAN returns logits tensor; BAGAN-GP returns tuple).
    """
    _log = print if verbose else (lambda *a, **k: None)
    if trainer.discriminator is None:
        raise RuntimeError(
            "Ab-8a retrain requires trainer.discriminator (BAGAN-GP critic)."
        )

    # -> Deficits (same as Ab-8) --
    target_counts = _compute_target_counts(pool_labels, config)
    deficits = {}
    for cls in config.minority_classes:
        current = (pool_labels == cls).sum().item()
        deficits[cls] = max(0, target_counts[cls] - current)

    max_deficit = max(deficits.values()) if deficits else 0
    num_al_cycles = max(1, math.ceil(max_deficit / config.ab8_samples_per_cycle))

    warmup = config.ab8_warmup_epochs
    consolidation = config.ab8_consolidation_epochs
    remaining = max(1, n_epochs - warmup - consolidation)
    epochs_per_cycle = max(1, remaining // num_al_cycles)

    _log(f"    Ab-8a retrain plan: warmup={warmup}, "
          f"{num_al_cycles} AL cycles x {epochs_per_cycle} ep, "
          f"consolidation={consolidation}, "
          f"aux_tau={config.ab8a_aux_tau} "
          f"(fallback {config.ab8a_aux_tau_fallback}), "
          f"pool_factor={config.ab8a_pool_size_factor}")

    best_state_dict = None
    best_metric = -1.0
    checkpoint_label = "init"
    es_counter = 0
    es_best_metric = -1.0

    def _maybe_checkpoint(label):
        nonlocal best_state_dict, best_metric, checkpoint_label
        nonlocal es_counter, es_best_metric
        if val_loader is None:
            return False
        _, metrics, _, _ = trainer._validate(
            model, val_loader, nn.CrossEntropyLoss()
        )
        _sel = config.selection_metric
        mf1 = metrics[_sel]
        star = ""
        if mf1 > best_metric:
            best_metric = mf1
            best_state_dict = copy.deepcopy(model.state_dict())
            checkpoint_label = label
            star = " *"
        if mf1 > es_best_metric:
            es_best_metric = mf1
            es_counter = 0
        else:
            es_counter += 1
        _log(f"      val_{_sel}={mf1:.4f} @ {label}{star}")
        return False

    real_labels_remapped = trainer._remap_labels(pool_labels)

    gen_was_training = trainer.generator.training
    disc_was_training = trainer.discriminator.training
    trainer.generator.eval()
    trainer.discriminator.eval()

    synth_data_list = []
    synth_label_list = []

    try:
        # -> Phase 1: Warm-up --
        train_ds = TensorDataset(pool_data, real_labels_remapped)
        train_loader = DeviceDataLoader(
            DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
            device,
        )
        for _ in range(warmup):
            trainer._train_epoch(model, train_loader, optimizer, criterion)
        _log(f"    Warm-up done ({warmup} epochs)")
        _maybe_checkpoint("warmup")

        # -> Phase 2: Gated AL loop --
        for cycle in range(num_al_cycles):
            for cls in config.minority_classes:
                remapped_c = trainer.label_to_index[cls]
                already_added = sum(
                    (lbl == remapped_c).sum().item()
                    for lbl in synth_label_list
                )
                n_add = min(
                    config.ab8_samples_per_cycle,
                    deficits[cls] - already_added,
                )
                if n_add <= 0:
                    continue

                n_candidates = n_add * config.ab8a_pool_size_factor
                z = trainer.latent_generator.sample_for_class_oversampling(
                    target_class=cls, n_samples=n_candidates
                )
                z = z.to(device)

                with torch.no_grad():
                    candidates = trainer.generator(z)

                    # -> Aux-head label gate --
                    aux_logits = trainer._adv_class_logits(trainer.discriminator, candidates)
                    if aux_logits.shape[1] > config.num_classes:
                        aux_logits = aux_logits[:, :config.num_classes]
                    aux_probs = torch.softmax(aux_logits, dim=1)
                    aux_argmax = aux_probs.argmax(dim=1)
                    aux_max = aux_probs.max(dim=1).values

                    intended_match = (aux_argmax == remapped_c)
                    gate_a = intended_match & (aux_max >= config.ab8a_aux_tau)

                    if gate_a.sum().item() >= n_add:
                        survivor_idx = gate_a.nonzero(as_tuple=False).squeeze(1)
                    else:
                        gate_b = intended_match & (aux_max >= config.ab8a_aux_tau_fallback)
                        if gate_b.sum().item() >= n_add:
                            survivor_idx = gate_b.nonzero(as_tuple=False).squeeze(1)
                        else:
                            survivor_idx = intended_match.nonzero(as_tuple=False).squeeze(1)

                    n_survivors = survivor_idx.numel()
                    if n_survivors == 0:
                        continue

                    k = min(n_add, n_survivors)
                    if n_survivors == k:
                        selected_local = torch.arange(n_survivors, device=candidates.device)
                    else:
                        survivor_cands = candidates[survivor_idx]
                        model.eval()
                        logits = model(survivor_cands)
                        model.train()
                        probs = torch.softmax(logits, dim=1)
                        sorted_probs, _ = probs.sort(dim=1, descending=True)
                        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                        _, local_idx = torch.topk(margin, k=k, largest=False)
                        selected_local = local_idx

                    selected = candidates[survivor_idx[selected_local]].cpu()

                synth_data_list.append(selected)
                synth_label_list.append(
                    torch.full((k,), remapped_c, dtype=torch.long)
                )

            if synth_data_list:
                all_synth_data = torch.cat(synth_data_list, dim=0)
                all_synth_labels = torch.cat(synth_label_list, dim=0)
                grown_data = torch.cat([pool_data, all_synth_data], dim=0)
                grown_labels = torch.cat(
                    [real_labels_remapped, all_synth_labels], dim=0
                )
            else:
                grown_data = pool_data
                grown_labels = real_labels_remapped

            grown_ds = TensorDataset(grown_data, grown_labels)
            grown_loader = DeviceDataLoader(
                DataLoader(grown_ds, batch_size=config.batch_size, shuffle=True),
                device,
            )

            for _ in range(epochs_per_cycle):
                trainer._train_epoch(model, grown_loader, optimizer, criterion)

            total_synth = sum(len(t) for t in synth_data_list)
            if (cycle + 1) % 5 == 0 or cycle == num_al_cycles - 1:
                _log(f"    AL cycle {cycle+1}/{num_al_cycles}: "
                      f"synth_pool={total_synth}")

            _maybe_checkpoint(f"cycle-{cycle+1}")

            if retrain_patience is not None and es_counter >= retrain_patience:
                _log(f"    Ab-8a AL early stopping at cycle {cycle+1} "
                      f"(patience={retrain_patience})")
                break

        # -> Phase 3: Consolidation --
        if synth_data_list:
            all_synth_data = torch.cat(synth_data_list, dim=0)
            all_synth_labels = torch.cat(synth_label_list, dim=0)
            final_data = torch.cat([pool_data, all_synth_data], dim=0)
            final_labels = torch.cat(
                [real_labels_remapped, all_synth_labels], dim=0
            )
        else:
            final_data = pool_data
            final_labels = real_labels_remapped

        final_ds = TensorDataset(final_data, final_labels)
        final_loader = DeviceDataLoader(
            DataLoader(final_ds, batch_size=config.batch_size, shuffle=True),
            device,
        )
        for _ in range(consolidation):
            trainer._train_epoch(model, final_loader, optimizer, criterion)

        _log(f"    Consolidation done ({consolidation} epochs)")
        _maybe_checkpoint("consolidation")

    finally:
        trainer.generator.train(mode=gen_was_training)
        trainer.discriminator.train(mode=disc_was_training)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        _log(f"    Best checkpoint: {checkpoint_label} "
              f"(macro_f1={best_metric:.4f})")

    _log(f"  Retrain complete.\n")
    return model


def _retrain_ab8b(
    trainer: ClassifierTrainer,
    model: CNNClassifier,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    pool_data: torch.Tensor,
    pool_labels: torch.Tensor,
    n_epochs: int,
    config,
    device: str,
    val_loader=None,
    retrain_patience=None,
    verbose: bool = True,
    mode: str = 'ab8b',
) -> CNNClassifier:
    """Ab-8b / Ab-8d retrain: Ab-8 3-phase loop with the margin selector swapped.

    Mirrors ``_retrain_ab8`` exactly EXCEPT for Phase-2 step 2a, where the
    acquisition rule is resolved from ``config.get_selection_params(mode)``:
    ``ab8b`` defaults to BADGE (k-means++ on the gradient embedding
    g_i = (p_i - e_yhat_i) (x) h_i, Ash et al. 2020), while ``ab8d`` applies the
    confidence gate first and then k-means++ over the penultimate features.
    ``h_i`` comes from a forward hook on the classifier's final ``nn.Linear``.
    No aux-gate, no discriminator usage.

    Candidate-latent SEEDING is also resolved per mode via
    ``_ab8c_seed_latents(..., mode=mode)``: Ab-8d reads ``ab8d_seed_mode``
    (132J default: ``"latent_smote"``, matching Proposed/Ab-8c), while Ab-8b
    has no dedicated field and keeps sampling the class-conditional prior.
    """
    _log = print if verbose else (lambda *a, **k: None)

    # -> Acquisition rule for THIS mode --
    _sel = config.get_selection_params(mode)
    _sel_mode = _sel['mode']
    _sel_min_conf = _sel['min_confidence']
    _sel_normalize = _sel['normalize']
    _sel_pool_factor = _sel['pool_factor']

    # -> Deficits (same as Ab-8) --
    target_counts = _compute_target_counts(pool_labels, config)
    deficits = {}
    for cls in config.minority_classes:
        current = (pool_labels == cls).sum().item()
        deficits[cls] = max(0, target_counts[cls] - current)

    max_deficit = max(deficits.values()) if deficits else 0
    num_al_cycles = max(1, math.ceil(max_deficit / config.ab8_samples_per_cycle))

    warmup = config.ab8_warmup_epochs
    consolidation = config.ab8_consolidation_epochs
    remaining = max(1, n_epochs - warmup - consolidation)
    epochs_per_cycle = max(1, remaining // num_al_cycles)

    _log(f"    {mode} retrain plan: warmup={warmup}, "
          f"{num_al_cycles} AL cycles x {epochs_per_cycle} ep, "
          f"consolidation={consolidation}, "
          f"pool_factor={_sel_pool_factor}, "
          f"kpp_normalize={_sel_normalize}, "
          f"selector={describe_selection(_sel_mode, _sel_min_conf)}")

    # Locate final nn.Linear for the penultimate-activation hook
    final_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            final_linear = m
    if final_linear is None:
        raise RuntimeError(
            f"{mode} retrain requires the classifier to contain at least "
            "one nn.Linear (none found in CNNClassifier)."
        )

    # -> Checkpoint tracking --
    best_state_dict = None
    best_metric = -1.0
    checkpoint_label = "init"
    es_counter = 0
    es_best_metric = -1.0

    def _maybe_checkpoint(label):
        nonlocal best_state_dict, best_metric, checkpoint_label
        nonlocal es_counter, es_best_metric
        if val_loader is None:
            return False
        _, metrics, _, _ = trainer._validate(
            model, val_loader, nn.CrossEntropyLoss()
        )
        _sel = config.selection_metric
        mf1 = metrics[_sel]
        star = ""
        if mf1 > best_metric:
            best_metric = mf1
            best_state_dict = copy.deepcopy(model.state_dict())
            checkpoint_label = label
            star = " *"
        if mf1 > es_best_metric:
            es_best_metric = mf1
            es_counter = 0
        else:
            es_counter += 1
        _log(f"      val_{_sel}={mf1:.4f} @ {label}{star}")
        return False

    real_labels_remapped = trainer._remap_labels(pool_labels)

    gen_was_training = trainer.generator.training
    trainer.generator.eval()

    synth_data_list = []
    synth_label_list = []

    try:
        # -> Phase 1: Warm-up on original imbalanced data --
        train_ds = TensorDataset(pool_data, real_labels_remapped)
        train_loader = DeviceDataLoader(
            DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
            device,
        )
        for epoch in range(warmup):
            trainer._train_epoch(model, train_loader, optimizer, criterion)
        _log(f"    Warm-up done ({warmup} epochs)")
        _maybe_checkpoint("warmup")

        # -> Phase 2: Active-learning acquisition loop --
        for cycle in range(num_al_cycles):
            # -> 2a. Generate + BADGE-select per minority class --
            for cls in config.minority_classes:
                remapped_c = trainer.label_to_index[cls]
                already_added = sum(
                    (lbl == remapped_c).sum().item()
                    for lbl in synth_label_list
                )
                n_add = min(
                    config.ab8_samples_per_cycle,
                    deficits[cls] - already_added,
                )
                if n_add <= 0:
                    continue

                n_candidates = n_add * _sel_pool_factor
                z = _ab8c_seed_latents(
                    trainer.latent_generator, cls, n_candidates, config,
                    mode=mode,
                )
                z = z.to(device)

                with torch.no_grad():
                    candidates = trainer.generator(z)

                    captured = {}

                    def _hook(_mod, inp, _out):
                        captured['h'] = inp[0].detach()

                    handle = final_linear.register_forward_hook(_hook)
                    try:
                        model.eval()
                        logits = model(candidates)
                        model.train()
                    finally:
                        handle.remove()

                    h = captured['h']  # [n_cand, d]
                    probs = torch.softmax(logits, dim=1)  # [n_cand, K]

                kpp_idx = select_indices(
                    probs, h, n_add, remapped_c,
                    mode=_sel_mode,
                    min_confidence=_sel_min_conf,
                    normalize=_sel_normalize,
                    seed=config.run_seed,   # algorithmic seed (varies per run)
                )
                k = int(kpp_idx.numel())
                selected = candidates[kpp_idx].cpu()

                synth_data_list.append(selected)
                synth_label_list.append(
                    torch.full((k,), remapped_c, dtype=torch.long)
                )

            # -> 2b. Build grown dataset --
            if synth_data_list:
                all_synth_data = torch.cat(synth_data_list, dim=0)
                all_synth_labels = torch.cat(synth_label_list, dim=0)
                grown_data = torch.cat([pool_data, all_synth_data], dim=0)
                grown_labels = torch.cat(
                    [real_labels_remapped, all_synth_labels], dim=0
                )
            else:
                grown_data = pool_data
                grown_labels = real_labels_remapped

            grown_ds = TensorDataset(grown_data, grown_labels)
            grown_loader = DeviceDataLoader(
                DataLoader(grown_ds, batch_size=config.batch_size, shuffle=True),
                device,
            )

            for epoch in range(epochs_per_cycle):
                trainer._train_epoch(model, grown_loader, optimizer, criterion)

            total_synth = sum(len(t) for t in synth_data_list)
            if (cycle + 1) % 5 == 0 or cycle == num_al_cycles - 1:
                _log(f"    AL cycle {cycle+1}/{num_al_cycles}: "
                      f"synth_pool={total_synth}")

            _maybe_checkpoint(f"cycle-{cycle+1}")

            if retrain_patience is not None and es_counter >= retrain_patience:
                _log(f"    Ab-8b AL early stopping at cycle {cycle+1} "
                      f"(patience={retrain_patience})")
                break

        # -> Phase 3: Consolidation --
        if synth_data_list:
            all_synth_data = torch.cat(synth_data_list, dim=0)
            all_synth_labels = torch.cat(synth_label_list, dim=0)
            final_data = torch.cat([pool_data, all_synth_data], dim=0)
            final_labels = torch.cat(
                [real_labels_remapped, all_synth_labels], dim=0
            )
        else:
            final_data = pool_data
            final_labels = real_labels_remapped

        final_ds = TensorDataset(final_data, final_labels)
        final_loader = DeviceDataLoader(
            DataLoader(final_ds, batch_size=config.batch_size, shuffle=True),
            device,
        )
        for epoch in range(consolidation):
            trainer._train_epoch(model, final_loader, optimizer, criterion)

        _log(f"    Consolidation done ({consolidation} epochs)")
        _maybe_checkpoint("consolidation")

    finally:
        trainer.generator.train(mode=gen_was_training)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        _log(f"    Best checkpoint: {checkpoint_label} "
              f"({config.selection_metric}={best_metric:.4f})")

    _log(f"  Retrain complete.\n")
    return model


def _retrain_ab8b_bagan(
    trainer: ClassifierTrainer,
    model: CNNClassifier,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    pool_data: torch.Tensor,
    pool_labels: torch.Tensor,
    n_epochs: int,
    config,
    device: str,
    val_loader=None,
    retrain_patience=None,
    verbose: bool = True,
) -> CNNClassifier:
    """Ab-8b-on-BAGAN retrain: Ab-8b BADGE + BAGAN-D acquisition filter.

    Mirrors ``_retrain_ab8b`` (frozen-generator BADGE acquisition, no aux
    gate, no GAN updates) and adds the Ab-8c-style D.aux acquisition
    filter using the VANILLA BAGAN discriminator's existing (n+1)-class
    softmax (read-only). The discriminator is never updated. This is the
    retrain twin of ``ClassifierTrainer._run_al_loop_ab8b_bagan``.
    """
    _log = print if verbose else (lambda *a, **k: None)
    from sklearn.cluster import kmeans_plusplus

    # -> Deficits (same as Ab-8 / Ab-8b) --
    target_counts = _compute_target_counts(pool_labels, config)
    deficits = {}
    for cls in config.minority_classes:
        current = (pool_labels == cls).sum().item()
        deficits[cls] = max(0, target_counts[cls] - current)

    max_deficit = max(deficits.values()) if deficits else 0
    num_al_cycles = max(1, math.ceil(max_deficit / config.ab8_samples_per_cycle))

    warmup = config.ab8_warmup_epochs
    consolidation = config.ab8_consolidation_epochs
    remaining = max(1, n_epochs - warmup - consolidation)
    epochs_per_cycle = max(1, remaining // num_al_cycles)

    _tau = config.ab8b_bagan_acq_filter_tau
    _use_filter = _tau > 0.0 and trainer.discriminator is not None

    _log(f"    Ab-8c_bagan retrain plan: warmup={warmup}, "
          f"{num_al_cycles} AL cycles x {epochs_per_cycle} ep, "
          f"consolidation={consolidation}, "
          f"pool_factor={config.ab8b_bagan_pool_size_factor}, "
          f"D.aux filter ?={_tau if _use_filter else 'off'}")

    # Locate final nn.Linear for BADGE penultimate hook
    final_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            final_linear = m
    if final_linear is None:
        raise RuntimeError(
            "Ab-8c_bagan retrain requires the classifier to contain at "
            "least one nn.Linear (none found in CNNClassifier)."
        )

    # -> Checkpoint tracking --
    best_state_dict = None
    best_metric = -1.0
    checkpoint_label = "init"
    es_counter = 0
    es_best_metric = -1.0

    def _maybe_checkpoint(label):
        nonlocal best_state_dict, best_metric, checkpoint_label
        nonlocal es_counter, es_best_metric
        if val_loader is None:
            return False
        _, metrics, _, _ = trainer._validate(
            model, val_loader, nn.CrossEntropyLoss()
        )
        # ab8b_bagan-only checkpoint-selection override: restore the
        # best-`ab8b_bagan_selection_metric` checkpoint (e.g. macro_gmean) when
        # set, else the global selection_metric. BADGE already peaks F2 at a
        # low-recall point; selecting on g-mean lifts recall/g-mean/specificity.
        _sel = getattr(config, 'ab8b_bagan_selection_metric', '') or config.selection_metric
        mf1 = metrics[_sel]
        star = ""
        if mf1 > best_metric:
            best_metric = mf1
            best_state_dict = copy.deepcopy(model.state_dict())
            checkpoint_label = label
            star = " *"
        if mf1 > es_best_metric:
            es_best_metric = mf1
            es_counter = 0
        else:
            es_counter += 1
        _log(f"      val_{_sel}={mf1:.4f} @ {label}{star}")
        return False

    real_labels_remapped = trainer._remap_labels(pool_labels)

    gen_was_training = trainer.generator.training
    trainer.generator.eval()
    disc_was_training = (
        trainer.discriminator.training
        if trainer.discriminator is not None else False
    )
    if trainer.discriminator is not None:
        trainer.discriminator.eval()

    synth_data_list = []
    synth_label_list = []

    try:
        # -> Phase 1: Warm-up on original imbalanced data --
        train_ds = TensorDataset(pool_data, real_labels_remapped)
        train_loader = DeviceDataLoader(
            DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
            device,
        )
        for epoch in range(warmup):
            trainer._train_epoch(model, train_loader, optimizer, criterion)
        _log(f"    Warm-up done ({warmup} epochs)")
        _maybe_checkpoint("warmup")

        # -> Phase 2: Active-learning acquisition loop --
        for cycle in range(num_al_cycles):
            for cls in config.minority_classes:
                remapped_c = trainer.label_to_index[cls]
                already_added = sum(
                    (lbl == remapped_c).sum().item()
                    for lbl in synth_label_list
                )
                n_add = min(
                    config.ab8_samples_per_cycle,
                    deficits[cls] - already_added,
                )
                if n_add <= 0:
                    continue

                n_candidates = n_add * config.ab8b_bagan_pool_size_factor
                z = trainer.latent_generator.sample_for_class_oversampling(
                    target_class=cls, n_samples=n_candidates
                )
                z = z.to(device)

                with torch.no_grad():
                    candidates = trainer.generator(z)

                # -> D.aux acquisition filter (BAGAN softmax, read-only) ->
                if _use_filter:
                    with torch.no_grad():
                        d_logits = trainer._adv_class_logits(trainer.discriminator, candidates)
                        d_p = torch.softmax(d_logits, dim=1)[:, remapped_c]
                    keep = d_p >= _tau
                    min_keep = max(
                        n_add,
                        int(config.ab8b_bagan_acq_filter_min_keep_frac
                            * candidates.size(0)),
                    )
                    if int(keep.sum().item()) >= min_keep:
                        candidates = candidates[keep]
                    else:
                        k_top = min(min_keep, d_p.numel())
                        top_idx = torch.topk(d_p, k=k_top).indices
                        candidates = candidates[top_idx]

                # -> BADGE k-means++ selection --
                with torch.no_grad():
                    captured = {}

                    def _hook(_mod, inp, _out):
                        captured['h'] = inp[0].detach()

                    handle = final_linear.register_forward_hook(_hook)
                    try:
                        model.eval()
                        logits = model(candidates)
                        model.train()
                    finally:
                        handle.remove()

                    h = captured['h']
                    probs = torch.softmax(logits, dim=1)
                    y_hat = probs.argmax(dim=1)
                    p_minus_e = probs.clone()
                    p_minus_e[torch.arange(p_minus_e.size(0)), y_hat] -= 1.0
                    emb = (p_minus_e.unsqueeze(2) * h.unsqueeze(1)).reshape(
                        p_minus_e.size(0), -1
                    )
                    if config.ab8b_bagan_kpp_seed_normalize:
                        emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-12)
                    emb_np = emb.cpu().numpy()

                k = min(n_add, emb_np.shape[0])
                _, kpp_idx = kmeans_plusplus(
                    emb_np, n_clusters=k, random_state=config.run_seed,
                )
                selected = candidates[kpp_idx].cpu()
                synth_data_list.append(selected)
                synth_label_list.append(
                    torch.full((k,), remapped_c, dtype=torch.long)
                )

            # -> Build grown dataset --
            if synth_data_list:
                all_synth_data = torch.cat(synth_data_list, dim=0)
                all_synth_labels = torch.cat(synth_label_list, dim=0)
                grown_data = torch.cat([pool_data, all_synth_data], dim=0)
                grown_labels = torch.cat(
                    [real_labels_remapped, all_synth_labels], dim=0
                )
            else:
                grown_data = pool_data
                grown_labels = real_labels_remapped

            grown_ds = TensorDataset(grown_data, grown_labels)
            grown_loader = DeviceDataLoader(
                DataLoader(grown_ds, batch_size=config.batch_size, shuffle=True),
                device,
            )
            for epoch in range(epochs_per_cycle):
                trainer._train_epoch(model, grown_loader, optimizer, criterion)

            total_synth = sum(len(t) for t in synth_data_list)
            if (cycle + 1) % 5 == 0 or cycle == num_al_cycles - 1:
                _log(f"    AL cycle {cycle+1}/{num_al_cycles}: "
                      f"synth_pool={total_synth}")

            _maybe_checkpoint(f"cycle-{cycle+1}")

            if retrain_patience is not None and es_counter >= retrain_patience:
                _log(f"    Ab-8c_bagan AL early stopping at cycle {cycle+1} "
                      f"(patience={retrain_patience})")
                break

        # -> Phase 3: Consolidation --
        if synth_data_list:
            all_synth_data = torch.cat(synth_data_list, dim=0)
            all_synth_labels = torch.cat(synth_label_list, dim=0)
            final_data = torch.cat([pool_data, all_synth_data], dim=0)
            final_labels = torch.cat(
                [real_labels_remapped, all_synth_labels], dim=0
            )
        else:
            final_data = pool_data
            final_labels = real_labels_remapped

        final_ds = TensorDataset(final_data, final_labels)
        final_loader = DeviceDataLoader(
            DataLoader(final_ds, batch_size=config.batch_size, shuffle=True),
            device,
        )
        for epoch in range(consolidation):
            trainer._train_epoch(model, final_loader, optimizer, criterion)

        _log(f"    Consolidation done ({consolidation} epochs)")
        _maybe_checkpoint("consolidation")

    finally:
        trainer.generator.train(mode=gen_was_training)
        if trainer.discriminator is not None:
            trainer.discriminator.train(mode=disc_was_training)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        _log(f"    Best checkpoint: {checkpoint_label} "
              f"({config.selection_metric}={best_metric:.4f})")

    _log(f"  Retrain complete.\n")
    return model


def _retrain_ab8c(
    trainer: ClassifierTrainer,
    model: CNNClassifier,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    pool_data: torch.Tensor,
    pool_labels: torch.Tensor,
    n_epochs: int,
    config,
    device: str,
    val_data: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
    val_loader=None,
    retrain_patience=None,
    verbose: bool = True,
) -> CNNClassifier:
    """Ab-8c retrain: acquisition + 3-player co-training (D/G/C) on the full pool.

    Mirrors ``ClassifierTrainer._run_al_loop_ab8c`` line-for-line EXCEPT
    that the GAN (generator + discriminator) is intentionally NOT
    restored to its pre-retrain state at exit, so the F Phase-1 FID
    block in E.1 sees the co-trained generator.

    Same warm-up ->AL cycles ->consolidation skeleton as the CV loop;
    each AL cycle runs an acquisition step on every minority class
    (rule from ``config.get_selection_params('ab8c')``; 131J default is the
    confidence-gated diverse rule), then K_D WGAN-GP critic steps, K_G
    generator steps (adv + aux + cls + ?_D anchor against frozen G_ref), then
    ``epochs_per_cycle`` C-step passes on (real ->cumulative synth). Early
    stopping uses the same patience semantics as ``_retrain_ab8b``.
    """
    _log = print if verbose else (lambda *a, **k: None)
    from bagan_gp.training.gan_trainer import wgan_gp_critic_loss

    # -> Acquisition rule for Ab-8c --
    _sel = config.get_selection_params('ab8c')
    _sel_mode = _sel['mode']
    _sel_min_conf = _sel['min_confidence']
    _sel_normalize = _sel['normalize']
    _sel_pool_factor = _sel['pool_factor']

    if trainer.discriminator is None:
        raise RuntimeError(
            "Ab-8c retrain requires a BAGAN-GP discriminator with an aux "
            "head; trainer.discriminator is None."
        )
    if getattr(config, 'gan_backend', 'bagan') != 'bagan_gp':
        raise RuntimeError(
            "Ab-8c retrain requires gan_backend='bagan_gp' (uses the "
            f"WGAN-GP critic loss). Current gan_backend = "
            f"{getattr(config, 'gan_backend', None)!r}."
        )

    # -> Cycle planning --
    target_counts = _compute_target_counts(pool_labels, config)
    deficits = {}
    for cls in config.minority_classes:
        current = (pool_labels == cls).sum().item()
        deficits[cls] = max(0, target_counts[cls] - current)
    max_deficit = max(deficits.values()) if deficits else 0

    # 131J: Ab-8c uses its OWN samples-per-cycle and warm-up so it matches
    # casestudy/methods/m10_ab8c.py, while Ab-8/8a/8b/8d keep sharing the
    # ab8_* budget. Falls back to the shared values if the fields are absent.
    _ab8c_spc = getattr(config, 'ab8c_samples_per_cycle', config.ab8_samples_per_cycle)
    _ab8c_warmup = getattr(config, 'ab8c_warmup_epochs', config.ab8_warmup_epochs)

    num_al_cycles = max(1, math.ceil(max_deficit / _ab8c_spc))

    warmup = _ab8c_warmup
    consolidation = config.ab8_consolidation_epochs
    remaining = max(1, n_epochs - warmup - consolidation)
    epochs_per_cycle = max(1, remaining // num_al_cycles)

    # WGAN-GP base hyperparameters (must match BAGANGPConfig defaults).
    BAGAN_GP_BASE_LR = 1e-4
    BAGAN_GP_BETAS = (0.0, 0.9)
    BAGAN_GP_GP_LAMBDA = 10.0

    _log(
        f"    Ab-8c retrain plan: warmup={warmup}, "
        f"{num_al_cycles} AL cycles x {epochs_per_cycle} ep, "
        f"consolidation={consolidation}, "
        f"pool_factor={_sel_pool_factor}, "
        f"selector={describe_selection(_sel_mode, _sel_min_conf)}, "
        f"K_D={config.ab8c_k_d}, K_G={config.ab8c_k_g}, "
        f"lr_d_ft={BAGAN_GP_BASE_LR * config.ab8c_lr_d_factor:.1e}, "
        f"lr_g_ft={BAGAN_GP_BASE_LR * config.ab8c_lr_g_factor:.1e}"
    )
    _log(
        f"    Lever1/2: balanced_sampler={config.ab8c_balanced_sampler}, "
        f"label_smoothing={config.ab8c_label_smoothing}"
    )
    _log(
        f"    125J-v2: freeze_generator={config.ab8c_freeze_generator} "
        f"(G-step {'SKIPPED' if config.ab8c_freeze_generator else 'ON'}), "
        f"D.aux acq filter ?={config.ab8c_acq_filter_tau} "
        f"(min_keep_frac={config.ab8c_acq_filter_min_keep_frac})"
    )
    _log(
        f"    125J-v3: G-refine epochs={config.ab8c_g_refine_epochs} "
        f"(steps/ep={config.ab8c_g_refine_steps_per_epoch}, "
        f"lr_factor={config.ab8c_g_refine_lr_factor}, "
        f"fm_w={config.ab8c_g_refine_fm_weight}, "
        f"anchor_w={config.ab8c_g_refine_anchor_weight}, "
        f"fid_samples={config.ab8c_g_refine_fid_samples})"
    )

    # Lever 2: override caller-provided criterion with label-smoothed CE.
    if float(getattr(config, 'ab8c_label_smoothing', 0.0)) > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=float(config.ab8c_label_smoothing))

    # Lever 1: class-balanced C-step DataLoader factory.
    def _build_c_step_loader(data: torch.Tensor, labels: torch.Tensor):
        ds = TensorDataset(data, labels)
        if getattr(config, 'ab8c_balanced_sampler', False) and labels.numel() > 0:
            lbl_cpu = labels.detach().cpu().long()
            counts = torch.bincount(lbl_cpu, minlength=config.num_classes).clamp(min=1)
            w_per_class = 1.0 / counts.float()
            sample_w = w_per_class[lbl_cpu]
            sampler = WeightedRandomSampler(
                weights=sample_w.tolist(),
                num_samples=len(ds),
                replacement=True,
            )
            inner = DataLoader(ds, batch_size=config.batch_size, sampler=sampler)
        else:
            inner = DataLoader(ds, batch_size=config.batch_size, shuffle=True)
        return DeviceDataLoader(inner, device)

    # Locate final nn.Linear for BADGE penultimate hook
    final_linear: Optional[nn.Linear] = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            final_linear = m
    if final_linear is None:
        raise RuntimeError(
            "Ab-8c retrain requires the classifier to contain at least "
            "one nn.Linear (none found in CNNClassifier)."
        )

    real_labels_remapped = trainer._remap_labels(pool_labels)

    # -> GAN fine-tune optimizers --
    opt_D = torch.optim.Adam(
        trainer.discriminator.parameters(),
        lr=BAGAN_GP_BASE_LR * config.ab8c_lr_d_factor,
        betas=BAGAN_GP_BETAS,
    )
    opt_G = torch.optim.Adam(
        trainer.generator.parameters(),
        lr=BAGAN_GP_BASE_LR * config.ab8c_lr_g_factor,
        betas=BAGAN_GP_BETAS,
    )

    # Determinism off during WGAN-GP backward (Conv2d double-backward).
    det_was_on = torch.are_deterministic_algorithms_enabled()
    if det_was_on:
        torch.use_deterministic_algorithms(False)

    gen_was_training = trainer.generator.training
    disc_was_training = trainer.discriminator.training

    # -> Checkpoint / early-stopping bookkeeping --
    best_state_dict = None
    # S1: snapshot G/D/synth alongside C whenever best F1 improves so the
    # downstream F FID block (Option C: GAN is NOT restored on exit) and
    # the Phase-3 consolidation see the best-cycle state, not the last
    # drifted one.
    best_G_state = None
    best_D_state = None
    best_synth_data: Optional[List[torch.Tensor]] = None
    best_synth_labels: Optional[List[torch.Tensor]] = None
    best_metric = -1.0
    checkpoint_label = "init"
    es_counter = 0
    es_best_metric = -1.0

    # -> Pre-compute remapped val labels for aux-real diagnostic --
    if val_data is not None and val_labels is not None:
        val_labels_idx = trainer._remap_labels(val_labels)
    else:
        val_labels_idx = None

    synth_data_list: List[torch.Tensor] = []
    synth_label_list: List[torch.Tensor] = []

    def _maybe_checkpoint(label):
        nonlocal best_state_dict, best_metric, checkpoint_label
        nonlocal best_G_state, best_D_state
        nonlocal best_synth_data, best_synth_labels
        nonlocal es_counter, es_best_metric
        if val_loader is None:
            return False
        _, metrics, _, _ = trainer._validate(
            model, val_loader, nn.CrossEntropyLoss()
        )
        # 123J-1A: checkpoint + early-stop on config.selection_metric
        # (macro_f2) so the retrained best state matches the F2 grading.
        mf1 = metrics[config.selection_metric]
        star = ""
        if mf1 > best_metric:
            best_metric = mf1
            best_state_dict = copy.deepcopy(model.state_dict())
            best_G_state = copy.deepcopy(trainer.generator.state_dict())
            best_D_state = copy.deepcopy(trainer.discriminator.state_dict())
            best_synth_data = [t.clone() for t in synth_data_list]
            best_synth_labels = [t.clone() for t in synth_label_list]
            checkpoint_label = label
            star = " *"
        if mf1 > es_best_metric:
            es_best_metric = mf1
            es_counter = 0
        else:
            es_counter += 1
        _log(f"      val_{config.selection_metric}={mf1:.4f} @ {label}{star}")
        return False
    G_ref: Optional[nn.Module] = None
    D_ref: Optional[nn.Module] = None
    aux_real_acc_warmup = 0.0

    try:
        # -> Phase 1: Warm-up classifier on raw real data --
        trainer.generator.eval()
        trainer.discriminator.eval()
        train_ds = TensorDataset(pool_data, real_labels_remapped)
        train_loader_real = DeviceDataLoader(
            DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
            device,
        )
        for _ in range(warmup):
            trainer._train_epoch(model, train_loader_real, optimizer, criterion)
        _log(f"    Warm-up done ({warmup} epochs)")
        _maybe_checkpoint("warmup")

        # Snapshot G_ref (frozen) for the ?_D anchor.
        G_ref = copy.deepcopy(trainer.generator)
        G_ref.eval()
        for p in G_ref.parameters():
            p.requires_grad_(False)

        # Snapshot D_ref (frozen warm-up critic) for the best-G export
        # guard (Strategy A). Scoring both generator candidates with the
        # SAME un-drifted critic keeps the FID-floor comparison unbiased.
        D_ref = copy.deepcopy(trainer.discriminator)
        D_ref.eval()
        for p in D_ref.parameters():
            p.requires_grad_(False)

        if val_data is not None and val_labels_idx is not None:
            aux_real_acc_warmup = trainer._ab8c_aux_real_val_acc(
                val_data, val_labels_idx
            )
            _log(f"      D.aux warm-up val acc={aux_real_acc_warmup:.3f}")

        # -> 125J-v3: Minority generator refinement (the real third player)
        # Runs ONCE here, after warm-up, before the AL cycles. Fine-tunes G
        # toward real minority features (critic frozen) with InceptionV3-FID
        # keep-best, so the exported G's minority FID is ->Ab-8b (strictly <
        # if headroom exists). The improved G is then frozen for the BADGE
        # loop below ->Ab-8b's exact acquisition now draws from a better G,
        # lifting test-F2 as well. G_ref above is the warm-up snapshot; we
        # refresh it to the refined G so the (now-skipped) per-cycle anchor
        # would reference the exported generator, not the pre-refine one.
        _refine_fid = trainer._ab8c_refine_generator(
            pool_data, pool_labels, log=_log,
        )
        if _refine_fid:
            G_ref = copy.deepcopy(trainer.generator)
            G_ref.eval()
            for p in G_ref.parameters():
                p.requires_grad_(False)

        # -> Phase 2: AL cycles --
        for cycle in range(num_al_cycles):
            # -> (a) BADGE acquisition per minority class --
            trainer.generator.eval()
            trainer.discriminator.eval()
            model.eval()

            cycle_S_list: List[torch.Tensor] = []
            cycle_y_list: List[torch.Tensor] = []
            for cls in config.minority_classes:
                remapped_c = trainer.label_to_index[cls]
                already_added = sum(
                    int((lbl == remapped_c).sum().item())
                    for lbl in synth_label_list
                )
                remaining_c = deficits[cls] - already_added
                n_add = min(_ab8c_spc, remaining_c)
                if n_add <= 0:
                    continue
                n_candidates = n_add * _sel_pool_factor
                z = _ab8c_seed_latents(
                    trainer.latent_generator, cls, n_candidates, config
                ).to(device)
                with torch.no_grad():
                    candidates = trainer.generator(z)

                # -> 125J-v2: D.aux acquisition filter --
                # Drop candidates the (D-step-sharpened) aux head won't
                # confidently assign to the target minority class, so BADGE
                # selects diversity from a clean, on-manifold pool.
                if config.ab8c_acq_filter_tau > 0.0:
                    with torch.no_grad():
                        _aux_cand = trainer._adv_class_logits(trainer.discriminator, candidates)
                        _aux_p = torch.softmax(_aux_cand, dim=1)[:, remapped_c]
                    _keep = _aux_p >= config.ab8c_acq_filter_tau
                    _min_keep = max(
                        n_add,
                        int(config.ab8c_acq_filter_min_keep_frac
                            * candidates.size(0)),
                    )
                    if int(_keep.sum().item()) >= _min_keep:
                        candidates = candidates[_keep]
                    else:
                        _k = min(_min_keep, _aux_p.numel())
                        _top = torch.topk(_aux_p, k=_k).indices
                        candidates = candidates[_top]

                captured: Dict[str, torch.Tensor] = {}

                def _hook(_mod, inp, _out, _store=captured):
                    _store['h'] = inp[0].detach()

                handle = final_linear.register_forward_hook(_hook)
                try:
                    with torch.no_grad():
                        logits = model(candidates)
                finally:
                    handle.remove()

                h = captured['h']
                probs = torch.softmax(logits, dim=1)
                kpp_idx = select_indices(
                    probs, h, n_add, remapped_c,
                    mode=_sel_mode,
                    min_confidence=_sel_min_conf,
                    normalize=_sel_normalize,
                    seed=config.run_seed,   # algorithmic seed (varies per run)
                )
                k = int(kpp_idx.numel())
                selected = candidates[kpp_idx].detach()
                y_selected = torch.full(
                    (k,), remapped_c, dtype=torch.long, device=device,
                )
                cycle_S_list.append(selected)
                cycle_y_list.append(y_selected)
                synth_data_list.append(selected.detach().cpu())
                synth_label_list.append(y_selected.detach().cpu())

            model.train()
            cycle_S = torch.cat(cycle_S_list, dim=0) if cycle_S_list else None
            cycle_y = torch.cat(cycle_y_list, dim=0) if cycle_y_list else None

            # -> (b) D-step: K_D minibatches of WGAN-GP critic loss --
            trainer.generator.eval()
            trainer.discriminator.train()
            grad_d_norms: List[float] = []
            n_real = pool_data.size(0)
            for _ in range(config.ab8c_k_d):
                idx = torch.randint(0, n_real, (config.batch_size,))
                real_batch = pool_data[idx].to(device, non_blocking=True)
                real_y_batch = real_labels_remapped[idx].to(
                    device, non_blocking=True
                )
                z, _ = trainer._ab8c_sample_balanced_latent(config.batch_size)
                with torch.no_grad():
                    fake_batch = trainer.generator(z)

                if cycle_S is not None and cycle_S.size(0) > 0:
                    s_sz = min(config.batch_size, cycle_S.size(0))
                    s_idx = torch.randint(0, cycle_S.size(0), (s_sz,))
                    extra_x = cycle_S[s_idx]
                    extra_y = cycle_y[s_idx]
                    extra_w = config.ab8c_lambda_aux_synth
                else:
                    extra_x = None
                    extra_y = None
                    extra_w = 0.0

                opt_D.zero_grad(set_to_none=True)
                loss_d, _info_d = wgan_gp_critic_loss(
                    trainer.discriminator,
                    real_batch, fake_batch, real_y_batch,
                    gp_lambda=BAGAN_GP_GP_LAMBDA,
                    aux_w=config.ab8c_lambda_aux_real,
                    extra_x=extra_x, extra_y=extra_y, extra_aux_w=extra_w,
                    device=device,
                )
                loss_d.backward()
                total_sq = 0.0
                for p in trainer.discriminator.parameters():
                    if p.grad is not None:
                        total_sq += float(p.grad.detach().norm().item() ** 2)
                grad_d_norms.append(total_sq ** 0.5)
                opt_D.step()

            # -> (c) G-step: K_G minibatches --
            # 125J-v2: when ab8c_freeze_generator is True the generator is
            # held at its strong warm-up state (the third player only sharpens
            # D.aux as an acquisition critic). Skipping the G-step guarantees
            # the exported generator == warm-up G, so minority FID ties Ab-8b.
            if not config.ab8c_freeze_generator:
                trainer.generator.train()
                trainer.discriminator.eval()
                for p in trainer.discriminator.parameters():
                    p.requires_grad_(False)
                try:
                    for _ in range(config.ab8c_k_g):
                        z, y_z = trainer._ab8c_sample_balanced_latent(
                            config.batch_size
                        )
                        fake = trainer.generator(z)
                        s_fake, aux_fake_logits = trainer.discriminator(fake)
                        adv = -s_fake.mean()
                        aux_g = nn.functional.cross_entropy(aux_fake_logits, y_z)
                        was_c_training = model.training
                        model.eval()
                        cls_logits = model(fake)
                        if was_c_training:
                            model.train()
                        cls_l = nn.functional.cross_entropy(cls_logits, y_z)
                        with torch.no_grad():
                            fake_ref = G_ref(z)
                            phi_ref = trainer._disc_features(
                                trainer.discriminator, fake_ref
                            ).detach()
                        phi = trainer._disc_features(trainer.discriminator, fake)
                        anchor = ((phi - phi_ref) ** 2).mean()
                        g_loss = (
                            config.ab8c_lambda_adv * adv
                            + config.ab8c_lambda_aux_g * aux_g
                            + config.ab8c_lambda_cls * cls_l
                            + config.ab8c_lambda_anchor * anchor
                        )
                        opt_G.zero_grad(set_to_none=True)
                        g_loss.backward()
                        opt_G.step()
                finally:
                    for p in trainer.discriminator.parameters():
                        p.requires_grad_(True)

            # -> (d) C-step: epochs_per_cycle on real ->cumulative S --
            trainer.generator.eval()
            trainer.discriminator.eval()
            if synth_data_list:
                all_synth_data = torch.cat(synth_data_list, dim=0)
                all_synth_labels = torch.cat(synth_label_list, dim=0)
                grown_data = torch.cat([pool_data, all_synth_data], dim=0)
                grown_labels = torch.cat(
                    [real_labels_remapped, all_synth_labels], dim=0
                )
            else:
                grown_data = pool_data
                grown_labels = real_labels_remapped
            grown_loader = _build_c_step_loader(grown_data, grown_labels)
            for _ in range(epochs_per_cycle):
                trainer._train_epoch(model, grown_loader, optimizer, criterion)

            # -> (e) Diagnostics + checkpoint --
            total_synth = sum(t.size(0) for t in synth_data_list)
            mean_grad_d = float(np.mean(grad_d_norms)) if grad_d_norms else 0.0
            if val_data is not None and val_labels_idx is not None:
                aux_real_acc = trainer._ab8c_aux_real_val_acc(
                    val_data, val_labels_idx
                )
            else:
                aux_real_acc = 0.0
            if (cycle + 1) % 5 == 0 or cycle == num_al_cycles - 1:
                _log(
                    f"    AL cycle {cycle+1}/{num_al_cycles}: "
                    f"synth_pool={total_synth}, "
                    f"D.aux val acc={aux_real_acc:.3f}, "
                    f"||grad_D||={mean_grad_d:.3f}"
                )
            _maybe_checkpoint(f"cycle-{cycle+1}")

            if retrain_patience is not None and es_counter >= retrain_patience:
                _log(
                    f"    Ab-8c AL early stopping at cycle {cycle+1} "
                    f"(patience={retrain_patience})"
                )
                break

        # -> Phase 3: Consolidation (C only) --
        trainer.generator.eval()
        trainer.discriminator.eval()
        # S1: restore the best-cycle (C, G, D, synth pool) triplet so the
        # Phase-3 consolidation refines C against the same context that
        # produced the peak F1. Since this is the retrain path (Option C),
        # the restored G/D also propagate out to F FID after the function
        # returns.
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            if best_G_state is not None:
                trainer.generator.load_state_dict(best_G_state)
            if best_D_state is not None:
                trainer.discriminator.load_state_dict(best_D_state)
            if best_synth_data is not None:
                synth_data_list = [t.clone() for t in best_synth_data]
                synth_label_list = [t.clone() for t in best_synth_labels]
        if synth_data_list:
            all_synth_data = torch.cat(synth_data_list, dim=0)
            all_synth_labels = torch.cat(synth_label_list, dim=0)
            final_data = torch.cat([pool_data, all_synth_data], dim=0)
            final_labels = torch.cat(
                [real_labels_remapped, all_synth_labels], dim=0
            )
        else:
            final_data = pool_data
            final_labels = real_labels_remapped
        final_loader = _build_c_step_loader(final_data, final_labels)
        for _ in range(consolidation):
            trainer._train_epoch(model, final_loader, optimizer, criterion)
        _log(f"    Consolidation done ({consolidation} epochs)")
        _maybe_checkpoint("consolidation")

        # -> Best-G export guard (Strategy A, real-anchored) --
        # Ab-8c mutates G via co-training; Ab-8b reuses the frozen warm-up
        # generator. Now that the BAGAN-GP warm-up G is strong (balanced
        # critic + shrinkage prior), an over-eager co-training cycle can
        # leave G *worse* than warm-up on the minority manifold, inflating
        # G FID above Ab-8b.
        #
        # IMPORTANT: the guard must NOT use the critic score s = D(G(z)). The
        # G-step explicitly maximizes that score, so the co-trained G always
        # "wins" a critic comparison even when its FID is worse (this is why
        # the earlier critic-score guard failed to enforce the floor). Instead
        # we use a REAL-data-anchored Frchet-style distance in the frozen
        # warm-up critic's feature space ?_D_ref: for each minority class we
        # match the mean+variance of ?_D_ref(real_c) against ?_D_ref(G(z_c)).
        # This is a genuine generative-quality proxy that the adversarial
        # objective cannot game. Export whichever G (co-trained vs warm-up
        # G_ref) is closer to the real minority features. Guarantees Ab-8c's
        # exported generator is never worse than Ab-8b's on this proxy, and
        # keeps the co-trained G only when it truly improves minority realism.
        # The swap touches only the generator ->the classifier (best_state_dict)
        # and thus F2 are unchanged.
        if config.ab8c_freeze_generator:
            _log(
                "    [G-guard] freeze_generator=True ->G held at warm-up "
                "snapshot; minority FID == Ab-8b floor by construction."
            )
        elif G_ref is not None and D_ref is not None and config.minority_classes:
            def _real_anchored_distance(gen_module, z_by_class, real_feat_stats):
                """Mean Frchet-style feat distance to real minority features.

                Lower = generated minority features closer to the real ones.
                Uses mean + diagonal-variance terms (robust with ~36 real
                samples, where a full covariance would be rank-deficient).
                """
                gen_module.eval()
                dists = []
                with torch.no_grad():
                    for _c, _z in z_by_class.items():
                        _phi = trainer._disc_features(
                            D_ref, gen_module(_z)
                        ).float()
                        _mu_f = _phi.mean(dim=0)
                        _var_f = _phi.var(dim=0, unbiased=False)
                        _mu_r, _var_r = real_feat_stats[_c]
                        _d_mu = ((_mu_f - _mu_r) ** 2).sum()
                        _d_var = ((_var_f.sqrt() - _var_r.sqrt()) ** 2).sum()
                        dists.append(float((_d_mu + _d_var).item()))
                return float(np.mean(dists)) if dists else 0.0

            # Real minority feature statistics in D_ref space (anchor).
            _real_feat_stats = {}
            with torch.no_grad():
                for _cls in config.minority_classes:
                    _rc = trainer.label_to_index[_cls]
                    _mask = (real_labels_remapped == _rc)
                    if int(_mask.sum().item()) == 0:
                        continue
                    _real_c = pool_data[_mask].to(device)
                    _phi_r = trainer._disc_features(D_ref, _real_c).float()
                    _real_feat_stats[_cls] = (
                        _phi_r.mean(dim=0),
                        _phi_r.var(dim=0, unbiased=False),
                    )

            # Identical latents for both candidates ->fair comparison.
            _guard_z = {
                _cls: trainer.latent_generator.sample_for_class_oversampling(
                    target_class=_cls, n_samples=256,
                ).to(device)
                for _cls in _real_feat_stats.keys()
            }

            if _guard_z:
                _dist_cotrained = _real_anchored_distance(
                    trainer.generator, _guard_z, _real_feat_stats
                )
                _cotrained_state = copy.deepcopy(trainer.generator.state_dict())
                trainer.generator.load_state_dict(G_ref.state_dict())
                _dist_ref = _real_anchored_distance(
                    trainer.generator, _guard_z, _real_feat_stats
                )
                if _dist_ref <= _dist_cotrained:
                    _log(
                        f"    [G-guard] warm-up G_ref closer to real "
                        f"(dist {_dist_ref:.1f} ->co-trained {_dist_cotrained:.1f}) "
                        f"->exporting warm-up G for FID (Ab-8b floor)."
                    )
                    # G_ref weights already loaded into trainer.generator.
                else:
                    _log(
                        f"    [G-guard] co-trained G closer to real "
                        f"(dist {_dist_cotrained:.1f} < warm-up {_dist_ref:.1f}) "
                        f"->keeping co-trained G (Ab-8c wins on minority FID)."
                    )
                    trainer.generator.load_state_dict(_cotrained_state)

    finally:
        # Intentionally NOT restoring trainer.generator/discriminator
        # state  Option C requires the co-trained G to persist so F
        # Phase-1 FID reflects ab8c's per-cycle G-step updates.
        trainer.generator.train(mode=gen_was_training)
        trainer.discriminator.train(mode=disc_was_training)
        if det_was_on:
            torch.use_deterministic_algorithms(True)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        _log(f"    Best checkpoint: {checkpoint_label} "
              f"({config.selection_metric}={best_metric:.4f})")

    _log(f"  Retrain complete.\n")
    return model


def _retrain_ab9(
    trainer: ClassifierTrainer,
    model: CNNClassifier,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    pool_data: torch.Tensor,
    pool_labels: torch.Tensor,
    n_epochs: int,
    config,
    device: str,
    val_loader=None,
    retrain_patience=None,
    verbose: bool = True,
) -> CNNClassifier:
    """Ab-9 retrain: pool-drain AL loop on full pool.

    525J enhancement: if val_loader is provided, validates after each
    phase (warmup, every drain cycle, consolidation) and keeps the
    best checkpoint by binary_f1.  Cycle-level early stopping terminates
    the drain loop when binary_f1 stalls for retrain_patience consecutive
    cycles.

    Mirrors _run_al_loop_ab9() but without early stopping.
    The epoch budget n_epochs is decomposed into:
        warmup + (num_al_cycles x epochs_per_cycle) + consolidation

    Args:
        trainer: ClassifierTrainer with BAGAN components stored.
        model: Fresh CNNClassifier to train.
        optimizer: Adam optimizer for model.
        criterion: CrossEntropyLoss.
        pool_data: Full pool images [N, 1, 64, 64].
        pool_labels: Full pool labels [N] ->original class labels.
        n_epochs: Total epoch budget.
        config: AblationConfig.
        device: 'cuda' or 'cpu'.
        val_loader: Optional DeviceDataLoader for monitoring hold-out.
                    If None, no validation or checkpointing is performed.
        retrain_patience: If set, stop the drain loop after this many
                         consecutive cycles without binary_f1 improvement.

    Returns:
        CNNClassifier with best checkpoint weights (by binary_f1),
        or last-epoch weights if val_loader is None.
    """
    _log = print if verbose else (lambda *a, **k: None)
    # -> Phase I: Build balanced pool (real + synthetic) --
    gen_was_training = trainer.generator.training
    trainer.generator.eval()

    try:
        aug_data, aug_labels = generate_bagan_augmented_dataset(
            pool_data, pool_labels,
            trainer.generator, trainer.latent_generator,
            config, device,
        )

        # Remap ALL labels to contiguous [0, num_classes-1]
        aug_labels = trainer._remap_labels(aug_labels)
        real_labels_remapped = trainer._remap_labels(pool_labels)

        n_real = len(pool_data)
        n_pool = len(aug_data)
        _log(f"    Ab-9 Phase I: pool={n_pool} "
              f"(real={n_real}, synthetic={n_pool - n_real})")

        # -> Seed: pick ab9_init_per_class REAL samples per class --
        seed_indices = []
        for c in range(config.num_classes):
            class_mask = (aug_labels[:n_real] == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(class_mask))
            n_pick = min(config.ab9_init_per_class, len(class_mask))
            seed_indices.append(class_mask[perm[:n_pick]])
        seed_indices = torch.cat(seed_indices)

        # Split pool into training_set (seed) and remaining
        mask = torch.ones(n_pool, dtype=torch.bool)
        mask[seed_indices] = False
        al_train_data = aug_data[seed_indices].clone()
        al_train_labels = aug_labels[seed_indices].clone()
        remaining_data = aug_data[mask].clone()
        remaining_labels = aug_labels[mask].clone()

        # Compute AL cycle count from pool size
        num_al_cycles = max(
            1,
            math.ceil(len(remaining_data) / config.ab9_samples_per_cycle),
        )

        # Decompose n_epochs into 3 phases
        warmup = config.ab9_warmup_epochs
        consolidation = config.ab9_consolidation_epochs
        remaining_budget = max(1, n_epochs - warmup - consolidation)
        epochs_per_cycle = max(1, remaining_budget // num_al_cycles)

        _log(f"    Ab-9 retrain plan: warmup={warmup}, "
              f"{num_al_cycles} AL cycles x {epochs_per_cycle} ep, "
              f"consolidation={consolidation}")

        # -> Checkpoint tracking (525J) --
        best_state_dict = None
        best_metric = -1.0
        checkpoint_label = "init"

        # -> Cycle-level early stopping --
        es_counter = 0
        es_best_metric = -1.0

        def _maybe_checkpoint(label):
            """Validate on hold-out and checkpoint if selection_metric improves."""
            nonlocal best_state_dict, best_metric, checkpoint_label
            nonlocal es_counter, es_best_metric
            if val_loader is None:
                return
            _, metrics, _, _ = trainer._validate(
                model, val_loader, nn.CrossEntropyLoss()
            )
            _sel = config.selection_metric
            bf1 = metrics[_sel]
            star = ""
            if bf1 > best_metric:
                best_metric = bf1
                best_state_dict = copy.deepcopy(model.state_dict())
                checkpoint_label = label
                star = " *"
            # Track cycle-level patience
            if bf1 > es_best_metric:
                es_best_metric = bf1
                es_counter = 0
            else:
                es_counter += 1
            _log(f"      val_{_sel}={bf1:.4f} @ {label}{star}")

        # -> Phase II-a: Warm-up on seed set --
        seed_ds = TensorDataset(al_train_data, al_train_labels)
        seed_loader = DeviceDataLoader(
            DataLoader(seed_ds, batch_size=config.batch_size, shuffle=True),
            device,
        )
        for epoch in range(warmup):
            trainer._train_epoch(model, seed_loader, optimizer, criterion)
        _log(f"    Warm-up done ({warmup} epochs)")
        _maybe_checkpoint("warmup")

        # -> Phase II-b: Active-learning drain loop --
        for cycle in range(num_al_cycles):
            if len(remaining_data) == 0:
                break

            n_acq = min(config.ab9_samples_per_cycle, len(remaining_data))

            # Score remaining pool by margin
            model.eval()
            with torch.no_grad():
                all_margins = []
                for i in range(0, len(remaining_data), config.batch_size):
                    batch = remaining_data[i:i + config.batch_size].to(device)
                    logits = model(batch)
                    probs = torch.softmax(logits, dim=1)
                    sorted_probs, _ = probs.sort(dim=1, descending=True)
                    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
                    all_margins.append(margins.cpu())
                all_margins = torch.cat(all_margins)

            # Acquire lowest-margin samples
            _, acq_idx = torch.topk(all_margins, k=n_acq, largest=False)

            al_train_data = torch.cat(
                [al_train_data, remaining_data[acq_idx]]
            )
            al_train_labels = torch.cat(
                [al_train_labels, remaining_labels[acq_idx]]
            )

            keep_mask = torch.ones(len(remaining_data), dtype=torch.bool)
            keep_mask[acq_idx] = False
            remaining_data = remaining_data[keep_mask]
            remaining_labels = remaining_labels[keep_mask]

            # Train on grown set
            grown_ds = TensorDataset(al_train_data, al_train_labels)
            grown_loader = DeviceDataLoader(
                DataLoader(
                    grown_ds, batch_size=config.batch_size, shuffle=True,
                ),
                device,
            )
            model.train()
            for epoch in range(epochs_per_cycle):
                trainer._train_epoch(
                    model, grown_loader, optimizer, criterion
                )

            if (cycle + 1) % 5 == 0 or cycle == num_al_cycles - 1:
                _log(f"    AL cycle {cycle+1}/{num_al_cycles}: "
                      f"train={len(al_train_data)}, "
                      f"pool={len(remaining_data)}")

            _maybe_checkpoint(f"cycle-{cycle+1}")

            # -> Cycle-level early stopping --
            if retrain_patience is not None and es_counter >= retrain_patience:
                _log(f"    Ab-9 AL early stopping at cycle {cycle+1} "
                      f"(patience={retrain_patience})")
                break

        # -> Phase III: Consolidation on consumed set --
        consol_ds = TensorDataset(al_train_data, al_train_labels)
        consol_loader = DeviceDataLoader(
            DataLoader(
                consol_ds, batch_size=config.batch_size, shuffle=True,
            ),
            device,
        )
        for epoch in range(consolidation):
            trainer._train_epoch(model, consol_loader, optimizer, criterion)

        _log(f"    Consolidation done ({consolidation} epochs)")
        _maybe_checkpoint("consolidation")

    finally:
        trainer.generator.train(mode=gen_was_training)

    # -> Restore best checkpoint (525J) --
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        _log(f"    Best checkpoint: {checkpoint_label} "
              f"(macro_f1={best_metric:.4f})")

    _log(f"  Retrain complete.\n")
    return model


def _print_fold_progress(
    fold_idx: int,
    n_folds: int,
    mode: str,
    metrics: Dict[str, float],
):
    """Print formatted progress update after a fold completes.
    
    Shows key metrics (F1, G-mean, AUC) for quick monitoring
    during long-running CV experiments.
    
    Output format:
        ->Fold 1/5 [AB4] ->F1: 0.847  G-mean: 0.847  AUC: 0.951
    
    Args:
        fold_idx: Current fold index (0-based, displayed as 1-based)
        n_folds: Total number of folds
        mode: Ablation mode (displayed uppercase)
        metrics: Complete raw metrics dict from train_fold()
    """
    f1 = metrics.get('binary_f1', 0.0)
    gmean = metrics.get('binary_gmean', 0.0)
    auc = metrics.get('binary_auc', 0.0)
    
    print(
        f"    ->Fold {fold_idx + 1}/{n_folds} [{mode.upper()}] ->"
        f"F1: {f1:.4f}  G-mean: {gmean:.4f}  AUC: {auc:.4f}"
    )

