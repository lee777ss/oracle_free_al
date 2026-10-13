"""
Unified classifier training framework supporting all augmentation modes.

Replaces 5 separate training functions (train_classifier_ab1, ab1a, ab4, ab5, ab6)
from 427J with a single mode-based ClassifierTrainer class.

427J had separate functions like:
    train_classifier_ab1()   ->imbalanced baseline
    train_classifier_ab1a()  ->classic oversampling
    train_classifier_on_bagan() / train_classifier_ab4() ->fixed BAGAN
    train_classifier_ab5()   ->dynamic BAGAN (fresh samples per epoch)
    train_classifier_ab6()   ->BAGAN + active learning (margin)

All of them shared the same training loop pattern:
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss, metrics = validate(...)
        early_stopping(model, val_loss)

The ONLY difference between ablation modes was "HOW training data was prepared":
    Ab-1:  Use raw imbalanced data (no change)
    Ab-1a: oversample_minority_classes() ->duplicate real minority samples
    Ab-4:  generate_bagan_augmented_dataset() ONCE before training loop
    Ab-5:  class-stratified batching with fresh BAGAN per batch (w/o replacement)
    Ab-6:  generate_bagan_al_augmented_dataset() EACH EPOCH (quality gate + margin)

This module unifies all 5 into one ClassifierTrainer class where only
_prepare_training_data() differs per mode.
"""

import copy
import math
from typing import Tuple, Dict, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from .models import CNNClassifier, DeviceDataLoader, EarlyStopping, to_device
from .metrics import compute_metrics
from .metrics_binary import compute_metrics_binary
from .augmentation import (
    _compute_target_counts,
    oversample_minority_classes,
    augment_with_real_reserve,
    generate_bagan_augmented_dataset,
)
from .selection import select_indices, describe as describe_selection


def _ab8c_seed_latents(latent_generator, target_class: int, n_samples: int, cfg,
                        mode: str = 'ab8c'):
    """Seed candidate latents under ``cfg.{mode}_seed_mode``.

    Shared by the CV loop (``trainer.py``) and the retrain loop (``kfold.py``)
    so callers cannot diverge. ``mode`` selects which config knob is read:
    Ab-8c (default) uses ``ab8c_seed_mode`` / ``ab8c_latent_smote_noise``;
    Ab-8d uses ``ab8d_seed_mode`` / ``ab8d_latent_smote_noise`` (132J - Variant
    2 now shares latent-SMOTE seeding with Proposed, so the Variant2 ->
    Proposed comparison isolates the backbone + co-training only). Modes with
    no dedicated ``{mode}_seed_mode`` field (Ab-8/8a/8b) fall back to
    ``"prior"``. ``"latent_smote"`` matches the case-study implementation;
    ``"prior"`` draws from the fitted class-conditional prior. Falls back to
    the prior if the fitted generator predates the ``sample_latent_smote`` API.
    """
    seed_mode = getattr(cfg, f'{mode}_seed_mode', 'prior')
    if seed_mode == 'latent_smote' and hasattr(latent_generator, 'sample_latent_smote'):
        return latent_generator.sample_latent_smote(
            target_class=target_class,
            n_samples=n_samples,
            noise_frac=getattr(cfg, f'{mode}_latent_smote_noise', 0.25),
        )
    return latent_generator.sample_for_class_oversampling(
        target_class=target_class, n_samples=n_samples,
    )

# All valid ablation mode identifiers
# ab20: Commit A variant of ab5 ->same per-batch dynamic-BAGAN training,
#       but BAGAN components come from the BAGAN-GP supervised autoencoder
#       (loaded by the caller from saved/bagan_gp_*_107J.pth).
# ab8d: Ab-8b with the BADGE selector replaced by the confidence-gated
#       diverse rule (isolates the selector swap; no co-training).
VALID_MODES = {'ab1', 'ab1f', 'ab1a', 'ab1b', 'ab4', 'ab5', 'ab6', 'ab7', 'ab8', 'ab8a', 'ab8b', 'ab8c', 'ab8d', 'ab8b_bagan', 'ab8b_without_gp', 'ab9', 'ab10', 'ab11', 'ab11a', 'ab20'}

# Modes that require a pre-trained BAGAN generator + latent_generator
BAGAN_MODES = {'ab4', 'ab5', 'ab6', 'ab7', 'ab8', 'ab8a', 'ab8b', 'ab8c', 'ab8d', 'ab8b_bagan', 'ab8b_without_gp', 'ab9', 'ab10', 'ab11', 'ab11a', 'ab20'}


class ClassifierTrainer:
    """Unified classifier training with mode-based augmentation.
    
    How the class works:
    --------------------
    Think of this as a "training recipe" that you configure once and
    then call train_fold() for each k-fold split. The mode parameter
    determines which data augmentation strategy is used.
    
    The training flow for each fold (train_fold method):
    
        1. PREPARE DATA ->_prepare_training_data() dispatches to the right
           augmentation function based on self.mode:
           
           Mode     ->Augmentation function           ->When called
           --
           'ab1'    ->None (use raw data)              ->Once before loop
           'ab1a'   ->oversample_minority_classes()    ->Once before loop
           'ab4'    ->generate_bagan_augmented_dataset()->Once before loop
           'ab5'    ->_train_epoch_ab5() stratified     ->Every batch
           'ab6'    ->_train_epoch_ab6() quality-gated    ->Every batch
           'ab7'    ->_train_epoch_ab7() margin-sampled   ->Every batch
           'ab10'   ->_train_epoch_ab10() ab7 + G finetune->Every batch
           
        2. CREATE DATALOADER ->wrap data in TensorDataset ->DataLoader ->
           DeviceDataLoader (auto-moves batches to GPU)
           
        3. TRAINING LOOP ->for each epoch:
           a. _train_epoch(): forward ->cross-entropy loss ->backward ->step
           b. _validate(): compute val_loss + all metrics (F1, precision, etc.)
           c. EarlyStopping: save best model, stop if no improvement
              for `patience` epochs
           
        4. RETURN ->(best_model, final_metrics)
    
    Modes:
        'ab1'  ->Imbalanced baseline (no augmentation)
                 Uses raw training data as-is with its natural class imbalance.
                 Purpose: Establish worst-case baseline performance.
                 
        'ab1a' ->Classic oversampling (balanced with real samples)
                 Duplicates real minority samples to reach target distribution.
                 No synthetic data. Purpose: Show effect of balancing alone.
                 
        'ab4'  ->BAGAN augmentation (fixed augmented dataset)
                 Generates BAGAN synthetic minority samples ONCE before training.
                 Same synthetic samples used in every epoch.
                 Purpose: Test if BAGAN samples are useful for augmentation.
                 
        'ab5'  ->Dynamic BAGAN (class-stratified per-batch augmentation)
                 Uses class-stratified WITHOUT-REPLACEMENT batching:
                 majority is shuffled and consumed sequentially; minority
                 cycles through all real samples across batches with fresh
                 BAGAN top-up per batch. 100% real data coverage per epoch.
                 Purpose: Test if sample diversity improves over fixed Ab-4.
                 
        'ab6'  ->BAGAN dynamic + quality gate
                 Like Ab-5 (class-stratified per-batch) but BAGAN top-up is
                 overproduced (config.overproduction_factor x) and filtered
                 by discriminator P(fake) ->keeps most realistic candidates.
                 Purpose: Test if quality-filtered synthesis beats random.
                 
        'ab7'  ->BAGAN dynamic + margin sampling (active learning)
                 Like Ab-5 (class-stratified per-batch) but BAGAN top-up is
                 overproduced (config.margin_overproduction_factor x) and
                 filtered by the classifier's own margin score
                 (P(top-1) - P(top-2)). Keeps most uncertain candidates.
                 Purpose: Test if classifier-driven selection beats random.
                 
        'ab10' ->BAGAN-classifier co-training (ab7 + separate G step)
                 Identical to Ab-7 (margin-sampled per-batch) for classifier
                 training, but adds a SEPARATE generator fine-tuning step
                 after each classifier batch: fresh z ->G(z) ->CE(C(G(z)),
                 target) ->backward through G ->g_optimizer.step().
                 G receives class-discriminative signal from the updated
                 classifier, raising the quality floor of overproduced
                 candidates. See 06_Plan for Ab-10.ipynb.
                 Purpose: Test if co-training G improves margin selection.
    
    Args:
        config: AblationConfig instance (hyperparameters, class setup)
        mode: Ablation mode identifier ->one of 'ab1','ab1a','ab4','ab5','ab6','ab7','ab10'
        generator: Pre-trained BAGAN generator (required for ab4/ab5/ab6/ab7/ab10)
        latent_generator: Pre-trained latent space generator (required for ab4/ab5/ab6/ab7/ab10)
        discriminator: Pre-trained BAGAN discriminator (required for ab6 quality gate).
                       Outputs (n+1) logits; last = P(fake). Used to filter
                       overproduced BAGAN candidates per batch.
        device: Device for training ('cuda' or 'cpu')
    
    Example:
        >>> trainer = ClassifierTrainer(
        ...     config=config,
        ...     mode='ab4',
        ...     generator=bagan_gen,
        ...     latent_generator=latent_gen,
        ...     device='cuda'
        ... )
        >>> model, metrics, val_logits, val_labels = trainer.train_fold(
        ...     train_data, train_labels, val_data, val_labels
        ... )
        >>> print(f"Val F1: {metrics['macro_f1']:.3f}")
    """
    
    def __init__(
        self,
        config,
        mode: str,
        generator=None,
        latent_generator=None,
        discriminator=None,
        reserve_data=None,
        reserve_labels=None,
        device: str = 'cuda'
    ):
        """Initialize classifier trainer with mode and components.
        
        Stores all configuration and pre-trained models needed for training.
        Validates that the mode is recognized and that required BAGAN
        components are provided for modes that need them.
        
        Args:
            config: AblationConfig ->contains num_epochs, batch_size,
                    learning_rate, patience, num_classes, etc.
            mode: Which ablation to run ->determines augmentation strategy.
            generator: BAGANGenerator ->unconditional generator that takes
                       latent vector z and outputs fake images [B,1,64,64].
                       Required for ab4, ab5, ab6, ab7. None for ab1, ab1a.
            latent_generator: ClassConditionalLatentGenerator ->has fitted
                              per-class distributions N(_c, ?I) and method
                              sample_for_class_oversampling(class, n_samples).
                              Required for ab4, ab5, ab6, ab7. None for ab1, ab1a.
            discriminator: BAGANDiscriminator ->outputs (n+1) logits, used in
                           Ab-6 quality gate to filter overproduced BAGAN
                           candidates per batch by P(fake).
                           Required for ab6. None for other modes.
            device: 'cuda' or 'cpu' for training.
        
        Raises:
            ValueError: If mode is not recognized, or BAGAN components
                        are missing when required.
        """
        # -> Validate mode --
        if mode not in VALID_MODES:
            raise ValueError(
                f"Unknown mode '{mode}'. Must be one of {sorted(VALID_MODES)}"
            )
        
        # -> Validate BAGAN dependencies --
        # Ab-4, Ab-5, Ab-6 all need BAGAN generator + latent_generator
        # because they generate synthetic minority samples.
        # Ab-1 and Ab-1a don't use BAGAN at all.
        if mode in BAGAN_MODES:
            if generator is None:
                raise ValueError(
                    f"Mode '{mode}' requires a pre-trained BAGAN generator"
                )
            if latent_generator is None:
                raise ValueError(
                    f"Mode '{mode}' requires a pre-trained latent_generator"
                )
        
        # Ab-6 needs discriminator for quality gate; Ab-11/Ab-11a need it for D-adversarial loss
        if mode in ('ab6', 'ab11', 'ab11a') and discriminator is None:
            raise ValueError(
                f"Mode '{mode}' requires a pre-trained discriminator"
            )
        
        # Ab-1b needs a reserve pool (disjoint real images for augmentation)
        if mode == 'ab1b' and reserve_data is None:
            raise ValueError(
                "Mode 'ab1b' requires reserve_data and reserve_labels "
                "(computed from pool_indices by run_kfold_ablation)"
            )
        
        # -> Store everything --
        self.config = config
        self.mode = mode
        self.generator = generator
        self.latent_generator = latent_generator
        self.discriminator = discriminator
        self.reserve_data = reserve_data
        self.reserve_labels = reserve_labels
        self.device = device
        
        # -> Build label ->index mappings --
        # 
        # WHY THIS IS NEEDED:
        # 511J uses non-contiguous class labels: [0, 2, 3]
        # But CrossEntropyLoss requires targets in range [0, num_classes-1].
        # With num_classes=3, valid indices are [0, 1, 2].
        # Passing label=3 directly ->"IndexError: Target 3 is out of bounds".
        #
        # Solution: Build a bidirectional mapping:
        #   label_to_index: {0->, 2->, 3->}  (for feeding to CrossEntropyLoss)
        #   index_to_label: {0->, 1->, 2->}  (for converting predictions back)
        #
        # 427J didn't need this because it used binary classes [0, 1] which
        # happen to be contiguous. This is a 511J-specific requirement.
        #
        # The mapping is built from config.selected_classes which is always
        # sorted (e.g., [0, 2, 3]), so index 0 = first class, etc.
        self.label_to_index = {
            c: i for i, c in enumerate(config.selected_classes)
        }
        self.index_to_label = {
            i: c for i, c in enumerate(config.selected_classes)
        }
        
        # -> Ab-10/Ab-11: epoch counter for warmup gating --
        self._current_epoch = 0
        
        # -> Ab-10/Ab-11: G optimizer for co-training --
        # When mode='ab10' or 'ab11', the generator is fine-tuned alongside
        # the classifier.  We create a separate optimizer for G here
        # so it persists across epochs (retains momentum/Adam state).
        self.g_optimizer = None
        if mode in ('ab10', 'ab11', 'ab11a') and generator is not None:
            if config.ab10_g_optimizer == 'sgd':
                self.g_optimizer = torch.optim.SGD(
                    generator.parameters(),
                    lr=config.ab10_generator_lr
                )
            else:  # 'adam'
                self.g_optimizer = torch.optim.Adam(
                    generator.parameters(),
                    lr=config.ab10_generator_lr,
                    betas=(0.5, 0.999)  # Match original GAN training betas
                )

        # -> 110J: backend-adapter helpers (BAGAN ->BAGAN-GP) --
        # `_realism_score`, `_adv_class_logits`, `_disc_features` translate
        # the discriminator's native output shape into the three abstract
        # signals trainer.py needs (lower-is-better realism, CE-friendly
        # class logits, flattened penultimate features). The vanilla 'bagan'
        # implementations are bit-equivalent to the inline code that 106J
        # used, so existing BAGAN runs are unchanged.
        from .gan_backend import get_gan_modules
        _gan = get_gan_modules(getattr(config, 'gan_backend', 'bagan'))
        self._realism_score = _gan.realism_score
        self._adv_class_logits = _gan.adv_class_logits
        self._disc_features = _gan.discriminator_features
    
    def train_fold(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: torch.Tensor,
        val_labels: torch.Tensor
    ) -> Tuple[CNNClassifier, Dict[str, float], np.ndarray, np.ndarray]:
        """Train classifier for one K-fold split.
        
        This is the main entry point. It runs the complete training pipeline:
        
        Step 1 ->PREPARE DATA:
            Call _prepare_training_data() based on mode.
            For ab1/ab1a/ab4: done once before the loop.
            For ab5/ab6: handled entirely by _train_epoch_ab5/ab6() per batch.
        
        Step 2 ->CREATE MODEL + OPTIMIZER + EARLY STOPPING:
            Fresh CNNClassifier, Adam optimizer, and EarlyStopping tracker.
            These are created new for each fold to ensure independence.
        
        Step 3 ->TRAINING LOOP (num_epochs iterations):
            a. For ab5: _train_epoch_ab5() does class-stratified batching
               For ab6: _train_epoch_ab6() does ab5 + quality gate
            b. _train_epoch(): one pass through all training batches
            c. _validate(): compute val_loss + metrics on val set
            d. EarlyStopping: check if we should stop
        
        Step 4 ->RETURN RESULTS:
            Return (model_with_best_weights, final_validation_metrics).
        
        Why fresh model per fold?
            In K-fold CV, each fold must be independent. If we reused
            the model from fold 1, fold 2's results would be contaminated.
            Fresh initialization ensures each fold trains from scratch.
        
        Args:
            train_data: Training images for this fold [N_train, 1, 64, 64]
            train_labels: Training labels [N_train] ->original class labels
                          (e.g., values in {0, 2, 3}, NOT remapped)
            val_data: Validation images [N_val, 1, 64, 64]
            val_labels: Validation labels [N_val]
        
        Returns:
            model: CNNClassifier ->trained model with best weights
                   (restored by EarlyStopping to the epoch with best binary_f1)
            metrics: Dict[str, float] ->validation metrics at best epoch:
                     {accuracy, macro_f1, macro_precision, macro_recall,
                      macro_specificity, macro_gmean, macro_auc,
                      stopping_epoch (int): the epoch number that achieved
                      the best binary_f1 ->used later to set retrain duration}
            best_val_logits: np.ndarray [N_val, num_classes] ->raw pre-softmax
                             logits at the best epoch on the validation set.
                             Used by Topic 7 (calibration / decision-rule
                             correction) without re-running forward passes.
            best_val_labels: np.ndarray [N_val] ->remapped contiguous labels
                             {0..num_classes-1} paired with best_val_logits.
        """
        # -> Ab-11a: clear per-fold caches --
        # `_real_minority_pool` and `_fm_real_ptr` are populated lazily
        # by `_train_epoch_ab11a` on its first call from this fold's data.
        # They MUST be cleared here so a previous fold's pool can't leak
        # into the next fold's FM real-anchor sampling.
        # Same goes for the per-epoch diagnostics dict, the per-fold
        # history of monitor metrics (Step E / Step H), the best-G
        # snapshot kept by the FID monitor, and its running best-FID scalar.
        if self.mode == 'ab11a':
            for attr in ('_real_minority_pool', '_fm_real_ptr',
                         '_ab11a_last_epoch_terms',
                         'fold_history', 'best_g_state_dict',
                         '_best_fid'):
                if hasattr(self, attr):
                    delattr(self, attr)
        # `fold_history` must exist for ALL modes ->`train_fold` unconditionally
        # exports `best_metrics['fold_history'] = dict(self.fold_history)` at
        # the end. Non-ab11a modes simply leave the dict empty.
        self.fold_history = {}

        # -> Step 1: Data preparation (for non-dynamic modes) --
        # Ab-1, Ab-1a, Ab-4: augment ONCE before training loop.
        # Ab-5, Ab-6: skip here ->they re-augment every epoch inside the loop.
        #
        # _is_dynamic_mode() tells us whether to refresh data each epoch.
        if not self._is_dynamic_mode():
            aug_data, aug_labels = self._prepare_training_data(
                train_data, train_labels
            )
            # Remap labels to contiguous indices for CrossEntropyLoss
            aug_labels = self._remap_labels(aug_labels)
        
        # -> Step 2: Create fresh model, optimizer, early stopping --
        #
        # CNNClassifier: 4-layer strided CNN ->FC head ->num_classes logits
        # See models.py for architecture details.
        model = CNNClassifier(
            input_channels=1,                   # Grayscale images
            num_classes=self.config.num_classes   # 3 for 511J
        ).to(self.device)
        
        # Adam optimizer with =(0.9, 0.999) ->PyTorch defaults.
        # 427J also used Adam with the same config.
        # learning_rate comes from AblationConfig (default: 0.001).
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay  # L2 regularization (default: 0.0, no decay)
        )
        
        # CrossEntropyLoss: combines LogSoftmax + NLLLoss in one step.
        # Expects raw logits (NOT softmax) + integer class labels.
        # Handles multi-class natively ->no need for one-hot encoding.
        criterion = nn.CrossEntropyLoss()
        
        # EarlyStopping: stop training when the monitored metric plateaus.
        # We monitor binary_f1 (mode='max') ->the F1 score with labels
        # binarized as {0}?egative (majority), {1,2}?ositive (minority).
        # This directly optimises the model for minority detection, which
        # is the 515J evaluation perspective (Chung et al. binary view).
        #
        # Ab-1f is a deliberate penalized baseline: keep FINAL epoch weights
        # and metrics (no best-checkpoint restore). We still instantiate the
        # helper for interface consistency, but bypass early-stop break/
        # improved-gating and never restore best weights for this mode.
        final_epoch_mode = (self.mode == 'ab1f')
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=0,                  # Any improvement counts
            restore_best_weights=not final_epoch_mode,
            mode='max'                    # Maximize binary_f1
        )
        
        # Validation DataLoader ->stays the same for all epochs.
        # We create it once here so we don't recreate it every epoch.
        # Remap val_labels to contiguous indices for CrossEntropyLoss.
        val_labels_idx = self._remap_labels(val_labels)
        val_dataset = TensorDataset(val_data, val_labels_idx)
        val_loader = DeviceDataLoader(
            DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False),
            self.device
        )
        
        # Note: val_labels_idx in the DataLoader are already in remapped
        # space {0,1,2}. _evaluate_epoch() uses these directly ->no need
        # to store original labels separately.
        
        # Track the best metrics (from the epoch with best val_loss)
        best_metrics = {}
        # Cached raw val logits/labels at the best epoch (Topic 7).
        best_val_logits = None
        best_val_labels = None
        
        # -> Ab-8 special dispatch --
        # Ab-8 replaces the entire epoch loop with its own AL cycle loop.
        # It manages warm-up, acquisition, and consolidation internally,
        # and returns the best model + metrics directly.
        if self.mode == 'ab8':
            return self._run_al_loop_ab8(
                train_data, train_labels, val_data, val_labels
                )

        # -> Ab-8a special dispatch --
        # Ab-8a = Ab-8 + BAGAN-GP aux-head label gate on the candidate
        # pool (independent-oracle selector). Same 3-phase outer loop.
        if self.mode == 'ab8a':
            return self._run_al_loop_ab8a(
                train_data, train_labels, val_data, val_labels
                )

        # -> Ab-8b / Ab-8d special dispatch --
        # Ab-8b = Ab-8 with margin ->BADGE selector swap (no aux-gate).
        # Ab-8d = the same loop with the confidence-gated diverse selector,
        # which isolates the acquisition-rule swap with no co-training.
        # Same 3-phase outer loop; step 2a resolves its rule from
        # config.get_selection_params(mode).
        # ab8b_without_gp is a pure alias of ab8b: identical BADGE active-
        # learning loop and ab8b_* hyperparameters. The only difference is
        # that it is intended to run on the vanilla BAGAN backend (no WGAN-GP
        # critic), which is determined by the backend wired in at call time.
        if self.mode in ('ab8b', 'ab8d', 'ab8b_without_gp'):
            return self._run_al_loop_ab8b(
                train_data, train_labels, val_data, val_labels
                )

        # -> Ab-8c special dispatch --
        # Ab-8c = Ab-8b's BADGE acquisition + 3-player co-training (D->)
        # of the WGAN-GP critic, generator, and classifier per AL cycle.
        # Uses a frozen end-of-warm-up snapshot of G as ?_D anchor.
        if self.mode == 'ab8c':
            return self._run_al_loop_ab8c(
                train_data, train_labels, val_data, val_labels
                )

        # -> Ab-8b-on-BAGAN special dispatch --
        # ab8b_bagan = Ab-8b's frozen-generator BADGE acquisition + a
        # read-only BAGAN-discriminator (softmax) acquisition filter. No
        # WGAN-GP, no G-step ->the vanilla-BAGAN counterpart of Ab-8c.
        if self.mode == 'ab8b_bagan':
            return self._run_al_loop_ab8b_bagan(
                train_data, train_labels, val_data, val_labels
                )

        # -> Ab-9 special dispatch --
        # Ab-9 replaces the entire epoch loop with a pool-drain AL cycle.
        # It builds a balanced pool upfront (Phase I), drains it by margin
        # (Phase II with patience-based stopping), and consolidates (Phase III).
        if self.mode == 'ab9':
            return self._run_al_loop_ab9(
                train_data, train_labels, val_data, val_labels
                )
        
        # -> Step 3: Training loop --
        for epoch in range(self.config.num_epochs):
            self._current_epoch = epoch  # ab11 warmup gating
            
            # -> 3a-c. Mode-dependent training step --
            if self.mode in ('ab5', 'ab20'):
                # Ab-5: Class-stratified without-replacement batching
                # Majority samples are shuffled and consumed sequentially;
                # real minority samples cycle across batches (round-robin);
                # each batch is balanced by topping up minority with fresh
                # BAGAN synthetic. 100% real data coverage per epoch.
                #
                # Ab-20 (Commit A): Identical training loop; the BAGAN
                # generator/latent_generator/discriminator passed into the
                # trainer come from the supervised-AE BAGAN-GP pipeline.
                train_loss = self._train_epoch_ab5(
                    model, train_data, train_labels, optimizer, criterion
                )
            elif self.mode == 'ab6':
                # Ab-6: Same per-batch strategy as Ab-5, but each batch's
                # BAGAN top-up is quality-gated: overproduce candidates
                # (config.overproduction_factor x), keep n_synthetic with
                # lowest P(fake) per discriminator.
                train_loss = self._train_epoch_ab6(
                    model, train_data, train_labels, optimizer, criterion
                )
            elif self.mode == 'ab7':
                # Ab-7: Same per-batch strategy as Ab-5, but each batch's
                # BAGAN top-up is filtered by classifier margin sampling:
                # overproduce candidates (config.margin_overproduction_factor x),
                # keep n_synthetic with lowest margin (most uncertain).
                train_loss = self._train_epoch_ab7(
                    model, train_data, train_labels, optimizer, criterion
                )
            elif self.mode == 'ab10':
                # Ab-10: Ab-7 margin selection + separate G fine-tuning step.
                # Phase 1 trains classifier (identical to Ab-7).
                # Phase 2 fine-tunes G so it generates images the
                # already-updated classifier classifies correctly.
                train_loss = self._train_epoch_ab10(
                    model, train_data, train_labels, optimizer, criterion
                )
            elif self.mode == 'ab11':
                # Ab-11: Ab-10 + D-adversarial loss, warmup, throttled G updates.
                # Phase 2 loss: _cls*CE(C(G(z)),y) + _adv*CE(D(G(z)),y)
                train_loss = self._train_epoch_ab11(
                    model, train_data, train_labels, optimizer, criterion
                )
            elif self.mode == 'ab11a':
                # Ab-11a: Ab-11 + Feature Matching + Margin-Selected G (V3).
                # Phase 2 loss: _cls*CE(C(G(z)),y) + _adv*CE(D(G(z)),y)
                #             + _fm*FM(?_real, ?_fake)
                # over an overproduced + margin-selected fake batch.
                # Diagnostics live in self._ab11a_last_epoch_terms.
                train_loss = self._train_epoch_ab11a(
                    model, train_data, train_labels, optimizer, criterion
                )
            else:
                # -> 3b. Create training DataLoader --
                train_dataset = TensorDataset(aug_data, aug_labels)
                train_loader = DeviceDataLoader(
                    DataLoader(
                        train_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=True
                    ),
                    self.device
                )
                
                # -> 3c. Train one epoch --
                train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            
            # -> 3d. Validate --
            # _validate returns (loss, metrics, logits, labels).
            # logits/labels are CPU tensors (torch.cat of .cpu() batches).
            val_loss, metrics, val_logits, val_lbls = self._validate(
                model, val_loader, criterion
            )
            
            # -> 3d-bis. Ab-11a per-epoch diagnostics + FID monitor --
            # (Step E) Print one summary line per fold-epoch and (when
            # `cfg.ab11a_fid_eval_every > 0`) run the cheap FID monitor ->
            # the actual FID compute lives in Step H's `ablation/fid.py`.
            if self.mode == 'ab11a':
                self._log_ab11a_epoch(epoch, train_loss, val_loss, metrics)
                self._maybe_run_fid_monitor(epoch)
            
            # -> 3e. Early stopping check --
            # Pass macro_f1 to EarlyStopping (mode='max').
            # macro_f1 = mean of per-class F1 across {0,1,2}; in a 1-vs-2
            # highly imbalanced setting it weights both minority classes
            # equally instead of collapsing them into one positive class.
            # binary_f1 is still computed and logged for comparison.
            # epoch= is tracked so we know the best epoch for retrain.
            if not final_epoch_mode and early_stopping(
                model, metrics['macro_f1'], epoch=epoch
            ):
                # Patience exhausted ->model weights already restored
                # to the best epoch by EarlyStopping.__call__()
                break
            
            # -> 3f. Track best metrics + cache best-epoch logits --
            # Save metrics AND raw val logits/labels from the epoch with
            # best binary_f1. Logits (pre-softmax) are needed by Topic 7
            # (temperature scaling, prior-shift correction, PR-curve
            # threshold tuning). Caching here avoids an extra forward pass
            # after training and guarantees the logits correspond exactly
            # to the weights EarlyStopping will restore.
            if final_epoch_mode:
                # Ab-1f: track the CURRENT epoch (final overwrite each iter).
                best_metrics = metrics.copy()
                best_val_logits = val_logits.numpy()
                best_val_labels = val_lbls.numpy()
            elif early_stopping.improved:
                best_metrics = metrics.copy()
                best_val_logits = val_logits.numpy()   # already CPU tensor
                best_val_labels = val_lbls.numpy()     # remapped {0..C-1}
        
        # -> Step 4: Return best model + metrics --
        # After the loop, model already has best weights (  opping
        # restored them either when patience ran out, or we manually
        # tracked the best epoch's metrics above).
        #
        # If training never triggered early stopping (ran all epochs),
        # we need to manually restore best weights:
        if (not final_epoch_mode) and early_stopping.best_model is not None:
            model.load_state_dict(early_stopping.best_model)
        
        # If best_metrics is empty (edge case: only 1 epoch ran),
        # use the last computed metrics
        if not best_metrics:
            best_metrics = metrics
        
        # Record the epoch that achieved the best binary_f1.
        # This is averaged across folds to set retrain duration (Step 6).
        if final_epoch_mode:
            # Full-length baseline: report "stopping" at the last epoch.
            best_metrics['stopping_epoch'] = self.config.num_epochs
        else:
            best_metrics['stopping_epoch'] = early_stopping.best_epoch

        # Expose per-epoch curves (g_fm, mean_selected_margin, fid_*, ...)
        # so the Ab-11a smoke run (Step G) and sweep harness (Step I) can
        # inspect Phase-2 dynamics without touching the trainer instance.
        best_metrics['fold_history'] = dict(self.fold_history)

        # -> Edge case: loop ran but `improved` was never True --
        # In practice epoch-0 always sets `improved=True`, but fall back
        # to the last computed logits/labels to be safe.
        if best_val_logits is None:
            best_val_logits = val_logits.numpy()
            best_val_labels = val_lbls.numpy()
        
        return model, best_metrics, best_val_logits, best_val_labels
    
    def _prepare_training_data(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        model: Optional[CNNClassifier] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation strategy based on self.mode.
        
        This is the KEY method that differentiates all 5 ablation modes.
        It dispatches to the appropriate augmentation function from
        augmentation.py based on self.mode.
        
        Dispatch table:
            'ab1'  ->return data unchanged (no augmentation)
            'ab1a' ->oversample_minority_classes()
            'ab4'  ->generate_bagan_augmented_dataset()
            'ab5'  ->NOT dispatched here (handled by _train_epoch_ab5)
            'ab6'  ->NOT dispatched here (handled by _train_epoch_ab6)
            'ab7'  ->NOT dispatched here (handled by _train_epoch_ab7)
        
        Note: Ab-5 and Ab-6 never call this method. Both build balanced batches
        directly inside their respective _train_epoch_*() methods via
        class-stratified without-replacement batching with per-batch BAGAN top-up.
        
        Args:
            train_data: Original (non-augmented) training data [N, 1, 64, 64]
            train_labels: Original training labels [N]
            model: Unused. Kept for interface compatibility.
        
        Returns:
            augmented_data: Training data after augmentation [N_aug, 1, 64, 64]
            augmented_labels: Labels for augmented data [N_aug]
        
        Raises:
            ValueError: If mode is somehow invalid (should never happen
                        since __init__ already validates).
        """
        if self.mode in ('ab1', 'ab1f'):
            # -> Ab-1: No augmentation ->use imbalanced data as-is --
            # The classifier trains on whatever the natural distribution is.
            # e.g., {class 0: 5400, class 2: 270, class 3: 270}
            return train_data, train_labels
        
        elif self.mode == 'ab1a':
            # -> Ab-1a: Classic oversampling --
            # Duplicate real minority samples with replacement until each
            # class reaches target count from config.augmentation_target.
            # No synthetic data involved ->purely random repetition.
            return oversample_minority_classes(
                train_data, train_labels, self.config
            )
        
        elif self.mode == 'ab1b':
            # -> Ab-1b: Augment with real reserve images --
            # Fill minority deficit with UNIQUE real images from a disjoint
            # reserve pool (Fashion-MNIST images not in pool_data).
            # Same target distribution as Ab-1a, but uses fresh real
            # images instead of duplicating training fold samples.
            return augment_with_real_reserve(
                train_data, train_labels,
                self.reserve_data, self.reserve_labels,
                self.config
            )
        
        elif self.mode == 'ab4':
            # -> Ab-4: BAGAN augmentation --
            # For each minority class:
            #   1. Compute deficit = target_count - current_count
            #   2. Sample z from N(_c, ?I) via latent_generator
            #   3. Generate images = generator(z)
            #   4. Concatenate with real data
            #
            # Ab-4: this is called ONCE (from train_fold, before loop)
            # Ab-5: this is called EVERY EPOCH (inside loop) ->fresh z
            return generate_bagan_augmented_dataset(
                train_data, train_labels,
                self.generator, self.latent_generator,
                self.config, self.device
            )
        
        else:
            # Should never reach here ->__init__ already validates
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _train_epoch(
        self,
        model: nn.Module,
        dataloader: DeviceDataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """Train for one epoch ->single pass through all training batches.
        
        Training loop pattern (identical across all 5 ablation modes):
            model.train()  ->enable dropout + batch norm training behavior
            for each batch:
                outputs = model(images)             ->forward pass
                loss = CrossEntropy(outputs, labels) ->compute loss
                optimizer.zero_grad()               ->clear old gradients
                loss.backward()                     ->compute new gradients
                optimizer.step()                    ->update weights
        
        This is the standard PyTorch training loop, unchanged from 427J.
        
        Why zero_grad()?
            PyTorch ACCUMULATES gradients by default (for multi-batch
            gradient accumulation). Since we don't use that technique,
            we must clear gradients before each backward() call.
            Without this, gradients from previous batches would add up.
        
        Args:
            model: CNNClassifier in training mode
            dataloader: DeviceDataLoader wrapping training TensorDataset
            optimizer: Adam optimizer for model parameters
            criterion: CrossEntropyLoss function
        
        Returns:
            avg_loss: Average training loss across all batches.
                      Used for monitoring only (not for EarlyStopping,
                      which uses validation loss instead).
        """
        model.train()   # Enable dropout + use batch norm running stats
        
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in dataloader:
            # images: [B, 1, 64, 64] already on self.device (DeviceDataLoader)
            # labels: [B] integer class labels on self.device
            
            # Forward pass: image ->conv features ->logits
            outputs = model(images)             # [B, num_classes]
            
            # Compute cross-entropy loss
            # CrossEntropyLoss internally applies LogSoftmax + NLLLoss:
            #   loss = -log(softmax(outputs)[correct_class])
            # Lower loss = higher probability assigned to correct class
            loss = criterion(outputs, labels)
            
            # Backward pass:
            optimizer.zero_grad()   # Clear accumulated gradients
            loss.backward()         # Compute gradients via backpropagation
            optimizer.step()        # Update model weights: w -= lr * grad
            
            total_loss += loss.item()   # .item() extracts scalar from tensor
            num_batches += 1
        
        # Return average loss per batch (not per sample)
        return total_loss / max(num_batches, 1)
    
    def _train_epoch_ab5(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Ab-5 class-stratified without-replacement BAGAN augmentation.
        
        Constructs balanced batches by drawing from each class separately,
        guaranteeing 100% real data coverage per epoch:
        
            Setup (once per epoch):
              1. Separate real indices by class
              2. Shuffle majority indices ->consume WITHOUT replacement
              3. Shuffle minority indices ->cycle via round-robin pointer
              4. num_batches = len(majority) // n_per_class
                 (anchored on majority so all majority samples are used)
        
            For each batch:
              1. Majority: take next n_per_class from shuffled majority
              2. Minority (each class): take next k real images (cycling),
                 top up to n_per_class with fresh BAGAN synthetic
              3. Shuffle the balanced batch ->forward ->backward ->step
        
        Compared to the old random-sampling approach:
          - 100% majority used (vs ~39% before ->rest was discarded)
          - 100% minority seen per epoch (vs ~63% with replacement)
          - ~2.5x more gradient updates (30 vs 12 batches for 640:64:64)
          - Each batch still gets fresh BAGAN synthetic samples
        
        Args:
            model: CNNClassifier to train
            train_data: [N, 1, 64, 64] real training images (original labels)
            train_labels: [N] original class labels (NOT yet remapped)
            optimizer: Adam optimizer
            criterion: CrossEntropyLoss
        
        Returns:
            avg_loss: Average training loss across all batches
        """
        model.train()
        
        n_per_class = self.config.batch_size // self.config.num_classes
        majority_class = self.config.majority_class
        
        # -> Step 1: Separate real indices by class --
        # .nonzero(as_tuple=True)[0] returns a 1-D tensor of positions
        # where the condition is True.
        majority_idx = (train_labels == majority_class).nonzero(as_tuple=True)[0]
        minority_idx = {}   # {class_label: 1-D tensor of indices}
        for cls in self.config.minority_classes:
            minority_idx[cls] = (train_labels == cls).nonzero(as_tuple=True)[0]
        
        # -> Step 2: Shuffle majority WITHOUT replacement --
        # After shuffling, we consume these sequentially in chunks of
        # n_per_class. Every majority sample is seen exactly once.
        majority_perm = majority_idx[torch.randperm(len(majority_idx))]
        
        # -> Step 3: Shuffle minority for round-robin cycling --
        # Each minority class has its own shuffled order and a pointer
        # that advances across batches (wrapping around when exhausted).
        minority_perm = {}  # {class_label: shuffled index tensor}
        minority_ptr = {}   # {class_label: int position in perm}
        for cls in self.config.minority_classes:
            idx = minority_idx[cls]
            minority_perm[cls] = idx[torch.randperm(len(idx))]
            minority_ptr[cls] = 0
        
        # -> Step 4: Determine num_batches from majority --
        # Anchored on majority class so ALL majority samples are consumed.
        # e.g. 640 majority / 21 per batch = 30 batches
        num_batches = max(1, len(majority_perm) // n_per_class)
        
        # Pre-compute how many real minority images to include per batch.
        # Distribute as evenly as possible: at least 1 real image per batch.
        # e.g. 64 minority / 30 batches ->base=2, remainder=4
        #      first 4 batches get 3 real, remaining 26 batches get 2 real.
        minority_per_batch_base = {}
        minority_per_batch_extra = {}
        for cls in self.config.minority_classes:
            n_minority = len(minority_perm[cls])
            minority_per_batch_base[cls] = max(1, n_minority // num_batches)
            minority_per_batch_extra[cls] = n_minority % num_batches
        
        # Put generator in eval mode for inference
        was_training = self.generator.training
        self.generator.eval()
        
        total_loss = 0.0

        # explain 'try/finally' here:
        # We use a try/finally block to ensure that we restore the generator's
        # training state at the end of this method, even if an error occurs.
        
        try:
            for batch_i in range(num_batches):
                final_images = []
                final_labels = []
                
                # -> Majority: next n_per_class WITHOUT replacement --
                maj_start = batch_i * n_per_class
                maj_end = min(maj_start + n_per_class, len(majority_perm))
                maj_batch_idx = majority_perm[maj_start:maj_end]
                maj_images = train_data[maj_batch_idx]
                n_maj = len(maj_images)
                remapped_maj = self.label_to_index[majority_class]
                final_images.append(maj_images)
                final_labels.append(torch.full(
                    (n_maj,), remapped_maj, dtype=torch.long
                ))
                
                # -> Minority (each class): cycle real + BAGAN top-up --
                for cls in self.config.minority_classes:
                    remapped_c = self.label_to_index[cls]
                    perm = minority_perm[cls]
                    n_total_minority = len(perm)
                    
                    # How many real minority images for THIS batch
                    base = minority_per_batch_base[cls]
                    extra = minority_per_batch_extra[cls]
                    n_real = base + (1 if batch_i < extra else 0)
                    
                    # Gather real minority via cycling pointer
                    real_indices = []
                    ptr = minority_ptr[cls]
                    for _ in range(n_real):
                        real_indices.append(perm[ptr % n_total_minority].item())
                        ptr += 1
                    minority_ptr[cls] = ptr
                    
                    # Add real minority images to batch
                    real_idx_tensor = torch.tensor(real_indices, dtype=torch.long)
                    real_images = train_data[real_idx_tensor]
                    final_images.append(real_images)
                    final_labels.append(torch.full(
                        (n_real,), remapped_c, dtype=torch.long
                    ))
                    
                    # BAGAN top-up: fill remaining slots with fresh synthetic
                    n_synthetic = n_per_class - n_real
                    if n_synthetic > 0:
                        z = self.latent_generator.sample_for_class_oversampling(
                            target_class=cls, n_samples=n_synthetic
                        )
                        z = z.to(self.device)
                        with torch.no_grad():
                            synthetic = self.generator(z).cpu()
                        final_images.append(synthetic)
                        final_labels.append(torch.full(
                            (n_synthetic,), remapped_c, dtype=torch.long
                        ))
                
                # -> Concatenate, shuffle, train --
                batch_images = torch.cat(final_images, dim=0)
                batch_labels = torch.cat(final_labels, dim=0)
                
                perm = torch.randperm(batch_images.size(0))
                batch_images = batch_images[perm].to(self.device)
                batch_labels = batch_labels[perm].to(self.device)
                
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        finally:
            # Restore generator training state
            self.generator.train(mode=was_training)
        
        return total_loss / max(num_batches, 1)
    
    def _train_epoch_ab6(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Ab-6 class-stratified BAGAN augmentation with quality gate.
        
        Identical to Ab-5 (class-stratified without-replacement batching)
        except the BAGAN top-up for each minority class in each batch is
        quality-gated:
        
            1. Overproduce: generate overproduction_factor x n_synthetic
               candidates via BAGAN latent sampling.
            2. Score each candidate with discriminator ->P(fake).
            3. Keep the n_synthetic candidates with LOWEST P(fake)
               (most realistic).
        
        This filters out obvious artifacts before they enter training,
        giving the classifier higher-quality synthetic minority examples
        without any additional classifier-in-the-loop cost.
        
        Everything else ->majority consumption, minority cycling, batch
        shuffling, gradient step ->is identical to _train_epoch_ab5().
        
        Args:
            model: CNNClassifier to train
            train_data: [N, 1, 64, 64] real training images (original labels)
            train_labels: [N] original class labels (NOT yet remapped)
            optimizer: Adam optimizer
            criterion: CrossEntropyLoss
        
        Returns:
            avg_loss: Average training loss across all batches
        """
        model.train()
        
        n_per_class = self.config.batch_size // self.config.num_classes
        majority_class = self.config.majority_class
        overproduction = self.config.overproduction_factor
        
        # -> Step 1: Separate real indices by class --
        majority_idx = (train_labels == majority_class).nonzero(as_tuple=True)[0]
        minority_idx = {}
        for cls in self.config.minority_classes:
            minority_idx[cls] = (train_labels == cls).nonzero(as_tuple=True)[0]
        
        # -> Step 2: Shuffle majority WITHOUT replacement --
        majority_perm = majority_idx[torch.randperm(len(majority_idx))]
        
        # -> Step 3: Shuffle minority for round-robin cycling --
        minority_perm = {}
        minority_ptr = {}
        for cls in self.config.minority_classes:
            idx = minority_idx[cls]
            minority_perm[cls] = idx[torch.randperm(len(idx))]
            minority_ptr[cls] = 0
        
        # -> Step 4: Determine num_batches from majority --
        num_batches = max(1, len(majority_perm) // n_per_class)
        
        # Pre-compute real minority images per batch
        minority_per_batch_base = {}
        minority_per_batch_extra = {}
        for cls in self.config.minority_classes:
            n_minority = len(minority_perm[cls])
            minority_per_batch_base[cls] = max(1, n_minority // num_batches)
            minority_per_batch_extra[cls] = n_minority % num_batches
        
        # Put generator + discriminator in eval mode for inference
        gen_was_training = self.generator.training
        disc_was_training = self.discriminator.training
        self.generator.eval()
        self.discriminator.eval()
        
        total_loss = 0.0
        
        try:
            for batch_i in range(num_batches):
                final_images = []
                final_labels = []
                
                # -> Majority: next n_per_class WITHOUT replacement --
                maj_start = batch_i * n_per_class
                maj_end = min(maj_start + n_per_class, len(majority_perm))
                maj_batch_idx = majority_perm[maj_start:maj_end]
                maj_images = train_data[maj_batch_idx]
                n_maj = len(maj_images)
                remapped_maj = self.label_to_index[majority_class]
                final_images.append(maj_images)
                final_labels.append(torch.full(
                    (n_maj,), remapped_maj, dtype=torch.long
                ))
                
                # -> Minority (each class): cycle real + quality-gated BAGAN ?
                for cls in self.config.minority_classes:
                    remapped_c = self.label_to_index[cls]
                    perm = minority_perm[cls]
                    n_total_minority = len(perm)
                    
                    # How many real minority images for THIS batch
                    base = minority_per_batch_base[cls]
                    extra = minority_per_batch_extra[cls]
                    n_real = base + (1 if batch_i < extra else 0)
                    
                    # Gather real minority via cycling pointer
                    real_indices = []
                    ptr = minority_ptr[cls]
                    for _ in range(n_real):
                        real_indices.append(perm[ptr % n_total_minority].item())
                        ptr += 1
                    minority_ptr[cls] = ptr
                    
                    # Add real minority images to batch
                    real_idx_tensor = torch.tensor(real_indices, dtype=torch.long)
                    real_images = train_data[real_idx_tensor]
                    final_images.append(real_images)
                    final_labels.append(torch.full(
                        (n_real,), remapped_c, dtype=torch.long
                    ))
                    
                    # -> Quality-gated BAGAN top-up --
                    # Overproduce candidates, keep most realistic ones.
                    n_synthetic = n_per_class - n_real
                    if n_synthetic > 0:
                        n_candidates = n_synthetic * overproduction
                        
                        z = self.latent_generator.sample_for_class_oversampling(
                            target_class=cls, n_samples=n_candidates
                        )
                        z = z.to(self.device)
                        with torch.no_grad():
                            candidates = self.generator(z)
                            # candidates: [n_candidates, 1, 64, 64] on device

                            # Score with discriminator ->"lower = more real".
                            # BAGAN backend returns P(fake)=softmax(.)[:, -1];
                            # BAGAN-GP returns -critic_score. Either way,
                            # topk(..., largest=False) keeps the most realistic.
                            p_fake = self._realism_score(self.discriminator, candidates)

                            # Keep n_synthetic with LOWEST score (most realistic)
                            _, best_idx = torch.topk(
                                p_fake, k=n_synthetic, largest=False
                            )
                            synthetic = candidates[best_idx].cpu()
                        
                        final_images.append(synthetic)
                        final_labels.append(torch.full(
                            (n_synthetic,), remapped_c, dtype=torch.long
                        ))
                
                # -> Concatenate, shuffle, train --
                batch_images = torch.cat(final_images, dim=0)
                batch_labels = torch.cat(final_labels, dim=0)
                
                perm = torch.randperm(batch_images.size(0))
                batch_images = batch_images[perm].to(self.device)
                batch_labels = batch_labels[perm].to(self.device)
                
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        finally:
            # Restore generator and discriminator training state
            self.generator.train(mode=gen_was_training)
            self.discriminator.train(mode=disc_was_training)
        
        return total_loss / max(num_batches, 1)

    def _train_epoch_ab7(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Ab-7 class-stratified BAGAN augmentation with margin sampling.
        
        Identical to Ab-5 (class-stratified without-replacement batching)
        except the BAGAN top-up for each minority class in each batch is
        filtered by the classifier's own uncertainty:
        
            1. Overproduce: generate margin_overproduction_factor x n_synthetic
               candidates via BAGAN latent sampling.
            2. Temporarily switch classifier to eval mode (disable dropout).
            3. Score each candidate with classifier ->softmax probabilities.
            4. Compute margin = P(top-1) ->P(top-2) for each candidate.
            5. Keep the n_synthetic candidates with LOWEST margin
               (most uncertain / most informative).
            6. Restore classifier to training mode.
        
        This implements active learning?tyle sample selection: the classifier
        sees the synthetic examples it is most confused about, creating a
        natural curriculum that adapts each epoch as the classifier improves.
        
        Everything else ->majority consumption, minority cycling, batch
        shuffling, gradient step ->is identical to _train_epoch_ab5().
        
        Args:
            model: CNNClassifier to train
            train_data: [N, 1, 64, 64] real training images (original labels)
            train_labels: [N] original class labels (NOT yet remapped)
            optimizer: Adam optimizer
            criterion: CrossEntropyLoss
        
        Returns:
            avg_loss: Average training loss across all batches
        """
        model.train()
        
        n_per_class = self.config.batch_size // self.config.num_classes
        majority_class = self.config.majority_class
        overproduction = self.config.margin_overproduction_factor
        
        # -> Step 1: Separate real indices by class --
        majority_idx = (train_labels == majority_class).nonzero(as_tuple=True)[0]
        minority_idx = {}
        for cls in self.config.minority_classes:
            minority_idx[cls] = (train_labels == cls).nonzero(as_tuple=True)[0]
        
        # -> Step 2: Shuffle majority WITHOUT replacement --
        majority_perm = majority_idx[torch.randperm(len(majority_idx))]
        
        # -> Step 3: Shuffle minority for round-robin cycling --
        minority_perm = {}
        minority_ptr = {}
        for cls in self.config.minority_classes:
            idx = minority_idx[cls]
            minority_perm[cls] = idx[torch.randperm(len(idx))]
            minority_ptr[cls] = 0
        
        # -> Step 4: Determine num_batches from majority --
        num_batches = max(1, len(majority_perm) // n_per_class)
        
        # Pre-compute real minority images per batch
        minority_per_batch_base = {}
        minority_per_batch_extra = {}
        for cls in self.config.minority_classes:
            n_minority = len(minority_perm[cls])
            minority_per_batch_base[cls] = max(1, n_minority // num_batches)
            minority_per_batch_extra[cls] = n_minority % num_batches
        
        # Put generator in eval mode for inference
        gen_was_training = self.generator.training
        self.generator.eval()
        
        total_loss = 0.0
        
        try:
            for batch_i in range(num_batches):
                final_images = []
                final_labels = []
                
                # -> Majority: next n_per_class WITHOUT replacement --
                maj_start = batch_i * n_per_class
                maj_end = min(maj_start + n_per_class, len(majority_perm))
                maj_batch_idx = majority_perm[maj_start:maj_end]
                maj_images = train_data[maj_batch_idx]
                n_maj = len(maj_images)
                remapped_maj = self.label_to_index[majority_class]
                final_images.append(maj_images)
                final_labels.append(torch.full(
                    (n_maj,), remapped_maj, dtype=torch.long
                ))
                
                # -> Minority (each class): cycle real + margin-sampled BAGAN ?
                for cls in self.config.minority_classes:
                    remapped_c = self.label_to_index[cls]
                    perm = minority_perm[cls]
                    n_total_minority = len(perm)
                    
                    # How many real minority images for THIS batch
                    base = minority_per_batch_base[cls]
                    extra = minority_per_batch_extra[cls]
                    n_real = base + (1 if batch_i < extra else 0)
                    
                    # Gather real minority via cycling pointer
                    real_indices = []
                    ptr = minority_ptr[cls]
                    for _ in range(n_real):
                        real_indices.append(perm[ptr % n_total_minority].item())
                        ptr += 1
                    minority_ptr[cls] = ptr
                    
                    # Add real minority images to batch
                    real_idx_tensor = torch.tensor(real_indices, dtype=torch.long)
                    real_images = train_data[real_idx_tensor]
                    final_images.append(real_images)
                    final_labels.append(torch.full(
                        (n_real,), remapped_c, dtype=torch.long
                    ))
                    
                    # -> Margin-sampled BAGAN top-up --
                    # Overproduce candidates, keep most uncertain ones.
                    n_synthetic = n_per_class - n_real
                    if n_synthetic > 0:
                        n_candidates = n_synthetic * overproduction
                        
                        z = self.latent_generator.sample_for_class_oversampling(
                            target_class=cls, n_samples=n_candidates
                        )
                        z = z.to(self.device)
                        with torch.no_grad():
                            candidates = self.generator(z)
                            # candidates: [n_candidates, 1, 64, 64] on device
                            
                            # Score with classifier ->margin
                            # Temporarily switch to eval to disable dropout
                            model.eval()
                            logits = model(candidates)
                            model.train()
                            
                            probs = torch.softmax(logits, dim=1)
                            sorted_probs, _ = probs.sort(dim=1, descending=True)
                            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                            
                            # Keep n_synthetic with LOWEST margin (most uncertain)
                            _, best_idx = torch.topk(
                                margin, k=n_synthetic, largest=False
                            )
                            # if torch.topk(largest=True), it gives highest margin (most certain)
                            synthetic = candidates[best_idx].cpu()
                        
                        final_images.append(synthetic)
                        final_labels.append(torch.full(
                            (n_synthetic,), remapped_c, dtype=torch.long
                        ))
                
                # -> Concatenate, shuffle, train --
                batch_images = torch.cat(final_images, dim=0)
                batch_labels = torch.cat(final_labels, dim=0)
                
                perm = torch.randperm(batch_images.size(0))
                batch_images = batch_images[perm].to(self.device)
                batch_labels = batch_labels[perm].to(self.device)
                
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        finally:
            # Restore generator training state
            self.generator.train(mode=gen_was_training)
        
        return total_loss / max(num_batches, 1)
    
    def _train_epoch_ab10(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Ab-10: Ab-7 margin selection + separate G fine-tuning step.
        
        Phase 1 (Classifier Training) is identical to Ab-7:
            Class-stratified batching with majority consumed without replacement,
            minority cycling with BAGAN margin-sampled top-up.
        
        Phase 2 (Generator Fine-tuning) is NEW ->runs after each classifier
        batch update (controlled by ab10_generator_update_every):
            1. Switch model ->eval (stable BN stats), generator ->train (grad).
            2. For each minority class: sample fresh z, generate with grad,
               forward through classifier, compute CrossEntropy targeting
               the correct minority class label.
            3. Backward through G only (g_optimizer.step), clip gradients.
            4. CRITICAL: clear C's accumulated gradients (model.zero_grad)
               to prevent gradient leakage into the next Phase 1 step.
            5. Restore model ->train, generator ->eval.
        
        The G step uses the ALREADY-UPDATED classifier from Phase 1, so the
        gradient signal reflects the most current decision boundary.
        
        Args:
            model: CNNClassifier to train
            train_data: [N, 1, 64, 64] real training images (original labels)
            train_labels: [N] original class labels (NOT yet remapped)
            optimizer: Adam optimizer for classifier
            criterion: CrossEntropyLoss
        
        Returns:
            avg_loss: Average classifier training loss across all batches
        """
        model.train()
        
        n_per_class = self.config.batch_size // self.config.num_classes
        majority_class = self.config.majority_class
        overproduction = self.config.margin_overproduction_factor
        
        # Ab-10-specific config
        cls_loss_weight = self.config.ab10_cls_loss_weight
        update_every = self.config.ab10_generator_update_every
        grad_clip = self.config.ab10_grad_clip_max_norm
        n_g_samples = self.config.ab10_g_update_samples or n_per_class
        
        # -> Step 1: Separate real indices by class --
        majority_idx = (train_labels == majority_class).nonzero(as_tuple=True)[0]
        minority_idx = {}
        for cls in self.config.minority_classes:
            minority_idx[cls] = (train_labels == cls).nonzero(as_tuple=True)[0]
        
        # -> Step 2: Shuffle majority WITHOUT replacement --
        majority_perm = majority_idx[torch.randperm(len(majority_idx))]
        
        # -> Step 3: Shuffle minority for round-robin cycling --
        minority_perm = {}
        minority_ptr = {}
        for cls in self.config.minority_classes:
            idx = minority_idx[cls]
            minority_perm[cls] = idx[torch.randperm(len(idx))]
            minority_ptr[cls] = 0
        
        # -> Step 4: Determine num_batches from majority --
        num_batches = max(1, len(majority_perm) // n_per_class)
        
        # Pre-compute real minority images per batch
        minority_per_batch_base = {}
        minority_per_batch_extra = {}
        for cls in self.config.minority_classes:
            n_minority = len(minority_perm[cls])
            minority_per_batch_base[cls] = max(1, n_minority // num_batches)
            minority_per_batch_extra[cls] = n_minority % num_batches
        
        # G starts in eval for Phase 1 (inference only, no grad)
        gen_was_training = self.generator.training
        self.generator.eval()
        
        total_loss = 0.0
        
        try:
            for batch_i in range(num_batches):
                # --
                # PHASE 1: Classifier Training (identical to Ab-7)
                # --
                final_images = []
                final_labels = []
                
                # -> Majority: next n_per_class WITHOUT replacement --
                maj_start = batch_i * n_per_class
                maj_end = min(maj_start + n_per_class, len(majority_perm))
                maj_batch_idx = majority_perm[maj_start:maj_end]
                maj_images = train_data[maj_batch_idx]
                n_maj = len(maj_images)
                remapped_maj = self.label_to_index[majority_class]
                final_images.append(maj_images)
                final_labels.append(torch.full(
                    (n_maj,), remapped_maj, dtype=torch.long
                ))
                
                # -> Minority (each class): cycle real + margin-sampled BAGAN ?
                for cls in self.config.minority_classes:
                    remapped_c = self.label_to_index[cls]
                    perm = minority_perm[cls]
                    n_total_minority = len(perm)
                    
                    # How many real minority images for THIS batch
                    base = minority_per_batch_base[cls]
                    extra = minority_per_batch_extra[cls]
                    n_real = base + (1 if batch_i < extra else 0)
                    
                    # Gather real minority via cycling pointer
                    real_indices = []
                    ptr = minority_ptr[cls]
                    for _ in range(n_real):
                        real_indices.append(perm[ptr % n_total_minority].item())
                        ptr += 1
                    minority_ptr[cls] = ptr
                    
                    # Add real minority images to batch
                    real_idx_tensor = torch.tensor(real_indices, dtype=torch.long)
                    real_images = train_data[real_idx_tensor]
                    final_images.append(real_images)
                    final_labels.append(torch.full(
                        (n_real,), remapped_c, dtype=torch.long
                    ))
                    
                    # -> Margin-sampled BAGAN top-up --
                    n_synthetic = n_per_class - n_real
                    if n_synthetic > 0:
                        n_candidates = n_synthetic * overproduction
                        
                        z = self.latent_generator.sample_for_class_oversampling(
                            target_class=cls, n_samples=n_candidates
                        )
                        z = z.to(self.device)
                        with torch.no_grad():
                            candidates = self.generator(z)
                            
                            # Score with classifier ->margin
                            model.eval()
                            logits = model(candidates)
                            model.train()
                            
                            probs = torch.softmax(logits, dim=1)
                            sorted_probs, _ = probs.sort(dim=1, descending=True)
                            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                            
                            # Keep n_synthetic with LOWEST margin (most uncertain)
                            _, best_idx = torch.topk(
                                margin, k=n_synthetic, largest=False
                            )
                            synthetic = candidates[best_idx].cpu()
                        
                        final_images.append(synthetic)
                        final_labels.append(torch.full(
                            (n_synthetic,), remapped_c, dtype=torch.long
                        ))
                
                # -> Concatenate, shuffle, train classifier --
                batch_images = torch.cat(final_images, dim=0)
                batch_labels = torch.cat(final_labels, dim=0)
                
                perm_shuffle = torch.randperm(batch_images.size(0))
                batch_images = batch_images[perm_shuffle].to(self.device)
                batch_labels = batch_labels[perm_shuffle].to(self.device)
                
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # --
                # PHASE 2: Generator Fine-tuning (NEW in Ab-10)
                # --
                # Fine-tune G so it generates images that the
                # ALREADY-UPDATED classifier classifies correctly.
                # Controlled by update_every (default=1 ->every batch).
                if batch_i % update_every == 0:
                    model.eval()            # Stable BN stats for scoring
                    self.generator.train()   # Enable G gradient tracking
                    
                    g_loss_total = torch.tensor(0.0, device=self.device)
                    
                    for cls in self.config.minority_classes:
                        remapped_c = self.label_to_index[cls]
                        z_fresh = self.latent_generator.sample_for_class_oversampling(
                            target_class=cls, n_samples=n_g_samples
                        ).to(self.device)
                        
                        # G graph is tracked (no torch.no_grad)
                        fake_images = self.generator(z_fresh)
                        # C forward ->gradients flow through C to G
                        g_logits = model(fake_images)
                        target = torch.full(
                            (n_g_samples,), remapped_c,
                            dtype=torch.long, device=self.device
                        )
                        g_loss_total = g_loss_total + (
                            cls_loss_weight * criterion(g_logits, target)
                        )
                    
                    # Backward through G (and C, but we only step G)
                    self.g_optimizer.zero_grad() # required pre-backward cleanup for generator params.
                    g_loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(), max_norm=grad_clip
                    )
                    self.g_optimizer.step()
                    
                    # CRITICAL: clear C's accumulated gradients from G step
                    # to prevent leakage into next Phase 1 optimizer.step()
                    self.g_optimizer.zero_grad() # post-step cleanup so generator grads don? linger after this mini-phase.
                    model.zero_grad()
                    
                    # Restore modes for next batch's Phase 1
                    self.generator.eval()   # Back to eval for inference
                    model.train()           # Back to train for classifier
        finally:
            # Restore generator training state
            self.generator.train(mode=gen_was_training)
        
        return total_loss / max(num_batches, 1)
    
    def _train_epoch_ab11(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Ab-11: Mixed D+C loss with warmup & throttled G updates.
        
        Phase 1 (Classifier Training) is identical to Ab-7 / Ab-10:
            Class-stratified batching with majority consumed without replacement,
            minority cycling with BAGAN margin-sampled top-up.
        
        Phase 2 (Generator Fine-tuning) extends Ab-10 with three changes:
            - Warmup: skipped entirely for the first ab11_warmup_epochs epochs.
            - Throttle: runs every ab11_generator_update_every batches (not every batch).
            - Mixed loss: g_loss = _cls*CE(C(G(z)),y) + _adv*CE(D(G(z)),y)
              where D is the pre-trained discriminator in eval mode with
              requires_grad=False (AC-GAN style: target = real minority class).
        
        Args:
            model: CNNClassifier to train
            train_data: [N, 1, 64, 64] real training images (original labels)
            train_labels: [N] original class labels (NOT yet remapped)
            optimizer: Adam optimizer for classifier
            criterion: CrossEntropyLoss
        
        Returns:
            avg_loss: Average classifier training loss across all batches
        """
        model.train()
        
        n_per_class = self.config.batch_size // self.config.num_classes
        majority_class = self.config.majority_class
        overproduction = self.config.margin_overproduction_factor
        
        # Ab-11-specific config
        cls_loss_weight = self.config.ab10_cls_loss_weight       # _cls = 1.0
        d_loss_weight = self.config.ab11_d_loss_weight           # _adv = 0.5
        warmup_epochs = self.config.ab11_warmup_epochs           # 5
        update_every = self.config.ab11_generator_update_every   # 5
        grad_clip = self.config.ab10_grad_clip_max_norm          # 1.0
        n_g_samples = self.config.ab10_g_update_samples or n_per_class
        
        # -> Step 1: Separate real indices by class --
        majority_idx = (train_labels == majority_class).nonzero(as_tuple=True)[0]
        minority_idx = {}
        for cls in self.config.minority_classes:
            minority_idx[cls] = (train_labels == cls).nonzero(as_tuple=True)[0]
        
        # -> Step 2: Shuffle majority WITHOUT replacement --
        majority_perm = majority_idx[torch.randperm(len(majority_idx))]
        
        # -> Step 3: Shuffle minority for round-robin cycling --
        minority_perm = {}
        minority_ptr = {}
        for cls in self.config.minority_classes:
            idx = minority_idx[cls]
            minority_perm[cls] = idx[torch.randperm(len(idx))]
            minority_ptr[cls] = 0
        
        # -> Step 4: Determine num_batches from majority --
        num_batches = max(1, len(majority_perm) // n_per_class)
        
        # Pre-compute real minority images per batch
        minority_per_batch_base = {}
        minority_per_batch_extra = {}
        for cls in self.config.minority_classes:
            n_minority = len(minority_perm[cls])
            minority_per_batch_base[cls] = max(1, n_minority // num_batches)
            minority_per_batch_extra[cls] = n_minority % num_batches
        
        # G starts in eval for Phase 1 (inference only, no grad)
        gen_was_training = self.generator.training
        disc_was_training = self.discriminator.training
        self.generator.eval()
        
        total_loss = 0.0
        in_warmup = self._current_epoch < warmup_epochs
        
        try:
            for batch_i in range(num_batches):
                # --
                # PHASE 1: Classifier Training (identical to Ab-7)
                # --
                final_images = []
                final_labels = []
                
                # -> Majority: next n_per_class WITHOUT replacement --
                maj_start = batch_i * n_per_class
                maj_end = min(maj_start + n_per_class, len(majority_perm))
                maj_batch_idx = majority_perm[maj_start:maj_end]
                maj_images = train_data[maj_batch_idx]
                n_maj = len(maj_images)
                remapped_maj = self.label_to_index[majority_class]
                final_images.append(maj_images)
                final_labels.append(torch.full(
                    (n_maj,), remapped_maj, dtype=torch.long
                ))
                
                # -> Minority (each class): cycle real + margin-sampled BAGAN ?
                for cls in self.config.minority_classes:
                    remapped_c = self.label_to_index[cls]
                    perm = minority_perm[cls]
                    n_total_minority = len(perm)
                    
                    # How many real minority images for THIS batch
                    base = minority_per_batch_base[cls]
                    extra = minority_per_batch_extra[cls]
                    n_real = base + (1 if batch_i < extra else 0)
                    
                    # Gather real minority via cycling pointer
                    real_indices = []
                    ptr = minority_ptr[cls]
                    for _ in range(n_real):
                        real_indices.append(perm[ptr % n_total_minority].item())
                        ptr += 1
                    minority_ptr[cls] = ptr
                    
                    # Add real minority images to batch
                    real_idx_tensor = torch.tensor(real_indices, dtype=torch.long)
                    real_images = train_data[real_idx_tensor]
                    final_images.append(real_images)
                    final_labels.append(torch.full(
                        (n_real,), remapped_c, dtype=torch.long
                    ))
                    
                    # -> Margin-sampled BAGAN top-up --
                    n_synthetic = n_per_class - n_real
                    if n_synthetic > 0:
                        n_candidates = n_synthetic * overproduction
                        
                        z = self.latent_generator.sample_for_class_oversampling(
                            target_class=cls, n_samples=n_candidates
                        )
                        z = z.to(self.device)
                        with torch.no_grad():
                            candidates = self.generator(z)
                            
                            # Score with classifier ->margin
                            model.eval()
                            logits = model(candidates)
                            model.train()
                            
                            probs = torch.softmax(logits, dim=1)
                            sorted_probs, _ = probs.sort(dim=1, descending=True)
                            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                            
                            # Keep n_synthetic with LOWEST margin (most uncertain)
                            _, best_idx = torch.topk(
                                margin, k=n_synthetic, largest=False
                            )
                            synthetic = candidates[best_idx].cpu()
                        
                        final_images.append(synthetic)
                        final_labels.append(torch.full(
                            (n_synthetic,), remapped_c, dtype=torch.long
                        ))
                
                # -> Concatenate, shuffle, train classifier --
                batch_images = torch.cat(final_images, dim=0)
                batch_labels = torch.cat(final_labels, dim=0)
                
                perm_shuffle = torch.randperm(batch_images.size(0))
                batch_images = batch_images[perm_shuffle].to(self.device)
                batch_labels = batch_labels[perm_shuffle].to(self.device)
                
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # --
                # PHASE 2: Generator Fine-tuning with Mixed D+C Loss
                # --
                # Skip Phase 2 entirely during warmup epochs.
                # After warmup, run every update_every batches.
                if not in_warmup and batch_i % update_every == 0:
                    model.eval()            # Stable BN stats for scoring
                    self.generator.train()   # Enable G gradient tracking
                    self.discriminator.eval()
                    self.discriminator.requires_grad_(False)  # skip D grad
                    
                    g_loss_total = torch.tensor(0.0, device=self.device)
                    
                    for cls in self.config.minority_classes:
                        remapped_c = self.label_to_index[cls]
                        z_fresh = self.latent_generator.sample_for_class_oversampling(
                            target_class=cls, n_samples=n_g_samples
                        ).to(self.device)
                        
                        # G forward ->graph tracked (no torch.no_grad)
                        fake_images = self.generator(z_fresh)
                        # C forward ->gradients flow through C to G
                        g_logits = model(fake_images)
                        # D forward ->gradients flow through D to G.
                        # Backend-aware: BAGAN returns [B, n+1] logits (full
                        # (n+1)-way softmax preserved for 106J equivalence);
                        # BAGAN-GP returns the auxiliary head's [B, n] logits.
                        # CE against `target ->[0, n-1]` works for both shapes.
                        d_logits = self._adv_class_logits(self.discriminator, fake_images)
                        
                        target = torch.full(
                            (n_g_samples,), remapped_c,
                            dtype=torch.long, device=self.device
                        )
                        
                        g_cls = criterion(g_logits, target)
                        g_adv = criterion(d_logits, target)  # AC-GAN: real class
                        
                        g_loss_total = g_loss_total + (
                            cls_loss_weight * g_cls + d_loss_weight * g_adv
                        )
                    
                    # Backward through G (and C/D graphs, but only step G)
                    self.g_optimizer.zero_grad()
                    g_loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(), max_norm=grad_clip
                    )
                    self.g_optimizer.step()
                    
                    # CRITICAL: clear C's accumulated gradients from G step
                    # to prevent leakage into next Phase 1 optimizer.step()
                    self.g_optimizer.zero_grad()
                    model.zero_grad()
                    
                    # Restore modes for next batch's Phase 1
                    self.discriminator.requires_grad_(True)
                    self.generator.eval()   # Back to eval for inference
                    model.train()           # Back to train for classifier
        finally:
            # Restore generator and discriminator training state
            self.generator.train(mode=gen_was_training)
            self.discriminator.train(mode=disc_was_training)
        
        return total_loss / max(num_batches, 1)
    
    # --
    # Ab-11a ->Step B helpers (feature matching + margin-selected G)
    # --
    # These three private helpers underpin `_train_epoch_ab11a` (added in
    # Step D). They are split out so each one is independently unit-testable
    # via `test_trainer_ab11a.py` without spinning up the full training
    # pipeline. None of them is called from any existing ablation mode, so
    # adding them is regression-safe.

    def _classifier_features(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Penultimate (64-dim) features from the *current fold's* CNNClassifier.
        
        Thin wrapper around ``model.extract_features(x)``. Required because
        the classifier model is fold-local (created fresh each fold inside
        ``train_fold``), so we cannot stash it on ``self`` at __init__ time.
        Caller passes the active model.
        """
        return model.extract_features(x)

    def _discriminator_features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate (8192-dim flattened) features from the frozen D.

        Backend-aware (110J): dispatches through ``self._disc_features`` so
        BAGAN reads ``discriminator.conv_layers(x)`` and BAGAN-GP reads
        ``discriminator.features(x)`` ->both produce the same 8192-dim
        flattened vector (the conv stack has identical spatial layout).
        Dropout is intentionally skipped (D should be in ``eval()`` when used
        as a feature extractor in Ab-11a Phase 2).
        """
        if self.discriminator is None:
            raise ValueError(
                "_discriminator_features requires a discriminator; pass "
                "discriminator= to ClassifierTrainer."
            )
        return self._disc_features(self.discriminator, x)   # (B, 8192)

    def _get_penultimate_features(
        self,
        x: torch.Tensor,
        which: str,
        *,
        classifier: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Dispatch penultimate-feature extraction by source.
        
        Used by the Ab-11a feature-matching loss to compute ? for both
        real minority and fake samples. The configured choice lives in
        ``config.ab11a_fm_feature_layer``.
        
        Args:
            x: Input images [B, 1, 64, 64] in [-1, 1].
            which: One of {'classifier_penultimate',
                'discriminator_penultimate', 'autoencoder_encoder'}.
            classifier: Required when ``which == 'classifier_penultimate'``;
                the fold-local CNNClassifier instance.
        
        Returns:
            Tensor of penultimate features (shape depends on ``which``).
        """
        if which == 'classifier_penultimate':
            if classifier is None:
                raise ValueError(
                    "classifier= is required when which='classifier_penultimate'"
                )
            return self._classifier_features(classifier, x)
        if which == 'discriminator_penultimate':
            return self._discriminator_features(x)
        if which == 'autoencoder_encoder':
            # Reserved for future ablations (Ab-11c?). The frozen AE encoder
            # is currently not threaded into ClassifierTrainer; it lives on
            # the BAGANAutoencoder built in the notebook. Wire it through if
            # you want to experiment with this ?.
            raise NotImplementedError(
                "which='autoencoder_encoder' is reserved; pass the encoder "
                "explicitly via a future arg."
            )
        raise ValueError(
            f"Unknown feature source '{which}'. Must be one of "
            "'classifier_penultimate', 'discriminator_penultimate', "
            "'autoencoder_encoder'."
        )

    def _margin_select_with_grad(
        self,
        fake_all: torch.Tensor,
        classifier: nn.Module,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pick the lowest-margin k samples while preserving G's grad path.
        
        Ab-11a Phase 2 generates ``ab11a_g_overproduction x n_g`` candidates
        from G, scores all of them with the (fold-local) classifier under
        ``no_grad``, and keeps the lowest-margin ``k`` for the G update.
        
        CRITICAL ->gradient flow:
            We score under ``no_grad`` to save memory + skip BN running-stat
            updates, but we MUST index the *original* differentiable
            ``fake_all`` tensor (not a detached copy) so the kept samples
            still carry G's autograd history. Returning ``fake_all[kept_idx]``
            preserves that path because advanced indexing is differentiable.
        
        Args:
            fake_all: G's overproduced batch [N, 1, 64, 64], requires_grad.
            classifier: Fold-local CNNClassifier (will be left in caller's
                training mode ->we restore it before returning).
            k: Number of samples to keep (k ->N).
        
        Returns:
            (fake_kept, kept_idx)
                fake_kept: [k, 1, 64, 64] ->grad path to G preserved
                kept_idx:  [k] ->long indices into fake_all
        """
        if k > fake_all.size(0):
            raise ValueError(
                f"_margin_select_with_grad: requested k={k} but only "
                f"{fake_all.size(0)} candidates available."
            )
        was_training = classifier.training
        classifier.eval()
        try:
            with torch.no_grad():
                logits = classifier(fake_all.detach())
                probs = F.softmax(logits, dim=1)
                top2 = probs.topk(2, dim=1).values            # (N, 2)
                margin = top2[:, 0] - top2[:, 1]              # (N,)
                kept_idx = margin.argsort(descending=False)[:k]
        finally:
            classifier.train(mode=was_training)
        return fake_all[kept_idx], kept_idx

    def _init_ab11a_real_pool(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
    ) -> None:
        """Build the per-fold real-minority pool used by feature matching.
        
        Called once at the start of each fold (from ``_train_epoch_ab11a``,
        Step D) before any Phase-2 update. Populates two pieces of state:
        
            self._real_minority_pool : Dict[int, Tensor]
                Maps *original* class label (e.g. 2 or 3 for the default [0, 2, 3])
                to the full set of that fold's real images for that class,
                shape [P_c, 1, 64, 64], placed on ``self.device``.
            self._fm_real_ptr : Dict[int, int]
                Round-robin read pointer per class (see ``_sample_real_minority``).
        
        Minority classes are inferred as everything except the most-frequent
        label in the training fold (matches Ab-11a's "anchor on real
        minority" semantics; the majority class is well-served by D and C
        already). Labels are accepted in *original* (pre-remap) form to
        match the convention used by ``train_fold``.
        """
        if not torch.is_tensor(train_labels):
            train_labels = torch.as_tensor(train_labels)
        labels_cpu = train_labels.detach().cpu()
        unique, counts = torch.unique(labels_cpu, return_counts=True)
        majority_label = int(unique[counts.argmax()].item())
        minority_labels = [int(l.item()) for l in unique if int(l.item()) != majority_label]

        pool: Dict[int, torch.Tensor] = {}
        for lbl in minority_labels:
            mask = (labels_cpu == lbl)
            samples = train_data[mask].to(self.device)
            pool[lbl] = samples
        self._real_minority_pool = pool
        self._fm_real_ptr = {lbl: 0 for lbl in minority_labels}

    def _sample_real_minority(
        self,
        class_c: int,
        n: Optional[int],
    ) -> torch.Tensor:
        """Round-robin sample of n real images of class ``class_c``.
        
        Deterministic per fold given the (epoch, batch_i) call sequence:
        each call advances ``self._fm_real_ptr[class_c]`` by ``n`` modulo
        the pool size. Passing ``n=None`` (the Ab-11a default for
        ``ab11a_fm_n_real``) returns the *entire* pool every call ->this
        gives the most stable FM target at the price of O(P) memory per
        Phase-2 step.
        
        Args:
            class_c: Original class label (key into ``_real_minority_pool``).
            n: Number to draw, or None for "all".
        
        Returns:
            Tensor of shape [n_eff, 1, 64, 64] on ``self.device``.
        """
        if not hasattr(self, '_real_minority_pool'):
            raise RuntimeError(
                "_sample_real_minority called before _init_ab11a_real_pool; "
                "make sure mode='ab11a' fold initialisation ran."
            )
        if class_c not in self._real_minority_pool:
            raise KeyError(
                f"class {class_c} not in real-minority pool; available: "
                f"{sorted(self._real_minority_pool.keys())}"
            )
        pool = self._real_minority_pool[class_c]
        P = pool.size(0)
        if n is None or n >= P:
            return pool
        ptr = self._fm_real_ptr.get(class_c, 0)
        idx = [(ptr + i) % P for i in range(n)]
        self._fm_real_ptr[class_c] = (ptr + n) % P
        return pool[idx]

    # --
    # Ab-11a ->Step E: per-epoch diagnostics & FID monitor wiring
    # --
    def _log_ab11a_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Dict[str, float],
    ) -> None:
        """Print one diagnostic line per fold-epoch and append to history.
        
        Reads `self._ab11a_last_epoch_terms` (populated by
        `_train_epoch_ab11a`) and divides each accumulator by
        `phase2_steps` to recover per-step means. Also records the line's
        components in `self.fold_history` for later plotting / sweep
        aggregation.
        
        No-op (silently) before Phase 2 has run any step (i.e. during the
        warm-up window) ->there are no terms to print yet, but train/val
        numbers are still appended to `fold_history` so the curves stay
        contiguous.
        """
        terms = getattr(self, '_ab11a_last_epoch_terms', None) or {}
        n_steps = max(int(terms.get('phase2_steps', 0)), 1)
        g_cls = terms.get('cls', 0.0) / n_steps
        g_adv = terms.get('adv', 0.0) / n_steps
        g_fm = terms.get('fm', 0.0) / n_steps
        sel_m = terms.get('sel_margin', 0.0) / n_steps
        
        # Always grow history (one row per epoch) so plotting code is simple.
        hist = self.fold_history
        hist.setdefault('epoch', []).append(epoch)
        hist.setdefault('train_loss', []).append(float(train_loss))
        hist.setdefault('val_loss', []).append(float(val_loss))
        hist.setdefault('val_macro_f1', []).append(float(metrics.get('macro_f1', float('nan'))))
        hist.setdefault('val_binary_f1', []).append(float(metrics.get('binary_f1', float('nan'))))
        hist.setdefault('g_cls', []).append(g_cls)
        hist.setdefault('g_adv', []).append(g_adv)
        hist.setdefault('g_fm', []).append(g_fm)
        hist.setdefault('sel_margin', []).append(sel_m)
        hist.setdefault('phase2_steps', []).append(int(terms.get('phase2_steps', 0)))
        
        if not self.config.ab11a_verbose_phase2:
            return
        if terms.get('phase2_steps', 0) == 0:
            # Warm-up: print a slim line for traceability.
            print(
                f"      [ab11a ep {epoch:>3}] train {train_loss:.3f} "
                f"val {val_loss:.3f} | val_macroF1 "
                f"{metrics.get('macro_f1', float('nan')):.3f} "
                f"| (warm-up ->Phase 2 idle)"
            )
            return
        print(
            f"      [ab11a ep {epoch:>3}] train {train_loss:.3f} "
            f"val {val_loss:.3f} "
            f"| val_macroF1 {metrics.get('macro_f1', float('nan')):.3f} "
            f"binF1 {metrics.get('binary_f1', float('nan')):.3f} "
            f"| g_cls {g_cls:.3f} g_adv {g_adv:.3f} g_fm {g_fm:.3f} "
            f"sel_margin {sel_m:.3f} (n={int(terms['phase2_steps'])})"
        )

    def _maybe_run_fid_monitor(self, epoch: int) -> None:
        """Cheap per-class FID monitor (Step E call site, Step H compute).
        
        When `cfg.ab11a_fid_eval_every > 0` and `(epoch + 1) % every == 0`,
        attempt to import `ablation.fid.fid_per_class` (added in Step H)
        and evaluate it against `cfg.minority_classes` with
        `cfg.ab11a_fid_monitor_samples` fakes per class.
        
        Side-effects:
            * `self.fold_history['fid_class_<c>']` gains a `(epoch, fid)`
              tuple for each minority class.
            * `self.fold_history['fid_mean']` gains `(epoch, mean_fid)`.
            * If mean FID strictly improves over `self._best_fid`, update
              both that scalar and `self.best_g_state_dict` (CPU clone).
        
        Silently no-ops when:
            * `ab11a_fid_eval_every == 0` (default), or
            * `epoch + 1` is not a multiple of the cadence, or
            * Step H's `ablation.fid` is not yet on the path (lets us land
              this wiring in Step E without breaking imports).
        """
        every = int(self.config.ab11a_fid_eval_every)
        if every <= 0:
            return
        if (epoch + 1) % every != 0:
            return
        try:
            from .fid import fid_per_class  # Step H
        except ImportError:
            # Step H hasn't shipped yet ->leave a single advisory line so
            # users who *did* enable the monitor know why it stayed silent.
            if self.config.ab11a_verbose_phase2:
                print(
                    "      [ab11a fid-monitor] cadence hit "
                    f"(epoch {epoch}) but `ablation.fid` is unavailable ->"
                    "Step H will provide it; skipping."
                )
            return
        
        n = int(self.config.ab11a_fid_monitor_samples)
        fid_dict: Dict[int, float] = fid_per_class(
            self.generator,
            self.latent_generator,
            classes=list(self.config.minority_classes),
            n_samples=n,
            device=self.device,
            ref_cache_dir=self.config.bagan_model_dir,
            date_suffix=self.config.date_suffix,
        )
        
        hist = self.fold_history
        for c, v in fid_dict.items():
            hist.setdefault(f'fid_class_{c}', []).append((epoch, float(v)))
        if fid_dict:
            mean_fid = float(sum(fid_dict.values()) / len(fid_dict))
            hist.setdefault('fid_mean', []).append((epoch, mean_fid))
            if mean_fid < getattr(self, '_best_fid', float('inf')):
                self._best_fid = mean_fid
                self.best_g_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.generator.state_dict().items()
                }
                if self.config.ab11a_verbose_phase2:
                    pretty = ' '.join(f'c{c}={v:.2f}' for c, v in fid_dict.items())
                    print(
                        f"      [ab11a fid-monitor ep {epoch}] "
                        f"mean {mean_fid:.2f} ({pretty}) "
                        f"\u2193 best ->snapshotted G"
                    )
            elif self.config.ab11a_verbose_phase2:
                pretty = ' '.join(f'c{c}={v:.2f}' for c, v in fid_dict.items())
                print(
                    f"      [ab11a fid-monitor ep {epoch}] "
                    f"mean {mean_fid:.2f} ({pretty})"
                )

    # --
    # Ab-11a ->Step C: epoch trainer
    # --
    def _train_epoch_ab11a(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Ab-11a: Ab-11 + Feature Matching + Margin-Selected G (V3 design).
        
        Phase 1 (Classifier Training) ->*byte-identical to Ab-11*:
            Class-stratified batching with majority WITHOUT replacement,
            minority cycling, BAGAN margin-sampled top-up. Copied verbatim
            so that turning all `ab11a_*` toggles off reproduces Ab-11.
        
        Phase 2 (Generator Fine-tuning) ->V3:
            * Overproduce: draw `over x n_g` candidate fakes per minority class.->
            * Margin-select-with-grad: keep the lowest-margin `n_g` while
              preserving the autograd path back to G (see
              `_margin_select_with_grad`).
            * Loss per class:
                `L_c = _cls * CE(C(G(z)), y)
                     + _adv * CE(D(G(z)), y)
                     + _fm  * FM(?_real, ?_fake)`
              where `FM` is squared mean (and optionally variance) distance
              of penultimate features ? from `ab11a_fm_feature_layer`.
            * Real anchor: `_sample_real_minority` returns the full per-fold
              minority pool by default (`ab11a_fm_n_real=None`).
        
        Toggles (regression-safe):
            * `ab11a_g_use_margin_selection=False` ->use raw `n_g` fakes.
            * `ab11a_g_use_feature_matching=False` ->drop the FM term;
              when *both* are False the Phase-2 loss reduces to Ab-11's
              `_cls * CE(C) + _adv * CE(D)` exactly.
            * `ab11a_fm_match_var=True` (default) ->mean-and-variance FM
              (cheap anti-mode-collapse signal while D is frozen). Set to
              False for plain mean FM (Salimans 2016).
        
        Side-effects beyond returning ``avg_loss``:
            * Lazy-initialises ``self._real_minority_pool`` and
              ``self._fm_real_ptr`` on the first call of each fold (Step D's
              ``train_fold`` will reset them when the fold changes).
            * Stashes per-epoch diagnostic loss components in
              ``self._ab11a_last_epoch_terms`` (Step E reads this).
        
        Args:
            model: CNNClassifier being trained this fold.
            train_data: [N, 1, 64, 64] real images, original labels.
            train_labels: [N] original (pre-remap) class labels.
            optimizer: Classifier optimizer.
            criterion: CrossEntropyLoss.
        
        Returns:
            avg_loss: Mean Phase-1 classifier loss across all batches
                      (matches `_train_epoch_ab11`'s contract ->Phase-2
                      loss components live in
                      ``self._ab11a_last_epoch_terms``).
        """
        model.train()
        cfg = self.config
        device = self.device
        
        n_per_class = cfg.batch_size // cfg.num_classes # per-class real+synthetic in Phase 1
        majority_class = cfg.majority_class
        overproduction = cfg.margin_overproduction_factor
        
        # -> Inherited Ab-11 hyperparameters --
        cls_loss_weight = cfg.ab10_cls_loss_weight
        d_loss_weight = cfg.ab11_d_loss_weight
        warmup_epochs = cfg.ab11_warmup_epochs
        update_every = cfg.ab11_generator_update_every
        grad_clip = cfg.ab10_grad_clip_max_norm
        n_g_samples = cfg.ab10_g_update_samples or n_per_class
        
        # -> New Ab-11a hyperparameters --
        fm_loss_weight = cfg.ab11a_fm_loss_weight
        fm_layer = cfg.ab11a_fm_feature_layer
        fm_match_var = cfg.ab11a_fm_match_var
        fm_n_real = cfg.ab11a_fm_n_real
        use_margin = cfg.ab11a_g_use_margin_selection
        g_over = cfg.ab11a_g_overproduction
        use_fm = cfg.ab11a_g_use_feature_matching
        diversity_w = cfg.ab11a_g_diversity_weight  # reserved (Step E may emit)
        
        # -> Lazy real-minority pool init (per fold) --
        # Step D's `train_fold` clears these at fold start; here we
        # just guarantee they exist before Phase 2 needs them.
        if not hasattr(self, '_real_minority_pool'):
            self._init_ab11a_real_pool(train_data, train_labels)
        
        # Diagnostic accumulator (Step E reads this after each epoch).
        epoch_terms = {
            'cls': 0.0, 'adv': 0.0, 'fm': 0.0,
            'sel_margin': 0.0, 'phase2_steps': 0,
        }
        
        # -> Step 1: Separate real indices by class --
        majority_idx = (train_labels == majority_class).nonzero(as_tuple=True)[0]
        minority_idx = {}
        for cls in cfg.minority_classes:
            minority_idx[cls] = (train_labels == cls).nonzero(as_tuple=True)[0]
        
        # -> Step 2: Shuffle majority WITHOUT replacement --
        majority_perm = majority_idx[torch.randperm(len(majority_idx))]
        
        # -> Step 3: Shuffle minority for round-robin cycling --
        minority_perm = {}
        minority_ptr = {}
        for cls in cfg.minority_classes:
            idx = minority_idx[cls]
            minority_perm[cls] = idx[torch.randperm(len(idx))]
            minority_ptr[cls] = 0
        
        # -> Step 4: Determine num_batches from majority --
        # n_per_class = cfg.batch_size // cfg.num_classes <- defined above for clarity
        # We could determine num_batches from the minority instead, but the
        # majority is the natural anchor for Phase 1's "cycle minority, top up with BAGAN" design, 
        # and it ensures we always use each majority sample exactly once per epoch 
        # (no risk of under- or over-sampling the majority).
        num_batches = max(1, len(majority_perm) // n_per_class)
        
        minority_per_batch_base = {}
        minority_per_batch_extra = {}
        for cls in cfg.minority_classes:
            n_minority = len(minority_perm[cls])
            minority_per_batch_base[cls] = max(1, n_minority // num_batches)
            minority_per_batch_extra[cls] = n_minority % num_batches
        
        gen_was_training = self.generator.training
        disc_was_training = self.discriminator.training
        self.generator.eval()
        
        total_loss = 0.0
        in_warmup = self._current_epoch < warmup_epochs
        
        try:
            for batch_i in range(num_batches):
                # --
                # PHASE 1 ->copied verbatim from `_train_epoch_ab11`
                # --
                final_images = []
                final_labels = []
                
                maj_start = batch_i * n_per_class
                maj_end = min(maj_start + n_per_class, len(majority_perm))
                maj_batch_idx = majority_perm[maj_start:maj_end]
                maj_images = train_data[maj_batch_idx]
                n_maj = len(maj_images)
                remapped_maj = self.label_to_index[majority_class]
                final_images.append(maj_images)
                final_labels.append(torch.full(
                    (n_maj,), remapped_maj, dtype=torch.long
                ))
                
                for cls in cfg.minority_classes:
                    remapped_c = self.label_to_index[cls]
                    perm = minority_perm[cls]
                    n_total_minority = len(perm)
                    
                    base = minority_per_batch_base[cls]
                    extra = minority_per_batch_extra[cls]
                    n_real = base + (1 if batch_i < extra else 0)
                    
                    real_indices = []
                    ptr = minority_ptr[cls]
                    for _ in range(n_real):
                        real_indices.append(perm[ptr % n_total_minority].item())
                        ptr += 1
                    minority_ptr[cls] = ptr
                    
                    real_idx_tensor = torch.tensor(real_indices, dtype=torch.long)
                    real_images = train_data[real_idx_tensor]
                    final_images.append(real_images)
                    final_labels.append(torch.full(
                        (n_real,), remapped_c, dtype=torch.long
                    ))
                    
                    n_synthetic = n_per_class - n_real
                    if n_synthetic > 0:
                        n_candidates = n_synthetic * overproduction
                        z = self.latent_generator.sample_for_class_oversampling(
                            target_class=cls, n_samples=n_candidates
                        ).to(device)
                        with torch.no_grad():
                            candidates = self.generator(z)
                            model.eval()
                            logits = model(candidates)
                            model.train()
                            probs = torch.softmax(logits, dim=1)
                            sorted_probs, _ = probs.sort(dim=1, descending=True)
                            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                            _, best_idx = torch.topk(
                                margin, k=n_synthetic, largest=False
                            )
                            synthetic = candidates[best_idx].cpu()
                        
                        final_images.append(synthetic)
                        final_labels.append(torch.full(
                            (n_synthetic,), remapped_c, dtype=torch.long
                        ))
                
                batch_images = torch.cat(final_images, dim=0)
                batch_labels = torch.cat(final_labels, dim=0)
                perm_shuffle = torch.randperm(batch_images.size(0))
                batch_images = batch_images[perm_shuffle].to(device)
                batch_labels = batch_labels[perm_shuffle].to(device)
                
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # --
                # PHASE 2 ->Ab-11a (FM + margin-selected G)
                # --
                if in_warmup or batch_i % update_every != 0:
                    continue
                
                # C in eval (stable BN), D frozen, G in train (need grads)
                model.eval()
                self.generator.train()
                self.discriminator.eval()
                self.discriminator.requires_grad_(False)
                
                self.g_optimizer.zero_grad(set_to_none=True)
                g_loss_total = torch.zeros((), device=device)
                
                for cls in cfg.minority_classes:
                    remapped_c = self.label_to_index[cls]
                    
                    # -> Overproduce candidates (with grad path to G) --
                    if use_margin:
                        n_cand = n_g_samples * g_over
                    else:
                        n_cand = n_g_samples
                    z_fresh = self.latent_generator.sample_for_class_oversampling(
                        target_class=cls, n_samples=n_cand
                    ).to(device)
                    fake_all = self.generator(z_fresh)  # WITH grad
                    
                    # -> Margin-select keeping the live G graph --
                    if use_margin:
                        fake, _kept_idx = self._margin_select_with_grad(
                            fake_all, model, k=n_g_samples
                        )
                        # Diagnostic: mean margin of the kept set
                        with torch.no_grad():
                            p_diag = F.softmax(model(fake.detach()), dim=1)
                            t2 = p_diag.topk(2, dim=1).values
                            epoch_terms['sel_margin'] += float(
                                (t2[:, 0] - t2[:, 1]).mean().item()
                            )
                    else:
                        fake = fake_all
                    
                    target = torch.full(
                        (fake.size(0),), remapped_c,
                        dtype=torch.long, device=device,
                    )
                    
                    # -> CE through frozen C and frozen D --
                    g_logits = model(fake)
                    # Backend-aware D logits (110J): BAGAN returns [B, n+1]
                    # (full softmax over n real + fake column, preserving
                    # 106J equivalence); BAGAN-GP returns the aux head [B, n].
                    # CE with target ->[0, n-1] is correct for both shapes.
                    d_logits = self._adv_class_logits(self.discriminator, fake)
                    g_cls = criterion(g_logits, target)
                    g_adv = criterion(d_logits, target)
                    L_c = cls_loss_weight * g_cls + d_loss_weight * g_adv
                    epoch_terms['cls'] += float(g_cls.item())
                    epoch_terms['adv'] += float(g_adv.item())
                    
                    # -> Feature-matching (FM) anchor against real minority --
                    if use_fm: # FM
                        x_real = self._sample_real_minority(cls, fm_n_real).to(device)
                        with torch.no_grad():
                            phi_real = self._get_penultimate_features(
                                x_real, fm_layer, classifier=model
                            )
                        phi_fake = self._get_penultimate_features(
                            fake, fm_layer, classifier=model
                        )
                        g_fm = (phi_real.mean(0) - phi_fake.mean(0)).pow(2).sum()
                        if fm_match_var: # cheapest direct anti-mode-collapse signal available while D is frozen. 
                            g_fm = g_fm + (
                                phi_real.var(0, unbiased=False)
                                - phi_fake.var(0, unbiased=False)
                            ).pow(2).sum()
                        L_c = L_c + fm_loss_weight * g_fm
                        epoch_terms['fm'] += float(g_fm.item())
                    
                    g_loss_total = g_loss_total + L_c
                
                # -> Single backward through G; D / C grads are zeroed --
                g_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), max_norm=grad_clip
                )
                self.g_optimizer.step()
                
                # Clean up grads accumulated on C / D / G
                self.g_optimizer.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)
                self.discriminator.zero_grad(set_to_none=True)
                
                # Restore modes for next batch's Phase 1
                self.discriminator.requires_grad_(True)
                self.generator.eval()
                model.train()
                
                epoch_terms['phase2_steps'] += 1
        finally:
            self.generator.train(mode=gen_was_training)
            self.discriminator.train(mode=disc_was_training)
        
        # Stash diagnostics for Step E (notebook reads after epoch).
        self._ab11a_last_epoch_terms = epoch_terms
        
        return total_loss / max(num_batches, 1)
    
    def _validate(
        self,
        model: nn.Module,
        dataloader: DeviceDataLoader,
        criterion: nn.Module
    ) -> Tuple[float, Dict[str, float], torch.Tensor, torch.Tensor]:
        """Validate model and compute all metrics on the validation set.
        
        Validation differs from training in two ways:
        1. model.eval()  ->disables dropout, uses batch norm running averages
        2. torch.no_grad() ->disables gradient computation (saves memory + time)
        
        We collect ALL predictions and labels first, then compute metrics
        once on the full validation set. This gives accurate global metrics
        (computing per-batch then averaging would be biased by batch size).
        
        Metrics computed (via compute_metrics from metrics.py):
            accuracy, macro_f1, macro_precision, macro_recall,
            macro_specificity, macro_gmean, macro_auc
        
        Args:
            model: CNNClassifier to evaluate
            dataloader: DeviceDataLoader wrapping validation TensorDataset
            criterion: CrossEntropyLoss for computing val_loss
        
        Returns:
            avg_loss: Average validation loss across all batches.
                      Used by EarlyStopping to decide when to stop.
            metrics: Dictionary of validation metrics.
                     Used for reporting and for selecting the best model.
            all_logits: [N_val, num_classes] raw model outputs (before softmax).
                        Returned so that post-hoc calibration (Topic 7) can
                        re-derive predictions with adjusted decision rules
                        without re-running the forward pass.
            all_labels: [N_val] ground-truth labels (contiguous {0,1,2}).
                        Paired with all_logits for calibration evaluation.
        """
        model.eval()    # Disable dropout, use batch norm running averages
        
        total_loss = 0.0
        num_batches = 0
        
        # Collect predictions across all batches before computing metrics
        all_logits = []     # Raw model outputs (before softmax)
        all_labels = []     # True labels
        
        with torch.no_grad():   # No gradient computation needed for eval
            for images, labels in dataloader:
                # Forward pass only (no backward)
                outputs = model(images)             # [B, num_classes]
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect for metrics computation
                all_logits.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # -> Concatenate all batch outputs --
        # all_logits: list of [B_i, num_classes] ->[N_val, num_classes]
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # -> Convert logits to predictions and probabilities --
        #
        # Softmax: converts raw logits into probabilities that sum to 1.
        #   logits [2.1, -0.3, 0.5] ->softmax ->[0.72, 0.07, 0.21]
        #
        # argmax: picks the class with highest probability as prediction.
        #   [0.72, 0.07, 0.21] ->argmax ->0 (class index 0)
        #
        # Both all_labels (from the DataLoader, remapped to {0,1,2})
        # and pred_indices (from argmax, naturally {0,1,2}) are in the
        # same contiguous index space. No mapping needed.
        probs = F.softmax(all_logits, dim=1)    # [N_val, num_classes]
        pred_indices = all_logits.argmax(dim=1)  # [N_val] ->class indices {0,1,2}
        
        # -> Compute all metrics --
        # all_labels from the val DataLoader are already remapped to
        # contiguous {0,1,2} (by _remap_labels in train_fold).
        # pred_indices from argmax are also {0,1,2}.
        # Both are in the same space ->compare directly.
        metrics = compute_metrics(
            y_true=all_labels.numpy(),           # [N_val] remapped labels {0,1,2}
            y_pred=pred_indices.numpy(),          # [N_val] predictions {0,1,2}
            y_prob=probs.numpy(),                # [N_val, num_classes] softmax probs
            num_classes=self.config.num_classes   # 3 for 511J
        )
        
        # Binary metrics (515J): binarize {0}?eg, {1,2}?os
        binary_metrics = compute_metrics_binary(
            y_true=all_labels.numpy(),
            y_pred=pred_indices.numpy(),
            y_prob=probs.numpy(),
            num_classes=self.config.num_classes,
        )
        metrics.update(binary_metrics) 
        # how does this line work?
        # Update the main metrics dict with the binary metrics (keys won't overlap)
        # How does dict.update() work?
        # dict.update() takes another dictionary as an argument and adds its key-value pairs to the
        # original dictionary. If there are overlapping keys, the values from the new dictionary will overwrite the existing ones. In this case, since the keys in binary_metrics are unique (e.g., 'binary_f1', 'binary_precision', etc.) and do not overlap with the keys in metrics (e.g., 'accuracy', 'macro_f1', etc.), all binary metrics will be added to the metrics dictionary without any overwriting.
        # can we remove 'metrics = compute_metrics' from the line above?
        # No, we still need to compute the main metrics first, then we update that dict with the binary metrics. 
        # The compute_metrics function computes the multi-class metrics (accuracy, macro_f1, etc.) while compute_metrics_binary computes the binary metrics after binarizing the labels. We want both sets of metrics in our final report, so we compute them separately and then merge them into one dictionary.
        # 
        #         
        # Return logits + labels alongside metrics so that post-hoc
        # calibration (step 7c/7d) can re-derive predictions without
        # re-running the forward pass through the model.
        return avg_loss, metrics, all_logits, all_labels
    
    def _is_dynamic_mode(self) -> bool:
        """Check if this mode needs dynamic data handling (skip initial prep).
        
        Returns True for modes that do NOT prepare a fixed augmented dataset
        before the training loop:
        - Ab-5: uses _train_epoch_ab5() for class-stratified per-batch BAGAN
        - Ab-6: uses _train_epoch_ab6() for Ab-5 + quality gate
        - Ab-7: uses _train_epoch_ab7() for Ab-5 + margin sampling
        - Ab-8: uses _run_al_loop_ab8() for iterative BAGAN acquisition
        - Ab-9: uses _run_al_loop_ab9() for pool-drain active learning
        - Ab-10: uses _train_epoch_ab10() for Ab-7 + G fine-tuning
        
        Static modes (ab1, ab1a, ab1b, ab4) prepare data once before the loop.
        
        Returns:
            True for ab5, ab6, ab7, ab8, ab9, ab10, ab11, ab11a, ab20.
            False for ab1, ab1a, ab1b, ab4.
        """
        return self.mode in ('ab5', 'ab6', 'ab7', 'ab8', 'ab8a', 'ab8b', 'ab8c', 'ab8d', 'ab8b_bagan', 'ab8b_without_gp', 'ab9', 'ab10', 'ab11', 'ab11a', 'ab20')
    
    def _remap_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Remap original class labels to contiguous indices [0, num_classes-1].
        
        WHY THIS IS NEEDED:
        CrossEntropyLoss requires targets in [0, num_classes-1].
        Our selected_classes may be non-contiguous (e.g., [0, 2, 3]),
        so label value 3 would be out of bounds for a 3-output classifier.
        
        This method converts:
            Original labels  -> Contiguous indices
            [0, 2, 3, 0, 3]  -> [0, 1, 2, 0, 2]
        
        using self.label_to_index = {0->, 2->, 3->} built in __init__.
        
        Args:
            labels: [N] tensor of original class labels (e.g., values in {0, 2, 3})
        
        Returns:
            [N] tensor of contiguous indices (e.g., values in {0, 1, 2})
        """
        return torch.tensor(
            [self.label_to_index[lbl.item()] for lbl in labels],
            dtype=labels.dtype
        )

    # ================================================================
    # Ab-8: Iterative BAGAN Acquisition (GAAL-style)
    # ================================================================
    #
    # Nielsen (2019) GAAL pipeline adapted for imbalanced multi-class:
    #   1. Warm-up: train on original imbalanced data for a few epochs.
    #   2. AL loop: each "active-learning cycle" adds a small batch of
    #      margin-selected BAGAN samples to the growing training set,
    #      then trains for several more epochs.
    #   3. Consolidation: a few final epochs on the complete balanced
    #      dataset to let the classifier settle its weights.
    #
    # The key idea: instead of dumping all synthetic data at once (Ab-4)
    # or refreshing every epoch (Ab-5/6/7), we incrementally grow the
    # pool across discrete "acquisition rounds" ->with cumulative data
    # retention and per-round margin selection.
    # ================================================================

    def _run_al_loop_ab8(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> Tuple['CNNClassifier', Dict[str, float]]:
        """Ab-8: Iterative BAGAN acquisition with margin sampling.

        This method REPLACES the standard epoch loop in train_fold() for Ab-8.
        It manages its own model, optimizer, early stopping, and epoch budget.

        Algorithm outline (refer to the comment block above for background):

            Phase 1 ->Warm-up
                Train on the original imbalanced data for ab8_warmup_epochs.
                This gives the classifier an initial decision boundary
                so that margin scoring is meaningful.

            Phase 2 ->Active-learning cycle loop
                For each AL cycle c = 1 ->num_al_cycles:
                    a. Generate candidates: ab8_pool_size_factor x ab8_samples_per_cycle
                       BAGAN candidates per minority class.
                    b. Margin-select: switch classifier to eval, compute margin
                       for each candidate, keep the ab8_samples_per_cycle with
                       LOWEST margin (most uncertain).
                    c. Append selected samples to the cumulative synthetic pool
                       (data grows across cycles ->never discarded).
                    d. Build the training set = original real data + cumulative
                       synthetic pool.
                    e. Train for epochs_per_cycle epochs on the grown dataset.
                    f. Validate ->track best binary_f1 ->early-stop check at
                       the cycle level (patience counted in AL cycles).

            Phase 3 ->Consolidation
                Restore best model weights, train ab8_consolidation_epochs
                more on the final balanced dataset (= real + all synthetic).

        Returns:
            model: CNNClassifier with best weights (by binary_f1).
            best_metrics: Dict of validation metrics at the best checkpoint,
                          including 'stopping_epoch' (total epochs trained
                          across warm-up + AL cycles + consolidation).
        """
        cfg = self.config

        # -> Compute deficit per minority class --
        # _compute_target_counts says how many samples each class should
        # have after augmentation (from config.augmentation_target).
        target_counts = _compute_target_counts(train_labels, cfg)

        deficits = {}
        for cls in cfg.minority_classes:
            current = (train_labels == cls).sum().item()
            deficits[cls] = max(0, target_counts[cls] - current)

        max_deficit = max(deficits.values()) if deficits else 0
        # How many AL cycles are needed to fill the largest deficit
        num_al_cycles = max(1, math.ceil(max_deficit / cfg.ab8_samples_per_cycle))

        # How many training epochs to run per AL cycle
        # Distribute the remaining epoch budget evenly across AL cycles
        remaining_epochs = max(1, cfg.num_epochs - cfg.ab8_warmup_epochs
                               - cfg.ab8_consolidation_epochs)
        epochs_per_cycle = max(1, remaining_epochs // num_al_cycles)

        print(f"      Ab-8 plan: {num_al_cycles} AL cycles x "
              f"{epochs_per_cycle} epochs/cycle, "
              f"warm-up={cfg.ab8_warmup_epochs}, "
              f"consolidation={cfg.ab8_consolidation_epochs}")
        print(f"      Deficits: { {cls: deficits[cls] for cls in cfg.minority_classes} }")

        # -> Create model, optimizer, criterion --
        model = CNNClassifier(
            input_channels=1,
            num_classes=cfg.num_classes
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # Remap labels once for the real data
        real_labels_remapped = self._remap_labels(train_labels)

        # Validation loader (fixed)
        val_labels_idx = self._remap_labels(val_labels)
        val_dataset = TensorDataset(val_data, val_labels_idx)
        val_loader = DeviceDataLoader(
            DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False),
            self.device
        )

        # Generator in eval mode for the entire AL loop
        gen_was_training = self.generator.training
        self.generator.eval()

        # -> Book-keeping for best-model tracking --
        # 125J: select / early-stop / log on the configured metric (macro_f2).
        _sel = cfg.selection_metric
        best_f1 = -1.0
        best_metrics = {}
        best_state = None               # copy.deepcopy of state_dict
        cycles_without_improvement = 0   # AL-cycle-level early stopping

        # Cumulative synthetic pool (grows each AL cycle)
        synth_data_list = []   # list of [n, 1, 64, 64] tensors
        synth_label_list = []  # list of [n] tensors (remapped labels)

        try:
            # --
            # PHASE 1 ->Warm-up on original imbalanced data
            # --
            train_ds = TensorDataset(train_data, real_labels_remapped)
            train_loader = DeviceDataLoader(
                DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
                self.device
            )

            for epoch in range(cfg.ab8_warmup_epochs):
                train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
                val_loss, metrics, _, _ = self._validate(model, val_loader, criterion)

                f1 = metrics[_sel]
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())

            print(f"      Warm-up done ->best {_sel} = {best_f1:.4f}")

            # --
            # PHASE 2 ->Active-learning acquisition loop
            # --
            for cycle in range(num_al_cycles):
                # -> 2a. Generate & margin-select per minority class --
                for cls in cfg.minority_classes:
                    remapped_c = self.label_to_index[cls]

                    # How many to add this cycle (don't overshoot deficit)
                    already_added = sum(
                        (lbl == remapped_c).sum().item()
                        for lbl in synth_label_list
                    )
                    remaining = deficits[cls] - already_added
                    n_add = min(cfg.ab8_samples_per_cycle, remaining)
                    if n_add <= 0:
                        continue

                    n_candidates = n_add * cfg.ab8_pool_size_factor

                    # Generate BAGAN candidates
                    z = self.latent_generator.sample_for_class_oversampling(
                        target_class=cls, n_samples=n_candidates
                    )
                    z = z.to(self.device)

                    with torch.no_grad():
                        candidates = self.generator(z)  # [n_candidates, 1, 64, 64]

                        # Margin scoring ->switch classifier to eval temporarily
                        model.eval()
                        logits = model(candidates)
                        model.train()

                        probs = torch.softmax(logits, dim=1)
                        sorted_probs, _ = probs.sort(dim=1, descending=True)
                        margin = sorted_probs[:, 0] - sorted_probs[:, 1]

                        # Keep n_add candidates with LOWEST margin
                        _, best_idx = torch.topk(margin, k=n_add, largest=False)
                        selected = candidates[best_idx].cpu()

                    synth_data_list.append(selected)
                    synth_label_list.append(
                        torch.full((n_add,), remapped_c, dtype=torch.long)
                    )

                # -> 2b. Build grown dataset (real + cumulative synthetic) ?
                if synth_data_list:
                    all_synth_data = torch.cat(synth_data_list, dim=0)
                    all_synth_labels = torch.cat(synth_label_list, dim=0)
                    grown_data = torch.cat([train_data, all_synth_data], dim=0)
                    grown_labels = torch.cat(
                        [real_labels_remapped, all_synth_labels], dim=0
                    )
                else:
                    grown_data = train_data
                    grown_labels = real_labels_remapped

                grown_ds = TensorDataset(grown_data, grown_labels)
                grown_loader = DeviceDataLoader(
                    DataLoader(grown_ds, batch_size=cfg.batch_size, shuffle=True),
                    self.device
                )

                ######################################################################
                # -> 2c. Train for epochs_per_cycle on grown dataset --
                for epoch in range(epochs_per_cycle):
                    train_loss = self._train_epoch(
                        model, grown_loader, optimizer, criterion
                    )
                ######################################################################
                
                # -> 2d. Validate & checkpoint --
                val_loss, metrics, _, _ = self._validate(
                    model, val_loader, criterion
                )
                f1 = metrics[_sel]

                total_synth = sum(len(t) for t in synth_data_list)
                print(f"      AL cycle {cycle+1}/{num_al_cycles}: "
                      f"synth_pool={total_synth}, "
                      f"{_sel}={f1:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())
                    cycles_without_improvement = 0
                else:
                    cycles_without_improvement += 1

                # -> 2e. Early stopping at the AL-cycle level --
                if cycles_without_improvement >= cfg.ab8_patience:
                    print(f"      Early stopping at AL cycle {cycle+1} "
                          f"(patience={cfg.ab8_patience})")
                    break

            # --
            # PHASE 3 ->Consolidation on final balanced dataset
            # --
            # Restore best model first, then fine-tune a few more epochs
            if best_state is not None:
                model.load_state_dict(best_state)

            # Re-build final grown dataset (same as last cycle)
            if synth_data_list:
                all_synth_data = torch.cat(synth_data_list, dim=0)
                all_synth_labels = torch.cat(synth_label_list, dim=0)
                final_data = torch.cat([train_data, all_synth_data], dim=0)
                final_labels = torch.cat(
                    [real_labels_remapped, all_synth_labels], dim=0
                )
            else:
                final_data = train_data
                final_labels = real_labels_remapped

            final_ds = TensorDataset(final_data, final_labels)
            final_loader = DeviceDataLoader(
                DataLoader(final_ds, batch_size=cfg.batch_size, shuffle=True),
                self.device
            )

            for epoch in range(cfg.ab8_consolidation_epochs):
                train_loss = self._train_epoch(
                    model, final_loader, optimizer, criterion
                )
                val_loss, metrics, _, _ = self._validate(
                    model, val_loader, criterion
                )
                f1 = metrics[_sel]
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())

            print(f"      Consolidation done ->final best {_sel} = {best_f1:.4f}")

        finally:
            # Restore generator training state
            self.generator.train(mode=gen_was_training)

        # Load absolute best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Record total epochs trained across all 3 phases.
        # cycle+1 = number of AL cycles actually completed (handles early stop).
        # This total is used as the retrain duration for Ab-8.
        al_cycles_completed = (cycle + 1) if num_al_cycles > 0 else 0
        total_epochs_trained = (
            cfg.ab8_warmup_epochs
            + al_cycles_completed * epochs_per_cycle
            + cfg.ab8_consolidation_epochs
        )
        best_metrics['stopping_epoch'] = total_epochs_trained

        # One extra _validate() on restored best weights to harvest
        # raw logits for Topic 7 (pragmatic asymmetry ->see 02a plan C2).
        _, _, best_val_logits, best_val_labels = self._validate(
            model, val_loader, criterion
        )
        return (
            model,
            best_metrics,
            best_val_logits.numpy(),
            best_val_labels.numpy(),
        )

    # --
    # Ab-8a: Ab-8 + BAGAN-GP aux-head label gate on candidates
    # --
    def _run_al_loop_ab8a(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> Tuple['CNNClassifier', Dict[str, float]]:
        """Ab-8a: Ab-8 with an independent-oracle label gate.

        Identical to ``_run_al_loop_ab8`` EXCEPT for Phase-2 step 2a:
        candidates are first filtered by the BAGAN-GP discriminator's
        auxiliary classifier head. Only candidates whose aux argmax
        matches the intended class AND whose aux softmax-max -- pass
        to the margin-based AL ranking.

        Requires self.discriminator to be a BAGAN-GP critic whose
        forward(x) returns (score, aux_logits). Verified for
        ``bagan_gp.models.discriminator.WGANAuxCriticNoBN``:
            - aux_head: nn.Linear(flat_dim, num_classes) ->size = num_classes
            - forward returns (score, aux_logits) tuple; we use index [1]
        """
        cfg = self.config

        if self.discriminator is None:
            raise RuntimeError(
                "Ab-8a requires a BAGAN-GP discriminator with an aux head; "
                "self.discriminator is None. Ensure _train_bagan_for_fold "
                "returned a discriminator and it was passed into "
                "ClassifierTrainer(discriminator=...)."
            )

        # -> Compute deficit per minority class (same as Ab-8) --
        target_counts = _compute_target_counts(train_labels, cfg)
        deficits = {}
        for cls in cfg.minority_classes:
            current = (train_labels == cls).sum().item()
            deficits[cls] = max(0, target_counts[cls] - current)

        max_deficit = max(deficits.values()) if deficits else 0
        num_al_cycles = max(1, math.ceil(max_deficit / cfg.ab8_samples_per_cycle))

        remaining_epochs = max(1, cfg.num_epochs - cfg.ab8_warmup_epochs
                               - cfg.ab8_consolidation_epochs)
        epochs_per_cycle = max(1, remaining_epochs // num_al_cycles)

        print(f"      Ab-8a plan: {num_al_cycles} AL cycles x "
              f"{epochs_per_cycle} epochs/cycle, "
              f"warm-up={cfg.ab8_warmup_epochs}, "
              f"consolidation={cfg.ab8_consolidation_epochs}, "
              f"aux_tau={cfg.ab8a_aux_tau} "
              f"(fallback {cfg.ab8a_aux_tau_fallback}), "
              f"pool_factor={cfg.ab8a_pool_size_factor}")
        print(f"      Deficits: { {cls: deficits[cls] for cls in cfg.minority_classes} }")

        # -> Model / optimizer / criterion --
        model = CNNClassifier(
            input_channels=1,
            num_classes=cfg.num_classes
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        real_labels_remapped = self._remap_labels(train_labels)
        val_labels_idx = self._remap_labels(val_labels)
        val_dataset = TensorDataset(val_data, val_labels_idx)
        val_loader = DeviceDataLoader(
            DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False),
            self.device
        )

        gen_was_training = self.generator.training
        disc_was_training = self.discriminator.training
        self.generator.eval()
        self.discriminator.eval()

        # 125J: select / early-stop / log on the configured metric (macro_f2).
        _sel = cfg.selection_metric
        best_f1 = -1.0
        best_metrics = {}
        best_state = None
        cycles_without_improvement = 0

        synth_data_list = []
        synth_label_list = []

        # Gate diagnostics (per-class per-cycle survivor counts)
        gate_log: List[str] = []

        try:
            # -> Phase 1: Warm-up on raw imbalanced real --
            train_ds = TensorDataset(train_data, real_labels_remapped)
            train_loader = DeviceDataLoader(
                DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
                self.device
            )
            for epoch in range(cfg.ab8_warmup_epochs):
                self._train_epoch(model, train_loader, optimizer, criterion)
                _, metrics, _, _ = self._validate(model, val_loader, criterion)
                f1 = metrics[_sel]
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())

            print(f"      Warm-up done ->best {_sel} = {best_f1:.4f}")

            # -> Phase 2: Active-learning acquisition loop --
            for cycle in range(num_al_cycles):
                # -> 2a. Gated generate & margin-select per minority class
                for cls in cfg.minority_classes:
                    remapped_c = self.label_to_index[cls]

                    already_added = sum(
                        (lbl == remapped_c).sum().item()
                        for lbl in synth_label_list
                    )
                    remaining = deficits[cls] - already_added
                    n_add = min(cfg.ab8_samples_per_cycle, remaining)
                    if n_add <= 0:
                        continue

                    n_candidates = n_add * cfg.ab8a_pool_size_factor

                    z = self.latent_generator.sample_for_class_oversampling(
                        target_class=cls, n_samples=n_candidates
                    )
                    z = z.to(self.device)

                    with torch.no_grad():
                        candidates = self.generator(z)

                        # -> Independent-oracle label gate --
                        aux_logits = self._adv_class_logits(self.discriminator, candidates)
                        # Defensive: if a discriminator with num_classes+1
                        # logits is ever wired in, drop the fake column.
                        # BAGAN-GP returns exactly num_classes ->no-op.
                        if aux_logits.shape[1] > cfg.num_classes:
                            aux_logits = aux_logits[:, :cfg.num_classes]
                        aux_probs = torch.softmax(aux_logits, dim=1)
                        aux_argmax = aux_probs.argmax(dim=1)
                        aux_max = aux_probs.max(dim=1).values

                        intended_match = (aux_argmax == remapped_c)
                        gate_a = intended_match & (aux_max >= cfg.ab8a_aux_tau)

                        if gate_a.sum().item() >= n_add:
                            survivor_idx = gate_a.nonzero(as_tuple=False).squeeze(1)
                            gate_used = 'A'
                        else:
                            gate_b = intended_match & (aux_max >= cfg.ab8a_aux_tau_fallback)
                            if gate_b.sum().item() >= n_add:
                                survivor_idx = gate_b.nonzero(as_tuple=False).squeeze(1)
                                gate_used = 'B'
                            else:
                                # Last resort: any argmax-correct candidate.
                                # NEVER inject argmax-mismatched samples.
                                survivor_idx = intended_match.nonzero(as_tuple=False).squeeze(1)
                                gate_used = 'C'

                        n_survivors = survivor_idx.numel()
                        if n_survivors == 0:
                            gate_log.append(
                                f"cycle {cycle+1} cls={cls}: 0 survivors "
                                f"({n_candidates} candidates) ->skipping"
                            )
                            continue

                        k = min(n_add, n_survivors)

                        if n_survivors == k:
                            selected_local = torch.arange(n_survivors, device=candidates.device)
                        else:
                            # Among survivors, keep lowest-margin (most informative)
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

                        gate_log.append(
                            f"cycle {cycle+1} cls={cls}: gate-{gate_used} "
                            f"{n_survivors}/{n_candidates} survivors, "
                            f"selected {k}"
                        )

                    synth_data_list.append(selected)
                    synth_label_list.append(
                        torch.full((k,), remapped_c, dtype=torch.long)
                    )

                # -> 2b. Build grown dataset --
                if synth_data_list:
                    all_synth_data = torch.cat(synth_data_list, dim=0)
                    all_synth_labels = torch.cat(synth_label_list, dim=0)
                    grown_data = torch.cat([train_data, all_synth_data], dim=0)
                    grown_labels = torch.cat(
                        [real_labels_remapped, all_synth_labels], dim=0
                    )
                else:
                    grown_data = train_data
                    grown_labels = real_labels_remapped

                grown_ds = TensorDataset(grown_data, grown_labels)
                grown_loader = DeviceDataLoader(
                    DataLoader(grown_ds, batch_size=cfg.batch_size, shuffle=True),
                    self.device
                )

                # -> 2c. Train classifier on grown set --
                for epoch in range(epochs_per_cycle):
                    self._train_epoch(
                        model, grown_loader, optimizer, criterion
                    )

                # -> 2d. Validate & checkpoint --
                _, metrics, _, _ = self._validate(
                    model, val_loader, criterion
                )
                f1 = metrics[_sel]

                total_synth = sum(len(t) for t in synth_data_list)
                print(f"      AL cycle {cycle+1}/{num_al_cycles}: "
                      f"synth_pool={total_synth}, "
                      f"{_sel}={f1:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())
                    cycles_without_improvement = 0
                else:
                    cycles_without_improvement += 1

                if cycles_without_improvement >= cfg.ab8_patience:
                    print(f"      Early stopping at AL cycle {cycle+1} "
                          f"(patience={cfg.ab8_patience})")
                    break

            # -> Phase 3: Consolidation --
            if best_state is not None:
                model.load_state_dict(best_state)

            if synth_data_list:
                all_synth_data = torch.cat(synth_data_list, dim=0)
                all_synth_labels = torch.cat(synth_label_list, dim=0)
                final_data = torch.cat([train_data, all_synth_data], dim=0)
                final_labels = torch.cat(
                    [real_labels_remapped, all_synth_labels], dim=0
                )
            else:
                final_data = train_data
                final_labels = real_labels_remapped

            final_ds = TensorDataset(final_data, final_labels)
            final_loader = DeviceDataLoader(
                DataLoader(final_ds, batch_size=cfg.batch_size, shuffle=True),
                self.device
            )

            for epoch in range(cfg.ab8_consolidation_epochs):
                self._train_epoch(
                    model, final_loader, optimizer, criterion
                )
                _, metrics, _, _ = self._validate(
                    model, val_loader, criterion
                )
                f1 = metrics[_sel]
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())

            print(f"      Consolidation done ->final best {_sel} = {best_f1:.4f}")

            if gate_log:
                tail = gate_log[-4:] if len(gate_log) > 4 else gate_log
                for line in tail:
                    print(f"      [gate] {line}")

        finally:
            self.generator.train(mode=gen_was_training)
            self.discriminator.train(mode=disc_was_training)

        if best_state is not None:
            model.load_state_dict(best_state)

        al_cycles_completed = (cycle + 1) if num_al_cycles > 0 else 0
        total_epochs_trained = (
            cfg.ab8_warmup_epochs
            + al_cycles_completed * epochs_per_cycle
            + cfg.ab8_consolidation_epochs
        )
        best_metrics['stopping_epoch'] = total_epochs_trained

        _, _, best_val_logits, best_val_labels = self._validate(
            model, val_loader, criterion
        )
        return (
            model,
            best_metrics,
            best_val_logits.numpy(),
            best_val_labels.numpy(),
        )

    # --
    # Ab-8b: Ab-8 with margin ->BADGE selector swap (no aux-gate)
    # --
    def _run_al_loop_ab8b(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> Tuple['CNNClassifier', Dict[str, float]]:
        """Ab-8b / Ab-8d: Ab-8 with the margin selector replaced.

        Identical to ``_run_al_loop_ab8`` EXCEPT for Phase-2 step 2a, where the
        acquisition rule is resolved from ``config.get_selection_params(mode)``:

        - ``ab8b`` defaults to **BADGE** (Ash et al. 2020) - k-means++ seeding
          on the gradient embedding g_i = (p_i - e_yhat_i) (x) h_i.
        - ``ab8d`` uses the **confidence-gated diverse** rule - reject any
          candidate the classifier does not confidently assign to the intended
          class, then k-means++ over the penultimate features of the survivors.

        Candidate-latent SEEDING is also resolved per mode via
        ``_ab8c_seed_latents(..., mode=self.mode)``: Ab-8d reads
        ``ab8d_seed_mode`` (132J default: ``"latent_smote"``, matching
        Proposed/Ab-8c), while Ab-8b has no dedicated field and keeps
        sampling the class-conditional prior.

        ``h_i`` is the penultimate activation captured by a forward hook on the
        classifier's final ``nn.Linear``. No aux-gate. Selection is per-class
        (mirrors the Ab-8 loop structure).
        """
        cfg = self.config

        # -> Acquisition rule for THIS mode (ab8b / ab8d / aliases) --
        _sel = cfg.get_selection_params(self.mode)
        _sel_mode = _sel['mode']
        _sel_min_conf = _sel['min_confidence']
        _sel_normalize = _sel['normalize']
        _sel_pool_factor = _sel['pool_factor']

        # -> Locate the final nn.Linear of the model template --
        # Used to register the penultimate-activation forward hook below.
        # We do this lazily inside the AL loop (after model is created).

        # -> Compute deficit per minority class (same as Ab-8) --
        target_counts = _compute_target_counts(train_labels, cfg)
        deficits = {}
        for cls in cfg.minority_classes:
            current = (train_labels == cls).sum().item()
            deficits[cls] = max(0, target_counts[cls] - current)

        max_deficit = max(deficits.values()) if deficits else 0
        num_al_cycles = max(1, math.ceil(max_deficit / cfg.ab8_samples_per_cycle))

        remaining_epochs = max(1, cfg.num_epochs - cfg.ab8_warmup_epochs
                               - cfg.ab8_consolidation_epochs)
        epochs_per_cycle = max(1, remaining_epochs // num_al_cycles)

        print(f"      {self.mode} plan: {num_al_cycles} AL cycles x "
              f"{epochs_per_cycle} epochs/cycle, "
              f"warm-up={cfg.ab8_warmup_epochs}, "
              f"consolidation={cfg.ab8_consolidation_epochs}, "
              f"pool_factor={_sel_pool_factor}, "
              f"kpp_normalize={_sel_normalize}, "
              f"selector={describe_selection(_sel_mode, _sel_min_conf)}")
        print(f"      Deficits: { {cls: deficits[cls] for cls in cfg.minority_classes} }")

        # -> Model / optimizer / criterion --
        model = CNNClassifier(
            input_channels=1,
            num_classes=cfg.num_classes
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # Locate final nn.Linear for BADGE penultimate hook
        final_linear = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                final_linear = m
        if final_linear is None:
            raise RuntimeError(
                "Ab-8b BADGE selector requires the classifier to contain "
                "at least one nn.Linear (none found in CNNClassifier)."
            )

        real_labels_remapped = self._remap_labels(train_labels)
        val_labels_idx = self._remap_labels(val_labels)
        val_dataset = TensorDataset(val_data, val_labels_idx)
        val_loader = DeviceDataLoader(
            DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False),
            self.device
        )

        gen_was_training = self.generator.training
        self.generator.eval()

        # 125J: select / early-stop / log on the configured selection metric
        # (macro_f2) so the CV-chosen stopping epoch matches how the model is
        # graded downstream (F test eval + retrain selection).
        _sel = cfg.selection_metric
        best_f1 = -1.0
        best_metrics = {}
        best_state = None
        cycles_without_improvement = 0

        synth_data_list = []
        synth_label_list = []

        try:
            # -> Phase 1: Warm-up on raw imbalanced real --
            train_ds = TensorDataset(train_data, real_labels_remapped)
            train_loader = DeviceDataLoader(
                DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
                self.device
            )
            for epoch in range(cfg.ab8_warmup_epochs):
                self._train_epoch(model, train_loader, optimizer, criterion)
                _, metrics, _, _ = self._validate(model, val_loader, criterion)
                f1 = metrics[_sel]
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())

            print(f"      Warm-up done ->best {_sel} = {best_f1:.4f}")

            # -> Phase 2: Active-learning acquisition loop --
            for cycle in range(num_al_cycles):
                # -> 2a. Generate + BADGE-select per minority class --
                for cls in cfg.minority_classes:
                    remapped_c = self.label_to_index[cls]

                    already_added = sum(
                        (lbl == remapped_c).sum().item()
                        for lbl in synth_label_list
                    )
                    remaining = deficits[cls] - already_added
                    n_add = min(cfg.ab8_samples_per_cycle, remaining)
                    if n_add <= 0:
                        continue

                    n_candidates = n_add * _sel_pool_factor

                    z = _ab8c_seed_latents(
                        self.latent_generator, cls, n_candidates, cfg,
                        mode=self.mode,
                    )
                    z = z.to(self.device)

                    with torch.no_grad():
                        candidates = self.generator(z)  # [n_cand, 1, 64, 64]

                        # -> Acquisition: penultimate features via forward hook ->
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

                    # Acquisition rule (BADGE or confidence-gated diverse) is
                    # resolved once per mode by config.get_selection_params().
                    kpp_idx = select_indices(
                        probs, h, n_add, remapped_c,
                        mode=_sel_mode,
                        min_confidence=_sel_min_conf,
                        normalize=_sel_normalize,
                        seed=cfg.run_seed,   # algorithmic seed (varies per run)
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
                    grown_data = torch.cat([train_data, all_synth_data], dim=0)
                    grown_labels = torch.cat(
                        [real_labels_remapped, all_synth_labels], dim=0
                    )
                else:
                    grown_data = train_data
                    grown_labels = real_labels_remapped

                grown_ds = TensorDataset(grown_data, grown_labels)
                grown_loader = DeviceDataLoader(
                    DataLoader(grown_ds, batch_size=cfg.batch_size, shuffle=True),
                    self.device
                )

                # -> 2c. Train classifier on grown set --
                for epoch in range(epochs_per_cycle):
                    self._train_epoch(
                        model, grown_loader, optimizer, criterion
                    )

                # -> 2d. Validate & checkpoint --
                _, metrics, _, _ = self._validate(
                    model, val_loader, criterion
                )
                f1 = metrics[_sel]

                total_synth = sum(len(t) for t in synth_data_list)
                print(f"      AL cycle {cycle+1}/{num_al_cycles}: "
                      f"synth_pool={total_synth}, "
                      f"{_sel}={f1:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())
                    cycles_without_improvement = 0
                else:
                    cycles_without_improvement += 1

                if cycles_without_improvement >= cfg.ab8_patience:
                    print(f"      Early stopping at AL cycle {cycle+1} "
                          f"(patience={cfg.ab8_patience})")
                    break

            # -> Phase 3: Consolidation --
            if best_state is not None:
                model.load_state_dict(best_state)

            if synth_data_list:
                all_synth_data = torch.cat(synth_data_list, dim=0)
                all_synth_labels = torch.cat(synth_label_list, dim=0)
                final_data = torch.cat([train_data, all_synth_data], dim=0)
                final_labels = torch.cat(
                    [real_labels_remapped, all_synth_labels], dim=0
                )
            else:
                final_data = train_data
                final_labels = real_labels_remapped

            final_ds = TensorDataset(final_data, final_labels)
            final_loader = DeviceDataLoader(
                DataLoader(final_ds, batch_size=cfg.batch_size, shuffle=True),
                self.device
            )

            for epoch in range(cfg.ab8_consolidation_epochs):
                self._train_epoch(
                    model, final_loader, optimizer, criterion
                )
                _, metrics, _, _ = self._validate(
                    model, val_loader, criterion
                )
                f1 = metrics[_sel]
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())

            print(f"      Consolidation done ->final best {_sel} = {best_f1:.4f}")

        finally:
            self.generator.train(mode=gen_was_training)

        if best_state is not None:
            model.load_state_dict(best_state)

        al_cycles_completed = (cycle + 1) if num_al_cycles > 0 else 0
        total_epochs_trained = (
            cfg.ab8_warmup_epochs
            + al_cycles_completed * epochs_per_cycle
            + cfg.ab8_consolidation_epochs
        )
        best_metrics['stopping_epoch'] = total_epochs_trained

        _, _, best_val_logits, best_val_labels = self._validate(
            model, val_loader, criterion
        )
        return (
            model,
            best_metrics,
            best_val_logits.numpy(),
            best_val_labels.numpy(),
        )

    # ================================================================
    # Ab-8c: BADGE Active Learning + 3-Player Co-Training (D ->G ->C)
    # ================================================================
    #
    # Same 3-phase outer skeleton as Ab-8b (warm-up ->AL cycles ->
    # consolidation), with BADGE k-means++ acquisition. Difference:
    # each AL cycle additionally fine-tunes the WGAN-GP critic (D) and
    # generator (G) using small learning rates and a static ?_D anchor
    # against ``G_ref`` (a frozen snapshot of G at end of warm-up).
    #
    # G_ref freeze rationale (Decision B = static reference):
    #   Anchoring against a moving target (current G) is degenerate
    #   (the anchor loss ->0 trivially). Freezing G_ref preserves the
    #   warm-up identity while letting G refine local detail.
    #
    # GAN-state restoration:
    #   Both G and D state_dicts are deep-copied on entry and restored
    #   in a ``finally`` block so downstream ablations (Ab-9, Ab-10, ->
    #   sharing the same fold's BAGAN see the original weights.
    #
    # Diagnostics (8c-R2 monitoring ->no auto-abort):
    #   Per cycle, log mean ``||grad_D||`` and ``D.aux`` accuracy on a
    #   subsample of the real validation split. A warning marker is
    #   printed if aux val acc drops more than 0.05 below the
    #   end-of-warm-up baseline.
    # ================================================================

    def _ab8c_refine_generator(
        self,
        pool_data: torch.Tensor,
        pool_labels: torch.Tensor,
        log=print,
    ) -> Dict[int, float]:
        """125J-v3: minority generator refinement (the real third player).

        Fine-tunes ``self.generator`` (critic ``self.discriminator`` frozen)
        to push minority output closer to the real minority distribution in
        the critic's feature space, validated by ACTUAL InceptionV3 FID with
        keep-best. Because epoch-0 == the warm-up G == Ab-8b's generator,
        keep-best guarantees the exported G's minority FID is never worse
        than Ab-8b's, and strictly better when headroom exists.

        Loss per minority class (G-only gradient, D frozen):
            fm     = mean( (mean_phi(G(z_c)) - mean_phi(real_c))^2
                           + (std_phi(G(z_c)) - std_phi(real_c))^2 )  # feat match
            anchor = mean( (G(z_c) - G_ref(z_c))^2 )                  # image drift

        Args:
            pool_data:   full-pool images (original-label space).
            pool_labels: full-pool ORIGINAL labels (not remapped).
            log:         logging callable.

        Returns:
            Best (lowest-mean) per-class minority FID dict achieved. Empty
            dict if refinement is disabled or no minority reals are present.
            Side effect: ``self.generator`` is left holding the best snapshot.
        """
        cfg = self.config
        if int(cfg.ab8c_g_refine_epochs) <= 0:
            return {}
        from .fid import fid_per_class

        device = self.device
        BAGAN_GP_BASE_LR = 1e-4
        BAGAN_GP_BETAS = (0.0, 0.9)

        # Real minority images keyed by ORIGINAL label (FID + feature target).
        real_by_class: Dict[int, torch.Tensor] = {}
        for cls in cfg.minority_classes:
            mask = (pool_labels == cls)
            if int(mask.sum().item()) == 0:
                continue
            real_by_class[int(cls)] = pool_data[mask].to(device)
        if not real_by_class:
            log("    [G-refine] no real minority samples ->skipping.")
            return {}

        D = self.discriminator
        D.eval()
        # Frozen real-minority feature statistics mean_phi, std_phi in D space.
        real_feat: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        with torch.no_grad():
            for cls, imgs in real_by_class.items():
                phi_r = self._disc_features(D, imgs).float()
                real_feat[cls] = (phi_r.mean(dim=0), phi_r.std(dim=0))

        # Frozen warm-up generator for the image-space drift anchor.
        G_ref = copy.deepcopy(self.generator)
        G_ref.eval()
        for p in G_ref.parameters():
            p.requires_grad_(False)
        # Freeze D for the entire refinement (pure feature matching ->no
        # adversarial gaming of a moving critic).
        d_req_grad = [p.requires_grad for p in D.parameters()]
        for p in D.parameters():
            p.requires_grad_(False)

        opt_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=BAGAN_GP_BASE_LR * cfg.ab8c_g_refine_lr_factor,
            betas=BAGAN_GP_BETAS,
        )

        def _measure_fid() -> Dict[int, float]:
            return fid_per_class(
                self.generator, self.latent_generator,
                classes=list(cfg.minority_classes),
                n_samples=int(cfg.ab8c_g_refine_fid_samples),
                device=device,
                ref_cache_dir=cfg.bagan_model_dir,
                date_suffix=cfg.date_suffix,
                real_images_by_class=real_by_class,
            )

        # FID measurement is non-deterministic (cuDNN Inception path); guard
        # it like the GP double-backward elsewhere.
        det_was_on = torch.are_deterministic_algorithms_enabled()
        if det_was_on:
            torch.use_deterministic_algorithms(False)

        try:
            base_fid = _measure_fid()
            best_mean = float(np.mean(list(base_fid.values())))
            best_fid = dict(base_fid)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in self.generator.state_dict().items()
            }
            _pretty = ' '.join(f'c{c}={v:.1f}' for c, v in base_fid.items())
            log(
                f"    [G-refine] warm-up minority FID mean={best_mean:.1f} "
                f"({_pretty}) ->baseline (== Ab-8b floor)."
            )

            for ep in range(int(cfg.ab8c_g_refine_epochs)):
                self.generator.train()
                for _ in range(int(cfg.ab8c_g_refine_steps_per_epoch)):
                    opt_G.zero_grad(set_to_none=True)
                    loss = torch.zeros((), device=device)
                    for cls in real_feat.keys():
                        z = self.latent_generator.sample_for_class_oversampling(
                            target_class=cls,
                            n_samples=int(cfg.ab8c_g_refine_batch),
                        ).to(device)
                        fake = self.generator(z)
                        phi = self._disc_features(D, fake).float()
                        mu_r, sd_r = real_feat[cls]
                        fm = (
                            ((phi.mean(dim=0) - mu_r) ** 2).mean()
                            + ((phi.std(dim=0) - sd_r) ** 2).mean()
                        )
                        with torch.no_grad():
                            fake_ref = G_ref(z)
                        anchor = ((fake - fake_ref) ** 2).mean()
                        loss = (
                            loss
                            + cfg.ab8c_g_refine_fm_weight * fm
                            + cfg.ab8c_g_refine_anchor_weight * anchor
                        )
                    loss.backward()
                    opt_G.step()

                fid_now = _measure_fid()
                mean_now = float(np.mean(list(fid_now.values())))
                improved = mean_now < best_mean
                if improved:
                    best_mean = mean_now
                    best_fid = dict(fid_now)
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.generator.state_dict().items()
                    }
                _pretty = ' '.join(f'c{c}={v:.1f}' for c, v in fid_now.items())
                log(
                    f"    [G-refine] epoch {ep+1}/{cfg.ab8c_g_refine_epochs}: "
                    f"minority FID mean={mean_now:.1f} ({_pretty})"
                    f"{'  best ->snapshotted G' if improved else ''}"
                )

            # Restore best-FID generator (>= Ab-8b quality by construction).
            self.generator.load_state_dict(best_state)
            _pretty = ' '.join(f'c{c}={v:.1f}' for c, v in best_fid.items())
            log(
                f"    [G-refine] done ->exporting best minority FID "
                f"mean={best_mean:.1f} ({_pretty})."
            )
            return best_fid
        finally:
            for p, req in zip(D.parameters(), d_req_grad):
                p.requires_grad_(req)
            if det_was_on:
                torch.use_deterministic_algorithms(True)

    def _run_al_loop_ab8c(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> Tuple[CNNClassifier, Dict[str, float], np.ndarray, np.ndarray]:
        """Ab-8c: BADGE AL + 3-player co-training (D -> G -> C).

          This method REPLACES the standard epoch loop in ``train_fold()`` for
          Ab-8c. It manages classifier training, per-cycle BADGE acquisition,
          WGAN-GP critic/generator fine-tuning, cycle-level early stopping,
          and the final consolidation stage.

          Algorithm outline:

                Phase 0 - Planning and safety setup
                     1. Compute minority deficits toward the configured target
                         distribution (via ``_compute_target_counts``).
                     2. Derive ``num_al_cycles`` from the largest deficit and
                         ``ab8_samples_per_cycle``.
                     3. Derive ``epochs_per_cycle`` from the remaining epoch budget
                         after warm-up and consolidation.
                     4. Snapshot original G/D state_dicts and RNG-determinism mode
                         so the method can restore shared GAN state on exit.

                Phase 1 - Warm-up classifier on real imbalanced data
                     1. Train C only for ``ab8_warmup_epochs`` on real fold data.
                     2. Track best checkpoint by ``cfg.selection_metric``.
                     3. Freeze a reference generator ``G_ref`` at warm-up end.
                         It anchors future G updates (static reference).
                     4. Measure warm-up D.aux validation accuracy baseline
                         (overall and per class) for later gate diagnostics.

                Phase 2 - Active-learning cycle loop (c = 1 .. num_al_cycles)
                     a. Acquisition (BADGE, per minority class)
                         - Generate ``ab8c_pool_size_factor * n_add`` candidates.
                         - Build BADGE embeddings from classifier penultimate
                            activations via final-linear hook.
                         - Select ``n_add`` samples with k-means++ seeding.
                         - Append selected samples to cumulative synthetic pool.

                     b. Quality gate decision (Plan1)
                         - Compare current per-class D.aux val accuracy against
                            warm-up baseline.
                         - Maintain hysteresis-based degraded class set.
                         - In ``mask`` mode: drop degraded-class synth for selected
                            steps. In ``rescue`` mode: keep but down-weight in C-step.

                     c. D-step (critic fine-tuning, repeated ``ab8c_k_d`` times)
                         - Train D with WGAN-GP critic loss on real vs fake.
                         - Include aux supervision on real labels and optional
                            synth extras depending on gate mode.
                         - Log ``||grad_D||`` diagnostic statistics.

                     d. G-step (generator fine-tuning, repeated ``ab8c_k_g`` times)
                         - Sample class-balanced latent batch.
                         - Optimize weighted objective:
                            adv + aux_g + cls + static feature anchor.
                         - Anchor uses discriminator penultimate features between
                            current ``G(z)`` and frozen ``G_ref(z)``.
                         - Anchor weight is class-dependent (majority/minority).

                     e. C-step (classifier retraining)
                         - Build grown dataset = real + cumulative synth
                            (after gate/mode handling).
                         - Train classifier for ``epochs_per_cycle`` epochs.

                     f. Optional synthetic pool refresh (Rec3)
                         - If enabled and still on improvement track, resample synth
                            content from current G while preserving class counts.
                         - Optional score guard rejects refreshes that degrade D
                            realism score beyond tolerance.

                     g. Validate + checkpoint + patience
                         - Validate by ``cfg.selection_metric``.
                         - Snapshot best C (and aligned G/D/synth context).
                         - Stop cycle loop when ``ab8c_patience`` is exceeded.

                Phase 3 - Consolidation
                     1. Restore best checkpoint context from cycles.
                     2. Train C for ``ab8_consolidation_epochs`` on final grown set.
                     3. Keep best checkpoint by ``cfg.selection_metric``.

                Phase 4 - Restore shared GAN state and return
                     1. Always restore original G/D state_dicts (finally block)._re
                     2. Restore deterministic-algorithms mode if it was enabled.
                     3. Return best classifier, best metrics, and best-val logits/
                         labels for downstream calibration.

                Returns:
                model: CNNClassifier with best weights.
                best_metrics: Validation metrics at the best checkpoint,
                     including ``stopping_epoch`` over warm-up + completed cycles
                     + consolidation.
                best_val_logits: Raw validation logits at the restored best model.
                best_val_labels: Validation labels paired with best_val_logits.
                """
        from bagan_gp.training.gan_trainer import wgan_gp_critic_loss

        cfg = self.config

        # -> Acquisition rule for Ab-8c (131J default: confidence-gated) --
        _sel = cfg.get_selection_params('ab8c')
        _sel_mode = _sel['mode']
        _sel_min_conf = _sel['min_confidence']
        _sel_normalize = _sel['normalize']
        _sel_pool_factor = _sel['pool_factor']

        if self.discriminator is None:
            raise RuntimeError(
                "Ab-8c requires a BAGAN-GP discriminator with an aux head; "
                "self.discriminator is None. Ensure _train_bagan_for_fold "
                "returned a discriminator and it was passed into "
                "ClassifierTrainer(discriminator=...)."
            )
        if getattr(cfg, 'gan_backend', 'bagan') != 'bagan_gp':
            raise RuntimeError(
                "Ab-8c requires gan_backend='bagan_gp' (uses the WGAN-GP "
                "critic loss). Current gan_backend = "
                f"{getattr(cfg, 'gan_backend', None)!r}."
            )

        # -> Cycle planning (mirrors Ab-8/8a/8b) --
        # 131J: Ab-8c uses its OWN samples-per-cycle and warm-up so it matches
        # casestudy/methods/m10_ab8c.py, while Ab-8/8a/8b/8d keep sharing the
        # ab8_* budget. Falls back to the shared values if the fields are absent.
        _ab8c_spc = getattr(cfg, 'ab8c_samples_per_cycle', cfg.ab8_samples_per_cycle)
        _ab8c_warmup = getattr(cfg, 'ab8c_warmup_epochs', cfg.ab8_warmup_epochs)

        target_counts = _compute_target_counts(train_labels, cfg)
        deficits = {}
        for cls in cfg.minority_classes:
            current = (train_labels == cls).sum().item()
            deficits[cls] = max(0, target_counts[cls] - current)
        max_deficit = max(deficits.values()) if deficits else 0
        num_al_cycles = max(1, math.ceil(max_deficit / _ab8c_spc))
        remaining_epochs = max(
            1,
            cfg.num_epochs - _ab8c_warmup - cfg.ab8_consolidation_epochs,
        )
        epochs_per_cycle = max(1, remaining_epochs // num_al_cycles)

        # WGAN-GP base hyperparameters (mirrors BAGANGPConfig defaults).
        # The Ab-8c fine-tune lrs are these scaled by ab8c_lr_d/g_factor.
        BAGAN_GP_BASE_LR = 1e-4
        BAGAN_GP_BETAS = (0.0, 0.9)
        BAGAN_GP_GP_LAMBDA = 10.0

        print(
            f"      Ab-8c plan: {num_al_cycles} AL cycles x "
            f"{epochs_per_cycle} epochs/cycle, "
            f"warm-up={_ab8c_warmup}, "
            f"consolidation={cfg.ab8_consolidation_epochs}, "
            f"pool_factor={_sel_pool_factor}, "
            f"selector={describe_selection(_sel_mode, _sel_min_conf)}, "
            f"K_D={cfg.ab8c_k_d}, K_G={cfg.ab8c_k_g}, "
            f"lr_d_ft={BAGAN_GP_BASE_LR * cfg.ab8c_lr_d_factor:.1e}, "
            f"lr_g_ft={BAGAN_GP_BASE_LR * cfg.ab8c_lr_g_factor:.1e}"
        )
        print(
            f"      : aux_real={cfg.ab8c_lambda_aux_real}, "
            f"aux_synth={cfg.ab8c_lambda_aux_synth}, "
            f"adv={cfg.ab8c_lambda_adv}, "
            f"aux_g={cfg.ab8c_lambda_aux_g}, "
            f"cls={cfg.ab8c_lambda_cls}, "
            f"anchor(maj/min)={cfg.ab8c_lambda_anchor_majority}/{cfg.ab8c_lambda_anchor_minority}"
        )
        print(
            f"      Plan1 gate: enabled={cfg.ab8c_aux_gate_enabled}, "
            f"mode={cfg.ab8c_aux_gate_mode}, drop={cfg.ab8c_aux_gate_drop}, "
            f"rescue_w={cfg.ab8c_aux_rescue_synth_weight}; "
            f"Rec3 refresh={cfg.ab8c_synth_refresh} "
            f"(Plan3 guard={cfg.ab8c_refresh_quality_guard}, tol={cfg.ab8c_refresh_score_tol})"
        )
        print(
            f"      Lever1/2: balanced_sampler={cfg.ab8c_balanced_sampler}, "
            f"label_smoothing={cfg.ab8c_label_smoothing}"
        )
        print(
            f"      125J-v2: freeze_generator={cfg.ab8c_freeze_generator} "
            f"(G-step {'SKIPPED' if cfg.ab8c_freeze_generator else 'ON'}), "
            f"D.aux acq filter ?={cfg.ab8c_acq_filter_tau} "
            f"(min_keep_frac={cfg.ab8c_acq_filter_min_keep_frac})"
        )
        print(f"      Deficits: { {cls: deficits[cls] for cls in cfg.minority_classes} }")

        # -> Model + optimizer + criterion --
        model = CNNClassifier(
            input_channels=1,
            num_classes=cfg.num_classes,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.ab8c_label_smoothing))

        # -> Lever 1: class-balanced C-step DataLoader factory --
        # Builds a loader whose minibatches are roughly class-balanced via
        # WeightedRandomSampler (per-sample w = 1 / class_count[y]). Used
        # for the per-cycle ``grown_loader`` and Phase-3 ``final_loader``.
        def _build_c_step_loader(data: torch.Tensor, labels: torch.Tensor):
            ds = TensorDataset(data, labels)
            if cfg.ab8c_balanced_sampler and labels.numel() > 0:
                lbl_cpu = labels.detach().cpu().long()
                counts = torch.bincount(lbl_cpu, minlength=cfg.num_classes).clamp(min=1)
                w_per_class = 1.0 / counts.float()
                sample_w = w_per_class[lbl_cpu]
                sampler = WeightedRandomSampler(
                    weights=sample_w.tolist(),
                    num_samples=len(ds),
                    replacement=True,
                )
                inner = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler)
            else:
                inner = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
            return DeviceDataLoader(inner, self.device)

        # Locate final nn.Linear for BADGE penultimate-feature hook.
        final_linear: Optional[nn.Linear] = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                final_linear = m
        if final_linear is None:
            raise RuntimeError(
                "Ab-8c BADGE selector requires the classifier to contain "
                "at least one nn.Linear (none found in CNNClassifier)."
            )

        real_labels_remapped = self._remap_labels(train_labels)
        val_labels_idx = self._remap_labels(val_labels)
        val_dataset = TensorDataset(val_data, val_labels_idx)
        val_loader = DeviceDataLoader(
            DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False),
            self.device,
        )

        # -> Save original GAN states; restore on exit --
        orig_G_state = copy.deepcopy(self.generator.state_dict())
        orig_D_state = copy.deepcopy(self.discriminator.state_dict())
        gen_was_training = self.generator.training
        disc_was_training = self.discriminator.training

        # -> GAN fine-tune optimizers --
        opt_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=BAGAN_GP_BASE_LR * cfg.ab8c_lr_d_factor,
            betas=BAGAN_GP_BETAS,
        )
        opt_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=BAGAN_GP_BASE_LR * cfg.ab8c_lr_g_factor,
            betas=BAGAN_GP_BETAS,
        )

        best_f1 = -1.0
        best_metrics: Dict[str, float] = {}
        best_state = None
        # S1: snapshot G/D/synth alongside C whenever best F1 improves so
        # Phase-3 consolidation refines C against the *same* (G, D, pool)
        # context that produced the peak. G/D snapshots are decorative for
        # the CV pass (orig G/D are restored on exit), but the synth pool
        # snapshot is substantive ->Phase-3 retrains C on this pool.
        best_G_state = None
        best_D_state = None
        best_synth_data: Optional[List[torch.Tensor]] = None
        best_synth_labels: Optional[List[torch.Tensor]] = None
        cycles_without_improvement = 0
        synth_data_list: List[torch.Tensor] = []  # CPU tensors
        synth_label_list: List[torch.Tensor] = []
        cycle = -1  # so al_cycles_completed = 0 if loop body never runs

        # ``use_deterministic_algorithms`` must be off during the gradient
        # penalty (Conv2d's double-backward path is non-deterministic).
        det_was_on = torch.are_deterministic_algorithms_enabled()
        if det_was_on:
            torch.use_deterministic_algorithms(False)

        G_ref: Optional[nn.Module] = None
        aux_real_acc_warmup = 0.0

        try:
            # -> Phase 1: Warm-up classifier on raw real data --
            self.generator.eval()
            self.discriminator.eval()
            train_ds = TensorDataset(train_data, real_labels_remapped)
            train_loader_real = DeviceDataLoader(
                DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
                self.device,
            )
            for _ in range(_ab8c_warmup):
                self._train_epoch(model, train_loader_real, optimizer, criterion)
                _, metrics, _, _ = self._validate(model, val_loader, criterion)
                # 123J-1A: select on cfg.selection_metric (macro_f2) so the
                # kept state matches how the model is graded downstream.
                f1 = metrics[cfg.selection_metric]
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())
                    # S1: synth pool is still empty during warm-up; G/D
                    # match orig states (Phase-1 only trains C).
                    best_G_state = copy.deepcopy(self.generator.state_dict())
                    best_D_state = copy.deepcopy(self.discriminator.state_dict())
                    best_synth_data = []
                    best_synth_labels = []

            # Snapshot G_ref (frozen)
            G_ref = copy.deepcopy(self.generator)
            G_ref.eval()
            for p in G_ref.parameters():
                p.requires_grad_(False)

            # Aux-real baseline (8c-R2 diagnostics)
            aux_real_acc_warmup = self._ab8c_aux_real_val_acc(val_data, val_labels_idx)
            # Per-class baseline (Rec 1 gate)
            aux_warmup_per_class = self._ab8c_aux_real_val_acc_per_class(
                val_data, val_labels_idx, cfg.num_classes,
            )
            print(
                f"      Warm-up done ->best {cfg.selection_metric}={best_f1:.4f}, "
                f"D.aux val acc={aux_real_acc_warmup:.3f}, "
                f"per-class={ {c: round(v, 3) for c, v in aux_warmup_per_class.items()} }"
            )

            # -> Plan2: per-class anchor weights (majority vs minority) ->
            minority_remapped = {
                self.label_to_index[cls] for cls in cfg.minority_classes
            }
            anchor_w_per_class = torch.tensor(
                [
                    cfg.ab8c_lambda_anchor_minority
                    if c in minority_remapped
                    else cfg.ab8c_lambda_anchor_majority
                    for c in range(cfg.num_classes)
                ],
                dtype=torch.float, device=self.device,
            )
            # -> Plan1: hysteresis state for the aux-quality gate --
            currently_gated: set = set()

            # -> Phase 2: AL cycles --
            for cycle in range(num_al_cycles):
                # -> (a) BADGE acquisition per minority class --
                self.generator.eval()
                self.discriminator.eval()
                model.eval()

                cycle_S_list: List[torch.Tensor] = []  # device tensors for this cycle
                cycle_y_list: List[torch.Tensor] = []
                for cls in cfg.minority_classes:
                    remapped_c = self.label_to_index[cls]
                    already_added = sum(
                        int((lbl == remapped_c).sum().item())
                        for lbl in synth_label_list
                    )
                    remaining = deficits[cls] - already_added
                    n_add = min(_ab8c_spc, remaining)
                    if n_add <= 0:
                        continue
                    n_candidates = n_add * _sel_pool_factor
                    z = _ab8c_seed_latents(
                        self.latent_generator, cls, n_candidates, cfg
                    ).to(self.device)
                    with torch.no_grad():
                        candidates = self.generator(z)

                    # -> 125J-v2: D.aux acquisition filter --
                    # Drop candidates the (D-step-sharpened) aux head won't
                    # confidently assign to the target minority class, so
                    # BADGE selects diversity from a clean, on-manifold pool.
                    if cfg.ab8c_acq_filter_tau > 0.0:
                        with torch.no_grad():
                            _aux_cand = self._adv_class_logits(self.discriminator, candidates)
                            _aux_p = torch.softmax(_aux_cand, dim=1)[:, remapped_c]
                        _keep = _aux_p >= cfg.ab8c_acq_filter_tau
                        _min_keep = max(
                            n_add,
                            int(cfg.ab8c_acq_filter_min_keep_frac
                                * candidates.size(0)),
                        )
                        if int(_keep.sum().item()) >= _min_keep:
                            candidates = candidates[_keep]
                        else:
                            # Too few pass tau ->fall back to the top-scoring
                            # candidates by aux prob so acquisition never starves.
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
                    # Acquisition rule resolved once per mode; 131J default is
                    # the confidence-gated diverse rule, matching the
                    # case-study implementation.
                    kpp_idx = select_indices(
                        probs, h, n_add, remapped_c,
                        mode=_sel_mode,
                        min_confidence=_sel_min_conf,
                        normalize=_sel_normalize,
                        seed=cfg.run_seed,   # algorithmic seed (varies per run)
                    )
                    k = int(kpp_idx.numel())
                    selected = candidates[kpp_idx].detach()
                    y_selected = torch.full(
                        (k,), remapped_c, dtype=torch.long, device=self.device,
                    )
                    cycle_S_list.append(selected)
                    cycle_y_list.append(y_selected)
                    # Persist to cumulative synth pool (CPU)
                    synth_data_list.append(selected.detach().cpu())
                    synth_label_list.append(y_selected.detach().cpu())

                model.train()

                cycle_S = torch.cat(cycle_S_list, dim=0) if cycle_S_list else None
                cycle_y = torch.cat(cycle_y_list, dim=0) if cycle_y_list else None

                # -> Plan1: gate with hysteresis + mode --
                # Enter gate when (base - cur) > drop; exit only when
                # recovered to within drop/2. Mode is 'mask' (legacy)
                # or 'rescue' (keep signals alive; only C-step downsamples).
                degraded_classes: set = set()
                if cfg.ab8c_aux_gate_enabled:
                    aux_now_pc = self._ab8c_aux_real_val_acc_per_class(
                        val_data, val_labels_idx, cfg.num_classes,
                    )
                    exit_thresh = cfg.ab8c_aux_gate_drop / 2.0
                    for c in range(cfg.num_classes):
                        base = aux_warmup_per_class.get(c, float('nan'))
                        cur = aux_now_pc.get(c, float('nan'))
                        if math.isnan(base) or math.isnan(cur):
                            continue
                        drop = base - cur
                        if c in currently_gated:
                            if drop > exit_thresh:
                                degraded_classes.add(c)  # still gated
                        else:
                            if drop > cfg.ab8c_aux_gate_drop:
                                degraded_classes.add(c)  # newly gated
                    currently_gated = degraded_classes.copy()
                gate_mode = cfg.ab8c_aux_gate_mode
                deg_t_cpu = (
                    torch.tensor(sorted(degraded_classes), dtype=torch.long)
                    if degraded_classes else None
                )
                deg_t_dev = (
                    deg_t_cpu.to(self.device) if deg_t_cpu is not None else None
                )

                # D-step extras: drop only in 'mask' mode; rescue keeps all
                # so D.aux receives synth aux signal too (the real-data
                # _aux_real=2.0 already rescues the head).
                if (
                    gate_mode == 'mask'
                    and cycle_S is not None
                    and deg_t_dev is not None
                ):
                    keep = (cycle_y.unsqueeze(1) != deg_t_dev.unsqueeze(0)).all(dim=1)
                    cycle_S_d = cycle_S[keep] if keep.any() else None
                    cycle_y_d = cycle_y[keep] if keep.any() else None
                else:
                    cycle_S_d = cycle_S
                    cycle_y_d = cycle_y

                # -> (b) D-step: K_D minibatches of WGAN-GP critic loss ->
                self.generator.eval()
                self.discriminator.train()
                grad_d_norms: List[float] = []
                n_real = train_data.size(0)
                for _ in range(cfg.ab8c_k_d):
                    idx = torch.randint(0, n_real, (cfg.batch_size,))
                    real_batch = train_data[idx].to(self.device, non_blocking=True)
                    real_y_batch = real_labels_remapped[idx].to(
                        self.device, non_blocking=True
                    )
                    z, _ = self._ab8c_sample_balanced_latent(cfg.batch_size)
                    with torch.no_grad():
                        fake_batch = self.generator(z)

                    if cycle_S_d is not None and cycle_S_d.size(0) > 0:
                        s_sz = min(cfg.batch_size, cycle_S_d.size(0))
                        s_idx = torch.randint(0, cycle_S_d.size(0), (s_sz,))
                        extra_x = cycle_S_d[s_idx]
                        extra_y = cycle_y_d[s_idx]
                        extra_w = cfg.ab8c_lambda_aux_synth
                    else:
                        extra_x = None
                        extra_y = None
                        extra_w = 0.0

                    opt_D.zero_grad(set_to_none=True)
                    loss_d, _info_d = wgan_gp_critic_loss(
                        self.discriminator,
                        real_batch, fake_batch, real_y_batch,
                        gp_lambda=BAGAN_GP_GP_LAMBDA,
                        aux_w=cfg.ab8c_lambda_aux_real,
                        extra_x=extra_x, extra_y=extra_y, extra_aux_w=extra_w,
                        device=self.device,
                    )
                    loss_d.backward()
                    # Diagnostic: ||grad_D||->(global)
                    total_sq = 0.0
                    for p in self.discriminator.parameters():
                        if p.grad is not None:
                            total_sq += float(p.grad.detach().norm().item() ** 2)
                    grad_d_norms.append(total_sq ** 0.5)
                    opt_D.step()

                # -> (c) G-step: K_G minibatches --
                # 125J-v2: when ab8c_freeze_generator is True the generator
                # is held at its strong warm-up state ->the third player (D)
                # only sharpens its aux head as an acquisition critic, and we
                # skip the G-step entirely. This guarantees minority FID ties
                # Ab-8b (the floor) and stops drifting-G samples poisoning C.
                if not cfg.ab8c_freeze_generator:
                    self.generator.train()
                    self.discriminator.eval()
                    # Freeze D's params during the G-step so its optimizer state
                    # is untouched and we don't accidentally update D.
                    for p in self.discriminator.parameters():
                        p.requires_grad_(False)
                    try:
                        for _ in range(cfg.ab8c_k_g):
                            z, y_z = self._ab8c_sample_balanced_latent(cfg.batch_size)
                            fake = self.generator(z)
                            s_fake, aux_fake_logits = self.discriminator(fake)
                            adv = -s_fake.mean()

                            # Plan1: in 'mask' mode build a per-sample keep mask;
                            # in 'rescue' mode keep all classes' G-step signals
                            # alive (only C-step downsamples).
                            if gate_mode == 'mask' and deg_t_dev is not None:
                                keep_mask = (
                                    y_z.unsqueeze(1) != deg_t_dev.unsqueeze(0)
                                ).all(dim=1).float()
                            else:
                                keep_mask = torch.ones_like(y_z, dtype=torch.float)
                            n_keep = keep_mask.sum().clamp(min=1.0)

                            aux_g_per = F.cross_entropy(
                                aux_fake_logits, y_z, reduction='none',
                            )
                            aux_g = (aux_g_per * keep_mask).sum() / n_keep
                            # CE(C(G(z)), y_z) ->C in eval mode (BN/dropout frozen).
                            # Rec 2: _cls is the dominant teacher; aux_g is off by default.
                            was_c_training = model.training
                            model.eval()
                            cls_logits = model(fake)
                            if was_c_training:
                                model.train()
                            cls_l_per = F.cross_entropy(
                                cls_logits, y_z, reduction='none',
                            )
                            cls_l = (cls_l_per * keep_mask).sum() / n_keep
                            # Static ?_D anchor against frozen G_ref, weighted
                            # per-class (Plan2): majority kept anchored, minority
                            # free to improve.
                            with torch.no_grad():
                                fake_ref = G_ref(z)
                                phi_ref = self._disc_features(
                                    self.discriminator, fake_ref
                                ).detach()
                            phi = self._disc_features(self.discriminator, fake)
                            anchor_per = ((phi - phi_ref) ** 2).mean(dim=1)
                            anchor_w_b = anchor_w_per_class[y_z]
                            anchor = (anchor_per * anchor_w_b).mean()

                            g_loss = (
                                cfg.ab8c_lambda_adv * adv
                                + cfg.ab8c_lambda_aux_g * aux_g
                                + cfg.ab8c_lambda_cls * cls_l
                                + anchor  # per-class weight already applied
                            )
                            opt_G.zero_grad(set_to_none=True)
                            g_loss.backward()
                            opt_G.step()
                    finally:
                        for p in self.discriminator.parameters():
                            p.requires_grad_(True)

                # -> (d) C-step: epochs_per_cycle on real ->(gated) S --
                self.generator.eval()
                self.discriminator.eval()
                if synth_data_list:
                    all_synth_data = torch.cat(synth_data_list, dim=0)
                    all_synth_labels = torch.cat(synth_label_list, dim=0)
                    if deg_t_cpu is not None:
                        if gate_mode == 'mask':
                            # legacy: drop degraded-class synth entirely
                            keep_cpu = (
                                all_synth_labels.unsqueeze(1)
                                != deg_t_cpu.unsqueeze(0)
                            ).all(dim=1)
                            all_synth_data = all_synth_data[keep_cpu]
                            all_synth_labels = all_synth_labels[keep_cpu]
                        else:
                            # Plan1 'rescue': downsample degraded-class synth
                            # to ``ab8c_aux_rescue_synth_weight`` (default 0.3)
                            # so C still sees them but with reduced weight.
                            keep_prob = float(cfg.ab8c_aux_rescue_synth_weight)
                            mask_full = torch.ones(
                                all_synth_labels.size(0), dtype=torch.bool,
                            )
                            for c in degraded_classes:
                                c_idx = (
                                    all_synth_labels == c
                                ).nonzero(as_tuple=True)[0]
                                n_c = c_idx.size(0)
                                n_keep_c = int(round(n_c * keep_prob))
                                if n_keep_c < n_c:
                                    perm = c_idx[torch.randperm(n_c)]
                                    drop_idx = perm[n_keep_c:]
                                    mask_full[drop_idx] = False
                            all_synth_data = all_synth_data[mask_full]
                            all_synth_labels = all_synth_labels[mask_full]
                    if all_synth_data.numel() > 0:
                        grown_data = torch.cat([train_data, all_synth_data], dim=0)
                        grown_labels = torch.cat(
                            [real_labels_remapped, all_synth_labels], dim=0
                        )
                    else:
                        grown_data = train_data
                        grown_labels = real_labels_remapped
                else:
                    grown_data = train_data
                    grown_labels = real_labels_remapped
                grown_loader = _build_c_step_loader(grown_data, grown_labels)
                for _ in range(epochs_per_cycle):
                    self._train_epoch(model, grown_loader, optimizer, criterion)

                # -> Rec 3: refresh synth pool from CURRENT G --
                # Preserve each remapped class's stored count; replace
                # content with samples decoded by the just-updated G.
                # BADGE selection is NOT re-run; this is a pure resample.
                # Plan3 guard: if current D scores the new batch lower than
                # the old samples (by more than ``refresh_score_tol``),
                # the refresh is rejected for that class ->keep old.
                # S2: freeze refresh once F1 stops improving. Uses the
                # previous cycle's ``cycles_without_improvement`` (which is
                # updated below, after validation), so refresh runs only
                # while still on the improvement track.
                refresh_rejected: List[int] = []
                if (
                    cfg.ab8c_synth_refresh
                    and cycles_without_improvement == 0
                    and synth_data_list
                ):
                    self.generator.eval()
                    self.discriminator.eval()
                    all_old_data_cpu = torch.cat(synth_data_list, dim=0)
                    all_old_labels = torch.cat(synth_label_list, dim=0)
                    refreshed_data: List[torch.Tensor] = []
                    refreshed_lbls: List[torch.Tensor] = []
                    for remapped_c in range(cfg.num_classes):
                        n_c = int((all_old_labels == remapped_c).sum().item())
                        if n_c == 0:
                            continue
                        orig_c = self.index_to_label[remapped_c]
                        z_r = self.latent_generator.sample_for_class_oversampling(
                            target_class=orig_c, n_samples=n_c,
                        ).to(self.device)
                        with torch.no_grad():
                            new_x = self.generator(z_r).detach()
                        accept = True
                        if cfg.ab8c_refresh_quality_guard:
                            old_x_c = all_old_data_cpu[
                                all_old_labels == remapped_c
                            ].to(self.device, non_blocking=True)
                            with torch.no_grad():
                                s_new = float(
                                    self.discriminator(new_x)[0].mean().item()
                                )
                                s_old = float(
                                    self.discriminator(old_x_c)[0].mean().item()
                                )
                            if s_new + cfg.ab8c_refresh_score_tol < s_old:
                                accept = False
                                refresh_rejected.append(remapped_c)
                                refreshed_data.append(old_x_c.detach().cpu())
                                refreshed_lbls.append(
                                    torch.full((old_x_c.size(0),),
                                               remapped_c, dtype=torch.long)
                                )
                            del old_x_c
                        if accept:
                            refreshed_data.append(new_x.detach().cpu())
                            refreshed_lbls.append(
                                torch.full((n_c,), remapped_c, dtype=torch.long)
                            )
                    synth_data_list = refreshed_data
                    synth_label_list = refreshed_lbls

                # -> (e) Validate + checkpoint + diagnostics --
                _, metrics, _, _ = self._validate(model, val_loader, criterion)
                # 123J-1A: select on cfg.selection_metric (macro_f2).
                f1 = metrics[cfg.selection_metric]
                total_synth = sum(t.size(0) for t in synth_data_list)
                aux_real_acc = self._ab8c_aux_real_val_acc(val_data, val_labels_idx)
                mean_grad_d = float(np.mean(grad_d_norms)) if grad_d_norms else 0.0

                warn_flags = []
                if aux_real_acc < aux_real_acc_warmup - 0.05:
                    warn_flags.append(
                        f"aux_drop={aux_real_acc_warmup - aux_real_acc:+.3f}"
                    )
                if degraded_classes:
                    warn_flags.append(
                        f"gated={sorted(degraded_classes)}({gate_mode})"
                    )
                if refresh_rejected:
                    warn_flags.append(
                        f"refresh_reject={sorted(refresh_rejected)}"
                    )
                warn_str = f" [WARN {' '.join(warn_flags)}]" if warn_flags else ""
                print(
                    f"      AL cycle {cycle+1}/{num_al_cycles}: "
                    f"synth_pool={total_synth}, {cfg.selection_metric}={f1:.4f}, "
                    f"D.aux val acc={aux_real_acc:.3f}, "
                    f"||grad_D||={mean_grad_d:.3f}{warn_str}"
                )

                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())
                    # S1: snapshot the full (C, G, D, synth pool) triplet.
                    best_G_state = copy.deepcopy(self.generator.state_dict())
                    best_D_state = copy.deepcopy(self.discriminator.state_dict())
                    best_synth_data = [t.clone() for t in synth_data_list]
                    best_synth_labels = [t.clone() for t in synth_label_list]
                    cycles_without_improvement = 0
                else:
                    cycles_without_improvement += 1
                if cycles_without_improvement >= cfg.ab8c_patience:
                    print(
                        f"      Early stopping at AL cycle {cycle+1} "
                        f"(patience={cfg.ab8c_patience})"
                    )
                    break

            # -> Phase 3: Consolidation (C only) --
            # S1: restore the full best (C, G, D, synth pool) triplet so
            # consolidation refines C against the same context that
            # produced the peak F1, not the last-cycle drifted state.
            if best_state is not None:
                model.load_state_dict(best_state)
                if best_G_state is not None:
                    self.generator.load_state_dict(best_G_state)
                if best_D_state is not None:
                    self.discriminator.load_state_dict(best_D_state)
                if best_synth_data is not None:
                    synth_data_list = [t.clone() for t in best_synth_data]
                    synth_label_list = [t.clone() for t in best_synth_labels]
            if synth_data_list:
                all_synth_data = torch.cat(synth_data_list, dim=0)
                all_synth_labels = torch.cat(synth_label_list, dim=0)
                final_data = torch.cat([train_data, all_synth_data], dim=0)
                final_labels = torch.cat(
                    [real_labels_remapped, all_synth_labels], dim=0
                )
            else:
                final_data = train_data
                final_labels = real_labels_remapped
            final_loader = _build_c_step_loader(final_data, final_labels)
            for _ in range(cfg.ab8_consolidation_epochs):
                self._train_epoch(model, final_loader, optimizer, criterion)
                _, metrics, _, _ = self._validate(model, val_loader, criterion)
                # 123J-1A: select on cfg.selection_metric (macro_f2).
                f1 = metrics[cfg.selection_metric]
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())
                    # S1: keep (G, D, synth) snapshots aligned with C.
                    best_G_state = copy.deepcopy(self.generator.state_dict())
                    best_D_state = copy.deepcopy(self.discriminator.state_dict())
                    best_synth_data = [t.clone() for t in synth_data_list]
                    best_synth_labels = [t.clone() for t in synth_label_list]
            print(f"      Consolidation done ->final best {cfg.selection_metric}={best_f1:.4f}")

        finally:
            # Always restore the GAN to its pre-Ab-8c state.
            self.generator.load_state_dict(orig_G_state)
            self.discriminator.load_state_dict(orig_D_state)
            self.generator.train(mode=gen_was_training)
            self.discriminator.train(mode=disc_was_training)
            if det_was_on:
                torch.use_deterministic_algorithms(True)

        if best_state is not None:
            model.load_state_dict(best_state)
        al_cycles_completed = (cycle + 1) if cycle >= 0 else 0
        total_epochs_trained = (
            cfg.ab8_warmup_epochs
            + al_cycles_completed * epochs_per_cycle
            + cfg.ab8_consolidation_epochs
        )
        best_metrics['stopping_epoch'] = total_epochs_trained

        _, _, best_val_logits, best_val_labels = self._validate(
            model, val_loader, criterion
        )
        return (
            model,
            best_metrics,
            best_val_logits.numpy(),
            best_val_labels.numpy(),
        )

    def _ab8c_sample_balanced_latent(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a class-balanced latent batch.

        Iterates over ``config.selected_classes`` (original labels) and
        delegates to ``self.latent_generator.sample_for_class_oversampling``
        (matching the convention used by Ab-8/8a/8b). The returned ``y``
        tensor uses CONTIGUOUS (remapped) class indices in ``[0, num_classes)``
        suitable for ``CrossEntropyLoss``.
        """
        classes = list(self.config.selected_classes)
        n_classes = len(classes)
        per_class = batch_size // n_classes
        zs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for i, cls in enumerate(classes):
            n = per_class if i < n_classes - 1 else (
                batch_size - per_class * (n_classes - 1)
            )
            if n <= 0:
                continue
            z = self.latent_generator.sample_for_class_oversampling(
                target_class=cls, n_samples=n,
            ).to(self.device)
            zs.append(z)
            ys.append(
                torch.full(
                    (n,), self.label_to_index[cls],
                    dtype=torch.long, device=self.device,
                )
            )
        return torch.cat(zs, dim=0), torch.cat(ys, dim=0)

    def _ab8c_aux_real_val_acc(
        self, val_data: torch.Tensor, val_labels_idx: torch.Tensor
    ) -> float:
        """Diagnostic: D.aux head accuracy on (a subsample of) real val.

        Returns 0.0 silently on any error so a diagnostic glitch does not
        crash the AL loop.
        """
        try:
            was_training = self.discriminator.training
            self.discriminator.eval()
            with torch.no_grad():
                n = min(len(val_data), 512)
                idx = torch.randperm(len(val_data))[:n]
                x = val_data[idx].to(self.device)
                y = val_labels_idx[idx].to(self.device)
                _, aux = self.discriminator(x)
                acc = float((aux.argmax(dim=1) == y).float().mean().item())
            if was_training:
                self.discriminator.train()
            return acc
        except Exception:
            return 0.0

    def _ab8c_aux_real_val_acc_per_class(
        self,
        val_data: torch.Tensor,
        val_labels_idx: torch.Tensor,
        num_classes: int,
    ) -> Dict[int, float]:
        """Per-class D.aux accuracy on a subsample of real val (Rec 1).

        Returns {remapped_class_index: acc}. Missing classes (no samples
        in subsample) map to ``float('nan')``. On any error, returns nans
        for all classes (gate becomes a no-op).
        """
        try:
            was_training = self.discriminator.training
            self.discriminator.eval()
            with torch.no_grad():
                n = min(len(val_data), 512)
                idx = torch.randperm(len(val_data))[:n]
                x = val_data[idx].to(self.device)
                y = val_labels_idx[idx].to(self.device)
                _, aux = self.discriminator(x)
                pred = aux.argmax(dim=1)
                out: Dict[int, float] = {}
                for c in range(num_classes):
                    mask = (y == c)
                    if int(mask.sum().item()) > 0:
                        out[c] = float((pred[mask] == y[mask]).float().mean().item())
                    else:
                        out[c] = float('nan')
            if was_training:
                self.discriminator.train()
            return out
        except Exception:
            return {c: float('nan') for c in range(num_classes)}

    # ================================================================
    # Ab-8b-on-BAGAN: BADGE + BAGAN-D acquisition filter (frozen G)
    # ================================================================
    #
    # The vanilla-BAGAN counterpart of Ab-8c. Ab-8c proper needs a
    # BAGAN-GP WGAN-GP critic + aux head + 3-player D/G/C co-training,
    # which the vanilla BAGAN discriminator (an (n+1)-class softmax) does
    # not provide. So this variant keeps Ab-8c's *acquisition* idea and
    # drops the WGAN-GP co-training:
    #   ->Generator FROZEN (no G-step) ->identical to Ab-8b.
    #   ->BADGE k-means++ selection on classifier gradient embeddings.
    #   ->D.aux acquisition filter: read the BAGAN discriminator's existing
    #     softmax prob for the target minority class and drop low-confidence
    #     candidates before BADGE picks for diversity.
    # Net: "Ab-8b + BAGAN-discriminator acquisition filter".
    # ================================================================

    def _run_al_loop_ab8b_bagan(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> Tuple['CNNClassifier', Dict[str, float]]:
        """Ab-8b-on-BAGAN CV loop: Ab-8b BADGE + BAGAN-D acquisition filter.

        Mirrors ``_run_al_loop_ab8`` (3-phase warm-up / AL cycles /
        consolidation, frozen generator) but replaces margin selection with
        BADGE k-means++ on gradient embeddings g_i = (p_i ->e__i) ->h_i
        and, before BADGE, optionally filters candidates by the BAGAN
        discriminator's softmax probability for the target minority class
        (``ab8b_bagan_acq_filter_tau``). The discriminator is used READ-ONLY
        (no D-step, no G-step).
        """
        from sklearn.cluster import kmeans_plusplus

        cfg = self.config

        # -> Deficit per minority class (same as Ab-8) --
        target_counts = _compute_target_counts(train_labels, cfg)
        deficits = {}
        for cls in cfg.minority_classes:
            current = (train_labels == cls).sum().item()
            deficits[cls] = max(0, target_counts[cls] - current)

        max_deficit = max(deficits.values()) if deficits else 0
        num_al_cycles = max(1, math.ceil(max_deficit / cfg.ab8_samples_per_cycle))
        remaining_epochs = max(1, cfg.num_epochs - cfg.ab8_warmup_epochs
                               - cfg.ab8_consolidation_epochs)
        epochs_per_cycle = max(1, remaining_epochs // num_al_cycles)

        _tau = cfg.ab8b_bagan_acq_filter_tau
        _use_filter = _tau > 0.0 and self.discriminator is not None
        print(f"      Ab-8c_bagan plan: {num_al_cycles} AL cycles x "
              f"{epochs_per_cycle} epochs/cycle, "
              f"warm-up={cfg.ab8_warmup_epochs}, "
              f"consolidation={cfg.ab8_consolidation_epochs}, "
              f"pool_factor={cfg.ab8b_bagan_pool_size_factor}, "
              f"D.aux filter ?={_tau if _use_filter else 'off'}")
        print(f"      Deficits: { {cls: deficits[cls] for cls in cfg.minority_classes} }")

        # -> Model, optimizer, criterion --
        model = CNNClassifier(
            input_channels=1,
            num_classes=cfg.num_classes,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # Locate final nn.Linear for BADGE penultimate-feature hook.
        final_linear: Optional[nn.Linear] = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                final_linear = m
        if final_linear is None:
            raise RuntimeError(
                "Ab-8c_bagan BADGE selector requires the classifier to "
                "contain at least one nn.Linear (none found in CNNClassifier)."
            )

        real_labels_remapped = self._remap_labels(train_labels)
        val_labels_idx = self._remap_labels(val_labels)
        val_dataset = TensorDataset(val_data, val_labels_idx)
        val_loader = DeviceDataLoader(
            DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False),
            self.device,
        )

        # Generator + discriminator are read-only for the whole loop.
        gen_was_training = self.generator.training
        self.generator.eval()
        disc_was_training = (
            self.discriminator.training if self.discriminator is not None else False
        )
        if self.discriminator is not None:
            self.discriminator.eval()

        _sel = cfg.selection_metric
        best_f1 = -1.0
        best_metrics: Dict[str, float] = {}
        best_state = None
        cycles_without_improvement = 0
        synth_data_list: List[torch.Tensor] = []
        synth_label_list: List[torch.Tensor] = []
        cycle = -1

        try:
            # -> Phase 1: Warm-up on original imbalanced data --
            train_ds = TensorDataset(train_data, real_labels_remapped)
            train_loader = DeviceDataLoader(
                DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
                self.device,
            )
            for epoch in range(cfg.ab8_warmup_epochs):
                self._train_epoch(model, train_loader, optimizer, criterion)
                _, metrics, _, _ = self._validate(model, val_loader, criterion)
                f1 = metrics[_sel]
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())
            print(f"      Warm-up done ->best {_sel} = {best_f1:.4f}")

            # -> Phase 2: Active-learning acquisition loop --
            for cycle in range(num_al_cycles):
                for cls in cfg.minority_classes:
                    remapped_c = self.label_to_index[cls]
                    already_added = sum(
                        int((lbl == remapped_c).sum().item())
                        for lbl in synth_label_list
                    )
                    n_add = min(
                        cfg.ab8_samples_per_cycle,
                        deficits[cls] - already_added,
                    )
                    if n_add <= 0:
                        continue

                    n_candidates = n_add * cfg.ab8b_bagan_pool_size_factor
                    z = self.latent_generator.sample_for_class_oversampling(
                        target_class=cls, n_samples=n_candidates,
                    ).to(self.device)
                    with torch.no_grad():
                        candidates = self.generator(z)

                    # -> D.aux acquisition filter (BAGAN softmax, read-only)
                    # Keep candidates the BAGAN discriminator confidently
                    # assigns to the target minority class. logits[:, :n] are
                    # real-class logits; logits[:, n] is the fake logit.
                    if _use_filter:
                        with torch.no_grad():
                            d_logits = self._adv_class_logits(self.discriminator, candidates)
                            d_p = torch.softmax(d_logits, dim=1)[:, remapped_c]
                        keep = d_p >= _tau
                        min_keep = max(
                            n_add,
                            int(cfg.ab8b_bagan_acq_filter_min_keep_frac
                                * candidates.size(0)),
                        )
                        if int(keep.sum().item()) >= min_keep:
                            candidates = candidates[keep]
                        else:
                            k_top = min(min_keep, d_p.numel())
                            top_idx = torch.topk(d_p, k=k_top).indices
                            candidates = candidates[top_idx]

                    # -> BADGE k-means++ selection --
                    captured: Dict[str, torch.Tensor] = {}

                    def _hook(_mod, inp, _out, _store=captured):
                        _store['h'] = inp[0].detach()

                    handle = final_linear.register_forward_hook(_hook)
                    try:
                        with torch.no_grad():
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
                    if cfg.ab8b_bagan_kpp_seed_normalize:
                        emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-12)
                    emb_np = emb.cpu().numpy()
                    k = min(n_add, emb_np.shape[0])
                    _, kpp_idx = kmeans_plusplus(
                        emb_np, n_clusters=k, random_state=cfg.run_seed,
                    )
                    selected = candidates[kpp_idx].cpu()
                    synth_data_list.append(selected)
                    synth_label_list.append(
                        torch.full((k,), remapped_c, dtype=torch.long)
                    )

                # -> Build grown dataset (real + cumulative synthetic) --
                if synth_data_list:
                    all_synth_data = torch.cat(synth_data_list, dim=0)
                    all_synth_labels = torch.cat(synth_label_list, dim=0)
                    grown_data = torch.cat([train_data, all_synth_data], dim=0)
                    grown_labels = torch.cat(
                        [real_labels_remapped, all_synth_labels], dim=0
                    )
                else:
                    grown_data = train_data
                    grown_labels = real_labels_remapped

                grown_ds = TensorDataset(grown_data, grown_labels)
                grown_loader = DeviceDataLoader(
                    DataLoader(grown_ds, batch_size=cfg.batch_size, shuffle=True),
                    self.device,
                )
                for epoch in range(epochs_per_cycle):
                    self._train_epoch(model, grown_loader, optimizer, criterion)

                _, metrics, _, _ = self._validate(model, val_loader, criterion)
                f1 = metrics[_sel]
                total_synth = sum(len(t) for t in synth_data_list)
                print(f"      AL cycle {cycle+1}/{num_al_cycles}: "
                      f"synth_pool={total_synth}, {_sel}={f1:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())
                    cycles_without_improvement = 0
                else:
                    cycles_without_improvement += 1

                if cycles_without_improvement >= cfg.ab8_patience:
                    print(f"      Early stopping at AL cycle {cycle+1} "
                          f"(patience={cfg.ab8_patience})")
                    break

            # -> Phase 3: Consolidation on final balanced dataset --
            if best_state is not None:
                model.load_state_dict(best_state)

            if synth_data_list:
                all_synth_data = torch.cat(synth_data_list, dim=0)
                all_synth_labels = torch.cat(synth_label_list, dim=0)
                final_data = torch.cat([train_data, all_synth_data], dim=0)
                final_labels = torch.cat(
                    [real_labels_remapped, all_synth_labels], dim=0
                )
            else:
                final_data = train_data
                final_labels = real_labels_remapped

            final_ds = TensorDataset(final_data, final_labels)
            final_loader = DeviceDataLoader(
                DataLoader(final_ds, batch_size=cfg.batch_size, shuffle=True),
                self.device,
            )
            for epoch in range(cfg.ab8_consolidation_epochs):
                self._train_epoch(model, final_loader, optimizer, criterion)
                _, metrics, _, _ = self._validate(model, val_loader, criterion)
                f1 = metrics[_sel]
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())
            print(f"      Consolidation done ->final best {_sel} = {best_f1:.4f}")

        finally:
            self.generator.train(mode=gen_was_training)
            if self.discriminator is not None:
                self.discriminator.train(mode=disc_was_training)

        if best_state is not None:
            model.load_state_dict(best_state)

        al_cycles_completed = (cycle + 1) if num_al_cycles > 0 else 0
        total_epochs_trained = (
            cfg.ab8_warmup_epochs
            + al_cycles_completed * epochs_per_cycle
            + cfg.ab8_consolidation_epochs
        )
        best_metrics['stopping_epoch'] = total_epochs_trained

        _, _, best_val_logits, best_val_labels = self._validate(
            model, val_loader, criterion
        )
        return (
            model,
            best_metrics,
            best_val_logits.numpy(),
            best_val_labels.numpy(),
        )

    # ================================================================
    # Ab-9: Pool-Drain Active Learning
    # ================================================================
    #
    # Ab-9 generates a balanced pool upfront (same as Ab-4's one-shot
    # generation), then drains it via margin-based active learning ->
    # feeding samples in order of classifier uncertainty.
    #
    # Three phases:
    #   1. Augmentation: build balanced pool = real + BAGAN synthetic.
    #   2. Pool-Drain AL: seed with real samples ->warm-up ->iteratively
    #      score ALL remaining pool samples by margin, acquire the most
    #      uncertain, retrain.  Patience-based early stopping halts the
    #      loop when binary_f1 plateaus.
    #   3. Consolidation: fine-tune the best model on the consumed
    #      portion of the pool.
    #
    # Key difference from Ab-8:
    #   Ab-8 generates fresh BAGAN candidates each cycle (infinite source).
    #   Ab-9 generates everything upfront and drains a finite pool ->
    #   the margin determines the ORDER of exposure (curriculum learning).
    #   Patience-based stopping also acts as quality filtering: if the
    #   remaining pool samples no longer help, the loop stops early.
    # ================================================================

    def _run_al_loop_ab9(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> Tuple['CNNClassifier', Dict[str, float]]:
        """Ab-9: Pool-Drain Active Learning with pre-generated BAGAN pool.

        This method REPLACES the standard epoch loop in train_fold() for Ab-9.
        It manages its own model, optimizer, early stopping, and epoch budget.

        Algorithm outline (refer to the comment block above for background):

            Phase I ->Augmentation
                Build a balanced pool = real data + BAGAN synthetic (same
                as Ab-4's one-shot generation, 33:33:33 target).

            Phase II ->Pool-Drain Active Learning
                Seed a small training set with ab9_init_per_class REAL
                samples per class.  Warm-up train on the seed set.
                Then iteratively: score ALL remaining pool samples by
                margin, acquire the N most uncertain, retrain (warm-start).
                Stop when pool is empty OR patience is exhausted.

            Phase III ->Consolidation
                Restore best model, fine-tune on the consumed portion
                of the pool for ab9_consolidation_epochs.

        Returns:
            model: CNNClassifier with best weights (by binary_f1).
            best_metrics: Dict of validation metrics at the best checkpoint,
                          including 'stopping_epoch' (total epochs trained).
        """
        cfg = self.config

        # --
        # PHASE I ->Build balanced pool (same approach as Ab-4)
        # --

        # Generator in eval mode for the entire method
        gen_was_training = self.generator.training
        self.generator.eval()

        try:
            # Generate balanced pool: real + synthetic
            # Returns data in ORIGINAL label space (e.g., {2, 4, 6})
            pool_data, pool_labels = generate_bagan_augmented_dataset(
                train_data, train_labels,
                self.generator, self.latent_generator,
                cfg, self.device
            )

            # Remap ALL pool labels to contiguous [0, num_classes-1]
            pool_labels = self._remap_labels(pool_labels)

            n_real = len(train_data)
            n_pool = len(pool_data)
            print(f"      Ab-9 Phase I: pool={n_pool} "
                  f"(real={n_real}, synthetic={n_pool - n_real})")

            # --
            # PHASE II ->Pool-Drain Active Learning
            # --

            # -> Seed: pick ab9_init_per_class REAL samples per class ->
            # Real samples occupy indices [0, n_real-1] in the pool.
            seed_indices = []
            for c in range(cfg.num_classes):
                # Find real samples of this class
                class_mask = (pool_labels[:n_real] == c).nonzero(as_tuple=True)[0]
                # Shuffle and pick first N
                perm = torch.randperm(len(class_mask))
                n_pick = min(cfg.ab9_init_per_class, len(class_mask))
                seed_indices.append(class_mask[perm[:n_pick]])
            seed_indices = torch.cat(seed_indices)

            # Split pool into training_set (seed) and remaining pool
            mask = torch.ones(n_pool, dtype=torch.bool)
            mask[seed_indices] = False

            al_train_data = pool_data[seed_indices].clone()
            al_train_labels = pool_labels[seed_indices].clone()
            remaining_data = pool_data[mask].clone()
            remaining_labels = pool_labels[mask].clone()

            print(f"      Ab-9 Phase II: seed={len(al_train_data)}, "
                  f"pool_remaining={len(remaining_data)}")

            # -> Create model, optimizer, criterion --
            model = CNNClassifier(
                input_channels=1,
                num_classes=cfg.num_classes
            ).to(self.device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
            criterion = nn.CrossEntropyLoss()

            # Validation loader (fixed)
            val_labels_idx = self._remap_labels(val_labels)
            val_dataset = TensorDataset(val_data, val_labels_idx)
            val_loader = DeviceDataLoader(
                DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False),
                self.device
            )

            # Best-model tracking
            best_f1 = -1.0
            best_metrics = {}
            best_state = None

            # -> Warm-up on seed set --
            seed_ds = TensorDataset(al_train_data, al_train_labels)
            seed_loader = DeviceDataLoader(
                DataLoader(seed_ds, batch_size=cfg.batch_size, shuffle=True),
                self.device
            )

            for epoch in range(cfg.ab9_warmup_epochs):
                train_loss = self._train_epoch(
                    model, seed_loader, optimizer, criterion
                )
                val_loss, metrics, _, _ = self._validate(
                    model, val_loader, criterion
                )
                f1 = metrics[cfg.ab9_checkpoint_metric]
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())

            print(f"      Warm-up done ->best {cfg.ab9_checkpoint_metric} "
                  f"= {best_f1:.4f}")

            # -> AL cycle loop with patience --
            cycle = 0
            stale_cycles = 0
            total_al_epochs = 0

            while len(remaining_data) > 0 and stale_cycles < cfg.ab9_patience:
                cycle += 1
                n_acq = min(cfg.ab9_samples_per_cycle, len(remaining_data))

                # -> Score ALL remaining pool samples by margin --
                model.eval()
                with torch.no_grad():
                    # Forward in batches to avoid OOM on large pools
                    all_margins = []
                    for i in range(0, len(remaining_data), cfg.batch_size):
                        batch = remaining_data[i:i + cfg.batch_size].to(self.device)
                        logits = model(batch)
                        probs = torch.softmax(logits, dim=1)
                        sorted_probs, _ = probs.sort(dim=1, descending=True)
                        margins = sorted_probs[:, 0] - sorted_probs[:, 1]
                        all_margins.append(margins.cpu())
                    all_margins = torch.cat(all_margins)

                # -> Acquire lowest-margin samples --
                _, acq_idx = torch.topk(all_margins, k=n_acq, largest=False)

                # Move acquired to training set
                al_train_data = torch.cat(
                    [al_train_data, remaining_data[acq_idx]]
                )
                al_train_labels = torch.cat(
                    [al_train_labels, remaining_labels[acq_idx]]
                )

                # Remove from remaining pool
                keep_mask = torch.ones(len(remaining_data), dtype=torch.bool)
                keep_mask[acq_idx] = False
                remaining_data = remaining_data[keep_mask]
                remaining_labels = remaining_labels[keep_mask]

                # -> Retrain (warm-start) on grown training set --
                train_ds = TensorDataset(al_train_data, al_train_labels)
                train_loader = DeviceDataLoader(
                    DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
                    self.device
                )

                model.train()
                for epoch in range(cfg.ab9_epochs_per_cycle):
                    train_loss = self._train_epoch(
                        model, train_loader, optimizer, criterion
                    )
                total_al_epochs += cfg.ab9_epochs_per_cycle

                # -> Evaluate, checkpoint, track patience --
                val_loss, metrics, _, _ = self._validate(
                    model, val_loader, criterion
                )
                f1 = metrics[cfg.ab9_checkpoint_metric]

                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())
                    stale_cycles = 0
                else:
                    stale_cycles += 1

                print(f"      [Ab-9] cycle {cycle}: "
                      f"pool={len(remaining_data)}, "
                      f"train={len(al_train_data)}, "
                      f"{cfg.ab9_checkpoint_metric}={f1:.4f}, "
                      f"stale={stale_cycles}/{cfg.ab9_patience}")

            # -> Log exit reason --
            if stale_cycles >= cfg.ab9_patience:
                print(f"      [Ab-9] Early stop ->patience exhausted at "
                      f"cycle {cycle} (pool remaining: "
                      f"{len(remaining_data)})")
            else:
                print(f"      [Ab-9] Pool fully drained at cycle {cycle}")

            # --
            # PHASE III ->Consolidation on consumed training set
            # --
            if best_state is not None:
                model.load_state_dict(best_state)

            # Build loader from consumed training set (NOT full pool)
            consol_ds = TensorDataset(al_train_data, al_train_labels)
            consol_loader = DeviceDataLoader(
                DataLoader(consol_ds, batch_size=cfg.batch_size, shuffle=True),
                self.device
            )

            for epoch in range(cfg.ab9_consolidation_epochs):
                train_loss = self._train_epoch(
                    model, consol_loader, optimizer, criterion
                )
                val_loss, metrics, _, _ = self._validate(
                    model, val_loader, criterion
                )
                f1 = metrics[cfg.ab9_checkpoint_metric]
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = metrics.copy()
                    best_state = copy.deepcopy(model.state_dict())

            print(f"      Consolidation done ->final best "
                  f"{cfg.ab9_checkpoint_metric} = {best_f1:.4f}")

        finally:
            # Restore generator training state
            self.generator.train(mode=gen_was_training)

        # Load absolute best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Record total epochs trained across all 3 phases
        al_cycles_completed = cycle if cycle > 0 else 0
        total_epochs_trained = (
            cfg.ab9_warmup_epochs
            + total_al_epochs
            + cfg.ab9_consolidation_epochs
        )
        best_metrics['stopping_epoch'] = total_epochs_trained

        # One extra _validate() on restored best weights to harvest
        # raw logits for Topic 7 (pragmatic asymmetry ->see 02a plan C2).
        _, _, best_val_logits, best_val_labels = self._validate(
            model, val_loader, criterion
        )
        return (
            model,
            best_metrics,
            best_val_logits.numpy(),
            best_val_labels.numpy(),
        )

