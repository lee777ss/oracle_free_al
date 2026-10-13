"""
Configuration dataclass for ablation study experiments.

Centralizes all hyperparameters and experimental settings for multi-class
imbalanced classification with k-fold cross-validation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import os

# Fashion-MNIST class names for visualization
FASHION_MNIST_CLASSES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}


@dataclass
# what does '@dataclass' do?
# The '@dataclass' decorator automatically generates special methods for the class, 
# such as __init__, __repr__, __eq__, and others, based on the class attributes.
class AblationConfig:
    """Configuration for ablation study experiments.
    
    Attributes:
        selected_classes: List of Fashion-MNIST class labels to use
        class_proportions: Proportion of samples for each class (first is majority)
        n_folds: Number of folds for stratified k-fold cross-validation
        random_seed: Seed for DATA PARTITIONING only (imbalanced pool, test set,
            k-fold assignment, retrain train/val split). Held fixed across the
            seeds of a multi-seed study so every seed and every mode sees the
            identical data.
        run_seed: Seed for ALGORITHMIC stochasticity (acquisition k-means++ /
            selection RNG). Varies across the seeds of a multi-seed study.
            Defaults to ``random_seed`` so single-seed runs are unchanged.
        num_epochs: Maximum training epochs per fold
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        patience: Early stopping patience (epochs without improvement)
        augmentation_target: Target class distribution as string "maj:min1:min2"
        latent_dim: BAGAN latent space dimensionality
        bagan_model_dir: Directory for saving/loading BAGAN models
        device: Device for training ('auto', 'cuda', 'cpu')
        date_suffix: Experiment identifier suffix
    """
    
    # Classes & imbalance
    # Current default scenario: [0, 2, 3] (T-shirt/top, Pullover, Dress).
    # Alternative class sets can still be tested by overriding selected_classes.

    selected_classes: List[int] = field(default_factory=lambda: [0, 2, 3])

    # Current default imbalance uses a capped majority count of 800 with
    # proportions [1.0, 0.05, 0.05], yielding 800:40:40 in train.
    # Example alternative retained for reference:
    # class_proportions: List[float] = field(default_factory=lambda: [1.0, 0.10, 0.10])
    # test_proportions: List[float] = field(default_factory=lambda: [1.0, 0.10, 0.10])

    class_proportions: List[float] = field(default_factory=lambda: [1.0, 0.05, 0.05])

    test_proportions: List[float] = field(default_factory=lambda: [1.0, 0.05, 0.05])

    

    # 514J: Cap majority class at 800 samples (Chung Table I).
    # Without this, proportion=1.0 keeps all 6000 ->minority scales against 6000.
    # With cap=800 and current prop=0.05: minority=0.05*800=40.
    # Set to None to keep all majority samples (513J behavior).
    max_samples_majority: Optional[int] = 800
    
    # K-fold cross-validation
    n_folds: int = 5

    # ─── Seeds ──────────────────────────────────────────────────────────────
    # random_seed  → DATA partitioning (pool, test set, k-fold, retrain split).
    #                Keep FIXED across a multi-seed study: every seed and every
    #                mode must be scored on identical data for the comparison
    #                across variants to be paired.
    # run_seed     → ALGORITHMIC stochasticity (acquisition selection RNG).
    #                VARIES across a multi-seed study. Global torch/numpy RNG
    #                state (network init, batch order, GAN noise) is driven
    #                separately by the notebook's set_seed(SEED).
    # Mirrors the case-study protocol in cs3j_runner.run_one(), where the split
    # is pinned to config.random_seed while the per-run seed drives the method.
    random_seed: int = 42
    run_seed: Optional[int] = None    # None → falls back to random_seed

    # Training hyperparameters
    num_epochs: int = 500
    # num_epochs: int = 50
    batch_size: int = 64
    # 514J: Lowered from 0.001 ->0.0005 to compensate for BatchNorm removal.
    # Without BN, gradients are less stable; a smaller LR prevents divergence.
    learning_rate: float = 0.0001
    
    weight_decay: float = 0.0001  # L2 regularization strength
   
    patience: int = 100  # EarlyStopping patience
    # patience: int = 7  # EarlyStopping patience

    # 123J-1A: metric used to select the best checkpoint/fold. Switched
    #   from 'macro_f1' to 'macro_f2' to emphasise recall on the imbalanced
    #   problem. Ab-8c's internal best-tracking and early-stop read this so
    #   the kept state matches how the model is graded downstream
    #   (evaluation.get_best_fold_model also defaults to 'macro_f2').
    selection_metric: str = 'macro_f2'

    if False:
        # Scenario 1 #########################################################################
        # "Let it overfit": long training, no regularization, no early stop
        num_epochs = 300
        learning_rate = 0.0005
        weight_decay = 0
        patience = 300  # effectively disabled
        #######################################################################################
        
        # Scenario 2 #########################################################################
        # "Noisy gradients": high LR, short patience
        num_epochs = 100
        learning_rate = 2e-3
        weight_decay = 0
        patience = 10
        #######################################################################################

        # Scenario 3 #########################################################################
        # "Weight-decay penalty": strong L2 + short training
        num_epochs = 50
        learning_rate = 5e-4
        weight_decay = 5e-3
        patience = 15
        #######################################################################################

        # Scenario 4 #########################################################################
        # "Slow soak": low LR, very long training, mild WD
        num_epochs = 500
        learning_rate = 1e-4
        weight_decay = 1e-4
        patience = 100
        #######################################################################################
        

        # Scenario 5 #########################################################################
        # "Fixed budget, no rescue": disable ES entirely
        num_epochs = 150
        learning_rate = 5e-4
        weight_decay = 0
        patience = 9999  # ES never triggers
        #######################################################################################


    # num_epochs = 500
    # learning_rate = 1e-4
    # weight_decay = 1e-4
    # patience = 100

    # Retrain monitoring: fraction of pool held out for checkpoint selection.
    # Used by retrain_on_full_pool() to select the best-epoch model.
    # Set to 0.0 to disable (reverts to original last-epoch behavior).
    retrain_val_fraction: float = 0.1
    
    # Per-mode retrain early-stopping patience.
    # Keys = mode strings ('ab1', 'ab4', etc.), values = patience epochs.
    # Modes not listed fall back to patience * 2 (default heuristic).
    # Example: {'ab1': 20, 'ab9': 30}
    # retrain_patience: Dict[str, int] = field(default_factory=lambda: {'ab4': 20})
    retrain_patience: Dict[str, int] = field(default_factory=dict)
    
    # Per-mode retrain max epoch cap (upper bound for ES loop).
    # Keys = mode strings, values = max epochs.
    # Modes not listed use max(avg_stopping_epoch * 2, 30).
    # Example: {'ab1': 60, 'ab10': 50}
    # retrain_max_epochs: Dict[str, int] = field(default_factory=lambda: {'ab4': 60})
    retrain_max_epochs: Dict[str, int] = field(default_factory=dict)
    
    # Augmentation targets (proportions relative to majority class size)
    # Format: "majority:minority1:minority2" ->equal proportions = fully balanced
    augmentation_target: str = "33:33:33"
    
    # BAGAN configuration
    latent_dim: int = 128
    bagan_model_dir: str = "saved"
    
    # Ab-6 quality gate: generate this many times more BAGAN candidates
    # than needed, then keep the most realistic ones via discriminator.
    # e.g., 3 ->generate 3 n_synthetic, keep top n_synthetic lowest P(fake)
    overproduction_factor: int = 3
    
    # Ab-7 margin sampling: generate this many times more BAGAN candidates
    # than needed, then keep the ones the classifier is most uncertain about
    # (lowest margin = P(top-1) - P(top-2)).
    margin_overproduction_factor: int = 3

    ##############################################################################
    # Ab-8: Iterative BAGAN Acquisition (GAAL-style) 
    # Nielsen (2019) GAAL pipeline: train classifier iteratively, adding
    # a small batch of margin-selected BAGAN samples each AL cycle.
    #
    # ab8_samples_per_cycle: how many synthetic images to add per minority
    #   class per AL cycle.  Small values (10) keep the curriculum gradual.
    ab8_samples_per_cycle: int = 10
    #
    # ab8_pool_size_factor: overproduction multiplier for candidate generation.
    #   Each cycle generates ab8_pool_size_factor  ab8_samples_per_cycle
    #   candidates per minority class, then keeps the most uncertain ones.
    ab8_pool_size_factor: int = 3
    #
    # ab8_warmup_epochs: epochs on the original imbalanced data before
    #   the acquisition loop starts.  Gives the classifier a baseline
    #   decision boundary so that margin scoring is meaningful from cycle 1.
    ab8_warmup_epochs: int = 5
    #
    # ab8_consolidation_epochs: extra epochs on the final (fully augmented)
    #   dataset after the AL loop finishes.  Lets the classifier settle
    #   its weights on the complete balanced data.
    ab8_consolidation_epochs: int = 5
    #
    # ab8_patience: AL-cycle-level early stopping patience.
    #   Counts consecutive AL cycles without binary_f1 improvement.
    #   Independent from the epoch-level 'patience' used by other ablations.
    ab8_patience: int = 7
    ##############################################################################

    ##############################################################################
    # Ab-8a: Independent-Oracle Selector (Ab-8 + BAGAN-GP aux-head gate) ?
    # Ab-8a is identical to Ab-8 EXCEPT that candidates are first filtered
    # by the BAGAN-GP discriminator's auxiliary classifier head ("oracle")
    # before margin selection. This rejects samples that the GAN itself
    # disagrees about the label of ->fixing the noisy-oracle failure mode
    # of margin-lowest AL when label-fidelity ->image-fidelity.
    #
    # ab8a_aux_tau: minimum softmax confidence required from the aux head
    #   for a candidate to pass the label gate (in addition to
    #   argmax(aux_logits) == intended_class).
    ab8a_aux_tau: float = 0.6
    #
    # ab8a_aux_tau_fallback: relaxed threshold used when the default threshold 
    #   does not yield enough survivors. Argmax-mismatched candidates are
    #   STILL rejected at this stage ->only the confidence floor relaxes.
    ab8a_aux_tau_fallback: float = 0.5
    #
    # ab8a_pool_size_factor: overproduction multiplier for Ab-8a. Larger
    #   than ab8_pool_size_factor because the aux gate rejects 30-50% of
    #   raw candidates. Defaults to 5 (vs Ab-8's 3).
    ab8a_pool_size_factor: int = 5
    ##############################################################################

    ##############################################################################
    # Ab-8b: BADGE Selector (Ab-8 with margin-BADGE swap) 
    # Ab-8b is identical to Ab-8 EXCEPT step 2a's selector: instead of
    # keeping the n_add candidates with lowest margin, it selects n_add
    # via k-means++ seeding on BADGE gradient embeddings
    #     g_i = (p_i ->e__i) ->h_i           (closed form; no backward pass)
    # where _i = argmax C(x_i), p_i = softmax(C(x_i)),
    # h_i is the penultimate activation captured by a forward hook on the
    # classifier's final nn.Linear, and ->is the (flattened) outer product.
    # Reference: Ash et al. 2020, "Deep Batch Active Learning by Diverse,
    # Uncertain Gradient Lower Bounds" (BADGE).
    #
    # ab8b_pool_size_factor: overproduction multiplier ->each cycle generates
    #   ab8b_pool_size_factor  ab8_samples_per_cycle candidates per minority
    #   class. Default 5 (matches Ab-8a; gives BADGE a wider diversity pool
    #   than Ab-8's margin selector needs).
    ab8b_pool_size_factor: int = 5
    #
    # ab8b_kpp_seed_normalize: if True, L2-normalize gradient embeddings
    #   before k-means++ seeding. Off by default ->pure BADGE.
    ab8b_kpp_seed_normalize: bool = False
    #
    # ab8b_selection_mode: acquisition rule. "badge" keeps Ab-8b as the
    #   literal Ash-et-al. selector (the published default, unchanged).
    #   "confident_diverse" swaps in the confidence-gated rule used by Ab-8d
    #   and Ab-8c. Exposed here so Ab-8b's rule can be varied without adding
    #   another mode.
    ab8b_selection_mode: str = "badge"
    #
    # ab8b_min_confidence: p(target class) floor, used only when
    #   ab8b_selection_mode == "confident_diverse".
    ab8b_min_confidence: float = 0.6
    ##############################################################################

    ##############################################################################
    # Ab-8d: Confidence-Gated Diverse Selector (Ab-8b with the selector swapped)
    # Ab-8d is identical to Ab-8b EXCEPT the acquisition rule. Instead of
    # k-means++ on BADGE gradient embeddings (which deliberately acquires the
    # most *uncertain* candidates), Ab-8d first rejects any candidate the
    # CLASSIFIER does not confidently assign to the intended class
    #     argmax C(x) == c   AND   p_C(c | x) >= ab8d_min_confidence
    # and then k-means++ seeds n_add *diverse* picks over the penultimate
    # features of the survivors.
    #
    # Rationale: BADGE assumes an oracle relabels each acquired point, so
    # ambiguity is informative. Here there is no oracle - an acquired sample
    # enters training carrying the class it was GENERATED for, so an ambiguous
    # generation is injected as label noise. The gate removes off-manifold and
    # mis-conditioned generations before the diversity step.
    #
    # Ab-8d isolates that selector swap: it is the direct BADGE -> confidence-
    # gated comparison, with no co-training. Ab-8c then adds the BAGAN-GP
    # backbone + 3-player co-training on top of this same rule + seeding.
    #
    # Re-uses ``ab8_samples_per_cycle``, ``ab8_warmup_epochs``,
    # ``ab8_consolidation_epochs`` and ``ab8_patience`` from Ab-8, exactly as
    # Ab-8b does, so the two differ ONLY in the selector.
    ab8d_pool_size_factor: int = 5
    ab8d_min_confidence: float = 0.6
    ab8d_kpp_seed_normalize: bool = True
    #
    # ab8d_seed_mode: candidate SEEDING mode, same options as
    #   ``ab8c_seed_mode`` ("prior" vs "latent_smote"). 132J: Variant 2 (Ab-8d)
    #   is Table 4's direct predecessor of Proposed (Ab-8c), so it now shares
    #   the SAME latent-SMOTE seeding -> the Variant2 -> Proposed comparison
    #   isolates ONLY the BAGAN-GP backbone + co-training, not the seeding.
    #   Independent of ``ab8c_seed_mode`` so it can be varied without
    #   affecting Ab-8c. Ab-8b (BADGE, excluded from the paper) has no
    #   ``ab8b_seed_mode`` field and therefore keeps sampling the prior.
    ab8d_seed_mode: str = "latent_smote"
    ab8d_latent_smote_noise: float = 0.25
    ##############################################################################

    ##############################################################################
    # Ab-8c: BADGE Active Learning + 3-Player Co-Training 
    # Ab-8c = Ab-8b's BADGE acquisition + per-cycle co-training of the
    # WGAN-GP critic/discriminator (D), generator (G), and classifier (C).
    # No aux-gate on the candidate pool (BADGE-only, like Ab-8b).
    #
    # Per AL cycle (after warm-up):
    #   (a) BADGE ->k-means++ on gradient embeddings (same as Ab-8b).
    #   (b) D-step ->K_D minibatches of WGAN-GP critic loss on real, with
    #       optional aux-CE on accepted synthetic samples (small lr).
    #   (c) G-step ->K_G minibatches: _adv(-D(x_fake)) + _aux_gCE(D.aux,y)
    #       + _clsCE(C(x_fake),y) + _anchor||?_D(G(z))-?_D(G_ref(z))||.
    #   (d) C-step ->epochs_per_cycle epochs on real ->cumulative synth.
    #   (e) Validate + AL-cycle patience.
    # ``G_ref`` is a deep-copy of G snapshotted at the end of warm-up
    # (Decision B = static reference).
    #
    # Re-uses ``ab8_samples_per_cycle``, ``ab8_warmup_epochs``,
    # ``ab8_consolidation_epochs``, and ``ab8_patience`` from Ab-8.
    #
    # ab8c_pool_size_factor: acquisition overproduction multiplier (per cycle,
    #   per minority class).
    #   131J: 5 -> 10 to match casestudy/methods/m10_ab8c.py (pool_factor=10).
    ab8c_pool_size_factor: int = 10
    #
    # 131J: Ab-8c-specific AL budget, so that the proposed variant matches the
    # case-study implementation WITHOUT disturbing Ab-8/8a/8b/8d, which keep
    # sharing ``ab8_samples_per_cycle`` / ``ab8_warmup_epochs``.
    #   m10_ab8c.samples_per_cycle = 20
    #   m10_ab8c.c_warmup_epochs   = 15
    ab8c_samples_per_cycle: int = 20
    ab8c_warmup_epochs: int = 15
    #
    # ab8c_kpp_seed_normalize: if True, L2-normalize BADGE gradient
    #   embeddings before k-means++ (matches Ab-8b's knob). Off = pure BADGE.
    #   123J-2: False ->True ->direction-over-magnitude BADGE picks improve
    #   minority coverage ->boosts recall, the F2 numerator.
    ab8c_kpp_seed_normalize: bool = True
    #
    # ab8c_selection_mode: acquisition rule for the proposed method.
    #   131J: switched "badge" -> "confident_diverse" so that the ablation's
    #   proposed variant uses the SAME acquisition rule as the case-study
    #   implementation (casestudy/methods/m10_ab8c.py). Before 131J the two
    #   disagreed: the ablation acquired the most uncertain candidates while
    #   the case studies acquired confidently-classified ones. Set back to
    #   "badge" to reproduce the pre-131J ablation numbers.
    ab8c_selection_mode: str = "confident_diverse"
    #
    # ab8c_min_confidence: p(target class) floor for the confidence gate,
    #   used only when ab8c_selection_mode == "confident_diverse". Matches
    #   the case-study default (m10_ab8c.min_confidence).
    ab8c_min_confidence: float = 0.6
    #
    # 131J: candidate SEEDING mode, matching m10_ab8c.seed_mode.
    #   "prior"        -> draw z ~ N(mu_c, sigma_c^2 I) from the fitted
    #                     class-conditional prior (the pre-131J behaviour).
    #   "latent_smote" -> interpolate between ENCODED REAL minority latents and
    #                     add jitter scaled by ab8c_latent_smote_noise * sigma_c.
    #                     Keeps candidates nearer the observed minority manifold;
    #                     the prior can place mass in low-density regions that
    #                     decode to off-manifold images.
    # 132J: Ab-8d now ALSO uses latent-SMOTE (via its own ``ab8d_seed_mode``
    # field, see the Ab-8d block above), so this field's default no longer
    # isolates Ab-8c's seeding uniquely - the Variant2 (Ab-8d) -> Proposed
    # (Ab-8c) comparison isolates the BAGAN-GP backbone + co-training only.
    # Ab-8/8a/8b keep sampling the prior (no ``ab8{mode}_seed_mode`` field).
    ab8c_seed_mode: str = "latent_smote"
    ab8c_latent_smote_noise: float = 0.25
    #
    # ab8c_kpp_pca_dim: optional PCA dim cap before k-means++. None = no PCA
    #   (current behavior). Reserved for forward-compat; not yet wired into
    #   the BADGE selector ->has no effect unless explicitly consumed.
    ab8c_kpp_pca_dim: Optional[int] = None
    #
    # ab8c_k_d: number of D minibatches per AL cycle.
    #   117J-tuned: lowered 30->5 to reduce per-cycle D drift (matches new k_g ratio).
    ab8c_k_d: int = 15
    #
    # ab8c_k_g: number of G minibatches per AL cycle.
    #   117J-tuned: lowered 20->0 ->log showed F1 peaks at cycle 2-> then decays,
    #   indicating G drift; fewer G updates per cycle slow the drift.
    ab8c_k_g: int = 10
    #
    # ab8c_lr_d_factor: multiplier on the WGAN-GP base D lr (1e-4) for the
    #   per-cycle D fine-tune optimizer. 117J-tuned: 0.1 ->0.05 (lr_d_ft = 5e-6)
    #   to keep D.aux head stable (observed real-val aux acc drift of 3-> pts).
    ab8c_lr_d_factor: float = 0.05
    #
    # ab8c_lr_g_factor: multiplier on the WGAN-GP base G lr (1e-4) for the
    #   per-cycle G fine-tune optimizer. 117J-tuned: 0.1 ->0.02 (lr_g_ft = 2e-6)
    #   ->slow G more than D since G drift is the dominant failure mode.
    #   123J-2: 0.02 ->0.01 ->further slow G drift past the F2 peak
    #   (complements lower _adv); fewer off-manifold minority samples.
    #   125J: 0.01 ->0.005 ->the balanced-critic + shrinkage warm-up G is now
    #   strong, so any drift is net-harmful; halve G lr to preserve it.
    ab8c_lr_g_factor: float = 0.05
    #
    # ab8c_lambda_aux_real: weight on CE(D.aux(real), y_real) in the D-step.
    #   117J-tuned: 1.0 ->2.0 to counter observed aux-head drift on real labels.
    ab8c_lambda_aux_real: float = 2.0
    #
    # ab8c_lambda_aux_synth: weight on CE(D.aux(S_c_mb), y_S_c) in the D-step,
    #   where S_c is the set of accepted synthetic candidates for this cycle.
    #   Set 0 to disable the synthetic-aux term.
    #   117J-tuned: 0.5 ->1.0 to consolidate conditional label semantics on synth.
    ab8c_lambda_aux_synth: float = 1.0
    #
    # ab8c_lambda_adv: weight on the WGAN adversarial term in the G-step.
    #   123J-2: 1.0 ->0.5 ->slow G drift past the F2 peak.
    #   125J: 0.5 ->0.25 ->the adversarial pull is the main driver of drift
    #   away from the now-strong warm-up manifold; halve it.
    ab8c_lambda_adv: float = 0.25
    #
    # ab8c_lambda_aux_g: weight on CE(D.aux(G(z)), y_z) in the G-step.
    #   117J-Rec2: 0.2 ->0.0 ->D.aux drifts per cycle so it is an unstable
    #   teacher for G. Replaced by C-teacher term (ab8c_lambda_cls).
    ab8c_lambda_aux_g: float = 0.0
    #
    # ab8c_lambda_cls: weight on CE(C(G(z)), y_z) in the G-step (pushes G to
    #   produce class-discriminative images for the current C).
    #   117J-Rec2: 0.0 ->0.5 ->C is the most stable of the three players
    #   (sees real data continuously, no adversarial term) so it is a better
    #   teacher than D.aux. Aligns G's gradient with the macro-F1 metric.
    #   C is set to eval() during the G-step's forward (BN/dropout frozen),
    #   then restored.
    #   123J-2: 0.5 ->0.2 ->weaken the "G chases current C" feedback that
    #   drives C to overfit a synthetic minority cluster (precision
    #   collapse). Lower cls-teacher keeps G closer to the real manifold.
    ab8c_lambda_cls: float = 0.2
    #
    # ab8c_lambda_anchor: weight on ``||?_D(G(z)) - ?_D(G_ref(z))||`` in
    #   the G-step. ?_D is the penultimate (8192-dim) feature of D's critic
    #   head, detached so the anchor does not push D. G_ref is the frozen
    #   end-of-warm-up snapshot of G. Acts as a regularizer that prevents G
    #   from drifting too far from its warm-up identity.
    #   117J-tuned: 0.1 ->0.5 ->anchor was too weak vs adv+aux_g+cls pull.
    #   125J: 0.5 ->1.0 ->the warm-up G is now the best generator we have,
    #   so anchor G hard to its strong warm-up identity (paired with the
    #   best-G export guard in _retrain_ab8c).
    #   NOTE: this scalar is the value actually used in the G-step; the
    #   per-class ``ab8c_lambda_anchor_majority/minority`` fields below are
    #   reserved and NOT yet wired into the retrain path.
    ab8c_lambda_anchor: float = 1.0
    #
    # 117J-Rec1: Per-class aux-quality gate 
    # If D.aux's per-class val accuracy drops more than ``ab8c_aux_gate_drop``
    # below its end-of-warm-up baseline, mark that class as DEGRADED for the
    # current cycle. Synthetic samples for degraded classes are then:
    #   ->dropped from D-step extras (no aux_synth signal),
    #   ->masked out of the G-step's aux_g / cls_g per-sample losses,
    #   ->excluded from the C-step's grown pool.
    # Acquisition continues unchanged (so deficit accounting still tracks).
    # Directly attacks observed Trouser FID collapse on Ab-8c.
    ab8c_aux_gate_enabled: bool = True
    ab8c_aux_gate_drop: float = 0.05
    #
    # 117J-Plan1: Gate behavior + hysteresis 
    # 'mask' (legacy Rec1): drop degraded-class synth entirely from
    #     D-extras, G-step aux_g/cls_l, and C-step pool.
    # 'rescue' (new): keep degraded synth signals alive ->D continues
    #     to see them (real-aux =2.0 already rescues D.aux head); G
    #     keeps full cls_l teacher signal on degraded classes (so G
    #     can *improve* the weak class instead of being frozen); C
    #     downsamples degraded synth to ``ab8c_aux_rescue_synth_weight``.
    # Hysteresis: once gated, a class only exits the gate when its
    #     drop recovers to ->``ab8c_aux_gate_drop`` / 2 (prevents per-
    #     cycle on/off flapping that was visible in 117J log).
    ab8c_aux_gate_mode: str = 'rescue'
    ab8c_aux_rescue_synth_weight: float = 0.3
    #
    # 117J-Plan2: Per-class anchor weighting (RESERVED ->not wired) ?
    # Intended to override the scalar ``ab8c_lambda_anchor`` with per-class
    # weights so the majority stays anchored while minorities are free to
    # improve under C-teacher gradients. NOTE (125J): these fields are NOT
    # currently consumed by ``_retrain_ab8c`` ->the G-step uses the scalar
    # ``ab8c_lambda_anchor`` uniformly. Kept for forward-compat / CV-path
    # use; changing them has no effect on the retrain/FID path.
    ab8c_lambda_anchor_majority: float = 0.5
    # 123J-2: 0.1 ->0.3 ->keep minority G closer to its real-data identity
    #   so it stops emitting off-distribution minority samples (precision).
    ab8c_lambda_anchor_minority: float = 0.3
    #
    # 117J-Rec3: Synth-pool refresh from current G 
    # When True, at the END of each AL cycle, re-sample the entire
    # cumulative synth pool from the CURRENT generator (preserving each
    # class's stored count). This eliminates stale early-cycle samples
    # (generated by the warm-up snapshot G) from poisoning the C-step
    # and from inflating per-class FID. BADGE selection is NOT re-run
    # during refresh ->only resampling z and decoding through current G.
    #   123J-1B: True ->False ->re-decoding the whole pool from the
    #   drifting G is a positive-feedback path (G->) that amplifies the
    #   spurious minority cluster C currently prefers. Disabling it keeps C
    #   anchored to the BADGE-selected, on-manifold acquisition samples.
    #   Primary lever on the precision collapse (0.632 ->target 0.70+).
    ab8c_synth_refresh: bool = False
    #
    # 117J-Plan3: Refresh quality guard 
    # During synth-pool refresh, compare current-D critic score s_fake
    # on the new (resampled) batch vs the stored old samples for that
    # class. If new score < old - ``ab8c_refresh_score_tol``, the
    # refresh is rejected for that class and the old samples are kept.
    # This prevents a transiently-bad G cycle from poisoning the pool.
    ab8c_refresh_quality_guard: bool = True
    ab8c_refresh_score_tol: float = 0.0
    #
    # 117J-Lever1: Class-balanced C-step sampler 
    # When True, the C-step (and Phase-3 consolidation) DataLoader uses
    # ``WeightedRandomSampler`` with weights = 1 / class_count[y] so every
    # minibatch is roughly class-balanced. Counteracts Plan-1 rescue
    # downsampling (which leaves the grown pool re-skewed toward real
    # cls-0) and the residual real-pool imbalance.
    ab8c_balanced_sampler: bool = False
    #
    # 117J-Lever2: Label smoothing on C-step CE 
    # Label smoothing only on the Ab-8c classifier loss. Reduces the
    # C's over-confidence on (noisier) synth labels and yields a small
    # macro-F1 gain on imbalanced multi-class problems. 0.0 disables.
    ab8c_label_smoothing: float = 0.00
    #
    # 123J-2: Ab-8c-specific AL-cycle patience 
    # Dedicated patience for the Ab-8c AL loop, keyed to ``selection_metric``
    # (macro-F2).
    #   131J: 4 -> 10 to match CaseStudyConfig.al_patience = 10.
    ab8c_patience: int = 10
    #
    # 125J-v2: Repurpose the third player (D) as an acquisition critic ?
    # The earlier 3-player design co-trained G against C/D. Once the warm-up
    # generator is strong (balanced-critic + shrinkage fixes), co-training G
    # only drifts it off-manifold (the E.1 best-G guard rejects it every
    # time) AND poisons the C-step, since BADGE acquires from the drifting G.
    # v2 removes that failure mode and gives D a *non-gameable* job:
    #
    #   ab8c_freeze_generator: SKIP the per-cycle co-training G-step.
    #
    #       125J-v2 set this True after observing that co-training G drifted it
    #       off-manifold and poisoned the C-step.
    #
    #       131J sets it back to False. Reason: the case-study implementation
    #       (casestudy/methods/m10_ab8c.py) runs the per-cycle G-step, and the
    #       paper reports the ablation and the case studies as the SAME method.
    #       Keeping them different meant "Ab-8c" named two algorithms. If the
    #       125J-v2 drift returns, it is a property of the method that belongs
    #       in the paper, not something to hide by disabling the step in one
    #       study only. Watch the AL cycle log for G drift.
    #
    #   ab8c_acq_filter_tau>0 ->D.aux ACQUISITION FILTER. DISABLED by default
    #       (=0.0) as of 125J-v3: empirically it starved acquisition and
    #       *dropped* test-F2 (0.865 ->0.807). Kept as an opt-in field for
    #       ablation. The case-study implementation has no such filter.
    #   ab8c_acq_filter_min_keep_frac: safety floor when the filter is on.
    ab8c_freeze_generator: bool = False
    ab8c_acq_filter_tau: float = 0.0
    ab8c_acq_filter_min_keep_frac: float = 0.25
    #
    # 125J-v3: Minority generator REFINEMENT ->the real third player ->
    # ROOT CAUSE this fixes: Ab-8b, Ab-8c and BAGAN-GP Static report
    # IDENTICAL FID because they all reuse the same frozen warm-up G. FID is
    # a pure function of the generator, so no selection/classifier change can
    # ever move it. To make Ab-8c BEAT Ab-8b on FID we must fine-tune G's
    # minority output, then feed that better G to Ab-8b's BADGE loop so F2
    # rises too.
    #
    # MECHANISM (runs once, after warm-up, BEFORE the AL cycles, retrain
    # path only  it is what produces the reported G FID/E test-F2):
    #   For ab8c_g_refine_epochs passes, take gradient steps on G ONLY
    #   (critic D frozen) minimizing a Salimans feature-matching loss to the
    #   REAL minority data in D's penultimate feature space ->match per-dim
    #   mean+std of ?_D(G(z_c)) to ?_D(real_c) for each minority class ->plus
    #   an image-space anchor ?(z) ->G_ref(z)->that keeps samples on the
    #   already-good warm-up manifold (prevents drift / FID-hacking). After
    #   each pass measure ACTUAL InceptionV3 minority FID and keep the
    #   best-FID snapshot. Because epoch-0 == warm-up G == Ab-8b, keep-best
    #   GUARANTEES Ab-8c minority FID ->Ab-8b (strictly < if any headroom
    #   exists). The refined G is then frozen for the BADGE loop.
    #
    # NOTE: with only ~40 real minority samples the minority-FID floor is
    # data-limited; if BAGAN-GP already converged the minority manifold this
    # ties Ab-8b rather than beating it. (131J measured exactly that: warm-up
    # 161.1 -> refined 159.7, a 1.4-point gain.)
    #
    # 131J: DISABLED (8 -> 0). The case-study implementation has no pre-AL
    # refinement stage; it updates G inside the AL loop instead. Keeping the
    # refinement here would leave the ablation's "proposed" variant a different
    # algorithm from the one the case studies evaluate. Set >0 to re-enable.
    ab8c_g_refine_epochs: int = 0
    ab8c_g_refine_steps_per_epoch: int = 50
    ab8c_g_refine_batch: int = 64
    ab8c_g_refine_lr_factor: float = 0.02      #  1e-4 WGAN-GP base lr
    ab8c_g_refine_fm_weight: float = 1.0       # feature-matching weight
    ab8c_g_refine_anchor_weight: float = 0.5   # image-space drift anchor
    ab8c_g_refine_fid_samples: int = 256       # fakes/class for keep-best FID
    ##############################################################################

    ##############################################################################
    # Ab-8b-on-BAGAN (mode 'ab8b_bagan') 
    # A vanilla-BAGAN counterpart of Ab-8c so it can sit in the SAME table
    # as the BAGAN-backend ablations (ab1/ab4) for a like-for-like compare.
    # Ab-8c proper REQUIRES gan_backend='bagan_gp' (WGAN-GP critic + aux
    # head + 3-player D/G/C co-training). The vanilla BAGAN discriminator
    # has no critic/aux head ->only an (n+1)-class softmax ->so this variant
    # is deliberately the SELECTION-ONLY core of Ab-8c:
    #   ->Generator FROZEN at its warm-up state (no G-step, no WGAN-GP).
    #   ->BADGE k-means++ acquisition on classifier gradient embeddings
    #     (identical selector to Ab-8b / Ab-8c).
    #   ->D.aux acquisition filter using the BAGAN discriminator's EXISTING
    #     softmax class logits (read-only): drop candidates the discriminator
    #     won't confidently assign to the target minority class before BADGE
    #     picks for diversity. This is the only thing it adds over Ab-8b.
    # Result = "Ab-8b + BAGAN-D acquisition filter", a clean isolation of the
    # discriminator-guided acquisition idea on the vanilla BAGAN backend.
    #
    # ab8b_bagan_pool_size_factor: BADGE overproduction multiplier per cycle.
    ab8b_bagan_pool_size_factor: int = 5
    # ab8b_bagan_kpp_seed_normalize: L2-normalize BADGE gradient embeddings
    #   before k-means++ (matches Ab-8c's normalize=True default).
    ab8b_bagan_kpp_seed_normalize: bool = True
    # ab8b_bagan_acq_filter_tau: keep candidates whose BAGAN-D softmax prob
    #   for the target minority class ->tau. 0.0 disables the filter (then
    #   this mode == Ab-8b on the bagan backend). Enabled by default since
    #   the discriminator filter is the whole point of this variant.
    ab8b_bagan_acq_filter_tau: float = 0.0
    # ab8b_bagan_acq_filter_min_keep_frac: safety floor ->if fewer than
    #   max(n_add, frac  n_candidates) candidates pass tau, fall back to the
    #   top-scoring ones by D prob so acquisition never starves.
    ab8b_bagan_acq_filter_min_keep_frac: float = 0.25    # ab8b_bagan_selection_metric: checkpoint-selection metric for THIS mode's
    #   retrain best-weight restore. Empty '' => fall back to the global
    #   ``selection_metric`` (macro_f2). Set to e.g. 'macro_gmean' to restore
    #   the balanced-recall checkpoint instead of the F2-optimal one, so the
    #   reported recall/g-mean/specificity rise (BADGE already wins F2). Only
    #   ab8b_bagan reads this field; no other mode is affected.
    ab8b_bagan_selection_metric: str = ''    ##############################################################################

    ##############################################################################
    # Ab-9: Pool-Drain Active Learning 
    # Ab-9 generates a balanced pool upfront (same as Ab-4), then drains it
    # via margin-based active learning ->feeding samples in order of
    # classifier uncertainty.  The margin determines exposure ORDER.
    # With patience-based stopping, the loop may terminate early when the
    # remaining pool samples no longer improve the classifier.
    #
    # ab9_init_per_class: how many REAL samples to seed per class.
    #   Prioritize real samples for seeding so the initial decision boundary
    #   is grounded in authentic data.
    ab9_init_per_class: int = 10
    #
    # ab9_samples_per_cycle: N_acq ->samples acquired per AL cycle.
    #   Larger values = fewer cycles = faster but coarser curriculum.
    ab9_samples_per_cycle: int = 10
    #
    # ab9_epochs_per_cycle: training epochs per AL cycle (warm-start).
    ab9_epochs_per_cycle: int = 10
    #
    # ab9_warmup_epochs: epochs on the seed set before AL loop starts.
    ab9_warmup_epochs: int = 5
    #
    # ab9_consolidation_epochs: fine-tuning epochs on the consumed pool
    #   after the AL loop finishes.
    ab9_consolidation_epochs: int = 5
    #
    # ab9_checkpoint_metric: metric used to checkpoint the best model.
    ab9_checkpoint_metric: str = 'macro_f2'
    #
    # ab9_patience: AL-cycle-level patience. Stop draining after this
    #   many consecutive cycles with no improvement in checkpoint metric.
    #   Set to a large value (e.g., 999) to effectively disable and drain fully.
    ab9_patience: int = 20
    ##############################################################################

    ##############################################################################
    # Ab-10: BAGAN-Classifier Co-Training (Separate G Step) 
    # Ab-10 extends Ab-7 (margin-sampled per-batch augmentation) by
    # fine-tuning the BAGAN generator alongside the classifier.
    # After each classifier batch update, G receives a gradient step from
    # CE(C(G(z)), target_class) ->pushing G to produce more class-
    # discriminative images.  See 06_Plan for Ab-10.ipynb for full details.
    #
    # ab10_generator_lr: Learning rate for G's optimizer during co-training.
    #   Must be MUCH lower than the original GAN lr (2e-4) to prevent
    #   catastrophic drift.  Default 1e-5 = 20 lower.
    ab10_generator_lr: float = 1e-5
    #
    # ab10_cls_loss_weight: Scaling factor _cls for the classifier CE loss
    #   applied to G.  g_loss = _cls  CE(C(G(z)), target_class)
    #   Lower values ->more conservative G updates.
    ab10_cls_loss_weight: float = 1.0
    #
    # ab10_generator_update_every: Update G every N classifier batches.
    #   1 = every batch, 3 = every 3rd batch, etc.
    #   Higher values reduce compute cost and slow G adaptation.
    #   Changed 1-> in 101J: reduces noisy early-epoch G updates.
    ab10_generator_update_every: int = 3
    #
    # ab10_grad_clip_max_norm: Maximum gradient norm for G updates.
    #   Prevents exploding gradients from classifier loss propagating
    #   through the deep C ->G graph.
    ab10_grad_clip_max_norm: float = 1.0
    #
    # ab10_g_optimizer: Which optimizer to use for G fine-tuning.
    #   'adam' ->adaptive lr, matches original GAN training dynamics
    #   'sgd'  ->simpler dynamics, no momentum state from prior training
    ab10_g_optimizer: str = 'adam'
    #
    # ab10_g_update_samples: Number of fresh samples per minority class
    #   for the G update step.  None = use n_per_class (batch_size // num_classes).
    ab10_g_update_samples: Optional[int] = None
    ##############################################################################

    ##############################################################################
    # Ab-11: Mixed D+C Loss with Warmup & Throttled G Updates 
    # Ab-11 extends Ab-10 by adding the pre-trained discriminator D as a
    # second loss signal: g_loss = _clsCE(C(G(z)),y) + _advCE(D(G(z)),y)
    # where D's target is the real minority class (AC-GAN style).
    # Additionally: warmup epochs (no G update) and throttled G updates.
    #
    # ab11_d_loss_weight: _adv ->weight for discriminator adversarial term.
    #   g_loss = _cls  CE(C(G(z)), y) + _adv  CE(D(G(z)), y)
    #   Smaller ->more conservative (classifier-dominated).
    #   Changed 0.5->.2 in 101J: stale D signal reduced from 33% to 17% of G loss.
    ab11_d_loss_weight: float = 0.2
    #
    # ab11_warmup_epochs: Number of initial epochs where Phase 2 (G update)
    #   is skipped entirely.  Lets C learn a reasonable boundary first.
    #   Changed 5->0 in 101J: scales warmup to patience=50 (ratio ~1:5).
    ab11_warmup_epochs: int = 10
    #
    # ab11_generator_update_every: Update G every N classifier batches.
    #   5 = every 5th batch.  Reduces G drift rate vs ab10 (default=1).
    ab11_generator_update_every: int = 5
    ##############################################################################

    ##############################################################################
    # Ab-11a: V3 = Ab-11 + Feature Matching + Margin-Selected G 
    # Ab-11a is an evolution of Ab-11 that fixes two diagnosed failure modes
    # (see 03_Ab11a_update_plan.ipynb 1.5):
    #   1. FID stalls because Ab-11's Phase-2 loss has no realism anchor.
    #      Cure: feature-matching (FM) on a frozen feature extractor ->anchors G(z) to the
    #      real minority feature manifold.
    #   2. Binary F1 stalls because Phase 2 trains G on uninformative fakes.
    #      Cure: mirror Phase 1's margin selection ->overproduce G(z),
    #      keep the lowest-margin subset (preserving the grad path to G).
    #
    # Phase 1 is identical to Ab-11. Phase 2 adds: overproduction ->margin-
    # selection ->CE(C) + _advCE(D) + _fmFM. Toggling all ab11a_g_use_*
    # off reproduces Ab-11 exactly (regression-safety; tested).
    #
    # Inherits unchanged from Ab-11: ab10_cls_loss_weight, ab11_d_loss_weight,
    # ab11_warmup_epochs, ab11_generator_update_every, ab10_grad_clip_max_norm,
    # ab10_g_update_samples.

    # ab11a_fm_loss_weight: _fm ->weight on the feature-matching term.
    #   FM losses are typically order-of-magnitude smaller in raw scale than
    #   CE; 10.0 is the recommended starting default (plan 2.4).
    ab11a_fm_loss_weight: float = 10.0
    #
    # ab11a_fm_feature_layer: which feature extractor to use for FM.
    #   'classifier_penultimate'   ->live C, task-aligned (moving target). Default.
    #   'discriminator_penultimate' ->frozen D, BAGAN-aligned (stable).
    #   'autoencoder_encoder'      ->frozen BAGAN encoder (most stable, off-task).
    ab11a_fm_feature_layer: str = 'classifier_penultimate'
    #
    # ab11a_fm_match_var: also match feature variance (mean+var FM).
    #   DEFAULT TRUE: cheapest direct anti-mode-collapse signal available
    #   while D is frozen. WGAN-GP-style cures require unfreezing D and
    #   are deferred to Ab-11b. Set False to reproduce plan 2.4 mean-only FM.
    ab11a_fm_match_var: bool = True
    #
    # ab11a_fm_n_real: real minority batch size for the FM expectation, per
    #   class, per Phase-2 step. None = use the full minority pool every
    #   step (locked decision: full pool, lowest-variance estimator on
    #   the ~24-img/class minority budget).
    ab11a_fm_n_real: Optional[int] = None
    #
    # ab11a_g_use_margin_selection: enable margin-based selection of G(z)
    #   in Phase 2. When True, overproduce ab11a_g_overproduction  n_g
    #   candidates, score with C, keep the lowest-margin n_g.
    ab11a_g_use_margin_selection: bool = True
    #
    # ab11a_g_overproduction: candidate-overproduction factor for margin
    #   selection in Phase 2. Mirrors margin_overproduction_factor used in
    #   Phase 1 top-up. Keep ->4 to bound compute.
    ab11a_g_overproduction: int = 4
    #
    # ab11a_g_dedupe_cosine_threshold: optional dedup of selected fakes in
    #   ?-space to prevent G winning by producing K copies of one hard fake.
    #   None = disabled. Typical value if enabled: 0.95.
    ab11a_g_dedupe_cosine_threshold: Optional[float] = None
    #
    # ab11a_g_diversity_weight: optional pairwise-diversity term on G(z)
    #   (Yang et al. 2019, anti-collapse). Off by default; enable only if
    #   mean+var FM proves insufficient.
    ab11a_g_diversity_weight: float = 0.0
    #
    # ab11a_g_use_feature_matching: master toggle for the FM term. With
    #   ab11a_g_use_margin_selection, lets V1/V2/V3 ablations be run from
    #   the same code path. Both False ->Ab-11a == Ab-11 (regression).
    ab11a_g_use_feature_matching: bool = True
    #
    # Step E: per-epoch diagnostics & cheap FID monitor 
    # ab11a_verbose_phase2: print one diagnostic line per fold-epoch
    #   summarising the Phase-2 loss components (g_cls, g_adv, g_fm,
    #   sel_margin) plus train/val numbers. Off only for sweeps where
    #   stdout noise hurts.
    ab11a_verbose_phase2: bool = True
    #
    # ab11a_fid_eval_every: cheap FID monitor cadence in epochs. 0 (default)
    #   ->disabled. When > 0 and `ablation.fid` is importable (Step H),
    #   compute per-class FID against a cached real ref every N epochs
    #   using `ab11a_fid_monitor_samples` fakes per class. Results land
    #   in self.fold_history['fid_class_<c>'] as (epoch, fid) pairs.
    ab11a_fid_eval_every: int = 0
    #
    # ab11a_fid_monitor_samples: N fakes per class for the cheap monitor
    #   (full Step-H sweep uses N=10_000; 2_000 is the cheap default).
    ab11a_fid_monitor_samples: int = 2000
    ##############################################################################

    # Device configuration
    device: str = "auto"  # auto-detect GPU, fallback to CPU

    # Output controls for CV/retrain orchestration.
    # - cv_verbose: if False, suppress most run_kfold_ablation progress logs.
    # - cv_show_bagan_plots: if False, disable BAGAN AE/GAN matplotlib displays.
    cv_verbose: bool = False
    cv_show_bagan_plots: bool = False
    
    # Experiment metadata
    # 514J: New experiment ->Chung-matched pool (800:80:80), balanced test,
    #        no BatchNorm in classifier, LR=0.0005
    date_suffix: str = "Ab_115J"

    # GAN backend selection (110J). Either 'bagan' (vanilla BAGAN, default ->
    # preserves 106J behaviour exactly) or 'bagan_gp' (WGAN-GP critic with
    # auxiliary classifier head). Validated in __post_init__.
    gan_backend: str = "bagan_gp"
    
    @property
    # what does '@property' do?
    # The '@property' decorator allows you to define methods in a class that can be accessed like attributes.
    def num_classes(self) -> int:
        """Number of classes in the experiment."""
        return len(self.selected_classes)
    
    @property
    def class_names(self) -> Dict[int, str]:
        """Get class names for selected classes."""
        return {c: FASHION_MNIST_CLASSES[c] for c in self.selected_classes}
    
    @property
    def majority_class(self) -> int:
        """Get the majority class label (first class with proportion 1.0)."""
        return self.selected_classes[0]
    
    @property
    def minority_classes(self) -> List[int]:
        """Get minority class labels (all except first)."""
        return self.selected_classes[1:]
    
    def get_device(self) -> str:
        """Resolve device setting, auto-detecting GPU if available.
        
        Returns:
            'cuda' if GPU available and device='auto' or 'cuda',
            'cpu' otherwise
        """
        import torch
        if self.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.device
    
    def get_retrain_patience(self, mode: str) -> int:
        """Get retrain early-stopping patience for a given mode.
        
        Returns mode-specific override if set, else 2x the CV patience.
        """
        return self.retrain_patience.get(mode, self.patience * 2)
        # return self.retrain_patience.get(mode, self.patience)
    
    def get_retrain_max_epochs(self, mode: str, avg_stopping_epoch: int) -> int:
        """Get retrain max epoch cap for a given mode.
        
        Returns mode-specific override if set, else max(avg_stop * 2, 30).
        """
        return self.retrain_max_epochs.get(mode, avg_stopping_epoch * 2)
        # return self.retrain_max_epochs.get(mode, avg_stopping_epoch)
    
    def get_selection_params(self, mode: str) -> Dict[str, object]:
        """Acquisition-rule settings for an Ab-8 family mode.

        Centralizes the per-mode lookup so the CV loop (``trainer.py``) and the
        retrain loop (``kfold.py``) can never drift apart on which rule a mode
        uses. Returns the keyword arguments expected by
        ``ablation.selection.select_indices`` plus the candidate pool factor.
        """
        table = {
            'ab8b': dict(
                mode=self.ab8b_selection_mode,
                min_confidence=self.ab8b_min_confidence,
                normalize=self.ab8b_kpp_seed_normalize,
                pool_factor=self.ab8b_pool_size_factor,
            ),
            'ab8d': dict(
                mode='confident_diverse',
                min_confidence=self.ab8d_min_confidence,
                normalize=self.ab8d_kpp_seed_normalize,
                pool_factor=self.ab8d_pool_size_factor,
            ),
            'ab8c': dict(
                mode=self.ab8c_selection_mode,
                min_confidence=self.ab8c_min_confidence,
                normalize=self.ab8c_kpp_seed_normalize,
                pool_factor=self.ab8c_pool_size_factor,
            ),
        }
        # ab8b aliases share ab8b's settings.
        table['ab8b_without_gp'] = table['ab8b']
        table['ab8b_bagan'] = table['ab8b']
        if mode not in table:
            raise KeyError(
                f"get_selection_params: mode {mode!r} has no acquisition rule; "
                f"expected one of {sorted(table)}"
            )
        return table[mode]
    
    def parse_augmentation_target(self) -> Tuple[float, ...]:
        """Parse augmentation_target string into proportions.
        
        Returns:
            Tuple of proportions (e.g., (50, 25, 25) for "50:25:25")
        
        Example:
            >>> config = AblationConfig(augmentation_target="33:33:33")
            >>> config.parse_augmentation_target()
            (33.0, 33.0, 33.0)
        """
        parts = self.augmentation_target.split(':')
        return tuple(float(p) for p in parts)
    
    def get_model_path(self, model_name: str) -> str:
        """Get full path for saving/loading a model.
        
        Args:
            model_name: Base name of the model (e.g., 'bagan_generator')
        
        Returns:
            Full path with directory and date suffix
            
        Example:
            >>> config = AblationConfig(date_suffix="511J", bagan_model_dir="saved")
            >>> config.get_model_path("bagan_generator")
            'saved/bagan_generator_511J.pth'
        """
        return os.path.join(self.bagan_model_dir, f"{model_name}_{self.date_suffix}.pth")
    
    def _resolve_run_seed(self) -> None:
        """Default ``run_seed`` to ``random_seed`` so single-seed runs are unchanged."""
        if self.run_seed is None:
            self.run_seed = self.random_seed

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._resolve_run_seed()

        if len(self.selected_classes) != len(self.class_proportions):
            raise ValueError(
                f"selected_classes length ({len(self.selected_classes)}) must match "
                f"class_proportions length ({len(self.class_proportions)})"
            )
        
        if self.num_classes < 2:
            raise ValueError(f"Must have at least 2 classes, got {self.num_classes}")
        
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        
        if not all(p > 0 for p in self.class_proportions):
            raise ValueError("All class_proportions must be positive")
        
        # Validate augmentation_target format
        try:
            parts = self.parse_augmentation_target()
            if len(parts) != self.num_classes:
                raise ValueError(
                    f"augmentation_target has {len(parts)} parts but num_classes is {self.num_classes}"
                )
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid augmentation_target format '{self.augmentation_target}': {e}")
        
        # Validate selected_classes are valid Fashion-MNIST labels
        for c in self.selected_classes:
            if c not in FASHION_MNIST_CLASSES:
                raise ValueError(f"Invalid class label {c}. Must be 0-9 for Fashion-MNIST.")
        
        # Validate device
        if self.device not in ('auto', 'cuda', 'cpu'):
            raise ValueError(f"device must be 'auto', 'cuda', or 'cpu', got '{self.device}'")

        # Validate GAN backend selection (110J).
        if self.gan_backend not in ('bagan', 'bagan_gp'):
            raise ValueError(
                f"gan_backend must be 'bagan' or 'bagan_gp', got {self.gan_backend!r}"
            )

