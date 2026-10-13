"""
Data augmentation strategies for imbalanced multi-class classification.

Implements augmentation modes for ablation studies:
- Ab-1:  No augmentation (imbalanced baseline) — handled by trainer, no function here
- Ab-1a: Classic oversampling — repeat real minority samples (oversample_minority_classes)
- Ab-4:  BAGAN static — generate fixed augmented dataset ONCE (generate_bagan_augmented_dataset)
- Ab-5:  BAGAN dynamic — class-stratified per-batch batching handled entirely in
         trainer._train_epoch_ab5() (does NOT call generate_bagan_augmented_dataset)
- Ab-6:  BAGAN dynamic + quality gate — like Ab-5 but BAGAN top-up is overproduced
         and filtered by discriminator P(fake). Handled entirely in
         trainer._train_epoch_ab6() (does NOT call any function here)

Architecture:
    Ab-4 uses generate_bagan_augmented_dataset() once before training.
    Ab-5 builds balanced batches directly inside _train_epoch_ab5() using
    class-stratified without-replacement batching + per-batch BAGAN top-up.
    Ab-6 extends Ab-5: same batching, but overproduces BAGAN candidates and
    keeps the most realistic ones (lowest discriminator P(fake)).

References:
    427J create_balanced_batch_bagan()           → line 17440
    427J create_balanced_batch_bagan_al_margin() → line 18422
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn.functional as F


# ============================================================================
# HELPER: Compute target counts per class from config
# ============================================================================
# This helper translates the augmentation_target string (e.g. "50:25:25") into
# actual sample counts. It is used by EVERY augmentation function below.
#
# Logic:
#   1. Parse "50:25:25" → proportions (50.0, 25.0, 25.0)
#   2. Count majority class samples in current data → n_majority
#   3. Scale each proportion relative to majority:
#      target_count[class_i] = n_majority × (proportion_i / proportion_majority)
#   4. For majority class: target = current count (keep all)
#   5. For minority classes: target = computed count (deficit = target - current)
# ============================================================================

def _compute_target_counts(
    train_labels: torch.Tensor,
    config
) -> dict:
    """Compute how many samples each class should have after augmentation.
    
    Args:
        train_labels: Current training labels [N]
        config: AblationConfig with selected_classes, augmentation_target
    
    Returns:
        Dict mapping class_label → target_count
        
    Example:
        >>> # config.augmentation_target = "50:25:25", classes = [0, 2, 3]
        >>> # Current: {0: 5400, 2: 270, 3: 270}
        >>> targets = _compute_target_counts(labels, config)
        >>> # proportions = (50, 25, 25) → ratio = 25/50 = 0.5
        >>> # targets = {0: 5400, 2: 5400*0.5=2700, 3: 5400*0.5=2700}
    """
    # Step 1: Parse the target proportions from config
    # "50:25:25" → (50.0, 25.0, 25.0)
    proportions = config.parse_augmentation_target()
    
    # Step 2: Count current samples per class
    # We iterate through config.selected_classes to ensure consistent ordering
    current_counts = {}
    for cls in config.selected_classes:
        current_counts[cls] = (train_labels == cls).sum().item()
    
    # Step 3: The majority class (index 0 in selected_classes) anchors the scaling
    # All other classes are scaled relative to this anchor
    majority_class = config.selected_classes[0]
    n_majority = current_counts[majority_class]
    majority_proportion = proportions[0]  # e.g. 50.0
    
    # Step 4: Compute target count for each class
    # target[i] = n_majority × (proportion[i] / proportion[majority])
    target_counts = {}
    for i, cls in enumerate(config.selected_classes):
        ratio = proportions[i] / majority_proportion  # e.g. 25/50 = 0.5
        target_counts[cls] = int(math.ceil(n_majority * ratio))
    
    return target_counts


# ============================================================================
# Ab-1a: CLASSIC OVERSAMPLING — Repeat Real Minority Samples
# ============================================================================
# 427J equivalent: WeightedRandomSampler-based balanced DataLoader
# 
# In 427J, Ab-1a used a WeightedRandomSampler to create a balanced DataLoader
# that oversampled minority indices within the standard PyTorch DataLoader loop.
# 
# For 511J, we take a simpler dataset-level approach:
#   1. Compute target counts per class (using _compute_target_counts)
#   2. For each minority class: randomly sample (with replacement) from its
#      real samples until we reach the target count
#   3. Concatenate all original data + duplicated minority samples
#
# "With replacement" means the same real image can appear multiple times.
# This is equivalent to WeightedRandomSampler over an epoch's worth of batches.
# ============================================================================

def oversample_minority_classes(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ab-1a: Classic oversampling — repeat real minority samples.
    
    No BAGAN, no synthetic data. Purely duplicates existing real minority
    samples randomly with replacement until each class reaches its target count
    defined by config.augmentation_target.
    
    Args:
        train_data: Training images [N, 1, 64, 64]
        train_labels: Training labels [N]
        config: AblationConfig with selected_classes, augmentation_target
    
    Returns:
        balanced_data: [N_balanced, 1, 64, 64] — all original + duplicated samples
        balanced_labels: [N_balanced]
    
    Example:
        >>> # config.augmentation_target = "50:25:25", classes = [0, 2, 3]
        >>> # Original: {0: 5400, 2: 270, 3: 270}
        >>> bal_data, bal_labels = oversample_minority_classes(data, labels, config)
        >>> # Result: {0: 5400, 2: 2700, 3: 2700}
        >>> # class 2: 270 original + 2430 duplicated = 2700
        >>> # class 3: 270 original + 2430 duplicated = 2700
    """
    # Compute how many samples each class should have
    target_counts = _compute_target_counts(train_labels, config)
    
    # Start with all original data (we keep everything, including majority)
    all_data = [train_data]
    all_labels = [train_labels]
    
    for cls in config.minority_classes:
        # Find indices of this minority class in the training data
        # cls_mask is a boolean tensor: True where label matches this class
        cls_mask = (train_labels == cls)
        cls_data = train_data[cls_mask]    # All real images of this class
        current_count = cls_data.size(0)   # How many we currently have
        target_count = target_counts[cls]  # How many we need
        
        # deficit = number of additional copies needed
        deficit = target_count - current_count
        
        if deficit > 0:
            # Sample indices WITH REPLACEMENT from available real samples
            # torch.randint(0, current_count, (deficit,)) picks `deficit` random
            # indices in range [0, current_count), allowing repeats
            repeat_indices = torch.randint(0, current_count, (deficit,))
            
            # Gather the duplicated images and create their labels
            duplicated_data = cls_data[repeat_indices]
            duplicated_labels = torch.full((deficit,), cls, dtype=train_labels.dtype)
            
            all_data.append(duplicated_data)
            all_labels.append(duplicated_labels)
    
    # Concatenate original + all duplicated samples along batch dimension
    balanced_data = torch.cat(all_data, dim=0)
    balanced_labels = torch.cat(all_labels, dim=0)
    
    return balanced_data, balanced_labels


# ============================================================================
# Ab-1b: REAL RESERVE AUGMENTATION — Inject Unique Disjoint Real Samples
# ============================================================================
# Ab-1b augments the imbalanced training fold with UNIQUE real images from
# a reserve pool that is disjoint from pool_data. This is the "oracle"
# baseline — balancing with real data only, no synthetic generation, no
# duplicate oversampling.
#
# Unlike Ab-1a (which duplicates training fold images), Ab-1b draws fresh
# real images that the classifier has never seen, giving the best possible
# real-data augmentation. If Ab-1b >> Ab-1, class imbalance is the bottleneck.
# If Ab-4 ≈ Ab-1b, BAGAN quality matches real data.
#
# The reserve pool is computed on-the-fly by run_kfold_ablation() from
# Fashion-MNIST images NOT in pool_data, ensuring complete disjointness
# from both training and validation folds.
# ============================================================================

def augment_with_real_reserve(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    reserve_data: torch.Tensor,
    reserve_labels: torch.Tensor,
    config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ab-1b: Fill minority deficit with unique real images from a reserve pool.
    
    Same target distribution as Ab-1a (config.augmentation_target), but instead
    of duplicating existing training fold images, injects UNIQUE real images
    from a disjoint reserve pool.
    
    Args:
        train_data: Training fold images [N, 1, 64, 64]
        train_labels: Training fold labels [N]
        reserve_data: Disjoint reserve images [M, 1, 64, 64] (preprocessed)
        reserve_labels: Reserve labels [M]
        config: AblationConfig with selected_classes, augmentation_target
    
    Returns:
        augmented_data: [N_aug, 1, 64, 64] — train + sampled reserve
        augmented_labels: [N_aug]
    """
    target_counts = _compute_target_counts(train_labels, config)
    
    all_data = [train_data]
    all_labels = [train_labels]
    
    for cls in config.minority_classes:
        current_count = (train_labels == cls).sum().item()
        target_count = target_counts[cls]
        deficit = target_count - current_count
        
        if deficit <= 0:
            continue
        
        # Get reserve images for this class
        cls_mask = (reserve_labels == cls)
        cls_reserve = reserve_data[cls_mask]
        n_available = cls_reserve.size(0)
        
        if n_available == 0:
            continue
        
        if deficit <= n_available:
            # Sample WITHOUT replacement — all unique images
            indices = torch.randperm(n_available)[:deficit]
            sampled = cls_reserve[indices]
        else:
            # Take all unique + oversample remainder (unlikely with large reserve)
            remainder = deficit - n_available
            repeat_indices = torch.randint(0, n_available, (remainder,))
            sampled = torch.cat([cls_reserve, cls_reserve[repeat_indices]], dim=0)
        
        sampled_labels = torch.full(
            (sampled.size(0),), cls, dtype=train_labels.dtype
        )
        all_data.append(sampled)
        all_labels.append(sampled_labels)
    
    return torch.cat(all_data, dim=0), torch.cat(all_labels, dim=0)


# ============================================================================
# Ab-4 / Ab-5: BAGAN AUGMENTATION — Generate Synthetic Minority Samples
# ============================================================================
# 427J equivalent: create_balanced_batch_bagan() at line 17440
#
# This is the core BAGAN payoff: use the pre-trained generator + latent_generator
# to synthesize minority class images and balance the dataset.
#
# Generation process for each minority class:
#   1. Compute deficit = target_count - current_count
#   2. Sample class-conditioned latent vectors:
#      z = latent_generator.sample_for_class_oversampling(class, deficit)
#      This draws from N(μ_c, σ_c²I) — the fitted class-specific distribution
#   3. Generate images: fake_images = generator(z)
#      BAGAN generator is UNCONDITIONAL — it takes only z, no class labels.
#      The class conditioning is entirely in the latent space (via μ_c, σ_c).
#   4. Concatenate real + synthetic
#
# NOTE: Ab-5 NO LONGER calls this function. It now uses class-stratified
# without-replacement batching inside trainer._train_epoch_ab5(), which
# handles majority/minority partitioning and BAGAN top-up per batch.
# This function is now used only by Ab-4 (and kept as a building block).
# ============================================================================

def generate_bagan_augmented_dataset(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    generator,
    latent_generator,
    config,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ab-4: Generate synthetic minority samples via BAGAN.
    
    For each minority class, compute the deficit (target - current), 
    sample class-conditioned latent vectors from latent_generator, 
    pass through the BAGAN generator, and concatenate with real data.
    
    Used by Ab-4 (called once before training → fixed augmented dataset).
    Ab-5 now handles augmentation directly in _train_epoch_ab5().
    
    Args:
        train_data: Training images [N, 1, 64, 64]
        train_labels: Training labels [N]
        generator: Pre-trained BAGANGenerator (unconditional: takes z only)
        latent_generator: Pre-trained ClassConditionalLatentGenerator
                          with fitted per-class distributions N(μ_c, σ_c²I)
        config: AblationConfig with selected_classes, augmentation_target
        device: Device for generation ('cuda' or 'cpu')
    
    Returns:
        augmented_data: [N_aug, 1, 64, 64] — real + synthetic images
        augmented_labels: [N_aug] — corresponding labels
    
    Example:
        >>> # config.augmentation_target = "50:25:25", classes = [0, 2, 3]
        >>> # Original: {0: 5400, 2: 270, 3: 270}
        >>> aug_data, aug_labels = generate_bagan_augmented_dataset(
        ...     data, labels, bagan_gen, latent_gen, config, 'cuda'
        ... )
        >>> # class 2: 270 real + 2430 synthetic = 2700
        >>> # class 3: 270 real + 2430 synthetic = 2700
        >>> # Result: {0: 5400, 2: 2700, 3: 2700}
    """
    # Compute how many samples each class should have
    target_counts = _compute_target_counts(train_labels, config)
    
    # Start with all original data
    all_data = [train_data]
    all_labels = [train_labels]
    
    # Put generator in eval mode (disables dropout, batchnorm uses running stats)
    # Save previous training state so we can restore it after generation
    was_training = generator.training
    generator.eval()
    
    try:
        # Disable gradient computation — we're only doing inference here
        with torch.no_grad():
            for cls in config.minority_classes: # Generate fake samples per minority class until deficit is met
                current_count = (train_labels == cls).sum().item()
                target_count = target_counts[cls]
                deficit = target_count - current_count
                
                if deficit <= 0:
                    continue  # This class already has enough samples
                
                # ── Step 1: Sample class-conditioned latent vectors ──
                # latent_generator.sample_for_class_oversampling() draws from
                # the fitted distribution N(μ_c, σ_c²I) for this specific class.
                # This is how BAGAN achieves class conditioning with an
                # unconditional generator: the class info is encoded in z.
                z = latent_generator.sample_for_class_oversampling(
                    target_class=cls,
                    n_samples=deficit
                )
                # z shape: [deficit, latent_dim] e.g. [2430, 128]
                
                # ── Step 2: Generate images in batches ──
                # We chunk into batches to avoid GPU OOM on large deficits
                gen_batch_size = config.batch_size  # typically 64
                synthetic_chunks = []
                
                for i in range(0, deficit, gen_batch_size):
                    z_batch = z[i:min(i + gen_batch_size, deficit)]
                    # Generator: z [batch, 128] → image [batch, 1, 64, 64]
                    fake_batch = generator(z_batch)
                    # Move to CPU immediately to free GPU memory
                    synthetic_chunks.append(fake_batch.cpu())
                
                # ── Step 3: Combine all generated chunks ──
                synthetic_images = torch.cat(synthetic_chunks, dim=0)
                synthetic_labels = torch.full(
                    (deficit,), cls, dtype=train_labels.dtype
                )
                
                all_data.append(synthetic_images)
                all_labels.append(synthetic_labels)
    finally:
        # Restore generator to its previous training state
        # This is important because the generator might be used elsewhere
        generator.train(mode=was_training)
    
    augmented_data = torch.cat(all_data, dim=0)
    augmented_labels = torch.cat(all_labels, dim=0)
    
    return augmented_data, augmented_labels
