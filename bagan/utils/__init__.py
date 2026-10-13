"""
BAGAN Utilities Package

This package provides helper utilities for BAGAN:
    - Weight transfer from Autoencoder to GAN models (Step 8)
    - Visualization tools for latent space and samples (Step 9)

Module Structure:
    bagan/utils/
    ├── __init__.py           ← This file (package exports)
    ├── weight_transfer.py    ← WeightTransfer class (Step 8) ✓
    └── visualization.py      ← BAGANVisualizer class (Step 9) ✓

Import Resolution:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ User code: from bagan import WeightTransfer, BAGANVisualizer        │
    │     ↓                                                               │
    │ bagan/__init__.py: from .utils import WeightTransfer, BAGANVisualizer│
    │     ↓                                                               │
    │ bagan/utils/__init__.py: from .weight_transfer import WeightTransfer│
    │                          from .visualization import BAGANVisualizer │
    │     ↓                                                               │
    │ bagan/utils/weight_transfer.py: class WeightTransfer                │
    │ bagan/utils/visualization.py: class BAGANVisualizer                 │
    └─────────────────────────────────────────────────────────────────────┘

BAGAN Pipeline (Steps 1-9):
    Step 1: BAGANConfig            - Configuration dataclass
    Step 2: BAGANAutoencoder       - Encoder + Decoder
    Step 3: BAGANGenerator         - Same architecture as Decoder
    Step 4: BAGANDiscriminator     - Conv layers same as Encoder
    Step 5: ClassConditionalLatentGenerator - Learn class Gaussians
    Step 6: AutoencoderTrainer     - Pre-train autoencoder
    Step 7: GANTrainer             - Train GAN with (n+1) loss
    Step 8: WeightTransfer         - Transfer AE → GAN weights
    Step 9: BAGANVisualizer        - Latent space & sample visualization ← THIS STEP
    
Weight Transfer Summary:
    ┌─────────────────┐     ┌─────────────────┐
    │  Autoencoder    │     │     GAN         │
    ├─────────────────┤     ├─────────────────┤
    │   Encoder       │────►│  Discriminator  │  (conv layers)
    │   (conv layers) │     │  (conv layers)  │
    ├─────────────────┤     ├─────────────────┤
    │   Decoder       │────►│  Generator      │  (all layers)
    │   (all layers)  │     │  (all layers)   │
    └─────────────────┘     └─────────────────┘

Visualization Features:
    - Latent space plots (PCA/t-SNE projections)
    - Class distribution visualization
    - Generated sample grids
    - Real vs. fake comparison
    - Training history curves
    - Sample progression over epochs
"""

# =============================================================================
# Export Weight Transfer (Step 8)
# =============================================================================
from .weight_transfer import WeightTransfer

# =============================================================================
# Export Visualization (Step 9)
# =============================================================================
from .visualization import BAGANVisualizer, show_samples, show_latent_space

# =============================================================================
# Backend-adapter helpers (used by ablation.gan_backend for BAGAN ↔ BAGAN-GP)
# =============================================================================
# These are intentionally simple — they translate the BAGAN discriminator's
# native output shape into the three "abstract" signals the ablation harness
# needs:
#
#   • realism_score(disc, x)        →  [B] tensor, lower-is-better realism
#   • adv_class_logits(disc, x)     →  [B, num_classes + 1] logits for CE
#   • discriminator_features(disc, x) → [B, D] flattened penultimate features
#
# Keeping the full [B, num_classes + 1] tensor in `adv_class_logits` preserves
# the exact softmax normalisation that 106J/110J's BAGAN runs use; switching
# to [B, num_classes] would silently change CE values. The :mod:`bagan_gp`
# package implements the same three names with WGAN-GP semantics, so the
# ablation harness can dispatch through `ablation.gan_backend.get_gan_modules`.
import torch as _torch


def realism_score(discriminator, x: _torch.Tensor) -> _torch.Tensor:
    """**Lower = more real.** For BAGAN this is ``P(fake)`` from the softmax
    over the discriminator's ``(num_classes + 1)``-way logits — i.e.
    ``softmax(disc(x))[:, -1]``."""
    logits = discriminator(x)
    probs = _torch.softmax(logits, dim=1)
    return probs[:, -1]


def adv_class_logits(discriminator, x: _torch.Tensor) -> _torch.Tensor:
    """Adversarial-training class logits for ``cross_entropy``.

    For BAGAN this returns the full ``[B, num_classes + 1]`` discriminator
    output so that CE softmax normalisation includes the fake column —
    matching the exact loss values produced by GAAL_Ablation_106J. The
    target tensor passed to CE is still in ``[0, num_classes - 1]``."""
    return discriminator(x)


def discriminator_features(discriminator, x: _torch.Tensor) -> _torch.Tensor:
    """Flattened penultimate features ``[B, D]``. For BAGAN, ``D = 8192``."""
    h = discriminator.conv_layers(x)
    return h.view(h.size(0), -1)


# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Step 8: Weight Transfer
    "WeightTransfer",

    # Step 9: Visualization
    "BAGANVisualizer",
    "show_samples",
    "show_latent_space",

    # Backend-adapter helpers (ablation.gan_backend)
    "realism_score",
    "adv_class_logits",
    "discriminator_features",
]
