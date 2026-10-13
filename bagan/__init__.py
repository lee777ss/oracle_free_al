"""
BAGAN: Balancing GAN for Imbalanced Dataset Augmentation

Implementation based on:
Mariani et al. (2018) "BAGAN: Data Augmentation with Balancing GAN"

Key components:
- Autoencoder-based initialization
- Class-conditional latent space with learned Gaussian distributions
- Single (n+1) discriminator output (classes + fake)

Module Structure:
    bagan/
    ├── __init__.py          ← This file (package root, exports public API)
    ├── config.py            ← BAGANConfig dataclass (Step 1) ✓
    ├── models/
    │   ├── __init__.py
    │   ├── autoencoder.py   ← BAGANEncoder, BAGANDecoder, BAGANAutoencoder (Step 2) ✓
    │   ├── generator.py     ← BAGANGenerator (Step 3) ✓
    │   └── discriminator.py ← BAGANDiscriminator (Step 4) ✓
    ├── training/
    │   ├── __init__.py
    │   ├── latent_space.py          ← ClassConditionalLatentGenerator (Step 5) ✓
    │   ├── autoencoder_trainer.py   ← AutoencoderTrainer (Step 6) ✓
    │   └── gan_trainer.py           ← GANTrainer (Step 7) ✓
    └── utils/
        ├── __init__.py
        ├── weight_transfer.py       ← WeightTransfer (Step 8) ✓
        └── visualization.py         ← BAGANVisualizer (Step 9) ✓
"""

# =============================================================================
# Configuration (Step 1)
# =============================================================================
# Relative import: The dot (.) means "current package" (bagan/)
# This imports BAGANConfig class from bagan/config.py
# Equivalent to: from bagan.config import BAGANConfig (when called from outside)
# This allows users to do: from bagan import BAGANConfig
from .config import BAGANConfig

# =============================================================================
# Models (Steps 2-4)
# =============================================================================
from .models import BAGANEncoder, BAGANDecoder, BAGANAutoencoder  # Step 2
from .models import BAGANGenerator                                  # Step 3
from .models import BAGANDiscriminator, BAGANDiscriminatorLoss      # Step 4

# =============================================================================
# Training (Steps 5-7)
# =============================================================================
# Import Resolution: Two-Level Import Chain
# ------------------------------------------
# These imports go through TWO __init__.py files:
#
# 1. First level: bagan/__init__.py (this file)
#    from .training import AutoencoderTrainer
#    The ".training" resolves to bagan/training/__init__.py
#
# 2. Second level: bagan/training/__init__.py
#    from .autoencoder_trainer import AutoencoderTrainer
#    The ".autoencoder_trainer" resolves to bagan/training/autoencoder_trainer.py
#
# Flow: User code → bagan/__init__.py → bagan/training/__init__.py → actual module
#
# This creates a clean public API:
#   ✓ Users write: from bagan import AutoencoderTrainer
#   ✗ Instead of:  from bagan.training.autoencoder_trainer import AutoencoderTrainer
#
# Benefits:
#   - Cleaner imports for users (shorter, more intuitive)
#   - Internal file structure can be refactored without breaking user code
#   - Clear namespace organization (bagan.training.* for training utilities)

from .training import ClassConditionalLatentGenerator  # Step 5
from .training import AutoencoderTrainer               # Step 6
from .training import GANTrainer                       # Step 7

# =============================================================================
# Utilities (Steps 8-9)
# =============================================================================
# Weight transfer from pre-trained Autoencoder to GAN models
# This is a key BAGAN innovation: initialize GAN with AE weights for stability
from .utils import WeightTransfer                      # Step 8

# Visualization utilities for latent space analysis and sample display
from .utils import BAGANVisualizer, show_samples, show_latent_space  # Step 9
from .utils import realism_score, adv_class_logits, discriminator_features  # backend-adapter helpers

__version__ = "1.0.0"
__author__ = "Based on Mariani et al. (2018)"

# =============================================================================
# Public API Definition
# =============================================================================
# __all__ defines what gets exported when using "from bagan import *"
# Organized by functionality for clarity

__all__ = [
    # Configuration (Step 1)
    "BAGANConfig",
    
    # Models (Steps 2-4)
    "BAGANEncoder",
    "BAGANDecoder", 
    "BAGANAutoencoder",
    "BAGANGenerator",
    "BAGANDiscriminator",
    "BAGANDiscriminatorLoss",
    
    # Training (Steps 5-7)
    "ClassConditionalLatentGenerator",
    "AutoencoderTrainer",
    "GANTrainer",
    
    # Utilities (Steps 8-9)
    "WeightTransfer",
    "BAGANVisualizer",
    "show_samples",
    "show_latent_space",
    # Backend-adapter helpers (used by ablation.gan_backend)
    "realism_score",
    "adv_class_logits",
    "discriminator_features",
]
