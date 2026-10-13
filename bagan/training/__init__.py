"""
BAGAN Training Module

Contains training utilities for BAGAN:
- ClassConditionalLatentGenerator: The key BAGAN innovation for class-conditional sampling
- AutoencoderTrainer: Training loop for autoencoder pretraining (Step 6)
- GANTrainer: Training loop for GAN with class-conditional generation (Step 7)

Module Structure:
    bagan/training/
    ├── __init__.py              ← This file (exports public API)
    ├── latent_space.py          ← ClassConditionalLatentGenerator (Step 5) ✓
    ├── autoencoder_trainer.py   ← AutoencoderTrainer (Step 6) ✓
    └── gan_trainer.py           ← GANTrainer (Step 7) ✓
"""

# =============================================================================
# Import Training Components
# =============================================================================
# This __init__.py acts as a MIDDLE LAYER in the import chain:
#
# Import Chain Visualization:
#   User Code
#      ↓
#   from bagan import AutoencoderTrainer
#      ↓ (resolves to bagan/__init__.py)
#   from .training import AutoencoderTrainer
#      ↓ (resolves to bagan/training/__init__.py - THIS FILE)
#   from .autoencoder_trainer import AutoencoderTrainer
#      ↓ (resolves to bagan/training/autoencoder_trainer.py)
#   class AutoencoderTrainer: ...
#
# Role of THIS file (bagan/training/__init__.py):
#   1. Import from actual implementation files (.latent_space, .autoencoder_trainer)
#   2. Re-export them as part of bagan.training module's public API
#   3. Allow parent module (bagan/__init__.py) to import from "training" namespace
#
# Without this file, users would need to write:
#   from bagan.training.autoencoder_trainer import AutoencoderTrainer  # Too verbose!
#
# With this file, users can write:
#   from bagan import AutoencoderTrainer                    # Clean! ✓
#   from bagan.training import AutoencoderTrainer          # Also works! ✓

from .latent_space import ClassConditionalLatentGenerator
from .autoencoder_trainer import AutoencoderTrainer
from .gan_trainer import GANTrainer

# =============================================================================
# Public API Definition
# =============================================================================
# __all__ defines what gets exported when using "from bagan.training import *"
# It also serves as documentation of the module's public interface

__all__ = [
    # Step 5: Class-Conditional Latent Sampling (THE KEY INNOVATION)
    'ClassConditionalLatentGenerator',
    
    # Step 6: Autoencoder Training
    'AutoencoderTrainer',
    
    # Step 7: GAN Training
    'GANTrainer',
]
