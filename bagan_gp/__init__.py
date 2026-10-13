"""bagan_gp — WGAN-GP backend (drop-in alternative to the ``bagan`` package).

Mirrors :mod:`bagan`'s public API name-for-name so callers can swap backends
with a single import line. The autoencoder, generator, latent generator,
visualiser, and AE-trainer are reused unchanged from :mod:`bagan`; only the
discriminator (``WGANAuxCriticNoBN``), GAN trainer (``WGANAuxTrainer``),
weight-transfer wrapper, and config (``BAGANGPConfig``) are new.

Public names (identical to :mod:`bagan`):
    BAGANConfig, BAGANEncoder, BAGANDecoder, BAGANAutoencoder,
    BAGANGenerator, BAGANDiscriminator, BAGANDiscriminatorLoss,
    AutoencoderTrainer, ClassConditionalLatentGenerator, GANTrainer,
    WeightTransfer, BAGANVisualizer, show_samples, show_latent_space.
"""

print("bagan_gp/__init__.py loaded")

from .config import BAGANGPConfig, BAGANConfig

from .models import (
    BAGANEncoder,
    BAGANDecoder,
    BAGANAutoencoder,
    BAGANGenerator,
    BAGANDiscriminator,
    BAGANDiscriminatorLoss,
    WGANAuxCriticNoBN,
)

from .training import (
    AutoencoderTrainer,
    ClassConditionalLatentGenerator,
    GANTrainer,
    WGANAuxTrainer,
)

from .utils import (
    WeightTransfer,
    BAGANVisualizer,
    show_samples,
    show_latent_space,
    realism_score,
    adv_class_logits,
    discriminator_features,
)

__all__ = [
    # Configuration
    "BAGANConfig",
    "BAGANGPConfig",
    # Models
    "BAGANEncoder",
    "BAGANDecoder",
    "BAGANAutoencoder",
    "BAGANGenerator",
    "BAGANDiscriminator",
    "BAGANDiscriminatorLoss",
    "WGANAuxCriticNoBN",
    # Training
    "ClassConditionalLatentGenerator",
    "AutoencoderTrainer",
    "GANTrainer",
    "WGANAuxTrainer",
    # Utilities
    "WeightTransfer",
    "BAGANVisualizer",
    "show_samples",
    "show_latent_space",
    # Backend adapter helpers (used by ablation.gan_backend)
    "realism_score",
    "adv_class_logits",
    "discriminator_features",
]
