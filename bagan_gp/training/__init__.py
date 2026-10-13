"""bagan_gp.training — re-exports AE trainer + latent generator from BAGAN,
exposes :class:`WGANAuxTrainer` under the unified ``GANTrainer`` name."""

from bagan.training.autoencoder_trainer import AutoencoderTrainer
from bagan.training.latent_space import ClassConditionalLatentGenerator

from .gan_trainer import (
    WGANAuxTrainer,
    GANTrainer,
    gradient_penalty,
    wgan_gp_critic_loss,
)

__all__ = [
    "AutoencoderTrainer",
    "ClassConditionalLatentGenerator",
    "GANTrainer",
    "WGANAuxTrainer",
    "gradient_penalty",
    "wgan_gp_critic_loss",
]
