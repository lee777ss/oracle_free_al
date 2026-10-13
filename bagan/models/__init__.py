"""
BAGAN Model Components

This module contains the neural network architectures for BAGAN:
- BAGANEncoder: Encodes images to latent space
- BAGANDecoder: Decodes latent vectors to images
- BAGANAutoencoder: Combines Encoder + Decoder for pre-training
- BAGANGenerator: Same architecture as Decoder (for GAN phase)
- BAGANDiscriminator: Encoder + (n+1) classification head
- Loss functions: BAGANDiscriminatorLoss, BAGANGeneratorLoss
"""

from .autoencoder import BAGANEncoder, BAGANDecoder, BAGANAutoencoder
from .generator import BAGANGenerator, create_generator_from_config
from .discriminator import (
    BAGANDiscriminator, 
    create_discriminator_from_config,
    BAGANDiscriminatorLoss,
    BAGANGeneratorLoss
)


# __all__ is a special Python variable that defines which names are exported when someone uses from module import *.
__all__ = [
    "BAGANEncoder",
    "BAGANDecoder", 
    "BAGANAutoencoder",
    "BAGANGenerator",
    "BAGANDiscriminator",
    "BAGANDiscriminatorLoss",
    "BAGANGeneratorLoss",
    "create_generator_from_config",
    "create_discriminator_from_config",
]
