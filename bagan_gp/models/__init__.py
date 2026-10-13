"""bagan_gp.models — public model API.

Re-exports vanilla BAGAN's :class:`BAGANEncoder`, :class:`BAGANDecoder`,
:class:`BAGANAutoencoder` and :class:`BAGANGenerator` unchanged (they are
identical between backends — the autoencoder is MSE-trained the same way,
and the generator architecture is reused), and provides the WGAN-GP critic
under the name :class:`BAGANDiscriminator`.
"""

from bagan.models.autoencoder import BAGANEncoder, BAGANDecoder, BAGANAutoencoder
from bagan.models.generator import BAGANGenerator

from .discriminator import WGANAuxCriticNoBN, BAGANDiscriminator

# ``BAGANDiscriminatorLoss`` exists in vanilla BAGAN but has no WGAN-GP
# analogue (loss is computed inline in :class:`WGANAuxTrainer`). Re-export
# the BAGAN version solely for import compatibility — it is unused under
# the WGAN-GP backend.
from bagan.models.discriminator import BAGANDiscriminatorLoss

__all__ = [
    "BAGANEncoder",
    "BAGANDecoder",
    "BAGANAutoencoder",
    "BAGANGenerator",
    "BAGANDiscriminator",
    "BAGANDiscriminatorLoss",
    "WGANAuxCriticNoBN",
]
