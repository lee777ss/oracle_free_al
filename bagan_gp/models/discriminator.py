"""WGAN-GP critic ("no-BN") — drop-in replacement for ``BAGANDiscriminator``.

Ported from ``Commit B_only.ipynb`` (cells B-only-3). The critic is identical
in convolutional layout to ``bagan.models.discriminator.BAGANDiscriminator``
(four stride-2 ``Conv2d`` blocks, no BatchNorm) so :class:`WeightTransfer`
can copy the autoencoder's encoder conv weights into ``self.features`` slots
by index.

Differences from the legacy ``BAGANDiscriminator``:

* ``forward`` returns a tuple ``(score, aux_logits)`` instead of a single
  ``(n+1)``-logit tensor.
* No BatchNorm anywhere — the gradient penalty requires per-sample gradients
  through the critic, and BN couples them.
* Two heads (Wasserstein scalar + n-way auxiliary classifier) instead of one.

The class is exported as ``BAGANDiscriminator`` (alias) so user code that
imports ``from bagan_gp import BAGANDiscriminator`` is interchangeable with
the BAGAN equivalent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("bagan_gp/models/discriminator.py loaded")


class WGANAuxCriticNoBN(nn.Module):
    """WGAN-GP critic with auxiliary classification head, no BatchNorm.

    Args:
        in_channels: Number of input image channels (1 for FashionMNIST).
        num_classes: Number of real classes (n). Critic does NOT add a "fake"
            output column; the realism signal comes from the scalar critic head.
        base_channels: Base channel count for the feature stack. Filters are
            ``base_channels * {1, 2, 4, 8}`` across the four conv blocks,
            matching the BAGAN encoder so weight transfer works.
        dropout_rate: Per-block Dropout2d rate, applied between conv blocks
            during training only. Default 0.0 (the no-BN critic relies on the
            gradient penalty for regularisation, not dropout).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3,
                 base_channels: int = 64, dropout_rate: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate

        # Convolutional feature stack — same shape as BAGAN's encoder so
        # WeightTransfer can map conv_layers[i].weight → features[2*i].weight.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels,        base_channels,     4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels,      base_channels * 2, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2,  base_channels * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4,  base_channels * 8, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        flat_dim = base_channels * 8 * 4 * 4   # 64×64 → 4×4 after 4 stride-2 conv
        self.critic_head = nn.Linear(flat_dim, 1)
        self.aux_head = nn.Linear(flat_dim, num_classes)

    # ------------------------------------------------------------------ #
    # API parity with bagan.BAGANDiscriminator
    # ------------------------------------------------------------------ #
    @property
    def conv_layers(self) -> nn.Sequential:
        """Alias for ``self.features`` so legacy callers reading
        ``discriminator.conv_layers`` keep working."""
        return self.features

    @property
    def num_discriminator_outputs(self) -> int:
        """The critic has 1 scalar + num_classes aux logits; this property is
        provided only for API parity with BAGANDiscriminator. It returns the
        BAGAN-style ``num_classes + 1`` and should not be used to size losses."""
        return self.num_classes + 1

    def forward(self, x: torch.Tensor):
        """Return ``(score, aux_logits)``.

        * ``score`` — shape ``[B]``, Wasserstein critic scalar.
        * ``aux_logits`` — shape ``[B, num_classes]``, auxiliary class logits
          (use with ``cross_entropy`` against the real-class label).
        """
        h = self.features(x)
        if self.dropout_rate > 0 and self.training:
            h = F.dropout2d(h, p=self.dropout_rate, training=True)
        h = h.view(h.size(0), -1)
        score = self.critic_head(h).view(-1)
        aux_logits = self.aux_head(h)
        return score, aux_logits


# API alias — keeps ``from bagan_gp import BAGANDiscriminator`` working.
BAGANDiscriminator = WGANAuxCriticNoBN
