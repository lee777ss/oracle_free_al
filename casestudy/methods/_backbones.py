"""casestudy.methods._backbones — shared 64×64 GAN network primitives.

Private helper module (no public method logic) reused by the GAN methods
m04–m09 so each method module holds only its *training loop*, not a redundant
copy of the generator/discriminator. All networks operate on ``[B, C, 64, 64]``
tensors in ``[-1, 1]`` with ``base_channels=64`` (filters 64/128/256/512 across
the four stride-2 blocks), matching the 130J BAGAN backbone family.
"""

from __future__ import annotations

import torch
import torch.nn as nn

print("casestudy/methods/_backbones.py loaded")


def init_weights(model: nn.Module) -> None:
    """DCGAN-style weight init (Conv/ConvT ~ N(0,0.02), BN ~ N(1,0.02))."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)


class Generator(nn.Module):
    """Latent (+ optional conditioning) vector → ``[B, out_channels, 64, 64]``.

    Architecture (matches BAGAN decoder/generator for weight-transfer parity):
        fc: in_dim → 512·4·4 → reshape (512,4,4)
        4× ConvTranspose2d(stride=2) with BatchNorm+ReLU → tanh, 4→8→16→32→64.

    ``in_dim`` is ``latent_dim`` for unconditional generators or
    ``latent_dim + cond_dim`` when a label embedding is concatenated to ``z``.
    """

    def __init__(self, in_dim: int, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.base_channels = base_channels
        self.fc = nn.Linear(in_dim, base_channels * 8 * 4 * 4)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        h = self.fc(v).view(v.size(0), self.base_channels * 8, 4, 4)
        return self.conv(h)


class DiscTrunk(nn.Module):
    """Image → flattened conv features ``[B, base*8*4*4]`` (64→4 via 4 stride-2).

    Args:
        in_channels: Input channels (image channels, optionally + label planes).
        base_channels: Base filter count (64).
        use_bn: Insert BatchNorm after each conv (DCGAN/BAGAN); disable for
            WGAN-GP critics where the gradient penalty needs per-sample grads.
        dropout: Dropout2d rate applied between conv blocks during training.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64,
                 use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.out_dim = base_channels * 8 * 4 * 4

        def block(cin: int, cout: int) -> list:
            layers: list = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.blocks = nn.ModuleList([
            nn.Sequential(*block(in_channels, base_channels)),
            nn.Sequential(*block(base_channels, base_channels * 2)),
            nn.Sequential(*block(base_channels * 2, base_channels * 4)),
            nn.Sequential(*block(base_channels * 4, base_channels * 8)),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if self.dropout > 0 and self.training and i < len(self.blocks) - 1:
                h = nn.functional.dropout2d(h, p=self.dropout, training=True)
        return h.view(h.size(0), -1)


def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: str | torch.device,
    aux: bool = False,
) -> torch.Tensor:
    """WGAN-GP gradient penalty on interpolates between ``real`` and ``fake``.

    If ``aux`` is True the critic returns ``(score, aux_logits)`` and only the
    scalar score is differentiated.
    """
    bsz = real.size(0)
    eps = torch.rand(bsz, 1, 1, 1, device=device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    out = critic(interp)
    score = out[0] if aux else out
    grads = torch.autograd.grad(
        outputs=score.sum(),
        inputs=interp,
        create_graph=True,
        retain_graph=True,
    )[0]
    grads = grads.view(bsz, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
