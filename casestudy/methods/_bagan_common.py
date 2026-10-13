"""casestudy.methods._bagan_common — shared BAGAN scaffolding (m08/m09).

Provides the pieces both BAGAN variants reuse (130J as algorithmic reference):
- :class:`Encoder` / :class:`AutoEncoder` — AE for the warm-start.
- :func:`pretrain_autoencoder` — MSE reconstruction pre-training.
- :class:`LatentPrior` — per-class **isotropic** Gaussian ``N(μ_c, σ_c² I)``
  fitted on encoded latents (one scalar σ per class — the only statistically
  sound choice for ~80 minority samples in 128-D, per Mariani et al. 2018).
- :func:`transfer_encoder_to_trunk` — copy encoder conv weights into a
  discriminator/critic ``DiscTrunk`` (warm-start), shape-matched by index.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ._backbones import DiscTrunk, Generator, init_weights

print("casestudy/methods/_bagan_common.py loaded")


class Encoder(nn.Module):
    """Image → latent vector (conv trunk + linear projection)."""

    def __init__(self, in_channels: int, latent_dim: int, base: int = 64):
        super().__init__()
        self.trunk = DiscTrunk(in_channels, base, use_bn=True)
        self.fc = nn.Linear(self.trunk.out_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.trunk(x))


class AutoEncoder(nn.Module):
    """Encoder + decoder (the decoder shares the GAN generator architecture)."""

    def __init__(self, in_channels: int, latent_dim: int, base: int = 64):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, base)
        self.decoder = Generator(latent_dim, in_channels, base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def pretrain_autoencoder(
    ae: AutoEncoder,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    batch_size: int,
    device: str,
    lr: float = 1e-3,
) -> AutoEncoder:
    """MSE reconstruction pre-training on the (real) pool."""
    init_weights(ae)
    ae.to(device).train()
    opt = torch.optim.Adam(ae.parameters(), lr=lr, betas=(0.5, 0.999))
    mse = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X),
        batch_size=min(batch_size, len(X)), shuffle=True, drop_last=False,
    )
    for _ in range(epochs):
        for (real,) in loader:
            real = real.to(device)
            opt.zero_grad()
            loss = mse(ae(real), real)
            loss.backward()
            opt.step()
    return ae


class LatentPrior:
    """Per-class isotropic Gaussian over encoded latents."""

    def __init__(self, latent_dim: int, eps: float = 1e-5):
        self.latent_dim = latent_dim
        self.eps = eps
        self.means: Dict[int, torch.Tensor] = {}
        self.stds: Dict[int, float] = {}

    @torch.no_grad()
    def fit(self, encoder: Encoder, X: torch.Tensor, y: torch.Tensor, device: str) -> "LatentPrior":
        encoder.eval()
        for c in torch.unique(y).tolist():
            Xc = X[y == c].to(device)
            z = encoder(Xc)
            mu = z.mean(dim=0)
            var = ((z - mu) ** 2).sum() / (z.numel())  # isotropic scalar variance
            self.means[int(c)] = mu.cpu()
            self.stds[int(c)] = float(var.clamp_min(self.eps).sqrt().item())
        return self

    def sample(self, class_label: int, n: int, device: str) -> torch.Tensor:
        mu = self.means[int(class_label)].to(device)
        std = self.stds[int(class_label)]
        eps = torch.randn(n, self.latent_dim, device=device)
        return mu.unsqueeze(0) + std * eps


def transfer_encoder_to_trunk(encoder: Encoder, trunk: DiscTrunk) -> None:
    """Copy encoder conv weights into a discriminator/critic trunk (warm-start).

    Shapes are identical by construction (both built from ``DiscTrunk``), so we
    copy block-by-block and skip any tensor whose shape differs (e.g. BN vs.
    no-BN critic).
    """
    src = encoder.trunk.state_dict()
    dst = trunk.state_dict()
    for key in dst:
        if key in src and src[key].shape == dst[key].shape:
            dst[key] = src[key].clone()
    trunk.load_state_dict(dst)
