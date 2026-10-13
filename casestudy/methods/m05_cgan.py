"""casestudy.methods.m05_cgan — conditional GAN (label-conditioned, §5.1).

A single conditional GAN: the class one-hot is concatenated to the noise for
the generator and broadcast as extra input planes for the discriminator
(label-projection conditioning). BCE adversarial loss; sample per target class.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .base import AugmentationMethod
from ._backbones import DiscTrunk, Generator, init_weights

print("casestudy/methods/m05_cgan.py loaded")


class _CondDiscriminator(nn.Module):
    """Image + label-plane channels → real/fake logit."""

    def __init__(self, in_channels: int, num_classes: int, base: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.trunk = DiscTrunk(in_channels + num_classes, base, use_bn=True)
        self.head = nn.Linear(self.trunk.out_dim, 1)

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        planes = y_onehot.view(y_onehot.size(0), self.num_classes, 1, 1)
        planes = planes.expand(-1, -1, x.size(2), x.size(3))
        return self.head(self.trunk(torch.cat([x, planes], dim=1))).view(-1)


class CGANMethod(AugmentationMethod):
    name = "cGAN"
    is_generative = True

    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        self.X_real, self.y_real = X_real, y_real
        self._G: Generator | None = None

    def _onehot(self, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(labels, self.num_classes).float()

    def _train(self) -> Generator:
        from casestudy.reproducibility import set_seed

        set_seed(self.seed)
        device = self.device
        K = self.num_classes
        G = Generator(self.latent_dim + K, self.channels).to(device)
        D = _CondDiscriminator(self.channels, K).to(device)
        init_weights(G); init_weights(D)
        optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_real, self.y_real),
            batch_size=min(self.config.batch_size, len(self.X_real)),
            shuffle=True, drop_last=False,
        )
        G.train(); D.train()
        for _ in range(self.config.gan_epochs):
            for real, lbl in loader:
                real, lbl = real.to(device), lbl.to(device)
                bsz = real.size(0)
                oh = self._onehot(lbl)
                z = torch.cat([torch.randn(bsz, self.latent_dim, device=device), oh], dim=1)
                fake = G(z)
                optD.zero_grad()
                loss_d = bce(D(real, oh), torch.ones(bsz, device=device)) + \
                    bce(D(fake.detach(), oh), torch.zeros(bsz, device=device))
                loss_d.backward(); optD.step()
                optG.zero_grad()
                loss_g = bce(D(fake, oh), torch.ones(bsz, device=device))
                loss_g.backward(); optG.step()
        G.eval()
        return G

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        deficits = self._deficits(target_counts)
        if not deficits:
            return self._empty()
        if self._G is None:
            self._G = self._train()
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for c, n in deficits.items():
            lbl = torch.full((n,), c, dtype=torch.long, device=self.device)
            oh = self._onehot(lbl)
            with torch.no_grad():
                z = torch.cat([torch.randn(n, self.latent_dim, device=self.device), oh], dim=1)
                fake = self._G(z).cpu().clamp(-1.0, 1.0)
            xs.append(fake)
            ys.append(torch.full((n,), c, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)
