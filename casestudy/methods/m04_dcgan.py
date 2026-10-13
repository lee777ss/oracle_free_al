"""casestudy.methods.m04_dcgan — DCGAN, one model per minority class (§5.1).

Trains an **unconditional** DCGAN (BCE adversarial loss, no label conditioning)
on each minority class's real samples, then samples to fill that class's
deficit. No active learning, no classifier in the loop.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .base import AugmentationMethod
from ._backbones import DiscTrunk, Generator, init_weights

print("casestudy/methods/m04_dcgan.py loaded")


class _DCDiscriminator(nn.Module):
    def __init__(self, in_channels: int, base: int = 64):
        super().__init__()
        self.trunk = DiscTrunk(in_channels, base, use_bn=True)
        self.head = nn.Linear(self.trunk.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x)).view(-1)


class DCGANMethod(AugmentationMethod):
    name = "DCGAN"
    is_generative = True

    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        self.X_real, self.y_real = X_real, y_real
        self._generators: Dict[int, Generator] = {}

    def _train_one_class(self, X_c: torch.Tensor) -> Generator:
        from casestudy.reproducibility import set_seed

        set_seed(self.seed)
        device = self.device
        G = Generator(self.latent_dim, self.channels).to(device)
        D = _DCDiscriminator(self.channels).to(device)
        init_weights(G)
        init_weights(D)
        optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_c),
            batch_size=min(self.config.batch_size, len(X_c)),
            shuffle=True, drop_last=False,
        )
        G.train(); D.train()
        for _ in range(self.config.gan_epochs):
            for (real,) in loader:
                real = real.to(device)
                bsz = real.size(0)
                z = torch.randn(bsz, self.latent_dim, device=device)
                fake = G(z)
                # D step
                optD.zero_grad()
                loss_d = bce(D(real), torch.ones(bsz, device=device)) + \
                    bce(D(fake.detach()), torch.zeros(bsz, device=device))
                loss_d.backward()
                optD.step()
                # G step
                optG.zero_grad()
                loss_g = bce(D(fake), torch.ones(bsz, device=device))
                loss_g.backward()
                optG.step()
        G.eval()
        return G

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        deficits = self._deficits(target_counts)
        if not deficits:
            return self._empty()
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for c, n in deficits.items():
            X_c = self.X_real[self.y_real == c]
            if c not in self._generators:
                self._generators[c] = self._train_one_class(X_c)
            G = self._generators[c]
            with torch.no_grad():
                z = torch.randn(n, self.latent_dim, device=self.device)
                fake = G(z).cpu().clamp(-1.0, 1.0)
            xs.append(fake)
            ys.append(torch.full((n,), c, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)
