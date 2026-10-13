"""casestudy.methods.m06_acgan — ACGAN (auxiliary classifier head, §5.1).

Conditional generator (noise ⊕ class one-hot) with a discriminator that has
two heads: a validity (real/fake) head and an auxiliary class-classification
head. Loss = adversarial BCE + auxiliary CE (Odena et al., 2017).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .base import AugmentationMethod
from ._backbones import DiscTrunk, Generator, init_weights

print("casestudy/methods/m06_acgan.py loaded")


class _ACDiscriminator(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base: int = 64):
        super().__init__()
        self.trunk = DiscTrunk(in_channels, base, use_bn=True)
        self.validity = nn.Linear(self.trunk.out_dim, 1)
        self.aux = nn.Linear(self.trunk.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.validity(h).view(-1), self.aux(h)


class ACGANMethod(AugmentationMethod):
    name = "ACGAN"
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
        D = _ACDiscriminator(self.channels, K).to(device)
        init_weights(G); init_weights(D)
        optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()
        ce = nn.CrossEntropyLoss()

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
                # D step: real validity + aux CE, fake validity + aux CE.
                optD.zero_grad()
                v_real, c_real = D(real)
                v_fake, c_fake = D(fake.detach())
                loss_d = (
                    bce(v_real, torch.ones(bsz, device=device))
                    + bce(v_fake, torch.zeros(bsz, device=device))
                    + ce(c_real, lbl) + ce(c_fake, lbl)
                )
                loss_d.backward(); optD.step()
                # G step: fool validity + correct aux class.
                optG.zero_grad()
                v_gen, c_gen = D(fake)
                loss_g = bce(v_gen, torch.ones(bsz, device=device)) + ce(c_gen, lbl)
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
