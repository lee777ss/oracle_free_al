"""casestudy.methods.m07_wgangp — WGAN-GP, one critic per minority class (§5.1).

Per-class Wasserstein GAN with gradient penalty (λ_gp=10, n_critic=5). The
critic uses no BatchNorm (the gradient penalty needs per-sample gradients).
No active learning, no classifier in the loop.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .base import AugmentationMethod
from ._backbones import DiscTrunk, Generator, gradient_penalty, init_weights

print("casestudy/methods/m07_wgangp.py loaded")


class _Critic(nn.Module):
    def __init__(self, in_channels: int, base: int = 64):
        super().__init__()
        self.trunk = DiscTrunk(in_channels, base, use_bn=False)
        self.head = nn.Linear(self.trunk.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x)).view(-1)


class WGANGPMethod(AugmentationMethod):
    name = "WGAN-GP"
    is_generative = True
    lambda_gp = 10.0
    n_critic = 5

    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        self.X_real, self.y_real = X_real, y_real
        self._generators: Dict[int, Generator] = {}

    def _train_one_class(self, X_c: torch.Tensor) -> Generator:
        from casestudy.reproducibility import set_seed

        set_seed(self.seed)
        device = self.device
        G = Generator(self.latent_dim, self.channels).to(device)
        C = _Critic(self.channels).to(device)
        init_weights(G); init_weights(C)
        optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
        optC = torch.optim.Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_c),
            batch_size=min(self.config.batch_size, len(X_c)),
            shuffle=True, drop_last=False,
        )
        G.train(); C.train()
        for _ in range(self.config.gan_epochs):
            for step, (real,) in enumerate(loader):
                real = real.to(device)
                bsz = real.size(0)
                # Critic update.
                z = torch.randn(bsz, self.latent_dim, device=device)
                fake = G(z).detach()
                optC.zero_grad()
                gp = gradient_penalty(C, real, fake, device, aux=False)
                loss_c = C(fake).mean() - C(real).mean() + self.lambda_gp * gp
                loss_c.backward(); optC.step()
                # Generator update every n_critic steps.
                if step % self.n_critic == 0:
                    z = torch.randn(bsz, self.latent_dim, device=device)
                    optG.zero_grad()
                    loss_g = -C(G(z)).mean()
                    loss_g.backward(); optG.step()
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
