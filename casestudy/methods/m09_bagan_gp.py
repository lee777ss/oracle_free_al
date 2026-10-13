"""casestudy.methods.m09_bagan_gp — BAGAN-GP (§5.1; strongest non-AL competitor).

BAGAN warm-start (AE pre-training + per-class isotropic latent prior + weight
transfer) combined with a WGAN-GP objective: a no-BatchNorm critic with a
scalar Wasserstein head **and** an auxiliary class head, trained with a
gradient penalty (λ_gp=10). This is the key contrast for H1/H2. Synthesis
samples each class's latent prior and decodes through the generator.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .base import AugmentationMethod
from ._backbones import DiscTrunk, Generator, gradient_penalty
from ._bagan_common import (
    AutoEncoder, LatentPrior, pretrain_autoencoder, transfer_encoder_to_trunk,
)

print("casestudy/methods/m09_bagan_gp.py loaded")


class _AuxCritic(nn.Module):
    """No-BN critic: image → (Wasserstein score, aux class logits)."""

    def __init__(self, in_channels: int, num_classes: int, base: int = 64):
        super().__init__()
        self.trunk = DiscTrunk(in_channels, base, use_bn=False)
        self.critic_head = nn.Linear(self.trunk.out_dim, 1)
        self.aux_head = nn.Linear(self.trunk.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.critic_head(h).view(-1), self.aux_head(h)


class BAGANGPMethod(AugmentationMethod):
    name = "BAGAN-GP"
    is_generative = True
    lambda_gp = 10.0
    n_critic = 5

    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        self.X_real, self.y_real = X_real, y_real
        self._G: Generator | None = None
        self._prior: LatentPrior | None = None

    def _build_and_train(self) -> None:
        from casestudy.reproducibility import set_seed

        set_seed(self.seed)
        device = self.device
        K = self.num_classes

        # 1) Autoencoder warm-start + 2) per-class isotropic latent prior.
        ae = AutoEncoder(self.channels, self.latent_dim)
        ae = pretrain_autoencoder(
            ae, self.X_real, self.y_real, self.config.gan_epochs,
            self.config.batch_size, device,
        )
        prior = LatentPrior(self.latent_dim).fit(ae.encoder, self.X_real, self.y_real, device)

        # 3) Weight transfer: decoder → generator, encoder → critic trunk.
        G = Generator(self.latent_dim, self.channels).to(device)
        G.load_state_dict(ae.decoder.state_dict())
        C = _AuxCritic(self.channels, K).to(device)
        transfer_encoder_to_trunk(ae.encoder, C.trunk)

        optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
        optC = torch.optim.Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))
        ce = nn.CrossEntropyLoss()

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_real, self.y_real),
            batch_size=min(self.config.batch_size, len(self.X_real)),
            shuffle=True, drop_last=False,
        )
        G.train(); C.train()
        for _ in range(self.config.gan_epochs):
            for step, (real, lbl) in enumerate(loader):
                real, lbl = real.to(device), lbl.to(device)
                bsz = real.size(0)
                cls = torch.randint(0, K, (bsz,), device=device)
                z = torch.cat([prior.sample(int(c), 1, device) for c in cls], dim=0)
                fake = G(z).detach()
                # Critic update: Wasserstein + GP + aux CE on real.
                optC.zero_grad()
                s_real, c_real = C(real)
                s_fake, _ = C(fake)
                gp = gradient_penalty(C, real, fake, device, aux=True)
                loss_c = s_fake.mean() - s_real.mean() + self.lambda_gp * gp + ce(c_real, lbl)
                loss_c.backward(); optC.step()
                # Generator update every n_critic steps: fool critic + aux class.
                if step % self.n_critic == 0:
                    z = torch.cat([prior.sample(int(c), 1, device) for c in cls], dim=0)
                    gen = G(z)
                    optG.zero_grad()
                    s_gen, c_gen = C(gen)
                    loss_g = -s_gen.mean() + ce(c_gen, cls)
                    loss_g.backward(); optG.step()
        G.eval()
        self._G, self._prior = G, prior

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        deficits = self._deficits(target_counts)
        if not deficits:
            return self._empty()
        if self._G is None:
            self._build_and_train()
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for c, n in deficits.items():
            with torch.no_grad():
                z = self._prior.sample(c, n, self.device)
                fake = self._G(z).cpu().clamp(-1.0, 1.0)
            xs.append(fake)
            ys.append(torch.full((n,), c, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)
