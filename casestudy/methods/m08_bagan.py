"""casestudy.methods.m08_bagan — BAGAN (AE warm-start + per-class prior, §5.1).

Standalone re-implementation (no ``bagan`` import) of BAGAN (Mariani et al.,
2018): autoencoder pre-training → per-class isotropic latent prior →
weight transfer (decoder→generator, encoder→discriminator trunk) →
(n+1)-class discriminator (real classes 0..n-1 + a "fake" class) trained with
cross-entropy on balanced batches. Synthesis samples each class's latent prior
and decodes through the generator.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .base import AugmentationMethod
from ._backbones import DiscTrunk, Generator
from ._bagan_common import (
    AutoEncoder, LatentPrior, pretrain_autoencoder, transfer_encoder_to_trunk,
)

print("casestudy/methods/m08_bagan.py loaded")


class _NPlus1Discriminator(nn.Module):
    """Image → (num_classes + 1) logits (last column = fake)."""

    def __init__(self, in_channels: int, num_classes: int, base: int = 64):
        super().__init__()
        self.trunk = DiscTrunk(in_channels, base, use_bn=True)
        self.head = nn.Linear(self.trunk.out_dim, num_classes + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


class BAGANMethod(AugmentationMethod):
    name = "BAGAN"
    is_generative = True
    # One-sided label smoothing on the discriminator's real targets (see below).
    # 0.1 reproduces the runs behind the submitted paper; the KBS revision runs
    # BAGAN as published (0.0) and reports the smoothed variant separately.
    label_smoothing = 0.1

    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        self.X_real, self.y_real = X_real, y_real
        self._G: Generator | None = None
        self._prior: LatentPrior | None = None

    def _build_and_train(self) -> None:
        from casestudy.reproducibility import set_seed

        set_seed(self.seed)
        device = self.device
        K = self.num_classes
        fake_label = K

        # 1) Autoencoder warm-start (AE epochs reuse the shared gan budget).
        ae = AutoEncoder(self.channels, self.latent_dim)
        ae = pretrain_autoencoder(
            ae, self.X_real, self.y_real, self.config.gan_epochs,
            self.config.batch_size, device,
        )
        # 2) Per-class isotropic latent prior.
        prior = LatentPrior(self.latent_dim).fit(ae.encoder, self.X_real, self.y_real, device)

        # 3) Weight transfer: decoder → generator, encoder → discriminator trunk.
        G = Generator(self.latent_dim, self.channels).to(device)
        G.load_state_dict(ae.decoder.state_dict())
        D = _NPlus1Discriminator(self.channels, K).to(device)
        transfer_encoder_to_trunk(ae.encoder, D.trunk)

        optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        # One-sided label smoothing on the discriminator's REAL targets
        # (Salimans et al. 2016, "Improved Techniques for Training GANs").
        # Without it the encoder-warm-started (n+1)-class D saturates
        # (loss_d -> ~0.005) and the generator collapses to noise late in
        # training. Smoothing keeps a learning signal for the full budget, so
        # the shared `gan_epochs` schedule is unchanged (no GAN-level
        # early-stop) and the fair-budget protocol still holds.
        ce_real = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
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
                # Sample balanced fake latents across the K classes.
                cls = torch.randint(0, K, (bsz,), device=device)
                z = torch.cat([prior.sample(int(c), 1, device) for c in cls], dim=0)
                fake = G(z)
                # D step: real → true class, fake → fake_label.
                optD.zero_grad()
                loss_d = ce_real(D(real), lbl) + ce(
                    D(fake.detach()), torch.full((bsz,), fake_label, device=device)
                )
                loss_d.backward(); optD.step()
                # G step: push fakes toward their intended real class.
                optG.zero_grad()
                loss_g = ce(D(fake), cls)
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
