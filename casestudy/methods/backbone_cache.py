"""casestudy.methods.backbone_cache - the shared autoencoder + BAGAN-GP backbone (KBS revision, S3).

The autoencoder, the class-conditional latent prior, the encoded real latents and the
gradient-penalised warm-up are trained once per (regime, seed, training data) and stored.
BAGAN-GP, the proposed method, every selection arm and every sensitivity setting load the
same file, so comparisons that involve the generator are paired on the same generator.

The recipe is the one of ``m09_bagan_gp.BAGANGPMethod._build_and_train`` and
``m10_ab8c.Ab8cMethod._warmup_bagan_gp`` (identical to each other), seeded from the
``backbone`` stream of the run seed. The file name carries a digest of the training data
and of the budget, so a backbone trained for a smoke test can never be loaded by a real run.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from ..locks import release, try_lock
from ..reproducibility import set_seed
from ..seeds import stream_seed
from ._backbones import Generator, gradient_penalty
from ._bagan_common import AutoEncoder, LatentPrior, pretrain_autoencoder, transfer_encoder_to_trunk
from .m09_bagan_gp import _AuxCritic

print("casestudy/methods/backbone_cache.py loaded")

LAMBDA_GP, N_CRITIC, BASE_LR, BASE_BETAS = 10.0, 5, 1e-4, (0.0, 0.9)


def save_root() -> Path:
    return Path(os.environ.get("KBS_SAVE_ROOT", "saved_kbs"))


@dataclass
class Backbone:
    G: Generator
    D: _AuxCritic
    prior: LatentPrior
    real_latents: Dict[int, torch.Tensor]
    timings: Dict[str, float] = field(default_factory=dict)
    trained_here: bool = False
    path: Path | None = None


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tensor_digest(*tensors: torch.Tensor) -> str:
    h = hashlib.sha1()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def backbone_path(regime: str, seed: int, X: torch.Tensor, y: torch.Tensor, config) -> Path:
    budget = f"{config.gan_epochs}|{config.batch_size}|{config.latent_dim}|{LAMBDA_GP}|{N_CRITIC}|{BASE_LR}|{BASE_BETAS}"
    digest = hashlib.sha1((budget + tensor_digest(X, y)).encode()).hexdigest()[:10]
    return save_root() / "backbones" / str(regime) / f"s{int(seed)}_{digest}.pt"


def train_backbone(X: torch.Tensor, y: torch.Tensor, config, channels: int, latent_dim: int,
                   num_classes: int, device: str):
    """Train the backbone from the current RNG state. Returns (G, D, prior, real_latents, timings)."""
    timings: Dict[str, float] = {}
    _sync()
    t0 = time.perf_counter()
    ae = AutoEncoder(channels, latent_dim)
    ae = pretrain_autoencoder(ae, X, y, config.gan_epochs, config.batch_size, device)
    prior = LatentPrior(latent_dim).fit(ae.encoder, X, y, device)
    ae.encoder.eval()
    real_latents: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for c in torch.unique(y).tolist():
            real_latents[int(c)] = ae.encoder(X[y == c].to(device)).detach().cpu()
    _sync()
    timings["ae_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    G = Generator(latent_dim, channels).to(device)
    G.load_state_dict(ae.decoder.state_dict())
    D = _AuxCritic(channels, num_classes).to(device)
    transfer_encoder_to_trunk(ae.encoder, D.trunk)
    optG = torch.optim.Adam(G.parameters(), lr=BASE_LR, betas=BASE_BETAS)
    optD = torch.optim.Adam(D.parameters(), lr=BASE_LR, betas=BASE_BETAS)
    ce = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=min(config.batch_size, len(X)), shuffle=True, drop_last=False,
    )
    G.train(); D.train()
    for _ in range(config.gan_epochs):
        for step, (real, lbl) in enumerate(loader):
            real, lbl = real.to(device), lbl.to(device)
            bsz = real.size(0)
            cls = torch.randint(0, num_classes, (bsz,), device=device)
            z = torch.cat([prior.sample(int(c), 1, device) for c in cls], dim=0)
            fake = G(z).detach()
            optD.zero_grad()
            s_real, c_real = D(real)
            s_fake, _ = D(fake)
            gp = gradient_penalty(D, real, fake, device, aux=True)
            loss_d = s_fake.mean() - s_real.mean() + LAMBDA_GP * gp + ce(c_real, lbl)
            loss_d.backward(); optD.step()
            if step % N_CRITIC == 0:
                z = torch.cat([prior.sample(int(c), 1, device) for c in cls], dim=0)
                gen = G(z)
                optG.zero_grad()
                s_gen, c_gen = D(gen)
                loss_g = -s_gen.mean() + ce(c_gen, cls)
                loss_g.backward(); optG.step()
    _sync()
    timings["backbone_s"] = time.perf_counter() - t1
    return G, D, prior, real_latents, timings


def _load(path: Path, channels: int, latent_dim: int, num_classes: int, device: str) -> Backbone:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    G = Generator(latent_dim, channels).to(device)
    G.load_state_dict(sd["G"])
    D = _AuxCritic(channels, num_classes).to(device)
    D.load_state_dict(sd["D"])
    prior = LatentPrior(latent_dim)
    prior.means = {int(c): m for c, m in sd["prior_means"].items()}
    prior.stds = {int(c): float(s) for c, s in sd["prior_stds"].items()}
    latents = {int(c): z for c, z in sd["real_latents"].items()}
    # Both networks are left in train mode, exactly as the end of the original warm-up leaves them.
    G.train(); D.train()
    return Backbone(G, D, prior, latents, dict(sd["timings"]), trained_here=False, path=path)


def get_backbone(regime: str, seed: int, X: torch.Tensor, y: torch.Tensor, config, channels: int,
                 latent_dim: int, num_classes: int, device: str, poll_s: float = 15.0) -> Backbone:
    """Load the backbone for (regime, seed, X, y), training it first if no process has.

    If another process is training the same backbone, wait for it rather than train twice.
    """
    path = backbone_path(regime, seed, X, y, config)
    lock = path.with_suffix(".lock")
    while True:
        if path.exists():
            return _load(path, channels, latent_dim, num_classes, device)
        if try_lock(lock, kind="backbone", regime=regime, seed=int(seed)):
            try:
                if path.exists():
                    return _load(path, channels, latent_dim, num_classes, device)
                set_seed(stream_seed(seed, "backbone"))
                G, D, prior, latents, timings = train_backbone(X, y, config, channels, latent_dim, num_classes, device)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                torch.save({
                    "G": G.state_dict(), "D": D.state_dict(),
                    "prior_means": {int(c): m.cpu() for c, m in prior.means.items()},
                    "prior_stds": {int(c): float(s) for c, s in prior.stds.items()},
                    "real_latents": latents, "timings": timings,
                    "meta": {"regime": str(regime), "seed": int(seed), "n_train": int(len(y)),
                             "gan_epochs": int(config.gan_epochs)},
                }, tmp)
                os.replace(tmp, path)
                return Backbone(G, D, prior, latents, timings, trained_here=True, path=path)
            finally:
                release(lock)
        time.sleep(poll_s)
