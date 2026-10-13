"""WGAN-GP auxiliary trainer — drop-in replacement for ``bagan.GANTrainer``.

Implements the training loop from ``Commit B_only.ipynb``:

* Critic update: minimise ``E[D(fake)] - E[D(real)] + λ·GP + α·CE(aux_real, y_real)``.
* Generator update (once per ``n_critic`` critic steps): minimise
  ``-E[D(fake)] + β·CE(aux_fake, y_fake)``.
* Class-conditional latent sampling via :class:`ClassConditionalLatentGenerator`.
* Balanced batch composition (equal samples per class within each generator batch).

The trainer is wrapped under the alias ``GANTrainer`` so the notebook's
``from bagan_gp import GANTrainer`` works identically to BAGAN's import.

Deterministic-algorithms note (Q-E):
The gradient penalty calls ``torch.autograd.grad`` through Conv2d operations
that lack deterministic CUDA implementations. ``fit()`` temporarily disables
``torch.use_deterministic_algorithms`` for its body and restores the previous
value in a ``finally`` block.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

print("bagan_gp/training/gan_trainer.py loaded")


# ────────────────────────────────────────────────────────────────────────────
# Public helpers — extracted from ``WGANAuxTrainer._critic_step`` so that
# downstream code (e.g. Ab-8c 3-player co-training in ``ablation.trainer``)
# can drive its own WGAN-GP critic update without re-implementing the loss
# formula.  These helpers are intentionally minimal — no optimizer state,
# no module state mutation — the caller is responsible for ``zero_grad``,
# ``backward``, ``step`` and any ``train()/eval()`` toggling on ``D``.
# ────────────────────────────────────────────────────────────────────────────


def gradient_penalty(
    D: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Standard WGAN-GP one-sided gradient penalty on real/fake interpolates.

    ``D`` must return ``(score, aux_logits)``; only the critic score is used.
    Builds the autograd graph so the result is differentiable wrt D's params.
    """
    if device is None:
        device = real.device
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=device)
    interp = (eps * real + (1.0 - eps) * fake).detach().requires_grad_(True)
    score, _ = D(interp)
    grads = torch.autograd.grad(
        outputs=score.sum(),
        inputs=interp,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(B, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def wgan_gp_critic_loss(
    D: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    real_y: torch.Tensor,
    *,
    gp_lambda: float = 10.0,
    aux_w: float = 1.0,
    extra_x: Optional[torch.Tensor] = None,
    extra_y: Optional[torch.Tensor] = None,
    extra_aux_w: float = 0.0,
    device: Optional[torch.device] = None,
):
    """Compute WGAN-GP critic loss with an aux-classification term.

    loss = (E[D(fake)] - E[D(real)])
           + gp_lambda * GP(real, fake)
           + aux_w * CE(D.aux(real), real_y)
           + extra_aux_w * CE(D.aux(extra_x), extra_y)   # optional

    Matches ``WGANAuxTrainer._critic_step`` exactly when ``extra_x`` is None.
    The optional ``extra_x``/``extra_y`` term lets Ab-8c teach the aux head
    on accepted synthetic samples in addition to real ones.

    Returns
    -------
    (loss, info) : ``info`` is a dict of detached scalar diagnostics.
    The caller is responsible for ``optimizer.zero_grad()``, ``loss.backward()``,
    and ``optimizer.step()``.
    """
    if device is None:
        device = real.device
    s_real, aux_real = D(real)
    s_fake, _aux_fake = D(fake)
    wd = s_fake.mean() - s_real.mean()
    gp = gradient_penalty(D, real, fake, device=device)
    aux = F.cross_entropy(aux_real, real_y)
    loss = wd + gp_lambda * gp + aux_w * aux
    extra_ce_val = 0.0
    if (
        extra_x is not None
        and extra_y is not None
        and extra_aux_w > 0.0
        and extra_x.size(0) > 0
    ):
        _, aux_extra = D(extra_x)
        ce_extra = F.cross_entropy(aux_extra, extra_y)
        loss = loss + extra_aux_w * ce_extra
        extra_ce_val = float(ce_extra.detach().item())
    info = {
        "loss": float(loss.detach().item()),
        "wd": float(wd.detach().item()),
        "gp": float(gp.detach().item()),
        "aux_real": float(aux.detach().item()),
        "aux_extra": extra_ce_val,
    }
    return loss, info


class WGANAuxTrainer:
    """WGAN-GP with auxiliary classifier (ACGAN-style aux head)."""

    def __init__(self, config, generator, discriminator, latent_generator, train_loader: DataLoader):
        """Mirrors :class:`bagan.GANTrainer.__init__` signature exactly.

        Args:
            config: A :class:`BAGANGPConfig` (or anything exposing the WGAN-GP
                knobs ``n_critic``, ``gp_lambda``, ``aux_weight_d``,
                ``aux_weight_g``, ``adam_betas``, ``lr_g``, ``lr_d``,
                ``gan_epochs``, ``num_classes``, ``device``).
            generator: :class:`BAGANGenerator`. Already moved to device & with
                AE-transferred weights from :meth:`WeightTransfer.initialize_gan_from_autoencoder`.
            discriminator: :class:`WGANAuxCriticNoBN` (aliased as
                ``BAGANDiscriminator``).
            latent_generator: Fitted :class:`ClassConditionalLatentGenerator`.
            train_loader: Yields ``(images, labels)`` with images in ``[-1, 1]``.
        """
        self.config = config
        self.device = config.device

        self.G = generator.to(self.device)
        self.D = discriminator.to(self.device)
        self.latent_gen = latent_generator
        self.loader = train_loader

        # WGAN-GP knobs.
        self.n_critic: int = getattr(config, "n_critic", 5)
        self.gp_lambda: float = getattr(config, "gp_lambda", 10.0)
        self.aux_w_d: float = getattr(config, "aux_weight_d", 1.0)
        self.aux_w_g: float = getattr(config, "aux_weight_g", 1.0)
        betas = getattr(config, "adam_betas", (0.0, 0.9))

        self.optG = torch.optim.Adam(self.G.parameters(), lr=config.lr_g, betas=betas)
        self.optD = torch.optim.Adam(self.D.parameters(), lr=config.lr_d, betas=betas)

        # ── Minority-FID fix #1: class-balanced real batches for the critic ──
        # When enabled, every critic step sees a ~33:33:33 real batch instead of
        # the raw 800:40:40 imbalance, so the global Wasserstein term, gradient
        # penalty, and aux-CE stop concentrating on the majority manifold.
        self.balanced_critic: bool = bool(getattr(config, "balanced_critic", False))
        self._real_pool: Dict[int, torch.Tensor] = {}
        if self.balanced_critic:
            self._build_real_pool()

        # History schema matches BAGAN's GANTrainer for plot compatibility.
        # d_acc_fake is filled with 0.0 (WGAN-GP has no "fake" class column).
        self.history: Dict[str, List[float]] = {
            "d_loss": [], "g_loss": [],
            "d_loss_real": [], "d_loss_fake": [],
            "d_acc_real": [], "d_acc_fake": [],
            "gp": [], "wd": [],
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _sample_balanced_latent(self, batch_size: int):
        """Balanced class-conditional latent batch (equal counts per class,
        with the remainder allocated to the last class to make the total
        exactly ``batch_size``).
        """
        n_classes = self.config.num_classes
        per_class = batch_size // n_classes
        zs, ys = [], []
        for c in range(n_classes):
            n = per_class if c < n_classes - 1 else batch_size - per_class * (n_classes - 1)
            if n <= 0:
                continue
            z = self.latent_gen.sample(class_label=c, batch_size=n)
            zs.append(z.to(self.device))
            ys.append(torch.full((n,), c, dtype=torch.long, device=self.device))
        return torch.cat(zs, dim=0), torch.cat(ys, dim=0)

    def _build_real_pool(self) -> None:
        """Index the training set by class once, so the critic can draw
        class-balanced real batches (minority-FID fix #1).

        Stores per-class real-image tensors on CPU (moved to device per batch)
        to avoid holding a second full copy of the dataset in GPU memory.
        """
        dataset = self.loader.dataset
        n_classes = self.config.num_classes
        buckets: Dict[int, List[torch.Tensor]] = {c: [] for c in range(n_classes)}
        for idx in range(len(dataset)):
            img, lbl = dataset[idx]
            c = int(lbl)
            if c in buckets:
                buckets[c].append(img.unsqueeze(0))
        for c in range(n_classes):
            if buckets[c]:
                self._real_pool[c] = torch.cat(buckets[c], dim=0)  # [N_c, 1, 64, 64] on CPU
        present = {c: int(v.size(0)) for c, v in self._real_pool.items()}
        print(f"  [balanced_critic] real pool per class: {present}")

    def _sample_balanced_real(self, batch_size: int):
        """Draw a class-balanced real batch (equal counts per class, remainder
        to the last class) by sampling with replacement from ``_real_pool``.

        Returns ``(real, real_y)`` already moved to the training device.
        """
        n_classes = self.config.num_classes
        per_class = batch_size // n_classes
        xs, ys = [], []
        for c in range(n_classes):
            n = per_class if c < n_classes - 1 else batch_size - per_class * (n_classes - 1)
            if n <= 0 or c not in self._real_pool:
                continue
            pool = self._real_pool[c]
            idx = torch.randint(0, pool.size(0), (n,))   # with replacement (minorities are tiny)
            xs.append(pool[idx])
            ys.append(torch.full((n,), c, dtype=torch.long))
        real = torch.cat(xs, dim=0).to(self.device, non_blocking=True)
        real_y = torch.cat(ys, dim=0).to(self.device, non_blocking=True)
        return real, real_y

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Standard WGAN-GP penalty: ``E[(‖∇ D(x̂)‖₂ − 1)²]`` on an
        interpolated batch ``x̂ = ε·real + (1−ε)·fake``."""
        B = real.size(0)
        eps = torch.rand(B, 1, 1, 1, device=self.device)
        interp = (eps * real + (1.0 - eps) * fake).detach().requires_grad_(True)
        score, _ = self.D(interp)
        grads = torch.autograd.grad(
            outputs=score.sum(), inputs=interp,
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        grads = grads.view(B, -1)
        gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
        return gp

    # ------------------------------------------------------------------ #
    # Per-step updates
    # ------------------------------------------------------------------ #
    def _critic_step(self, real: torch.Tensor, real_y: torch.Tensor) -> Dict[str, float]:
        B = real.size(0)
        z, _ = self._sample_balanced_latent(B)
        with torch.no_grad():
            fake = self.G(z)

        s_real, aux_real = self.D(real)
        s_fake, aux_fake = self.D(fake)

        wd = s_fake.mean() - s_real.mean()                 # Wasserstein loss
        gp = self._gradient_penalty(real, fake)
        aux = F.cross_entropy(aux_real, real_y)            # aux on REAL only
        loss = wd + self.gp_lambda * gp + self.aux_w_d * aux

        self.optD.zero_grad(set_to_none=True)
        loss.backward()
        self.optD.step()

        with torch.no_grad():
            acc_real = (aux_real.argmax(dim=1) == real_y).float().mean().item()

        return {
            "d_loss": float(loss.item()),
            "d_loss_real": float(-s_real.mean().item()),
            "d_loss_fake": float(s_fake.mean().item()),
            "wd": float(wd.item()),
            "gp": float(gp.item()),
            "d_acc_real": acc_real,
        }

    def _generator_step(self, batch_size: int) -> Dict[str, float]:
        z, fake_y = self._sample_balanced_latent(batch_size)
        fake = self.G(z)
        s_fake, aux_fake = self.D(fake)
        adv = -s_fake.mean()
        aux = F.cross_entropy(aux_fake, fake_y)
        loss = adv + self.aux_w_g * aux

        self.optG.zero_grad(set_to_none=True)
        loss.backward()
        self.optG.step()

        return {"g_loss": float(loss.item())}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, epochs: Optional[int] = None, save_every: int = 10,
            verbose: bool = True) -> Dict[str, List[float]]:
        """Run the WGAN-GP training loop.

        Returns the ``self.history`` dict (same keys as BAGAN's GANTrainer
        plus ``'wd'`` and ``'gp'``).

        Q-E: deterministic algorithms are turned off for the body of this
        method to allow the gradient-penalty autograd path through CUDA
        ``conv2d`` kernels; the previous setting is restored on exit.
        """
        epochs = int(epochs if epochs is not None else self.config.gan_epochs)

        prev_det = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        try:
            for epoch in range(epochs):
                ep_d = ep_g = ep_dr = ep_df = ep_ar = ep_gp = ep_wd = 0.0
                n_d = n_g = 0
                critic_buffer = 0

                self.G.train(); self.D.train()
                for real, real_y in self.loader:
                    if self.balanced_critic:
                        # Replace the imbalanced loader batch with a class-balanced
                        # real draw (same size); loader still drives epoch length.
                        real, real_y = self._sample_balanced_real(real.size(0))
                    else:
                        real = real.to(self.device, non_blocking=True)
                        real_y = real_y.to(self.device, non_blocking=True)

                    # --- Critic update ---
                    s = self._critic_step(real, real_y)
                    ep_d += s["d_loss"]; ep_dr += s["d_loss_real"]; ep_df += s["d_loss_fake"]
                    ep_ar += s["d_acc_real"]; ep_gp += s["gp"]; ep_wd += s["wd"]
                    n_d += 1
                    critic_buffer += 1

                    # --- Generator update once per n_critic critic steps ---
                    if critic_buffer >= self.n_critic:
                        gs = self._generator_step(real.size(0))
                        ep_g += gs["g_loss"]; n_g += 1
                        critic_buffer = 0

                # ---- record averages ----
                self.history["d_loss"].append(ep_d / max(n_d, 1))
                self.history["g_loss"].append(ep_g / max(n_g, 1) if n_g > 0 else 0.0)
                self.history["d_loss_real"].append(ep_dr / max(n_d, 1))
                self.history["d_loss_fake"].append(ep_df / max(n_d, 1))
                self.history["d_acc_real"].append(ep_ar / max(n_d, 1))
                # WGAN-GP critic has no "fake-class" head; pad for schema parity.
                self.history["d_acc_fake"].append(0.0)
                self.history["gp"].append(ep_gp / max(n_d, 1))
                self.history["wd"].append(ep_wd / max(n_d, 1))

                if verbose:
                    print(f"  [E{epoch + 1:>3}/{epochs}] "
                          f"D={self.history['d_loss'][-1]:+.4f}  "
                          f"G={self.history['g_loss'][-1]:+.4f}  "
                          f"WD={self.history['wd'][-1]:+.4f}  "
                          f"GP={self.history['gp'][-1]:.4f}  "
                          f"aux_real={self.history['d_acc_real'][-1]:.1%}")

            return self.history
        finally:
            torch.use_deterministic_algorithms(prev_det)

    # ------------------------------------------------------------------ #
    # API parity with bagan.GANTrainer (no-op stubs)
    # ------------------------------------------------------------------ #
    def generate_class_samples(self, target_class: int, n_samples: int) -> torch.Tensor:
        """Sample ``n_samples`` synthetic images for ``target_class``."""
        self.G.eval()
        with torch.no_grad():
            z = self.latent_gen.sample(class_label=target_class, batch_size=n_samples).to(self.device)
            return self.G(z)

    # Deprecated alias kept for API parity.
    generate_minority_samples = generate_class_samples


# Public alias — ``from bagan_gp import GANTrainer`` works like BAGAN's import.
GANTrainer = WGANAuxTrainer
