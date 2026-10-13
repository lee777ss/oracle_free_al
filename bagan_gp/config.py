"""
BAGAN-GP Configuration (WGAN-GP backend).

Mirrors ``bagan.config.BAGANConfig`` field-for-field (Q3 — duplicate, not
subclass) and adds the WGAN-GP-specific knobs that ``WGANAuxTrainer`` reads.

The ``model_prefix`` defaults to ``'bagan_gp_'`` (Q4 — separate checkpoints),
so AE/GAN artefacts written by this backend never collide with vanilla BAGAN's
``'bagan_'`` files on disk.

Knob defaults reproduce ``Commit B_only.ipynb`` (Q5 — confirmed):
    n_critic=5, gp_lambda=10.0, aux_weight_d=aux_weight_g=1.0,
    lr_g=lr_d=1e-4, adam_betas=(0.0, 0.9).

`BAGANConfig` is exported as an alias of ``BAGANGPConfig`` so user code that
imports ``from bagan_gp import BAGANConfig`` works identically to
``from bagan import BAGANConfig``.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple
from pathlib import Path
import torch

print("bagan_gp/config.py loaded")


@dataclass
class BAGANGPConfig:
    """WGAN-GP variant of :class:`bagan.BAGANConfig`.

    See module docstring for the WGAN-GP-specific fields. All other fields are
    duplicated from ``BAGANConfig`` so this class is drop-in compatible with
    every consumer of the vanilla BAGAN config.
    """

    # --- Image parameters (duplicated from BAGANConfig) ---
    img_size: int = 64
    img_channels: int = 1

    # --- Model architecture (duplicated) ---
    latent_dim: int = 128
    num_classes: int = 2
    encoder_channels: Tuple[int, ...] = (64, 128, 256, 512)
    decoder_channels: Tuple[int, ...] = (512, 256, 128, 64)

    # --- Training hyperparameters (duplicated) ---
    batch_size: int = 64

    # Autoencoder (same as BAGAN — AE is identical between backends)
    ae_epochs: int = 30
    lr_ae: float = 0.0002

    # GAN — defaults overridden by Commit-B-only Q5 confirmation.
    gan_epochs: int = 200
    lr_g: float = 1e-4   # Q5: 1e-4 (WGAN-GP standard, was 2e-4 for BAGAN)
    lr_d: float = 1e-4   # Q5: 1e-4 (was 1e-4 for BAGAN already)

    # Adam betas — WGAN-GP standard is (0, 0.9).
    beta1: float = 0.0   # Q5: 0.0 (WGAN-GP), was 0.5 in BAGAN
    beta2: float = 0.9   # Q5: 0.9 (WGAN-GP), was 0.999 in BAGAN

    # Training-strategy fields retained for API compatibility. WGAN-GP uses
    # ``n_critic`` instead of ``d_steps_per_g_step``/``g_steps_per_d_step``,
    # but downstream code may still inspect these so we keep matching names.
    d_steps_per_g_step: int = 1
    g_steps_per_d_step: int = 1
    label_smoothing: float = 0.0    # not used by WGAN-GP
    d_dropout_rate: float = 0.0     # WGAN-GP critic uses 0 dropout by default

    # --- Latent space (duplicated) ---
    latent_regularization: float = 1e-5

    # --- Paths (duplicated, but with bagan_gp_ prefix) ---
    save_dir: str = "saved"
    model_prefix: str = "bagan_gp_"     # Q4: separate checkpoints per backend
    date_suffix: str = "110J"

    # --- Training control flags (duplicated) ---
    train_ae_from_scratch: bool = True
    train_gan_from_scratch: bool = True
    save_checkpoints: bool = False

    # =========================================================================
    # WGAN-GP-specific knobs (Q5 defaults confirmed)
    # =========================================================================
    n_critic: int = 5
    gp_lambda: float = 10.0
    aux_weight_d: float = 1.0
    aux_weight_g: float = 1.0
    adam_betas: Tuple[float, float] = (0.0, 0.9)

    # =========================================================================
    # Minority-FID improvements (default ON for the WGAN-GP backend)
    # =========================================================================
    # Root cause of "WGAN-GP improves only the majority": every class signal in
    # the training loop is dominated by the 800:40:40 imbalance, and the
    # minority latent prior is estimated from only ~40 samples.
    #
    # 1) balanced_critic — feed the critic class-balanced (33:33:33) real
    #    batches so the global Wasserstein term + gradient penalty + aux-CE no
    #    longer concentrate on the majority manifold.
    # 2) latent_cov_type — replace the brittle single-scalar isotropic σ_c with
    #    a Ledoit–Wolf-style *shrinkage* covariance (per-dimension variance
    #    shrunk toward the isotropic mean), giving a usable minority prior even
    #    with few samples. 'isotropic' reproduces the old behaviour.
    # 3) latent_shrinkage — λ ∈ [0, 1]: 1.0 = isotropic, 0.0 = full diagonal.
    balanced_critic: bool = True
    latent_cov_type: str = "shrinkage"   # 'isotropic' | 'shrinkage' | 'diagonal'
    latent_shrinkage: float = 0.5

    # --- Computed properties (duplicated) ---
    @property
    def num_discriminator_outputs(self) -> int:
        """For API parity with BAGANConfig. The WGAN-GP critic has two heads
        (scalar score + n-class aux); this property returns ``num_classes + 1``
        only so legacy assertions checking the BAGAN shape do not break."""
        return self.num_classes + 1

    @property
    def autoencoder_path(self) -> Path:
        return Path(f"{self.save_dir}/{self.model_prefix}autoencoder_{self.date_suffix}.pth")

    @property
    def latent_params_path(self) -> Path:
        return Path(f"{self.save_dir}/{self.model_prefix}latent_params_{self.date_suffix}.pth")

    @property
    def generator_path(self) -> Path:
        return Path(f"{self.save_dir}/{self.model_prefix}generator_{self.date_suffix}.pth")

    @property
    def discriminator_path(self) -> Path:
        # The WGAN-GP critic is conceptually a "discriminator" from the
        # ablation harness's POV; named consistently for path interoperability.
        return Path(f"{self.save_dir}/{self.model_prefix}critic_{self.date_suffix}.pth")

    @property
    def visualization_dir(self) -> Path:
        return Path(f"{self.save_dir}/visualizations_{self.date_suffix}")

    # --- Device ---
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        assert self.num_classes >= 2, \
            f"num_classes must be >= 2, got {self.num_classes}"
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.img_size in [32, 64, 128], "img_size must be 32, 64, or 128"
        assert self.n_critic >= 1, "n_critic must be >= 1"
        assert self.gp_lambda >= 0, "gp_lambda must be >= 0"
        assert self.latent_cov_type in ("isotropic", "shrinkage", "diagonal"), \
            f"latent_cov_type must be isotropic|shrinkage|diagonal, got {self.latent_cov_type!r}"
        assert 0.0 <= self.latent_shrinkage <= 1.0, \
            f"latent_shrinkage must be in [0, 1], got {self.latent_shrinkage}"

    # --- Factory methods (duplicated API) ---
    @classmethod
    def from_ablation_config(cls, ablation_config) -> "BAGANGPConfig":
        """Mirror of :meth:`BAGANConfig.from_ablation_config` — syncs shared
        fields from an :class:`AblationConfig`."""
        return cls(
            num_classes=ablation_config.num_classes,
            latent_dim=ablation_config.latent_dim,
            save_dir=ablation_config.bagan_model_dir,
            date_suffix=ablation_config.date_suffix,
            device=ablation_config.get_device(),
        )

    def for_cv(
        self,
        ae_epochs: int = 50,
        gan_epochs: Optional[int] = None,
    ) -> "BAGANGPConfig":
        """Reduced-epoch copy for per-fold CV training.

        WGAN-GP knob: the generator is updated once per ``n_critic`` critic
        steps, so to match vanilla BAGAN's per-fold effective G-epochs we
        default ``gan_epochs`` to ``300 * n_critic`` (Commit-B-equivalent
        budget; with ``n_critic=5`` this is 1500 outer epochs = 300 effective
        generator epochs). Callers may override explicitly.
        """
        if gan_epochs is None:
            gan_epochs = 300 * self.n_critic    # n_critic=5 -> 1500
        return replace(self, ae_epochs=ae_epochs, gan_epochs=gan_epochs)


# Alias so ``from bagan_gp import BAGANConfig`` works identically to the
# vanilla import. The two backends expose the same public name; only the
# underlying class differs.
BAGANConfig = BAGANGPConfig
