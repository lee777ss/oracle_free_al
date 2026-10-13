"""
BAGAN Configuration (N-Class)

Hyperparameters and settings for BAGAN training.
Supports arbitrary N-class classification (not limited to binary).

Architecture Overview:
    BAGAN uses an (N+1)-class discriminator:
        - N real classes (class_0, class_1, ..., class_{N-1})
        - 1 fake class (index = N)
    
    The generator is unconditional — class conditioning is achieved via
    class-specific latent distributions (ClassConditionalLatentGenerator).

History:
    427J: Binary (2-class) hardcoded defaults
    511J: Generalized to N-class with AblationConfig bridge
"""

from dataclasses import dataclass, field, replace
from typing import Tuple, Optional
from pathlib import Path
import torch


# @dataclass decorator automatically generates:
#   - __init__(): Constructor from class attributes
#   - __repr__(): String representation for debugging
#   - __eq__(): Equality comparison
# Instead of writing: def __init__(self, image_size=64, channels=1, ...): self.image_size = image_size; ...
# We just declare class attributes with type hints and default values
@dataclass
class BAGANConfig:
    """
    Configuration for BAGAN training pipeline.
    
    Supports N-class classification. The discriminator uses (N+1) outputs:
    N real class logits + 1 fake class logit. The number of discriminator
    outputs is computed automatically as a @property (num_classes + 1),
    so it always stays in sync with num_classes.
    
    Usage:
        # Direct construction (specify num_classes explicitly):
        config = BAGANConfig(num_classes=3)
        
        # From AblationConfig (recommended — syncs shared fields):
        config = BAGANConfig.from_ablation_config(ablation_config)
        
        # Access discriminator output count (auto-computed):
        config.num_discriminator_outputs  # Returns 4 when num_classes=3
    """
    
    # ==========================================================================
    # Image Parameters
    # ==========================================================================
    img_size: int = 64 # I can directly access this as config.img_size due to @dataclass
    img_channels: int = 1  # Grayscale for FashionMNIST
    
    # ==========================================================================
    # Model Architecture
    # ==========================================================================
    latent_dim: int = 128
    num_classes: int = 2  # Number of real classes (N). Discriminator will have N+1 outputs.
    
    # --- num_discriminator_outputs ---
    # Previously (427J): was a separate dataclass field `num_discriminator_outputs: int = 3`
    # Problem: Having it as a mutable field risked going out of sync with num_classes.
    #   e.g., BAGANConfig(num_classes=3, num_discriminator_outputs=3) would be WRONG
    #   (should be 4 = 3+1), and the error would only be caught at __post_init__.
    #
    # Now (511J): Replaced with a @property that COMPUTES it automatically.
    # This makes it impossible to get out of sync — it always returns num_classes + 1.
    # @property is accessed same as a field: config.num_discriminator_outputs (no parens)
    #
    # All downstream code (discriminator, gan_trainer, etc.) continues to read
    # config.num_discriminator_outputs identically — no changes needed there.
    
    # Encoder/Decoder channel progression
    # These define the number of filters at each conv/deconv layer:
    # Encoder: 1→64→128→256→512 (progressively more feature maps, smaller spatial)
    # Decoder: 512→256→128→64→1 (mirror of encoder)
    # why do we need to define channel progressions in config?
    # - Centralizes architecture hyperparameters in one place
    # - Makes it easy to modify architecture without changing model code
    # where is this used?
    # - BAGANEncoder and BAGANDecoder will read these channel configurations
    encoder_channels: Tuple[int, ...] = (64, 128, 256, 512)
    decoder_channels: Tuple[int, ...] = (512, 256, 128, 64)
    
    # ==========================================================================
    # Training Hyperparameters (shared where possible)
    # ==========================================================================
    batch_size: int = 64  # Used for both AE and GAN training
    
    # Autoencoder Training
    ae_epochs: int = 30
    lr_ae: float = 0.0002  # Autoencoder learning rate
    
    # GAN Training  
    gan_epochs: int = 200
    lr_g: float = 0.0002  # Generator learning rate
    lr_d: float = 0.0001  # Discriminator learning rate (matched to G; 0.0001 caused D starvation)
    
    # Adam optimizer parameters (shared)
    beta1: float = 0.5  # Adam beta1 (standard for GANs)
    beta2: float = 0.999  # Adam beta2
    
    # Training strategy
    d_steps_per_g_step: int = 1  # Discriminator steps per generator step
    g_steps_per_d_step: int = 5  # Generator steps per discriminator step (5 caused G to overwhelm D → collapse)
    label_smoothing: float = 0.2  # Smooth D targets: real→(1-ε), fake→ε (0.2 too aggressive for 4-class head)
    d_dropout_rate: float = 0.3  # Dropout2d rate for discriminator conv blocks (0 = disabled)
    
    # GAN training stability
    # (Collapse detection removed — was triggering premature stops.
    #  Cosine LR scheduler in GANTrainer.fit() handles convergence instead.)

    # Per-class FID best-checkpoint monitoring (opt-in; 0 preserves old behavior
    # of always using the final-epoch generator). When > 0, GANTrainer.fit()
    # evaluates ablation.fid.fid_per_class every N epochs and snapshots the
    # generator whenever mean FID improves, so training instability late in a
    # run (e.g. D overpowering G on a well-represented class) doesn't silently
    # ship a worse-than-earlier generator. Requires GANTrainer to be
    # constructed with `real_images_by_class` (used to build/reuse the
    # InceptionV3 real-reference cache).
    fid_monitor_every: int = 0
    fid_monitor_samples: int = 500  # fake samples/class for the monitor (cheaper than the final N_FID_SAMPLES eval)

    # ==========================================================================
    # Latent Space
    # ==========================================================================
    latent_regularization: float = 1e-5  # For covariance matrix stability
    
    # ==========================================================================
    # Paths
    # ==========================================================================
    save_dir: str = "saved"
    model_prefix: str = "bagan_"
    date_suffix: str = "512J"  # Experiment identifier suffix
    
    # Training control flags
    train_ae_from_scratch: bool = True   # Whether to retrain autoencoder regardless of checkpoint
    train_gan_from_scratch: bool = True # Whether to retrain GAN regardless of checkpoint
    save_checkpoints: bool = False  # Whether to save model checkpoints, latent params, and viz to disk

    # ==========================================================================
    # Computed Properties
    # ==========================================================================
    
    @property
    def num_discriminator_outputs(self) -> int:
        """
        Number of discriminator output logits: N real classes + 1 fake class.
        
        Computed automatically from num_classes to prevent desynchronization.
        
        Previously (427J): This was a separate dataclass field with default=3.
        Now (511J): It's a @property that always returns num_classes + 1.
        
        Example:
            config = BAGANConfig(num_classes=3)
            config.num_discriminator_outputs  # → 4 (classes 0,1,2 + fake)
            
            config = BAGANConfig(num_classes=2)
            config.num_discriminator_outputs  # → 3 (classes 0,1 + fake)
        
        Why @property instead of a field?
            - Impossible to get out of sync with num_classes
            - No need for validation assertion (it's computed, not user-provided)
            - @property is accessed identically to a field (no parentheses)
            - Read-only: cannot be accidentally overwritten
        """
        return self.num_classes + 1

    # Model checkpoint paths
    # @property decorator turns a method into a "getter" that acts like an attribute
    # Usage: config.autoencoder_path (no parentheses needed)
    # Benefits: Computed dynamically each time, read-only, cleaner syntax
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
        return Path(f"{self.save_dir}/{self.model_prefix}discriminator_{self.date_suffix}.pth")
    
    @property
    def visualization_dir(self) -> Path:
        """Directory for saving training visualizations (sample grids, loss curves)."""
        return Path(f"{self.save_dir}/visualizations_{self.date_suffix}")
    
    # ==========================================================================
    # Device
    # ==========================================================================
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # num_discriminator_outputs assertion REMOVED:
        # Previously checked: num_discriminator_outputs == num_classes + 1
        # No longer needed because num_discriminator_outputs is now a @property
        # that automatically computes num_classes + 1. It CAN'T be wrong.
        
        assert self.num_classes >= 2, \
            f"num_classes must be >= 2 for meaningful classification, got {self.num_classes}"
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.img_size in [32, 64, 128], "image_size must be 32, 64, or 128"
    
    # ==========================================================================
    # Factory Methods
    # ==========================================================================
    
    # what does @classmethod do?
    # It allows us to define a method that is called on the CLASS itself, not on an instance.
    # This is useful for alternative constructors or factory methods that create instances
    @classmethod
    def from_ablation_config(cls, ablation_config) -> 'BAGANConfig':
        """
        Create BAGANConfig from an AblationConfig, syncing shared fields.
        
        This is the RECOMMENDED way to create BAGANConfig in the 511J pipeline.
        It ensures num_classes, latent_dim, paths, and device are consistent
        between the ablation experiment and the BAGAN training.
        
        How it works:
            @classmethod means this method is called on the CLASS, not an instance:
                config = BAGANConfig.from_ablation_config(abl_config)
            
            'cls' refers to BAGANConfig itself (like 'self' refers to an instance).
            We call cls(...) which is equivalent to BAGANConfig(...).
        
        Fields synced from AblationConfig:
            - num_classes: len(selected_classes) — e.g., 3 for [0, 2, 3]
            - latent_dim: Latent space dimensionality (default 128 in both configs)
            - save_dir: Directory for BAGAN model checkpoints (bagan_model_dir)
            - date_suffix: Experiment identifier (e.g., "511J")
            - device: Resolved device string ("cuda" or "cpu")
        
        Fields NOT synced (BAGAN-specific, use BAGANConfig defaults):
            - img_size, img_channels: Image parameters (always 64x64 grayscale)
            - encoder/decoder_channels: Architecture (fixed)
            - lr_g, lr_d, lr_ae: Learning rates (BAGAN-specific tuning)
            - gan_epochs, ae_epochs: Training duration
            - All other training hyperparameters
        
        Args:
            ablation_config: AblationConfig instance from ablation/config.py.
                Must have: num_classes (property), latent_dim, bagan_model_dir,
                           date_suffix, get_device() method.
        
        Returns:
            BAGANConfig instance with synced fields.
        
        Example:
            >>> from ablation.config import AblationConfig
            >>> from bagan.config import BAGANConfig
            >>> 
            >>> abl = AblationConfig(selected_classes=[0, 2, 3])  # 3 classes
            >>> bagan = BAGANConfig.from_ablation_config(abl)
            >>> bagan.num_classes                   # 3
            >>> bagan.num_discriminator_outputs      # 4 (auto-computed)
            >>> bagan.date_suffix                    # "511J"
        """
        return cls(
            num_classes=ablation_config.num_classes,      # len(selected_classes) → e.g., 3
            latent_dim=ablation_config.latent_dim,         # 128 (shared default)
            save_dir=ablation_config.bagan_model_dir,      # "saved"
            date_suffix=ablation_config.date_suffix,       # "511J"
            device=ablation_config.get_device(),           # auto-resolved "cuda" or "cpu"
        )

    def for_cv(self, ae_epochs: int = 30, gan_epochs: int = 200) -> 'BAGANConfig':
        """
        Create a reduced-epoch copy of this config for per-fold CV training.

        During cross-validation, each fold trains its own BAGAN from scratch.
        With 5 folds × multiple BAGAN modes, shorter epochs keep wall-clock
        time practical while maintaining sufficient augmentation quality.

        Args:
            ae_epochs: Autoencoder epochs for CV (default 30, vs 50 full).
            gan_epochs: GAN epochs for CV (default 300, vs full config value).

        Returns:
            New BAGANConfig with reduced epochs; all other fields inherited.

        Example:
            >>> bagan_config = BAGANConfig.from_ablation_config(config)
            >>> bagan_config_cv = bagan_config.for_cv(ae_epochs=30, gan_epochs=300)
            >>> bagan_config_cv.ae_epochs   # 30
            >>> bagan_config_cv.gan_epochs  # 300
        """
        return replace(self, ae_epochs=ae_epochs, gan_epochs=gan_epochs)
