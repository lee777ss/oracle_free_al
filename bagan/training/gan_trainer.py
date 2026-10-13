"""
BAGAN GAN Trainer

This module provides the training loop for the BAGAN GAN phase.
The GAN is trained AFTER the autoencoder phase, using:
    1. Generator initialized with Decoder weights
    2. Discriminator initialized with Encoder weights (conv layers only)
    3. Class-conditional latent sampling via ClassConditionalLatentGenerator

GAN Training Overview:
    BAGAN uses an (n+1)-class discriminator architecture:
    - n real classes (0 to n-1)
    - 1 fake class (index = n)
    
    Discriminator Training:
        - Real images → should be classified as their true class (0 to n-1)
        - Fake images → should be classified as fake (class n)
        
    Generator Training:
        - Generate images from class-conditional latents
        - Fool D into classifying fake images as the target real class

Loss Functions:
    - D_loss = CrossEntropy(D(real), real_labels) + CrossEntropy(D(fake), fake_label)
    - G_loss = CrossEntropy(D(fake), target_class)  # G wants D to think fake is real

Reference: Mariani et al. (2018) "BAGAN: Data Augmentation with Balancing GAN"

Usage:
    >>> from bagan.training import GANTrainer, ClassConditionalLatentGenerator
    >>> from bagan.models import BAGANGenerator, BAGANDiscriminator
    >>> from bagan.config import BAGANConfig
    >>> 
    >>> config = BAGANConfig()
    >>> 1. generator = BAGANGenerator(latent_dim=config.latent_dim)
    >>> 2. discriminator = BAGANDiscriminator(num_classes=config.num_classes)
    >>> 3. latent_gen = ClassConditionalLatentGenerator(config)
    >>> 
    >>> trainer = GANTrainer(config, generator, discriminator, latent_gen, train_loader)
    >>> history = trainer.fit()
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import os
import sys
import random  # For balanced real sampling per BAGAN paper

# =============================================================================
# Import Handling for Both Module and Direct Execution
# =============================================================================
# This pattern allows the file to be:
#   1. Imported as part of the bagan package: from bagan.training import GANTrainer
#   2. Run directly for testing: python bagan/training/gan_trainer.py

if __name__ == "__main__":
    # Direct execution: Add parent directories to Python's module search path
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from bagan.config import BAGANConfig
    from bagan.models.generator import BAGANGenerator
    from bagan.models.discriminator import BAGANDiscriminator, BAGANDiscriminatorLoss, BAGANGeneratorLoss
    from bagan.training.latent_space import ClassConditionalLatentGenerator
else:
    # Module import: Use relative imports (preferred for package code)
    from ..config import BAGANConfig
    from ..models.generator import BAGANGenerator
    from ..models.discriminator import BAGANDiscriminator, BAGANDiscriminatorLoss, BAGANGeneratorLoss
    from .latent_space import ClassConditionalLatentGenerator


class GANTrainer:
    """
    Trainer for BAGAN's GAN training phase.
    
    =========================================================================
    BAGAN Paper-Exact Implementation (Mariani et al., 2018)
    =========================================================================
    
    Key Quote from Paper:
        "For each batch we supply, 1/(n+1) of the total images are fake,
         i.e. we provide the best possible balance for the fake class."
        
        "The fake data is generated as output of G that takes as inputs
         latent vectors Zc extracted from the class-conditional latent
         vector generator. In turn, the class-conditional latent vector
         generator takes as input uniformly distributed class labels c,
         i.e. the fake images are uniformly distributed between the
         problem-specific classes."

    * What does alternating mean? Alternating means:
    You do NOT update the Generator and Discriminator at the same time.
    Instead, you update them one after the other, in turns.

    Concretely:

    1. Fix the Generator, update the Discriminator
    2. Fix the Discriminator, update the Generator
    3. Repeat

    This back-and-forth process is called alternating optimization.


    This class encapsulates the complete GAN training loop with:
        - Alternating Discriminator and Generator updates
        - BALANCED real sampling for D (paper-exact)
        - Class-conditional latent sampling for balanced generation
        - (n+1)-class discriminator loss
        - Training metrics tracking (losses, accuracies)
        - Sample visualization during training
        - Checkpoint save/load functionality
    
    BAGAN Paper-Exact Training Flow (N classes):
        ┌─────────────────────────────────────────────────────────────────┐
        │                     TRAINING LOOP (per batch)                   │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                 │
        │  ┌──────────────────────────────────────────────────────────┐   │
        │  │ STEP 1: DISCRIMINATOR UPDATE (Paper-Exact Balancing)     │   │
        │  │                                                          │   │
        │  │  *** BALANCED Real Sampling (NOT from imbalanced data) *** │
        │  │  Sample k images from each of the N classes (balanced)   │   │
        │  │  Real batch = N·k balanced samples                       │   │
        │  │      ↓                                                   │   │
        │  │  D(x_real) → logits for (N+1) classes                   │   │
        │  │      ↓                                                   │   │
        │  │  L_real = CrossEntropy(D(x), y)  # y ∈ {0, ..., N-1}    │   │
        │  │                                                          │   │
        │  │  Generate k FAKE images (uniform class distribution)     │   │
        │  │  Sample z ~ N(μ_c, σ_c² I) with uniform c               │   │
        │  │      ↓                                                   │   │
        │  │  G(z) → k fake images                                    │   │
        │  │      ↓                                                   │   │
        │  │  D(G(z)) → logits                                        │   │
        │  │      ↓                                                   │   │
        │  │  L_fake = CrossEntropy(D(G(z)), fake_label=N)           │   │
        │  │                                                          │   │
        │  │  Total D batch = (N+1)·k (N·k real + k fake)             │   │
        │  │  L_D = L_real + L_fake                                   │   │
        │  │  Update D parameters                                      │   │
        │  └──────────────────────────────────────────────────────────┘   │
        │                                                                 │
        │  ┌──────────────────────────────────────────────────────────┐   │
        │  │ STEP 2: GENERATOR UPDATE                                  │   │
        │  │                                                          │   │
        │  │  Sample z ~ N(μ_c, σ_c² I) with UNIFORM class c         │   │
        │  │  (sample_balanced: equal samples per class)              │   │
        │  │      ↓                                                   │   │
        │  │  G(z) → fake images                                      │   │
        │  │      ↓                                                   │   │
        │  │  D(G(z)) → logits                                        │   │
        │  │      ↓                                                   │   │
        │  │  L_G = CrossEntropy(D(G(z)), target_class=c)            │   │
        │  │  (G wants D to classify fake as real class c)           │   │
        │  │                                                          │   │
        │  │  Update G parameters                                      │   │
        │  └──────────────────────────────────────────────────────────┘   │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘
    
    Attributes:
        config (BAGANConfig): Configuration object with hyperparameters
        generator (BAGANGenerator): Generator network
        discriminator (BAGANDiscriminator): Discriminator network
        latent_generator (ClassConditionalLatentGenerator): For class-conditional z sampling
        train_loader (DataLoader): Training data iterator
        optimizer_g (torch.optim.Adam): Generator optimizer
        optimizer_d (torch.optim.Adam): Discriminator optimizer
        d_loss_fn (BAGANDiscriminatorLoss): Discriminator loss function
        g_loss_fn (BAGANGeneratorLoss): Generator loss function
        device (torch.device): Training device (cuda/cpu)
        history (Dict): Training metrics history
        current_epoch (int): Current training epoch (for resuming)
    
    Example:
        >>> trainer = GANTrainer(config, generator, discriminator, latent_gen, train_loader)
        >>> history = trainer.fit(epochs=500)
        >>> trainer.visualize_samples(class_label=0, n_samples=16)  # Any class
    """
    
    def __init__(
        self,
        config: BAGANConfig,
        generator: BAGANGenerator,
        discriminator: BAGANDiscriminator,
        latent_generator: ClassConditionalLatentGenerator,
        train_loader: DataLoader,
        real_images_by_class: Optional[Dict[int, torch.Tensor]] = None,
    ):
        """
        Initialize the GAN trainer.
        
        Args:
            config: BAGANConfig instance with hyperparameters:
                - gan_epochs: Number of training epochs (default: 500)
                - gan_lr_g: Generator learning rate (default: 0.0002)
                - gan_lr_d: Discriminator learning rate (default: 0.0002)
                - gan_beta1: Adam β₁ parameter (default: 0.5)
                - gan_beta2: Adam β₂ parameter (default: 0.999)
                - d_steps_per_g_step: D updates per G update (default: 1)
                - device: Training device string
                
            generator: BAGANGenerator instance
                - Should be initialized with Decoder weights (load_decoder_weights)
                
            discriminator: BAGANDiscriminator instance
                - Should be initialized with Encoder weights (load_encoder_weights)
                
            latent_generator: ClassConditionalLatentGenerator instance
                - Must be fitted (fit() called on encoder + training data)
                - Provides class-conditional sampling z ~ N(μ_c, σ_c² I)
                
            train_loader: DataLoader yielding (images, labels) tuples
                - images: Tensor of shape [batch, 1, 64, 64] in range [-1, 1]
                - labels: Tensor of shape [batch] with values in {0, 1}

            real_images_by_class: Optional {class_id: Tensor} of real reference
                images, needed only when config.fid_monitor_every > 0. Enables
                the periodic per-class FID checkpoint monitor (see
                `_maybe_run_fid_monitor`). Ignored when the monitor is off.

        Raises:
            RuntimeError: If latent_generator is not fitted
        """
        # =====================================================================
        # Validate Prerequisites
        # =====================================================================
        if not latent_generator.is_fitted:
            raise RuntimeError(
                "ClassConditionalLatentGenerator must be fitted before GAN training!\n"
                "Call latent_generator.fit(encoder, train_loader) first."
            )
        
        # =====================================================================
        # Store Configuration and Models
        # =====================================================================
        self.config = config
        self.generator = generator 
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.train_loader = train_loader
        self.real_images_by_class = real_images_by_class

        # FID best-checkpoint monitor state (populated only if
        # config.fid_monitor_every > 0 — see _maybe_run_fid_monitor).
        self.best_g_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.best_fid_epoch: Optional[int] = None
        self._best_fid_mean: float = float('inf')
        self.fid_monitor_history: Dict[str, List[Tuple[int, float]]] = {}

        # =====================================================================
        # Device Setup
        # =====================================================================
        self.device = torch.device(config.device)
        
        # Move models to device (GPU if available)
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        
        # =====================================================================
        # Loss Functions
        # =====================================================================
        # Discriminator loss: (n+1)-class CrossEntropy
        # - Real images → target = true class (0 to n-1)
        # - Fake images → target = fake class (n)
        self.d_loss_fn = BAGANDiscriminatorLoss(
            num_classes=config.num_classes,
            label_smoothing=config.label_smoothing
        )
        
        # Generator loss: CrossEntropy
        # - Fake images → target = intended real class (0 to n-1)
        # - G wants D to think fake images belong to real classes
        self.g_loss_fn = BAGANGeneratorLoss()
        
        # =====================================================================
        # Optimizers Setup: Separate Adam optimizers for G and D
        # =====================================================================
        # Why separate optimizers?
        #   - G and D have different objectives (adversarial)
        #   - May need different learning rates
        #   - Standard practice in GAN training
        #
        # Adam with β₁=0.5 is standard for GANs:
        #   - Lower momentum helps with adversarial dynamics
        #   - More responsive to changing loss landscape
        
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.lr_g,       # Learning rate for G (default: 0.0002)
            betas=(config.beta1, config.beta2)  # (0.5, 0.999)
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr_d,       # Learning rate for D (default: 0.0002)
            betas=(config.beta1, config.beta2)
        )
        
        # =====================================================================
        # Training Configuration
        # =====================================================================
        # d_steps_per_g_step: How many D updates per G update
        # Common values: 1 (balanced), 5 (WGAN-style)
        # For BAGAN with (n+1) loss, 1 usually works well
        self.d_steps_per_g_step = config.d_steps_per_g_step
        # g_steps_per_d_step: How many G updates per D update
        # Use > 1 when D dominates (D acc >> 50%) — gives G more time to learn
        self.g_steps_per_d_step = config.g_steps_per_d_step
        
        # =====================================================================
        # Training State and History
        # =====================================================================
        self.current_epoch = 0
        self.global_step = 0
        
        # Comprehensive training history
        self.history: Dict[str, List[float]] = {
            # Losses per epoch
            'd_loss': [],           # Total discriminator loss
            'd_loss_real': [],      # D loss on real images
            'd_loss_fake': [],      # D loss on fake images
            'g_loss': [],           # Generator loss
            
            # Accuracies per epoch
            'd_acc_real': [],       # D accuracy on real (correct class)
            'd_acc_fake': [],       # D accuracy on fake (labeled as fake)
            
            # Generation quality metrics (optional)
            'samples_generated': [],  # Number of synthetic samples generated
        }
        
        # =====================================================================
        # BAGAN Paper-Exact: Build Class-Indexed Storage for Balanced Sampling
        # =====================================================================
        # Per BAGAN paper: "For each batch we supply, 1/(n+1) of the total 
        # images are fake, i.e. we provide the best possible balance for the
        # fake class." This requires BALANCED real samples during D training.
        #
        # We store indices of samples by class to enable balanced sampling
        # even when the dataset is imbalanced (e.g., some classes have fewer samples)
        self._build_class_indices()
        
        # =====================================================================
        # Logging and Display
        # =====================================================================
        self._print_trainer_info()
    
    def _print_trainer_info(self) -> None:
        """Print trainer configuration summary."""
        print("\n" + "=" * 70)
        print("GANTrainer Initialized")
        print("=" * 70)
        print(f"  Generator: BAGANGenerator (latent_dim={self.config.latent_dim})")
        print(f"  Discriminator: BAGANDiscriminator (outputs={self.config.num_discriminator_outputs})")
        print(f"  Device: {self.device}")
        print(f"  Training epochs: {self.config.gan_epochs}")
        print(f"  Learning rate (G): {self.config.lr_g}")
        print(f"  Learning rate (D): {self.config.lr_d}")
        print(f"  Adam betas: ({self.config.beta1}, {self.config.beta2})")
        print(f"  D steps per G step: {self.d_steps_per_g_step}")
        print(f"  G steps per D step: {self.g_steps_per_d_step}")
        print(f"  Label smoothing:    {self.config.label_smoothing}")
        
        # Access underlying dataset from DeviceDataLoader
        if hasattr(self.train_loader, 'dl'):
            dataset = self.train_loader.dl.dataset  # type: ignore
            batch_size = self.train_loader.dl.batch_size  # type: ignore
        else:
            dataset = self.train_loader.dataset  # type: ignore
            batch_size = self.train_loader.batch_size  # type: ignore
        
        print(f"  Training samples: {len(dataset)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches per epoch: {len(self.train_loader)}")
        print(f"  Latent generator: {'Fitted ✓' if self.latent_generator.is_fitted else 'Not fitted ✗'}")
        print(f"  Generator path: {self.config.generator_path}")
        print(f"  Discriminator path: {self.config.discriminator_path}")
        print("=" * 70)
    
    # =========================================================================
    # PHASE 1: BAGAN Paper-Exact Balanced Real Sampling Infrastructure
    # =========================================================================
    # Per BAGAN paper (Mariani et al., 2018):
    #   "For each batch we supply, 1/(n+1) of the total images are fake"
    #   "The fake data is generated... the fake images are uniformly distributed"
    #
    # This requires D to see BALANCED real samples during training, NOT the
    # original imbalanced distribution → should become k:k per class)
    # =========================================================================
    
    def _build_class_indices(self) -> None:
        """
        Build an index mapping class labels to dataset indices.
        
        This enables efficient balanced sampling from the training dataset,
        even when classes are highly imbalanced. For example, with varying
        class sizes, we can still sample k from each class.
        
        Per BAGAN paper: D training requires balanced real samples (1/n+1 per class)
        
        Stores:
            self.class_indices: Dict mapping class_label → List[dataset_indices]
            self.class_counts: Dict mapping class_label → count
        """
        print("  Building class indices for balanced sampling...", end=" ")
        
        self.class_indices: Dict[int, List[int]] = {
            c: [] for c in range(self.config.num_classes)
        }
        # {0: [], 1: [], ..., N-1: []}
        
        # Access underlying dataset from DeviceDataLoader
        # DeviceDataLoader has a 'dl' attribute that contains the PyTorch DataLoader
        if hasattr(self.train_loader, 'dl'):
            dataset = self.train_loader.dl.dataset  # type: ignore
        else:
            dataset = self.train_loader.dataset  # type: ignore
        
        # Iterate through dataset and record indices per class
        for idx in range(len(dataset)):  # type: ignore
            _, label = dataset[idx]
            # Handle both tensor and int labels, ensure int type for dict key
            label_int: int = label.item() if isinstance(label, torch.Tensor) else int(label)
            self.class_indices[label_int].append(idx)
        
        # Record counts for reference
        self.class_counts = {c: len(indices) for c, indices in self.class_indices.items()}
        
        print(f"Done. Class counts: {self.class_counts}")
    
    def _sample_balanced_real(self, k_per_class: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample k real images from EACH class for balanced D training.
        
        BAGAN Paper-Exact Implementation:
            Per paper: "For each batch we supply, 1/(n+1) of the total images are fake"
            This means real samples must be balanced: k samples per class.
            
            For N classes with k samples each:
                - k Real Class 0      (1/(N+1) of D batch)
                - k Real Class 1      (1/(N+1) of D batch)
                - ...
                - k Real Class N-1    (1/(N+1) of D batch)
                - k Fake (added separately) (1/(N+1) of D batch)
        
        Uses sampling WITH REPLACEMENT to handle underrepresented classes
        (which may have fewer than k samples).
        
        Args:
            k_per_class: Number of samples to draw from each class
            
        Returns:
            images: Tensor of shape [n*k, 1, 64, 64] with balanced real samples
            labels: Tensor of shape [N*k] with corresponding class labels (0 to N-1)
            
        Example:
            For N=3, k=21:
                Returns (images[63, 1, 64, 64], labels[63])
                where labels contains 21 each of 0, 1, 2 (shuffled)
        """
        images_list = []
        labels_list = []
        
        # Access underlying dataset from DeviceDataLoader
        if hasattr(self.train_loader, 'dl'): # True
            dataset = self.train_loader.dl.dataset  # type: ignore
        else:
            dataset = self.train_loader.dataset  # type: ignore
        
        for class_idx in range(self.config.num_classes): # iterate over all N classes
            # Sample k indices WITH REPLACEMENT from this class
            # random.choices allows replacement, essential for underrepresented classes
            sampled_indices = random.choices(
                self.class_indices[class_idx], 
                k=k_per_class
            ) # this is the key: select k_per_class samples from each class
            # random.choice() → one element
            # random.sample() → without replacement
            # random.choices() → with replacement ✅

            # self.class_indices = {0: [idx, ...], 1: [idx, ...], ..., N-1: [idx, ...]}
            # self.class_indices[class_idx]: List of all dataset indices for this class
            

            for idx in sampled_indices:
                img, lbl = dataset[idx]
                images_list.append(img)
                # Handle both tensor and int labels
                lbl = lbl.item() if isinstance(lbl, torch.Tensor) else int(lbl)
                labels_list.append(lbl)
        
        # Stack into tensors
        images = torch.stack(images_list).to(self.device)
        labels = torch.tensor(labels_list, dtype=torch.long).to(self.device)
        
        # Shuffle to mix classes within the batch
        # This prevents any ordering bias in batch normalization
        perm = torch.randperm(len(labels))
        
        return images[perm], labels[perm]
    
    def fit(
        self,
        epochs: Optional[int] = None,
        verbose: bool = True,
        save_every: int = 50,
        visualize_every: int = 50,
        n_vis_samples: int = 16,
        show_plots: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the GAN for the specified number of epochs.
        
        =========================================================================
        BAGAN Paper-Exact Training (Mariani et al., 2018)
        =========================================================================
        
        This training loop implements the EXACT batch composition from the paper:
        
        Quote: "For each batch we supply, 1/(n+1) of the total images are fake,
               i.e. we provide the best possible balance for the fake class."
        
        D Batch Composition (N classes, k samples per class):
            • 1/(N+1) Real Class 0 (k samples)  → balanced real
            • 1/(N+1) Real Class 1 (k samples)  → balanced real
            • ...                                → balanced real
            • 1/(N+1) Real Class N-1 (k samples) → balanced real
            • 1/(N+1) Fake (k samples)           → uniform class distribution
            Total = (N+1)·k samples
        
        G Batch Composition:
            • sample_balanced(n*k) → uniform class labels → N·k samples
            Quote: "the class-conditional latent vector generator takes as input
                    uniformly distributed class labels c"
        
        Training Flow:
            for epoch in range(epochs):
                for batch_idx in range(num_batches):
                    # D step: Sample BALANCED real batch + generate fake
                    discriminator_step(k_per_class)
                    
                    # G step: Generate balanced fake, fool D
                    generator_step(n * k_per_class)
        
        Args:
            epochs: Number of epochs to train. If None, uses config.gan_epochs
            verbose: Whether to print progress every epoch
            save_every: Save checkpoint every N epochs (0 to disable)
            visualize_every: Show generated samples every N epochs (0 to disable)
            n_vis_samples: Number of samples to show in visualization
        
        Returns:
            history: Dictionary containing training metrics per epoch
        
        Example:
            >>> history = trainer.fit(epochs=500, save_every=100)
            >>> plt.plot(history['d_loss'], label='D Loss')
            >>> plt.plot(history['g_loss'], label='G Loss')
            >>> plt.legend()
            >>> plt.show()
        """
        epochs = epochs or self.config.gan_epochs 
        
        # The or operator:
        # returns the first operand that is truthy;
        # if the first operand is falsy, it returns the second operand.
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"Starting GAN Training: {epochs} epochs")
            print("=" * 70)
        
        # Track training time
        start_time = time.time()
        
        # Visualization save directory
        if self.config.save_checkpoints:
            vis_dir = Path(self.config.visualization_dir)
            vis_dir.mkdir(parents=True, exist_ok=True)
        else:
            vis_dir = None
        
        # =================================================================
        # Learning Rate Schedulers (cosine annealing for stable convergence)
        # =================================================================
        scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=epochs, eta_min=self.config.lr_g * 0.1
        )
        scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d, T_max=epochs, eta_min=self.config.lr_d * 0.1
        )
        
        # =====================================================================
        # Main Training Loop
        # =====================================================================
        for epoch in range(self.current_epoch, epochs):
            epoch_start_time = time.time()
            
            # -----------------------------------------------------------------
            # Train One Epoch
            # -----------------------------------------------------------------
            epoch_metrics = self._train_epoch()
            
            # Record metrics in history
            for key, value in epoch_metrics.items():
                if key in self.history:
                    self.history[key].append(value)

            # -----------------------------------------------------------------
            # Per-class FID best-checkpoint monitor (opt-in, see config docstring)
            # -----------------------------------------------------------------
            self._maybe_run_fid_monitor(epoch)

            # -----------------------------------------------------------------
            # Progress Logging
            # -----------------------------------------------------------------
            epoch_time = time.time() - epoch_start_time
            
            if verbose:
                msg = f"Epoch [{epoch + 1:4d}/{epochs}] | "
                msg += f"D: {epoch_metrics['d_loss']:.4f} "
                msg += f"(real:{epoch_metrics['d_loss_real']:.4f}, fake:{epoch_metrics['d_loss_fake']:.4f}) | "
                msg += f"G: {epoch_metrics['g_loss']:.4f} | "
                msg += f"D_acc: {epoch_metrics['d_acc_real']:.1%}/{epoch_metrics['d_acc_fake']:.1%} | "
                msg += f"{epoch_time:.1f}s"
                print(msg)
            
            # -----------------------------------------------------------------
            # Checkpoint Saving (periodic)
            # -----------------------------------------------------------------
            if save_every > 0 and self.config.save_checkpoints and (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch=epoch + 1)
            
            # -----------------------------------------------------------------
            # Sample Visualization (display + save to file)
            # -----------------------------------------------------------------
            if show_plots and visualize_every > 0 and (epoch + 1) % visualize_every == 0:
                sp = str(vis_dir / f"samples_epoch_{epoch + 1:04d}.png") if vis_dir else None
                self._visualize_samples_epoch(
                    epoch + 1, n_samples=n_vis_samples, save_path=sp
                )
            
            # -----------------------------------------------------------------
            # Step Learning Rate Schedulers
            # -----------------------------------------------------------------
            scheduler_g.step()
            scheduler_d.step()
            
            # Update current epoch for checkpoint resumption
            self.current_epoch = epoch + 1
        
        # =====================================================================
        # Save Final Epoch Checkpoint + Visualizations
        # =====================================================================
        if self.config.save_checkpoints:
            self.save_checkpoint(epoch=epochs)
            if show_plots:
                final_samples_path = vis_dir / "samples_final.png"
                self._visualize_samples_epoch(
                    epochs, n_samples=n_vis_samples, save_path=str(final_samples_path)
                )
                curves_path = vis_dir / "training_curves.png"
                self.plot_training_curves(save_path=str(curves_path))
        else:
            if show_plots:
                self._visualize_samples_epoch(epochs, n_samples=n_vis_samples, save_path=None)
                self.plot_training_curves(save_path=None)
        
        # =====================================================================
        # Training Complete
        # =====================================================================
        total_time = time.time() - start_time
        
        if verbose:
            print("\n" + "=" * 70)
            print("GAN Training Complete!")
            print("=" * 70)
            print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"  Final D loss: {self.history['d_loss'][-1]:.4f}")
            print(f"  Final G loss: {self.history['g_loss'][-1]:.4f}")
            print(f"  Final D accuracy (real/fake): "
                  f"{self.history['d_acc_real'][-1]:.1%} / {self.history['d_acc_fake'][-1]:.1%}")
            print(f"  Saved final-epoch weights to: {self.config.generator_path}")
            print(f"  Saved final-epoch weights to: {self.config.discriminator_path}")
            if show_plots:
                print(f"  Visualizations saved to: {vis_dir}/")
            print("=" * 70)

        if self.config.fid_monitor_every > 0 and self.best_g_state_dict is not None:
            print(f"\n  [fid-monitor] best mean FID {self._best_fid_mean:.2f} "
                  f"@ epoch {self.best_fid_epoch} (final epoch = {epochs})")
            print(f"  [fid-monitor] call trainer.restore_best_generator() to "
                  f"swap the generator to that checkpoint.")

        return self.history

    def _maybe_run_fid_monitor(self, epoch: int) -> None:
        """Cheap per-class FID monitor + best-checkpoint snapshot.

        Runs every `config.fid_monitor_every` epochs (no-op if 0, the
        default). Ports the pattern used for the Ab-11a ablation arm
        (ablation/trainer.py `_maybe_run_fid_monitor`) into the plain-BAGAN
        warm-up trainer, generalized to cover ALL classes rather than just
        the minority ones — added after seed-44 of the 140J run showed the
        final-epoch generator's class-0 (majority) FID regress past the
        pretrained autoencoder's FID, while seeds 42/43/45/46 all improved
        on it. Class 0 has 720 real images vs. 36 for the minority classes,
        so D can overfit/dominate there; without this monitor, a bad
        late-epoch snapshot for that one class ships silently.

        Side effects:
            * `self.fid_monitor_history['fid_class_<c>']` gains an
              `(epoch, fid)` tuple per class.
            * `self.fid_monitor_history['fid_mean']` gains `(epoch, mean_fid)`.
            * If mean FID strictly improves over `self._best_fid_mean`,
              snapshots `self.best_g_state_dict` (CPU clone) and updates
              `self.best_fid_epoch`.

        Silently no-ops when the monitor is off, the cadence isn't hit, no
        `real_images_by_class` was passed to the constructor, or the FID
        reference cache hasn't been built yet (caller error — surfaced once
        as a warning, not raised, so a misconfigured monitor doesn't crash
        an otherwise-fine GAN training run).
        """
        every = int(self.config.fid_monitor_every)
        if every <= 0:
            return
        if (epoch + 1) % every != 0:
            return
        if not self.real_images_by_class:
            if not getattr(self, '_fid_monitor_warned', False):
                print("  [fid-monitor] config.fid_monitor_every > 0 but no "
                      "real_images_by_class was passed to GANTrainer — skipping.")
                self._fid_monitor_warned = True
            return

        try:
            from ablation.fid import fid_per_class
        except ImportError:
            if not getattr(self, '_fid_monitor_warned', False):
                print("  [fid-monitor] `ablation.fid` unavailable — skipping.")
                self._fid_monitor_warned = True
            return

        try:
            fid_dict: Dict[int, float] = fid_per_class(
                self.generator,
                self.latent_generator,
                classes=sorted(self.real_images_by_class.keys()),
                n_samples=int(self.config.fid_monitor_samples),
                device=self.device,
                # Reuses the on-disk reference cache if it exists (built in
                # §D.6 in the normal notebook flow); builds+caches it on
                # first call otherwise, so the monitor doesn't require a
                # specific call order.
                real_images_by_class=self.real_images_by_class,
                ref_cache_dir=self.config.save_dir,
                date_suffix=self.config.date_suffix,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            # sqrtm(sigma1 @ sigma2) can produce a large spurious imaginary
            # part (see ablation/shift_metrics.py::_frechet_distance) when
            # either covariance is severely rank-deficient — e.g. a
            # mid-training generator whose fake-feature spread collapses, or
            # too few fid_monitor_samples relative to the 2048-dim Inception
            # feature space. That's a transient numerical failure of the
            # metric, not a reason to abort an otherwise fine training run,
            # so skip this epoch's reading rather than propagate.
            print(f"  [fid-monitor ep {epoch + 1}] FID computation failed "
                  f"({exc}) — skipping this reading.")
            return

        hist = self.fid_monitor_history
        for c, v in fid_dict.items():
            hist.setdefault(f'fid_class_{c}', []).append((epoch, float(v)))
        if not fid_dict:
            return
        mean_fid = float(sum(fid_dict.values()) / len(fid_dict))
        hist.setdefault('fid_mean', []).append((epoch, mean_fid))

        pretty = ' '.join(f'c{c}={v:.2f}' for c, v in sorted(fid_dict.items()))
        if mean_fid < self._best_fid_mean:
            self._best_fid_mean = mean_fid
            self.best_fid_epoch = epoch + 1
            self.best_g_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in self.generator.state_dict().items()
            }
            print(f"  [fid-monitor ep {epoch + 1}] mean {mean_fid:.2f} ({pretty}) ↓ best -> snapshotted G")
        else:
            print(f"  [fid-monitor ep {epoch + 1}] mean {mean_fid:.2f} ({pretty})")

    def restore_best_generator(self) -> bool:
        """Load the best-FID generator snapshot into `self.generator` in place.

        Returns True if a snapshot existed and was restored, False if the
        monitor never ran or never improved on its first reading (nothing to
        restore — the final-epoch generator IS the best one seen).
        """
        if self.best_g_state_dict is None:
            print("  [fid-monitor] no snapshot to restore (monitor off, or "
                  "final epoch was already the best).")
            return False
        self.generator.load_state_dict(self.best_g_state_dict)
        self.generator.to(self.device)
        print(f"  [fid-monitor] restored generator to epoch {self.best_fid_epoch} "
              f"(mean FID {self._best_fid_mean:.2f}).")
        return True

    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch and return average metrics.
        
        BAGAN Paper-Exact Implementation:
            Per paper: "For each batch we supply, 1/(n+1) of the total images are fake"
            
            For N classes, each D batch is composed of:
                - k Real Class 0      (1/(N+1) of batch)
                - k Real Class 1      (1/(N+1) of batch)
                - ...
                - k Real Class N-1    (1/(N+1) of batch)
                - k Fake              (1/(N+1) of batch, uniform class distribution)
            
            Total D batch = (N+1)·k samples
            
            This is DIFFERENT from the original implementation that used
            imbalanced real batches from the DataLoader.
        
        One epoch = one complete pass through iterations (based on DataLoader length).
        
        Returns:
            Dict of average metrics over the epoch:
                - d_loss, d_loss_real, d_loss_fake
                - g_loss
                - d_acc_real, d_acc_fake
        """
        # Set models to training mode
        # - BatchNorm: use batch statistics, update running stats
        # - Dropout: randomly drop neurons (if used)
        self.generator.train()
        self.discriminator.train()
        
        # Accumulators for epoch metrics
        epoch_d_loss = 0.0
        epoch_d_loss_real = 0.0
        epoch_d_loss_fake = 0.0
        epoch_g_loss = 0.0
        epoch_d_correct_real = 0
        epoch_d_correct_fake = 0
        epoch_total_real = 0
        epoch_total_fake = 0
        
        num_batches = len(self.train_loader)
        
        # =====================================================================
        # BAGAN Paper-Exact: Compute k_per_class for Balanced Batches
        # =====================================================================
        # Per paper: D batch = (n+1) equal parts
        # For batch_size=64, N=3: k = 64 // 4 = 16
        # Actual D batch = 3*21 = 63 samples (21 real class0, 21 real class1, 21 fake)
        #
        # We use the DataLoader's batch_size as reference to maintain similar
        # computational cost per iteration.
        #
        # IMPORTANT: When using custom batch_sampler (e.g., NaturalMinBatchSampler),
        # PyTorch DataLoader doesn't set the batch_size attribute (it becomes None).
        # We need to extract batch_size from the batch_sampler itself.
        if hasattr(self.train_loader, 'dl'):
            # DeviceDataLoader wrapper case
            dl = self.train_loader.dl  # type: ignore
            if hasattr(dl, 'batch_sampler') and hasattr(dl.batch_sampler, 'batch_size'):
                # Custom batch sampler with batch_size attribute
                reference_batch_size = dl.batch_sampler.batch_size
            else:
                # Standard DataLoader with batch_size parameter
                reference_batch_size = dl.batch_size or 64
        else:
            # Direct DataLoader case
            if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'batch_size'):  # type: ignore
                reference_batch_size = self.train_loader.batch_sampler.batch_size  # type: ignore
            else:
                reference_batch_size = self.train_loader.batch_size or 64  # type: ignore
        
        k_per_class = max(1, reference_batch_size // (self.config.num_classes + 1))
        # how to determine k_per_class:
        # - reference_batch_size: DataLoader batch size (e.g., 64)
        # - num_classes: N (e.g., 3)
        # - (N+1) = 4 for N=3
        # - k_per_class = 64 // 4 = 16
        
        # =====================================================================
        # Iterate Over Training Batches
        # =====================================================================
        # Note: We iterate over DataLoader to maintain epoch structure, but
        # D step now samples its own balanced batch instead of using DataLoader batch
        for batch_idx, _ in enumerate(self.train_loader):
            
            # -----------------------------------------------------------------
            # DISCRIMINATOR STEP(S) - BAGAN Paper-Exact
            # -----------------------------------------------------------------
            # D now internally samples balanced real batches and generates
            # balanced fake batches, per paper specification
            for _ in range(self.d_steps_per_g_step):
                d_loss, d_loss_real, d_loss_fake, d_metrics = self._discriminator_step(
                    k_per_class
                )
            
            # Accumulate D metrics (from last D step)
            epoch_d_loss += d_loss
            epoch_d_loss_real += d_loss_real
            epoch_d_loss_fake += d_loss_fake
            epoch_d_correct_real += d_metrics['correct_real']
            epoch_d_correct_fake += d_metrics['correct_fake']
            epoch_total_real += d_metrics['total_real']
            epoch_total_fake += d_metrics['total_fake']
            
            # -----------------------------------------------------------------
            # GENERATOR STEP(S)
            # -----------------------------------------------------------------
            # G uses balanced sampling (sample_balanced), which is correct per paper:
            # "the class-conditional latent vector generator takes as input
            #  uniformly distributed class labels c"
            #
            # g_steps_per_d_step > 1 compensates when D dominates:
            # G gets more gradient updates per D update to close the gap.
            g_batch_size = k_per_class * self.config.num_classes
            g_step_losses = []
            for _ in range(self.g_steps_per_d_step):
                g_loss = self._generator_step(g_batch_size)
                g_step_losses.append(g_loss)
            epoch_g_loss += sum(g_step_losses) / len(g_step_losses)
            
            self.global_step += 1
        
        # =====================================================================
        # Compute Epoch Averages
        # =====================================================================
        return {
            'd_loss': epoch_d_loss / num_batches,
            'd_loss_real': epoch_d_loss_real / num_batches,
            'd_loss_fake': epoch_d_loss_fake / num_batches,
            'g_loss': epoch_g_loss / num_batches,
            'd_acc_real': epoch_d_correct_real / max(epoch_total_real, 1),
            'd_acc_fake': epoch_d_correct_fake / max(epoch_total_fake, 1),
        }
    
    def _discriminator_step(
        self, 
        k_per_class: int
    ) -> Tuple[float, float, float, Dict[str, int]]:
        """
        Perform one discriminator update step with BAGAN paper-exact batch composition.
        
        =========================================================================
        BAGAN Paper-Exact Implementation (Mariani et al., 2018):
        =========================================================================
        
        Quote from paper:
            "For each batch we supply, 1/(n+1) of the total images are fake,
             i.e. we provide the best possible balance for the fake class."
            
            "The fake data is generated as output of G that takes as inputs
             latent vectors Zc extracted from the class-conditional latent
             vector generator. In turn, the class-conditional latent vector
             generator takes as input uniformly distributed class labels c,
             i.e. the fake images are uniformly distributed between the
             problem-specific classes."
        
        Batch Composition for N classes:
            ┌─────────────────────────────────────────────────────────────────┐
            │  D TRAINING BATCH (Total = (N+1)·k samples)                    │
            ├─────────────────────────────────────────────────────────────────┤
            │  • k Real Class 0 samples    → Target: label 0   (1/(N+1))     │
            │  • k Real Class 1 samples    → Target: label 1   (1/(N+1))     │
            │  • ...                                                         │
            │  • k Real Class N-1 samples  → Target: label N-1 (1/(N+1))     │
            │  • k Fake samples (uniform)  → Target: label N   (1/(N+1))     │
            │    └─ Internally: ~k/N generated per class (uniform)            │
            └─────────────────────────────────────────────────────────────────┘
        
        This is DIFFERENT from the naive approach:
            ❌ Old: Use imbalanced real data + balanced fake
            ✅ New: Use balanced real (k per class) + balanced fake
        
        Why this matters:
            - Prevents D from learning spurious correlation based on class prevalence
            - D sees perfect (N+1)-way classification balance
            - Ensures D judges image quality, not class prevalence
        
        Args:
            k_per_class: Number of samples per real class (total real = N*k)
            N: number of real classes
        
        Returns:
            Tuple of:
                - d_loss: Total discriminator loss (float)
                - d_loss_real: Loss on real images (float)
                - d_loss_fake: Loss on fake images (float)
                - metrics: Dict with accuracy counts
        """
        n = self.config.num_classes  # N real classes
        
        # ---------------------------------------------------------------------
        # Step 1: Zero Gradients
        # ---------------------------------------------------------------------
        self.optimizer_d.zero_grad()
        
        # ---------------------------------------------------------------------
        # Step 2: Sample BALANCED Real Images (k per class)
        # ---------------------------------------------------------------------
        # Per BAGAN paper: D sees equal samples from each real class
        # This is achieved via _sample_balanced_real() with replacement
        real_images, real_labels = self._sample_balanced_real(k_per_class)
        # real_images.shape = [N*k, 1, 64, 64]
        # real_labels.shape = [N*k] with k labels per class
        
        num_real = real_images.size(0)  # = n*k
        
        # ---------------------------------------------------------------------
        # Step 3: Forward Pass on REAL Images
        # ---------------------------------------------------------------------
        real_logits = self.discriminator(real_images)
        
        # Count correct predictions on real images
        real_preds = torch.argmax(real_logits, dim=1)
        correct_real = (real_preds == real_labels).sum().item()
        
        # ---------------------------------------------------------------------
        # Step 4: Generate FAKE Images (k total, uniform class distribution)
        # ---------------------------------------------------------------------
        # Per paper: "1/(n+1) of the total images are fake" 
        # and "fake images are uniformly distributed between... classes"
        #
        # sample_balanced(k) returns k total samples with k/n per class
        # This gives us k fake with uniform class distribution
        num_fake = k_per_class  # k fake samples = 1/(n+1) of total batch
        z, fake_target_classes = self.latent_generator.sample_balanced(num_fake)
        
        # Generate fake images
        # detach() prevents gradients flowing to G during D training
        fake_images = self.generator(z).detach()
        
        # ---------------------------------------------------------------------
        # Step 5: Forward Pass on FAKE Images
        # ---------------------------------------------------------------------
        fake_logits = self.discriminator(fake_images)
        
        # Count correct predictions on fake images
        # D should output class n (fake class index) for all fake images
        fake_preds = torch.argmax(fake_logits, dim=1)
        fake_label = self.config.num_classes  # = N (fake class index)
        correct_fake = (fake_preds == fake_label).sum().item()
        
        # ---------------------------------------------------------------------
        # Step 6: Compute Discriminator Loss
        # ---------------------------------------------------------------------
        # L_D = CrossEntropy(D(real), real_labels) + CrossEntropy(D(fake), fake_label)
        d_loss, d_loss_real, d_loss_fake = self.d_loss_fn(
            real_logits, real_labels, fake_logits
        )
        
        # ---------------------------------------------------------------------
        # Step 7: Backward Pass and Parameter Update
        # ---------------------------------------------------------------------
        d_loss.backward()
        self.optimizer_d.step()
        
        # ---------------------------------------------------------------------
        # Return Metrics
        # ---------------------------------------------------------------------
        metrics = {
            'correct_real': correct_real,
            'correct_fake': correct_fake,
            'total_real': num_real,      # = N*k (e.g., 48 for k=16, N=3)
            'total_fake': num_fake,      # = k (e.g., 16)
        }
        
        return d_loss.item(), d_loss_real.item(), d_loss_fake.item(), metrics
    
    def _generator_step(self, batch_size: int) -> float:
        """
        Perform one generator update step.
        
        Generator Training Objective:
            - Fool D into classifying fake images as REAL classes (0 to N-1)
            - NOT as fake (class N)
        
        Loss Function:
            L_G = CrossEntropy(D(G(z_c)), target_class=c)
            
            Where z_c is sampled from class c's distribution N(μ_c, σ_c² I)
        
        Key Insight:
            - G generates images from class-conditional latents
            - If z was sampled for class 1, G wants D to output class 1
            - This teaches G to generate class-specific images
        
        Args:
            batch_size: Number of fake samples to generate
        
        Returns:
            g_loss: Generator loss (float)
        """
        # ---------------------------------------------------------------------
        # Step 1: Zero Gradients
        # ---------------------------------------------------------------------
        self.optimizer_g.zero_grad()
        
        # ---------------------------------------------------------------------
        # Step 2: Sample Class-Conditional Latent Vectors
        # ---------------------------------------------------------------------
        # z_c ~ N(μ_c, σ_c² I) for balanced classes
        # target_classes tells us which class each z was sampled for
        z, target_classes = self.latent_generator.sample_balanced(batch_size)
        
        # ---------------------------------------------------------------------
        # Step 3: Generate Fake Images
        # ---------------------------------------------------------------------
        # NO detach() here! We WANT gradients to flow through G
        # G(z) computes the forward pass with gradient tracking
        fake_images = self.generator(z)
        
        # ---------------------------------------------------------------------
        # Step 4: Get Discriminator's Opinion on Fake Images
        # ---------------------------------------------------------------------
        # D(G(z)) → logits
        # G wants these logits to favor target_classes (real classes)
        fake_logits = self.discriminator(fake_images)
        
        # ---------------------------------------------------------------------
        # Step 5: Compute Generator Loss
        # ---------------------------------------------------------------------
        # G wants D to classify fake images as their target real class
        # If z was sampled for class 1, D should output class 1
        # This is the OPPOSITE of D's goal (which wants D to output fake=N)
        g_loss = self.g_loss_fn(fake_logits, target_classes)
        
        # ---------------------------------------------------------------------
        # Step 6: Backward Pass and Parameter Update
        # ---------------------------------------------------------------------
        # Compute gradients: ∂L_G/∂θ_G
        # This backpropagates through: L_G → D → G(z) → G.parameters()
        # D's parameters are NOT updated (they're not in optimizer_g)
        g_loss.backward()
        
        # Update generator parameters only
        self.optimizer_g.step()
        
        return g_loss.item()
    
    def _visualize_samples_epoch(
        self, 
        epoch: int, 
        n_samples: int = 16, # per class
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize generated samples during training.
        
        Shows a grid of generated images for each class to monitor
        training progress. Useful for detecting mode collapse.
        
        Args:
            epoch: Current epoch number (for title)
            n_samples: Number of samples per class to show
            save_path: If provided, save figure to this path
        """
        self.generator.eval()
        
        with torch.no_grad():
            images_per_class = []
            
            for class_idx in range(self.config.num_classes):
                # Sample latents for this class
                z = self.latent_generator.sample(class_idx, n_samples)
                
                # Generate images
                fake_images = self.generator(z)
                images_per_class.append(fake_images.cpu())
        
        self.generator.train()
        
        # Create visualization grid
        self._plot_class_samples(
            images_per_class, 
            title=f"Epoch {epoch} Generated Samples",
            save_path=save_path
        )
    
    def visualize_samples(
        self, 
        n_samples: int = 16,
        class_label: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Generate and visualize samples.
        
        Args:
            n_samples: Number of samples to generate
            class_label: Specific class to generate. If None, generates balanced.
            save_path: If provided, save figure to this path
        
        Returns:
            Generated images tensor [n_samples, 1, 64, 64]
        """
        self.generator.eval()
        
        with torch.no_grad():
            if class_label is not None:
                # Generate for specific class
                z = self.latent_generator.sample(class_label, n_samples)
                target_classes = torch.full((n_samples,), class_label, 
                                           dtype=torch.long, device=self.device)
            else:
                # Balanced sampling
                z, target_classes = self.latent_generator.sample_balanced(n_samples)
            
            fake_images = self.generator(z)
        
        self.generator.train()
        
        # Visualize
        self._plot_samples(
            fake_images.cpu(), 
            target_classes.cpu(),
            title=f"Generated Samples (Class: {'All' if class_label is None else class_label})",
            save_path=save_path
        )
        
        return fake_images
    
    def _plot_samples(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        title: str = "Generated Samples",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot a grid of generated samples.
        
        Args:
            images: Generated images [N, 1, 64, 64]
            labels: Class labels [N]
            title: Plot title
            save_path: Optional path to save figure
        """
        n_samples = images.shape[0]
        n_cols = min(8, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            row, col = i // n_cols, i % n_cols
            
            # Denormalize: [-1, 1] → [0, 1]
            img = (images[i].squeeze() + 1) / 2
            img = torch.clamp(img, 0, 1)
            
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f"c={labels[i].item()}", fontsize=8)
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(n_samples, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(title, fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Samples saved to: {save_path}")
        
        plt.show()
    
    def _plot_class_samples(
        self,
        images_per_class: List[torch.Tensor],
        title: str = "Class Samples",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot samples organized by class in rows.
        
        Args:
            images_per_class: List of image tensors, one per class
            title: Plot title
            save_path: If provided, save figure to this path
        """
        num_classes = len(images_per_class)
        n_samples = images_per_class[0].shape[0]
        
        fig, axes = plt.subplots(num_classes, n_samples, 
                                figsize=(n_samples * 1.2, num_classes * 1.5))
        
        class_names = [f'Class {i}' for i in range(num_classes)]
        
        for class_idx, class_images in enumerate(images_per_class):
            for i in range(n_samples):
                # Denormalize
                img = (class_images[i].squeeze() + 1) / 2
                img = torch.clamp(img, 0, 1)
                
                if num_classes == 1:
                    ax = axes[i]
                else:
                    ax = axes[class_idx, i]
                
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                
                # Add class name label to the first image of each row
                if i == 0:
                    ax.text(-0.1, 0.5, class_names[class_idx], 
                           transform=ax.transAxes, fontsize=12, fontweight='bold',
                           rotation=90, va='center', ha='right')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.show()
    
    def save_checkpoint(
        self, 
        path_g: Optional[str] = None,
        path_d: Optional[str] = None,
        epoch: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        Save model checkpoints for resumption or deployment.
        
        Saves both Generator and Discriminator states, including:
            - Model state dict (learned weights)
            - Optimizer state dict (Adam momentum buffers)
            - Training history
            - Current epoch
            - Config snapshot
        
        Args:
            path_g: Generator save path. If None, uses config.generator_path
            path_d: Discriminator save path. If None, uses config.discriminator_path
            epoch: Epoch number to include in checkpoint
        
        Returns:
            Tuple of (generator_path, discriminator_path)
        """
        if not self.config.save_checkpoints:
            print("Checkpoint saving disabled (save_checkpoints=False)")
            return None, None
        
        path_g = path_g or self.config.generator_path
        path_d = path_d or self.config.discriminator_path
        
        # Ensure directories exist
        Path(path_g).parent.mkdir(parents=True, exist_ok=True)
        Path(path_d).parent.mkdir(parents=True, exist_ok=True)
        
        # Build checkpoint dictionaries
        checkpoint_g = {
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer_g.state_dict(),
            'epoch': epoch or self.current_epoch,
            'global_step': self.global_step,
            'history': self.history,
            'config': {
                'latent_dim': self.config.latent_dim,
                'lr_g': self.config.lr_g,
            }
        }
        
        checkpoint_d = {
            'model_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer_d.state_dict(),
            'epoch': epoch or self.current_epoch,
            'global_step': self.global_step,
            'config': {
                'num_classes': self.config.num_classes,
                'lr_d': self.config.lr_d,
            }
        }
        
        # Save checkpoints
        torch.save(checkpoint_g, path_g)
        torch.save(checkpoint_d, path_d)
        
        print(f"Checkpoints saved:")
        print(f"  Generator: {path_g}")
        print(f"  Discriminator: {path_d}")
        
        return path_g, path_d
    
    def load_checkpoint(
        self, 
        path_g: Optional[str] = None,
        path_d: Optional[str] = None,
        load_optimizer: bool = True
    ) -> 'GANTrainer':
        """
        Load checkpoints for training resumption.
        
        Args:
            path_g: Generator checkpoint path
            path_d: Discriminator checkpoint path
            load_optimizer: Whether to restore optimizer states
        
        Returns:
            self: For method chaining
        """
        path_g = path_g or self.config.generator_path
        path_d = path_d or self.config.discriminator_path
        
        # Load checkpoints
        # weights_only=False: Allow loading full checkpoint with optimizer state
        checkpoint_g = torch.load(path_g, map_location=self.device, weights_only=False)
        checkpoint_d = torch.load(path_d, map_location=self.device, weights_only=False)
        
        # Restore model weights
        self.generator.load_state_dict(checkpoint_g['model_state_dict'])
        self.discriminator.load_state_dict(checkpoint_d['model_state_dict'])
        
        # Optionally restore optimizer states
        if load_optimizer:
            if 'optimizer_state_dict' in checkpoint_g:
                self.optimizer_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
            if 'optimizer_state_dict' in checkpoint_d:
                self.optimizer_d.load_state_dict(checkpoint_d['optimizer_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint_g.get('epoch', 0)
        self.global_step = checkpoint_g.get('global_step', 0)
        self.history = checkpoint_g.get('history', self.history)
        
        print(f"Checkpoints loaded:")
        print(f"  Generator: {path_g}")
        print(f"  Discriminator: {path_d}")
        print(f"  Resuming from epoch: {self.current_epoch}")
        
        return self
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history for analysis and plotting.
        
        Returns:
            Dict containing training metrics per epoch
        """
        return self.history.copy()
    
    def plot_training_curves(
        self, 
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot training loss curves.
        
        Creates a figure showing:
            - D loss (total, real, fake) over epochs
            - G loss over epochs
            - D accuracy (real, fake) over epochs
        
        Args:
            save_path: If provided, save figure to this path
        """
        if not self.history['d_loss']:
            print("No training history to plot. Train the model first.")
            return
        
        epochs = range(1, len(self.history['d_loss']) + 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Losses
        axes[0].plot(epochs, self.history['d_loss'], 'b-', label='D Total', linewidth=2)
        axes[0].plot(epochs, self.history['d_loss_real'], 'b--', label='D Real', alpha=0.7)
        axes[0].plot(epochs, self.history['d_loss_fake'], 'b:', label='D Fake', alpha=0.7)
        axes[0].plot(epochs, self.history['g_loss'], 'r-', label='G Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Losses')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: D Accuracy
        axes[1].plot(epochs, self.history['d_acc_real'], 'g-', label='D Acc (Real)', linewidth=2)
        axes[1].plot(epochs, self.history['d_acc_fake'], 'm-', label='D Acc (Fake)', linewidth=2)
        axes[1].axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Random')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Discriminator Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        # Plot 3: G vs D Loss ratio
        g_d_ratio = [g / (d + 1e-8) for g, d in zip(self.history['g_loss'], self.history['d_loss'])]
        axes[2].plot(epochs, g_d_ratio, 'purple', linewidth=2)
        axes[2].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Balanced')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('G/D Loss Ratio')
        axes[2].set_title('Training Balance (G/D)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('GAN Training Progress', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        plt.show()
    
    def generate_class_samples(
        self, 
        target_class: int,
        n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic samples for a specified class.
        
        This is the primary use case for BAGAN: generate synthetic samples
        to balance an imbalanced dataset. Can target any class.
        
        Args:
            target_class: Class label to generate samples for (0 to n-1)
            n_samples: Number of samples to generate
        
        Returns:
            Tuple of:
                - images: Generated images [n_samples, 1, 64, 64]
                - labels: Class labels (all target_class) [n_samples]
        
        Example:
            >>> # Generate 520 samples for class 2 to balance the dataset
            >>> fake_images, fake_labels = trainer.generate_class_samples(2, 520)
            >>> # Combine with real data
            >>> balanced_dataset = torch.cat([real_images, fake_images])
        """
        self.generator.eval()
        
        with torch.no_grad():
            z = self.latent_generator.sample(class_label=target_class, batch_size=n_samples)
            fake_images = self.generator(z)
            labels = torch.full((n_samples,), target_class, 
                               dtype=torch.long, device=self.device)
        
        self.generator.train()
        
        return fake_images, labels

    # Backward-compatible alias (deprecated)
    generate_minority_samples = generate_class_samples


# =============================================================================
# Module-level convenience function
# =============================================================================

def create_gan_trainer(
    generator: BAGANGenerator,
    discriminator: BAGANDiscriminator,
    latent_generator: ClassConditionalLatentGenerator,
    train_loader: DataLoader,
    config: Optional[BAGANConfig] = None,
    **kwargs
) -> GANTrainer:
    """
    Factory function to create a GANTrainer.
    
    Args:
        generator: BAGANGenerator instance
        discriminator: BAGANDiscriminator instance
        latent_generator: Fitted ClassConditionalLatentGenerator
        train_loader: Training data loader
        config: BAGANConfig instance. If None, uses default.
        **kwargs: Additional arguments passed to GANTrainer
    
    Returns:
        GANTrainer instance
    """
    if config is None:
        config = BAGANConfig()
    return GANTrainer(config, generator, discriminator, latent_generator, train_loader, **kwargs)


# =============================================================================
# Test code (runs only when executed directly)
# =============================================================================
# $env:KMP_DUPLICATE_LIB_OK="TRUE"
if __name__ == "__main__":
    """
    Test the GANTrainer with synthetic data.
    
    This test verifies:
        1. Trainer initialization works correctly
        2. Training loop runs without errors
        3. D and G steps execute properly
        4. Checkpointing saves and loads properly
    """
    # Fix OpenMP library conflict error
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("\n" + "=" * 70)
    print("Testing GANTrainer")
    print("=" * 70)
    
    # =========================================================================
    # Setup: Create synthetic dataset and models
    # =========================================================================
    config = BAGANConfig(num_classes=3)  # Test with 3 classes (N-class generality)
    
    # Create synthetic dataset
    n_samples = 128
    batch_size = 32
    
    synthetic_images = torch.randn(n_samples, 1, 64, 64)
    synthetic_labels = torch.randint(0, config.num_classes, (n_samples,))
    
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(synthetic_images, synthetic_labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create models
    generator = BAGANGenerator(
        out_channels=config.img_channels,
        latent_dim=config.latent_dim
    )
    
    discriminator = BAGANDiscriminator(
        in_channels=config.img_channels,
        num_classes=config.num_classes
    )
    
    # Create and FIT latent generator (required before GAN training)
    latent_gen = ClassConditionalLatentGenerator(config)
    
    # Simulate fitted state
    latent_gen.means[0] = torch.randn(128, device=latent_gen.device)
    latent_gen.means[1] = torch.randn(128, device=latent_gen.device) + 2
    latent_gen.means[2] = torch.randn(128, device=latent_gen.device) + 4
    latent_gen.stds[0] = torch.tensor([1.0], device=latent_gen.device)
    latent_gen.stds[1] = torch.tensor([0.8], device=latent_gen.device)
    latent_gen.stds[2] = torch.tensor([0.9], device=latent_gen.device)
    latent_gen.n_samples = {0: 500, 1: 300, 2: 80}
    latent_gen.is_fitted = True
    
    # =========================================================================
    # Test 1: Trainer Initialization
    # =========================================================================
    print("\nTest 1: Initialize trainer")
    trainer = GANTrainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        latent_generator=latent_gen,
        train_loader=train_loader
    )
    print("  ✓ Trainer initialized successfully")
    
    # =========================================================================
    # Test 2: Training Loop
    # =========================================================================
    print("\nTest 2: Train for 2 epochs")
    history = trainer.fit(epochs=2, verbose=True, save_every=0, visualize_every=0)
    print(f"  ✓ Training completed")
    print(f"  ✓ D losses: {[f'{l:.4f}' for l in history['d_loss']]}")
    print(f"  ✓ G losses: {[f'{l:.4f}' for l in history['g_loss']]}")
    
    # =========================================================================
    # Test 3: Sample Generation
    # =========================================================================
    print("\nTest 3: Generate class samples")
    fake_images, fake_labels = trainer.generate_class_samples(target_class=2, n_samples=8)
    print(f"  ✓ Generated images shape: {fake_images.shape}")
    print(f"  ✓ Labels: {fake_labels.tolist()}")
    
    # =========================================================================
    # Test 4: Checkpoint Save/Load
    # =========================================================================
    print("\nTest 4: Checkpoint save/load")
    test_g_path = "saved/test_gan_g.pth"
    test_d_path = "saved/test_gan_d.pth"
    trainer.save_checkpoint(path_g=test_g_path, path_d=test_d_path)
    
    # Create new trainer and load
    new_trainer = GANTrainer(
        config=config,
        generator=BAGANGenerator(out_channels=config.img_channels, latent_dim=config.latent_dim),
        discriminator=BAGANDiscriminator(in_channels=config.img_channels, num_classes=config.num_classes),
        latent_generator=latent_gen,
        train_loader=train_loader
    )
    new_trainer.load_checkpoint(path_g=test_g_path, path_d=test_d_path)
    print(f"  ✓ Checkpoint saved and loaded successfully")
    print(f"  ✓ Resumed from epoch: {new_trainer.current_epoch}")
    
    # Clean up test checkpoints
    if os.path.exists(test_g_path):
        os.remove(test_g_path)
    if os.path.exists(test_d_path):
        os.remove(test_d_path)
    print(f"  ✓ Test checkpoints cleaned up")
    
    # =========================================================================
    # All Tests Passed
    # =========================================================================
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
