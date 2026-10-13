"""
BAGAN Class-Conditional Latent Space Generator

This module implements the KEY INNOVATION of BAGAN (Mariani et al., 2018):
    - Learn class-specific distributions in the autoencoder's latent space
    - Sample from these distributions to generate class-conditional fake samples
    - Initialize GAN generator with semantically meaningful latent codes

Implementation: ISOTROPIC COVARIANCE (Option C)
    
    For each class c:
        z ~ N(μ_c, σ_c² I)
    
    Where:
        - μ_c = mean latent vector for class c (shape: [latent_dim])
        - σ_c = scalar standard deviation for class c (1 parameter)
        - I = identity matrix

Why Isotropic Covariance?
    Given only ~80 minority samples and 128-dimensional latent space:
    - Full Covariance: 8,256 parameters → 0.01 samples/param (IMPOSSIBLE)
    - Diagonal Covariance: 128 parameters → 0.625 samples/param (RISKY)
    - Isotropic Covariance: 1 parameter → 80 samples/param (EXCELLENT)
    
    Isotropic covariance is the ONLY statistically sound choice for extreme data scarcity.

Usage:
    >>> from bagan.training import ClassConditionalLatentGenerator
    >>> from bagan.config import BAGANConfig
    >>> 
    >>> config = BAGANConfig()
    >>> latent_gen = ClassConditionalLatentGenerator(config)
    >>> 
    >>> # Fit on training data using pretrained encoder
    >>> latent_gen.fit(encoder, train_loader)
    >>> 
    >>> # Sample for specific class
    >>> z = latent_gen.sample(class_label=1, batch_size=32)  # Minority class samples
    >>> 
    >>> # Sample balanced across classes
    >>> z, labels = latent_gen.sample_balanced(batch_size=64)  # Equal per class
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from collections import defaultdict
import numpy as np
import os
import sys

# Fix import path for both module import and direct execution
if __name__ == "__main__":
    # When running directly, add parent directories to path
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from bagan.config import BAGANConfig
else:
    # When imported as module, use relative import
    from ..config import BAGANConfig

    # Relative Import Explanation: from ..config import BAGANConfig
    # ================================================================
    # 
    # The ".." notation means "go up one directory level from current module"
    # 
    # Current file location:
    #   bagan/training/latent_space.py  ← We are here
    # 
    # Import resolution:
    #   from ..config import BAGANConfig
    #        ││
    #        │└─ First dot:  current package (bagan/training/)
    #        └── Second dot: go up one level → parent package (bagan/)
    # 
    # So ".." means: bagan/training/ → bagan/
    # Then "config" means: bagan/config.py
    # Finally "import BAGANConfig" imports the BAGANConfig class from that file
    # 
    # Directory structure:
    #   bagan/
    #   ├── __init__.py
    #   ├── config.py           ← Target file (contains BAGANConfig class)
    #   ├── models/
    #   │   ├── __init__.py
    #   │   ├── autoencoder.py
    #   │   └── ...
    #   └── training/
    #       ├── __init__.py
    #       └── latent_space.py ← Current file
    # 
    # Equivalent absolute import (when bagan/ is in sys.path):
    #   from bagan.config import BAGANConfig
    # 
    # Why use relative imports?
    #   ✓ Package-relative: Works regardless of where bagan/ is installed
    #   ✓ Refactoring-friendly: Rename package without breaking internal imports
    #   ✓ Avoids circular import issues in complex packages
    #   ✓ Clearer intent: Shows this is an internal package import
    # 
    # Dot notation reference:
    #   .config    → bagan/training/config.py (same directory - doesn't exist)
    #   ..config   → bagan/config.py (parent directory) ✓ This is what we use
    #   ...config  → parent's parent (would be outside bagan/) ✗ Error
    


class ClassConditionalLatentGenerator:
    """
    Generate class-conditional latent vectors for BAGAN.
    
    This class learns per-class distributions in the latent space of a pretrained
    autoencoder. During GAN training, it samples from these distributions to
    generate class-specific fake samples.
    
    Mathematical Formulation (Isotropic Covariance):
        For each class c ∈ {0, 1, ..., num_classes-1}:
            μ_c = (1/N_c) Σ_i z_i    where z_i belongs to class c
            σ_c² = (1/N_c·d) Σ_i ||z_i - μ_c||²    (isotropic: single scalar)
            
        Sampling:
            z ~ N(μ_c, σ_c² I)  →  z = μ_c + σ_c · ε, where ε ~ N(0, I)
    
    Attributes:
        config: BAGANConfig instance with hyperparameters
        num_classes: Number of real classes (excluding 'fake')
        latent_dim: Dimensionality of latent space
        device: Torch device (cuda/cpu)
        means: Dict mapping class_label → mean vector μ_c (shape: [latent_dim])
        stds: Dict mapping class_label → scalar std σ_c (single float)
        n_samples: Dict mapping class_label → number of samples used for estimation
        is_fitted: Boolean indicating if fit() has been called
    
    Example:
        >>> latent_gen = ClassConditionalLatentGenerator(config)
        >>> latent_gen.fit(encoder, train_loader)
        >>> 
        >>> # Generate 32 samples for a target class
        >>> z = latent_gen.sample(class_label=target_class, batch_size=32)
        >>> fake_images = generator(z)  # Pass to generator
    """
    
    def __init__(self, config: BAGANConfig):
        """
        Initialize the class-conditional latent generator.
        
        Args:
            config: BAGANConfig instance containing:
                - latent_dim: Dimensionality of latent space (default: 128)
                - num_classes: Number of real classes (default: 2)
                - device: Torch device string (default: auto-detect)
                - latent_regularization: Minimum std for numerical stability
        """
        self.config = config
        self.num_classes = config.num_classes
        self.latent_dim = config.latent_dim
        self.device = torch.device(config.device)
        self.eps = config.latent_regularization  # For numerical stability
        
        # Per-class statistics (to be computed in fit())
        self.means: Dict[int, torch.Tensor] = {}  # μ_c for each class
        self.stds: Dict[int, torch.Tensor] = {}   # σ_c (scalar) for each class
        self.n_samples: Dict[int, int] = {}       # Sample count per class
        # Encoded latents of the REAL samples, kept per class so that
        # ``sample_latent_smote`` can interpolate between them. Held on CPU;
        # only the small minority classes are ever drawn from in practice.
        self.class_latents: Dict[int, torch.Tensor] = {}
        
        self.is_fitted = False
        
        print("=" * 70)
        print("ClassConditionalLatentGenerator initialized")
        print("=" * 70)
        print(f"  Covariance type: ISOTROPIC (σ_c² I)")
        print(f"  Latent dimension: {self.latent_dim}")     # 128 (default)
        print(f"  Number of classes: {self.num_classes}")   # N (from config)
        print(f"  Device: {self.device}")                   # auto-detected
        print(f"  Parameters per class: 1 (single σ_c)")
        print(f"  Numerical stability eps: {self.eps}")     # latent_regularization (default 1e-5)
        print("=" * 70)
    
    @torch.no_grad() # Disable gradient computation for efficiency
    def fit(
        self, 
        encoder: nn.Module, 
        dataloader: torch.utils.data.DataLoader,
        use_vae_mean: bool = False  # VAE support: Future-proofing (default=False for standard AE)
    ) -> 'ClassConditionalLatentGenerator':
        """
        Fit class-conditional distributions from encoded training data.
        
        This method:
        1. Encodes all training images through the pretrained encoder
        2. Groups latent vectors by class label
        3. Computes per-class statistics (mean μ_c, isotropic std σ_c)
        
        Args:
            encoder: Pretrained encoder network (from autoencoder)
                     Should have forward() that returns latent vector z
                     
                     BAGAN Implementation Note:
                     ---------------------------
                     Current BAGAN uses STANDARD AUTOENCODER (not VAE).
                     
                     However, this code supports both architectures for flexibility:
                     
                     Standard Autoencoder (current):
                       - encoder(x) returns: z  (single tensor)
                       - use_vae_mean: ignored (leave as default False)
                     
                     Variational Autoencoder (optional future extension):
                       - encoder(x) returns: (z, mu, logvar)  (tuple of 3 tensors)
                       - use_vae_mean=True: use mu (mean) for statistics (more stable)
                       - use_vae_mean=False: use z (sampled) for statistics
                       
                     Why include VAE support?
                       ✓ Future-proofing: Easy to experiment with VAE initialization
                       ✓ Code reusability: Same class works for both architectures
                       ✓ No overhead: Standard AE path is simple (no tuple unpacking)
                       
            dataloader: Training data loader yielding (images, labels) tuples
            
            use_vae_mean: [Default: False] Only relevant for VAE encoders.
                         - False: Use sampled z for computing class statistics (default)
                         - True: Use deterministic mu (mean) instead of stochastic z
                         
                         ⚠️ For standard autoencoder (current BAGAN): This parameter is ignored.
                         The encoder returns a single tensor z, not (z, mu, logvar).
        
        Returns:
            self: For method chaining
        
        Note:
            - Encoder should be in eval mode (this method ensures it)
            - All computation is done without gradients
            - Statistics are stored on the configured device
        """
        print("\n" + "=" * 70)
        print("Fitting Class-Conditional Latent Distributions")
        print("=" * 70)
        
        # Ensure encoder is in eval mode
        encoder.eval()
        encoder = encoder.to(self.device)
        
        # Collect latent vectors grouped by class
        latents_by_class: Dict[int, list] = defaultdict(list)
        total_samples = 0
        
        print("Encoding training data...")
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Encode images to latent space
            output = encoder(images) # 128-dim latent vectors
            
            # Handle different encoder output formats
            # ----------------------------------------
            # BAGAN uses standard autoencoder → encoder returns single tensor z
            # VAE support included for future flexibility → encoder returns (z, mu, logvar) : logvar (log variance)
            if isinstance(output, tuple):
                # VAE encoder returns (z, mu, logvar)
                z, mu, logvar = output
                if use_vae_mean:
                    z = mu  # Use mean for more stable statistics
            else:
                # Standard autoencoder returns just z (BAGAN current implementation)
                z = output
            
            # Group by class label
            for i, label in enumerate(labels): # labels shape: [batch_size]
                class_idx = label.item()
                latents_by_class[class_idx].append(z[i].cpu())  # Move to CPU for aggregation
                # Since z is the batch of encoded latent vectors with shape [batch_size, latent_dim] (e.g., [32, 128]):
                # z[i] extracts the i-th latent vector from the batch
                # Shape: [latent_dim] → [128] (a single 128-dimensional vector)

            total_samples += len(labels)
            
            # Progress indicator every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches ({total_samples} samples)")
        
        print(f"\nTotal samples encoded: {total_samples}")
        
        # Compute per-class statistics
        print("\nComputing per-class statistics (Isotropic Covariance):")
        print("-" * 70)
        print(f"{'Class':<8} {'Samples':<10} {'Mean Norm':<15} {'Std (σ_c)':<15} {'Status':<10}")
        print("-" * 70)
        
        # Covariance model (default 'isotropic' preserves the original single-σ
        # behaviour; 'shrinkage'/'diagonal' give a per-dimension σ vector that
        # is far more robust for small minority classes — see BAGANGPConfig).
        cov_type = getattr(self.config, "latent_cov_type", "isotropic")
        shrink_lambda = float(getattr(self.config, "latent_shrinkage", 0.5))
        if cov_type == "diagonal":
            shrink_lambda = 0.0  # full per-dimension variance, no shrink
        if cov_type != "isotropic":
            print(f"  Covariance model: {cov_type.upper()} "
                  f"(λ={shrink_lambda:.2f}; 1=isotropic, 0=diagonal)")
        
        for class_idx in range(self.num_classes):
            if class_idx not in latents_by_class or len(latents_by_class[class_idx]) == 0:
                # No samples for this class - use standard normal as fallback
                print(f"{class_idx:<8} {'0':<10} {'N/A':<15} {'1.0 (default)':<15} {'⚠️ Fallback':<10}")
                self.means[class_idx] = torch.zeros(self.latent_dim, device=self.device) # 128-dim zero vector with value 0.0
                self.stds[class_idx] = torch.ones(1, device=self.device)    # 1-dim tensor with value 1.0
                self.n_samples[class_idx] = 0
                self.class_latents[class_idx] = torch.empty(0, self.latent_dim)
                continue
            
            # Stack all latent vectors for this class
            class_latents = torch.stack(latents_by_class[class_idx])  # Shape: [N_c, latent_dim]
            n_c = class_latents.shape[0] # number of samples in class c
            # Keep the encoded real latents (CPU) for latent-SMOTE seeding.
            self.class_latents[class_idx] = class_latents.detach().cpu()
            
            # Compute mean μ_c
            mu_c = class_latents.mean(dim=0)  # Shape: [latent_dim]
            
            # Compute isotropic standard deviation σ_c
            # σ_c² = (1 / (N_c * d)) * Σ_i ||z_i - μ_c||²
            # This is the average squared distance from mean, divided by dimension
            centered = class_latents - mu_c  # Shape: [N_c, latent_dim]
            
            if n_c == 1:
                # Single sample - use default std
                sigma_c = torch.ones(1)  # Default std for single sample
                status = "⚠️ Single"
            elif cov_type == "isotropic":
                # Isotropic: single variance parameter
                # Variance = average of all squared deviations
                variance = (centered ** 2).mean()  # Scalar
                sigma_c = variance.sqrt()
                
                # Numerical stability: ensure minimum std
                sigma_c = torch.clamp(sigma_c, min=self.eps)
                status = "✓ OK"
            else:
                # Shrinkage / diagonal: per-dimension variance shrunk toward the
                # isotropic target (Ledoit–Wolf style). Robust for small N_c:
                #   var_shrunk_j = (1-λ)·var_j + λ·mean_j(var_j)
                #   σ_c = sqrt(var_shrunk)   (shape [latent_dim])
                var_dim = (centered ** 2).mean(dim=0)            # [latent_dim]
                var_iso = var_dim.mean()                         # scalar target
                var_shrunk = (1.0 - shrink_lambda) * var_dim + shrink_lambda * var_iso
                sigma_c = var_shrunk.sqrt()
                sigma_c = torch.clamp(sigma_c, min=self.eps)     # [latent_dim]
                status = "✓ shrink"
            
            # Store on device
            self.means[class_idx] = mu_c.to(self.device)
            self.stds[class_idx] = sigma_c.to(self.device)
            self.n_samples[class_idx] = n_c
            
            # Print statistics
            mean_norm = mu_c.norm().item()
            # what does norm do here? It computes the Euclidean norm (L2 norm) of the mean vector μ_c.
             # The norm is calculated as:
            # ||μ_c|| = sqrt(Σ_j (μ_c[j])²)
            print(f"{class_idx:<8} {n_c:<10} {mean_norm:<15.4f} {sigma_c.mean().item():<15.6f} {status:<10}")
        
        print("-" * 70)
        
        # =====================================================================
        # N-Class Pairwise Separation Analysis
        # =====================================================================
        # For N classes we compute C(N,2) = N*(N-1)/2 pairwise separation ratios.
        # Each ratio measures how well two classes are separated in latent space:
        #
        #   ratio(i,j) = ||μ_i - μ_j|| / σ_avg(i,j)
        #
        # where σ_avg(i,j) = (σ_i + σ_j) / 2
        #
        # Interpretation:
        #   ratio > 2  → well-separated (< 5% overlap under Gaussian assumption)
        #   ratio ≤ 2  → significant overlap (may affect generation quality)
        #
        # A summary (min / mean / max) gives a quick overview of overall
        # latent space quality across all class pairs.
        # =====================================================================
        n_pairs = self.num_classes * (self.num_classes - 1) // 2  # C(N,2)
        ratios = []  # Collect all pairwise ratios for summary statistics
        
        print(f"\nClass Separation Analysis ({n_pairs} pair{'s' if n_pairs != 1 else ''}):")
        print(f"  {'Pair':<12} {'||Δμ||':<12} {'σ_avg':<12} {'Ratio':<10} {'Status'}")
        print(f"  {'-'*56}")
        
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                mu_diff = (self.means[i] - self.means[j]).norm().item()
                avg_std = (self.stds[i].mean().item() + self.stds[j].mean().item()) / 2
                ratio = mu_diff / avg_std if avg_std > 0 else float('inf')
                ratios.append(ratio)
                
                status = "✓ separated" if ratio > 2 else "⚠ overlap"
                print(f"  {i} ↔ {j:<8} {mu_diff:<12.4f} {avg_std:<12.6f} {ratio:<10.2f} {status}")
        
        # Summary statistics across all pairs
        if ratios:
            min_r, mean_r, max_r = min(ratios), np.mean(ratios), max(ratios)
            print(f"\n  Summary — Min / Mean / Max ratio: {min_r:.2f} / {mean_r:.2f} / {max_r:.2f}")
            
            if min_r > 2:
                print(f"  → All classes are well-separated in latent space ✓")
            else:
                n_overlap = sum(1 for r in ratios if r <= 2)
                print(f"  → {n_overlap}/{n_pairs} pair(s) have significant overlap")
        
        self.is_fitted = True
        print("\n" + "=" * 70)
        print("Fitting complete! Ready for class-conditional sampling.")
        print("=" * 70)
        
        return self
    
    def sample(
        self, 
        class_label: int, 
        batch_size: int,
        return_mean: bool = False
    ) -> torch.Tensor:
        """
        Sample latent vectors for a specific class.
        
        Sampling formula (Isotropic Covariance):
            z = μ_c + σ_c · ε,  where ε ~ N(0, I)
        
        Args:
            class_label: Target class to sample (0 to num_classes-1)
            batch_size: Number of samples to generate
            return_mean: If True, return μ_c without noise (for visualization)
        
        Returns:
            z: Latent vectors of shape [batch_size, latent_dim]
        
        Raises:
            RuntimeError: If fit() has not been called
            ValueError: If class_label is invalid
        """
        if not self.is_fitted:
            raise RuntimeError("ClassConditionalLatentGenerator not fitted. Call fit() first.")
        
        if class_label not in self.means:
            raise ValueError(f"Invalid class_label: {class_label}. Valid labels: {list(self.means.keys())}")
        
        mu_c = self.means[class_label]      # Shape: [latent_dim]
        sigma_c = self.stds[class_label]    # Scalar
        
        if return_mean:
            # Return mean without noise (useful for visualization)
            return mu_c.unsqueeze(0).expand(batch_size, -1)
        
        # Sample: z = μ_c + σ_c · ε
        epsilon = torch.randn(batch_size, self.latent_dim, device=self.device)
        z = mu_c.unsqueeze(0) + sigma_c * epsilon  # Broadcasting: [1, d] + scalar * [B, d]
        
        # Broadcasting Explanation:
        # ------------------------
        # mu_c shape:              [latent_dim]           e.g., [128]
        # mu_c.unsqueeze(0) shape: [1, latent_dim]        e.g., [1, 128]
        # sigma_c shape:           scalar or [1]          e.g., 0.8
        # epsilon shape:           [batch_size, latent_dim]  e.g., [32, 128]
        # 
        # Step 1: sigma_c * epsilon
        #   scalar * [B, d] → [B, d]
        #   The scalar σ_c is multiplied element-wise with every element in epsilon
        #   Result: [32, 128] (each of the 32×128 values scaled by σ_c)
        #
        # Step 2: mu_c.unsqueeze(0) + (sigma_c * epsilon)
        #   [1, d] + [B, d] → Broadcasting automatically expands [1, d] to [B, d]
        #   [1, 128] + [32, 128] → PyTorch repeats the first row 32 times
        #   
        #   Conceptually:
        #   [[μ_c]]          [[μ_c],          [[μ_c + σ·ε₁],
        #      +        →     [μ_c],    +      [μ_c + σ·ε₂],
        #   [σ·ε₁,            ...             ...
        #    σ·ε₂,            [μ_c]]           [μ_c + σ·ε₃₂]]
        #    ...
        #    σ·ε₃₂]
        #
        # Final z shape: [batch_size, latent_dim] → [B, d] → [32, 128] ✓
        # Each row is an independent sample: z[i] = μ_c + σ_c * ε[i]
        
        return z
    
    def sample_balanced(
        self, 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample balanced latent vectors from all classes.
        
        Generates approximately equal samples from each class, ensuring
        that the minority class is well-represented in each batch.
        
        Args:
            batch_size: Total number of samples (will be distributed across classes)
        
        Returns:
            z: Latent vectors of shape [batch_size, latent_dim]
            labels: Class labels of shape [batch_size]
        
        Example:
            >>> z, labels = latent_gen.sample_balanced(batch_size=64)
            >>> # z.shape: [64, 128]
            >>> # labels: balanced across all num_classes classes
        """
        if not self.is_fitted:
            raise RuntimeError("ClassConditionalLatentGenerator not fitted. Call fit() first.")
        
        samples_per_class = batch_size // self.num_classes
        remainder = batch_size % self.num_classes
        
        z_list = []
        labels_list = []
        
        for class_idx in range(self.num_classes):
            # Handle remainder by giving extra sample to first classes
            n_samples = samples_per_class + (1 if class_idx < remainder else 0)
            
            z_class = self.sample(class_idx, n_samples)
            z_list.append(z_class)
            labels_list.append(torch.full((n_samples,), class_idx, 
                                          dtype=torch.long, device=self.device))
        
        z = torch.cat(z_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        # Shuffle to mix classes
        perm = torch.randperm(batch_size, device=self.device)
        z = z[perm]
        labels = labels[perm]
        
        return z, labels
    
    def sample_for_class_oversampling(
        self, 
        target_class: int,
        n_samples: int
    ) -> torch.Tensor:
        """
        Sample latent vectors specifically for class oversampling.
        
        This is the primary use case for BAGAN: generate synthetic samples
        for underrepresented classes to balance the training data.
        
        Args:
            target_class: Class to oversample (0 to num_classes-1)
            n_samples: Number of samples to generate
        
        Returns:
            z: Latent vectors of shape [n_samples, latent_dim]
        
        Example:
            >>> # Generate synthetic samples to balance any minority class
            >>> z = latent_gen.sample_for_class_oversampling(
            ...     target_class=minority_class_idx,
            ...     n_samples=520   # To reach balanced count
            ... )
            >>> synthetic_images = generator(z)
        """
        return self.sample(target_class, n_samples)
    
    def sample_latent_smote(
        self,
        target_class: int,
        n_samples: int,
        noise_frac: float = 0.25,
    ) -> torch.Tensor:
        """Seed latents by SMOTE-style interpolation of encoded REAL latents.

        Mirrors ``casestudy/methods/m10_ab8c.py`` (``seed_mode="latent_smote"``).
        For a target class ``c`` with encoded real latents
        :math:`\\{z_1 \\dots z_{N_c}\\}`, each sample is

        .. math::
            z = z_i + \\alpha (z_j - z_i) + \\epsilon,\\quad
            \\alpha \\sim U(0, 1),\\quad
            \\epsilon \\sim \\mathcal{N}(0, (\\text{noise\\_frac}\\cdot\\sigma_c)^2)

        with ``i``, ``j`` drawn uniformly from the class's real latents.

        Compared with sampling the fitted prior :math:`\\mathcal{N}(\\mu_c,
        \\sigma_c^2 I)`, this stays closer to the observed minority manifold —
        the prior can place mass in low-density regions that the decoder maps to
        off-manifold images. It needs at least one real latent for the class; if
        the class had no samples at fit time it falls back to the prior.

        Args:
            target_class: Class to seed for (0 to num_classes-1).
            n_samples: Number of latent vectors to produce.
            noise_frac: Jitter as a fraction of the class std :math:`\\sigma_c`.

        Returns:
            ``z`` of shape ``[n_samples, latent_dim]`` on ``self.device``.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "sample_latent_smote() requires a fitted generator; call fit() first."
            )

        reals = self.class_latents.get(target_class)
        if reals is None or reals.shape[0] == 0:
            # No real latents recorded for this class - prior is the only option.
            return self.sample(target_class, n_samples)

        reals = reals.to(self.device)
        n_c = reals.shape[0]

        if n_c == 1:
            # Interpolation is degenerate with a single anchor; jitter it instead.
            base = reals.expand(n_samples, -1)
        else:
            i = torch.randint(0, n_c, (n_samples,), device=self.device)
            j = torch.randint(0, n_c, (n_samples,), device=self.device)
            alpha = torch.rand(n_samples, 1, device=self.device)
            base = reals[i] + alpha * (reals[j] - reals[i])

        sigma_c = self.stds[target_class].to(self.device)
        # sigma_c is a scalar for the isotropic model and [latent_dim] for the
        # shrinkage/diagonal model; both broadcast against [n_samples, d].
        noise = torch.randn(n_samples, self.latent_dim, device=self.device)
        return base + noise_frac * sigma_c * noise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get fitted statistics for inspection and debugging.
        
        Returns:
            Dict containing:
                - 'means': Dict of class means
                - 'stds': Dict of class standard deviations
                - 'n_samples': Dict of sample counts per class
                - 'is_fitted': Boolean indicating fit status
        """
        return {
            'means': {k: v.cpu().numpy() for k, v in self.means.items()},
            'stds': {k: (v.cpu().item() if v.numel() == 1 else v.cpu().numpy())
                     for k, v in self.stds.items()},
            'n_samples': self.n_samples.copy(),
            'is_fitted': self.is_fitted,
            'covariance_type': getattr(self.config, 'latent_cov_type', 'isotropic')
        }
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save fitted parameters to disk.
        
        Args:
            path: Save path. If None, uses config.latent_params_path
        
        Returns:
            str: Path where parameters were saved
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted generator. Call fit() first.")
        
        if not self.config.save_checkpoints:
            print("Latent param saving disabled (save_checkpoints=False)")
            return ""
        
        save_path = path or self.config.latent_params_path
        
        state = {
            'means': {k: v.cpu() for k, v in self.means.items()},
            'stds': {k: v.cpu() for k, v in self.stds.items()},
            'n_samples': self.n_samples,
            'config': {
                'num_classes': self.num_classes,
                'latent_dim': self.latent_dim,
                'eps': self.eps
            },
            'covariance_type': 'isotropic'
        }
        
        torch.save(state, save_path)
        print(f"Latent parameters saved to: {save_path}")
        
        return save_path
    
    def load(self, path: Optional[str] = None) -> 'ClassConditionalLatentGenerator':
        """
        Load fitted parameters from disk.
        
        Args:
            path: Load path. If None, uses config.latent_params_path
        
        Returns:
            self: For method chaining
        """
        load_path = path or self.config.latent_params_path
        
        # weights_only=False: Allow loading full state dict with nested Python objects
        # Required because state contains dicts of tensors and Python objects
        # Note: Only use weights_only=False for files you trust (your own saves)
        state = torch.load(load_path, map_location=self.device, weights_only=False)
        
        # Validate configuration match
        if state['config']['latent_dim'] != self.latent_dim:
            raise ValueError(f"Latent dim mismatch: saved={state['config']['latent_dim']}, "
                           f"current={self.latent_dim}")
        
        # Restore state
        self.means = {k: v.to(self.device) for k, v in state['means'].items()}
        self.stds = {k: v.to(self.device) for k, v in state['stds'].items()}
        self.n_samples = state['n_samples']
        self.is_fitted = True
        
        print(f"Latent parameters loaded from: {load_path}")
        print(f"  Covariance type: {state.get('covariance_type', 'unknown')}")
        print(f"  Classes: {list(self.means.keys())}")
        print(f"  Samples per class: {self.n_samples}")
        
        return self
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"ClassConditionalLatentGenerator("
                f"num_classes={self.num_classes}, "
                f"latent_dim={self.latent_dim}, "
                f"covariance=isotropic, "
                f"status={status})")


# =============================================================================
# Module-level convenience functions
# =============================================================================

def create_latent_generator(config: Optional[BAGANConfig] = None) -> ClassConditionalLatentGenerator:
    """
    Factory function to create a ClassConditionalLatentGenerator.
    
    Args:
        config: BAGANConfig instance. If None, uses default config.
    
    Returns:
        ClassConditionalLatentGenerator instance
    """
    if config is None:
        config = BAGANConfig()
    return ClassConditionalLatentGenerator(config)


# =============================================================================
# Test code (runs only when executed directly)
# =============================================================================

if __name__ == "__main__":
    # Suppress OpenMP warnings for testing
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("\n" + "=" * 70)
    print("Testing ClassConditionalLatentGenerator (N-Class)")
    print("=" * 70)
    
    # Create config with 3 classes to test N-class behaviour
    config = BAGANConfig(num_classes=3)
    latent_gen = ClassConditionalLatentGenerator(config)
    
    # Simulate fitted state for testing with 3 classes
    # Each class gets a distinct offset so pairwise separations are measurable
    latent_gen.means[0] = torch.randn(128, device=latent_gen.device)
    latent_gen.means[1] = torch.randn(128, device=latent_gen.device) + 2   # Offset +2
    latent_gen.means[2] = torch.randn(128, device=latent_gen.device) - 1.5 # Offset -1.5
    latent_gen.stds[0] = torch.tensor([1.0], device=latent_gen.device)
    latent_gen.stds[1] = torch.tensor([0.8], device=latent_gen.device)
    latent_gen.stds[2] = torch.tensor([0.9], device=latent_gen.device)
    latent_gen.n_samples = {0: 720, 1: 80, 2: 300}
    latent_gen.is_fitted = True
    
    # Test sampling from each class
    for c in range(config.num_classes):
        print(f"\nTest: Sample from class {c}")
        z = latent_gen.sample(class_label=c, batch_size=4)
        print(f"  Shape: {z.shape}")
        print(f"  Mean: {z.mean().item():.4f}")
    
    print("\nTest: Balanced sampling")
    z_bal, labels = latent_gen.sample_balanced(batch_size=9)  # 3 per class
    print(f"  z shape: {z_bal.shape}")
    print(f"  labels: {labels.tolist()}")
    
    print("\nTest: Statistics")
    stats = latent_gen.get_statistics()
    print(f"  n_samples: {stats['n_samples']}")
    print(f"  stds: {stats['stds']}")
    
    # Test N-class pairwise separation analysis
    # We call fit-internal logic manually by reusing the means/stds already set
    print("\nTest: Pairwise Separation (C(3,2) = 3 pairs)")
    n_pairs = config.num_classes * (config.num_classes - 1) // 2
    ratios = []
    for i in range(config.num_classes):
        for j in range(i + 1, config.num_classes):
            mu_diff = (latent_gen.means[i] - latent_gen.means[j]).norm().item()
            avg_std = (latent_gen.stds[i].item() + latent_gen.stds[j].item()) / 2
            ratio = mu_diff / avg_std if avg_std > 0 else float('inf')
            ratios.append(ratio)
            print(f"  {i} ↔ {j}: ||Δμ||={mu_diff:.4f}, ratio={ratio:.2f}")
    print(f"  Min/Mean/Max: {min(ratios):.2f} / {np.mean(ratios):.2f} / {max(ratios):.2f}")
    
    # Validate that num_discriminator_outputs property works correctly
    print(f"\nConfig check: num_classes={config.num_classes}, "
          f"num_discriminator_outputs={config.num_discriminator_outputs}")
    assert config.num_discriminator_outputs == config.num_classes + 1, \
        "num_discriminator_outputs should be num_classes + 1"
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
