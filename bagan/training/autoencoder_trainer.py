"""
BAGAN Autoencoder Trainer

This module provides the training loop for the BAGAN autoencoder (standard AE, not VAE).
The autoencoder is trained BEFORE GAN training to:
    1. Learn a meaningful latent representation of all training images
    2. Provide weight initialization for Generator (from Decoder) and Discriminator (from Encoder)
    3. Enable class-conditional latent sampling via ClassConditionalLatentGenerator

Loss Function:
    L = MSE(x, x_reconstructed) = (1/N) Σ ||x_i - Decoder(Encoder(x_i))||²
    
    BAGAN uses simple L2 loss (MSE) - NOT the VAE's combined MSE + KL divergence loss.
    Reference: Mariani et al. (2018) Section 4 - "we apply l2 loss minimization"

Training Strategy:
    - Train on ALL data (both majority and minority classes)
    - No class conditioning during autoencoder phase
    - Learns shared feature representation useful for both classes
    - Adam optimizer with β₁=0.5, β₂=0.999 (standard for image generation)

Usage:
    >>> from bagan.training import AutoencoderTrainer
    >>> from bagan.models import BAGANAutoencoder
    >>> from bagan.config import BAGANConfig
    >>> 
    >>> config = BAGANConfig()
    >>> autoencoder = BAGANAutoencoder(latent_dim=config.latent_dim)
    >>> trainer = AutoencoderTrainer(config, autoencoder, train_loader)
    >>> 
    >>> # Train the autoencoder
    >>> history = trainer.fit()
    >>> 
    >>> # Visualize reconstructions
    >>> trainer.visualize_reconstructions(sample_images)
    >>> 
    >>> # Save checkpoint
    >>> trainer.save_checkpoint()
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

# =============================================================================
# Import Handling for Both Module and Direct Execution
# =============================================================================
# This pattern allows the file to be:
#   1. Imported as part of the bagan package: from bagan.training import AutoencoderTrainer
#   2. Run directly for testing: python bagan/training/autoencoder_trainer.py

if __name__ == "__main__":
    # Direct execution: Add parent directories to Python's module search path
    # Path(__file__) = current file path (autoencoder_trainer.py)
    # .parent = bagan/training/
    # .parent.parent = bagan/
    # .parent.parent.parent = project root (where bagan/ folder lives)
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from bagan.config import BAGANConfig
    from bagan.models.autoencoder import BAGANAutoencoder
else:
    # Module import: Use relative imports (preferred for package code)
    # ".." goes up from bagan/training/ to bagan/
    from ..config import BAGANConfig
    from ..models.autoencoder import BAGANAutoencoder


class AutoencoderTrainer:
    """
    Trainer for BAGAN's autoencoder pre-training phase.
    
    This class encapsulates:
        - Training loop with MSE reconstruction loss
        - Adam optimizer with configurable hyperparameters
        - Learning rate scheduling (optional)
        - Training history tracking for visualization
        - Checkpoint save/load functionality
        - Reconstruction visualization
    
    The trained autoencoder serves two purposes in BAGAN:
        1. Weight Initialization: Encoder → Discriminator, Decoder → Generator
        2. Latent Space Learning: Encoder outputs used for class-conditional sampling
    
    Attributes:
        config (BAGANConfig): Configuration object with hyperparameters
        autoencoder (BAGANAutoencoder): The model to train
        train_loader (DataLoader): Training data iterator
        val_loader (Optional[DataLoader]): Validation data iterator
        optimizer (torch.optim.Adam): Adam optimizer instance
        scheduler (Optional): Learning rate scheduler
        device (torch.device): Training device (cuda/cpu)
        history (Dict): Training metrics history
        current_epoch (int): Current training epoch (for resuming)
    
    Example:
        >>> # Create trainer
        >>> trainer = AutoencoderTrainer(config, autoencoder, train_loader)
        >>> 
        >>> # Train for configured number of epochs
        >>> history = trainer.fit()
        >>> 
        >>> # Plot training curve
        >>> plt.plot(history['train_loss'])
        >>> plt.xlabel('Epoch')
        >>> plt.ylabel('MSE Loss')
        >>> plt.show()
    """
    
    def __init__(
        self,
        config: BAGANConfig,
        autoencoder: BAGANAutoencoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        use_scheduler: bool = True,
        scheduler_step_size: int = 20,
        scheduler_gamma: float = 0.5
    ):
        """
        Initialize the autoencoder trainer.
        
        Args:
            config: BAGANConfig instance with hyperparameters:
                - ae_epochs: Number of training epochs (default: 50)
                - lr_ae: Learning rate (default: 0.0002)
                - beta1: Adam β₁ parameter (default: 0.5)
                - beta2: Adam β₂ parameter (default: 0.999)
                - device: Training device string
                - autoencoder_path: Path for saving model
                
            autoencoder: BAGANAutoencoder instance to train
                - Should be freshly initialized (random weights)
                - Will be modified in-place during training
                
            train_loader: DataLoader yielding (images, labels) tuples
                - images: Tensor of shape [batch, 1, 64, 64] in range [-1, 1]
                - labels: Tensor of shape [batch] (not used during AE training)
                
            val_loader: Optional validation DataLoader
                - Same format as train_loader
                - Used for monitoring generalization (not for early stopping by default)
                
            use_scheduler: Whether to use learning rate scheduling
                - True: StepLR scheduler reduces LR by gamma every step_size epochs
                - False: Constant learning rate throughout training
                
            scheduler_step_size: Epochs between LR reductions (default: 20)
                - After this many epochs, LR is multiplied by gamma
                - Example: At step_size=20, gamma=0.5: LR halves at epochs 20, 40, ...
                
            scheduler_gamma: LR reduction factor (default: 0.5)
                - New LR = current LR × gamma
                - 0.5 means halving the learning rate
        
        Notes:
            - Autoencoder is moved to config.device automatically
            - Optimizer is created with config's Adam parameters
            - Training history is initialized empty
        """
        # =====================================================================
        # Store Configuration and Model
        # =====================================================================
        self.config = config
        self.autoencoder = autoencoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # =====================================================================
        # Device Setup
        # =====================================================================
        # torch.device() converts string like "cuda" to proper device object
        # This allows consistent device handling across the codebase
        self.device = torch.device(config.device)
        
        # Move autoencoder to device (GPU if available)
        # .to(device) returns self, but also modifies in-place for nn.Module
        self.autoencoder = self.autoencoder.to(self.device)
        
        # =====================================================================
        # Optimizer Setup: Adam with BAGAN-specific parameters
        # =====================================================================
        # Adam optimizer with β₁=0.5 is standard for GANs and autoencoders
        # Lower β₁ (0.5 vs default 0.9) provides less momentum, more responsive updates
        # This helps with the non-stationary nature of adversarial training
        #
        # Parameters explained:
        #   lr: Learning rate - step size for weight updates
        #   betas: (β₁, β₂) - exponential decay rates for moment estimates
        #       β₁: Decay for first moment (mean of gradients) - default 0.9, we use 0.5
        #       β₂: Decay for second moment (uncentered variance) - usually 0.999
        self.optimizer = optim.Adam(
            self.autoencoder.parameters(),     # All trainable parameters
            lr=config.lr_ae,                   # Learning rate (default: 0.0002)
            betas=(config.beta1, config.beta2)  # (0.5, 0.999)
        )
        
        # =====================================================================
        # Learning Rate Scheduler (Optional)
        # =====================================================================
        # StepLR: Reduces learning rate by gamma every step_size epochs
        # This helps fine-tune the model in later epochs with smaller updates
        #
        # Learning rate schedule example (step_size=20, gamma=0.5, initial_lr=0.0002):
        #   Epoch 0-19:  lr = 0.0002
        #   Epoch 20-39: lr = 0.0001 (0.0002 × 0.5)
        #   Epoch 40-49: lr = 0.00005 (0.0001 × 0.5)
        self.use_scheduler = use_scheduler
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_step_size,  # Epochs between LR decay
                gamma=scheduler_gamma           # Multiplicative factor
            )
        else:
            self.scheduler = None
        
        # =====================================================================
        # Training State and History
        # =====================================================================
        # current_epoch tracks progress for checkpoint resumption
        self.current_epoch = 0
        
        # history stores metrics for visualization and analysis
        # Dict structure allows easy extension with new metrics
        self.history: Dict[str, List[float]] = {
            'train_loss': [],      # MSE loss per epoch
            'val_loss': [],        # Validation loss per epoch (if val_loader provided)
            'learning_rate': [],   # LR per epoch (useful with scheduler)
        }
        
        # =====================================================================
        # Logging and Display
        # =====================================================================
        self._print_trainer_info()
    
    def _print_trainer_info(self) -> None:
        """Print trainer configuration summary."""
        print("\n" + "=" * 70)
        print("AutoencoderTrainer Initialized")
        print("=" * 70)
        print(f"  Model: BAGANAutoencoder")
        print(f"  Device: {self.device}")
        print(f"  Training epochs: {self.config.ae_epochs}")
        print(f"  Learning rate: {self.config.lr_ae}")
        print(f"  Adam betas: ({self.config.beta1}, {self.config.beta2})")
        print(f"  LR Scheduler: {'StepLR' if self.use_scheduler else 'None'}")
        
        # Handle custom DeviceDataLoader wrapper vs standard DataLoader
        # DeviceDataLoader stores the underlying DataLoader in .dl attribute
        if hasattr(self.train_loader, 'dl'):
            # Custom DeviceDataLoader wrapper
            underlying_loader = self.train_loader.dl
            num_samples = len(underlying_loader.dataset)
            batch_size = underlying_loader.batch_size
        elif hasattr(self.train_loader, 'dataset'):
            # Standard PyTorch DataLoader
            num_samples = len(self.train_loader.dataset)
            batch_size = self.train_loader.batch_size
        else:
            # Fallback: can't determine, use len()
            num_samples = "Unknown"
            batch_size = "Unknown"
        
        print(f"  Training samples: {num_samples}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches per epoch: {len(self.train_loader)}")
        
        if self.val_loader:
            if hasattr(self.val_loader, 'dl'):
                val_samples = len(self.val_loader.dl.dataset)
            elif hasattr(self.val_loader, 'dataset'):
                val_samples = len(self.val_loader.dataset)
            else:
                val_samples = "Unknown"
            print(f"  Validation samples: {val_samples}")
        
        print(f"  Save path: {self.config.autoencoder_path}")
        print("=" * 70)
    
    def fit(
        self,
        epochs: Optional[int] = None,
        verbose: bool = True,
        save_every: int = 10,
        visualize_every: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the autoencoder for the specified number of epochs.
        
        This is the main training loop that:
            1. Iterates over epochs
            2. Trains on all batches in train_loader
            3. Optionally validates on val_loader
            4. Saves checkpoints periodically
            5. Optionally visualizes reconstructions
        
        Training Flow:
            for epoch in range(epochs):
                1. Set model to training mode
                2. For each batch: compute loss, backward, optimizer step
                3. Track average epoch loss
                4. (Optional) Compute validation loss
                5. (Optional) Step learning rate scheduler
                6. (Optional) Save checkpoint
                7. (Optional) Visualize reconstructions
        
        Args:
            epochs: Number of epochs to train. If None, uses config.ae_epochs
            verbose: Whether to print progress every epoch
            save_every: Save checkpoint every N epochs (0 to disable)
            visualize_every: Show reconstructions every N epochs (0 to disable)
        
        Returns:
            history: Dictionary containing:
                - 'train_loss': List of training losses per epoch
                - 'val_loss': List of validation losses per epoch (if val_loader)
                - 'learning_rate': List of learning rates per epoch
        
        Example:
            >>> history = trainer.fit(epochs=50, save_every=10)
            >>> print(f"Final loss: {history['train_loss'][-1]:.4f}")
        """
        # Use config epochs if not specified
        epochs = epochs or self.config.ae_epochs
        
        print("\n" + "=" * 70)
        print(f"Starting Autoencoder Training: {epochs} epochs")
        print("=" * 70)
        
        # Track training time
        start_time = time.time()
        
        # =====================================================================
        # Main Training Loop
        # =====================================================================
        for epoch in range(self.current_epoch, epochs):
            epoch_start_time = time.time()
            
            # -----------------------------------------------------------------
            # Training Phase
            # -----------------------------------------------------------------
            # _train_epoch returns the average MSE loss over all batches
            train_loss = self._train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # -----------------------------------------------------------------
            # Validation Phase (Optional)
            # -----------------------------------------------------------------
            if self.val_loader is not None:
                val_loss = self._validate()
                self.history['val_loss'].append(val_loss)
            else:
                val_loss = None
            
            # -----------------------------------------------------------------
            # Learning Rate Scheduler Step
            # -----------------------------------------------------------------
            # Get current learning rate for logging
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Step scheduler at end of epoch (not batch)
            if self.scheduler is not None:
                self.scheduler.step()
            
            # -----------------------------------------------------------------
            # Progress Logging
            # -----------------------------------------------------------------
            epoch_time = time.time() - epoch_start_time
            
            if verbose:
                # Build progress message
                msg = f"Epoch [{epoch + 1:3d}/{epochs}] | "
                msg += f"Train Loss: {train_loss:.6f} | "
                if val_loss is not None:
                    msg += f"Val Loss: {val_loss:.6f} | "
                msg += f"LR: {current_lr:.6f} | "
                msg += f"Time: {epoch_time:.1f}s"
                print(msg)
            
            # -----------------------------------------------------------------
            # Checkpoint Saving
            # -----------------------------------------------------------------
            # save_every > 0 enables periodic saving
            # Also save at the final epoch
            if save_every > 0 and self.config.save_checkpoints and ((epoch + 1) % save_every == 0 or epoch == epochs - 1):
                self.save_checkpoint(epoch=epoch + 1)
            
            # -----------------------------------------------------------------
            # Reconstruction Visualization
            # -----------------------------------------------------------------
            if visualize_every > 0 and (epoch + 1) % visualize_every == 0:
                self._visualize_reconstructions_epoch(epoch + 1)
            
            # Update current epoch for checkpoint resumption
            self.current_epoch = epoch + 1
        
        # =====================================================================
        # Training Complete
        # =====================================================================
        total_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Final train loss: {self.history['train_loss'][-1]:.6f}")
        if self.history['val_loss']:
            print(f"  Final val loss: {self.history['val_loss'][-1]:.6f}")
        print(f"  Model saved to: {self.config.autoencoder_path}")
        print("=" * 70)
        
        return self.history
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch and return average loss.
        
        One epoch = one complete pass through all training data.
        
        Training Steps per Batch:
            1. Move data to device (GPU)
            2. Zero optimizer gradients
            3. Forward pass: x → Encoder → z → Decoder → x_recon
            4. Compute MSE loss: ||x - x_recon||²
            5. Backward pass: compute gradients
            6. Optimizer step: update weights
            7. Accumulate loss for averaging
        
        Returns:
            avg_loss: Average MSE loss over all batches
        
        Notes:
            - autoencoder is set to train() mode for dropout/batchnorm
            - Gradients are zeroed before each forward pass
            - Loss is averaged over the epoch for stability
        """
        # Set model to training mode
        # This affects layers like Dropout and BatchNorm:
        #   - Dropout: Randomly drops neurons (active)
        #   - BatchNorm: Uses batch statistics and updates running stats
        self.autoencoder.train()
        
        # Accumulate loss across all batches
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Iterate over all batches in training data
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # -----------------------------------------------------------------
            # Step 1: Move data to device
            # -----------------------------------------------------------------
            # images: [batch_size, 1, 64, 64] - grayscale FashionMNIST
            # labels: [batch_size] - class labels (not used in AE training)
            images = images.to(self.device)
            # Note: labels are not used because autoencoder is unconditional
            
            # -----------------------------------------------------------------
            # Step 2: Zero gradients
            # -----------------------------------------------------------------
            # PyTorch accumulates gradients by default
            # We need to zero them before each backward pass
            # Otherwise gradients from previous batches would accumulate
            self.optimizer.zero_grad()
            
            # -----------------------------------------------------------------
            # Step 3: Forward pass through autoencoder
            # -----------------------------------------------------------------
            # BAGANAutoencoder.forward() returns (x_recon, z)
            #   x_recon: Reconstructed image [batch, 1, 64, 64]
            #   z: Latent representation [batch, latent_dim]
            #   x -> Encoder -> z -> Decoder -> x_recon
            x_recon, z = self.autoencoder(images)
            
            # -----------------------------------------------------------------
            # Step 4: Compute reconstruction loss (MSE)
            # -----------------------------------------------------------------
            # MSE Loss = (1/N) Σ ||x_i - x_recon_i||²
            # Uses the static method from BAGANAutoencoder for consistency
            loss = BAGANAutoencoder.reconstruction_loss(images, x_recon)
            
            # -----------------------------------------------------------------
            # Step 5: Backward pass (compute gradients)
            # -----------------------------------------------------------------
            # loss.backward() computes ∂loss/∂θ for all parameters θ
            # Gradients are stored in parameter.grad attributes
            loss.backward()
            
            # -----------------------------------------------------------------
            # Step 6: Optimizer step (update weights)
            # -----------------------------------------------------------------
            # For Adam: θ = θ - lr * m / (√v + ε)
            # where m and v are the first and second moment estimates
            self.optimizer.step()
            
            # -----------------------------------------------------------------
            # Step 7: Accumulate loss for epoch average
            # -----------------------------------------------------------------
            # .item() converts single-element tensor to Python float
            # This also releases the computation graph for memory efficiency
            total_loss += loss.item()
        
        # Compute average loss over all batches
        avg_loss = total_loss / num_batches
        
        return avg_loss
    
    @torch.no_grad()  # Decorator that disables gradient computation for efficiency
    def _validate(self) -> float:
        """
        Compute validation loss without training.
        
        Validation is used to:
            - Monitor generalization (detect overfitting)
            - Compare model performance across epochs
            - NOT used for weight updates (no backward pass)
        
        The @torch.no_grad() decorator:
            - Disables gradient computation for all operations inside
            - Reduces memory usage (no gradient tensors stored)
            - Speeds up computation (no gradient tracking overhead)
        
        Returns:
            avg_loss: Average MSE loss over validation set
        """
        # Set model to evaluation mode
        # This affects layers like Dropout and BatchNorm:
        #   - Dropout: All neurons active (no dropping)
        #   - BatchNorm: Uses running statistics (not batch statistics)
        self.autoencoder.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            
            # Forward pass only (no backward needed)
            x_recon, z = self.autoencoder(images)
            loss = BAGANAutoencoder.reconstruction_loss(images, x_recon)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        return avg_loss
    
    def _visualize_reconstructions_epoch(self, epoch: int, n_samples: int = 8) -> None:
        """
        Visualize reconstructions during training.
        
        Shows a grid comparing original images with their reconstructions.
        This is useful for:
            - Monitoring training progress visually
            - Debugging poor reconstructions
            - Verifying the autoencoder is learning
        
        Args:
            epoch: Current epoch number (for title)
            n_samples: Number of image pairs to show
        """
        # Get a batch of images
        images, labels = next(iter(self.train_loader))
        images = images[:n_samples].to(self.device)
        
        # Generate reconstructions
        self.autoencoder.eval()
        with torch.no_grad():
            x_recon, z = self.autoencoder(images)
        
        # Move to CPU for plotting
        images = images.cpu()
        x_recon = x_recon.cpu()
        
        # Create visualization
        self._plot_reconstructions(
            images, x_recon, 
            title=f"Epoch {epoch} Reconstructions"
        )
    
    def visualize_reconstructions(
        self, 
        images: torch.Tensor,
        title: str = "Autoencoder Reconstructions",
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize reconstructions for given images.
        
        Creates a side-by-side comparison:
            Row 1: Original images
            Row 2: Reconstructed images
        
        Args:
            images: Input images [N, 1, 64, 64], normalized to [-1, 1]
            title: Plot title
            save_path: If provided, save figure to this path
        
        Example:
            >>> sample_images = next(iter(train_loader))[0][:8]
            >>> trainer.visualize_reconstructions(sample_images)
        """
        # Ensure model is in eval mode
        self.autoencoder.eval()
        
        # Move images to device and generate reconstructions
        images = images.to(self.device)
        with torch.no_grad():
            x_recon, z = self.autoencoder(images)
        
        # Move to CPU for plotting
        images = images.cpu()
        x_recon = x_recon.cpu()
        
        self._plot_reconstructions(images, x_recon, title, save_path)
    
    def _plot_reconstructions(
        self,
        originals: torch.Tensor,
        reconstructions: torch.Tensor,
        title: str = "Reconstructions",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create reconstruction comparison plot.
        
        Internal helper for visualization methods.
        
        Args:
            originals: Original images [N, 1, 64, 64]
            reconstructions: Reconstructed images [N, 1, 64, 64]
            title: Plot title
            save_path: Optional path to save figure
        """
        n_samples = originals.shape[0]
        
        # Create figure with 2 rows: originals on top, reconstructions below
        fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.5, 3))
        
        for i in range(n_samples):
            # Original images (top row)
            # Denormalize: [-1, 1] → [0, 1] for display
            img_orig = (originals[i].squeeze() + 1) / 2
            axes[0, i].imshow(img_orig, cmap='gray')
            axes[0, i].set_title("Original", fontsize=8)
            axes[0, i].axis('off')
            
            # Reconstructed images (bottom row)
            img_recon = (reconstructions[i].squeeze() + 1) / 2
            # Clamp to [0, 1] to handle any slight out-of-range values
            img_recon = torch.clamp(img_recon, 0, 1)
            axes[1, i].imshow(img_recon, cmap='gray')
            axes[1, i].set_title("Reconstruction", fontsize=8)
            axes[1, i].axis('off')
        
        plt.suptitle(title, fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Reconstruction plot saved to: {save_path}")
        
        plt.show()
    
    def save_checkpoint(
        self, 
        path: Optional[str] = None,
        epoch: Optional[int] = None
    ) -> str:
        """
        Save model checkpoint for resumption or deployment.
        
        Checkpoint includes:
            - Model state dict (all learned weights)
            - Optimizer state dict (Adam momentum buffers)
            - Scheduler state dict (if using scheduler)
            - Training history (losses per epoch)
            - Current epoch (for resumption)
            - Config snapshot (for reproducibility)
        
        Args:
            path: Save path. If None, uses config.autoencoder_path
            epoch: Epoch number to include in checkpoint
        
        Returns:
            save_path: Path where checkpoint was saved
        
        Example:
            >>> # Save during training
            >>> trainer.save_checkpoint()
            >>> 
            >>> # Later: load and resume
            >>> trainer.load_checkpoint()
            >>> trainer.fit(epochs=100)  # Continues from saved epoch
        """
        if not self.config.save_checkpoints:
            print("Checkpoint saving disabled (save_checkpoints=False)")
            return ""
        
        save_path = path or self.config.autoencoder_path
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Build checkpoint dictionary
        checkpoint = {
            # Model weights - the learned parameters
            'model_state_dict': self.autoencoder.state_dict(),
            
            # Optimizer state - momentum buffers for Adam
            # Important for smooth training resumption
            'optimizer_state_dict': self.optimizer.state_dict(),
            
            # Training progress
            'epoch': epoch or self.current_epoch,
            'history': self.history,
            
            # Configuration snapshot for reproducibility
            'config': {
                'latent_dim': self.config.latent_dim,
                'lr_ae': self.config.lr_ae,
                'ae_epochs': self.config.ae_epochs,
                'beta1': self.config.beta1,
                'beta2': self.config.beta2,
            }
        }
        
        # Add scheduler state if using
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to: {save_path}")
        
        return save_path
    
    def load_checkpoint(
        self, 
        path: Optional[str] = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> 'AutoencoderTrainer':
        """
        Load checkpoint for training resumption.
        
        Args:
            path: Load path. If None, uses config.autoencoder_path
            load_optimizer: Whether to restore optimizer state
                - True: Resume training smoothly (recommended)
                - False: Fresh optimizer (useful for fine-tuning with different LR)
            load_scheduler: Whether to restore scheduler state
        
        Returns:
            self: For method chaining
        
        Example:
            >>> # Resume interrupted training
            >>> trainer.load_checkpoint()
            >>> trainer.fit(epochs=100)  # Continues from checkpoint
        """
        load_path = path or self.config.autoencoder_path
        
        # Load checkpoint to correct device
        # weights_only=False: Allow loading full checkpoint dict (not just model weights)
        # This is required because we save optimizer state, history dict, etc.
        # Note: Only use weights_only=False for checkpoints you trust (your own saves)
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        
        # Restore model weights
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        
        # Optionally restore optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Optionally restore scheduler state
        if load_scheduler and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.history = checkpoint.get('history', {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        })
        
        print(f"Checkpoint loaded from: {load_path}")
        print(f"  Resuming from epoch: {self.current_epoch}")
        if self.history['train_loss']:
            print(f"  Last train loss: {self.history['train_loss'][-1]:.6f}")
        
        return self
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history for analysis and plotting.
        
        Returns:
            Dict containing:
                - 'train_loss': List of training losses per epoch
                - 'val_loss': List of validation losses (if applicable)
                - 'learning_rate': List of learning rates per epoch
        """
        return self.history.copy()
    
    def plot_training_curve(
        self, 
        save_path: Optional[str] = None,
        show_lr: bool = True
    ) -> None:
        """
        Plot training loss curve.
        
        Creates a figure showing:
            - Training loss over epochs (primary y-axis)
            - Validation loss if available
            - Learning rate if show_lr=True (secondary y-axis)
        
        Args:
            save_path: If provided, save figure to this path
            show_lr: Whether to show learning rate on secondary axis
        
        Example:
            >>> trainer.fit()
            >>> trainer.plot_training_curve(save_path='training_curve.png')
        """
        if not self.history['train_loss']:
            print("No training history to plot. Train the model first.")
            return
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Primary y-axis: Loss
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss', color='tab:blue')
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        
        if self.history['val_loss']:
            ax1.plot(epochs, self.history['val_loss'], 'b--', label='Val Loss', linewidth=2)
        
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Secondary y-axis: Learning rate
        if show_lr and self.history['learning_rate']:
            ax2 = ax1.twinx()
            ax2.set_ylabel('Learning Rate', color='tab:red')
            ax2.plot(epochs, self.history['learning_rate'], 'r:', 
                    label='LR', linewidth=1.5, alpha=0.7)
            ax2.tick_params(axis='y', labelcolor='tab:red')
            ax2.legend(loc='upper right')
        
        plt.title('Autoencoder Training Progress')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curve saved to: {save_path}")
        
        plt.show()
    
    def get_encoder(self) -> nn.Module:
        """
        Get the trained encoder for latent space analysis.
        
        The encoder is used by ClassConditionalLatentGenerator to:
            1. Encode all training images to latent vectors
            2. Compute per-class mean and variance
        
        Returns:
            encoder: The trained encoder module
        """
        return self.autoencoder.encoder
    
    def get_decoder(self) -> nn.Module:
        """
        Get the trained decoder for weight transfer.
        
        The decoder weights are transferred to the GAN generator
        for meaningful initialization.
        
        Returns:
            decoder: The trained decoder module
        """
        return self.autoencoder.decoder


# =============================================================================
# Module-level convenience functions
# =============================================================================

def create_trainer(
    autoencoder: BAGANAutoencoder,
    train_loader: DataLoader,
    config: Optional[BAGANConfig] = None,
    **kwargs
) -> AutoencoderTrainer:
    """
    Factory function to create an AutoencoderTrainer.
    
    Args:
        autoencoder: BAGANAutoencoder to train
        train_loader: Training data loader
        config: BAGANConfig instance. If None, uses default config.
        **kwargs: Additional arguments passed to AutoencoderTrainer
    
    Returns:
        AutoencoderTrainer instance
    
    Example:
        >>> trainer = create_trainer(autoencoder, train_loader)
        >>> trainer.fit()
    """
    if config is None:
        config = BAGANConfig()
    return AutoencoderTrainer(config, autoencoder, train_loader, **kwargs)


# =============================================================================
# Test code (runs only when executed directly)
# =============================================================================
# $env:KMP_DUPLICATE_LIB_OK="TRUE"; python your_script.py

if __name__ == "__main__":
    """
    Test the AutoencoderTrainer with synthetic data.
    
    This test verifies:
        1. Trainer initialization works correctly
        2. Training loop runs without errors
        3. Checkpointing saves and loads properly
        4. Visualization functions work
    """
    # Fix OpenMP library conflict error
    # ===================================
    # Problem: Multiple copies of OpenMP runtime loaded by PyTorch/NumPy/MKL
    # Solution: Set KMP_DUPLICATE_LIB_OK=TRUE to allow duplicate libraries
    # 
    # This MUST be set BEFORE importing torch/numpy, but since we already
    # imported them at the top of this file, this line only helps for
    # subprocess spawns or module reloads.
    #
    # To fix the error yourself in future scripts, add this line at the
    # VERY TOP of your script (before any torch/numpy imports):
    #   import os
    #   os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    #
    # Or set it in your terminal before running Python:
    #   PowerShell: $env:KMP_DUPLICATE_LIB_OK="TRUE"
    #   Bash: export KMP_DUPLICATE_LIB_OK=TRUE
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("\n" + "=" * 70)
    print("Testing AutoencoderTrainer")
    print("=" * 70)
    
    # =========================================================================
    # Setup: Create synthetic dataset and model
    # =========================================================================
    config = BAGANConfig(num_classes=3)
    
    # Create synthetic dataset
    # Random images simulating FashionMNIST format
    n_samples = 128
    batch_size = 32
    
    # Create random images in [-1, 1] range (matching normalization)
    synthetic_images = torch.randn(n_samples, 1, 64, 64)
    synthetic_labels = torch.randint(0, 3, (n_samples,))  # 3-class labels
    
    # Create TensorDataset and DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(synthetic_images, synthetic_labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create autoencoder
    autoencoder = BAGANAutoencoder(
        in_channels=config.img_channels,
        latent_dim=config.latent_dim
    )
    
    # =========================================================================
    # Test 1: Trainer Initialization
    # =========================================================================
    print("\nTest 1: Initialize trainer")
    trainer = AutoencoderTrainer(
        config=config,
        autoencoder=autoencoder,
        train_loader=train_loader,
        use_scheduler=True
    )
    print("  ✓ Trainer initialized successfully")
    
    # =========================================================================
    # Test 2: Single Epoch Training
    # =========================================================================
    print("\nTest 2: Train for 2 epochs")
    history = trainer.fit(epochs=2, verbose=True, save_every=0, visualize_every=0)
    print(f"  ✓ Training completed")
    print(f"  ✓ Train losses: {[f'{l:.4f}' for l in history['train_loss']]}")
    
    # =========================================================================
    # Test 3: Checkpoint Save/Load
    # =========================================================================
    print("\nTest 3: Checkpoint save/load")
    test_checkpoint_path = "saved/test_ae_checkpoint.pth"
    trainer.save_checkpoint(path=test_checkpoint_path)
    
    # Create new trainer and load
    new_trainer = AutoencoderTrainer(
        config=config,
        autoencoder=BAGANAutoencoder(
            in_channels=config.img_channels,
            latent_dim=config.latent_dim
        ),
        train_loader=train_loader
    )
    new_trainer.load_checkpoint(path=test_checkpoint_path)
    print(f"  ✓ Checkpoint saved and loaded successfully")
    print(f"  ✓ Resumed from epoch: {new_trainer.current_epoch}")
    
    # Clean up test checkpoint
    if os.path.exists(test_checkpoint_path):
        os.remove(test_checkpoint_path)
        print(f"  ✓ Test checkpoint cleaned up")
    
    # =========================================================================
    # Test 4: Get Encoder/Decoder
    # =========================================================================
    print("\nTest 4: Access encoder and decoder")
    encoder = trainer.get_encoder()
    decoder = trainer.get_decoder()
    print(f"  ✓ Encoder type: {type(encoder).__name__}")
    print(f"  ✓ Decoder type: {type(decoder).__name__}")
    
    # Test encoder forward pass
    test_input = torch.randn(4, 1, 64, 64).to(trainer.device)
    with torch.no_grad():
        z = encoder(test_input)
    print(f"  ✓ Encoder output shape: {z.shape}")
    
    # =========================================================================
    # All Tests Passed
    # =========================================================================
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
