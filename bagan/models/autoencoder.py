"""
BAGAN Autoencoder Components

Standard Autoencoder (not VAE) for BAGAN initialization.
Uses MSE loss only - no KL divergence term.

Architecture matches the notebook's ConditionalVAE but without:
- Class conditioning (autoencoder processes all images unconditionally)
- KL divergence loss (simple reconstruction loss only)

Reference: Mariani et al. (2018) Section 4 - Autoencoder Training
"""

import torch
import torch.nn as nn
from typing import Tuple


class BAGANEncoder(nn.Module):
    """
    BAGAN Encoder: Image (1, 64, 64) → Latent vector (latent_dim,)
    
    Architecture:
        Conv1: 1 → 64, stride=2 → (64, 32, 32)
        Conv2: 64 → 128, stride=2 → (128, 16, 16)
        Conv3: 128 → 256, stride=2 → (256, 8, 8)
        Conv4: 256 → 512, stride=2 → (512, 4, 4)
        Flatten: 512 × 4 × 4 = 8192
        FC: 8192 → latent_dim (128)
    
    Note: NO class conditioning - autoencoder is unconditional.
    This encoder's weights will be transferred to Discriminator's De layers.
    """
    
    def __init__(
        self,
        in_channels: int = 1, # Grayscale input
        latent_dim: int = 128,
        base_channels: int = 64
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Convolutional layers: progressively downsample spatial dimensions
        # Each layer halves spatial size: 64 → 32 → 16 → 8 → 4
        self.conv_layers = nn.Sequential(
            # Layer 1: (1, 64, 64) → (64, 32, 32)
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: (64, 32, 32) → (128, 16, 16)
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: (128, 16, 16) → (256, 8, 8)
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: (256, 8, 8) → (512, 4, 4)
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Flatten and project to latent space
        # 512 * 4 * 4 = 8192 → latent_dim
        self.fc = nn.Linear(base_channels * 8 * 4 * 4, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent vector.
        
        Args:
            x: Input image tensor of shape (batch, 1, 64, 64)
            
        Returns:
            Latent vector of shape (batch, latent_dim)
        """
        # Convolutional encoding
        features = self.conv_layers(x)  # (batch, 512, 4, 4)
        
        # Flatten and project
        features = features.view(features.size(0), -1)  # (batch, 8192)
        z = self.fc(features)  # (batch, latent_dim)
        
        return z


class BAGANDecoder(nn.Module):
    """
    BAGAN Decoder: Latent vector (latent_dim,) → Image (1, 64, 64)
    
    Architecture:
        FC: latent_dim → 8192 (= 512 × 4 × 4)
        Reshape: (512, 4, 4)
        ConvT1: 512 → 256, stride=2 → (256, 8, 8)
        ConvT2: 256 → 128, stride=2 → (128, 16, 16)
        ConvT3: 128 → 64, stride=2 → (64, 32, 32)
        ConvT4: 64 → 1, stride=2 → (1, 64, 64)
        Tanh activation: output in [-1, 1]
    
    Note: NO class conditioning - autoencoder is unconditional.
    This decoder's weights will be transferred to Generator.
    """
    
    def __init__(
        self,
        out_channels: int = 1,
        latent_dim: int = 128,
        base_channels: int = 64
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        
        # Project latent to feature map size
        self.fc = nn.Linear(latent_dim, base_channels * 8 * 4 * 4)
        
        # Transposed convolutional layers: progressively upsample
        # Each layer doubles spatial size: 4 → 8 → 16 → 32 → 64
        self.conv_layers = nn.Sequential(
            # Reshape happens before this in forward()
            # Input: (512, 4, 4)
            
            # Layer 1: (512, 4, 4) → (256, 8, 8)
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Layer 2: (256, 8, 8) → (128, 16, 16)
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Layer 3: (128, 16, 16) → (64, 32, 32)
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Layer 4: (64, 32, 32) → (1, 64, 64)
            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # Output in [-1, 1] to match image normalization
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.
        
        Args:
            z: Latent vector of shape (batch, latent_dim)
            
        Returns:
            Reconstructed image of shape (batch, 1, 64, 64)
        """
        # Project and reshape
        features = self.fc(z)  # (batch, 8192)
        features = features.view(features.size(0), self.base_channels * 8, 4, 4)  # (batch, 512, 4, 4)
        
        # Transposed convolution decoding
        x_recon = self.conv_layers(features)  # (batch, 1, 64, 64)
        
        return x_recon


class BAGANAutoencoder(nn.Module):
    """
    BAGAN Autoencoder: Complete Encoder + Decoder for pre-training.
    
    This is a standard autoencoder (NOT VAE):
    - No reparameterization trick
    - No KL divergence loss
    - Simple MSE reconstruction loss
    
    Training:
        - Train on ALL data (majority + minority classes)
        - No class conditioning
        - Loss = MSE(input, reconstruction)
    
    After training:
        - Encoder weights → Discriminator's De layers
        - Decoder weights → Generator
        - Latent vectors used to compute class-conditional Gaussians
    
    Reference: Mariani et al. (2018) Section 4 - "Autoencoder training"
    "In this work we apply l2 loss minimization for the autoencoder training."
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        base_channels: int = 64
    ):
        super().__init__()
        
        self.encoder = BAGANEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels
        )
        
        self.decoder = BAGANDecoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent vector."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.
        
        Args:
            x: Input image tensor of shape (batch, 1, 64, 64)
            
        Returns:
            Tuple of:
                - x_recon: Reconstructed image (batch, 1, 64, 64)
                - z: Latent vector (batch, latent_dim)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    # @staticmethod decorator means this method doesn't use 'self' (no instance access)
    # Benefits:
    #   - Can be called without creating an instance: BAGANAutoencoder.reconstruction_loss(x, recon)
    #   - Also callable on instance: model.reconstruction_loss(x, recon)
    #   - Clearer intent: this is a utility function that belongs to the class logically
    @staticmethod
    def reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE reconstruction loss.
        
        BAGAN uses simple L2 loss, not the VAE's combined MSE + KL loss.
        
        Args:
            x: Original image
            x_recon: Reconstructed image
            
        Returns:
            MSE loss (scalar)
        """
        return nn.functional.mse_loss(x_recon, x, reduction='mean')
