"""
BAGAN Generator

Generator network for BAGAN with architecture identical to Decoder.
This allows direct weight transfer from pre-trained autoencoder.

Key points:
- Same architecture as BAGANDecoder (for weight transfer)
- NO class conditioning (unconditional generation)
- Latent sampling uses class-conditional Gaussians (external)
- Weights initialized from trained Decoder

Reference: Mariani et al. (2018) Section 4 - "GAN training"
"""

import torch
import torch.nn as nn
from typing import Optional


class BAGANGenerator(nn.Module):
    """
    BAGAN Generator: Latent vector (latent_dim,) → Image (1, 64, 64)
    
    Architecture (IDENTICAL to BAGANDecoder for weight transfer):
        FC: latent_dim → 8192 (= 512 × 4 × 4)
        Reshape: (512, 4, 4)
        ConvT1: 512 → 256, stride=2 → (256, 8, 8)
        ConvT2: 256 → 128, stride=2 → (128, 16, 16)
        ConvT3: 128 → 64, stride=2 → (64, 32, 32)
        ConvT4: 64 → 1, stride=2 → (1, 64, 64)
        Tanh activation: output in [-1, 1]
    
    Note: 
        - NO class embedding! Generator is unconditional.
        - Class-conditioning happens via latent space sampling (Cholesky decomposition)
        - The latent z is already class-conditioned when passed to generate()
    
    BAGAN vs AC-GAN Generator:
        - AC-GAN: G(z, c) where c is class embedding concatenated to z
        - BAGAN: G(z_c) where z_c = μ_c + L_c · ε is class-conditioned latent
    """
    
    def __init__(
        self,
        out_channels: int = 1,
        latent_dim: int = 128,
        base_channels: int = 64
    ):
        """
        Initialize BAGAN Generator.
        
        Args:
            out_channels: Number of output image channels (1 for grayscale, 3 for RGB)
            latent_dim: Dimension of input latent vector
            base_channels: Base number of channels (multiplied for deeper layers)
        """
        super().__init__()
        
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        
        # Project latent to feature map size
        # latent_dim → 512 * 4 * 4 = 8192
        self.fc = nn.Linear(latent_dim, base_channels * 8 * 4 * 4)
        
        # Transposed convolutional layers: progressively upsample
        # Each layer doubles spatial size: 4 → 8 → 16 → 32 → 64
        self.conv_layers = nn.Sequential(
            # Input: (512, 4, 4) after reshape
            
            # Layer 1: (512, 4, 4) → (256, 8, 8)
            nn.ConvTranspose2d(
                base_channels * 8, base_channels * 4, 
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Layer 2: (256, 8, 8) → (128, 16, 16)
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2, 
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Layer 3: (128, 16, 16) → (64, 32, 32)
            nn.ConvTranspose2d(
                base_channels * 2, base_channels, 
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Layer 4: (64, 32, 32) → (out_channels, 64, 64)
            nn.ConvTranspose2d(
                base_channels, out_channels, 
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.Tanh(),  # Output in [-1, 1] to match image normalization
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate image from latent vector.
        
        The input z should already be class-conditioned via Cholesky sampling:
            z_c = μ_c + L_c · ε
        where μ_c and L_c are the mean and Cholesky factor for class c.
        
        Args:
            z: Latent vector of shape (batch, latent_dim)
               Already class-conditioned through ClassConditionalLatentGenerator
        
        Returns:
            Generated image of shape (batch, out_channels, 64, 64)
            Pixel values in range [-1, 1]
        """
        # Project latent to spatial features
        features = self.fc(z)  # (batch, 8192)
        features = features.view(features.size(0), self.base_channels * 8, 4, 4)  # (batch, 512, 4, 4)
        
        # Transposed convolution upsampling
        images = self.conv_layers(features)  # (batch, out_channels, 64, 64)
        
        return images
    
    def generate(
        self, 
        num_samples: int, 
        latent_generator: Optional['ClassConditionalLatentGenerator'] = None,
        target_class: Optional[int] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Convenience method to generate samples.
        
        Args:
            num_samples: Number of images to generate
            latent_generator: ClassConditionalLatentGenerator for class-conditioned sampling
                            If None, uses standard normal sampling
            target_class: Class to generate (required if latent_generator provided)
            device: Device to generate on (defaults to model's device)
        
        Returns:
            Generated images of shape (num_samples, out_channels, 64, 64)
        """
        if device is None:
            device = next(self.parameters()).device
        
        if latent_generator is not None and target_class is not None:
            # Class-conditioned sampling via Cholesky decomposition
            z = latent_generator.sample(num_samples, target_class, device)
        else:
            # Standard normal sampling (no class conditioning)
            z = torch.randn(num_samples, self.latent_dim, device=device)
        
        with torch.no_grad():
            images = self.forward(z)
        
        return images
    
    def load_decoder_weights(self, decoder: nn.Module) -> dict:
        """
        Load weights from a trained BAGANDecoder.
        
        This is a key step in BAGAN initialization:
        After autoencoder training, decoder weights are transferred to generator.
        
        Args:
            decoder: Trained BAGANDecoder instance
        
        Returns:
            dict: Transfer statistics (matched keys, total params, etc.)
        
        Raises:
            ValueError: If architectures don't match
        """
        decoder_state = decoder.state_dict()
        generator_state = self.state_dict()
        
        # Verify architectures match
        if set(decoder_state.keys()) != set(generator_state.keys()):
            decoder_keys = set(decoder_state.keys())
            generator_keys = set(generator_state.keys())
            missing = generator_keys - decoder_keys
            extra = decoder_keys - generator_keys
            raise ValueError(
                f"Architecture mismatch!\n"
                f"Missing in decoder: {missing}\n"
                f"Extra in decoder: {extra}"
            )
        
        # Verify shapes match
        mismatched = []
        for key in decoder_state:
            if decoder_state[key].shape != generator_state[key].shape:
                mismatched.append(
                    f"{key}: decoder {decoder_state[key].shape} vs generator {generator_state[key].shape}"
                )
        
        if mismatched:
            raise ValueError(f"Shape mismatch!\n" + "\n".join(mismatched))
        
        # Load weights
        self.load_state_dict(decoder_state)
        
        # Compute stats
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'matched_keys': len(decoder_state),
            'total_parameters': total_params,
            'success': True
        }


# Convenience function for creating generator from config
def create_generator_from_config(config) -> BAGANGenerator:
    """
    Create BAGANGenerator from BAGANConfig.
    
    Args:
        config: BAGANConfig instance
    
    Returns:
        BAGANGenerator instance
    """
    return BAGANGenerator(
        out_channels=config.img_channels,
        latent_dim=config.latent_dim,
        base_channels=config.encoder_channels[0]  # 64
    )
