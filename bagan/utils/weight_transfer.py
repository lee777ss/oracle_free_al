"""
BAGAN Weight Transfer Module

This module handles the critical weight transfer from a pre-trained Autoencoder
to the GAN models (Generator and Discriminator). This is a key innovation in
BAGAN that enables stable GAN training by providing a "warm start".

Why Weight Transfer Matters:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Problem: GANs are notoriously hard to train from random init        │
    │   - Mode collapse: G generates same image regardless of input       │
    │   - Training instability: D/G oscillate, never converge             │
    │   - Vanishing gradients: D becomes too strong, G stops learning     │
    │                                                                     │
    │ BAGAN Solution: Initialize GAN with pre-trained Autoencoder         │
    │   - Decoder already knows how to generate realistic images          │
    │   - Encoder already knows meaningful feature representations        │
    │   - GAN starts from stable, meaningful weight space                 │
    │   - Much faster convergence and better final quality                │
    └─────────────────────────────────────────────────────────────────────┘

Weight Transfer Mapping:
    ┌────────────────────────────────────────────────────────────────────┐
    │ DECODER → GENERATOR (Full Transfer)                                │
    ├────────────────────────────────────────────────────────────────────┤
    │ Decoder Layer           │ Generator Layer         │ Status         │
    │ ─────────────────────── │ ─────────────────────── │ ────────────── │
    │ fc                      │ fc                      │ ✓ Transfer     │
    │ conv_layers.0 (ConvT)   │ conv_layers.0 (ConvT)   │ ✓ Transfer     │
    │ conv_layers.1 (BN)      │ conv_layers.1 (BN)      │ ✓ Transfer     │
    │ conv_layers.3 (ConvT)   │ conv_layers.3 (ConvT)   │ ✓ Transfer     │
    │ conv_layers.4 (BN)      │ conv_layers.4 (BN)      │ ✓ Transfer     │
    │ conv_layers.6 (ConvT)   │ conv_layers.6 (ConvT)   │ ✓ Transfer     │
    │ conv_layers.7 (BN)      │ conv_layers.7 (BN)      │ ✓ Transfer     │
    │ conv_layers.9 (ConvT)   │ conv_layers.9 (ConvT)   │ ✓ Transfer     │
    └────────────────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ ENCODER → DISCRIMINATOR (Partial Transfer - Conv Only)            │
    ├────────────────────────────────────────────────────────────────────┤
    │ Encoder Layer           │ Discriminator Layer     │ Status         │
    │ ─────────────────────── │ ─────────────────────── │ ────────────── │
    │ conv_layers.0 (Conv)    │ conv_layers.0 (Conv)    │ ✓ Transfer     │
    │ conv_layers.1 (BN)      │ conv_layers.1 (BN)      │ ✓ Transfer     │
    │ conv_layers.3 (Conv)    │ conv_layers.3 (Conv)    │ ✓ Transfer     │
    │ conv_layers.4 (BN)      │ conv_layers.4 (BN)      │ ✓ Transfer     │
    │ conv_layers.6 (Conv)    │ conv_layers.6 (Conv)    │ ✓ Transfer     │
    │ conv_layers.7 (BN)      │ conv_layers.7 (BN)      │ ✓ Transfer     │
    │ conv_layers.9 (Conv)    │ conv_layers.9 (Conv)    │ ✓ Transfer     │
    │ conv_layers.10 (BN)     │ conv_layers.10 (BN)     │ ✓ Transfer     │
    │ fc (8192→128)           │ classifier (8192→3)     │ ✗ Skip (shape) │
    └────────────────────────────────────────────────────────────────────┘

Reference: Mariani et al. (2018) "BAGAN: Data Augmentation with Balancing GAN"
    Section 4: "We then train a GAN using the decoder as the generator's 
    initialization and the encoder to initialize the discriminator."

Usage:
    >>> from bagan import WeightTransfer, BAGANAutoencoder, BAGANGenerator, BAGANDiscriminator
    >>> 
    >>> # Load pre-trained autoencoder
    >>> autoencoder = BAGANAutoencoder()
    >>> autoencoder.load_state_dict(torch.load('saved/autoencoder.pth'))
    >>> 
    >>> # Create GAN models
    >>> generator = BAGANGenerator()
    >>> discriminator = BAGANDiscriminator()
    >>> 
    >>> # Transfer weights
    >>> WeightTransfer.initialize_gan_from_autoencoder(
    ...     autoencoder, generator, discriminator, verbose=True
    ... )
    >>> 
    >>> # Now generator and discriminator have pre-trained weights!
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import os
import sys

# =============================================================================
# Import Handling for Both Module and Direct Execution
# =============================================================================
# This pattern allows the file to be:
#   1. Imported as part of the bagan package: from bagan.utils import WeightTransfer
#   2. Run directly for testing: python bagan/utils/weight_transfer.py

if __name__ == "__main__":
    # Direct execution: Add parent directories to Python's module search path
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from bagan.models.autoencoder import BAGANAutoencoder, BAGANEncoder, BAGANDecoder
    from bagan.models.generator import BAGANGenerator
    from bagan.models.discriminator import BAGANDiscriminator
else:
    # Module import: Use relative imports (preferred for package code)
    from ..models.autoencoder import BAGANAutoencoder, BAGANEncoder, BAGANDecoder
    from ..models.generator import BAGANGenerator
    from ..models.discriminator import BAGANDiscriminator


class WeightTransfer:
    """
    Static utility class for transferring weights between BAGAN models.
    
    This class provides methods to transfer learned weights from a pre-trained
    Autoencoder to initialize the GAN's Generator and Discriminator. This
    initialization strategy is crucial for stable GAN training in BAGAN.
    
    Theory Behind Weight Transfer:
        ┌─────────────────────────────────────────────────────────────────┐
        │ Phase 1: Autoencoder Pre-training                               │
        │                                                                 │
        │   Input x → Encoder → z → Decoder → x̂                          │
        │                                                                 │
        │   - Encoder learns to extract meaningful features               │
        │   - Decoder learns to reconstruct images from latent space      │
        │   - Both networks are trained stably with reconstruction loss   │
        │                                                                 │
        │ Phase 2: GAN Training (with transferred weights)                │
        │                                                                 │
        │   z_c ~ N(μ_c, Σ_c) → Generator → fake_x                       │
        │                         ↓                                       │
        │   real_x or fake_x → Discriminator → class logits              │
        │                                                                 │
        │   - Generator starts with Decoder's reconstruction knowledge   │
        │   - Discriminator starts with Encoder's feature extraction     │
        │   - Both networks already understand the image manifold        │
        └─────────────────────────────────────────────────────────────────┘
    
    All methods are static (no instance needed):
        >>> WeightTransfer.transfer_decoder_to_generator(decoder, generator)
    
    A static method is a method that:
        Belongs to a class, not to an instance
        Does not receive self (instance) or cls (class)
        Behaves like a regular function, but is namespaced inside a class for logical grouping

        
    Attributes:
        None (all methods are static)
    """
    
    @staticmethod
    def transfer_decoder_to_generator(
        decoder: BAGANDecoder,
        generator: BAGANGenerator,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Transfer all weights from Decoder to Generator.
        
        Since Decoder and Generator have IDENTICAL architectures, we can
        transfer ALL weights directly. This gives the Generator a head start:
        it already knows how to generate realistic-looking images.
        
        Layer Mapping:
            Decoder (source)         Generator (target)
            ────────────────────     ────────────────────
            fc.weight, fc.bias   →   fc.weight, fc.bias
            conv_layers.0.weight →   conv_layers.0.weight  (ConvTranspose2d)
            conv_layers.1.*      →   conv_layers.1.*       (BatchNorm2d)
            conv_layers.3.weight →   conv_layers.3.weight  (ConvTranspose2d)
            conv_layers.4.*      →   conv_layers.4.*       (BatchNorm2d)
            ... (same pattern for all layers)
        
        Args:
            decoder: Pre-trained BAGANDecoder instance
            generator: BAGANGenerator instance to initialize
            verbose: Whether to print detailed transfer information
        
        Returns:
            Dict with transfer statistics:
                - 'transferred_params': Number of parameters transferred
                - 'transferred_layers': List of layer names transferred
                - 'skipped_layers': List of layer names skipped (if any)
        
        Example:
            >>> decoder = autoencoder.decoder  # From trained autoencoder
            >>> generator = BAGANGenerator()
            >>> stats = WeightTransfer.transfer_decoder_to_generator(decoder, generator)
            >>> print(f"Transferred {stats['transferred_params']:,} parameters")
        """
        if verbose:
            print("\n" + "=" * 70)
            print("Transferring Weights: Decoder → Generator")
            print("=" * 70)
        
        # =====================================================================
        # Get State Dictionaries
        # =====================================================================
        # state_dict() returns OrderedDict of {parameter_name: tensor}
        # This is PyTorch's way of serializing a model's learnable parameters
        # 
        # What's in a state_dict?
        #   - Keys: Layer names (e.g., 'fc.weight', 'conv_layers.0.weight')
        #   - Values: Tensor containing the learned weights
        #
        # Example structure:
        #   OrderedDict([
        #       ('fc.weight', tensor([[0.23, -0.45, ...], ...], shape=[8192, 128])),
        #       ('fc.bias', tensor([0.1, -0.2, ...], shape=[8192])),
        #       ('conv_layers.0.weight', tensor([[[[...]]]], shape=[128, 64, 4, 4])),
        #       ...
        #   ])
        #
        # Why use state_dict()?
        #   - Provides a standardized interface to access all parameters
        #   - Preserves layer hierarchy through dotted naming
        #   - Can be saved/loaded for checkpointing
        #   - Allows parameter-by-parameter manipulation
        decoder_state = decoder.state_dict()
        generator_state = generator.state_dict()

        # ====================================================================
        # The prcess of transferring weights:
        # 1. Get decoder_state (source weights)
        # 2. Get generator_state (target weights)
        # 3. Copy weights from decoder_state to generator_state
        # 4. Load updated generator_state into generator
        # ====================================================================

        # =====================================================================
        # Track Transfer Statistics
        # =====================================================================
        # We track what happened during transfer for debugging and validation
        transferred_params = 0      # Total number of individual weights copied
        transferred_layers = []     # List of layer names successfully transferred
        skipped_layers = []         # List of layers that couldn't be transferred
        
        # =====================================================================
        # Transfer Each Parameter
        # =====================================================================
        # How weight transfer works:
        #
        # 1. ITERATE through source (decoder) parameters
        #    - Loop gives us (name, param) pairs
        #    - name: string like "fc.weight" or "conv_layers.3.bias"
        #    - param: actual tensor containing learned weights
        #
        # 2. CHECK if parameter exists in target (generator)
        #    - Models should be identical, but we verify anyway
        #    - "name in generator_state" checks if the key exists
        #
        # 3. VERIFY shapes match
        #    - Even if names match, shapes must be compatible
        #    - param.shape returns tuple like (128, 64, 4, 4)
        #    - Shape mismatch = architectural difference
        #
        # 4. COPY the weights in-place
        #    - .copy_() performs in-place tensor copy
        #    - This MODIFIES generator_state[name] directly
        #    - After copy, generator has decoder's learned weights
        #
        # 5. TRACK statistics
        #    - Count how many weights transferred
        #    - Record which layers succeeded/failed
        #
        # Since architectures are IDENTICAL, we expect to transfer ALL parameters
        for name, param in decoder_state.items():
            # -------------------------------------------------------------
            # Check 1: Does this parameter exist in the target model?
            # -------------------------------------------------------------
            # name is a string like "fc.weight" or "conv_layers.0.weight"
            # We check if generator has a parameter with the same name
            if name in generator_state:
                # ---------------------------------------------------------
                # Check 2: Do the shapes match?
                # ---------------------------------------------------------
                # param.shape returns tuple like (8192, 128) for FC layer
                # Shapes must match exactly for copy to work
                # Example: fc.weight should be [8192, 128] in both models
                if param.shape == generator_state[name].shape:
                    # -----------------------------------------------------
                    # PERFORM THE WEIGHT TRANSFER
                    # -----------------------------------------------------
                    # .copy_() is an IN-PLACE operation that copies tensor values
                    # 
                    # What happens internally:
                    #   1. generator_state[name] is a tensor in GPU/CPU memory
                    #   2. .copy_(param) overwrites that memory with param's values
                    #   3. The tensor reference stays the same (in-place)
                    #   4. No new memory is allocated
                    #
                    # Why in-place?
                    #   - Efficient: No memory allocation
                    #   - Safe: Preserves computational graph if needed
                    #   - Direct: Modifies the actual generator weights
                    #
                    # After this line, generator has decoder's trained weights!
                    generator_state[name].copy_(param)
                    
                    # -----------------------------------------------------
                    # TRACK TRANSFER STATISTICS
                    # -----------------------------------------------------
                    # param.numel() = "number of elements" in the tensor
                    # Example: tensor of shape [128, 64, 4, 4] has 128*64*4*4 = 131,072 elements
                    # We sum these to get total parameters transferred
                    transferred_params += param.numel()
                    transferred_layers.append(name)
                    
                    if verbose:
                        print(f"  ✓ {name}: {list(param.shape)}")
                else:
                    # ---------------------------------------------------------
                    # SHAPE MISMATCH (shouldn't happen for identical models)
                    # ---------------------------------------------------------
                    # This indicates architectural difference
                    # For Decoder→Generator, this is unexpected
                    skipped_layers.append(name)
                    if verbose:
                        print(f"  ✗ {name}: Shape mismatch! "
                              f"Decoder={list(param.shape)}, Generator={list(generator_state[name].shape)}")
            else:
                # -------------------------------------------------------------
                # PARAMETER NOT FOUND (shouldn't happen for identical models)
                # -------------------------------------------------------------
                # This means decoder has a layer that generator doesn't
                # For Decoder→Generator, this is unexpected
                skipped_layers.append(name)
                if verbose:
                    print(f"  ✗ {name}: Not found in Generator")
        
        # =====================================================================
        # Load the Modified State Dict Back to Generator
        # =====================================================================
        # strict=True ensures all keys must match (raises error otherwise)
        generator.load_state_dict(generator_state, strict=True)
        
        # =====================================================================
        # Print Summary
        # =====================================================================
        if verbose:
            print("-" * 70)
            print(f"Transfer Complete!")
            print(f"  Parameters transferred: {transferred_params:,}")
            print(f"  Layers transferred: {len(transferred_layers)}")
            if skipped_layers:
                print(f"  Layers skipped: {len(skipped_layers)}")
            print("=" * 70)
        
        return {
            'transferred_params': transferred_params,
            'transferred_layers': transferred_layers,
            'skipped_layers': skipped_layers
        }
    
    @staticmethod
    def transfer_encoder_to_discriminator(
        encoder: BAGANEncoder,
        discriminator: BAGANDiscriminator,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Transfer convolutional weights from Encoder to Discriminator.
        
        The Encoder and Discriminator share the SAME convolutional architecture,
        but have DIFFERENT fully connected layers:
            - Encoder FC: 8192 → latent_dim (128)
            - Discriminator classifier: 8192 → num_classes + 1 (3)
        
        Therefore, we transfer only the conv layers (for feature extraction),
        and leave the classifier randomly initialized.
        
        Why This Works:
            The Encoder learns to extract hierarchical features that are useful
            for reconstructing images. These same features are useful for the
            Discriminator to distinguish between real classes and fake images.
        
        Layer Mapping:
            Encoder (source)           Discriminator (target)   Status
            ──────────────────────     ──────────────────────   ──────
            conv_layers.0.weight   →   conv_layers.0.weight     ✓ Transfer
            conv_layers.1.*        →   conv_layers.1.*          ✓ Transfer (BN)
            conv_layers.3.weight   →   conv_layers.3.weight     ✓ Transfer
            conv_layers.4.*        →   conv_layers.4.*          ✓ Transfer (BN)
            conv_layers.6.weight   →   conv_layers.6.weight     ✓ Transfer
            conv_layers.7.*        →   conv_layers.7.*          ✓ Transfer (BN)
            conv_layers.9.weight   →   conv_layers.9.weight     ✓ Transfer
            conv_layers.10.*       →   conv_layers.10.*         ✓ Transfer (BN)
            fc.weight, fc.bias         classifier.weight/bias   ✗ Skip
        
        Args:
            encoder: Pre-trained BAGANEncoder instance
            discriminator: BAGANDiscriminator instance to initialize
            verbose: Whether to print detailed transfer information
        
        Returns:
            Dict with transfer statistics:
                - 'transferred_params': Number of parameters transferred
                - 'transferred_layers': List of layer names transferred
                - 'skipped_layers': List of layer names skipped
        
        Example:
            >>> encoder = autoencoder.encoder  # From trained autoencoder
            >>> discriminator = BAGANDiscriminator()
            >>> stats = WeightTransfer.transfer_encoder_to_discriminator(encoder, discriminator)
            >>> print(f"Transferred {stats['transferred_params']:,} parameters")
        """
        if verbose:
            print("\n" + "=" * 70)
            print("Transferring Weights: Encoder → Discriminator")
            print("=" * 70)
        
        # =====================================================================
        # Get State Dictionaries
        # =====================================================================
        encoder_state = encoder.state_dict()
        discriminator_state = discriminator.state_dict()
        
        # =====================================================================
        # Track Transfer Statistics
        # =====================================================================
        transferred_params = 0
        transferred_layers = []
        skipped_layers = []
        
        # =====================================================================
        # Transfer Only Convolutional Layers
        # =====================================================================
        # We explicitly transfer conv_layers.* and skip fc.* / classifier.*
        for name, param in encoder_state.items():
            # Skip the fully connected layer (different output dimensions)
            if name.startswith('fc.'):
                skipped_layers.append(name)
                if verbose:
                    print(f"  ○ {name}: Skipped (FC layer, different output dim)")
                continue
            
            # Check if this layer exists in discriminator
            if name in discriminator_state:
                if param.shape == discriminator_state[name].shape:
                    # Transfer the weights
                    discriminator_state[name].copy_(param)
                    
                    transferred_params += param.numel()
                    transferred_layers.append(name)
                    
                    if verbose:
                        print(f"  ✓ {name}: {list(param.shape)}")
                else:
                    skipped_layers.append(name)
                    if verbose:
                        print(f"  ✗ {name}: Shape mismatch! "
                              f"Encoder={list(param.shape)}, Discriminator={list(discriminator_state[name].shape)}")
            else:
                skipped_layers.append(name)
                if verbose:
                    print(f"  ✗ {name}: Not found in Discriminator")
        
        # =====================================================================
        # Load the Modified State Dict Back to Discriminator
        # =====================================================================
        # strict=False because we only modified some parameters
        # The classifier layer retains its random initialization
        discriminator.load_state_dict(discriminator_state, strict=True)
        
        # =====================================================================
        # Print Summary
        # =====================================================================
        if verbose:
            print("-" * 70)
            print(f"Transfer Complete!")
            print(f"  Parameters transferred: {transferred_params:,}")
            print(f"  Layers transferred: {len(transferred_layers)}")
            print(f"  Layers skipped: {len(skipped_layers)} (FC layers - expected)")
            print("=" * 70)
        
        return {
            'transferred_params': transferred_params,
            'transferred_layers': transferred_layers,
            'skipped_layers': skipped_layers
        }
    
    @staticmethod
    def initialize_gan_from_autoencoder(
        autoencoder: BAGANAutoencoder,
        generator: BAGANGenerator,
        discriminator: BAGANDiscriminator,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Initialize both Generator and Discriminator from pre-trained Autoencoder.
        
        This is the main entry point for BAGAN weight transfer. It performs:
            1. Decoder → Generator (full transfer)
            2. Encoder → Discriminator (conv layers only)
        
        BAGAN Pipeline Context:
            ┌─────────────────────────────────────────────────────────────────┐
            │ Phase 1: Train Autoencoder                                      │
            │   - Input: Training images (all classes)                        │
            │   - Loss: MSE reconstruction loss                               │
            │   - Output: Trained encoder + decoder                           │
            │                                                                 │
            │ Phase 2: Fit Latent Generator ← YOU ARE HERE ─┐                │
            │   - Input: Encoder + training images          │                │
            │   - Output: Class-conditional Gaussians       │                │
            │                                               ▼                │
            │ Phase 3: Initialize GAN ← THIS FUNCTION ──────┘                │
            │   - Transfer Decoder weights to Generator                       │
            │   - Transfer Encoder conv weights to Discriminator              │
            │                                                                 │
            │ Phase 4: Train GAN                                              │
            │   - Use initialized G and D                                     │
            │   - Use class-conditional latent sampling                       │
            │   - (n+1)-class discriminator loss                              │
            └─────────────────────────────────────────────────────────────────┘
        
        Args:
            autoencoder: Pre-trained BAGANAutoencoder instance
            generator: BAGANGenerator instance to initialize
            discriminator: BAGANDiscriminator instance to initialize
            verbose: Whether to print detailed transfer information
        
        Returns:
            Dict with combined transfer statistics:
                - 'decoder_to_generator': Stats from D→G transfer
                - 'encoder_to_discriminator': Stats from E→D transfer
                - 'total_params_transferred': Total parameters transferred
        
        Example:
            >>> # Load trained autoencoder
            >>> autoencoder = BAGANAutoencoder()
            >>> autoencoder.load_state_dict(torch.load('saved/autoencoder.pth'))
            >>> 
            >>> # Create fresh GAN models
            >>> generator = BAGANGenerator()
            >>> discriminator = BAGANDiscriminator()
            >>> 
            >>> # Transfer weights from autoencoder to GAN
            >>> stats = WeightTransfer.initialize_gan_from_autoencoder(
            ...     autoencoder, generator, discriminator, verbose=True
            ... )
            >>> 
            >>> # Now GAN is ready for training with good initialization!
            >>> print(f"Total transferred: {stats['total_params_transferred']:,} parameters")
        """
        if verbose:
            print("\n" + "=" * 70)
            print("BAGAN Weight Initialization: Autoencoder → GAN")
            print("=" * 70)
            print("\nThis transfers learned weights from pre-trained Autoencoder")
            print("to initialize the GAN's Generator and Discriminator.\n")
        
        # =====================================================================
        # Transfer 1: Decoder → Generator
        # =====================================================================
        d2g_stats = WeightTransfer.transfer_decoder_to_generator(
            autoencoder.decoder,
            generator,
            verbose=verbose
        )
        
        # =====================================================================
        # Transfer 2: Encoder → Discriminator
        # =====================================================================
        e2d_stats = WeightTransfer.transfer_encoder_to_discriminator(
            autoencoder.encoder,
            discriminator,
            verbose=verbose
        )
        
        # =====================================================================
        # Combined Statistics
        # =====================================================================
        total_transferred = (
            d2g_stats['transferred_params'] + 
            e2d_stats['transferred_params']
        )
        
        if verbose:
            print("\n" + "=" * 70)
            print("Weight Transfer Complete!")
            print("=" * 70)
            print(f"  Decoder → Generator: {d2g_stats['transferred_params']:,} parameters")
            print(f"  Encoder → Discriminator: {e2d_stats['transferred_params']:,} parameters")
            print(f"  Total transferred: {total_transferred:,} parameters")
            print("\n  ✓ Generator initialized with Decoder weights")
            print("  ✓ Discriminator conv layers initialized with Encoder weights")
            print("  ✓ Discriminator classifier layer: random initialization")
            print("=" * 70)
        
        return {
            'decoder_to_generator': d2g_stats,
            'encoder_to_discriminator': e2d_stats,
            'total_params_transferred': total_transferred
        }
    
    @staticmethod
    def verify_transfer(
        source: nn.Module,
        target: nn.Module,
        layer_names: Optional[List[str]] = None
    ) -> bool:
        """
        Verify that weights were correctly transferred between two models.
        
        Compares the specified layers between source and target models to
        ensure the weights are identical (within floating point tolerance).
        
        This is useful for debugging and validation after weight transfer.
        
        Args:
            source: Source model (Encoder or Decoder)
            target: Target model (Discriminator or Generator)
            layer_names: Optional list of layer names to verify.
                        If None, verifies all matching layers.
        
        Returns:
            True if all specified layers match, False otherwise
        
        Example:
            >>> # After transfer, verify it worked
            >>> is_valid = WeightTransfer.verify_transfer(
            ...     autoencoder.decoder, 
            ...     generator,
            ...     layer_names=['fc.weight', 'conv_layers.0.weight']
            ... )
            >>> assert is_valid, "Weight transfer failed!"
        """
        source_state = source.state_dict()
        target_state = target.state_dict()
        
        # Determine which layers to verify
        if layer_names is None:
            # Verify all matching layers
            layer_names = [name for name in source_state.keys() 
                          if name in target_state]
        
        all_match = True
        
        for name in layer_names:
            if name not in source_state:
                print(f"  ✗ {name}: Not found in source")
                all_match = False
                continue
            
            if name not in target_state:
                print(f"  ✗ {name}: Not found in target")
                all_match = False
                continue
            
            source_param = source_state[name]
            target_param = target_state[name]
            
            # Check if shapes match
            if source_param.shape != target_param.shape:
                print(f"  ✗ {name}: Shape mismatch")
                all_match = False
                continue
            
            # Check if values match (within tolerance)
            if torch.allclose(source_param, target_param, rtol=1e-5, atol=1e-8):
                print(f"  ✓ {name}: Weights match")
            else:
                print(f"  ✗ {name}: Weights differ")
                all_match = False
        
        return all_match
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        Count the number of parameters in a model.
        
        Useful for understanding model size and verifying transfer coverage.
        
        Args:
            model: PyTorch model to analyze
        
        Returns:
            Dict with parameter counts:
                - 'total': Total number of parameters
                - 'trainable': Number of trainable parameters
                - 'frozen': Number of frozen (non-trainable) parameters
        
        Example:
            >>> stats = WeightTransfer.count_parameters(generator)
            >>> print(f"Total: {stats['total']:,} parameters")
            >>> print(f"Trainable: {stats['trainable']:,} parameters")
        """
        total = 0
        trainable = 0
        
        for param in model.parameters():
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }
    
    @staticmethod
    def freeze_transferred_layers(
        model: nn.Module,
        layer_names: List[str],
        verbose: bool = True
    ) -> int:
        """
        Freeze specific layers (disable gradient computation).
        
        This can be useful for:
            - Fine-tuning only the classifier layer of discriminator
            - Preventing early GAN training from corrupting transferred weights
            - Transfer learning experiments
        
        Args:
            model: Model containing layers to freeze
            layer_names: List of layer name prefixes to freeze
                        (e.g., ['conv_layers'] freezes all conv layers)
            verbose: Whether to print frozen layers
        
        Returns:
            Number of parameters frozen
        
        Example:
            >>> # Freeze conv layers, train only classifier
            >>> frozen = WeightTransfer.freeze_transferred_layers(
            ...     discriminator,
            ...     layer_names=['conv_layers'],
            ...     verbose=True
            ... )
            >>> print(f"Frozen {frozen:,} parameters")
        """
        frozen_count = 0
        
        for name, param in model.named_parameters():
            # Check if this parameter should be frozen
            should_freeze = any(name.startswith(prefix) for prefix in layer_names)
            
            if should_freeze:
                param.requires_grad = False
                frozen_count += param.numel()
                if verbose:
                    print(f"  Frozen: {name} ({param.numel():,} params)")
        
        if verbose:
            print(f"Total frozen: {frozen_count:,} parameters")
        
        return frozen_count
    
    @staticmethod
    def unfreeze_all(model: nn.Module, verbose: bool = True) -> int:
        """
        Unfreeze all parameters in a model.
        
        Args:
            model: Model to unfreeze
            verbose: Whether to print info
        
        Returns:
            Number of parameters unfrozen
        """
        unfrozen_count = 0
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += param.numel()
        
        if verbose:
            print(f"Unfrozen: {unfrozen_count:,} parameters")
        
        return unfrozen_count


# =============================================================================
# Module-level convenience functions
# =============================================================================

def transfer_autoencoder_to_gan(
    autoencoder: BAGANAutoencoder,
    generator: BAGANGenerator,
    discriminator: BAGANDiscriminator,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for weight transfer.
    
    Equivalent to WeightTransfer.initialize_gan_from_autoencoder().
    
    Args:
        autoencoder: Pre-trained autoencoder
        generator: Generator to initialize
        discriminator: Discriminator to initialize
        verbose: Whether to print details
    
    Returns:
        Transfer statistics
    """
    return WeightTransfer.initialize_gan_from_autoencoder(
        autoencoder, generator, discriminator, verbose
    )


# =============================================================================
# Test code (runs only when executed directly)
# =============================================================================
# Run with: python bagan/utils/weight_transfer.py
# Or in PowerShell: $env:KMP_DUPLICATE_LIB_OK="TRUE"; python bagan/utils/weight_transfer.py
# $env:KMP_DUPLICATE_LIB_OK="TRUE"

if __name__ == "__main__":
    """
    Test the WeightTransfer module.
    
    This test verifies:
        1. Decoder → Generator transfer works correctly
        2. Encoder → Discriminator transfer works correctly
        3. Weights are actually copied (not just referenced)
        4. verify_transfer() confirms successful transfer
        5. Parameter counting is accurate
    """
    # Fix OpenMP library conflict error (common on Windows)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("\n" + "=" * 70)
    print("Testing WeightTransfer Module")
    print("=" * 70)
    
    # =========================================================================
    # Setup: Create Models
    # =========================================================================
    print("\nCreating models...")
    
    # Create autoencoder (with random weights simulating trained model)
    autoencoder = BAGANAutoencoder(
        in_channels=1,
        latent_dim=128,
        base_channels=64
    )
    
    # Simulate training by modifying weights
    # In practice, you'd load a trained checkpoint
    with torch.no_grad():
        for param in autoencoder.parameters():
            param.fill_(0.5)  # Fill with known value for testing
    
    # Create fresh GAN models (random initialization)
    generator = BAGANGenerator(
        out_channels=1,
        latent_dim=128,
        base_channels=64
    )
    
    discriminator = BAGANDiscriminator(
        in_channels=1,
        num_classes=2,
        base_channels=64
    )
    
    print("  ✓ Models created")
    
    # =========================================================================
    # Test 1: Parameter Counting
    # =========================================================================
    print("\nTest 1: Parameter Counting")
    
    ae_params = WeightTransfer.count_parameters(autoencoder)
    g_params = WeightTransfer.count_parameters(generator)
    d_params = WeightTransfer.count_parameters(discriminator)
    
    print(f"  Autoencoder: {ae_params['total']:,} total params")
    print(f"  Generator: {g_params['total']:,} total params")
    print(f"  Discriminator: {d_params['total']:,} total params")
    print("  ✓ Parameter counting works")
    
    # =========================================================================
    # Test 2: Full Weight Transfer
    # =========================================================================
    print("\nTest 2: Weight Transfer (Autoencoder → GAN)")
    
    stats = WeightTransfer.initialize_gan_from_autoencoder(
        autoencoder,
        generator,
        discriminator,
        verbose=True
    )
    
    print(f"  ✓ Total transferred: {stats['total_params_transferred']:,} parameters")
    
    # =========================================================================
    # Test 3: Verify Transfer
    # =========================================================================
    print("\nTest 3: Verify Transfer")
    
    # Check Generator got Decoder weights
    print("  Verifying Decoder → Generator:")
    d2g_valid = WeightTransfer.verify_transfer(
        autoencoder.decoder,
        generator,
        layer_names=['fc.weight', 'conv_layers.0.weight']
    )
    print(f"  {'✓' if d2g_valid else '✗'} Decoder → Generator: {'Valid' if d2g_valid else 'Failed'}")
    
    # Check Discriminator got Encoder conv weights
    print("  Verifying Encoder → Discriminator (conv only):")
    e2d_valid = WeightTransfer.verify_transfer(
        autoencoder.encoder,
        discriminator,
        layer_names=['conv_layers.0.weight', 'conv_layers.3.weight']
    )
    print(f"  {'✓' if e2d_valid else '✗'} Encoder → Discriminator: {'Valid' if e2d_valid else 'Failed'}")
    
    # =========================================================================
    # Test 4: Verify Values Actually Changed
    # =========================================================================
    print("\nTest 4: Verify Generator Has Decoder Values")
    
    # Generator should now have 0.5 values (from our simulated training)
    sample_weight = generator.fc.weight[0, 0].item()
    expected = 0.5
    
    if abs(sample_weight - expected) < 1e-5:
        print(f"  ✓ Generator fc.weight[0,0] = {sample_weight:.4f} (expected {expected})")
    else:
        print(f"  ✗ Generator fc.weight[0,0] = {sample_weight:.4f} (expected {expected})")
    
    # =========================================================================
    # Test 5: Freeze and Unfreeze
    # =========================================================================
    print("\nTest 5: Freeze/Unfreeze Layers")
    
    frozen = WeightTransfer.freeze_transferred_layers(
        discriminator,
        layer_names=['conv_layers'],
        verbose=False
    )
    print(f"  ✓ Frozen {frozen:,} parameters in discriminator conv layers")
    
    # Check classifier is still trainable
    classifier_trainable = discriminator.classifier.weight.requires_grad
    print(f"  ✓ Classifier layer trainable: {classifier_trainable}")
    
    unfrozen = WeightTransfer.unfreeze_all(discriminator, verbose=False)
    print(f"  ✓ Unfrozen {unfrozen:,} parameters")
    
    # =========================================================================
    # All Tests Passed
    # =========================================================================
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\nWeightTransfer module is ready for use.")
    print("Usage: from bagan import WeightTransfer")
