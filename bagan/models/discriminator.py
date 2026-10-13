"""
BAGAN Discriminator

Discriminator network with (n+1) output classification.
- n real class outputs (class_0, class_1, ..., class_{n-1})
- 1 fake output (class_n = fake)

Key differences from AC-GAN:
- AC-GAN: Dual head (real/fake sigmoid + class softmax)
- BAGAN: Single (n+1) head with CrossEntropyLoss

The conv layers match BAGANEncoder for weight transfer from pre-trained autoencoder.

Reference: Mariani et al. (2018) Section 3.2 - "(n+1) class discriminator"
"We train the discriminator D to learn to classify samples in n+1 classes:
the n known classes plus an additional 'fake' class"
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class BAGANDiscriminator(nn.Module):
    """
    BAGAN Discriminator: Image (1, 64, 64) → (n+1) class logits
    
    Architecture:
        Conv layers (SAME as BAGANEncoder for weight transfer):
            Conv1: 1 → 64, stride=2 → (64, 32, 32)      # same as encoder
            Conv2: 64 → 128, stride=2 → (128, 16, 16)   # same as encoder
            Conv3: 128 → 256, stride=2 → (256, 8, 8)    # same as encoder
            Conv4: 256 → 512, stride=2 → (512, 4, 4)    # same as encoder
        Flatten: 512 × 4 × 4 = 8192                     # same as encoder
        Classification head: 8192 → (n+1) logits        # ONLY DIFFERENT LAYER #
    
    Output interpretation:
        - logits[0]: class 0 (majority)
        - logits[1]: class 1 (minority)
        - logits[2]: fake class
    
    Training labels:
        - Real class 0 images → target = 0
        - Real class 1 images → target = 1
        - Generated images → target = 2 (fake)
    
    CRITICAL: Output is LOGITS (no softmax!)
    Use nn.CrossEntropyLoss which expects raw logits.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 64,
        dropout_rate: float = 0.0
    ):
        """
        Initialize BAGAN Discriminator.
        
        Args:
            in_channels: Number of input image channels (1 for grayscale)
            num_classes: Number of real classes (n). Output will be n+1.
            base_channels: Base number of channels (64)
            dropout_rate: Dropout2d rate applied between conv blocks (0 = disabled)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_outputs = num_classes + 1  # n classes + 1 fake
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate
        
        # Convolutional layers: IDENTICAL to BAGANEncoder (for weight transfer)
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
        
        # Classification head: 8192 → (n+1)
        # This is the ONLY layer that differs from Encoder
        # Encoder: 8192 → latent_dim (128)
        # Discriminator: 8192 → num_outputs (3)
        self.classifier = nn.Linear(base_channels * 8 * 4 * 4, self.num_outputs)
        
        # NO activation here! Output is raw logits.
        # CrossEntropyLoss expects logits, not probabilities.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify image as one of (n+1) classes.
        
        Args:
            x: Input image tensor of shape (batch, 1, 64, 64)
        
        Returns:
            Logits of shape (batch, n+1) where:
                - logits[:, 0:n] = real class logits
                - logits[:, n] = fake class logit
            
            IMPORTANT: These are RAW LOGITS, not probabilities!
            Use CrossEntropyLoss directly, or apply softmax for probabilities.
        """
        # Convolutional feature extraction with optional dropout between blocks.
        # Each block is 3 Sequential children: Conv2d + BatchNorm2d + LeakyReLU.
        # Dropout2d is applied BETWEEN blocks (not inside conv_layers) so
        # conv_layers state_dict keys stay identical to BAGANEncoder.
        if self.dropout_rate > 0 and self.training:
            layers = list(self.conv_layers.children())
            block_size = 3  # Conv2d + BN + LeakyReLU
            features = x
            for i in range(0, len(layers), block_size):
                for layer in layers[i:i + block_size]:
                    features = layer(features)
                if i + block_size < len(layers):  # dropout between blocks, not after last
                    features = nn.functional.dropout2d(features, p=self.dropout_rate, training=True)
        else:
            features = self.conv_layers(x)  # (batch, 512, 4, 4)
        
        # Flatten
        features = features.view(features.size(0), -1)  # (batch, 8192)
        
        # Classification (no activation!)
        logits = self.classifier(features)  # (batch, n+1)
        
        return logits
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities (softmax of logits).
        
        Use this for inference/visualization, NOT for training loss.
        
        Args:
            x: Input image tensor
        
        Returns:
            Probabilities of shape (batch, n+1), summing to 1
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predicted class and confidence.
        
        Args:
            x: Input image tensor
        
        Returns:
            Tuple of:
                - predicted_class: (batch,) tensor of class indices
                - confidence: (batch,) tensor of prediction confidence
        """
        probs = self.get_probabilities(x)
        confidence, predicted_class = torch.max(probs, dim=1)
        return predicted_class, confidence
    
    def is_real(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check if discriminator thinks images are real (any real class).
        
        Args:
            x: Input image tensor
        
        Returns:
            Boolean tensor (batch,) - True if predicted as real class
        """
        predicted_class, _ = self.predict(x)
        return predicted_class < self.num_classes  # Real classes are 0 to n-1
    
    def load_encoder_weights(self, encoder: nn.Module) -> dict:
        """
        Load conv layer weights from a trained BAGANEncoder.
        
        This is a key step in BAGAN initialization:
        After autoencoder training, encoder conv weights are transferred
        to discriminator's conv layers. Only the classifier head is new.
        
        Args:
            encoder: Trained BAGANEncoder instance
        
        Returns:
            dict: Transfer statistics
        
        Note:
            Only conv_layers weights are transferred.
            The classifier head (8192 → n+1) is randomly initialized
            because Encoder has a different head (8192 → latent_dim).
        """
        encoder_state = encoder.state_dict()
        discriminator_state = self.state_dict()
        
        # We only transfer conv_layers, not fc (which becomes classifier)
        transferred = 0
        skipped = 0
        
        for key in encoder_state:
            if key.startswith('conv_layers'):
                if key in discriminator_state:
                    if encoder_state[key].shape == discriminator_state[key].shape:
                        discriminator_state[key] = encoder_state[key]
                        transferred += 1
                    else:
                        print(f"Shape mismatch for {key}: "
                              f"encoder {encoder_state[key].shape} vs "
                              f"discriminator {discriminator_state[key].shape}")
                        skipped += 1
            else:
                # Skip fc layer (encoder has latent projection, we have classifier)
                skipped += 1
        
        self.load_state_dict(discriminator_state)
        
        # Count parameters
        conv_params = sum(
            p.numel() for name, p in self.named_parameters() 
            if 'conv_layers' in name
        )
        classifier_params = sum(
            p.numel() for name, p in self.named_parameters() 
            if 'classifier' in name
        )
        
        return {
            'transferred_keys': transferred,
            'skipped_keys': skipped,
            'conv_parameters': conv_params,
            'classifier_parameters': classifier_params,
            'classifier_randomly_initialized': True,
            'success': True
        }


# Convenience function for creating discriminator from config
def create_discriminator_from_config(config) -> BAGANDiscriminator:
    """
    Create BAGANDiscriminator from BAGANConfig.
    
    Args:
        config: BAGANConfig instance
    
    Returns:
        BAGANDiscriminator instance
    """
    return BAGANDiscriminator(
        in_channels=config.img_channels,
        num_classes=config.num_classes,
        base_channels=config.encoder_channels[0],  # 64
        dropout_rate=getattr(config, 'd_dropout_rate', 0.0)
    )


# Loss function for BAGAN discriminator
class BAGANDiscriminatorLoss(nn.Module):
    """
    Loss function for BAGAN Discriminator training.
    
    Uses CrossEntropyLoss with (n+1) class labels:
        - Real images: label = actual class (0 or 1)
        - Fake images: label = n (fake class)
    
    IMPORTANT: The discriminator itself doesn't "know" if an image is real or fake!
    It simply outputs 3 logits for ANY input image. The distinction between
    real and fake happens in this loss function:
    
        Training loop (pseudocode):
            # 1. Get real images from dataset
            real_images, real_labels = next(dataloader)  # labels are 0 or 1
            
            # 2. Generate fake images
            fake_images = generator(z)
            
            # 3. Discriminator treats all images the same way
            [real_logits[0], real_logits[1], real_logits[2]] = real_logits = discriminator(real_images)  # D doesn't know these are real
            [fake_logits[0], fake_logits[1], fake_logits[2]] = fake_logits = discriminator(fake_images)  # D doesn't know these are fake
            
            # 4. THIS loss function applies different TARGET labels
            loss = loss_fn(real_logits, real_labels, fake_logits)
            #                          ↑ target: 0 or 1    ↑ target: 2 (fake)
    
    The discriminator LEARNS to distinguish real vs fake through backpropagation,
    not by being told which is which at inference time.
    
    Reference: Mariani et al. (2018)
    "sparse categorical cross-entropy as D loss"
    """
    
    def __init__(self, num_classes: int = 2, label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.fake_label = num_classes  # Fake class is the (n+1)th class (index 2)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(
        self, 
        real_logits: torch.Tensor, 
        real_labels: torch.Tensor,
        fake_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute discriminator loss.
        
        Args:
            real_logits: Discriminator output for real images (batch, n+1)
                        NOTE: "real" here means images FROM DATASET, not that
                        discriminator identified them as real. D just sees images.
            real_labels: True class labels for real images (batch,) with values 
                        in [0, n-1], not one-hot encoded. These are ground truth
                        labels from the dataset.
            fake_logits: Discriminator output for fake images (batch, n+1)
                        NOTE: "fake" here means images FROM GENERATOR.
                        D processes these exactly like real images.
            
            ** real_logits and fake_logits are 3 digitis each (for n=2). **
            ** real_labels are either 0 or 1. **
        
        Returns:
            Tuple of:
                - total_loss: Combined loss
                - real_loss: Loss on real images (target = actual class 0 or 1)
                - fake_loss: Loss on fake images (target = 2, the fake class)
        """
        # Real images should be classified as their true class (0 or 1)
        # This teaches D to recognize real class-0 and class-1 images
        # [real_logits[0], real_logits[1], real_logits[2]] = real_logits
        # real_labels are in [0, 1]
        real_loss = self.criterion(real_logits, real_labels)
        
        # Fake images should be classified as fake (class index 2)
        # This teaches D to recognize generated images as fake
        batch_size = fake_logits.size(0)
        fake_labels = torch.full(
            (batch_size,), 
            self.fake_label,  # = 2 (the fake class index)
            dtype=torch.long, 
            device=fake_logits.device
        )

        # [fake_logits[0], fake_logits[1], fake_logits[2]] = fake_logits
        # fake_labels = 2
        fake_loss = self.criterion(fake_logits, fake_labels)
        
        # Total loss: D wants to correctly classify both real AND fake
        total_loss = real_loss + fake_loss
        
        return total_loss, real_loss, fake_loss


class BAGANGeneratorLoss(nn.Module):
    """
    Loss function for BAGAN Generator training.
    
    Generator wants discriminator to classify fake images as REAL classes,
    not as the fake class. 
    The target class should match the "class-conditioned latent vector" 
    used to generate the image.
    
    Reference: Mariani et al. (2018)
    "The generator is trained to fool D into categorizing its 
    output as belonging to one of the n known classes"
    """
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        fake_logits: torch.Tensor, 
        target_class: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute generator loss.
        
        Args:
            fake_logits: Discriminator output for fake images (batch, n+1)
            target_class: Intended class for each fake image (batch,)
                         Values should be in [0, n-1] (real classes, not fake)
        
        Returns:
            Generator loss (scalar)
        """
        # Generator wants fake images classified as the target real class
        return self.criterion(fake_logits, target_class)
