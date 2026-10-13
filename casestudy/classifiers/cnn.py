"""casestudy.classifiers.cnn — small CNN backbone (plan §6).

Standalone re-implementation of the 130J ``CNNClassifier`` (no import of
``ablation``).  Four strided conv blocks (channels 16→32→64→128), LeakyReLU(0.2),
no BatchNorm, FC bottleneck 2048→64→num_classes with dropout.  Input is the
shared 64×64 resolution; ``input_channels`` follows the dataset (1 grayscale,
3 color).  Returns raw logits (use with ``CrossEntropyLoss``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

print("casestudy/classifiers/cnn.py loaded")


class SmallCNN(nn.Module):
    """Small 4-block CNN for 64×64 inputs (Fashion-MNIST / BloodMNIST).

    Args:
        input_channels: 1 (grayscale) or 3 (color).
        num_classes: Number of output classes (3 for the case studies).
        dropout_rate: Dropout probability on the FC bottleneck.
    """

    def __init__(self, input_channels: int = 1, num_classes: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Feature extractor: 64→32→16→8→4 (stride-2 convs, no BatchNorm).
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # Classifier head: 128*4*4 = 2048 → 64 → num_classes.
        self.feature_size = 128 * 4 * 4
        self.fc1 = nn.Linear(self.feature_size, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate 64-dim features (post-fc1 activation, pre-dropout/fc2)."""
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.view(x.size(0), -1)
        return self.activation(self.fc1(x))
