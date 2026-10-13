"""casestudy.classifiers.resnet — ResNet backbones (CIFAR-10 / CIFAR-100; plan §6).

Self-contained ResNet implementations (no torchvision-version coupling, no
import of ``ablation``):

- ``resnet32``: CIFAR-style ResNet-32 (BasicBlock, 3 stages × 5 blocks → 6n+2,
  n=5), 3×3 stem, channels 16→32→64, global average pool. Suited to small
  images upscaled to 64×64.
- ``resnet18``: ResNet-18 (BasicBlock [2,2,2,2]), 3×3 stem (no max-pool),
  channels 64→128→256→512.

Both accept ``in_channels`` (3 for CIFAR) and ``num_classes`` (3 here) and
return raw logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

print("casestudy/classifiers/resnet.py loaded")


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (two 3×3 convs + identity/projection skip)."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet(nn.Module):
    """Generic ResNet for small images upscaled to 64×64.

    Args:
        num_blocks: Blocks per stage.
        in_channels: Input image channels (3 for CIFAR).
        num_classes: Output classes.
        stem_planes: Channel count after the stem / first stage.
        widths: Per-stage channel widths.
        strides: Per-stage strides (first stage usually 1).
    """

    def __init__(self, num_blocks, in_channels=3, num_classes=3,
                 stem_planes=16, widths=(16, 32, 64), strides=(1, 2, 2)):
        super().__init__()
        self.num_classes = num_classes
        self.in_planes = stem_planes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_planes),
            nn.ReLU(inplace=True),
        )

        stages = []
        for width, n, stride in zip(widths, num_blocks, strides):
            stages.append(self._make_stage(width, n, stride))
        self.stages = nn.Sequential(*stages)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[-1] * BasicBlock.expansion, num_classes)

    def _make_stage(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet32(in_channels: int = 3, num_classes: int = 3) -> ResNet:
    """CIFAR-style ResNet-32 (6n+2, n=5): 3 stages × 5 BasicBlocks."""
    return ResNet(
        num_blocks=(5, 5, 5), in_channels=in_channels, num_classes=num_classes,
        stem_planes=16, widths=(16, 32, 64), strides=(1, 2, 2),
    )


def resnet18(in_channels: int = 3, num_classes: int = 3) -> ResNet:
    """ResNet-18 (BasicBlock [2,2,2,2]) with a 3×3 stem (no max-pool)."""
    return ResNet(
        num_blocks=(2, 2, 2, 2), in_channels=in_channels, num_classes=num_classes,
        stem_planes=64, widths=(64, 128, 256, 512), strides=(1, 2, 2, 2),
    )

