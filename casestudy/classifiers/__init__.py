"""casestudy.classifiers — shared downstream backbones (plan §6).

Implemented (roadmap Step 3):
- ``cnn.SmallCNN``: small CNN for Fashion-MNIST / BloodMNIST.
- ``resnet.resnet32`` / ``resnet.resnet18``: ResNet backbones for CIFAR.
- ``trainer``: shared training/eval loop + ``build_classifier`` factory +
  ``EarlyStopping`` (early-stop on validation macro-F1).
"""

from __future__ import annotations

from .cnn import SmallCNN
from .resnet import ResNet, BasicBlock, resnet18, resnet32
from .trainer import (
    EarlyStopping,
    build_classifier,
    train_classifier,
    evaluate_classifier,
)

__all__ = [
    "SmallCNN",
    "ResNet",
    "BasicBlock",
    "resnet18",
    "resnet32",
    "EarlyStopping",
    "build_classifier",
    "train_classifier",
    "evaluate_classifier",
]
