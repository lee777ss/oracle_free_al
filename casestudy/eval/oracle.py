"""casestudy.eval.oracle - the reference classifier that audits synthetic labels (KBS revision, S5).

An ensemble of ImageNet-pretrained ResNet-18 networks, fine-tuned on every official training
image of a dataset's three classes. It takes no part in any method's training; it only labels
candidates afterwards, so that "does the internal classifier's confidence predict whether a
synthetic label is correct?" can be answered by a judge independent of that classifier.

Inputs are the 64 x 64 images in [-1, 1] that every method works with; the network upsamples
them to 128 x 128 and applies ImageNet normalisation. Trained by ``kbs/oracle.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def oracle_dir(dataset: str) -> Path:
    return Path(os.environ.get("KBS_SAVE_ROOT", "saved_kbs")) / "oracle" / dataset


class RefNet(nn.Module):
    """ResNet-18 for 64 x 64 inputs in [-1, 1] (grey or colour)."""

    def __init__(self, num_classes: int, pretrained: bool = True, size: int = 128):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        self.size = size
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self._prep(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The 512-d pooled features that feed the final layer (used for the diversity audit)."""
        n = self.net
        h = n.maxpool(n.relu(n.bn1(n.conv1(self._prep(x)))))
        h = n.layer4(n.layer3(n.layer2(n.layer1(h))))
        return torch.flatten(n.avgpool(h), 1)


class Oracle:
    """Mean softmax of the ensemble members; returns probabilities on the CPU."""

    def __init__(self, members: List[RefNet], device: str):
        self.members, self.device = members, device

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, batch: int = 256) -> torch.Tensor:
        out = []
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(self.device)
            out.append(torch.stack([torch.softmax(m(xb), dim=1) for m in self.members]).mean(0).cpu())
        return torch.cat(out) if out else torch.empty(0)


def load_oracle(dataset: str, device: str) -> Optional[Oracle]:
    """The trained ensemble for ``dataset``, or None if ``kbs/oracle.py`` has not produced it yet."""
    d = oracle_dir(dataset)
    summary = d / "summary.json"
    if not summary.exists():
        return None
    info = json.loads(summary.read_text(encoding="utf-8"))
    members = []
    for name in info["members"]:
        m = RefNet(info["num_classes"], pretrained=False)
        m.load_state_dict(torch.load(d / name, map_location="cpu", weights_only=True))
        members.append(m.to(device).eval())
    return Oracle(members, device)
