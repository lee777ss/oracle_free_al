"""casestudy.eval.fid — per-class Fréchet Inception Distance (plan §7).

Standalone re-implementation (no ``ablation`` import) of FID between real and
generated minority samples, using pretrained InceptionV3 ``pool_3`` features
(2048-D). Accepts case-study image tensors ``[N, C, H, W]`` in ``[-1, 1]`` with
``C in {1, 3}``; internally maps to ``[0, 1]``, expands grayscale to RGB, and
resizes to 299×299 for Inception.

Reference: Heusel et al. (2017), "GANs Trained by a Two Time-Scale Update Rule
Converge to a Local Nash Equilibrium" (https://arxiv.org/abs/1706.08500).
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("casestudy/eval/fid.py loaded")

_INCEPTION: nn.Module | None = None


def _get_inception(device: str | torch.device) -> nn.Module:
    """Lazy singleton InceptionV3 pool_3 (2048-D) feature extractor."""
    global _INCEPTION
    if _INCEPTION is None:
        from torchvision.models import Inception_V3_Weights, inception_v3

        model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True,
            transform_input=True,
        )
        model.fc = nn.Identity()
        model.AuxLogits = None
        model.eval()
        _INCEPTION = model
    _INCEPTION.to(device)
    return _INCEPTION


def _prep_for_inception(x: torch.Tensor) -> torch.Tensor:
    """``[N, C, H, W]`` in ``[-1, 1]`` -> ``[N, 3, 299, 299]`` in ``[0, 1]``."""
    if x.dim() != 4:
        raise ValueError(f"expected 4D tensor, got shape {tuple(x.shape)}")
    x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] != 3:
        raise ValueError(f"expected 1 or 3 channels, got {x.shape[1]}")
    if x.shape[-1] != 299 or x.shape[-2] != 299:
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return x


@torch.no_grad()
def _extract_features(
    images: torch.Tensor,
    model: nn.Module,
    device: str | torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Run Inception in batches; return ``[N, 2048]`` float64 array."""
    feats: list[np.ndarray] = []
    n = images.shape[0]
    for i in range(0, n, batch_size):
        batch = images[i : i + batch_size].to(device)
        batch = _prep_for_inception(batch)
        out = model(batch)
        if isinstance(out, tuple):  # safety; aux disabled
            out = out[0]
        feats.append(out.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(feats, axis=0)


def _frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """``||mu1-mu2||^2 + Tr(s1 + s2 - 2*sqrt(s1@s2))`` with a stability fix."""
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"sqrtm produced large imaginary part: {m}")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: str | torch.device,
    batch_size: int = 32,
) -> float:
    """FID between two image tensors in ``[-1, 1]``. Lower is better."""
    model = _get_inception(device)
    feats_r = _extract_features(real_images, model, device, batch_size)
    feats_f = _extract_features(fake_images, model, device, batch_size)
    mu_r, mu_f = feats_r.mean(axis=0), feats_f.mean(axis=0)
    sigma_r = np.cov(feats_r, rowvar=False)
    sigma_f = np.cov(feats_f, rowvar=False)
    return _frechet_distance(mu_r, sigma_r, mu_f, sigma_f)


def fid_per_class(
    real_by_class: Mapping[int, torch.Tensor],
    fake_by_class: Mapping[int, torch.Tensor],
    device: str | torch.device,
    batch_size: int = 32,
) -> Dict[int, float]:
    """Per-class FID between generated and real samples.

    Args:
        real_by_class: ``{class_id: real_images[-1,1]}`` (e.g. minority classes).
        fake_by_class: ``{class_id: generated_images[-1,1]}`` for the same classes.
        device: torch device for Inception.
        batch_size: Inception batch size.

    Returns:
        ``{class_id: fid}`` for every class present in both mappings.
    """
    model = _get_inception(device)
    out: Dict[int, float] = {}
    for class_id, real in real_by_class.items():
        if class_id not in fake_by_class:
            continue
        feats_r = _extract_features(real, model, device, batch_size)
        feats_f = _extract_features(fake_by_class[class_id], model, device, batch_size)
        mu_r, mu_f = feats_r.mean(axis=0), feats_f.mean(axis=0)
        sigma_r = np.cov(feats_r, rowvar=False)
        sigma_f = np.cov(feats_f, rowvar=False)
        out[int(class_id)] = _frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
    return out
