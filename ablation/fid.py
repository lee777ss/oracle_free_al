"""Per-class FID helpers for Ab-11a diagnostics.

This module backs the optional Ab-11a FID monitor in ``trainer.py`` and the
notebook Step H cache-building workflow. It reuses ``shift_metrics`` for the
InceptionV3 feature extractor and Frechet distance implementation.
"""

from __future__ import annotations

import os
from typing import Mapping

import numpy as np
import torch

from .shift_metrics import _extract_features, _frechet_distance, _get_inception

print("fid.py loaded")


def _ref_cache_path(class_id: int, ref_cache_dir: str, date_suffix: str | None) -> str:
    name = (
        f"fid_ref_class_{class_id}_{date_suffix}.npz"
        if date_suffix
        else f"fid_ref_class_{class_id}.npz"
    )
    return os.path.join(ref_cache_dir, name)


def _load_ref_stats(path: str) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path) as handle:
        mu = handle["mu"].astype(np.float64)
        sigma = handle["sigma"].astype(np.float64)
        n = int(handle["n"]) if "n" in handle.files else -1
    return mu, sigma, n


def _save_ref_stats(path: str, mu: np.ndarray, sigma: np.ndarray, n: int) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez(path, mu=mu.astype(np.float64), sigma=sigma.astype(np.float64), n=np.int64(n))


def _stats_from_images(
    images: torch.Tensor,
    inception: torch.nn.Module,
    device: str | torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    feats = _extract_features(images, inception, device, batch_size)
    return feats.mean(axis=0), np.cov(feats, rowvar=False), int(feats.shape[0])


def build_real_reference_cache(
    real_images_by_class: Mapping[int, torch.Tensor],
    device: str | torch.device,
    ref_cache_dir: str = "saved",
    date_suffix: str | None = None,
    batch_size: int = 32,
    overwrite: bool = False,
) -> dict[int, str]:
    """Build/load cached real Inception feature statistics for each class."""
    inception = _get_inception(device)
    paths: dict[int, str] = {}

    for class_id, images in real_images_by_class.items():
        class_int = int(class_id)
        path = _ref_cache_path(class_int, ref_cache_dir, date_suffix)
        if not overwrite and os.path.isfile(path):
            paths[class_int] = path
            continue

        mu, sigma, n = _stats_from_images(images, inception, device, batch_size)
        _save_ref_stats(path, mu, sigma, n)
        paths[class_int] = path

    return paths


@torch.no_grad()
def _generate_class_samples(
    generator: torch.nn.Module,
    latent_generator,
    class_id: int,
    n_samples: int,
    device: str | torch.device,
    gen_batch_size: int,
) -> torch.Tensor:
    was_training = generator.training
    generator.eval()
    chunks: list[torch.Tensor] = []
    remaining = int(n_samples)

    while remaining > 0:
        batch_size = min(gen_batch_size, remaining)
        z = latent_generator.sample(class_label=class_id, batch_size=batch_size).to(device)
        chunks.append(generator(z).detach().cpu())
        remaining -= batch_size

    generator.train(mode=was_training)
    return torch.cat(chunks, dim=0)


def fid_per_class(
    generator: torch.nn.Module,
    latent_generator,
    classes,
    n_samples: int = 10_000,
    device: str | torch.device = "cuda",
    real_images_by_class: Mapping[int, torch.Tensor] | None = None,
    ref_cache_dir: str = "saved",
    date_suffix: str | None = None,
    batch_size: int = 32,
    gen_batch_size: int = 64,
) -> dict[int, float]:
    """Compute per-class FID from generated samples to cached real references."""
    inception = _get_inception(device)
    out: dict[int, float] = {}

    for class_id in classes:
        class_int = int(class_id)
        path = _ref_cache_path(class_int, ref_cache_dir, date_suffix)

        if os.path.isfile(path):
            mu_real, sigma_real, _ = _load_ref_stats(path)
        elif real_images_by_class is not None and class_int in real_images_by_class:
            mu_real, sigma_real, n_real = _stats_from_images(
                real_images_by_class[class_int], inception, device, batch_size
            )
            _save_ref_stats(path, mu_real, sigma_real, n_real)
        else:
            raise FileNotFoundError(
                f"FID reference for class {class_int} not found at {path!r}; "
                "build the real-reference cache first."
            )

        fake_images = _generate_class_samples(
            generator, latent_generator, class_int, n_samples, device, gen_batch_size
        )
        mu_fake, sigma_fake, _ = _stats_from_images(fake_images, inception, device, batch_size)
        out[class_int] = _frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    return out


@torch.no_grad()
def ae_fid_per_class(
    autoencoder,
    latent_generator,
    classes,
    n_samples: int = 1000,
    device: str | torch.device = "cuda",
    real_images_by_class: Mapping[int, torch.Tensor] | None = None,
    ref_cache_dir: str = "saved",
    date_suffix: str | None = None,
    batch_size: int = 32,
    gen_batch_size: int = 64,
) -> dict[int, float]:
    """Per-class FID of AUTOENCODER-decoded class samples vs cached real references.

    The autoencoder's decoder shares the ``(latent) -> image`` signature of the GAN
    generator, so this simply forwards the decoder to :func:`fid_per_class` and reuses
    the same InceptionV3 real-reference cache. ``autoencoder`` may be a full
    ``BAGANAutoencoder`` (its ``.decoder`` is used) or a bare decoder module.
    """
    decoder = getattr(autoencoder, "decoder", autoencoder)
    return fid_per_class(
        generator=decoder,
        latent_generator=latent_generator,
        classes=classes,
        n_samples=n_samples,
        device=device,
        real_images_by_class=real_images_by_class,
        ref_cache_dir=ref_cache_dir,
        date_suffix=date_suffix,
        batch_size=batch_size,
        gen_batch_size=gen_batch_size,
    )