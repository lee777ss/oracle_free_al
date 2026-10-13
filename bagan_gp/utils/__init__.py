"""bagan_gp.utils — public utility API.

Mirrors :mod:`bagan.utils` and adds the three discriminator-adapter helpers
(``realism_score``, ``adv_class_logits``, ``discriminator_features``) that
:mod:`ablation.gan_backend` uses to abstract over the two GAN backends.
"""

import torch

# Re-export visualisation utilities unchanged (visualisation is backend-agnostic).
from bagan.utils.visualization import BAGANVisualizer, show_samples, show_latent_space

from .weight_transfer import WeightTransfer

print("bagan_gp/utils/__init__.py loaded")


def _parse_discriminator_output(discriminator, x: torch.Tensor):
    """Normalize discriminator outputs across BAGAN-GP and BAGAN variants.

    Returns:
        score_like: Tensor [B], higher means more real-like before sign handling.
        class_logits: Tensor [B, C] (real classes only).
    """
    out = discriminator(x)

    # WGAN-GP path: (critic_score[B], aux_logits[B, C])
    if isinstance(out, tuple):
        if len(out) < 2:
            raise ValueError("Discriminator tuple output must contain at least 2 elements")
        score, aux = out[0], out[1]
        return score, aux

    # Vanilla BAGAN path: logits[B, C+1] with final column = fake class.
    logits = out
    if logits.ndim != 2:
        raise ValueError(f"Expected discriminator logits with shape [B, K], got {tuple(logits.shape)}")

    if logits.size(1) < 2:
        raise ValueError(
            "Expected at least 2 discriminator outputs (real classes + fake class)"
        )

    p_fake = torch.softmax(logits, dim=1)[:, -1]
    class_logits = logits[:, :-1]
    # Convert to "higher is more real-like" score for shared handling.
    score = -p_fake
    return score, class_logits


# ---------------------------------------------------------------------- #
# Backend-adapter helpers (BAGAN-GP-specific)
# ---------------------------------------------------------------------- #
def realism_score(discriminator, x: torch.Tensor) -> torch.Tensor:
    """**Lower = more real.**

    For the WGAN-GP critic, the realism signal is the scalar critic head.
    Higher critic output ⇒ more "real-like" (Wasserstein convention), so we
    return ``-score`` to keep the lower-is-better convention shared with the
    BAGAN backend (where BAGAN returns ``P(fake)``).
    """
    score, _ = _parse_discriminator_output(discriminator, x)
    return -score


def adv_class_logits(discriminator, x: torch.Tensor) -> torch.Tensor:
    """Class logits suitable for ``cross_entropy(..., target ∈ [0, n-1])``.

    For BAGAN-GP this is the auxiliary head's output, shape ``[B, num_classes]``.
    (Vanilla BAGAN's equivalent helper returns the full ``[B, num_classes + 1]``
    tensor to preserve 106J softmax-normalisation equivalence.)
    """
    _, aux = _parse_discriminator_output(discriminator, x)
    return aux


def discriminator_features(discriminator, x: torch.Tensor) -> torch.Tensor:
    """Flattened penultimate features, shape ``[B, D]``.

    For BAGAN-GP the conv stack is ``discriminator.features`` (the WGAN-GP
    critic has no BatchNorm but the spatial layout matches BAGAN's encoder
    exactly, so the output dim is the same 8192).
    """
    h = discriminator.features(x)
    return h.view(h.size(0), -1)


__all__ = [
    "WeightTransfer",
    "BAGANVisualizer",
    "show_samples",
    "show_latent_space",
    "realism_score",
    "adv_class_logits",
    "discriminator_features",
]
