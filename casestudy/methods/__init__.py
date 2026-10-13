"""casestudy.methods — one module per competing augmentation method (plan §5).

Uniform interface (``base.AugmentationMethod``):
    fit(X_real, y_real) -> None          # train generator / fit sampler
    synthesize(target_counts) -> (X, y)  # produce balanced minority set

Module map (fixed order):
    m00_imbalanced  Ab-1 baseline (no augmentation)
    m01_smote       SMOTE
    m02_bsmote      Borderline-SMOTE
    m03_adasyn      ADASYN
    m04_dcgan       DCGAN
    m05_cgan        cGAN
    m06_acgan       ACGAN
    m07_wgangp      WGAN-GP
    m08_bagan       BAGAN
    m09_bagan_gp    BAGAN-GP
    m10_ab8c        Ab-8c (proposed; BADGE + 3-player co-training)

Populated in roadmap Steps 5 (m00–m09) and 6 (m10).
"""

from .base import (
    AugmentationMethod, balance_targets, balance_targets_ratio, build_method,
)

__all__ = [
    "AugmentationMethod",
    "balance_targets",
    "balance_targets_ratio",
    "build_method",
]
