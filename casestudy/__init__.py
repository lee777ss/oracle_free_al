"""casestudy — standalone package for the open-source data case studies (CS1J).

Self-contained benchmarking of Ab-8c against SOTA imbalanced-learning methods
on Fashion-MNIST, CIFAR-10, CIFAR-100, and BloodMNIST.  This package does NOT
import ``ablation`` or ``bagan``; GAAL_Ablation_130J is used only as the
algorithmic reference for the standalone Ab-8c re-implementation.

Blueprint: ``110_Plan for Open source data case studies.ipynb``.
"""

from __future__ import annotations

from .config import (
    CaseStudyConfig,
    DatasetSpec,
    METHOD_ORDER,
    METHOD_LABELS,
    GENERATIVE_METHODS,
    METRICS,
)

print("casestudy package loaded")

__all__ = [
    "CaseStudyConfig",
    "DatasetSpec",
    "METHOD_ORDER",
    "METHOD_LABELS",
    "GENERATIVE_METHODS",
    "METRICS",
]
