"""casestudy.methods.base — uniform augmentation-method interface (plan §4.1).

Every competing method subclasses :class:`AugmentationMethod` and implements
``fit`` / ``synthesize`` so the notebook's per-method loop is identical:

    method = build_method("m08_bagan", config, spec, device, seed)
    method.fit(X_real, y_real)                      # train-pool partition only
    X_syn, y_syn = method.synthesize(targets)       # NEW samples to add

        Conventions (match the case-study data layer):
        - ``X_real``: float tensor ``[N, C, 64, 64]`` in ``[-1, 1]``.
        - ``y_real``: long tensor ``[N]`` with **contiguous** labels ``{0..K-1}``
            (0 = majority, 1..K-1 = minorities).
    - ``synthesize(target_counts)``: ``target_counts`` is the desired *final*
      count per class; the method returns only the **new synthetic samples**
      needed to reach those counts (``max(0, target - current)`` per class).
      The baseline returns empty tensors.

``is_generative`` flags whether per-class FID applies (GAN methods) vs. the
baseline / classical resamplers (which report "—" for FID, plan §7).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch

print("casestudy/methods/base.py loaded")


def balance_targets(y_real: torch.Tensor, majority_label: int = 0) -> Dict[int, int]:
    """Final per-class target = the majority count (fill minorities up to it).

    Mirrors 130J's balancing convention by filling every minority class up to
    the majority count.
    """
    vals, counts = torch.unique(y_real, return_counts=True)
    counts_by_class = {int(v): int(c) for v, c in zip(vals, counts)}
    majority_count = counts_by_class.get(majority_label, max(counts_by_class.values()))
    return {c: majority_count for c in counts_by_class}


def balance_targets_ratio(y_real: torch.Tensor, ratio="full",
                          majority_label: int = 0) -> Dict[int, int]:
    """Fair per-class targets with a synthetic:real ratio cap.

    ``ratio="full"`` (or ``None``) reproduces :func:`balance_targets` (fill every
    minority up to the majority count). A numeric ``r`` sets each minority target
    to ``min(majority_count, max(real_c, round(r * real_c)))`` — at most ``r x``
    its real count, never below its real count, never above the majority. The
    majority class keeps its own count (no synthesis). Applied identically to
    every method, so it stays a fair benchmark knob.
    """
    vals, counts = torch.unique(y_real, return_counts=True)
    counts_by_class = {int(v): int(c) for v, c in zip(vals, counts)}
    majority_count = counts_by_class.get(majority_label, max(counts_by_class.values()))
    if ratio is None or (isinstance(ratio, str) and ratio.lower() == "full"):
        return {c: majority_count for c in counts_by_class}
    r = float(ratio)
    out: Dict[int, int] = {}
    for c, n in counts_by_class.items():
        if c == majority_label:
            out[c] = n
        else:
            out[c] = int(min(majority_count, max(n, round(r * n))))
    return out


class AugmentationMethod(ABC):
    """Abstract base for every minority-augmentation method (§4.1 / §5)."""

    name: str = "base"
    is_generative: bool = False

    def __init__(self, config, spec, device: str = "cpu", seed: int = 42):
        self.config = config
        self.spec = spec
        self.device = device
        self.seed = seed
        self.channels = spec.channels
        self.image_size = config.image_size
        self.latent_dim = config.latent_dim
        self.X_real: torch.Tensor | None = None
        self.y_real: torch.Tensor | None = None

    @property
    def num_classes(self) -> int:
        if self.y_real is None:
            if not self.spec.native_imbalance and self.spec.selected_classes:
                return len(self.spec.selected_classes)
            return 3
        return int(self.y_real.max().item()) + 1

    @abstractmethod
    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        """Train the generator / fit the sampler on the given (train-pool) data."""

    @abstractmethod
    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return only the NEW synthetic samples needed to reach ``target_counts``."""

    # ── shared helpers ────────────────────────────────────────────────
    def _deficits(self, target_counts: Dict[int, int]) -> Dict[int, int]:
        """Per-class number of NEW samples to generate (``target - current``)."""
        assert self.y_real is not None, "fit() must be called before synthesize()"
        vals, counts = torch.unique(self.y_real, return_counts=True)
        have = {int(v): int(c) for v, c in zip(vals, counts)}
        out: Dict[int, int] = {}
        for c, target in target_counts.items():
            deficit = int(target) - have.get(int(c), 0)
            if deficit > 0:
                out[int(c)] = deficit
        return out

    def _empty(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Empty ``(X, y)`` with correct dims/dtypes (no synthesis)."""
        x = torch.empty(0, self.channels, self.image_size, self.image_size)
        y = torch.empty(0, dtype=torch.long)
        return x, y


# ── Registry ──────────────────────────────────────────────────────────────
def build_method(key: str, config, spec, device: str = "cpu", seed: int = 42) -> AugmentationMethod:
    """Instantiate a method by its ``method_order`` key (e.g. ``"m08_bagan"``)."""
    from . import (
        m00_imbalanced, m01_smote, m02_bsmote, m03_adasyn, m04_dcgan,
        m05_cgan, m06_acgan, m07_wgangp, m08_bagan, m09_bagan_gp, m10_ab8c,
    )

    registry = {
        "m00_imbalanced": m00_imbalanced.ImbalancedBaseline,
        "m01_smote": m01_smote.SMOTEMethod,
        "m02_bsmote": m02_bsmote.BorderlineSMOTEMethod,
        "m03_adasyn": m03_adasyn.ADASYNMethod,
        "m04_dcgan": m04_dcgan.DCGANMethod,
        "m05_cgan": m05_cgan.CGANMethod,
        "m06_acgan": m06_acgan.ACGANMethod,
        "m07_wgangp": m07_wgangp.WGANGPMethod,
        "m08_bagan": m08_bagan.BAGANMethod,
        "m09_bagan_gp": m09_bagan_gp.BAGANGPMethod,
        "m10_ab8c": m10_ab8c.Ab8cMethod,
        "m11_imbalanced_naive": m00_imbalanced.ImbalancedBaselineNaive,
    }
    if key not in registry:
        raise KeyError(f"Unknown method key {key!r}; known: {sorted(registry)}")
    return registry[key](config, spec, device, seed)
