"""casestudy.methods.m00_imbalanced — Ab-1 baseline (no augmentation, §5.1).

Control method: the classifier trains on the raw imbalanced pool, so
``synthesize`` produces nothing. Establishes the no-augmentation floor
(validation gate d: must reproduce 130J's no-aug result on Fashion-MNIST).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from .base import AugmentationMethod

print("casestudy/methods/m00_imbalanced.py loaded")


class ImbalancedBaseline(AugmentationMethod):
    name = "Imbalanced (Ab-1)"
    is_generative = False

    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        self.X_real, self.y_real = X_real, y_real

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._empty()


class ImbalancedBaselineNaive(ImbalancedBaseline):
    """Second baseline: identical to Ab-1 (no augmentation) but the downstream
    classifier is model-selected on validation ACCURACY instead of macro-F2.
    On imbalanced data the best-accuracy epoch favors the majority class and
    ignores minorities -> low macro-F2, giving a genuinely weak, majority-biased
    'naive practitioner' control. The accuracy monitor is enforced by the runner
    via ``config.accuracy_baseline_methods`` / ``config.selection_metric_for``.
    """

    name = "Imbalanced-naive (acc-sel)"
