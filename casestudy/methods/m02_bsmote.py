"""casestudy.methods.m02_bsmote — Borderline-SMOTE on raw pixels (§5.1).

``imblearn.over_sampling.BorderlineSMOTE(kind='borderline-1')`` concentrates
synthesis on borderline minority samples; same flatten/reshape interface as
m01 via :func:`casestudy.methods.m01_smote.resample`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from .base import AugmentationMethod
from .m01_smote import resample

print("casestudy/methods/m02_bsmote.py loaded")


class BorderlineSMOTEMethod(AugmentationMethod):
    name = "Borderline-SMOTE"
    is_generative = False

    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        self.X_real, self.y_real = X_real, y_real

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        from imblearn.over_sampling import BorderlineSMOTE

        deficits = self._deficits(target_counts)
        return resample(
            lambda strat, k: BorderlineSMOTE(sampling_strategy=strat, k_neighbors=k,
                                             kind="borderline-1", random_state=self.seed),
            self.X_real, self.y_real, deficits, self.channels, self.image_size,
        )
