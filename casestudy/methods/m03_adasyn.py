"""casestudy.methods.m03_adasyn — ADASYN on raw pixels (§5.1).

Density-adaptive oversampling (``imblearn.over_sampling.ADASYN``): more samples
are generated where minority density is low. ADASYN's ``n_neighbors`` plays the
role of m01's ``k_neighbors``; it only approximately reaches the requested
count, so the synthesize result is trimmed/topped-up to the exact deficit.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from .base import AugmentationMethod
from .m01_smote import resample

print("casestudy/methods/m03_adasyn.py loaded")


class ADASYNMethod(AugmentationMethod):
    name = "ADASYN"
    is_generative = False

    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        self.X_real, self.y_real = X_real, y_real

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        from imblearn.over_sampling import ADASYN

        deficits = self._deficits(target_counts)
        if not deficits:
            return self._empty()
        X_new, y_new = resample(
            lambda strat, k: ADASYN(sampling_strategy=strat, n_neighbors=k,
                                    random_state=self.seed),
            self.X_real, self.y_real, deficits, self.channels, self.image_size,
        )
        # ADASYN only approximately honours the target count; resize each class
        # to the exact deficit. Over-production → take the first d; under-
        # production → top up by sampling (with replacement) from that class's
        # synthetics (or its real samples if ADASYN produced none).
        g = torch.Generator().manual_seed(self.seed)
        xs, ys = [], []
        for c, d in deficits.items():
            X_c = X_new[y_new == c]
            if len(X_c) >= d:
                X_c = X_c[:d]
            else:
                pool = X_c if len(X_c) > 0 else self.X_real[self.y_real == c]
                extra = d - len(X_c)
                pick = torch.randint(0, len(pool), (extra,), generator=g)
                X_c = torch.cat([X_c, pool[pick]], dim=0)
            xs.append(X_c)
            ys.append(torch.full((d,), c, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)
