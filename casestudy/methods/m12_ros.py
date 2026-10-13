"""m12 - random oversampling (KBS revision, S2; requested by Reviewer 3, comment 5).

Duplicates real minority images, drawn uniformly with replacement, until each class
reaches the shared augmentation target. The simplest remedy for imbalance: it adds no
new image, only repetitions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .base import AugmentationMethod

print("casestudy/methods/m12_ros.py loaded")


class ROSMethod(AugmentationMethod):
    name = "Random oversampling"
    is_generative = False

    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        self.X_real, self.y_real = X_real, y_real

    def _picks(self, deficits: Dict[int, int]) -> List[Tuple[int, torch.Tensor]]:
        g = torch.Generator().manual_seed(int(self.seed))
        out = []
        for c, d in deficits.items():
            idx_c = torch.where(self.y_real == c)[0]
            out.append((c, idx_c[torch.randint(0, len(idx_c), (d,), generator=g)]))
        return out

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        deficits = self._deficits(target_counts)
        if not deficits:
            return self._empty()
        xs, ys = [], []
        for c, pick in self._picks(deficits):
            xs.append(self.X_real[pick].clone())
            ys.append(torch.full((len(pick),), c, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)
