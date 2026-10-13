"""m13 / m14 - standard image augmentation (KBS revision, S2; Reviewer 3, comment 5).

``StdAugMethod`` (m13, budget-matched): draws real minority images with replacement, as
random oversampling does, and returns a randomly shifted, rotated, scaled and (except on
GTSRB) mirrored copy of each, up to the shared augmentation target.

``StdAugOnline`` (m14, conventional): adds no image. Instead the downstream classifier
sees every training batch freshly augmented, through ``train_transform``. It is the only
method in the field that is not budget-matched, by construction.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from ..data.augment import OnlineAugment, random_affine
from .m00_imbalanced import ImbalancedBaseline
from .m12_ros import ROSMethod

print("casestudy/methods/m13_stdaug.py loaded")


def _hflip_allowed(spec) -> bool:
    """Mirroring changes the class of a traffic sign, so it is off for GTSRB."""
    return getattr(spec, "loader_key", "") != "gtsrb"


class StdAugMethod(ROSMethod):
    name = "Standard augmentation"
    is_generative = False

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        deficits = self._deficits(target_counts)
        if not deficits:
            return self._empty()
        g = torch.Generator().manual_seed(int(self.seed) + 1)
        xs, ys = [], []
        for c, pick in self._picks(deficits):
            xs.append(random_affine(self.X_real[pick].clone(), g, hflip=_hflip_allowed(self.spec)))
            ys.append(torch.full((len(pick),), c, dtype=torch.long))
        return torch.cat(xs).clamp(-1.0, 1.0), torch.cat(ys)


class StdAugOnline(ImbalancedBaseline):
    name = "Standard augmentation (on the fly)"
    is_generative = False

    def train_transform(self, device: str):
        """Per-batch augmentation for the downstream classifier, seeded by the run seed."""
        return OnlineAugment(seed=int(self.seed) + 2, device=device, hflip=_hflip_allowed(self.spec))
