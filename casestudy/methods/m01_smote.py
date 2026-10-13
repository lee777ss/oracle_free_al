"""casestudy.methods.m01_smote — SMOTE on raw pixel vectors (§5.1, decision 3).

Uses ``imblearn.over_sampling.SMOTE`` on flattened ``C×64×64`` pixel vectors to
oversample each minority class up to the target (majority) count. Returns only
the NEW synthetic samples (imblearn appends them after the originals).

The shared flatten → resample → reshape machinery lives in :func:`resample`
so the Borderline-SMOTE (m02) and ADASYN (m03) modules reuse it.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from .base import AugmentationMethod

print("casestudy/methods/m01_smote.py loaded")


def resample(
    sampler_factory,
    X_real: torch.Tensor,
    y_real: torch.Tensor,
    deficits: Dict[int, int],
    channels: int,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten → oversample → reshape; return only NEW synthetic samples.

    Args:
        sampler_factory: ``callable(sampling_strategy, k_neighbors) -> imblearn sampler``.
        X_real, y_real: real pool tensors (``[N,C,S,S]`` in ``[-1,1]``, ``[N]``).
        deficits: per-class number of NEW samples to synthesize.
        channels, image_size: geometry for reshaping back to images.
    """
    if not deficits:
        x = torch.empty(0, channels, image_size, image_size)
        return x, torch.empty(0, dtype=torch.long)

    X_np = X_real.view(X_real.size(0), -1).cpu().numpy().astype(np.float64)
    y_np = y_real.cpu().numpy().astype(int)

    counts = {int(c): int((y_np == c).sum()) for c in np.unique(y_np)}
    # imblearn needs final target counts >= current; only oversample deficit classes.
    strategy = {c: counts[c] + d for c, d in deficits.items()}
    min_class = min(counts[c] for c in deficits)
    k = max(1, min(5, min_class - 1))

    sampler = sampler_factory(strategy, k)
    X_res, y_res = sampler.fit_resample(X_np, y_np)

    # New synthetic rows are appended after the originals.
    n_orig = len(y_np)
    X_new = X_res[n_orig:]
    y_new = y_res[n_orig:]

    X_t = torch.from_numpy(X_new.astype(np.float32)).view(-1, channels, image_size, image_size)
    X_t = X_t.clamp(-1.0, 1.0)
    y_t = torch.from_numpy(np.asarray(y_new)).long()
    return X_t, y_t


class SMOTEMethod(AugmentationMethod):
    name = "SMOTE"
    is_generative = False

    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        self.X_real, self.y_real = X_real, y_real

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        from imblearn.over_sampling import SMOTE

        deficits = self._deficits(target_counts)
        return resample(
            lambda strat, k: SMOTE(sampling_strategy=strat, k_neighbors=k,
                                   random_state=self.seed),
            self.X_real, self.y_real, deficits, self.channels, self.image_size,
        )
