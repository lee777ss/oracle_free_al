"""kbs2.baselines — the three simple remedies requested by Reviewer 3 (comment 3.5).

All four classes follow ``casestudy.methods.base.AugmentationMethod`` and receive
exactly the same downstream treatment as every augmenter (``kbs2.protocol.treatment``);
they differ only in the training images they supply.

* ``ClassWeightedNoAug``  — no synthetic images; the remedy is the class-weighted
  loss. (In arm U of E8 it coincides with the plain-CE baseline m00.)
* ``RandomOversampling``  — duplicates of real minority images, drawn uniformly with
  replacement, as many per class as every augmenter adds (Delta_c = target - N_c).
* ``StdAugBudget``        — the same budget of augmented copies (source drawn as in
  ROS, then one random ``pad_crop_flip`` each).
* ``StdAugOnline``        — conventional on-the-fly augmentation: no synthetic images;
  every training batch (all classes) is transformed by ``pad_crop_flip``.
* ``StdAugBudgetPad10``   — variant of ``StdAugBudget`` run in E2p (author request, 11 Oct
  2026): identical except for 10 px padding (shift up to +-10 px instead of +-8 px).
* ``StdAugBudgetPad12``   — the same with 12 px padding (E2q, author request, 11 Oct 2026).
* ``StdAugBudgetPad6``    — the same with 6 px padding (E2r, author request, 11 Oct 2026).
* ``StdAugBudgetZoom32``  — zoom-crop variant (E2s, author request, 11 Oct 2026): pad 10 px, a 32x32
  window anywhere in the padded 84x84 image, flip p = 0.5, bilinear upscale back to 64x64.
* ``RandomOversamplingPerEpoch`` — no stored duplicates; the downstream loader redraws Delta_c minority
  duplicates every epoch (``kbs2.per_epoch``; E2v, author request, 11 Oct 2026).
* ``RandomOversamplingBatch`` — target-ratio balanced mini-batches drawn with replacement
  (``kbs2.per_epoch`` mode "batch"; E2x, author request, 11 Oct 2026).
* ``EqualCopyOversampling`` — offline ROS with every minority image duplicated equally often
  (floor(Delta_c / N_c) copies each; any remainder drawn without replacement); E2y, 11 Oct 2026.

The augmentation recipe is fixed here, before any result: the conventional
pad-and-crop + horizontal flip (He et al., 2016) scaled to 64x64 — zero padding
of 8 px (12.5% of the side; pixel value -1 in the [-1, 1] range), a random 64x64
crop, and a horizontal flip with probability 0.5. No rotation, scaling or colour
change. The same recipe is used for every dataset.

Randomness: each method draws from its own ``torch.Generator`` seeded with the
run seed, never from the global RNG, so the downstream classifier's RNG stream
(initialisation, dropout, batch order) is identical to every other method's.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

from casestudy.methods.base import AugmentationMethod

PAD = 8
FILL = -1.0
FLIP_P = 0.5


def pad_crop_flip(x: torch.Tensor, generator: torch.Generator, pad: int = PAD) -> torch.Tensor:
    """Random pad-and-crop + horizontal flip of a batch ``[B, C, H, W]``.

    Uses only ``generator`` (on the same device as ``x``) for randomness.
    """
    b, c, h, w = x.shape
    dev = x.device
    padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), value=FILL)
    oy = torch.randint(0, 2 * pad + 1, (b,), generator=generator, device=dev)
    ox = torch.randint(0, 2 * pad + 1, (b,), generator=generator, device=dev)
    flip = torch.rand(b, generator=generator, device=dev) < FLIP_P
    rows = oy[:, None] + torch.arange(h, device=dev)[None, :]            # [B, H]
    cols = ox[:, None] + torch.arange(w, device=dev)[None, :]            # [B, W]
    cols = torch.where(flip[:, None], cols.flip(1), cols)
    bi = torch.arange(b, device=dev)[:, None, None]
    out = padded.permute(0, 2, 3, 1)[bi, rows[:, :, None], cols[:, None, :]]  # [B, H, W, C]
    return out.permute(0, 3, 1, 2).contiguous()


def zoom_crop_flip(x: torch.Tensor, generator: torch.Generator, pad: int = 10, crop: int = 32) -> torch.Tensor:
    """Pad, take a random ``crop`` x ``crop`` window anywhere in the padded image, flip, resize back.

    The window offset is uniform over the whole padded image (0 .. H + 2*pad - crop), so corner
    windows can be largely border. Uses only ``generator`` for randomness.
    """
    b, c, h, w = x.shape
    dev = x.device
    padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), value=FILL)
    oy = torch.randint(0, h + 2 * pad - crop + 1, (b,), generator=generator, device=dev)
    ox = torch.randint(0, w + 2 * pad - crop + 1, (b,), generator=generator, device=dev)
    flip = torch.rand(b, generator=generator, device=dev) < FLIP_P
    rows = oy[:, None] + torch.arange(crop, device=dev)[None, :]          # [B, crop]
    cols = ox[:, None] + torch.arange(crop, device=dev)[None, :]
    cols = torch.where(flip[:, None], cols.flip(1), cols)
    bi = torch.arange(b, device=dev)[:, None, None]
    out = padded.permute(0, 2, 3, 1)[bi, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
    out = torch.nn.functional.interpolate(out.contiguous(), size=(h, w), mode="bilinear", align_corners=False)
    return out.clamp(-1.0, 1.0).contiguous()


class _RealOnly(AugmentationMethod):
    is_generative = False

    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        self.X_real, self.y_real = X_real, y_real

    def _draw_sources(self, target_counts: Dict[int, int]):
        """Indices of real images to copy: uniform with replacement per minority class."""
        g = torch.Generator().manual_seed(int(self.seed))
        out = []
        for c, n in sorted(self._deficits(target_counts).items()):
            idx_c = torch.where(self.y_real == c)[0]
            pick = idx_c[torch.randint(0, len(idx_c), (n,), generator=g)]
            out.append((c, pick))
        return out, g


class ClassWeightedNoAug(_RealOnly):
    name = "Class-weighted"

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._empty()


class RandomOversampling(_RealOnly):
    name = "ROS"

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        draws, _ = self._draw_sources(target_counts)
        if not draws:
            return self._empty()
        idx = torch.cat([p for _, p in draws])
        self.source_index = idx.clone()
        return self.X_real[idx].clone(), self.y_real[idx].clone()


class StdAugBudget(_RealOnly):
    name = "StdAug (budget-matched)"
    pad = PAD

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        draws, g = self._draw_sources(target_counts)
        if not draws:
            return self._empty()
        idx = torch.cat([p for _, p in draws])
        self.source_index = idx.clone()
        x = pad_crop_flip(self.X_real[idx], g, self.pad)    # CPU generator, CPU tensors
        return x, self.y_real[idx].clone()


class StdAugBudgetPad10(StdAugBudget):
    name = "StdAug (budget-matched, pad 10)"
    pad = 10


class StdAugBudgetPad12(StdAugBudget):
    name = "StdAug (budget-matched, pad 12)"
    pad = 12


class StdAugBudgetPad6(StdAugBudget):
    name = "StdAug (budget-matched, pad 6)"
    pad = 6


class EqualCopyOversampling(_RealOnly):
    name = "ROS (equal copies)"

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(int(self.seed))
        parts = []
        for c, n in sorted(self._deficits(target_counts).items()):
            idx_c = torch.where(self.y_real == c)[0]
            k, rem = divmod(int(n), len(idx_c))
            parts.append(idx_c.repeat(k))
            if rem:
                parts.append(idx_c[torch.randperm(len(idx_c), generator=g)[:rem]])
        if not parts:
            return self._empty()
        idx = torch.cat(parts)
        self.source_index = idx.clone()
        return self.X_real[idx].clone(), self.y_real[idx].clone()


class RandomOversamplingPerEpoch(_RealOnly):
    name = "ROS (per epoch)"

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._empty()      # the oversampling happens in the downstream loader


class RandomOversamplingBatch(RandomOversamplingPerEpoch):
    name = "ROS (target-ratio balanced batches)"


class StdAugBudgetZoom32(_RealOnly):
    name = "StdAug (budget-matched, pad 10, 32x32 zoom crop)"

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        draws, g = self._draw_sources(target_counts)
        if not draws:
            return self._empty()
        idx = torch.cat([p for _, p in draws])
        self.source_index = idx.clone()
        x = zoom_crop_flip(self.X_real[idx], g, pad=10, crop=32)
        return x, self.y_real[idx].clone()


class StdAugOnline(_RealOnly):
    name = "StdAug (on-the-fly)"

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._empty()


def online_transform(seed: int, device: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Batch transform for ``train_classifier(train_transform=...)`` with its own stream."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    def _t(xb: torch.Tensor) -> torch.Tensor:
        # With one channel the [B, 1, H, W] result of pad_crop_flip also counts as channels_last,
        # which the convolutions propagate and the CNN's .view() then rejects (E2, 10 Oct). Force the
        # standard layout; the values are unchanged.
        return pad_crop_flip(xb, g).clone(memory_format=torch.contiguous_format)

    return _t


REGISTRY = {
    "n01_class_weighted": ClassWeightedNoAug,
    "n02_ros": RandomOversampling,
    "n03_stdaug_budget": StdAugBudget,
    "n04_stdaug_online": StdAugOnline,
    "n05_stdaug_budget_p10": StdAugBudgetPad10,
    "n06_stdaug_budget_p12": StdAugBudgetPad12,
    "n07_stdaug_budget_p6": StdAugBudgetPad6,
    "n08_stdaug_budget_zc32": StdAugBudgetZoom32,
    "n09_ros_per_epoch": RandomOversamplingPerEpoch,
    "n10_ros_batch": RandomOversamplingBatch,
    "n11_ros_equal": EqualCopyOversampling,
}
PER_EPOCH = {"n09_ros_per_epoch": "epoch", "n10_ros_batch": "batch"}   # key -> kbs2.per_epoch mode
ONLINE = {"n04_stdaug_online"}
