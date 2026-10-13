"""kbs2.per_epoch — per-epoch random oversampling for the downstream classifier (E2v, author request 11 Oct 2026).

Offline ROS (``n02_ros``) adds a fixed set of Delta_c duplicates of real minority images once. Per-epoch ROS
stores no duplicates: every epoch the loader yields all real training images once plus Delta_c fresh draws
(uniform, with replacement) from the real images of each minority class c, so the per-epoch class counts equal
offline ROS's fixed counts (the r x N_c targets) while the duplicated images change from epoch to epoch.

``casestudy/`` must stay byte-identical to the submission snapshot, so ``per_epoch_ros(...)`` swaps two names in
the trainer module for the duration of one ``train_classifier`` call and always restores them:

* ``DataLoader``: the training loader (the only call with ``shuffle=True``) is built with ``PerEpochSampler``;
  the evaluation loaders pass through unchanged.
* ``_class_weights``: inverse frequency computed on the per-epoch counts (the targets), i.e. exactly the weights
  offline ROS gets from its augmented label vector, so both ROS forms receive the same treatment.

Randomness: the minority draws use their own generator seeded with the run seed (so the ten seeds differ);
the order of each epoch is a permutation drawn from the trainer's loader generator (classifier seed), as for
every other method.

Mode "batch" (E2x, author request 11 Oct 2026) — target-ratio balanced mini-batches: every epoch has
B = ceil(sum_c t_c / batch_size) batches; each class total t_c is split over the batches as evenly as possible
(per-batch counts differ by at most 1; the batches that get the extra one are drawn at random), and each batch
draws its quota of class c uniformly WITH replacement from the real images of class c. Every batch thus has the
target class proportions, and each epoch has exactly t_c samples of class c (offline ROS's counts). Draws use the
run-seed generator; the order of the batches within an epoch is a permutation from the loader generator.
"""

from __future__ import annotations

import contextlib
from typing import Dict, Iterator

import torch
from torch.utils.data import Sampler


class PerEpochSampler(Sampler):
    def __init__(self, labels: torch.Tensor, deficits: Dict[int, int], draw_seed: int,
                 order_generator: torch.Generator):
        self.n_real = int(labels.shape[0])
        self.pools = {int(c): torch.where(labels == c)[0] for c in deficits}
        self.deficits = {int(c): int(n) for c, n in deficits.items() if int(n) > 0}
        self.draw_gen = torch.Generator().manual_seed(int(draw_seed))
        self.order_gen = order_generator
        self.length = self.n_real + sum(self.deficits.values())

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        parts = [torch.arange(self.n_real)]
        for c, n in sorted(self.deficits.items()):
            pool = self.pools[c]
            parts.append(pool[torch.randint(0, len(pool), (n,), generator=self.draw_gen)])
        idx = torch.cat(parts)
        perm = torch.randperm(len(idx), generator=self.order_gen)
        return iter(idx[perm].tolist())


class BalancedBatchSampler(Sampler):
    def __init__(self, labels: torch.Tensor, targets: Dict[int, int], batch_size: int, draw_seed: int,
                 order_generator: torch.Generator):
        self.pools = {int(c): torch.where(labels == c)[0] for c in targets}
        self.targets = {int(c): int(t) for c, t in targets.items()}
        self.n_batches = -(-sum(self.targets.values()) // int(batch_size))
        self.draw_gen = torch.Generator().manual_seed(int(draw_seed))
        self.order_gen = order_generator

    def __len__(self) -> int:
        return self.n_batches

    def quotas(self) -> torch.Tensor:
        """[n_batches, K] class counts per batch for one epoch (column sums = targets)."""
        q = torch.zeros(self.n_batches, len(self.targets), dtype=torch.long)
        for j, (c, t) in enumerate(sorted(self.targets.items())):
            q[:, j] = t // self.n_batches
            extra = torch.randperm(self.n_batches, generator=self.draw_gen)[: t % self.n_batches]
            q[extra, j] += 1
        return q

    def __iter__(self):
        q = self.quotas()
        classes = sorted(self.targets)
        batches = []
        for b in range(self.n_batches):
            parts = []
            for j, c in enumerate(classes):
                pool = self.pools[c]
                parts.append(pool[torch.randint(0, len(pool), (int(q[b, j]),), generator=self.draw_gen)])
            batches.append(torch.cat(parts).tolist())
        for b in torch.randperm(self.n_batches, generator=self.order_gen).tolist():
            yield batches[b]


@contextlib.contextmanager
def per_epoch_ros(train_labels: torch.Tensor, targets: Dict[int, int], draw_seed: int,
                  mode: str = "epoch") -> Iterator[None]:
    if mode not in ("epoch", "batch"):
        raise ValueError(mode)
    from torch.utils.data import DataLoader as _DL
    from casestudy.classifiers import trainer

    counts = torch.bincount(train_labels, minlength=len(targets))
    deficits = {int(c): max(0, int(t) - int(counts[int(c)])) for c, t in targets.items()}
    target_labels = torch.cat([torch.full((int(t),), int(c), dtype=torch.long) for c, t in sorted(targets.items())])
    orig_loader, orig_weights = trainer.DataLoader, trainer._class_weights
    built = []

    def _loader(dataset, batch_size=1, shuffle=False, generator=None, worker_init_fn=None, **kw):
        if not shuffle:
            return orig_loader(dataset, batch_size=batch_size, shuffle=False, generator=generator,
                               worker_init_fn=worker_init_fn, **kw)
        if mode == "batch":
            bs = BalancedBatchSampler(dataset.tensors[1], targets, batch_size, draw_seed, generator)
            built.append(bs)
            return orig_loader(dataset, batch_sampler=bs, worker_init_fn=worker_init_fn, **kw)
        sampler = PerEpochSampler(dataset.tensors[1], deficits, draw_seed, generator)
        built.append(sampler)
        return orig_loader(dataset, batch_size=batch_size, sampler=sampler, worker_init_fn=worker_init_fn, **kw)

    def _weights(labels, num_classes, device):
        return orig_weights(target_labels, num_classes, device)

    trainer.DataLoader, trainer._class_weights = _loader, _weights
    try:
        yield
    finally:
        trainer.DataLoader, trainer._class_weights = orig_loader, orig_weights
    if len(built) != 1:
        raise RuntimeError(f"per-epoch ROS: expected one training loader, built {len(built)}")
