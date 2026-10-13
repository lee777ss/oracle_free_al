"""casestudy.reproducibility — self-contained seeding helpers (plan §8).

Faithful, standalone re-implementation of the seeding behavior used in
GAAL_Ablation_130J (``ablation/reproducibility.py``).  The ``casestudy``
package never imports ``ablation``; this module provides the same guarantees
so every stochastic step in the case studies is reproducible.

Call ``set_seed(42)`` before every stochastic step (model init, training,
synthesis).  Use ``make_loader_generator`` / ``seed_worker`` for DataLoaders.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn

print("casestudy/reproducibility.py loaded")


def set_seed(seed: int = 42) -> torch.Generator:
    """Reset every RNG affecting training; return a seeded ``torch.Generator``.

    Side effects: seeds ``random``, ``numpy``, ``torch`` (CPU+CUDA); enables
    ``cudnn.deterministic`` and disables ``cudnn.benchmark``.

    Note: for strict bitwise determinism on CUDA, set
    ``os.environ["CUBLAS_WORKSPACE_CONFIG"]`` before importing torch and
    enable ``torch.use_deterministic_algorithms(True)`` in the notebook.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for DataLoader: seed numpy/random per worker."""
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def make_loader_generator(seed: Optional[int] = None) -> torch.Generator:
    """Build a fresh ``torch.Generator`` for a DataLoader (reproducible shuffle)."""
    g = torch.Generator()
    if seed is None:
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
    g.manual_seed(int(seed))
    return g
