"""
Centralized reproducibility helpers.

Single source of truth for seeding RNGs across the notebook and the
ablation package.  Phase 2 of the reproducibility remediation plan.

IMPORTANT:
    ``os.environ["CUBLAS_WORKSPACE_CONFIG"]`` must be set BEFORE the first
    ``import torch`` in the process.  This module does NOT set it (too late
    by the time anyone imports us).  The notebook's first cell is
    responsible for setting it before importing torch.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn


print("reproducibility.py loaded")


# ── Phase 3.6 strict-determinism flags ──────────────────────────────────
# Toggle via env vars BEFORE first `import ablation.reproducibility`.
#   BAGAN_REPRO_STRICT=0       → keep `use_deterministic_algorithms(False)`
#                                escape hatches in kfold._retrain_ab8c and
#                                trainer (ab11 fine-tune) ACTIVE.  Required
#                                if WGAN-GP double-backward raises on CUDA.
#   BAGAN_REPRO_WARN_ONLY=1    → keep `set_seed` in soft mode (silent
#                                fallback on non-deterministic kernels).
# Defaults: strict on, warn-only off → bitwise reproducibility.
BAGAN_REPRO_STRICT: bool = os.environ.get("BAGAN_REPRO_STRICT", "1") != "0"
BAGAN_REPRO_WARN_ONLY: bool = os.environ.get("BAGAN_REPRO_WARN_ONLY", "0") == "1"


def set_seed(seed: int = 42) -> torch.Generator:
    """Reset every RNG that affects training, return a seeded torch.Generator.

    The returned generator can be passed to ``DataLoader(generator=...)`` so
    that shuffling is reproducible *per call site* even when many DataLoaders
    are created during one run.

    Side effects:
        - seeds ``random``, ``numpy.random``, ``torch`` (CPU+CUDA)
        - enables ``cudnn.deterministic`` and disables ``cudnn.benchmark``
        - enables ``torch.use_deterministic_algorithms(True)``

    Pre-condition (caller's responsibility):
        ``os.environ["CUBLAS_WORKSPACE_CONFIG"]`` is set BEFORE ``import torch``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    # Phase 3.6: honor module-level BAGAN_REPRO_WARN_ONLY flag.  Default is
    # strict (warn_only=False) — any non-deterministic kernel raises.  Set
    # env BAGAN_REPRO_WARN_ONLY=1 to fall back to legacy soft behavior.
    torch.use_deterministic_algorithms(True, warn_only=BAGAN_REPRO_WARN_ONLY)

    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for DataLoader to seed numpy/random per worker."""
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def make_loader_generator(seed: Optional[int] = None) -> torch.Generator:
    """Build a fresh torch.Generator for a DataLoader.

    If ``seed`` is None the generator inherits its seed from the current
    torch RNG state (so it changes every time after ``set_seed`` is called).
    Pass an explicit ``seed`` only when you need a *named*, predictable
    stream (e.g. for tests).
    """
    g = torch.Generator()
    if seed is None:
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
    g.manual_seed(int(seed))
    return g


def tensor_hash(t: torch.Tensor, n_bytes: int = 12) -> str:
    """Stable short hex hash of a tensor's contents (for audit logs)."""
    arr = t.detach().to("cpu").contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()[: n_bytes * 2]


def mode_seed(base_seed: int, mode: str) -> int:
    """Stable per-mode seed derived from ``(base_seed, mode)``.

    Pure function — independent of process RNG state, ``MODES_TO_RUN``
    ordering, or any prior RNG consumption.  Used to reseed each mode's
    retrain so its result depends only on ``(config.random_seed, mode)``.
    """
    h = int.from_bytes(hashlib.sha256(mode.encode("utf-8")).digest()[:4], "big")
    return (int(base_seed) ^ h) & 0x7FFFFFFF


def collect_env_info() -> dict:
    """Snapshot the runtime environment for audit logs."""
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": cudnn.version(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "deterministic": cudnn.deterministic,
        "benchmark": cudnn.benchmark,
        "bagan_repro_strict": BAGAN_REPRO_STRICT,
        "bagan_repro_warn_only": BAGAN_REPRO_WARN_ONLY,
    }


# ── Phase 3.6 BAGAN snapshot helpers ────────────────────────────────────

def sd_hash(state: Dict[str, Any], n_bytes: int = 12) -> str:
    """Stable short hex hash of a state_dict-shaped mapping.

    Iterates keys in sorted order so the hash is independent of insertion
    order.  Values may be tensors, plain dicts of tensors, or scalars.
    """
    h = hashlib.sha256()
    for k in sorted(state.keys()):
        h.update(str(k).encode("utf-8"))
        v = state[k]
        if isinstance(v, torch.Tensor):
            h.update(v.detach().to("cpu").contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            h.update(sd_hash(v, n_bytes).encode("ascii"))
        else:
            h.update(repr(v).encode("utf-8"))
    return h.hexdigest()[: n_bytes * 2]


def save_bagan_snapshot(
    path: str,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    latent_gen: Any,
) -> str:
    """Write the post-§D BAGAN trio to one .pth file.

    The snapshot is the canonical state restored at the top of every
    mode's retrain in §E.1 (Phase 3.6 Step 3.6.1 / 3.6.2).  All tensors
    are stored on CPU so the file is device-agnostic.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "generator": {k: v.detach().cpu() for k, v in generator.state_dict().items()},
        "discriminator": {k: v.detach().cpu() for k, v in discriminator.state_dict().items()},
        "latent_means": {k: v.detach().cpu() for k, v in latent_gen.means.items()},
        "latent_stds": {k: v.detach().cpu() for k, v in latent_gen.stds.items()},
        "latent_n_samples": getattr(latent_gen, "n_samples", None),
        "format_version": 1,
    }
    torch.save(payload, path)
    return path


def load_bagan_snapshot(
    path: str,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    latent_gen: Any,
) -> None:
    """Restore the BAGAN trio in-place from a snapshot file.

    Loads onto whichever device the modules currently live on.  Used at
    the top of every retrain (Phase 3.6 Step 3.6.2) to eliminate cross-
    mode state mutation.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError(
            f"Unsupported bagan snapshot format_version: {payload.get('format_version')}"
        )
    g_device = next(generator.parameters()).device
    d_device = next(discriminator.parameters()).device
    l_device = getattr(latent_gen, "device", g_device)

    generator.load_state_dict(
        {k: v.to(g_device) for k, v in payload["generator"].items()}
    )
    discriminator.load_state_dict(
        {k: v.to(d_device) for k, v in payload["discriminator"].items()}
    )
    latent_gen.means = {k: v.to(l_device) for k, v in payload["latent_means"].items()}
    latent_gen.stds = {k: v.to(l_device) for k, v in payload["latent_stds"].items()}
    if payload.get("latent_n_samples") is not None:
        latent_gen.n_samples = payload["latent_n_samples"]
    # Mark as fitted so downstream sampling doesn't refit.
    if hasattr(latent_gen, "is_fitted"):
        latent_gen.is_fitted = True
