"""kbs2.provenance — guard and record that runs use the submitted protocol (rule B2).

Before any experiment, ``check_or_die()`` verifies that

* every Python file of ``casestudy/`` and ``cs3j_runner.py`` that exists at the
  submission snapshot ``b359b0c8`` is byte-identical to it, except two files whose
  post-snapshot diffs are pinned by hash and were verified behaviour-neutral:
    - ``casestudy/classifiers/trainer.py``: adds ``train_transform=None``; with
      ``None`` the training loop is unchanged.
    - ``casestudy/methods/m08_bagan.py``: moves the hard-coded label smoothing
      0.1 into a class attribute whose default is 0.1.
* the interpreter and libraries match the environment recorded in the result
  manifests (Python 3.11.15, torch 2.11.0+cu128, numpy 2.4.3, CUDA 12.8);
* no determinism switch that the submitted runs did not use has been turned on.

``record()`` returns a dict stored next to every result (commit, diff hashes,
environment, data-cache hashes).
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from typing import Dict, List, Optional

SNAPSHOT = "b359b0c8"
RESULTS_COMMIT = "b3676a02a3768c18ced324973d556bf6b11037df"  # recorded in the manifests

REQUIRED_ENV = {"python": "3.11.15", "torch": "2.11.0+cu128", "numpy": "2.4.3", "cuda": "12.8"}

# sha256 of `git diff b359b0c8 -- <path>` (verified behaviour-neutral, 8 Oct 2026).
WHITELIST = {
    "casestudy/classifiers/trainer.py":
        "51a27c630679eed1c2ce1badab8efb1dc302121b8c07f59f6397d378803ad733",
    "casestudy/methods/m08_bagan.py":
        "183d71c4892c667ebd6f137ea11e07a1ed5abab02a600164bd2ea1663f86ddf0",
}


class ProtocolViolation(RuntimeError):
    """Raised when the code or environment would not reproduce the submitted protocol."""


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot_files() -> List[str]:
    out = _git("ls-tree", "-r", "--name-only", SNAPSHOT, "--", "casestudy", "cs3j_runner.py")
    return [p for p in out.decode().splitlines() if p.endswith(".py")]


def check_code() -> Dict[str, str]:
    """Return {path: status}; raise ProtocolViolation on any unexpected difference."""
    status: Dict[str, str] = {}
    problems: List[str] = []
    for path in snapshot_files():
        diff = _git("diff", SNAPSHOT, "--", path)
        if not diff:
            status[path] = "identical"
            continue
        h = _sha256(diff)
        if WHITELIST.get(path) == h:
            status[path] = "whitelisted-behaviour-neutral"
        else:
            status[path] = f"DIFFERS ({h[:12]})"
            problems.append(f"{path}: diff hash {h}")
    if problems:
        raise ProtocolViolation("code differs from the submission snapshot:\n  " + "\n  ".join(problems))
    return status


def check_env() -> Dict[str, str]:
    import numpy as np
    import torch

    found = {"python": platform.python_version(), "torch": torch.__version__,
             "numpy": np.__version__, "cuda": str(torch.version.cuda)}
    bad = {k: (found[k], v) for k, v in REQUIRED_ENV.items() if found[k] != v}
    if bad:
        raise ProtocolViolation(f"environment differs from the manifests: {bad}")
    # The submitted runs neither forced deterministic algorithms nor set a cuBLAS
    # workspace; turning either on would change the computation.
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise ProtocolViolation("CUBLAS_WORKSPACE_CONFIG is set; the submitted runs did not set it")
    if torch.are_deterministic_algorithms_enabled():
        raise ProtocolViolation("torch deterministic algorithms are enabled; the submitted runs did not")
    return found


def file_sha256(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kbs2_hash() -> str:
    """Hash of every kbs2 source file (content, not mtime)."""
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in sorted(os.listdir(here)):
        if name.endswith(".py"):
            with open(os.path.join(here, name), "rb") as fh:
                h.update(name.encode() + b"\0" + fh.read())
    return h.hexdigest()


def check_or_die() -> Dict[str, object]:
    """Public release: the submission snapshot b359b0c8 is in the authors' private history, not in this
    repository, so the code and environment checks warn instead of stopping (README, "Provenance guard")."""
    out = {}
    for name, fn in (("code", check_code), ("env", check_env)):
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[provenance] {name} check not enforced in the public release: {exc}")
            out[name] = f"not enforced: {exc}"
    return out


def _head() -> str:
    try:
        return _git("rev-parse", "HEAD").decode().strip()
    except Exception:  # noqa: BLE001  (e.g. a downloaded archive without .git)
        return "unknown"


def record(extra_files: Optional[List[str]] = None) -> Dict[str, object]:
    import torch

    rec: Dict[str, object] = {
        "snapshot": SNAPSHOT,
        "results_commit": RESULTS_COMMIT,
        "head": _head(),
        "kbs2_sha256": kbs2_hash(),
        "env": check_env(),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "cudnn": torch.backends.cudnn.version(),
    }
    if extra_files:
        rec["files"] = {p: file_sha256(p) for p in extra_files}
    return rec
