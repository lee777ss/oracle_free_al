"""casestudy.locks - cross-process file locks for the KBS revision queue (S7).

A lock is a small JSON file created with ``O_CREAT | O_EXCL``, so only one process can
hold it. It records the owner's pid and creation time; a lock whose owner is gone (or
whose pid now belongs to a newer process) is stale and is removed by the next caller.
Never use ``os.kill(pid, 0)`` for the liveness test on Windows: it terminates the target.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import psutil


def _owner_alive(info: dict) -> bool:
    pid, created = info.get("pid"), info.get("pid_created")
    if not pid or not psutil.pid_exists(pid):
        return False
    try:
        return created is None or abs(psutil.Process(pid).create_time() - created) < 1.0
    except psutil.Error:
        return False


def read_lock(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def try_lock(path: Path, **meta) -> bool:
    """Acquire ``path`` if free (or stale). Returns True if this process now holds it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            info = read_lock(path)
            if info is not None and _owner_alive(info):
                return False
            if info is None:
                # Either half-written by a live owner (give it a moment) or debris.
                time.sleep(0.5)
                info = read_lock(path)
                if info is not None and _owner_alive(info):
                    return False
            try:
                path.unlink()
            except OSError:
                return False
            continue
        me = psutil.Process(os.getpid())
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"pid": me.pid, "pid_created": me.create_time(),
                       "started": time.strftime("%Y-%m-%d %H:%M:%S"), **meta}, fh)
        return True
    return False


def release(path: Path) -> None:
    """Remove ``path`` if this process holds it."""
    info = read_lock(path)
    if info is not None and info.get("pid") == os.getpid():
        try:
            Path(path).unlink()
        except OSError:
            pass


def held_by_other(path: Path) -> bool:
    """True if a live process other than this one holds ``path``."""
    info = read_lock(path)
    return bool(info) and info.get("pid") != os.getpid() and _owner_alive(info)
