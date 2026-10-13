"""casestudy.seeds - named random streams derived from a run seed (KBS revision, S3).

``stream_seed(42, "backbone")`` always returns the same 32-bit integer, and different
names give unrelated integers. Seeding each stage of a run from its own stream means
a stage gives the same result whether earlier stages were run in this process or
loaded from a cache, and lets several selection rules draw identical candidates.
"""

from __future__ import annotations

import hashlib


def stream_seed(seed: int, name: str) -> int:
    """Deterministic 32-bit seed for the stream ``name`` of run seed ``seed``."""
    digest = hashlib.sha1(f"{int(seed)}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
