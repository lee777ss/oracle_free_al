"""casestudy.eval — metrics, FID & statistics (plan §7 / §9).

Populated in roadmap Step 4:
- ``metrics.py``: macro-recall, G-mean, macro-F2, macro-specificity.
- ``fid.py``: per-class FID (Inception features).
- ``stats.py``: CV aggregation + Friedman / Wilcoxon-Holm significance tests.
"""

from .metrics import aggregate_folds, compute_metrics, per_class_recall
from .fid import compute_fid, fid_per_class
from .stats import aggregate_cv, average_ranks, friedman_test, wilcoxon_holm

__all__ = [
    "compute_metrics",
    "per_class_recall",
    "aggregate_folds",
    "compute_fid",
    "fid_per_class",
    "aggregate_cv",
    "average_ranks",
    "friedman_test",
    "wilcoxon_holm",
]
