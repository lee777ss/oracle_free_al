"""casestudy.reporting — tables & figures (plan §9).

Populated in roadmap Step 9:
- ``tables.py``: LaTeX + CSV comparison tables.
- ``plots.py``: bar charts, ranking heatmaps, CD diagram, sample grids, FID plots.
"""

from .tables import (
    TEST_METRICS,
    METRIC_LABELS,
    load_per_dataset_csvs,
    build_master_table,
    master_to_latex,
    cross_dataset_ranking,
    significance_table,
)
from .plots import (
    set_pub_style,
    grouped_bar_charts,
    ranking_heatmap,
    cd_diagram,
    per_class_fid_bars,
    sample_grid,
)

__all__ = [
    "TEST_METRICS",
    "METRIC_LABELS",
    "load_per_dataset_csvs",
    "build_master_table",
    "master_to_latex",
    "cross_dataset_ranking",
    "significance_table",
    "set_pub_style",
    "grouped_bar_charts",
    "ranking_heatmap",
    "cd_diagram",
    "per_class_fid_bars",
    "sample_grid",
]
