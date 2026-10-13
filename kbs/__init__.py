"""kbs - drivers for the Knowledge-Based Systems revision (KNOSYS-D-26-20956).

See ``400_kbs_revision_plan.ipynb``. ``jobs`` defines the experiment stages, ``runq`` runs them as a
resumable queue, ``verify`` checks the results, ``oracle`` trains the reference classifiers.
"""

import os as _os
from pathlib import Path as _Path

# Where tables, figures and generated LaTeX go: the manuscript folder of this repository by default; set
# KBS_PAPER_DIR to write them elsewhere (for example in a clone of the public release).
PAPER_DIR = _Path(_os.environ.get("KBS_PAPER_DIR", "Journal_paper_02_kbs_revise"))
