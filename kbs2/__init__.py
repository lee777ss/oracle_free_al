"""kbs2 — experiment infrastructure for the second KBS revision (KNOSYS-D-26-20956).

Implements P0 of ``410_kbs_revise2_plan.md``: the submitted protocol, reproduction
gate (E1), the three requested baselines (E2), the weighting comparison with
synthetic-data reuse (E8), the complementarity experiment (E7), grid completion
(E3) and the pre-specified analysis (E4/A1).

Nothing here changes the submitted methods. ``casestudy/`` and ``cs3j_runner.py``
are used as they stand at the submission snapshot; ``kbs2.provenance`` refuses to
run if they differ from it beyond two verified behaviour-neutral edits.

Binding rules B1-B8 of the plan apply. In particular no experiment runs without
its approval file (``saved_kbs2/approvals/<EXP>.json``), see ``kbs2.runner``.
"""
