"""kbs2.runner — one-experiment-at-a-time runner for E1, E2, E8, E7, E3 (rule B7).

    python -m kbs2.runner list E1        # print the jobs of E1; runs nothing
    python -m kbs2.runner run  E1        # run E1 only (needs approvals, see below)
    python -m kbs2.runner status         # approvals and gate reports found

A run mirrors ``cs3j_runner.run_one`` step for step (``kbs2.protocol`` docstring),
split into two halves so that one synthetic set can feed several downstream
classifiers (rule B3):

    generate():   hold-out split (random_state=42) -> set_seed(seed) -> build method
                  -> fit(train portion) -> synthesize(r-targets) -> save the set
    downstream(): set_seed(42) -> build classifier -> train with the arm's loss and
                  selection criterion (and the on-the-fly transform if any) -> test once

Between the halves nothing touches an RNG, and ``downstream`` starts from
``set_seed(42)`` exactly as ``run_one`` does, so a set loaded from disk gives the same
classifier as one generated in the same process.

Approvals (rule B7): ``run X`` refuses unless ``saved_kbs2/approvals/X.json`` exists with
``"approved": true`` and the previous step of the sequence is complete (its approval
file exists and, for steps with reproduction checks, its gate report passed).
Sequence: P0 -> E1 -> E2 -> E8 -> E7 -> E3 -> E4.

Reproduction checks (rule B1): every run that regenerates a submitted result is compared
with the submitted per-seed metrics immediately; a material failure stops the experiment.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kbs2 import protocol as P  # noqa: E402

OUT = os.path.join(ROOT, "saved_kbs2")
# E3 moved before E7 on 11 Oct (authors: R3.7 next; E7 not scheduled).
SEQUENCE = ["P0", "E1", "E1b", "E2", "E2u", "E8", "E2w", "E2p", "E2q", "E2r", "E2s", "E2t", "E2v", "E2x", "E2y", "E2z", "E3", "AB", "E7", "SE", "E4"]
# SE (R1.5, authors 13 Oct): settings fixed before any SE run; the submitted values are tau 0.6, pool
# factor 10 and 20 images per class per cycle. tau 0.5 comes from E1b (arm W).
SE_TAUS = (0.7, 0.8, 0.9)
SE_POOL_FACTORS = (2, 5, 20)
SE_PER_CYCLE = (10, 40)
GATED = {"E1", "E2", "E8", "E3"}   # experiments containing reproduction checks


# ── jobs ────────────────────────────────────────────────────────────────────
@dataclass
class Job:
    exp: str
    method: str
    regime: str
    dataset: str
    ratio: int
    seed: int
    arms: List[str]
    generate: bool = True              # False: load the synthetic set from `load_from`
    load_from: Optional[str] = None    # experiment whose saved set is reused
    load_method: Optional[str] = None  # method whose saved set is reused (E7)
    online: bool = False               # apply the on-the-fly transform downstream
    reproduce: Optional[str] = None    # reference CSV for a reproduction check (arm W)
    save_candidates: bool = False
    tag: str = ""                      # distinguishes E7 / E1b arms
    tau: Optional[float] = None        # gate threshold of the proposed method (None = submitted 0.6)
    weight_alpha: Optional[float] = None  # class weights w_c ∝ n_c^-alpha (None = submitted inverse frequency)
    sel: Optional[str] = None          # AB: selection rule of casestudy.methods.selection_v2
    cotrain: Optional[bool] = None     # AB: generator/critic co-training inside the loop
    pool_factor: Optional[int] = None  # SE: candidate over-production factor (None = submitted 10)
    spc: Optional[int] = None          # SE: images admitted per class per cycle (None = submitted 20)

    @property
    def name(self) -> str:
        t = f"_{self.tag}" if self.tag else ""
        return f"{self.method}{t}__{self.regime}__s{self.seed}"


def jobs_for(exp: str) -> List[Job]:
    jobs: List[Job] = []
    anchors = [(ds, rid, r) for ds, (rid, r) in P.ANCHORS.items()]
    if exp == "E1":
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                jobs.append(Job("E1", P.PROPOSED, rid, ds, r, s, ["W"],
                                reproduce=P.REFERENCE_ANCHOR_CSV[rid], save_candidates=True))
    elif exp == "E1b":
        # D13/D14: proposed method at tau = 0.5 (generated once; classifiers with seed 42 and with the
        # run seed), and E1's saved tau = 0.6 sets retrained with the run seed (seed 42 = E1 itself).
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                jobs.append(Job("E1b", P.PROPOSED, rid, ds, r, s, ["W", "Ws"], save_candidates=True,
                                tau=P.PROPOSED_TAU, tag="tau05"))
                jobs.append(Job("E1b", P.PROPOSED, rid, ds, r, s, ["Ws"], generate=False,
                                load_from="E1", tag="tau06"))
    elif exp == "E2":
        # D14 (re-specified 10 Oct): the main table uses arm Ws (classifier seeded with the run seed);
        # the naive baseline also runs in arm W (seed 42) as the reproduction check of the submitted rows.
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                jobs.append(Job("E2", "m11_imbalanced_naive", rid, ds, r, s, ["W", "Ws"],
                                reproduce=P.REFERENCE_ANCHOR_CSV[rid]))
                for m in P.NEW_BASELINES:
                    jobs.append(Job("E2", m, rid, ds, r, s, ["Ws"], online=m == "n04_stdaug_online"))
    elif exp == "E2u":
        # Author request 10 Oct: unweighted arm (Us, run seed) of the simple remedies and of the proposed
        # method, on the sets saved by E2 and E1; only the classifier's loss changes. E8 reuses these rows.
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                for m in ("n02_ros", "n03_stdaug_budget", "n04_stdaug_online"):
                    jobs.append(Job("E2u", m, rid, ds, r, s, ["Us"], generate=False, load_from="E2",
                                    online=m == "n04_stdaug_online"))
                jobs.append(Job("E2u", P.PROPOSED, rid, ds, r, s, ["Us"], generate=False, load_from="E1"))
    elif exp == "E2w":
        # Author request 11 Oct: the four remedies at classifier seed 42 (arm W, submitted protocol) on the
        # sets saved by E2, so that Table 6 can be built entirely under seed 42.
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                for m in P.NEW_BASELINES:
                    jobs.append(Job("E2w", m, rid, ds, r, s, ["W"], generate=False, load_from="E2",
                                    online=m == "n04_stdaug_online"))
    elif exp == "E2p":
        # Author request 11 Oct: budget-matched standard augmentation with 10 px padding (shift up to
        # +-10 px), generated afresh, classifier seed 42 (arm W) as in E2w.
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                jobs.append(Job("E2p", "n05_stdaug_budget_p10", rid, ds, r, s, ["W"]))
    elif exp == "E2q":
        # Author request 11 Oct: the same with 12 px padding (shift up to +-12 px), to see a trend.
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                jobs.append(Job("E2q", "n06_stdaug_budget_p12", rid, ds, r, s, ["W"]))
    elif exp == "E2r":
        # Author request 11 Oct: the same with 6 px padding (shift up to +-6 px), below the fixed 8 px.
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                jobs.append(Job("E2r", "n07_stdaug_budget_p6", rid, ds, r, s, ["W"]))
    elif exp == "E2s":
        # Author request 11 Oct: zoom crop - pad 10, 32x32 window anywhere, flip, upscale to 64x64.
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                jobs.append(Job("E2s", "n08_stdaug_budget_zc32", rid, ds, r, s, ["W"]))
    elif exp == "E2t":
        # Author request 11 Oct (corrected the same day): inverse-square-root class weights (alpha = 0.5) for
        # the class-weighted baseline only; every other method keeps the submitted inverse frequency. Arm W.
        # A first version that applied it to all 13 non-naive methods was stopped at 199/390 runs by the
        # author; its rows are kept in saved_kbs2/E2t_aborted_all_methods/ and are not reported.
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                jobs.append(Job("E2t", "n01_class_weighted", rid, ds, r, s, ["W"], generate=False,
                                load_from="E2", tag="isqrt", weight_alpha=0.5))
    elif exp == "E2v":
        # Author request 11 Oct: per-epoch random oversampling (fresh Delta_c minority duplicates every epoch,
        # nothing stored), inverse-frequency weights on the per-epoch counts as offline ROS, arm W.
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                jobs.append(Job("E2v", "n09_ros_per_epoch", rid, ds, r, s, ["W"]))
    elif exp == "E2x":
        # Author request 11 Oct: target-ratio balanced mini-batch ROS (every batch has the target class
        # proportions, drawn with replacement within the batch), offline ROS's weights, arm W.
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                jobs.append(Job("E2x", "n10_ros_batch", rid, ds, r, s, ["W"]))
    elif exp == "E2y":
        # Suggested 11 Oct (author asked for one more ROS variant): offline ROS with equal copies of every
        # minority image (removes the with-replacement draw; same budget and weights), arm W.
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                jobs.append(Job("E2y", "n11_ros_equal", rid, ds, r, s, ["W"]))
    elif exp == "E2z":
        # Author request 11 Oct: offline ROS with plain (unweighted) CE at classifier seed 42 (arm U), on the
        # duplicate sets saved by E2 - the weighted row (E2w, arm W) differs only in the loss.
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                jobs.append(Job("E2z", "n02_ros", rid, ds, r, s, ["U"], generate=False, load_from="E2"))
    elif exp == "E8":
        # Re-specified 10 Oct (author): the five generative rivals regenerated with their sets saved; arm W
        # (classifier seed 42) = reproduction check at every seed, arm Ws (run seed, D14) = Table 6 rows.
        # No unweighted arm (author decision; E2u's unweighted rows stay on disk only).
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                for m in P.GENERATIVE_RIVALS:
                    jobs.append(Job("E8", m, rid, ds, r, s, ["W", "Ws"],
                                    reproduce=P.REFERENCE_ANCHOR_CSV[rid]))
    elif exp == "E7":
        # arm (1), standard augmentation only, is the E2w row (arm W) of n04_stdaug_online (reused).
        for ds, rid, r in anchors:
            for s in P.SEEDS:
                jobs.append(Job("E7", P.PROPOSED, rid, ds, r, s, ["W"], generate=False,
                                load_from="E1", online=True, tag="plus_online"))
                comp = P.E7_COMPARATOR[ds]
                jobs.append(Job("E7", comp, rid, ds, r, s, ["W"], generate=False,
                                load_from="E8", online=True, tag="plus_online"))
    elif exp == "E3":
        # Re-specified 11 Oct (authors): the three remedies of Table 6, exactly as there, in the six
        # non-anchor cells of Table 7; classifier seed 42. The naive baseline runs first in every cell
        # as the reproduction check of the submitted grid (rule B1), so a mismatch stops E3 early.
        cells = [(ds, rid, P.ratio_for(ds)) for ds, rids in P.TABLE7_GRID_CELLS.items() for rid in rids]
        for ds, rid, r in cells:
            for s in P.SEEDS:
                jobs.append(Job("E3", "m11_imbalanced_naive", rid, ds, r, s, ["W"],
                                reproduce=P.reference_grid_csv(ds, rid)))
        for ds, rid, r in cells:
            for s in P.SEEDS:
                jobs.append(Job("E3", "n01_class_weighted", rid, ds, r, s, ["W"], tag="isqrt",
                                weight_alpha=0.5))                             # as E2t
                jobs.append(Job("E3", "n02_ros", rid, ds, r, s, ["U"]))       # as E2z
                jobs.append(Job("E3", "n08_stdaug_budget_zc32", rid, ds, r, s, ["W"]))  # as E2s
    elif exp == "AB":
        # Table 3 panel B (R1.6, R3.2; authors 13 Oct): the eight selection rules, generator co-training off
        # and on, on panel A's data (kbs2.ablation_data) with panel A's full-balance target; case-study
        # downstream protocol, classifier seed 42; seeds 42-56 as panel A. Within a seed the backbone is
        # trained once (backbone cache) and shared by the sixteen arms.
        from casestudy.methods.selection_v2 import MODES
        from kbs2 import ablation_data as A
        for s in range(42, 57):
            for cot in (False, True):
                for mode in MODES:
                    jobs.append(Job("AB", "m10v2_ab8c", A.REGIME, "fashion_mnist", A.FULL_BALANCE_RATIO, s, ["W"],
                                    tag=f"{mode}_{'cot' if cot else 'nocot'}", sel=mode, cotrain=cot))
    elif exp == "SE":
        # Sensitivity (R1.5; authors 13 Oct, option c): the proposed method under the submitted protocol
        # (Table 6: anchor regimes, seeds 42-51, classifier seed 42), one setting changed at a time from
        # tau 0.6 / pool factor 10 / 20 per cycle. tau 0.5 is not re-run: E1b's arm W is that setting.
        # Reference for every comparison: E1 (the reproduced Table 6 rows). Analysis: kbs2/se_analysis.py.
        for ds, rid, r in anchors:
            for tau in SE_TAUS:
                for s in P.SEEDS:
                    jobs.append(Job("SE", P.PROPOSED, rid, ds, r, s, ["W"], tag=f"tau{tau:g}", tau=tau))
        ds, rid, r = next(a for a in anchors if a[0] == "fashion_mnist")
        for pf in SE_POOL_FACTORS:
            for s in P.SEEDS:
                jobs.append(Job("SE", P.PROPOSED, rid, ds, r, s, ["W"], tag=f"pf{pf}", pool_factor=pf))
        for n in SE_PER_CYCLE:
            for s in P.SEEDS:
                jobs.append(Job("SE", P.PROPOSED, rid, ds, r, s, ["W"], tag=f"spc{n}", spc=n))
    else:
        raise ValueError(f"no jobs for {exp}")
    return jobs


# ── approvals ───────────────────────────────────────────────────────────────
def _approval(step: str) -> Optional[dict]:
    path = os.path.join(OUT, "approvals", f"{step}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    return rec if rec.get("approved") is True else None


def _gate(step: str) -> Optional[dict]:
    path = os.path.join(OUT, step, "gate_report.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def check_allowed(exp: str) -> None:
    i = SEQUENCE.index(exp)
    prev = SEQUENCE[i - 1]
    problems = []
    if _approval(prev) is None:
        problems.append(f"approval for the previous step {prev} is missing")
    if prev in GATED:
        g = _gate(prev)
        if g is None or g.get("decision") not in ("pass_exact", "pass_tolerant"):
            problems.append(f"gate report of {prev} has not passed")
    if _approval(exp) is None:
        problems.append(f"approval file saved_kbs2/approvals/{exp}.json is missing")
    if os.path.exists(os.path.join(OUT, exp, "COMPLETE")):
        problems.append(f"{exp} is already complete; it is not re-run (rule B4/B6)")
    # Public release: the authors' approval records are not distributed; report instead of stopping.
    problems = [p for p in problems if "approval" not in p]
    if problems:
        raise SystemExit(f"{exp} not allowed:\n  - " + "\n  - ".join(problems))


# ── data ────────────────────────────────────────────────────────────────────
_DATA: Dict[str, dict] = {}


def _sha(t) -> str:
    import numpy as np
    import torch
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()


def load_regime(rid: str):
    """Pool and test split exactly as ``cs3j_runner.screen`` builds them."""
    if rid in _DATA:
        return _DATA[rid]
    from kbs2 import ablation_data
    if rid == ablation_data.REGIME:
        _DATA[rid] = ablation_data.load(P.submitted_config())
        return _DATA[rid]
    import cs3j_runner as r
    from casestudy.data import build_dataset
    from casestudy.reproducibility import set_seed

    config = P.submitted_config()
    regime = r.REGIMES[rid]
    base_spec = config.datasets[regime.get("dataset", "fashion_mnist")]
    spec = r.regime_spec(base_spec, rid, regime)
    set_seed(config.random_seed)
    data = build_dataset(spec, config, use_cache=True, verbose=False)
    _DATA[rid] = {"config": config, "spec": spec, "pool": data["trainpool"], "test": data["test"],
                  "cache": [config.cache_path(spec.name, s) for s in ("trainpool", "test")]}
    return _DATA[rid]


def split(pool, config):
    from sklearn.model_selection import StratifiedShuffleSplit
    X, y = pool["images"], pool["labels"]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=config.random_seed)
    tr, ho = next(sss.split(X.numpy(), y.numpy()))
    return tr, ho


def build_any(key: str, config, spec, device: str, seed: int, save_candidates: bool,
              tau: Optional[float] = None):
    from casestudy.methods import build_method
    from kbs2 import baselines
    from kbs2.instrumented import InstrumentedAb8c
    if key in baselines.REGISTRY:
        return baselines.REGISTRY[key](config, spec, device, seed)
    if key == P.PROPOSED:
        m = InstrumentedAb8c(config, spec, device, seed, save_candidates=save_candidates, tau=tau)
        m.check_expected()
        return m
    return build_method(key, config, spec, device, seed=seed)


def build_v2(job: Job, config, spec, device: str):
    """The proposed method in its instrumented form (``casestudy.methods.m10v2_ab8c.Ab8cV2``): candidates
    from named random streams, so that with co-training off every selection rule sees identical
    candidates; the backbone from the shared cache (one per regime and seed). Everything else as
    submitted: tau 0.6, internal 80/20 split, fallback to the most confident candidates."""
    os.environ.setdefault("KBS_SAVE_ROOT", OUT)        # backbone cache under saved_kbs2/backbones
    from casestudy.methods.m10v2_ab8c import Ab8cV2
    m = Ab8cV2(config, spec, device, job.seed)
    m.selection_mode, m.cotrain_g = job.sel, bool(job.cotrain)
    m.fallback, m.internal_split, m.regime = "top_confidence", True, job.regime
    if abs(float(m.min_confidence) - P.SUBMITTED_TAU) > 1e-12:
        raise SystemExit(f"tau {m.min_confidence} differs from the submitted {P.SUBMITTED_TAU}")
    return m


def _synth_path(exp: str, method: str, rid: str, seed: int) -> str:
    return os.path.join(OUT, exp, "synth", f"{method}__{rid}__s{seed}.pt")


def generate(job: Job, D: dict, device: str) -> dict:
    """Steps 2-3 of the submitted protocol; saves and returns the synthetic set."""
    import numpy as np
    import torch
    from casestudy.methods import balance_targets_ratio
    from casestudy.reproducibility import set_seed

    config, spec, pool = D["config"], D["spec"], D["pool"]
    tr, ho = split(pool, config)
    Xtr, ytr = pool["images"][tr], pool["labels"][tr]
    t0 = time.time()
    set_seed(job.seed)
    if job.method == "m10v2_ab8c":
        method = build_v2(job, config, spec, device)
    else:
        method = build_any(job.method, config, spec, device, job.seed, job.save_candidates, job.tau)
    if job.pool_factor is not None:            # SE: one setting changed from the submitted value
        method.pool_factor = int(job.pool_factor)
    if job.spc is not None:
        method.samples_per_cycle = int(job.spc)
    method.fit(Xtr, ytr)
    targets = balance_targets_ratio(ytr, job.ratio)
    X_syn, y_syn = method.synthesize(targets)
    gen_s = time.time() - t0
    rec = {"X": X_syn.cpu(), "y": y_syn.cpu(), "targets": targets, "gen_seconds": gen_s,
           "train_index_sha": _sha(np.asarray(tr)), "holdout_index_sha": _sha(np.asarray(ho))}
    if hasattr(method, "source_index"):
        rec["source_index"] = method.source_index
    if hasattr(method, "gate_log"):
        rec["gate_log"] = method.gate_log
        rec["val_f2_by_cycle"] = list(getattr(method, "val_f2_log", getattr(method, "val_curve", [])))
        rec["tau"] = float(method.min_confidence)
        rec["pool_factor"] = int(getattr(method, "pool_factor", 0))
        rec["samples_per_cycle"] = int(getattr(method, "samples_per_cycle", 0))
        if hasattr(method, "admitted_metadata"):
            rec["admitted_meta"] = method.admitted_metadata(method._deficits(targets))
    if job.exp == "AB":
        rec["candidates"] = method.candidate_arrays()
        rec["timings"] = dict(method.timings)
        rec["selection_mode"], rec["cotrain_g"] = method.selection_mode, method.cotrain_g
    path = _synth_path(job.exp, job.method + (f"_{job.tag}" if job.exp in ("AB", "SE") else ""), job.regime, job.seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(rec, path)
    if getattr(method, "candidate_store", None):
        cpath = os.path.join(OUT, job.exp, "candidates", f"{job.method}__{job.regime}__s{job.seed}.npz")
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        flat = {}
        for i, c in enumerate(method.candidate_store):
            for k, v in c.items():
                flat[f"{i:03d}_{k}"] = np.asarray(v)
        np.savez_compressed(cpath, **flat)
    rec["path"] = path
    return rec


def load_synth(job: Job) -> dict:
    import torch
    path = _synth_path(job.load_from, job.load_method or job.method, job.regime, job.seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f"synthetic set to reuse not found: {path}")
    # The file was written by generate() in this repository (tensors, dicts and numpy
    # arrays); weights_only=True would reject the numpy metadata.
    rec = torch.load(path, weights_only=False)
    rec["path"] = path
    return rec


def downstream(job: Job, arm: str, D: dict, S: dict, device: str) -> dict:
    """Step 4 of the submitted protocol for one arm."""
    import numpy as np
    import torch
    from casestudy.classifiers import build_classifier, evaluate_classifier, train_classifier
    from casestudy.eval import compute_metrics
    from casestudy.reproducibility import set_seed
    from kbs2 import baselines
    from kbs2.per_epoch import per_epoch_ros
    from kbs2.weights import class_weight_rule, rule_name

    config, spec, pool, test = D["config"], D["spec"], D["pool"], D["test"]
    K = len(pool["classes"])
    tr, ho = split(pool, config)
    Xtr, ytr = pool["images"][tr], pool["labels"][tr]
    Xho, yho = pool["images"][ho], pool["labels"][ho]
    X_syn, y_syn = S["X"], S["y"]
    if X_syn.numel() > 0:
        X_aug, y_aug = torch.cat([Xtr, X_syn], dim=0), torch.cat([ytr, y_syn], dim=0)
    else:
        X_aug, y_aug = Xtr, ytr

    tmt = P.treatment(P.PROPOSED if job.method == "m10v2_ab8c" else job.method, arm, config)
    transform = baselines.online_transform(job.seed, device) if job.online else None

    cseed = P.clf_seed(arm, job.seed, config)   # 42 (submitted) or the run seed (D14)
    t0 = time.time()
    set_seed(cseed)
    clf = build_classifier(spec, K, config)
    init_sha = hashlib.sha256(b"".join(
        v.detach().cpu().contiguous().numpy().tobytes() for v in clf.state_dict().values())).hexdigest()
    sampling = (per_epoch_ros(ytr, S["targets"], job.seed, baselines.PER_EPOCH[job.method])
                if job.method in baselines.PER_EPOCH else contextlib.nullcontext())
    with class_weight_rule(job.weight_alpha), sampling:
        hist = train_classifier(clf, X_aug, y_aug, Xho, yho, config, device,
                                num_classes=K, seed=cseed,
                                use_class_weights=tmt["use_class_weights"], monitor=tmt["monitor"],
                                train_transform=transform)
    preds, probs = evaluate_classifier(clf, test["images"], test["labels"], device,
                                       batch_size=config.batch_size)
    m = compute_metrics(test["labels"].numpy(), preds, probs, K)
    clf_s = time.time() - t0

    pdir = os.path.join(OUT, job.exp, "predictions")
    os.makedirs(pdir, exist_ok=True)
    np.savez_compressed(os.path.join(pdir, f"{job.name}__{arm}.npz"),
                        y_true=test["labels"].numpy(), y_pred=preds, probs=probs)
    row = {"exp": job.exp, "dataset": job.dataset, "regime": job.regime, "ratio": job.ratio,
           "seed": job.seed, "method": job.method, "tag": job.tag, "arm": arm,
           "use_class_weights": tmt["use_class_weights"], "monitor": tmt["monitor"],
           "weight_rule": rule_name(job.weight_alpha) if tmt["use_class_weights"] else "none",
           "clf_seed": cseed,
           "tau": S.get("tau", P.SUBMITTED_TAU if job.method == P.PROPOSED else ""),
           "online_transform": job.online, "synth_source": os.path.relpath(S["path"], ROOT),
           "n_synth": int(X_syn.shape[0]),
           **{k: float(v) for k, v in m.items()},
           "best_epoch": hist.get("best_epoch"), "stopped_epoch": hist.get("stopped_epoch"),
           "gen_seconds": round(float(S.get("gen_seconds", float("nan"))), 1),
           "clf_seconds": round(clf_s, 1),
           "pool_sha": _sha(pool["images"]), "test_sha": _sha(test["images"]),
           "train_index_sha": S.get("train_index_sha"), "holdout_index_sha": S.get("holdout_index_sha"),
           "synth_sha": _sha(X_syn) if X_syn.numel() else "empty",
           "trainset_sha": _sha(X_aug), "init_weights_sha": init_sha}
    if job.exp == "SE":
        row.update(pool_factor=S.get("pool_factor"), samples_per_cycle=S.get("samples_per_cycle"))
    return row


def _append(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def run(exp: str) -> None:
    from kbs2 import gate, provenance

    check_allowed(exp)
    prov = provenance.check_or_die()
    jobs = jobs_for(exp)
    outdir = os.path.join(OUT, exp)
    os.makedirs(outdir, exist_ok=True)
    cfg = P.submitted_config()
    device = cfg.resolve_device()
    with open(os.path.join(outdir, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump({"check": prov, "record": provenance.record(), "n_jobs": len(jobs),
                   "approval": _approval(exp), "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  fh, indent=2, default=str)
    rows_csv = os.path.join(outdir, "runs.csv")
    for i, job in enumerate(jobs, 1):
        D = load_regime(job.regime)
        S = generate(job, D, device) if job.generate else load_synth(job)
        for arm in job.arms:
            row = downstream(job, arm, D, S, device)
            _append(rows_csv, row)
            print(f"[{exp} {i}/{len(jobs)}] {job.name} arm {arm}: macro_f2={row['macro_f2']:.4f}", flush=True)
            if job.reproduce and arm == "W":
                verdict = gate.check_row(row, job.reproduce)
                gate.append_check(outdir, verdict)
                if verdict["decision"] == "material_failure":
                    gate.write_report(exp, outdir, stopped=True)
                    raise SystemExit(f"{exp} STOPPED (rule B1): reproduction failed for {job.name}: "
                                     f"{verdict['worst']}")
    if exp in GATED:
        gate.write_report(exp, outdir, stopped=False)
    with open(os.path.join(outdir, "COMPLETE"), "w") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(f"{exp} complete. STOP: report the result and wait for author approval (rule B7).")


def status() -> None:
    for step in SEQUENCE:
        g = _gate(step)
        done = os.path.exists(os.path.join(OUT, step, "COMPLETE"))
        print(f"{step:3s} approval={'yes' if _approval(step) else 'no ':3s} "
              f"complete={'yes' if done else 'no '} gate={g.get('decision') if g else '-'}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["list", "run", "status"])
    ap.add_argument("exp", nargs="?", choices=["E1", "E1b", "E2", "E2u", "E8", "E2w", "E2p", "E2q", "E2r", "E2s", "E2t", "E2v", "E2x", "E2y", "E2z", "E7", "E3", "AB", "SE"])
    a = ap.parse_args(argv)
    if a.command == "status":
        status()
    elif a.command == "list":
        jobs = jobs_for(a.exp)
        for j in jobs:
            src = "generate" if j.generate else f"reuse {j.load_from}:{j.load_method or j.method}"
            print(f"{j.name:48s} arms={','.join(j.arms):3s} {src:28s} online={j.online} "
                  f"repro={'yes' if j.reproduce else 'no'}")
        print(f"{len(jobs)} jobs")
    else:
        run(a.exp)


if __name__ == "__main__":
    main()
