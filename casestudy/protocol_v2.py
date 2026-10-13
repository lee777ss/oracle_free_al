"""casestudy.protocol_v2 - the corrected evaluation protocol of the KBS revision (S1).

One run with seed ``s``:

1. Pool: the majority class is subsampled to its cap and each minority class to rho times
   that, drawn with seed ``s`` from the official training split minus the development set
   (and minus any image that also occurs in a test split).
2. Hold-out: a stratified 10% of the pool, split with seed ``s`` before anything is trained.
3. Augmenter: fitted on the remaining 90% and asked for the shared target
   ``balance_targets_ratio(y, r)``; its randomness comes from the ``augmenter`` stream of ``s``.
4. Downstream classifier: built after ``set_seed(s)``, so its initial weights are identical
   for every method within a seed; class-weighted cross-entropy and macro-F2 model selection
   on the hold-out for every method (``b0_plain`` keeps plain cross-entropy).
5. Test: scored once on the cached test set of the submitted runs ("test") and on every
   official test image of the three classes, re-weighted to the class prior of "test"
   ("test_full"; for OCTMNIST this includes the official validation split and is the
   primary evaluation). Pilot runs are scored on the development set only ("dev").

``protocol="v1"`` reproduces ``cs3j_runner.run_one`` (pool, split and classifier pinned to
seed 42, per-method loss and monitor) and exists only for the regression check.

Every run leaves ``result.json`` (written last and atomically), ``preds.npz``, ``synth.npz``
and, where they exist, ``gate.json``, ``candidates.npz``, ``gallery.npz`` (admitted and rejected
example images of the proposed method) and ``generator.pt`` in
``saved_kbs/runs/<stage>/<regime>/<method>[+<variant>]/s<seed>/``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from .classifiers.trainer import build_classifier, evaluate_classifier, train_classifier
from .data.loaders import _load_raw, _remap, build_dataset
from .data.transforms import to_normalized_tensor
from .eval.metrics import compute_metrics
from .methods.base import balance_targets_ratio, build_method
from .reproducibility import set_seed
from .seeds import stream_seed

print("casestudy/protocol_v2.py loaded")

DEV_SEED = 20261002
DEV_PER_CLASS = {"fashion_mnist": 1000, "cifar10": 1000, "stl10": 60}
FULL_IS_PRIMARY = {"octmnist"}

# regime: (dataset, rho, augmentation ratio r, role). r is held fixed per dataset (D9).
REGIMES_V2: Dict[str, dict] = {
    "R1_confusable":  dict(dataset="fashion_mnist", rho=0.05, ratio=4, role="anchor"),
    "C2_cifar_paper": dict(dataset="cifar10", rho=0.30, ratio=2, role="anchor"),
    "S2_stl_mod":     dict(dataset="stl10", rho=0.10, ratio=3, role="anchor"),
    **{f"R1_rho{k:04d}": dict(dataset="fashion_mnist", rho=k / 1000, ratio=4, role="grid") for k in (25, 100, 150, 200)},
    **{f"C2_rho{k:04d}": dict(dataset="cifar10", rho=k / 1000, ratio=2, role="grid") for k in (100, 150, 200, 500)},
    **{f"S2_rho{k:04d}": dict(dataset="stl10", rho=k / 1000, ratio=3, role="grid") for k in (50, 150, 200, 300)},
    "OCT1_pathology_severe": dict(dataset="octmnist", rho=0.05, ratio=3, role="domain"),
    "OCT2_pathology_mod":    dict(dataset="octmnist", rho=0.10, ratio=3, role="domain"),
    "G1_signs_severe":       dict(dataset="gtsrb", rho=0.05, ratio=3, role="domain"),
    "G2_signs_mod":          dict(dataset="gtsrb", rho=0.10, ratio=3, role="domain"),
}

# BAGAN in the field is the variant with label smoothing 0.1, as in the submitted runs: without smoothing it
# collapsed to noise-like images in about half of its runs on Fashion-MNIST and STL-10 (decided 5 October, Q1);
# the unsmoothed variant m08_bagan is reported in the appendix.
FIELD = ["b0_plain", "b1_weighted", "m12_ros", "m13_stdaug", "m14_stdaug_online", "m01_smote", "m02_bsmote",
         "m03_adasyn", "m04_dcgan", "m05_cgan", "m06_acgan", "m07_wgangp", "m08_bagan_ls", "m09_bagan_gp", "p_proposed"]
LABELS = {
    "b0_plain": "No augmentation (plain CE)", "b1_weighted": "No augmentation (class-weighted CE)",
    "m11_naive": "No augmentation, accuracy-selected (submitted)", "m12_ros": "Random oversampling",
    "m13_stdaug": "Standard augmentation (budget-matched)", "m14_stdaug_online": "Standard augmentation (on the fly)",
    "m01_smote": "SMOTE", "m02_bsmote": "Borderline-SMOTE", "m03_adasyn": "ADASYN", "m04_dcgan": "DCGAN",
    "m05_cgan": "cGAN", "m06_acgan": "ACGAN", "m07_wgangp": "WGAN-GP", "m08_bagan": "BAGAN, no label smoothing",
    "m08_bagan_ls": "BAGAN", "m09_bagan_gp": "BAGAN-GP", "p_proposed": "Proposed",
}
PLAIN_CE = {"b0_plain", "m11_naive"}


def save_root() -> Path:
    return Path(os.environ.get("KBS_SAVE_ROOT", "saved_kbs"))


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _sha(*arrays) -> str:
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------- configuration
def make_config(overrides: Optional[dict] = None):
    """The full-budget configuration of the submitted runs (``cs3j_runner.quick_config(quick=False)``)."""
    import cs3j_runner as R

    cfg = R.quick_config(quick=False)
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


def regime_spec(regime: str, cfg):
    import cs3j_runner as R

    ds = REGIMES_V2[regime]["dataset"]
    return R.regime_spec(cfg.datasets[ds], regime, R.REGIMES[regime])


# ---------------------------------------------------------------------------- data
_RAW: Dict[str, dict] = {}
_REG: Dict[Tuple[str, str, int], "RegimeData"] = {}


def _raw(dataset: str, spec, cfg) -> dict:
    """Official splits of the three classes, the development set and the reservoir, once per process."""
    if dataset in _RAW:
        return _RAW[dataset]
    classes = list(spec.selected_classes)
    tr_x, tr_y = _load_raw(spec.loader_key, True, cfg.data_root)
    te_x, te_y = _load_raw(spec.loader_key, False, cfg.data_root)
    if spec.loader_key == "octmnist":       # the official validation split joins the full test evaluation
        npz = np.load(os.path.join(cfg.data_root, "octmnist.npz"))
        te_x = np.concatenate([npz["val_images"], te_x])
        te_y = np.concatenate([npz["val_labels"].reshape(-1), np.asarray(te_y).reshape(-1)])
    tr_y, te_y = np.asarray(tr_y).reshape(-1), np.asarray(te_y).reshape(-1)
    m_tr, m_te = np.isin(tr_y, classes), np.isin(te_y, classes)
    trx, trl = tr_x[m_tr], _remap(tr_y[m_tr], classes)
    tex, tel = te_x[m_te], _remap(te_y[m_te], classes)

    dev_idx = np.array([], dtype=int)
    if DEV_PER_CLASS.get(dataset):
        rng = np.random.default_rng(DEV_SEED)
        dev_idx = np.sort(np.concatenate([rng.permutation(np.where(trl == c)[0])[:DEV_PER_CLASS[dataset]]
                                          for c in range(len(classes))]))
    keep = np.ones(len(trl), dtype=bool)
    keep[dev_idx] = False
    test_hashes = {hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest() for img in tex}
    dup = np.array([hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest() in test_hashes for img in trx])
    n_dup = int((dup & keep).sum())
    keep &= ~dup
    _RAW[dataset] = dict(classes=classes, res_x=trx[keep], res_y=trl[keep], dev_x=trx[dev_idx], dev_y=trl[dev_idx],
                         full_x=tex, full_y=tel, n_dup=n_dup)
    return _RAW[dataset]


@dataclass
class RegimeData:
    regime: str
    dataset: str
    spec: object
    rho: float
    ratio: object
    K: int
    res_x: np.ndarray
    res_y: np.ndarray
    test: Tuple[torch.Tensor, torch.Tensor]
    test_full: Tuple[torch.Tensor, torch.Tensor]
    dev: Optional[Tuple[torch.Tensor, torch.Tensor]]
    prior: np.ndarray
    primary: str
    test_hash: str
    n_dup_removed: int
    counts: dict = field(default_factory=dict)


def get_regime(regime: str, cfg) -> RegimeData:
    key = (regime, cfg.date_suffix, cfg.image_size)
    if key in _REG:
        return _REG[key]
    info = REGIMES_V2[regime]
    spec = regime_spec(regime, cfg)
    raw = _raw(info["dataset"], spec, cfg)
    to_t = lambda x: to_normalized_tensor(x, spec.channels, cfg.image_size, cfg.normalize_range)
    cached = build_dataset(spec, cfg, use_cache=True, verbose=False)["test"]    # the submitted runs' test set
    Xt, yt = cached["images"], cached["labels"]
    K = len(spec.selected_classes)
    prior = np.bincount(yt.numpy(), minlength=K) / len(yt)
    dev = (to_t(raw["dev_x"]), torch.from_numpy(raw["dev_y"]).long()) if len(raw["dev_y"]) else None
    rd = RegimeData(
        regime=regime, dataset=info["dataset"], spec=spec, rho=float(info["rho"]), ratio=info["ratio"], K=K,
        res_x=raw["res_x"], res_y=raw["res_y"], test=(Xt, yt),
        test_full=(to_t(raw["full_x"]), torch.from_numpy(raw["full_y"]).long()), dev=dev, prior=prior,
        primary="test_full" if info["dataset"] in FULL_IS_PRIMARY else "test",
        test_hash=_sha(Xt.numpy(), yt.numpy()), n_dup_removed=raw["n_dup"],
        counts={"reservoir": np.bincount(raw["res_y"], minlength=K).tolist(),
                "dev": np.bincount(raw["dev_y"], minlength=K).tolist(),
                "test": np.bincount(yt.numpy(), minlength=K).tolist(),
                "test_full": np.bincount(raw["full_y"], minlength=K).tolist()},
    )
    _REG[key] = rd
    return rd


def sample_pool(labels: np.ndarray, majority_cap: Optional[int], rho: float, seed: int) -> np.ndarray:
    """Indices of the imbalanced pool; the same draws as ``loaders._apply_imbalance``."""
    rng = np.random.default_rng(seed)
    maj = np.where(labels == 0)[0]
    n_maj = len(maj) if majority_cap is None else min(len(maj), majority_cap)
    chosen = [rng.choice(maj, size=n_maj, replace=False) if n_maj < len(maj) else maj]
    n_min = int(round(rho * n_maj))
    for c in sorted(int(v) for v in np.unique(labels) if int(v) != 0):
        ci = np.where(labels == c)[0]
        k = min(n_min, len(ci))
        chosen.append(rng.choice(ci, size=k, replace=False) if k < len(ci) else ci)
    return np.sort(np.concatenate(chosen))


# ---------------------------------------------------------------------------- methods
def build_method_v2(key: str, cfg, spec, device: str, seed: int, regime: str, overrides: Optional[dict] = None):
    from .methods.m00_imbalanced import ImbalancedBaseline, ImbalancedBaselineNaive
    from .methods.m08_bagan import BAGANMethod
    from .methods.m10v2_ab8c import Ab8cV2, BAGANGPCached
    from .methods.m12_ros import ROSMethod
    from .methods.m13_stdaug import StdAugMethod, StdAugOnline

    table = {"b0_plain": ImbalancedBaseline, "b1_weighted": ImbalancedBaseline, "m11_naive": ImbalancedBaselineNaive,
             "m12_ros": ROSMethod, "m13_stdaug": StdAugMethod, "m14_stdaug_online": StdAugOnline,
             "m08_bagan": BAGANMethod, "m08_bagan_ls": BAGANMethod, "m09_bagan_gp": BAGANGPCached, "p_proposed": Ab8cV2}
    method = table[key](cfg, spec, device, seed) if key in table else build_method(key, cfg, spec, device, seed)
    defaults = {"m08_bagan": {"label_smoothing": 0.0}, "m08_bagan_ls": {"label_smoothing": 0.1}}
    for k, v in {**defaults.get(key, {}), **(overrides or {})}.items():
        if not hasattr(method, k):
            raise AttributeError(f"{key}: no attribute {k!r} to override")
        setattr(method, k, v)
    if hasattr(method, "regime"):
        method.regime = regime
    return method


def export_generator(method) -> Optional[dict]:
    gens = getattr(method, "_generators", None)
    if gens:
        return {"per_class": {int(c): g.state_dict() for c, g in gens.items()}}
    G = getattr(method, "_G", None)
    if G is None:
        return None
    out = {"G": G.state_dict()}
    prior = getattr(method, "_prior", None)
    if prior is not None:
        out["prior_means"] = {int(c): m.cpu() for c, m in prior.means.items()}
        out["prior_stds"] = {int(c): float(s) for c, s in prior.stds.items()}
    return out


def _n_params(module) -> int:
    return int(sum(p.numel() for p in module.parameters())) if module is not None else 0


# ---------------------------------------------------------------------------- metrics
def full_metrics(y_true, y_pred, y_prob, K: int, prior: Optional[np.ndarray] = None) -> dict:
    """Headline, extra and per-class metrics; with ``prior`` the confusion matrix is re-weighted row by
    row to that class prior first (recall and G-mean do not depend on it)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(K))).astype(float)
    n_c = cm.sum(axis=1)
    cmw = cm * (np.asarray(prior) * cm.sum() / np.clip(n_c, 1, None))[:, None] if prior is not None else cm
    tp = np.diag(cmw)
    fn = cmw.sum(axis=1) - tp
    fp = cmw.sum(axis=0) - tp
    tn = cmw.sum() - tp - fn - fp
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    rec, prec, spec = div(tp, tp + fn), div(tp, tp + fp), div(tn, tn + fp)
    f2, f1 = div(5 * tp, 5 * tp + 4 * fn + fp), div(2 * tp, 2 * tp + fn + fp)
    base = compute_metrics(y_true, y_pred, y_prob, K)
    return {
        "macro_f2": float(f2.mean()), "gmean": float(np.prod(rec) ** (1 / K)) if np.all(rec > 0) else 0.0,
        "macro_recall": float(rec.mean()), "macro_specificity": float(spec.mean()),
        "macro_precision": float(prec.mean()), "macro_f1": float(f1.mean()),
        "accuracy": float(tp.sum() / cmw.sum()), "macro_auc": float(base["macro_auc"]),
        "reweighted": prior is not None, "n": int(cm.sum()),
        "per_class": {"precision": prec.tolist(), "recall": rec.tolist(), "f2": f2.tolist(), "specificity": spec.tolist()},
        "confusion": cm.astype(int).tolist(),
    }


# ---------------------------------------------------------------------------- one run
@dataclass
class Job:
    stage: str
    regime: str
    method: str
    seed: int
    variant: str = ""
    overrides: dict = field(default_factory=dict)
    protocol: str = "v2"
    eval_split: str = "test"           # "test" -> test and test_full; "dev" -> development set only
    save_generator: bool = False
    oracle: bool = False
    config_overrides: dict = field(default_factory=dict)
    ratio: object = None               # augmentation ratio r; None -> the regime's fixed value (E9 varies it)
    reuse: Optional[str] = None        # key of a finished run whose synthetic set is reused; no augmenter is trained (E12)
    weighted: Optional[bool] = None    # downstream loss; None -> class-weighted unless the method is in PLAIN_CE

    @property
    def name(self) -> str:
        """Folder-safe run name; characters Windows forbids in file names become underscores."""
        variant = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in self.variant)
        return self.method + (f"+{variant}" if variant else "")

    @property
    def key(self) -> str:
        return f"{self.stage}/{self.regime}/{self.name}/s{self.seed}"

    @property
    def run_dir(self) -> Path:
        return save_root() / "runs" / self.stage / self.regime / self.name / f"s{self.seed}"


_ENV: Optional[dict] = None


def env_info() -> dict:
    global _ENV
    if _ENV is None:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            commit = "nogit"
        _ENV = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "git_commit": commit}
    return _ENV


def run_one_v2(job: Job, device: Optional[str] = None, write: bool = True) -> dict:
    """Run one job end to end; returns the result record (also written to ``result.json``)."""
    t_all = time.perf_counter()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    cfg = make_config(job.config_overrides)
    device = device or cfg.resolve_device()
    rd = get_regime(job.regime, cfg)
    s, K, v1 = int(job.seed), rd.K, job.protocol == "v1"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    T: Dict[str, float] = {}

    # 1-2. pool and hold-out
    t = time.perf_counter()
    if v1:
        pool = build_dataset(rd.spec, cfg, use_cache=True, verbose=False)["trainpool"]
        X, y = pool["images"], pool["labels"]
        pool_hash = _sha(y.numpy(), X.numpy())
        split_seed = clf_seed = cfg.random_seed
    else:
        idx = sample_pool(rd.res_y, rd.spec.majority_cap, rd.rho, s)
        X = to_normalized_tensor(rd.res_x[idx], rd.spec.channels, cfg.image_size, cfg.normalize_range)
        y = torch.from_numpy(rd.res_y[idx]).long()
        pool_hash = _sha(idx)
        split_seed = clf_seed = s
    tr, ho = next(StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=split_seed)
                  .split(np.zeros(len(y)), y.numpy()))
    Xtr, ytr, Xho, yho = X[tr], y[tr], X[ho], y[ho]
    T["data_s"] = time.perf_counter() - t

    # 3. augmenter
    ratio = rd.ratio if job.ratio is None else job.ratio
    if job.reuse:                       # E12: the synthetic set of a finished run, as saved (uint8)
        src = save_root() / "runs" / job.reuse
        src_res = json.loads((src / "result.json").read_text(encoding="utf-8"))
        if src_res.get("status") != "ok" or src_res["hashes"]["pool"] != pool_hash:
            raise RuntimeError(f"reuse source {job.reuse} is not a finished run on the same pool")
        method = SimpleNamespace()
        t = time.perf_counter()
        targets = balance_targets_ratio(ytr, ratio)
        if (src / "synth.npz").exists():
            z = np.load(src / "synth.npz")
            X_syn, y_syn = torch.from_numpy(z["x"]).float() / 127.5 - 1.0, torch.from_numpy(z["y"].astype(np.int64))
        else:
            X_syn, y_syn = torch.empty(0, *Xtr.shape[1:]), torch.empty(0, dtype=torch.long)
        T["augment_s"] = time.perf_counter() - t
    else:
        if v1:
            set_seed(s)
            method = build_method(job.method, cfg, rd.spec, device, seed=s)
            for k, v in job.overrides.items():
                setattr(method, k, v)
        else:
            method = build_method_v2(job.method, cfg, rd.spec, device, s, job.regime, job.overrides)
            if job.oracle:
                from .eval.oracle import load_oracle
                method.oracle = load_oracle(rd.dataset, device)
                if method.oracle is None:
                    raise RuntimeError(f"reference classifier for {rd.dataset} not trained yet (E1)")
            set_seed(stream_seed(s, "augmenter"))
        _sync(); t = time.perf_counter()
        method.fit(Xtr, ytr)
        targets = balance_targets_ratio(ytr, ratio)
        X_syn, y_syn = method.synthesize(targets)
        _sync(); T["augment_s"] = time.perf_counter() - t
    n_target = int(sum(max(0, targets[c] - int((ytr == c).sum())) for c in targets))
    X_aug = torch.cat([Xtr, X_syn]) if X_syn.numel() else Xtr
    y_aug = torch.cat([ytr, y_syn]) if X_syn.numel() else ytr

    # 4. downstream classifier
    if v1:
        weighted, monitor = cfg.uses_class_weights(job.method), cfg.selection_metric_for(job.method)
    else:
        weighted = job.method not in PLAIN_CE if job.weighted is None else bool(job.weighted)
        monitor = "accuracy" if job.method == "m11_naive" else "macro_f2"
    set_seed(clf_seed)
    clf = build_classifier(rd.spec, K, cfg)
    clf_hash = _sha(*[t_.detach().cpu().numpy() for t_ in clf.state_dict().values()])
    transform = method.train_transform(device) if hasattr(method, "train_transform") else None
    _sync(); t = time.perf_counter()
    hist = train_classifier(clf, X_aug, y_aug, Xho, yho, cfg, device, num_classes=K, seed=clf_seed,
                            use_class_weights=weighted, monitor=monitor, train_transform=transform)
    _sync(); T["downstream_s"] = time.perf_counter() - t

    # 5. evaluation
    t = time.perf_counter()
    splits = ["dev"] if job.eval_split == "dev" else (["test"] if v1 else ["test", "test_full"])
    sources = {"test": rd.test, "test_full": rd.test_full, "dev": rd.dev}
    metrics, preds = {}, {}
    for sp in splits:
        Xe, ye = sources[sp]
        p_, pr_ = evaluate_classifier(clf, Xe, ye, device, batch_size=cfg.batch_size)
        metrics[sp] = full_metrics(ye.numpy(), p_, pr_, K, None if sp == "test" else rd.prior)
        preds[sp] = (ye.numpy(), p_, pr_)
    _sync(); T["inference_s"] = time.perf_counter() - t
    T["total_s"] = time.perf_counter() - t_all

    gens = getattr(method, "_generators", None) or {}
    result = {
        "stage": job.stage, "regime": job.regime, "dataset": rd.dataset, "rho": rd.rho, "ratio": str(ratio),
        "method": job.method, "variant": job.variant, "label": LABELS.get(job.method, job.method), "seed": s,
        "protocol": job.protocol, "status": "ok", "primary": rd.primary if job.eval_split != "dev" else "dev",
        "overrides": job.overrides, "config_overrides": job.config_overrides,
        "metrics": metrics, "n_synth": int(X_syn.shape[0]), "n_synth_target": n_target, "reuse": job.reuse,
        "shortfall": {str(k): v for k, v in getattr(method, "shortfall", {}).items()},
        "n_train": int(len(ytr)), "n_holdout": int(len(yho)),
        "downstream": {"weighted_ce": bool(weighted), "monitor": monitor, "best_epoch": int(hist["best_epoch"]),
                       "stopped_epoch": int(hist["stopped_epoch"]), "best_score": float(hist["best_score"])},
        "timings_s": T, "method_timings_s": {k: (v.item() if hasattr(v, "item") else v)
                                              for k, v in getattr(method, "timings", {}).items()},
        "peak_mem_mb": float(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0.0,
        "params": {"downstream": _n_params(clf),
                   "generator": sum(_n_params(g) for g in gens.values()) if gens else _n_params(getattr(method, "_G", None)),
                   "critic": _n_params(getattr(method, "_D", None)),
                   "internal_clf": _n_params(getattr(method, "_C", None))},
        "hashes": {"pool": pool_hash, "holdout": _sha(ho), "clf_init": clf_hash, "test": rd.test_hash},
        "gate": _gate_summary(getattr(method, "gate_log", None)),
        "audit": _audit_summary(method),
        "config_digest": hashlib.sha1(json.dumps(dataclasses.asdict(cfg), default=str, sort_keys=True).encode()).hexdigest()[:12],
        "env": env_info(), "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if write:
        _write(job, result, method, X_syn, y_syn, preds)
    return result


def _gate_summary(log) -> dict:
    if not log:
        return {}
    return {"class_cycles": len(log), "fallback_cycles": int(sum(r["fell_back"] for r in log)),
            "admitted": int(sum(r["n_admitted"] for r in log)), "violating": int(sum(r["n_violating"] for r in log)),
            "violating_argmax": int(sum(r["violating_argmax"] for r in log)),
            "violating_below_tau": int(sum(r["violating_below_tau"] for r in log)),
            "mean_pass_rate": float(np.mean([r["pass_rate"] for r in log]))}


def _audit_summary(method) -> dict:
    arrays = method.candidate_arrays() if hasattr(method, "candidate_arrays") else {}
    if "oracle_pred" not in arrays:
        return {}
    agree = arrays["oracle_pred"] == arrays["cls"]
    adm = arrays["admitted"]
    return {"agree_pool": float(agree.mean()), "agree_admitted": float(agree[adm].mean()) if adm.any() else float("nan"),
            "agree_rejected": float(agree[~adm].mean()) if (~adm).any() else float("nan"), "n_candidates": int(len(agree))}


def _to_uint8(x: torch.Tensor) -> np.ndarray:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()


def _write(job: Job, result: dict, method, X_syn, y_syn, preds) -> None:
    d = job.run_dir
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "preds.npz", **{f"{sp}_{k}": v for sp, (yt, yp, pr) in preds.items()
                                            for k, v in (("y_true", yt.astype(np.int8)), ("y_pred", yp.astype(np.int8)),
                                                         ("y_prob", pr.astype(np.float16)))})
    if X_syn.numel() and not job.reuse:             # a reused set stays with the run that made it
        np.savez_compressed(d / "synth.npz", x=_to_uint8(X_syn), y=y_syn.numpy().astype(np.int8))
    if getattr(method, "gate_log", None):
        (d / "gate.json").write_text(json.dumps(method.gate_log, indent=0), encoding="utf-8")
    if hasattr(method, "candidate_arrays") and method.candidate_arrays():
        np.savez_compressed(d / "candidates.npz", **method.candidate_arrays())
    if hasattr(method, "gallery_arrays") and method.gallery_arrays():
        np.savez_compressed(d / "gallery.npz", **method.gallery_arrays())
    if job.save_generator:
        g = export_generator(method)
        if g is not None:
            torch.save(g, d / "generator.pt")
    tmp = d / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, d / "result.json")


def write_error(job: Job, err: str) -> None:
    d = job.run_dir
    d.mkdir(parents=True, exist_ok=True)
    rec = {"stage": job.stage, "regime": job.regime, "method": job.method, "variant": job.variant, "seed": job.seed,
           "status": "error", "error": err, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = d / "result.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, d / "result.json")
