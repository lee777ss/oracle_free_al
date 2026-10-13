"""cs3j_runner — autonomous screener for the Ab-8c 'best in case study' effort (CS3J).

Phase-0/1 tooling for [130_Ab8c for Casestudy.ipynb]. Runs a *fair* held-out-test
comparison of Ab-8c vs. selected rivals across fashion_mnist regimes at reduced
(QUICK) budget, using the new fair ``synth_ratio`` target knob. Screening only —
the full 5-seed confirmation lives in the case-study notebook.

Fairness: identical data pool, classifier, budget and seed schedule for every
method; only Ab-8c uses active learning. The baseline (m00) keeps plain CE.

Usage (from the anaconda 'torch' env):
    python cs3j_runner.py                     # default screen (R1,R2 x ratios)
    python -c "import cs3j_runner as r; print(r.screen(quick=True))"
"""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

import casestudy as cs
from casestudy.reproducibility import set_seed
from casestudy.data import build_dataset
from casestudy.classifiers import build_classifier, train_classifier, evaluate_classifier
from casestudy.methods import build_method, balance_targets_ratio
from casestudy.eval import compute_metrics

print("cs3j_runner.py loaded")

# Fashion-MNIST label -> human name (for building regime specs).
_FMNIST_NAMES = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot",
}
_CIFAR10_NAMES = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck",
}
# BloodMNIST (MedMNIST) label -> cell type (8 native classes).
_BLOODMNIST_NAMES = {
    0: "basophil", 1: "eosinophil", 2: "erythroblast", 3: "immature_granulocyte",
    4: "lymphocyte", 5: "monocyte", 6: "neutrophil", 7: "platelet",
}
# OrganAMNIST (MedMNIST) label -> organ (11 classes, abdominal CT axial).
_ORGANAMNIST_NAMES = {
    0: "bladder", 1: "femur-left", 2: "femur-right", 3: "heart", 4: "kidney-left",
    5: "kidney-right", 6: "liver", 7: "lung-left", 8: "lung-right", 9: "pancreas",
    10: "spleen",
}
_NAME_MAPS = {
    "fashion_mnist": _FMNIST_NAMES, "cifar10": _CIFAR10_NAMES,
    "bloodmnist": _BLOODMNIST_NAMES, "organamnist": _ORGANAMNIST_NAMES,
    "cells": {0: "red blood cell", 1: "ring", 2: "schizont", 3: "trophozoite"},
    "gtsrb": {0: "20km/h", 1: "30km/h", 2: "50km/h", 3: "60km/h", 4: "70km/h",
              5: "80km/h", 6: "end80", 7: "100km/h", 8: "120km/h"},
    "flowers": {0: "daisy", 1: "dandelion", 2: "roses", 3: "sunflowers", 4: "tulips"},
    "stl10": {0: "airplane", 1: "bird", 2: "car", 3: "cat", 4: "deer", 5: "dog",
              6: "horse", 7: "monkey", 8: "ship", 9: "truck"},
    # EMNIST "letters" split is 1-indexed (a=1 .. z=26).
    "emnist_letters": {i: chr(ord('a') + i - 1) for i in range(1, 27)},
    "octmnist": {0: "CNV", 1: "DME", 2: "drusen", 3: "normal"},
    # KMNIST 10 hiragana (romaji): 0 o, 1 ki, 2 su, 3 tsu, 4 na, 5 ha, 6 ma, 7 ya, 8 re, 9 wo.
    "kmnist": {0: "o", 1: "ki", 2: "su", 3: "tsu", 4: "na", 5: "ha", 6: "ma", 7: "ya", 8: "re", 9: "wo"},
}

# Candidate regimes (see plan §5). classes[0] is the majority; "dataset"
# defaults to fashion_mnist for the original R* regimes.
REGIMES: Dict[str, dict] = {
    "R1_confusable": dict(classes=[2, 4, 6], rho=0.05),           # Pullover/Coat/Shirt
    "R2_5class":     dict(classes=[0, 2, 3, 4, 6], rho=0.025),    # +T-shirt/Dress
    # CIFAR-10 confusable animals (cat/dog/deer), ResNet backbone.
    "C1_cifar_confusable": dict(dataset="cifar10", classes=[3, 5, 4], rho=0.10),
    # CIFAR-10 paper-faithful (Chung et al. 2022, Table III): Airplane majority,
    # Automobile + Ship minorities, ratio 0.30, majority 1000 / minority 300.
    "C2_cifar_paper": dict(dataset="cifar10", classes=[0, 1, 8], rho=0.30, majority_cap=1000),
    # CIFAR-10 confusable animals (cat/dog/deer) WITH adequate data (maj 1000,
    # ratio 0.30 -> 300 minority): weak baseline (confusable) + enough samples.
    "C3_cifar_confusable_data": dict(dataset="cifar10", classes=[3, 5, 4], rho=0.30, majority_cap=1000),
    # CIFAR-10 paper-faithful CLASSIFIER too: Chung et al. 2022 used one small
    # 4-block CNN for every dataset (Table VIII), not ResNet -> weaker baseline
    # with room for augmentation to win (matches their baseline ~0.62).
    "C4_cifar_paper_cnn": dict(dataset="cifar10", classes=[0, 1, 8], rho=0.30,
                               majority_cap=1000, classifier="cnn"),
    "C5_cifar_confusable_cnn": dict(dataset="cifar10", classes=[3, 5, 4], rho=0.30,
                                    majority_cap=1000, classifier="cnn"),
    # CIFAR-10 genuinely imbalanced (rho=0.15 -> 150 minority): baseline honestly
    # weak (~0.62 vs 0.72 at rho0.30) but GANs not starved (unlike rho0.05).
    "C6_cifar_severe": dict(dataset="cifar10", classes=[0, 1, 8], rho=0.15, majority_cap=1000),
    # BloodMNIST (RGB 28x28, cnn) SYNTHETIC imbalance, medically-motivated trio:
    # neutrophil (most abundant WBC) majority; eosinophil + basophil (genuinely
    # rare WBCs) minorities -> honest clinical imbalance. Sweep rho to find a
    # genuine-imbalance operating point (weak baseline, GANs not starved).
    "B1_blood_severe": dict(dataset="bloodmnist", classes=[6, 1, 0], rho=0.05,
                            majority_cap=800, native_imbalance=False),
    "B2_blood_mod":    dict(dataset="bloodmnist", classes=[6, 1, 0], rho=0.10,
                            majority_cap=800, native_imbalance=False),
    "B3_blood_mild":   dict(dataset="bloodmnist", classes=[6, 1, 0], rho=0.20,
                            majority_cap=800, native_imbalance=False),
    # OrganAMNIST (GRAYSCALE abdominal CT, cnn) SYNTHETIC imbalance over a
    # confusable soft-tissue trio: liver majority; spleen + pancreas minorities
    # (genuinely hard to tell apart on a single axial slice).  Grayscale ->
    # matches Ab-8c's faithful-reconstruction strength; ~34k train -> no GAN
    # starvation.  Sweep rho to find a genuine-imbalance operating point.
    "O1_organ_severe": dict(dataset="organamnist", classes=[6, 10, 9], rho=0.05,
                            majority_cap=800, native_imbalance=False),
    "O2_organ_mod":    dict(dataset="organamnist", classes=[6, 10, 9], rho=0.10,
                            majority_cap=800, native_imbalance=False),
    "O3_organ_mild":   dict(dataset="organamnist", classes=[6, 10, 9], rho=0.20,
                            majority_cap=800, native_imbalance=False),
    # Cells (Huang & Jafari BAGAN-GP benchmark, BBBC041 malaria) used with its
    # NATIVE 4-class imbalance (no classes/rho override -> spec used as-is).
    "CELL_native": dict(dataset="cells"),
    # GTSRB confusable speed-limit trio (50 majority; 30 + 80 minorities) with
    # SYNTHETIC imbalance.  Near-identical signs -> pixel-interpolation resamplers
    # blur the digit; the paper's BAGAN win dataset.  Sweep rho.
    "G1_signs_severe": dict(dataset="gtsrb", classes=[2, 1, 5], rho=0.05, majority_cap=800),
    "G2_signs_mod":    dict(dataset="gtsrb", classes=[2, 1, 5], rho=0.10, majority_cap=800),
    "G3_signs_mild":   dict(dataset="gtsrb", classes=[2, 1, 5], rho=0.20, majority_cap=800),
    # Finer rho grid for the honest rho-dependence curve (Ab-8c vs BAGAN-GP).
    "G0_signs_extreme": dict(dataset="gtsrb", classes=[2, 1, 5], rho=0.025, majority_cap=800),
    "G1b_signs":        dict(dataset="gtsrb", classes=[2, 1, 5], rho=0.075, majority_cap=800),
    "G2b_signs":        dict(dataset="gtsrb", classes=[2, 1, 5], rho=0.15, majority_cap=800),
    # Flowers (BAGAN paper 5-class) SEVERE imbalance: dandelion majority; daisy/
    # roses/sunflowers/tulips minorities.  Natural photos = the paper's hard case.
    "F1_flowers_severe": dict(dataset="flowers", classes=[1, 0, 2, 3, 4], rho=0.05, majority_cap=700),
    "F0_flowers_extreme": dict(dataset="flowers", classes=[1, 0, 2, 3, 4], rho=0.025, majority_cap=700),
    # STL-10 confusable animal trio cat(majority)/dog/deer; natural photos, CIFAR-
    # like.  rho-sweep 0.05 / 0.10.
    "S1_stl_severe": dict(dataset="stl10", classes=[3, 5, 4], rho=0.05, majority_cap=400),
    "S2_stl_mod":    dict(dataset="stl10", classes=[3, 5, 4], rho=0.10, majority_cap=400),
    # EMNIST-Letters confusable thin-stroke trio i(majority)/l/j (1-indexed 9/12/10).
    # Grayscale + confusable + weak baseline + abundant data = the Fashion-win
    # profile.  rho-sweep 0.05 (severe) / 0.10 (moderate), majority cap 800.
    "E1_letters_severe": dict(dataset="emnist_letters", classes=[9, 12, 10], rho=0.05, majority_cap=800),
    "E2_letters_mod":    dict(dataset="emnist_letters", classes=[9, 12, 10], rho=0.10, majority_cap=800),
    # Extreme-scarcity point for the severity-crossover test (Flowers thesis):
    # at rho=0.025 (800:20:20) pixel-interpolation resamplers degenerate; does
    # WGAN-GP's edge narrow vs Ab-8c's AE + latent-SMOTE sample efficiency?
    "E0_letters_extreme": dict(dataset="emnist_letters", classes=[9, 12, 10], rho=0.025, majority_cap=800),
    # OCTMNIST GRAYSCALE retinal-OCT pathology trio CNV(majority)/DME/drusen
    # (drop 'normal').  Texture-defined confusable pathologies + large real pool
    # -> grayscale-medical win-profile probe.  Synthetic imbalance, rho sweep.
    "OCT1_pathology_severe": dict(dataset="octmnist", classes=[0, 1, 2], rho=0.05, majority_cap=800),
    "OCT2_pathology_mod":    dict(dataset="octmnist", classes=[0, 1, 2], rho=0.10, majority_cap=800),
    # Wider rho-curve points (0.15/0.20/0.30) for the OCT severity-dependence
    # sweep -> more minority samples (less noisy) but stronger baseline.
    "OCT3_pathology_015": dict(dataset="octmnist", classes=[0, 1, 2], rho=0.15, majority_cap=800),
    "OCT4_pathology_020": dict(dataset="octmnist", classes=[0, 1, 2], rho=0.20, majority_cap=800),
    "OCT5_pathology_030": dict(dataset="octmnist", classes=[0, 1, 2], rho=0.30, majority_cap=800),
    # KMNIST data-driven confusable hiragana trio: majority 6=ma (central),
    # minorities 1=ki, 2=su (most-confused trio from a balanced confusion
    # matrix).  Grayscale, harder-than-MNIST baseline, abundant pool.  rho sweep.
    "K1_kana_severe": dict(dataset="kmnist", classes=[6, 1, 2], rho=0.05, majority_cap=800),
    "K2_kana_mod":    dict(dataset="kmnist", classes=[6, 1, 2], rho=0.10, majority_cap=800),
    "K3_kana_mild":   dict(dataset="kmnist", classes=[6, 1, 2], rho=0.15, majority_cap=800),

    # ── rho-dependence sweep (3 datasets x 5 rho) for the operating-regime
    # analysis in [GAAL_Case study_4J.ipynb].  Class trio, majority_cap,
    # classifier backbone and synth_ratio are held FIXED per dataset; rho is the
    # only varied factor, so the minority count is the sole moving part.
    # The anchor rho of each dataset is the existing 4J headline regime
    # (R1_confusable / C2_cifar_paper / S2_stl_mod) and is NOT duplicated here.
    # Fashion-MNIST [2,4,6] Pullover/Coat/Shirt, majority_cap 800 (anchor rho=0.05).
    "R1_rho0025": dict(classes=[2, 4, 6], rho=0.025, majority_cap=800),          # 800:20:20
    "R1_rho0100": dict(classes=[2, 4, 6], rho=0.100, majority_cap=800),          # 800:80:80
    "R1_rho0150": dict(classes=[2, 4, 6], rho=0.150, majority_cap=800),          # 800:120:120
    "R1_rho0200": dict(classes=[2, 4, 6], rho=0.200, majority_cap=800),          # 800:160:160
    # CIFAR-10 [0,1,8] airplane/automobile/ship, majority_cap 1000 (anchor rho=0.30).
    "C2_rho0100": dict(dataset="cifar10", classes=[0, 1, 8], rho=0.100, majority_cap=1000),
    "C2_rho0150": dict(dataset="cifar10", classes=[0, 1, 8], rho=0.150, majority_cap=1000),
    "C2_rho0200": dict(dataset="cifar10", classes=[0, 1, 8], rho=0.200, majority_cap=1000),
    "C2_rho0500": dict(dataset="cifar10", classes=[0, 1, 8], rho=0.500, majority_cap=1000),
    # STL-10 [3,5,4] cat/dog/deer, majority_cap 400 (anchor rho=0.10).
    "S2_rho0050": dict(dataset="stl10", classes=[3, 5, 4], rho=0.050, majority_cap=400),
    "S2_rho0150": dict(dataset="stl10", classes=[3, 5, 4], rho=0.150, majority_cap=400),
    "S2_rho0200": dict(dataset="stl10", classes=[3, 5, 4], rho=0.200, majority_cap=400),
    "S2_rho0300": dict(dataset="stl10", classes=[3, 5, 4], rho=0.300, majority_cap=400),
}

# ── rho-dependence grid metadata (consumed by _cs3j_rho3x3.py and the 4J
# notebook).  ``synth_ratio`` is the dataset's 4J headline value, held fixed
# across rho so the augmentation budget is never re-tuned per operating point.
# ``anchor_csv`` points at the already-computed 10-seed confirmation for the
# anchor rho, so that cell is reused instead of recomputed.
RHO3X3: Dict[str, dict] = {
    "fashion_mnist": dict(
        label="Fashion-MNIST", synth_ratio=4, majority_cap=800,
        anchor=("R1_confusable", 0.05, "saved_casestudy/cs3j_fashion_R1_reduced8.csv"),
        sweep=[("R1_rho0025", 0.025), ("R1_rho0100", 0.100),
               ("R1_rho0150", 0.150), ("R1_rho0200", 0.200)],
    ),
    "cifar10": dict(
        label="CIFAR-10", synth_ratio=2, majority_cap=1000,
        anchor=("C2_cifar_paper", 0.30, "saved_casestudy/cs3j_cifar_C2_reduced8.csv"),
        sweep=[("C2_rho0100", 0.100), ("C2_rho0150", 0.150),
               ("C2_rho0200", 0.200), ("C2_rho0500", 0.500)],
    ),
    "stl10": dict(
        label="STL-10", synth_ratio=3, majority_cap=400,
        anchor=("S2_stl_mod", 0.10, "saved_casestudy/cs3j_stl10_S2_reduced8.csv"),
        sweep=[("S2_rho0050", 0.050), ("S2_rho0150", 0.150),
               ("S2_rho0200", 0.200), ("S2_rho0300", 0.300)],
    ),
}

# Regime id -> (dataset_key, rho) for every cell of the rho grid (anchor included).
RHO3X3_REGIME_INFO: Dict[str, tuple] = {
    rid: (ds, rho)
    for ds, meta in RHO3X3.items()
    for rid, rho in [(meta["anchor"][0], meta["anchor"][1]), *meta["sweep"]]
}

# The reduced 8-method field used by the 4J headline cut (m00/m01/m03/m09 dropped).
REDUCED_METHODS: List[str] = ["m11_imbalanced_naive", "m02_bsmote", "m04_dcgan",
                              "m05_cgan", "m06_acgan", "m07_wgangp", "m08_bagan",
                              "m10_ab8c"]

# Ab-8c + current strongest rivals (cheap screening set).
SCREEN_METHODS: List[str] = ["m10_ab8c", "m01_smote", "m07_wgangp", "m05_cgan"]
ALL_METHODS: List[str] = list(cs.CaseStudyConfig().method_order)

# Synthetic:real ratio sweep to map the quality-vs-quantity curve.
RATIOS = (2, 3, 4, "full")

PRIMARY = "macro_f2"
SECONDARY = ["gmean", "macro_recall", "macro_specificity"]


def quick_config(base: Optional[cs.CaseStudyConfig] = None,
                 quick: bool = True) -> cs.CaseStudyConfig:
    """A CS3J config; reduced budget when ``quick`` (for screening)."""
    base = base or cs.CaseStudyConfig()
    if not quick:
        return dataclasses.replace(base, date_suffix="CS3J")
    return dataclasses.replace(
        base, date_suffix="CS3J",
        gan_epochs=80, clf_max_epochs=60, clf_patience=12,
        n_folds=2, max_al_cycles=8, al_patience=4,
    )


def _env_info() -> dict:
    """Environment fingerprint for reproducibility (versions, GPU, git hash)."""
    import platform
    import subprocess
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        info["torch"] = torch.__version__
        info["numpy"] = np.__version__
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover
        info["version_err"] = str(exc)
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        info["git_commit"] = "nogit"
    return info


def write_manifest(out_path: str, config, regimes: dict, methods, ratios, seeds) -> str:
    """Write a JSON reproducibility sidecar next to ``out_path`` (journal-grade).

    Captures the exact regime defs, method list, ratios, seeds, full config, and
    environment fingerprint so the CSV can be regenerated identically later.
    """
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "out_csv": out_path,
        "seeds": list(seeds),
        "ratios": [str(r) for r in ratios],
        "methods": list(methods),
        "regimes": regimes,
        "config": dataclasses.asdict(config),
        "env": _env_info(),
    }
    mpath = out_path.rsplit(".", 1)[0] + ".manifest.json"
    import os
    os.makedirs(os.path.dirname(mpath) or ".", exist_ok=True)
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return mpath


def regime_spec(base_spec: cs.DatasetSpec, regime_id: str, regime: dict) -> cs.DatasetSpec:
    """Build a DatasetSpec variant for a regime (distinct name -> own cache).

    ``base_spec`` must be the base spec for ``regime['dataset']`` (or fashion_mnist).
    """
    ds = regime.get("dataset", base_spec.name)
    if "classes" not in regime:
        return base_spec  # native dataset used as-is (e.g. cells)
    names = _NAME_MAPS.get(ds, {})
    classes = regime["classes"]
    kwargs = dict(
        name=f"{ds}_{regime_id}",
        selected_classes=list(classes),
        class_names=[names.get(c, str(c)) for c in classes],
        imbalance_ratio=float(regime["rho"]),
    )
    if "majority_cap" in regime:
        kwargs["majority_cap"] = int(regime["majority_cap"])
    if "classifier" in regime:
        kwargs["classifier"] = regime["classifier"]
    if "native_imbalance" in regime:
        kwargs["native_imbalance"] = bool(regime["native_imbalance"])
    return dataclasses.replace(base_spec, **kwargs)


def run_one(method_key: str, spec, pool, test, config, device: str,
            synth_ratio, seed: int, overrides: Optional[dict] = None) -> Dict[str, float]:
    """Fair held-out-test evaluation for one method (mirrors notebook §F run_test).

    Fit/synthesize on the retrain-train portion only; early-stop on a stratified
    hold-out; score ONCE on the held-out test set. ``synth_ratio`` controls the
    fair augmentation target for every method. ``overrides`` sets attributes on
    the built method (used to tune Ab-8c, e.g. {"min_confidence": 0.7}).
    """
    X, y = pool["images"], pool["labels"]
    Xte, yte = test["images"], test["labels"]
    K = len(pool["classes"])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=config.random_seed)
    tr, ho = next(sss.split(X.numpy(), y.numpy()))
    Xtr, ytr = X[tr], y[tr]
    Xho, yho = X[ho], y[ho]

    set_seed(seed)
    method = build_method(method_key, config, spec, device, seed=seed)
    if overrides:
        for k, v in overrides.items():
            setattr(method, k, v)
    method.fit(Xtr, ytr)
    targets = balance_targets_ratio(ytr, synth_ratio)
    X_syn, y_syn = method.synthesize(targets)
    if X_syn.numel() > 0:
        X_aug = torch.cat([Xtr, X_syn], dim=0)
        y_aug = torch.cat([ytr, y_syn], dim=0)
    else:
        X_aug, y_aug = Xtr, ytr

    set_seed(config.random_seed)
    clf = build_classifier(spec, K, config)
    # Naive baseline (m11) model-selects on val ACCURACY (majority-biased ->
    # weak macro-F2); every other method uses config.selection_metric (macro-F2).
    train_classifier(clf, X_aug, y_aug, Xho, yho, config, device,
                     num_classes=K, seed=config.random_seed,
                     use_class_weights=config.uses_class_weights(method_key),
                     monitor=config.selection_metric_for(method_key))
    preds, probs = evaluate_classifier(clf, Xte, yte, device, batch_size=config.batch_size)
    m = compute_metrics(yte.numpy(), preds, probs, K)
    return {"n_synth": int(X_syn.shape[0]), **{k: float(m[k]) for k in [PRIMARY, *SECONDARY]}}


def screen(regimes: Optional[Dict[str, dict]] = None,
           methods: Optional[List[str]] = None,
           ratios=RATIOS,
           seeds=(42,),
           quick: bool = True,
           out_path: Optional[str] = None, 
           verbose: bool = True) -> pd.DataFrame:
    """Run methods x regimes x synth_ratios x seeds on fashion_mnist; return a DataFrame.

    Robust: each run is wrapped in try/except (a single failure never aborts the
    sweep) and every completed row is appended to ``out_path`` immediately, so an
    interrupted multi-hour run keeps all progress. Per (regime, ratio), prints the
    MEAN-over-seeds ranking and where Ab-8c lands (rank + gap to best).
    """
    regimes = regimes or REGIMES
    methods = methods or SCREEN_METHODS
    seeds = tuple(seeds)
    config = quick_config(quick=quick)
    device = config.resolve_device()
    tag = "quick" if quick else "full"
    out_path = out_path or f"saved_casestudy/cs3j_{tag}_{int(time.time())}.csv"
    write_manifest(out_path, config, regimes, methods, list(ratios), list(seeds))
    rows: List[dict] = []
    t_start = time.time()
    n_total = len(regimes) * len(list(ratios)) * len(methods) * len(seeds)
    n_done = 0

    for rid, regime in regimes.items():
        base_spec = config.datasets[regime.get("dataset", "fashion_mnist")]
        spec = regime_spec(base_spec, rid, regime)
        set_seed(config.random_seed)
        data = build_dataset(spec, config, use_cache=True, verbose=False)
        pool, test = data["trainpool"], data["test"]
        if verbose:
            print(f"\n=== {rid}: classes={regime.get('classes', spec.selected_classes)} "
                  f"rho={regime.get('rho', 'native')} "
                  f"| pool={len(pool['labels'])} test={len(test['labels'])} "
                  f"| device={device} budget={tag} seeds={seeds} ===", flush=True)
        for ratio in ratios:
            for mkey in methods:
                for sd in seeds:
                    t0 = time.time()
                    try:
                        res = run_one(mkey, spec, pool, test, config, device, ratio, sd)
                        err = ""
                    except Exception as exc:  # keep the sweep alive
                        res = {"n_synth": -1, PRIMARY: float("nan"),
                               **{k: float("nan") for k in SECONDARY}}
                        err = f"{type(exc).__name__}: {exc}"
                    dt = time.time() - t0
                    n_done += 1
                    row = {"regime": rid, "ratio": str(ratio), "seed": sd, "method": mkey,
                           "label": cs.METHOD_LABELS[mkey], **res,
                           "runtime_s": round(dt, 1), "error": err}
                    rows.append(row)
                    _append_row(out_path, row)  # persist progress immediately
                    if verbose:
                        if err:
                            print(f"  [{rid}|r={ratio}|s{sd}|{mkey}] FAILED {err} ({dt:.0f}s) "
                                  f"[{n_done}/{n_total}]", flush=True)
                        else:
                            el = time.time() - t_start
                            eta = el / n_done * (n_total - n_done)
                            print(f"  [{rid}|r={ratio}|s{sd}|{mkey}] {PRIMARY}={res[PRIMARY]:.4f} "
                                  f"gmean={res['gmean']:.4f} n_synth={res['n_synth']} "
                                  f"({dt:.0f}s) [{n_done}/{n_total} eta~{eta/60:.0f}m]",
                                  flush=True)
            _report_block(rows, rid, str(ratio), verbose)

    df = pd.DataFrame(rows)
    if verbose:
        print(f"\nsaved sweep -> {out_path} | total {(time.time()-t_start)/60:.1f} min", flush=True)
    return df


def _append_row(path: str, row: dict) -> None:
    """Append one result row to a CSV (write header once); non-fatal on lock."""
    import csv
    import os
    try:
        exists = os.path.exists(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)
    except Exception as exc:
        print(f"[warn] could not append row: {exc}", flush=True)


def _report_block(rows: List[dict], rid: str, ratio: str, verbose: bool) -> None:
    """Print Ab-8c's rank + gap to best (MEAN over seeds) for one (regime, ratio) block."""
    import collections
    block = [r for r in rows if r["regime"] == rid and r["ratio"] == ratio and not r["error"]]
    if not block:
        return
    agg = collections.defaultdict(list)
    for r in block:
        agg[r["method"]].append(r[PRIMARY])
    means = {m: sum(v) / len(v) for m, v in agg.items()}
    order = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    if "m10_ab8c" not in means:
        return
    rank = [m for m, _ in order].index("m10_ab8c") + 1
    best_m, best_v = order[0]
    gap = best_v - means["m10_ab8c"]
    nseed = len(agg["m10_ab8c"])
    verdict = "WIN" if rank == 1 else ("close" if gap <= 0.01 else "behind")
    if verbose:
        print(f"  -> Ab-8c rank {rank}/{len(order)} on mean {PRIMARY} over {nseed} seed(s) "
              f"(best={cs.METHOD_LABELS[best_m]} {best_v:.4f}, gap={gap:+.4f}) [{verdict}]",
              flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="CS3J fashion_mnist ratio sweep")
    ap.add_argument("--quick", action="store_true", help="reduced budget (integrity check)")
    ap.add_argument("--methods", default="all", choices=["all", "screen"],
                    help="'all' = 11 methods (default), 'screen' = Ab-8c + 3 rivals")
    ap.add_argument("--regimes", default="all", help="comma list of regime ids, or 'all'")
    ap.add_argument("--ratios", default="", help="comma list e.g. 2,3,4,full (default all)")
    ap.add_argument("--seeds", default="42", help="comma list e.g. 42,43,44,45,46")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()

    _methods = ALL_METHODS if args.methods == "all" else SCREEN_METHODS
    _regimes = REGIMES if args.regimes == "all" else {k: REGIMES[k] for k in args.regimes.split(",")}
    if args.ratios:
        _ratios = tuple(int(x) if x.strip().isdigit() else x.strip() for x in args.ratios.split(","))
    else:
        _ratios = RATIOS
    _seeds = tuple(int(x) for x in args.seeds.split(","))
    screen(regimes=_regimes, methods=_methods, ratios=_ratios, seeds=_seeds,
           quick=args.quick, out_path=args.out)
