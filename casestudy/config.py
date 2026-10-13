"""casestudy.config — configuration for the open-source data case studies (CS1J).

Single source of truth for the four case-study datasets, their class-subset
imbalance recipes, the fixed method order, the metric list, and global run
settings. Mirrors the role of ``ablation/config.py`` in GAAL_Ablation_130J,
but is fully self-contained: the ``casestudy`` package never imports
``ablation`` or ``bagan``.

See plan §3 (datasets/imbalance), §4.1 (package layout), §13 (locked
configuration decisions) in ``110_Plan for Open source data case studies.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

print("casestudy/config.py loaded")


# ── Fixed method order (plan §5) ─────────────────────────────────────────
# Keys map 1:1 to casestudy/methods/m00..m10.  Only Ab-8c uses active learning.
METHOD_ORDER: List[str] = [
    "m00_imbalanced",  # Ab-1 baseline (no augmentation)
    "m01_smote",
    "m02_bsmote",
    "m03_adasyn",
    "m04_dcgan",
    "m05_cgan",
    "m06_acgan",
    "m07_wgangp",
    "m08_bagan",
    "m09_bagan_gp",
    "m10_ab8c",        # proposed (BADGE + 3-player co-training)
    "m11_imbalanced_naive",  # 2nd baseline (no early-stop / no checkpoint restore)
]

# Human-readable labels for reporting tables (plan §5 / §9).
METHOD_LABELS: Dict[str, str] = {
    "m00_imbalanced": "Imbalanced (Ab-1)",
    "m01_smote": "SMOTE",
    "m02_bsmote": "Borderline-SMOTE",
    "m03_adasyn": "ADASYN",
    "m04_dcgan": "DCGAN",
    "m05_cgan": "cGAN",
    "m06_acgan": "ACGAN",
    "m07_wgangp": "WGAN-GP",
    "m08_bagan": "BAGAN",
    "m09_bagan_gp": "BAGAN-GP",
    "m10_ab8c": "Ab-8c (proposed)",
    "m11_imbalanced_naive": "Imbalanced-naive (acc-sel)",
}

# Generative methods (per-class FID is computed only for these; plan §7/§9).
GENERATIVE_METHODS: List[str] = [
    "m04_dcgan", "m05_cgan", "m06_acgan", "m07_wgangp",
    "m08_bagan", "m09_bagan_gp", "m10_ab8c",
]

# ── Metric list (plan §7) ────────────────────────────────────────────────
METRICS: List[str] = [
    "macro_recall",
    "gmean",
    "macro_f2",
    "macro_specificity",
    "fid",  # per-class FID (generative methods only)
]


@dataclass
class DatasetSpec:
    """Per-dataset imbalance recipe (plan §3.2 / §13).

    Attributes:
        name: Short identifier used in cache/CSV filenames (e.g. "fashion_mnist").
        loader_key: Which raw loader to use ("fmnist"/"cifar10"/"cifar100"/"bloodmnist").
        selected_classes: Original (possibly non-contiguous) class indices;
            element 0 is always treated as the majority class and every later
            element is treated as a minority class.
        class_names: Human-readable names aligned to ``selected_classes``.
        majority_cap: Cap on majority-class count for the training pool.
        imbalance_ratio: ρ = minority / majority for the synthetic construction.
        native_imbalance: If True (BloodMNIST), use the dataset's native
            long-tailed counts instead of the synthetic ρ recipe.
        channels: Number of image channels after loading (1 grayscale, 3 color).
        classifier: Shared backbone key for this dataset ("cnn"/"resnet").
    """

    name: str
    loader_key: str
    selected_classes: List[int]
    class_names: List[str]
    channels: int
    classifier: str
    majority_cap: Optional[int] = 800
    imbalance_ratio: float = 0.10
    native_imbalance: bool = False
    balanced_test: bool = False

    def __post_init__(self) -> None:
        if self.native_imbalance:
            return
        if len(self.selected_classes) < 2:
            raise ValueError(
                f"{self.name}: selected_classes must contain at least 2 classes, "
                f"got {self.selected_classes!r}"
            )
        if len(self.class_names) != len(self.selected_classes):
            raise ValueError(
                f"{self.name}: class_names length ({len(self.class_names)}) must match "
                f"selected_classes length ({len(self.selected_classes)})"
            )


def _default_datasets() -> Dict[str, DatasetSpec]:
    """The four locked case-study datasets (plan §3.2 table + §13)."""
    return {
        # 1 · Fashion-MNIST — first class is majority; all later classes are minorities.
        #   Target = paper Table I/III ablation subset (T-shirt majority).
        #   ρ=0.10 → 800:80 PER minority class (the paper's "80" is per-class,
        #   NOT a shared total).  ρ=0.025 would give only 20/class — ~4× too few
        #   real minority samples, which starves the GANs and pushes every
        #   method below 0.6 (the symptom seen in CS1J/CS2J).
        #   3-class current target = [0, 2, 3] (T-shirt, Pullover, Dress).
        #   5-class paper open-source subset = [0, 2, 3, 4, 6] (+ Coat, Shirt).
        "fashion_mnist": DatasetSpec(
            name="fashion_mnist",
            loader_key="fmnist",
            selected_classes=[0, 2, 3],
            class_names=["T-shirt/top", "Pullover", "Dress"],
            channels=1,
            classifier="cnn",
            majority_cap=800,
            imbalance_ratio=0.05,  # 0.05 → 800:40:40 (per-class, matches paper)
        ),
        # 2 · CIFAR-10 — ρ=0.20 → 800:160:160 (airplane → bird, cat).
        "cifar10": DatasetSpec(
            name="cifar10",
            loader_key="cifar10",
            selected_classes=[0, 2, 3],
            class_names=["airplane", "bird", "cat"],
            channels=3,
            classifier="resnet",
            majority_cap=800,
            imbalance_ratio=0.20,
        ),
        # 3 · CIFAR-100 — ρ=0.30 → 400:120:120 (bus → pickup_truck, streetcar).
        "cifar100": DatasetSpec(
            name="cifar100",
            loader_key="cifar100",
            selected_classes=[13, 58, 81],
            class_names=["bus", "pickup_truck", "streetcar"],
            channels=3,
            classifier="resnet",
            majority_cap=400,
            imbalance_ratio=0.30,
        ),
        # 4 · BloodMNIST — native imbalance, used as-is (plan §13 item 1).
        "bloodmnist": DatasetSpec(
            name="bloodmnist",
            loader_key="bloodmnist",
            selected_classes=[],  # resolved at load time (3 native classes)
            class_names=[],       # resolved at load time
            channels=3,
            classifier="cnn",
            majority_cap=None,
            imbalance_ratio=0.0,
            native_imbalance=True,
        ),
        # 5 · OrganAMNIST — GRAYSCALE abdominal-CT organ slices (MedMNIST v2).
        #   Grayscale (Ab-8c's reconstruction strength) + medical narrative +
        #   abundant data (~34k train) so GANs are never starved.  Synthetic
        #   imbalance over a confusable soft-tissue trio (liver majority;
        #   spleen + pancreas minorities).  cs3j regimes override rho/classes.
        "organamnist": DatasetSpec(
            name="organamnist",
            loader_key="organamnist",
            selected_classes=[6, 10, 9],
            class_names=["liver", "spleen", "pancreas"],
            channels=1,
            classifier="cnn",
            majority_cap=800,
            imbalance_ratio=0.10,
        ),
        # 6 · Cells (Huang & Jafari BAGAN-GP benchmark) — BBBC041 malaria crops.
        #   4 classes, native severe imbalance (red blood cell majority; ring /
        #   schizont / trophozoite parasite stages rare & confusable).  This is
        #   the dataset BAGAN-GP (Ab-8c's backbone) was designed to win on.
        #   Built by _prep_cells.py into data/cells_processed/*.npy.
        "cells": DatasetSpec(
            name="cells",
            loader_key="cells",
            selected_classes=[0, 1, 2, 3],
            class_names=["red blood cell", "ring", "schizont", "trophozoite"],
            channels=3,
            classifier="cnn",
            majority_cap=None,
            imbalance_ratio=0.0,
            native_imbalance=True,
        ),
        # 7 · GTSRB — traffic signs (BAGAN paper's win, Mariani et al. 2018).
        #   Confusable speed-limit subset (50 / 30 / 80 km/h): near-identical
        #   circles differing only by the printed digit -> pixel-interpolation
        #   resamplers (SMOTE/ADASYN) blur the digit, while a structure-learning
        #   generator can keep it crisp.  Synthetic imbalance over the trio.
        #   Built by _prep_gtsrb.py into data/gtsrb_processed/*.npy.
        "gtsrb": DatasetSpec(
            name="gtsrb",
            loader_key="gtsrb",
            selected_classes=[2, 1, 5],
            class_names=["50km/h", "30km/h", "80km/h"],
            channels=3,
            classifier="cnn",
            majority_cap=800,
            imbalance_ratio=0.10,
        ),
        # 8 · Flowers (BAGAN paper's 5-class Google flower_photos).
        #   Natural photos (the paper's HARD case: mirror-aug beats BAGAN there).
        #   Dandelion (most populous) is the majority; severe imbalance over the
        #   4 minorities.  Built by _prep_flowers.py into data/flowers_processed.
        "flowers": DatasetSpec(
            name="flowers",
            loader_key="flowers",
            selected_classes=[1, 0, 2, 3, 4],
            class_names=["dandelion", "daisy", "roses", "sunflowers", "tulips"],
            channels=3,
            classifier="cnn",
            majority_cap=700,
            imbalance_ratio=0.05,
            balanced_test=True,
        ),
        # 9 · STL-10 — natural images (CIFAR-like, 96px native -> 64).  Confusable
        #   animal trio cat(majority)/dog/deer; severe imbalance; balanced test.
        #   Built by _prep_stl10.py into data/stl10_processed/*.npy.
        "stl10": DatasetSpec(
            name="stl10",
            loader_key="stl10",
            selected_classes=[3, 5, 4],
            class_names=["cat", "dog", "deer"],
            channels=3,
            classifier="cnn",
            majority_cap=400,
            imbalance_ratio=0.05,
            balanced_test=True,
        ),
        # 10 · EMNIST-Letters — GRAYSCALE handwritten letters (the closest analog
        #   to the Fashion-MNIST win: grayscale + genuinely confusable classes +
        #   weak baseline + abundant data so GANs are never starved).  Confusable
        #   thin vertical-stroke trio i (majority) / l / j (i-vs-l is the canonical
        #   EMNIST hard pair; case-merged 'i' contains 'I' ~ 'l').  Synthetic
        #   imbalance; cs3j regimes override rho.  Loaded directly via torchvision.
        "emnist_letters": DatasetSpec(
            name="emnist_letters",
            loader_key="emnist_letters",
            selected_classes=[9, 12, 10],  # 1-indexed: i, l, j
            class_names=["i", "l", "j"],
            channels=1,
            classifier="cnn",
            majority_cap=800,
            imbalance_ratio=0.05,
        ),
        # 11 · OCTMNIST — GRAYSCALE retinal OCT (MedMNIST v2).  Grayscale medical
        #   setting with genuinely confusable, texture-defined pathologies.  We
        #   use the pathology trio CNV (majority) / DME / drusen and DROP the
        #   comparatively separable 'normal' class to keep the confusability high.
        #   Large train pool (>50k across these 3) → GANs never starved even at
        #   rho 0.05.  cs3j regimes override rho.
        "octmnist": DatasetSpec(
            name="octmnist",
            loader_key="octmnist",
            selected_classes=[0, 1, 2],  # CNV (majority), DME, drusen
            class_names=["CNV", "DME", "drusen"],
            channels=1,
            classifier="cnn",
            majority_cap=800,
            imbalance_ratio=0.05,
        ),
        # 12 · KMNIST (Kuzushiji-MNIST) — GRAYSCALE cursive hiragana, harder than
        #   MNIST (weak baseline).  Data-driven confusable trio from a balanced
        #   10-class confusion matrix (_kmnist_confusion.py): majority 6 = ま (ma,
        #   central) confused with minorities 1 = き (ki) and 2 = す (su)
        #   (2<->6=18, 1<->6=9 confusions).  Abundant data (6000/class) -> GANs
        #   never starved.  cs3j regimes override rho.
        "kmnist": DatasetSpec(
            name="kmnist",
            loader_key="kmnist",
            selected_classes=[6, 1, 2],  # ma (majority), ki, su
            class_names=["ma", "ki", "su"],
            channels=1,
            classifier="cnn",
            majority_cap=800,
            imbalance_ratio=0.05,
        ),
    }


@dataclass
class CaseStudyConfig:
    """Global configuration for the CS1J case studies (plan §4.1 / §8 / §13).

    The notebook's ``SELECTED_DATASETS`` / ``SELECTED_METHODS`` switches
    default to ``dataset_order`` / ``method_order`` (set them to ``None`` to
    run the full grid, or a subset preserving order).
    """

    # Datasets & methods
    datasets: Dict[str, DatasetSpec] = field(default_factory=_default_datasets)
    method_order: List[str] = field(default_factory=lambda: list(METHOD_ORDER))
    metrics: List[str] = field(default_factory=lambda: list(METRICS))

    # Image format — all datasets upscaled to a shared 64×64 backbone family.
    image_size: int = 64
    normalize_range: tuple = (-1.0, 1.0)  # matches generator Tanh output

    # Cross-validation & reproducibility (plan §8 / §13)
    n_folds: int = 5
    random_seed: int = 42

    # Compute budget (plan §8 fairness policy)
    gan_epochs: int = 500          # fixed shared generator budget, no GAN early-stop
    clf_max_epochs: int = 500      # classifier upper bound (early-stop on val macro-F1)
    clf_patience: int = 100        # classifier early-stopping patience
    al_patience: int = 10          # Ab-8c active-learning patience
    max_al_cycles: int = 20        # Ab-8c max active-learning cycles in mixed-fill mode
    strict_badge_fill: bool = True  # If True, Ab-8c fills every deficit via BADGE-selected samples
    batch_size: int = 64
    latent_dim: int = 128

    # Fair augmentation target (applied identically to ALL methods).
    #   "full" -> fill every minority up to the majority count (original CS2J
    #   behavior). A numeric r -> cap each minority at r x its real count
    #   (never below real, never above majority). Lower r reduces the synthetic
    #   :real ratio so on-manifold sample QUALITY matters more than quantity.
    synth_ratio: object = "full"

    # Classifier optimizer (matches 130J "slow soak": Adam, lr 1e-4, wd 1e-4)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout_rate: float = 0.3
    cnn_backbone: str = "small_cnn"   # backbone key for spec.classifier == "cnn"
    resnet_backbone: str = "resnet32"  # backbone key for spec.classifier == "resnet"
    selection_metric: str = "macro_f2"  # early-stop monitor (val), mode='max'

    # Which methods' downstream classifier uses inverse-frequency class-weighted
    # CE.  Augmentation methods balance the pool to majority count first, so the
    # weights collapse to ~uniform and the flag is a no-op for them; the ONLY
    # method materially affected is the imbalanced baseline (m00).  Excluding
    # m00 here makes Ab-1 a faithful "no imbalance handling" control (matches
    # the paper's plain-CE baseline) instead of a silently class-balanced one.
    class_weighted_methods: tuple = (
        "m01_smote", "m02_bsmote", "m03_adasyn", "m04_dcgan", "m05_cgan",
        "m06_acgan", "m07_wgangp", "m08_bagan", "m09_bagan_gp", "m10_ab8c",
    )

    # Second baseline (m11): downstream classifier is model-SELECTED on validation
    # ACCURACY instead of macro-F2. On imbalanced data the best-accuracy epoch
    # predicts the majority well and ignores minorities -> low macro-F2, i.e. a
    # genuinely weak, majority-biased "naive practitioner" control vs the
    # recall-aware macro-F2-selected baseline (m00). Both use plain CE.
    accuracy_baseline_methods: tuple = ("m11_imbalanced_naive",)

    # Paths & artifacts (plan §3.1 / §8 — experiment suffix "CS2J")
    date_suffix: str = "CS2J"
    data_root: str = "./data"
    cache_dir: str = "./data/casestudy_cache"
    saved_dir: str = "saved_casestudy"
    figures_dir: str = "reporting/figures"

    device: str = "auto"  # 'auto' | 'cuda' | 'cpu'

    @property
    def dataset_order(self) -> List[str]:
        """Canonical dataset iteration order (insertion order of ``datasets``)."""
        return list(self.datasets.keys())

    def cache_path(self, dataset_name: str, split: str) -> str:
        """Cache filename for a (dataset, split) pair, e.g.
        ``data/casestudy_cache/fashion_mnist_trainpool_CS2J.npz``.

        Args:
            dataset_name: Key into ``self.datasets``.
            split: "trainpool" (from the train split) or "test" (from the test split).
        """
        return f"{self.cache_dir}/{dataset_name}_{split}_{self.date_suffix}.npz"

    def resolve_device(self) -> str:
        """Resolve the 'auto' device to 'cuda' if available else 'cpu'."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def uses_class_weights(self, method_key: str) -> bool:
        """Whether ``method_key``'s downstream classifier uses class-weighted CE.

        Returns False for the imbalanced baseline (``m00_imbalanced``) so it is
        trained with plain CE — a faithful "no imbalance handling" control.
        For every balanced method the weights are ~uniform, so this flag has no
        practical effect on them.
        """
        return method_key in self.class_weighted_methods

    def selection_metric_for(self, method_key: str) -> str:
        """Validation monitor for early-stop/checkpoint. The naive baseline (m11)
        selects on ACCURACY (majority-biased -> weak macro-F2); every other method
        uses ``selection_metric`` (default macro-F2).
        """
        if method_key in self.accuracy_baseline_methods:
            return "accuracy"
        return self.selection_metric

