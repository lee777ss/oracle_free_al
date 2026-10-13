"""casestudy.classifiers.trainer — shared training/eval loop (plan §6 / §8).

Standalone (no import of ``ablation``).  Provides:
- ``EarlyStopping``: monitor a validation metric (mode='max'), restore best weights.
- ``build_classifier``: factory mapping a ``DatasetSpec`` to a backbone.
- ``train_classifier``: class-weighted CE, Adam, early-stop on val macro-F2
  (monitor set by ``config.selection_metric``; default macro-F2).
- ``evaluate_classifier``: forward pass → (preds, probs) for the eval layer.

The same loop trains every method's classifier on the training pool (+ its
synthetics) and is reused for both CV (train/val partition) and the final
held-out test retrain (plan §8).  The model is re-initialized from the same
seed for every method to keep comparisons fair.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..reproducibility import set_seed, seed_worker, make_loader_generator
from .cnn import SmallCNN
from .resnet import resnet18, resnet32

print("casestudy/classifiers/trainer.py loaded")


# ── Early stopping ────────────────────────────────────────────────────────
class EarlyStopping:
    """Early stopping on a monitored validation metric (mode='max' for F1).

    Mirrors the 130J behavior: track the best metric, deep-copy the best
    weights, restore them when patience is exhausted.
    """

    def __init__(self, patience: int = 100, min_delta: float = 0.0,
                 restore_best_weights: bool = True, mode: str = "max"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_metric: Optional[float] = None
        self.best_model = None
        self.best_epoch = 0
        self.counter = 0
        self.improved = False

    def _is_improvement(self, metric: float) -> bool:
        if self.mode == "max":
            return metric > self.best_metric + self.min_delta
        return metric < self.best_metric - self.min_delta

    def __call__(self, model: nn.Module, metric: float, epoch: int = 0) -> bool:
        self.improved = False
        if self.best_metric is None or self._is_improvement(metric):
            self.best_metric = metric
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
            self.improved = True
            return False
        self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_model is not None:
                model.load_state_dict(self.best_model)
            return True
        return False


# ── Metric helper (early-stop monitor only; full battery is in eval/) ─────
def _macro_fbeta(targets: np.ndarray, preds: np.ndarray, num_classes: int,
                 beta: float = 2.0) -> float:
    """Macro-averaged F-beta over ``num_classes`` (no external deps).

    beta=1 -> macro-F1, beta=2 -> macro-F2 (recall-weighted, our headline).
    """
    b2 = beta * beta
    fs = []
    for c in range(num_classes):
        tp = int(np.sum((preds == c) & (targets == c)))
        fp = int(np.sum((preds == c) & (targets != c)))
        fn = int(np.sum((preds != c) & (targets == c)))
        denom = (1 + b2) * tp + b2 * fn + fp
        fs.append(((1 + b2) * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(fs))


def _beta_for(config) -> float:
    """Resolve the early-stop F-beta from ``config.selection_metric``."""
    name = str(getattr(config, "selection_metric", "macro_f2")).lower()
    return 2.0 if name.endswith("f2") else 1.0


# ── Factory ───────────────────────────────────────────────────────────────
def build_classifier(spec, num_classes: int, config) -> nn.Module:
    """Instantiate the shared backbone for a dataset (plan §6).

    ``spec.classifier == 'cnn'`` → small CNN; ``'resnet'`` → ResNet-32/18
    (per ``config.resnet_backbone``).  ``spec.channels`` sets input channels.
    """
    if spec.classifier == "cnn":
        return SmallCNN(input_channels=spec.channels, num_classes=num_classes,
                        dropout_rate=config.dropout_rate)
    if spec.classifier == "resnet":
        builder = {"resnet32": resnet32, "resnet18": resnet18}[config.resnet_backbone]
        return builder(in_channels=spec.channels, num_classes=num_classes)
    raise ValueError(f"Unknown classifier key: {spec.classifier!r}")


def _class_weights(labels: torch.Tensor, num_classes: int, device: str) -> torch.Tensor:
    """Inverse-frequency class weights (macro-F-aware CE; plan §6)."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


# ── Training / evaluation ───────────────────────────────────────────────────
def train_classifier(
    model: nn.Module,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    config,
    device: str,
    num_classes: int = 3,
    seed: Optional[int] = None,
    max_epochs: Optional[int] = None,
    use_class_weights: bool = True,
    use_early_stopping: bool = True,
    monitor: Optional[str] = None,
    verbose: bool = False,
    train_transform=None,
) -> Dict:
    """Train with class-weighted CE + early-stop on the validation macro-F-beta
    monitor selected by ``config.selection_metric`` (default macro-F2).

    Reproducibility contract (plan §8): for fully reproducible *weights*, call
    ``set_seed(seed)`` immediately **before** ``build_classifier`` so the model
    is initialized at a known RNG state; passing the same ``seed`` here then
    makes dropout and DataLoader shuffling deterministic too.

    Args:
        model: A freshly built classifier (will be moved to ``device``).
        train_images/labels: Training tensors (real + synthetic).
        val_images/labels: Validation tensors for early-stopping/selection.
        config: ``CaseStudyConfig`` (lr, weight_decay, batch_size, patience).
        device: 'cuda' or 'cpu'.
        num_classes: Number of classes (3 in the case studies).
        seed: If given, ``set_seed(seed)`` before init/training.
        max_epochs: Override ``config.clf_max_epochs`` (used for quick checks).
        use_class_weights: If True (default), use inverse-frequency class-weighted
            CE. Set False for the imbalanced baseline so it gets no free
            loss-level imbalance correction (no effect on balanced methods).
        verbose: Print per-epoch progress.
        train_transform: Optional callable applied to every training batch on the
            device before the forward pass (on-the-fly augmentation, KBS revision
            method ``m14_stdaug_online``). ``None`` leaves training unchanged.

    Returns:
        ``{"history", "best_macro_f1", "best_epoch", "stopped_epoch"}``.
        ``model`` holds the best weights (restored in-place).
    """
    if seed is not None:
        set_seed(seed)

    model = model.to(device)
    max_epochs = max_epochs if max_epochs is not None else config.clf_max_epochs

    if use_class_weights:
        weights = _class_weights(train_labels, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        # Plain CE — used for the imbalanced baseline (Ab-1) so it does NOT
        # receive free loss-level imbalance correction. Augmentation methods
        # train on balanced data, where inverse-freq weights are ~uniform
        # anyway, so this flag only materially affects the imbalanced baseline.
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)

    g = make_loader_generator(seed if seed is not None else config.random_seed)
    train_loader = DataLoader(
        TensorDataset(train_images, train_labels), batch_size=config.batch_size,
        shuffle=True, generator=g, worker_init_fn=seed_worker,
    )

    stopper = EarlyStopping(patience=config.clf_patience, mode="max") if use_early_stopping else None
    beta = _beta_for(config)
    monitor_name = (monitor or getattr(config, "selection_metric", "macro_f2")).lower()
    monitor_accuracy = monitor_name in ("accuracy", "acc")
    history = {"train_loss": [], "val_score": []}
    stopped_epoch = max_epochs - 1
    val_score = 0.0

    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if train_transform is not None:
                with torch.no_grad():
                    xb = train_transform(xb)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * xb.size(0)
        train_loss = running / len(train_loader.dataset)

        val_preds, _ = evaluate_classifier(model, val_images, val_labels, device,
                                            batch_size=config.batch_size)
        _yv = val_labels.cpu().numpy()
        val_score = (float((val_preds == _yv).mean()) if monitor_accuracy
                     else _macro_fbeta(_yv, val_preds, num_classes, beta))
        history["train_loss"].append(train_loss)
        history["val_score"].append(val_score)

        if verbose:
            print(f"    epoch {epoch:03d}  loss={train_loss:.4f}  "
                  f"val_macroF{int(beta)}={val_score:.4f}")

        # Naive baseline (use_early_stopping=False): no best-checkpoint restore,
        # train the full fixed budget and keep the FINAL weights.
        if stopper is not None and stopper(model, val_score, epoch=epoch):
            stopped_epoch = epoch
            if verbose:
                print(f"    early stop @ epoch {epoch} (best epoch {stopper.best_epoch})")
            break

    return {
        "history": history,
        "best_score": (stopper.best_metric if stopper is not None else val_score),
        "best_epoch": (stopper.best_epoch if stopper is not None else stopped_epoch),
        "stopped_epoch": stopped_epoch,
    }


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward pass → ``(preds, probs)`` numpy arrays.

    The eval layer (Step 4) consumes ``preds`` for macro-recall/G-mean/F2/
    specificity and may use ``probs`` for calibration diagnostics.
    """
    model = model.to(device)
    model.eval()
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)
    all_preds, all_probs = [], []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(probs.argmax(dim=1).cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_probs)
