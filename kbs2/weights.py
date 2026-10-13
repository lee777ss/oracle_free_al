"""kbs2.weights — power-law class weights for the downstream classifier (E2t, author request 11 Oct 2026).

The submitted trainer (``casestudy.classifiers.trainer._class_weights``) uses inverse-frequency weights
``w_c = N / (K n_c)``. This module provides the family ``w_c ∝ n_c^(-alpha)`` (alpha = 1: inverse
frequency, alpha = 0.5: inverse square root), normalised to mean 1 — PyTorch's weighted mean CE divides
by the sum of the batch weights, so only the ratios matter.

``casestudy/`` must stay byte-identical to the submission snapshot (``kbs2.provenance``), so the rule is
not edited there: ``class_weight_rule(alpha)`` swaps the trainer's module-level ``_class_weights`` for the
duration of one ``train_classifier`` call and always restores it. ``alpha = 1`` (or ``None``) leaves the
original function in place, so every existing result is reproduced bit for bit.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Optional

import torch


def power_law_weights(labels: torch.Tensor, num_classes: int, device: str, alpha: float) -> torch.Tensor:
    """``w_c ∝ n_c^(-alpha)`` with mean 1 (counts clamped at 1, as in the submitted rule)."""
    counts = torch.clamp(torch.bincount(labels, minlength=num_classes).float(), min=1.0)
    w = counts.pow(-float(alpha))
    return (w * num_classes / w.sum()).to(device)


def rule_name(alpha: Optional[float]) -> str:
    return "inverse_frequency" if alpha is None or float(alpha) == 1.0 else f"power_law_alpha_{alpha:g}"


@contextlib.contextmanager
def class_weight_rule(alpha: Optional[float]) -> Iterator[None]:
    from casestudy.classifiers import trainer

    if alpha is None or float(alpha) == 1.0:
        yield
        return
    original = trainer._class_weights

    def _patched(labels, num_classes, device):
        return power_law_weights(labels, num_classes, device, alpha)

    trainer._class_weights = _patched
    try:
        yield
    finally:
        trainer._class_weights = original
