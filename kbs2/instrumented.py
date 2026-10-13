"""kbs2.instrumented — logging-only subclass of the submitted proposed method (E1).

``InstrumentedAb8c`` inherits ``casestudy.methods.m10_ab8c.Ab8cMethod`` unchanged and
overrides only ``_acquire``: it calls the submitted ``_acquire`` first, then, for the
record, recomputes the internal classifier's softmax on the same candidates.

Why this cannot change the computation:
* the submitted ``_quality_select`` leaves ``C`` in eval mode; the extra forward pass
  is also in eval mode under ``no_grad`` (BatchNorm uses running statistics and is not
  updated; there is no dropout in eval mode), and ``C`` is left in eval mode as before;
* the pass draws no random numbers, so every RNG stream is untouched;
* the returned indices are those of the submitted rule, unmodified.
Whether this holds in practice is exactly what the E1 gate tests (bit-identical
metrics against the submitted per-seed results).

What is recorded per (class, cycle): candidates generated, admission quota n_add,
number passing the gate (p_c >= tau and argmax = c), whether the fallback fired
(fewer than n_add passed), and for every admitted image its p_c and gate status.
Optionally every candidate (uint8) and its p_c, for the label audit and the
near-copy check.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from casestudy.methods.m10_ab8c import Ab8cMethod

# Settings of the submitted runs (class defaults of Ab8cMethod + config); E1 step 0
# asserts them so that the instrumented run cannot silently use another rule.
EXPECTED = {
    "selection_mode": "confident_diverse",
    "seed_mode": "latent_smote",
    "min_confidence": 0.6,
    "realism_filter": False,
    "latent_smote_noise": 0.25,
    "val_frac": 0.2,
}


def to_uint8(x: torch.Tensor) -> np.ndarray:
    return ((x.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()


class InstrumentedAb8c(Ab8cMethod):
    def __init__(self, config, spec, device: str = "cpu", seed: int = 42,
                 save_candidates: bool = False, tau: Optional[float] = None):
        super().__init__(config, spec, device, seed)
        self.save_candidates = save_candidates
        # Decision D13: the gate threshold is a declared hyperparameter of the run; None keeps
        # the submitted class default (0.6).
        self.expected_tau = EXPECTED["min_confidence"] if tau is None else float(tau)
        if tau is not None:
            self.min_confidence = float(tau)
        self.gate_log: List[Dict[str, object]] = []
        self.val_f2_log: List[float] = []
        self.candidate_store: List[Dict[str, object]] = []
        self._cycle_of_class: Dict[int, int] = {}
        self._admitted_meta: Dict[int, List[Dict[str, np.ndarray]]] = {}

    def _val_f2(self, C, Xv, yv, device):
        """Record the internal classifier's diagnostic macro-F2 on S_val (Algorithm 1, line 26).

        The submitted loop calls this once per cycle and, with strict fill on, uses the value
        for nothing; the value itself is computed by the submitted method unchanged.
        """
        v = super()._val_f2(C, Xv, yv, device)
        self.val_f2_log.append(float(v))
        return v

    def check_expected(self) -> Dict[str, object]:
        expected = dict(EXPECTED, min_confidence=self.expected_tau)
        found = {k: getattr(self, k) for k in expected}
        found["strict_badge_fill"] = bool(getattr(self.config, "strict_badge_fill", False))
        bad = {k: (found[k], v) for k, v in expected.items() if found[k] != v}
        if not found["strict_badge_fill"]:
            bad["strict_badge_fill"] = (False, True)
        if bad:
            raise RuntimeError(f"proposed-method settings differ from the submitted runs: {bad}")
        return found

    def _acquire(self, C, candidates, n_add, target_class, Xva=None, yva=None):
        idx = super()._acquire(C, candidates, n_add, target_class, Xva, yva)

        c = int(target_class)
        cycle = self._cycle_of_class.get(c, 0)
        self._cycle_of_class[c] = cycle + 1
        with torch.no_grad():                       # C is already in eval mode here
            probs = torch.softmax(C(candidates.to(self.device)), dim=1).cpu()
        p_c = probs[:, c]
        passed = (probs.argmax(dim=1) == c) & (p_c >= self.min_confidence)
        idx_cpu = idx.cpu()
        n_pass = int(passed.sum())
        self.gate_log.append({
            "class": c, "cycle": cycle, "n_candidates": int(candidates.size(0)),
            "n_add": int(n_add), "n_pass": n_pass, "fallback": bool(n_pass < n_add),
            "n_admitted": int(idx_cpu.numel()),
            "n_admitted_failing_gate": int((~passed[idx_cpu]).sum()),
        })
        self._admitted_meta.setdefault(c, []).append({
            "cycle": np.full(idx_cpu.numel(), cycle, dtype=np.int16),
            "p_c": p_c[idx_cpu].numpy().astype(np.float32),
            "passed_gate": passed[idx_cpu].numpy(),
        })
        if self.save_candidates:
            self.candidate_store.append({
                "class": c, "cycle": cycle, "images_uint8": to_uint8(candidates),
                "p_c": p_c.numpy().astype(np.float32), "probs": probs.numpy().astype(np.float32),
                "passed_gate": passed.numpy(), "admitted_index": idx_cpu.numpy(),
            })
        return idx

    def admitted_metadata(self, deficits: Dict[int, int]) -> Optional[Dict[str, np.ndarray]]:
        """Per-image metadata aligned with the output of ``synthesize`` (class order,
        cycle order within class, truncated to the deficit as ``synthesize`` does)."""
        out = {"class": [], "cycle": [], "p_c": [], "passed_gate": []}
        for c, n in deficits.items():
            parts = self._admitted_meta.get(int(c), [])
            if not parts:
                continue
            cyc = np.concatenate([p["cycle"] for p in parts])[:n]
            out["class"].append(np.full(len(cyc), int(c), dtype=np.int16))
            out["cycle"].append(cyc)
            out["p_c"].append(np.concatenate([p["p_c"] for p in parts])[:n])
            out["passed_gate"].append(np.concatenate([p["passed_gate"] for p in parts])[:n])
        if not out["class"]:
            return None
        return {k: np.concatenate(v) for k, v in out.items()}
