"""casestudy.methods.selection_v2 - candidate selection rules for the KBS revision (S3).

For one minority class ``c`` in one acquisition cycle, a rule chooses ``n`` of the ``P``
candidates, given the internal classifier's softmax ``probs`` [P, K] and its penultimate
features ``h`` [P, d]:

    random          n uniformly at random                         no selection
    margin          the n smallest top-1 minus top-2 margins      uncertainty
    badge           k-means++ on gradient embeddings (Ash et al.)  uncertainty + diversity
    agree_margin    among argmax == c, the n smallest margins      uncertainty under agreement
    top_confidence  the n largest p(c)                             confidence ranking
    gate_random     among the gate passers, n at random            confidence only
    diversity_only  k-means++ on h over all candidates             diversity only
    gate_diverse    k-means++ on h over the gate passers           confidence + diversity (proposed)

The gate is ``argmax == c and p(c) >= tau``. For the gated rules, ``fallback`` decides what
happens when fewer than ``n`` candidates pass: ``"top_confidence"`` fills with the most
confident candidates (the submitted rule), ``"none"`` returns the passers only.

``gate_diverse`` with ``fallback="top_confidence"`` returns exactly what
``Ab8cMethod._quality_select`` returns for the same inputs; ``badge`` matches
``Ab8cMethod._badge_select`` with normalisation on.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

MODES = ("random", "margin", "badge", "agree_margin", "top_confidence",
         "gate_random", "diversity_only", "gate_diverse")
GATED = ("gate_random", "gate_diverse")
# Audit strata (E5, Reviewer 3.1): the same budget drawn at random from one confidence tercile of the
# candidates, or from the lower / upper half of confidence among the candidates the reference
# classifier labels as the intended class.
STRATA = ("conf_low", "conf_mid", "conf_high", "correct_low", "correct_high")
NEEDS_ORACLE = ("correct_low", "correct_high")


def score(C: torch.nn.Module, candidates: torch.Tensor, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax probabilities and penultimate features of ``C`` on ``candidates`` (both on CPU)."""
    captured: Dict[str, torch.Tensor] = {}
    handle = C.head.register_forward_hook(lambda _m, inp, _o: captured.__setitem__("h", inp[0].detach()))
    try:
        C.eval()
        with torch.no_grad():
            logits = C(candidates.to(device))
    finally:
        handle.remove()
    return torch.softmax(logits, dim=1).cpu(), captured["h"].cpu()


def _kpp(feats: torch.Tensor, k: int, seed: int, normalize: bool = True) -> torch.Tensor:
    """k-means++ seeding on the rows of ``feats``; returns ``min(k, rows)`` row indices."""
    from sklearn.cluster import kmeans_plusplus

    f = feats / (feats.norm(dim=1, keepdim=True) + 1e-12) if normalize else feats
    k = min(int(k), f.size(0))
    if k <= 1:
        return torch.arange(k, dtype=torch.long)
    _, idx = kmeans_plusplus(f.numpy(), n_clusters=k, random_state=seed)
    return torch.as_tensor(idx, dtype=torch.long)


def select(mode: str, probs: torch.Tensor, h: torch.Tensor, n: int, c: int, tau: float, seed: int,
           fallback: str = "top_confidence", oracle_agree: torch.Tensor | None = None) -> Tuple[torch.Tensor, dict]:
    """Indices of the admitted candidates, and a record of what the gate saw.

    ``oracle_agree`` (bool per candidate: the reference classifier labels it class ``c``) is required by
    the ``correct_*`` strata and ignored otherwise.
    """
    if mode not in MODES + STRATA:
        raise ValueError(f"unknown selection mode {mode!r}; expected one of {MODES + STRATA}")
    if mode in NEEDS_ORACLE and oracle_agree is None:
        raise ValueError(f"mode {mode!r} needs the reference classifier's verdicts")
    if fallback not in ("top_confidence", "none"):
        raise ValueError(f"unknown fallback {fallback!r}")
    P = probs.size(0)
    p_c = probs[:, c]
    pred = probs.argmax(dim=1)
    top2 = probs.topk(min(2, probs.size(1)), dim=1).values
    margin = top2[:, 0] - top2[:, 1] if top2.size(1) > 1 else top2[:, 0]
    gate = (pred == c) & (p_c >= tau)
    n_pass = int(gate.sum())
    rng = np.random.default_rng(seed)
    fell_back = False

    if mode == "random":
        idx = torch.as_tensor(rng.choice(P, size=min(n, P), replace=False), dtype=torch.long)
    elif mode == "margin":
        idx = torch.argsort(margin)[:n]
    elif mode == "badge":
        p_minus_e = probs.clone()
        p_minus_e[torch.arange(P), pred] -= 1.0
        emb = (p_minus_e.unsqueeze(2) * h.unsqueeze(1)).reshape(P, -1)
        idx = _kpp(emb, n, seed, normalize=True)
    elif mode == "agree_margin":
        agree = torch.where(pred == c)[0]
        order = agree[torch.argsort(margin[agree])]
        if order.numel() >= n:
            idx = order[:n]
        else:
            fell_back = True
            rest = torch.where(pred != c)[0]
            rest = rest[torch.argsort(margin[rest])]
            idx = torch.cat([order, rest[: n - order.numel()]])
    elif mode == "top_confidence":
        idx = torch.argsort(p_c, descending=True)[:n]
    elif mode == "diversity_only":
        idx = _kpp(h, n, seed, normalize=True)
    elif mode in STRATA:
        if mode in NEEDS_ORACLE:
            ok = torch.where(oracle_agree)[0]
            ok = ok[torch.argsort(p_c[ok])]
            part = ok[: ok.numel() // 2] if mode == "correct_low" else ok[ok.numel() // 2:]
        else:
            order = torch.argsort(p_c)
            third = P // 3
            part = {"conf_low": order[:third], "conf_mid": order[third:2 * third], "conf_high": order[2 * third:]}[mode]
        k = min(n, part.numel())
        fell_back = k < n                    # the stratum held fewer candidates than the budget
        idx = part[torch.as_tensor(rng.choice(part.numel(), size=k, replace=False), dtype=torch.long)] if k else part
    else:  # gated rules
        keep = torch.where(gate)[0]
        if keep.numel() < n:
            fell_back = True
            if fallback == "top_confidence":
                keep = torch.argsort(p_c, descending=True)[:max(n, keep.numel())]
        if mode == "gate_diverse":
            idx = keep[_kpp(h[keep], n, seed, normalize=True)] if keep.numel() else keep
        else:
            k = min(n, keep.numel())
            idx = keep[torch.as_tensor(rng.choice(keep.numel(), size=k, replace=False), dtype=torch.long)] if k else keep

    idx = idx.long()
    adm_gate = gate[idx]
    conf = p_c[idx]
    stats = {
        "mode": mode, "n_pool": P, "n_req": int(n), "n_pass": n_pass, "pass_rate": n_pass / max(P, 1),
        "fell_back": bool(fell_back), "n_admitted": int(idx.numel()),
        "n_violating": int((~adm_gate).sum()),
        "violating_argmax": int((pred[idx] != c).sum()),
        "violating_below_tau": int(((pred[idx] == c) & (p_c[idx] < tau)).sum()),
        "conf_admitted_mean": float(conf.mean()) if conf.numel() else float("nan"),
        "conf_admitted_min": float(conf.min()) if conf.numel() else float("nan"),
    }
    return idx, stats
