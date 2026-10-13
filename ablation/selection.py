"""ablation.selection - candidate acquisition rules for the Ab-8 family.

Both acquisition rules score a pool of *generated* candidates for one minority
class and return the indices to keep. They are factored out here because the
same choice is made in four places (CV loop and retrain loop, for each of
``ab8b`` / ``ab8c``), and the two loops must not be allowed to drift apart.

``badge``
    Ash et al. (2020). Build the closed-form gradient embedding
    ``g_i = (p_i - e_{yhat_i}) (x) h_i`` and k-means++ seed ``n_add`` picks from
    it. Large ``||g_i||`` means the classifier is *uncertain* about candidate
    ``i``, so BADGE deliberately acquires ambiguous points.

``confident_diverse``
    Keep only candidates the classifier confidently assigns to the intended
    class (``argmax == target`` and ``p(target) >= min_confidence``), then
    k-means++ seed ``n_add`` diverse picks over the penultimate features.

The distinction matters because these acquisitions are used for different
purposes. BADGE assumes an oracle will supply a true label for each acquired
point, so ambiguity is informative. Here no oracle exists: an acquired sample
is injected into training carrying the class it was *generated* for. An
ambiguous generation therefore enters as label noise, which is why the
confidence gate is applied before the diversity step.
"""

from typing import Optional

import torch

print("ablation/selection.py loaded")

VALID_SELECTION_MODES = ("badge", "confident_diverse")


def _kmeans_plusplus_indices(emb: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """k-means++ seeding over ``emb`` rows; returns ``k`` row indices (CPU)."""
    from sklearn.cluster import kmeans_plusplus

    emb_np = emb.detach().cpu().numpy()
    k = min(k, emb_np.shape[0])
    _, idx = kmeans_plusplus(emb_np, n_clusters=k, random_state=seed)
    return torch.as_tensor(idx, dtype=torch.long)


def _badge_indices(probs: torch.Tensor, h: torch.Tensor, n_add: int,
                   normalize: bool, seed: int) -> torch.Tensor:
    """BADGE: k-means++ over gradient embeddings ``(p - e_yhat) (x) h``."""
    y_hat = probs.argmax(dim=1)
    p_minus_e = probs.clone()
    p_minus_e[torch.arange(p_minus_e.size(0)), y_hat] -= 1.0
    emb = (p_minus_e.unsqueeze(2) * h.unsqueeze(1)).reshape(p_minus_e.size(0), -1)
    if normalize:
        emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-12)
    return _kmeans_plusplus_indices(emb, n_add, seed)


def _confident_diverse_indices(probs: torch.Tensor, h: torch.Tensor, n_add: int,
                               target_class: int, min_confidence: float,
                               normalize: bool, seed: int,
                               stats: Optional[dict] = None) -> torch.Tensor:
    """Confidence gate on the classifier, then k-means++ over features ``h``."""
    p_c = probs[:, target_class]
    keep = (probs.argmax(dim=1) == target_class) & (p_c >= min_confidence)
    keep_idx = torch.where(keep)[0]
    n_pass = int(keep_idx.numel())
    fell_back = n_pass < n_add

    if fell_back:
        # Too few candidates cleared the gate. Relax to the highest-confidence
        # candidates so the caller still receives min(n_add, pool) indices and
        # the per-cycle fill schedule is unchanged.
        #
        # This path matters: if it fires routinely the gate is doing no work
        # and the rule degenerates to "take the most confident candidates",
        # losing the diversity that k-means++ is there to provide. Warn so it
        # shows up in the run log rather than silently in the results.
        order = torch.argsort(p_c, descending=True)
        keep_idx = order[: max(n_add, n_pass)]
        if n_pass * 2 < n_add:
            print(
                f"      [selection] confidence gate starved: only {n_pass}/"
                f"{probs.size(0)} candidates for class {target_class} cleared "
                f"tau={min_confidence}; falling back to top-confidence "
                f"(diversity step weakened)."
            )

    feats = h[keep_idx]
    if normalize:
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)

    local_idx = _kmeans_plusplus_indices(feats, n_add, seed)
    picked = keep_idx[local_idx]

    if stats is not None:
        stats['n_pool'] = int(probs.size(0))
        stats['n_gate_pass'] = n_pass
        stats['fell_back'] = bool(fell_back)

    return picked


def select_indices(probs: torch.Tensor, h: torch.Tensor, n_add: int,
                   target_class: int, mode: str, min_confidence: float,
                   normalize: bool, seed: int,
                   stats: Optional[dict] = None) -> torch.Tensor:
    """Pick ``n_add`` candidate indices under the requested acquisition rule.

    Args:
        probs: ``[P, K]`` softmax outputs of the classifier on the candidates.
        h: ``[P, d]`` penultimate activations for the same candidates.
        n_add: number of candidates to acquire.
        target_class: remapped class the candidates were generated for.
        mode: ``"badge"`` or ``"confident_diverse"``.
        min_confidence: ``p(target_class)`` floor; ``confident_diverse`` only.
        normalize: L2-normalize the embedding before k-means++ seeding.
        seed: ``random_state`` for k-means++ seeding.
        stats: optional dict; filled with gate diagnostics when
            ``mode == "confident_diverse"``.

    Returns:
        ``LongTensor`` of ``min(n_add, P)`` indices into the candidate pool.
    """
    if mode not in VALID_SELECTION_MODES:
        raise ValueError(
            f"Unknown selection mode {mode!r}; expected one of {VALID_SELECTION_MODES}"
        )

    probs = probs.detach().cpu()
    h = h.detach().cpu()

    if mode == "badge":
        return _badge_indices(probs, h, n_add, normalize, seed)
    return _confident_diverse_indices(
        probs, h, n_add, target_class, min_confidence, normalize, seed, stats
    )


def describe(mode: str, min_confidence: Optional[float] = None) -> str:
    """One-line human-readable description for run banners."""
    if mode == "badge":
        return "BADGE (k-means++ on gradient embeddings)"
    return f"confidence-gated diverse (tau={min_confidence}, k-means++ on features)"
