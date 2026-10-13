"""m10 · Ab-8c (proposed) — standalone re-implementation (plan §5.1 row 10).

The **only** method permitted to use active learning. Pipeline:

    AE warm-start  →  BAGAN-GP backbone  →  BADGE acquisition  →
    3-player (D / G / C) co-training

A faithful, self-contained re-implementation of the GAAL_Ablation_130J Ab-8c
algorithm (130J is the *algorithmic reference only* — this module does **not**
import ``ablation`` or ``bagan``). It reuses the case-study BAGAN-GP building
blocks (``_bagan_common`` / ``_backbones`` / ``m09``'s aux critic) for the
warm-up, then adds the two mechanisms that are exclusive to Ab-8c:

1. **BADGE acquisition** (Ash et al. 2020) — each AL cycle, over-produce a
   candidate pool per minority class from the generator, embed each candidate
   with the closed-form gradient ``g_i = (p_i − e_ŷᵢ) ⊗ h_i`` (where ``h_i`` is
   the penultimate activation captured by a forward hook on the classifier's
   final ``nn.Linear``), then seed ``k`` diverse+uncertain picks via
   k-means++. Selected samples accumulate across cycles.
2. **3-player co-training** — per cycle, fine-tune the WGAN-GP critic (D), the
   generator (G), and a classifier (C) together:
       D-step:  WGAN-GP critic + λ_aux·CE(aux(real),y) + CE(aux(synth),y)
       G-step:  λ_adv·(−D(G(z))) + λ_aux_g·CE(aux(G(z)),y)
                + λ_cls·CE(C(G(z)),y) + λ_anchor·‖φ_D(G(z))−φ_D(G_ref(z))‖²
       C-step:  epochs on real → cumulative selected synthetics
   ``G_ref`` is a frozen snapshot of G at the end of warm-up (static anchor).

``C`` is an *internal* co-training player only; the downstream evaluation
classifier is trained separately by the notebook (§E/§F). In strict mode,
``synthesize`` returns only BADGE-selected synthetics and raises if the selected
set cannot cover the requested deficits; mixed-fill mode keeps the older
refined-generator top-up behavior.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .base import AugmentationMethod
from ._backbones import Generator, gradient_penalty, init_weights
from ._bagan_common import (
    AutoEncoder, LatentPrior, pretrain_autoencoder, transfer_encoder_to_trunk,
)
from .m09_bagan_gp import _AuxCritic

print("casestudy/methods/m10_ab8c.py loaded")


class _CoCNN(nn.Module):
    """Small internal classifier (the third co-training player, ``C``).

    Operates on ``[B, C, 64, 64]`` and exposes a final ``nn.Linear`` so the
    BADGE gradient embedding can hook its penultimate activation. Independent
    of the downstream evaluation backbone (kept deliberately compact).
    """

    def __init__(self, in_channels: int, num_classes: int, base: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, 2, 1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, 2, 1), nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, 2, 1), nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(base * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.head(h)


class Ab8cMethod(AugmentationMethod):
    """Ab-8c — BAGAN-GP backbone + BADGE acquisition + 3-player co-training."""

    name = "Ab-8c (proposed)"
    is_generative = True

    # WGAN-GP warm-up hyperparameters (match m09).
    lambda_gp = 10.0
    n_critic = 5
    base_lr = 1e-4
    base_betas = (0.0, 0.9)

    # Co-training defaults (mirror 130J Ab-8c spirit; see ablation/config.py).
    samples_per_cycle = 20      # BADGE picks per minority class per cycle
    pool_factor = 10             # candidate over-production multiplier
    k_d = 15                    # D minibatches per cycle
    k_g = 10                    # G minibatches per cycle
    lr_d_factor = 0.05          # per-cycle D fine-tune lr = base_lr * this
    lr_g_factor = 0.05          # per-cycle G fine-tune lr = base_lr * this
    lambda_aux_real = 2.0       # CE(aux(real), y) in D-step
    lambda_aux_synth = 1.0      # CE(aux(selected synth), y) in D-step
    lambda_adv = 0.25           # −D(G(z)) in G-step
    lambda_aux_g = 0.0          # CE(aux(G(z)), y) in G-step (off; C is teacher)
    lambda_cls = 0.2            # CE(C(G(z)), y) in G-step (classifier teacher)
    lambda_anchor = 1.0         # ‖φ_D(G(z)) − φ_D(G_ref(z))‖² in G-step
    kpp_seed_normalize = True   # L2-normalize BADGE embeddings before k-means++
    c_lr = 1e-3                 # internal classifier lr
    c_warmup_epochs = 15        # epochs to train C on real BEFORE active learning
    c_epochs_per_cycle = 1      # C-step epochs per AL cycle
    val_frac = 0.2              # internal stratified val split for AL patience

    # ── acquisition strategy ──────────────────────────────────────────────
    # "confident_diverse" (default): reject off-manifold / mis-conditioned
    #   candidates (C's argmax != target class, or p(target) < min_confidence),
    #   then k-means++ for diversity among the surviving high-quality pool.
    #   This is the correct criterion when the acquired samples are *used as
    #   pseudo-labeled training data* (no oracle relabels them), and it is what
    #   lets Ab-8c inject clean, on-manifold minority samples.
    # "badge": original Ash-et-al. uncertainty+diversity gradient embedding
    #   (kept for reproducibility of earlier runs). NOTE: uncertainty selection
    #   deliberately picks the most ambiguous generated samples, which become
    #   label noise once injected with a hard class label — hence it hurts the
    #   downstream classifier here.
    selection_mode = "confident_diverse"
    min_confidence = 0.6        # p(target class) gate for confident_diverse
    # Opt-in realism/typicality filter (default OFF -> preserves every prior
    # run, incl. the Fashion win).  Ranks confidence-gated candidates by cosine
    # similarity to the REAL minority feature centroid and keeps the most
    # typical fraction, removing blurry/atypical synth that classifier
    # confidence cannot detect (a blurry sign is still confidently classified).
    realism_filter = False
    realism_keep_frac = 0.5

    # ── candidate seeding (A4) ────────────────────────────────────────────
    # "prior" (default): sample the per-class isotropic Gaussian latent prior.
    # "latent_smote": seed from SMOTE-style interpolations of ENCODED real
    #   minority latents, giving generated candidates the on-manifold
    #   reliability of resamplers while keeping GAN diversity + AL selection.
    #   Default: measured +macro-F2 and lower variance vs "prior" (CS3J A4).
    seed_mode = "latent_smote"
    latent_smote_noise = 0.25   # jitter as a fraction of per-class prior std

    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        self.X_real, self.y_real = X_real, y_real
        self._G: Generator | None = None
        self._prior: LatentPrior | None = None
        self._selected: Dict[int, torch.Tensor] = {}  # class → accumulated synth
        self._real_latents: Dict[int, torch.Tensor] = {}  # class → encoded reals (A4)

    # ── warm-up: BAGAN-GP backbone (identical recipe to m09) ──────────────
    def _warmup_bagan_gp(self, device: str) -> Tuple[Generator, _AuxCritic, LatentPrior]:
        K = self.num_classes
        ae = AutoEncoder(self.channels, self.latent_dim)
        ae = pretrain_autoencoder(
            ae, self.X_real, self.y_real, self.config.gan_epochs,
            self.config.batch_size, device,
        )
        prior = LatentPrior(self.latent_dim).fit(ae.encoder, self.X_real, self.y_real, device)

        # A4: cache per-class encoded real latents for latent-SMOTE seeding.
        ae.encoder.eval()
        self._real_latents = {}
        with torch.no_grad():
            for c in torch.unique(self.y_real).tolist():
                Xc = self.X_real[self.y_real == c].to(device)
                self._real_latents[int(c)] = ae.encoder(Xc).detach().cpu()

        G = Generator(self.latent_dim, self.channels).to(device)
        G.load_state_dict(ae.decoder.state_dict())
        C = _AuxCritic(self.channels, K).to(device)
        transfer_encoder_to_trunk(ae.encoder, C.trunk)

        optG = torch.optim.Adam(G.parameters(), lr=self.base_lr, betas=self.base_betas)
        optC = torch.optim.Adam(C.parameters(), lr=self.base_lr, betas=self.base_betas)
        ce = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_real, self.y_real),
            batch_size=min(self.config.batch_size, len(self.X_real)),
            shuffle=True, drop_last=False,
        )
        G.train(); C.train()
        for _ in range(self.config.gan_epochs):
            for step, (real, lbl) in enumerate(loader):
                real, lbl = real.to(device), lbl.to(device)
                bsz = real.size(0)
                cls = torch.randint(0, K, (bsz,), device=device)
                z = torch.cat([prior.sample(int(c), 1, device) for c in cls], dim=0)
                fake = G(z).detach()
                optC.zero_grad()
                s_real, c_real = C(real)
                s_fake, _ = C(fake)
                gp = gradient_penalty(C, real, fake, device, aux=True)
                loss_c = s_fake.mean() - s_real.mean() + self.lambda_gp * gp + ce(c_real, lbl)
                loss_c.backward(); optC.step()
                if step % self.n_critic == 0:
                    z = torch.cat([prior.sample(int(c), 1, device) for c in cls], dim=0)
                    gen = G(z)
                    optG.zero_grad()
                    s_gen, c_gen = C(gen)
                    loss_g = -s_gen.mean() + ce(c_gen, cls)
                    loss_g.backward(); optG.step()
        return G, C, prior

    # ── internal helpers ──────────────────────────────────────────────────
    def _stratified_val_split(self) -> Tuple[torch.Tensor, ...]:
        """Split the real pool into train/val (stratified) for AL patience."""
        g = torch.Generator().manual_seed(self.seed)
        tr_idx, va_idx = [], []
        for c in torch.unique(self.y_real).tolist():
            ci = torch.where(self.y_real == c)[0]
            ci = ci[torch.randperm(len(ci), generator=g)]
            n_val = max(1, int(round(self.val_frac * len(ci))))
            va_idx.append(ci[:n_val]); tr_idx.append(ci[n_val:])
        tr = torch.cat(tr_idx); va = torch.cat(va_idx)
        return self.X_real[tr], self.y_real[tr], self.X_real[va], self.y_real[va]

    def _sample_balanced_z(self, prior: LatentPrior, n: int, device: str):
        """Sample ``n`` latents with roughly balanced class labels."""
        K = self.num_classes
        labels = torch.arange(n) % K
        z = torch.cat([prior.sample(int(c), 1, device) for c in labels], dim=0)
        return z, labels.to(device)

    def _latent_smote_sample(self, c: int, n: int, device: str, prior: LatentPrior) -> torch.Tensor:
        """Latent-space SMOTE: interpolate random pairs of encoded real class-c latents."""
        Z = self._real_latents.get(int(c))
        if Z is None or Z.size(0) < 2:
            return prior.sample(c, n, device)
        Z = Z.to(device)
        nc = Z.size(0)
        i = torch.randint(0, nc, (n,), device=device)
        j = torch.randint(0, nc, (n,), device=device)
        lam = torch.rand(n, 1, device=device)
        z = Z[i] + lam * (Z[j] - Z[i])
        if self.latent_smote_noise > 0:
            std = float(prior.stds.get(int(c), 0.0))
            z = z + self.latent_smote_noise * std * torch.randn(n, self.latent_dim, device=device)
        return z

    def _seed_latents(self, c: int, n: int, device: str, prior: LatentPrior) -> torch.Tensor:
        """Dispatch candidate-latent seeding by ``seed_mode`` (prior | latent_smote)."""
        if self.seed_mode == "latent_smote":
            return self._latent_smote_sample(c, n, device, prior)
        return prior.sample(c, n, device)

    @staticmethod
    def _badge_select(C: _CoCNN, candidates: torch.Tensor, n_add: int,
                      normalize: bool, seed: int, device: str) -> torch.Tensor:
        """BADGE: gradient-embed candidates, k-means++ seed ``n_add`` picks."""
        from sklearn.cluster import kmeans_plusplus

        captured: Dict[str, torch.Tensor] = {}

        def _hook(_m, inp, _o):
            captured["h"] = inp[0].detach()

        handle = C.head.register_forward_hook(_hook)
        try:
            C.eval()
            with torch.no_grad():
                logits = C(candidates.to(device))
        finally:
            handle.remove()
        h = captured["h"]                                   # [P, d]
        probs = torch.softmax(logits, dim=1)                # [P, K]
        y_hat = probs.argmax(dim=1)
        p_minus_e = probs.clone()
        p_minus_e[torch.arange(p_minus_e.size(0)), y_hat] -= 1.0
        emb = (p_minus_e.unsqueeze(2) * h.unsqueeze(1)).reshape(p_minus_e.size(0), -1)
        if normalize:
            emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-12)
        emb_np = emb.cpu().numpy()
        k = min(n_add, emb_np.shape[0])
        _, idx = kmeans_plusplus(emb_np, n_clusters=k, random_state=seed)
        return torch.as_tensor(idx, dtype=torch.long)

    @staticmethod
    def _quality_select(C: _CoCNN, candidates: torch.Tensor, n_add: int,
                        target_class: int, min_confidence: float,
                        seed: int, device: str,
                        real_centroid=None, keep_frac: float = 0.5) -> torch.Tensor:
        """Confidence-gated diverse acquisition (default).

        Unlike BADGE (which seeks *uncertain* points for an oracle to relabel),
        this keeps only candidates the internal classifier confidently assigns
        to ``target_class`` — i.e. on-manifold, correctly class-conditioned
        generations — then k-means++ seeds ``n_add`` *diverse* picks among them
        (diversity over the penultimate features, not the uncertainty gradient).
        Falls back to the top-confidence candidates if too few clear the gate,
        so strict fill always returns ``min(n_add, P)`` indices.
        """
        from sklearn.cluster import kmeans_plusplus

        captured: Dict[str, torch.Tensor] = {}

        def _hook(_m, inp, _o):
            captured["h"] = inp[0].detach()

        handle = C.head.register_forward_hook(_hook)
        try:
            C.eval()
            with torch.no_grad():
                logits = C(candidates.to(device))
        finally:
            handle.remove()
        # Keep all index bookkeeping on CPU so the returned indices match the
        # CPU-resident candidate tensor (candidates stay on CPU in the caller).
        h = captured["h"].cpu()                             # [P, d]
        probs = torch.softmax(logits, dim=1).cpu()          # [P, K]
        p_c = probs[:, target_class]
        pred = probs.argmax(dim=1)

        keep = (pred == target_class) & (p_c >= min_confidence)
        keep_idx = torch.where(keep)[0]
        if keep_idx.numel() < n_add:
            # Relax the gate: take the highest-confidence candidates so we can
            # still cover the requested count (guarantees strict fill).
            order = torch.argsort(p_c, descending=True)
            keep_idx = order[:max(n_add, keep_idx.numel())]

        if real_centroid is not None and keep_idx.numel() > n_add:
            # Typicality filter: among gated candidates, keep the fraction most
            # similar (cosine, feature space) to the REAL minority centroid, so
            # atypical/blurry synth is dropped before the diversity step.
            hk = h[keep_idx]
            hk = hk / (hk.norm(dim=1, keepdim=True) + 1e-12)
            typ = hk @ real_centroid                               # [m]
            keep_n = max(n_add, int(round(keep_frac * keep_idx.numel())))
            keep_idx = keep_idx[torch.argsort(typ, descending=True)[:keep_n]]

        h_keep = h[keep_idx]
        h_keep = h_keep / (h_keep.norm(dim=1, keepdim=True) + 1e-12)
        k = min(n_add, h_keep.size(0))
        if k <= 1:
            return keep_idx[:k]
        _, sub = kmeans_plusplus(h_keep.numpy(), n_clusters=k, random_state=seed)
        return keep_idx[torch.as_tensor(sub, dtype=torch.long)]

    def _class_feat_centroid(self, C: _CoCNN, X: torch.Tensor, device: str) -> torch.Tensor:
        """Normalized mean penultimate feature of ``X`` under classifier ``C``."""
        capt: Dict[str, torch.Tensor] = {}
        handle = C.head.register_forward_hook(lambda m, i, o: capt.__setitem__("h", i[0].detach()))
        try:
            C.eval()
            with torch.no_grad():
                C(X.to(device))
        finally:
            handle.remove()
        f = capt["h"].cpu()
        f = f / (f.norm(dim=1, keepdim=True) + 1e-12)
        mu = f.mean(dim=0)
        return mu / (mu.norm() + 1e-12)

    def _acquire(self, C: _CoCNN, candidates: torch.Tensor, n_add: int,
                 target_class: int, Xva: torch.Tensor | None = None,
                 yva: torch.Tensor | None = None) -> torch.Tensor:
        """Dispatch to the configured acquisition strategy."""
        if self.selection_mode == "badge":
            return self._badge_select(
                C, candidates, n_add, self.kpp_seed_normalize, self.seed, self.device)
        if self.selection_mode == "val_f2_greedy" and Xva is not None:
            return self._valf2_select(C, candidates, n_add, target_class, Xva, yva)
        real_centroid = None
        if getattr(self, "realism_filter", False) and self.X_real is not None:
            real_c = self.X_real[self.y_real == target_class]
            if real_c.numel() > 0:
                real_centroid = self._class_feat_centroid(C, real_c, self.device)
        return self._quality_select(
            C, candidates, n_add, target_class, self.min_confidence,
            self.seed, self.device, real_centroid, self.realism_keep_frac)

    def _valf2_select(self, C: _CoCNN, candidates: torch.Tensor, n_add: int,
                      target_class: int, Xva: torch.Tensor, yva: torch.Tensor) -> torch.Tensor:
        """Validation-F2-guided acquisition (Path A).

        Directly targets the reported metric: among confident, correctly
        class-conditioned candidates, prefer those that resemble the *val
        minority samples the internal classifier currently MISSES* (recall
        errors — what macro-F2 rewards fixing), then k-means++ for diversity.
        Falls back to confident-diverse when C already recalls this class.
        """
        from sklearn.cluster import kmeans_plusplus

        device = self.device

        def _feats(x):
            capt: Dict[str, torch.Tensor] = {}
            h = C.head.register_forward_hook(lambda m, i, o: capt.__setitem__("h", i[0].detach()))
            try:
                C.eval()
                with torch.no_grad():
                    logits = C(x.to(device))
            finally:
                h.remove()
            return capt["h"].cpu(), logits.cpu()

        hc, logits = _feats(candidates)
        probs = torch.softmax(logits, dim=1)
        p_c = probs[:, target_class]
        pred = probs.argmax(dim=1)
        keep = (pred == target_class) & (p_c >= self.min_confidence)
        keep_idx = torch.where(keep)[0]
        if keep_idx.numel() < n_add:
            keep_idx = torch.argsort(p_c, descending=True)[:max(n_add, keep_idx.numel())]

        hk = hc[keep_idx]
        hk = hk / (hk.norm(dim=1, keepdim=True) + 1e-12)

        hv, vlogits = _feats(Xva)
        vpred = vlogits.argmax(dim=1)
        missed = (yva == target_class) & (vpred != target_class)  # recall errors for class c
        if missed.any():
            mu = hv[missed]
            mu = mu / (mu.norm(dim=1, keepdim=True) + 1e-12)
            mu = mu.mean(dim=0, keepdim=True)
            mu = mu / (mu.norm() + 1e-12)
            sim = (hk @ mu.t()).squeeze(1)
            topn = min(hk.size(0), max(n_add, n_add * 3))
            top = torch.argsort(sim, descending=True)[:topn]
            sub_idx, hsub = keep_idx[top], hk[top]
        else:  # recall already perfect on val -> just diversify confident picks
            sub_idx, hsub = keep_idx, hk

        k = min(n_add, hsub.size(0))
        if k <= 1:
            return sub_idx[:k]
        _, sel = kmeans_plusplus(hsub.numpy(), n_clusters=k, random_state=self.seed)
        return sub_idx[torch.as_tensor(sel, dtype=torch.long)]


    def _train_classifier(self, C: _CoCNN, X: torch.Tensor, y: torch.Tensor,
                          opt, epochs: int, device: str) -> None:
        ce = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y),
            batch_size=min(self.config.batch_size, len(X)), shuffle=True,
        )
        C.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = ce(C(xb), yb)
                loss.backward(); opt.step()

    @torch.no_grad()
    def _val_accuracy(self, C: _CoCNN, Xv: torch.Tensor, yv: torch.Tensor, device: str) -> float:
        C.eval()
        pred = C(Xv.to(device)).argmax(dim=1).cpu()
        return float((pred == yv).float().mean().item())

    @torch.no_grad()
    def _val_f2(self, C: _CoCNN, Xv: torch.Tensor, yv: torch.Tensor, device: str) -> float:
        """Internal macro-F2 on the val split (AL patience monitor; matches headline)."""
        C.eval()
        pred = C(Xv.to(device)).argmax(dim=1).cpu().numpy()
        yt = yv.cpu().numpy()
        fs = []
        for c in range(self.num_classes):
            tp = int(((pred == c) & (yt == c)).sum())
            fp = int(((pred == c) & (yt != c)).sum())
            fn = int(((pred != c) & (yt == c)).sum())
            denom = 5 * tp + 4 * fn + fp
            fs.append((5 * tp / denom) if denom > 0 else 0.0)
        return float(sum(fs) / len(fs))

    # ── full Ab-8c pipeline ────────────────────────────────────────────────
    def _build_and_train(self, deficits: Dict[int, int]) -> None:
        from casestudy.reproducibility import set_seed

        set_seed(self.seed)
        device = self.device
        K = self.num_classes

        # 1) BAGAN-GP warm-up + frozen reference generator (static anchor).
        G, D, prior = self._warmup_bagan_gp(device)
        G_ref = copy.deepcopy(G).eval()
        for p in G_ref.parameters():
            p.requires_grad_(False)

        # 2) Internal classifier C (third player) + warm-up on real pool.
        Xtr, ytr, Xva, yva = self._stratified_val_split()
        C = _CoCNN(self.channels, K).to(device)
        init_weights(C)
        optC_cls = torch.optim.Adam(C.parameters(), lr=self.c_lr)
        self._train_classifier(C, Xtr, ytr, optC_cls, epochs=self.c_warmup_epochs, device=device)

        # AL-cycle plan: strict mode guarantees enough BADGE acquisitions to
        # fill the largest deficit. Mixed-fill mode keeps the older capped
        # behavior and lets synthesize() top up any remaining shortfall.
        max_deficit = max(deficits.values()) if deficits else 0
        required_cycles = max(1, math.ceil(max_deficit / self.samples_per_cycle))
        strict_badge_fill = bool(getattr(self.config, "strict_badge_fill", False))
        if strict_badge_fill:
            n_cycles = required_cycles
        else:
            n_cycles = min(required_cycles, int(self.config.max_al_cycles))
        epochs_per_cycle = self.c_epochs_per_cycle

        optD = torch.optim.Adam(D.parameters(), lr=self.base_lr * self.lr_d_factor, betas=self.base_betas)
        optG = torch.optim.Adam(G.parameters(), lr=self.base_lr * self.lr_g_factor, betas=self.base_betas)
        ce = nn.CrossEntropyLoss()

        selected: Dict[int, List[torch.Tensor]] = {c: [] for c in deficits}
        best_acc, best_selected = -1.0, None
        no_improve = 0

        for _cycle in range(n_cycles):
            # (a) BADGE acquisition per minority class.
            for c, deficit in deficits.items():
                have = sum(t.size(0) for t in selected[c])
                n_add = min(self.samples_per_cycle, deficit - have)
                if n_add <= 0:
                    continue
                n_cand = max(n_add, n_add * self.pool_factor)
                with torch.no_grad():
                    z = self._seed_latents(c, n_cand, device, prior)
                    cand = G(z).clamp(-1.0, 1.0).cpu()
                idx = self._acquire(C, cand, n_add, c, Xva, yva)
                selected[c].append(cand[idx])

            # Concatenate accumulated synthetics for this cycle's steps.
            syn_x = [t for ts in selected.values() for t in ts]
            syn_y = [torch.full((t.size(0),), c, dtype=torch.long)
                     for c, ts in selected.items() for t in ts]
            S_x = torch.cat(syn_x) if syn_x else None
            S_y = torch.cat(syn_y) if syn_y else None

            # (b) D-step: WGAN-GP critic + aux CE on real (+ accepted synth).
            G.eval(); D.train()
            n_real = self.X_real.size(0)
            for _ in range(self.k_d):
                bidx = torch.randint(0, n_real, (min(self.config.batch_size, n_real),))
                real = self.X_real[bidx].to(device)
                real_y = self.y_real[bidx].to(device)
                z, _ = self._sample_balanced_z(prior, real.size(0), device)
                with torch.no_grad():
                    fake = G(z)
                optD.zero_grad()
                s_real, a_real = D(real)
                s_fake, _ = D(fake)
                gp = gradient_penalty(D, real, fake, device, aux=True)
                loss_d = (s_fake.mean() - s_real.mean() + self.lambda_gp * gp
                          + self.lambda_aux_real * ce(a_real, real_y))
                if S_x is not None and S_x.size(0) > 0:
                    sidx = torch.randint(0, S_x.size(0), (min(self.config.batch_size, S_x.size(0)),))
                    _, a_syn = D(S_x[sidx].to(device))
                    loss_d = loss_d + self.lambda_aux_synth * ce(a_syn, S_y[sidx].to(device))
                loss_d.backward(); optD.step()

            # (c) G-step: adv + aux_g + classifier-teacher + static anchor.
            G.train(); D.eval()
            for p in D.parameters():
                p.requires_grad_(False)
            for _ in range(self.k_g):
                z, y_z = self._sample_balanced_z(prior, self.config.batch_size, device)
                fake = G(z)
                s_fake, a_fake = D(fake)
                adv = -s_fake.mean()
                aux_g = ce(a_fake, y_z)
                C.eval()
                cls_l = ce(C(fake), y_z)
                with torch.no_grad():
                    phi_ref = D.trunk(G_ref(z)).detach()
                phi = D.trunk(fake)
                anchor = ((phi - phi_ref) ** 2).mean()
                loss_g = (self.lambda_adv * adv + self.lambda_aux_g * aux_g
                          + self.lambda_cls * cls_l + self.lambda_anchor * anchor)
                optG.zero_grad(); loss_g.backward(); optG.step()
            for p in D.parameters():
                p.requires_grad_(True)

            # (d) C-step: retrain C on real (train split) + cumulative synth.
            if S_x is not None and S_x.size(0) > 0:
                grow_x = torch.cat([Xtr, S_x]); grow_y = torch.cat([ytr, S_y])
            else:
                grow_x, grow_y = Xtr, ytr
            self._train_classifier(C, grow_x, grow_y, optC_cls, epochs_per_cycle, device)

            # (e) Validate + AL-cycle patience (keep best synth snapshot).
            acc = self._val_f2(C, Xva, yva, device)
            if acc > best_acc:
                best_acc = acc
                best_selected = {c: [t.clone() for t in ts] for c, ts in selected.items()}
                no_improve = 0
            else:
                no_improve += 1
                if (not strict_badge_fill) and no_improve >= int(self.config.al_patience):
                    break

        G.eval()
        self._G, self._prior = G, prior
        snap = selected if strict_badge_fill else (best_selected if best_selected is not None else selected)
        empty = torch.empty(0, self.channels, self.image_size, self.image_size)
        self._selected = {c: (torch.cat(ts) if ts else empty) for c, ts in snap.items()}

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        deficits = self._deficits(target_counts)
        if not deficits:
            return self._empty()
        if self._G is None:
            self._build_and_train(deficits)
        empty = torch.empty(0, self.channels, self.image_size, self.image_size)
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for c, n in deficits.items():
            picked = self._selected.get(c, empty)
            if picked.size(0) >= n:
                X_c = picked[:n]
            elif bool(getattr(self.config, "strict_badge_fill", False)):
                raise RuntimeError(
                    f"Strict BADGE fill failed for class {c}: "
                    f"needed {n}, selected {picked.size(0)}."
                )
            else:
                # Top up from the refined generator via the class prior.
                extra = n - picked.size(0)
                with torch.no_grad():
                    z = self._prior.sample(c, extra, self.device)
                    gen = self._G(z).cpu().clamp(-1.0, 1.0)
                X_c = torch.cat([picked, gen]) if picked.size(0) > 0 else gen
            xs.append(X_c)
            ys.append(torch.full((n,), c, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)
