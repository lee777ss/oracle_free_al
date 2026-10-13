"""m10 v2 - the proposed method, instrumented for the KBS revision (S3).

``Ab8cV2`` subclasses the submitted ``Ab8cMethod`` (``m10_ab8c.py`` is not edited) and changes
only what the revision needs:

* the autoencoder and BAGAN-GP warm-up come from the shared backbone cache;
* no internal 80/20 split: the internal classifier trains on the whole training portion
  (Reviewer 3, comment 4); ``internal_split=True`` restores the submitted split;
* the selection rule is any rule of ``selection_v2`` (Reviewer 1.6, 3.2);
* ``fallback``: ``"top_confidence"`` is the submitted rule, ``"regenerate"`` draws fresh
  candidates (up to ``max_regen_rounds``) and otherwise admits a short batch, so the gate is
  never bypassed (Reviewer 3.3);
* ``cotrain_g``: generator and critic updates inside the loop on (submitted) or off;
* candidates are drawn from named random streams, so every rule sees identical candidates;
* every cycle is logged (``gate_log``, candidate records) and every stage is timed.

Everything else follows ``Ab8cMethod._build_and_train``: the generator stays in train mode
during acquisition, as in the submitted loop; the critic and generator steps are unchanged.

``BAGANGPCached`` is BAGAN-GP drawing its generator from the same cache, so that the step from
BAGAN-GP to the proposed method is measured on one generator.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..reproducibility import set_seed
from ..seeds import stream_seed
from . import backbone_cache, selection_v2
from ._backbones import gradient_penalty, init_weights
from .m09_bagan_gp import BAGANGPMethod
from .m10_ab8c import Ab8cMethod, _CoCNN

print("casestudy/methods/m10v2_ab8c.py loaded")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class BAGANGPCached(BAGANGPMethod):
    """BAGAN-GP whose autoencoder and warm-up come from the shared backbone cache."""

    regime: Optional[str] = None

    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        super().fit(X_real, y_real)
        self.timings: Dict[str, float] = {}

    def _build_and_train(self) -> None:
        bb = backbone_cache.get_backbone(self.regime, self.seed, self.X_real, self.y_real, self.config,
                                         self.channels, self.latent_dim, self.num_classes, self.device)
        self._G, self._prior = bb.G.eval(), bb.prior
        self.timings.update({"backbone_cached": not bb.trained_here, **bb.timings})

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        deficits = self._deficits(target_counts)
        if not deficits:
            return self._empty()
        if self._G is None:
            self._build_and_train()
        set_seed(stream_seed(self.seed, "synthesize"))
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for c, n in deficits.items():
            with torch.no_grad():
                z = self._prior.sample(c, n, self.device)
                xs.append(self._G(z).cpu().clamp(-1.0, 1.0))
            ys.append(torch.full((n,), c, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)


class Ab8cV2(Ab8cMethod):
    """The proposed method with the configuration switches and logging of the revision."""

    name = "Proposed"
    selection_mode = "gate_diverse"
    fallback = "top_confidence"
    cotrain_g = True
    max_regen_rounds = 5
    internal_split = False
    regime: Optional[str] = None
    oracle = None            # callable images -> probabilities, set by the protocol for audit runs

    def fit(self, X_real: torch.Tensor, y_real: torch.Tensor) -> None:
        super().fit(X_real, y_real)
        self.gate_log: List[dict] = []
        self._cand: List[Dict[str, np.ndarray]] = []
        self._gallery: List[Dict[str, np.ndarray]] = []
        self.timings: Dict[str, float] = {}
        self.shortfall: Dict[int, int] = {}
        self.val_curve: List[float] = []
        self._D = self._C = None

    # -- candidates ---------------------------------------------------------------
    def _latents(self, c: int, n: int, gen: torch.Generator, prior) -> torch.Tensor:
        """Latent seeds for class ``c`` from the CPU generator ``gen`` (latent-SMOTE or prior)."""
        std = float(prior.stds.get(int(c), 0.0))
        Z = self._real_latents.get(int(c))
        if self.seed_mode != "latent_smote" or Z is None or Z.size(0) < 2:
            return prior.means[int(c)].unsqueeze(0) + std * torch.randn(n, self.latent_dim, generator=gen)
        nc = Z.size(0)
        i = torch.randint(0, nc, (n,), generator=gen)
        j = torch.randint(0, nc, (n,), generator=gen)
        lam = torch.rand(n, 1, generator=gen)
        z = Z[i] + lam * (Z[j] - Z[i])
        if self.latent_smote_noise > 0:
            z = z + self.latent_smote_noise * std * torch.randn(n, self.latent_dim, generator=gen)
        return z

    def _log_candidates(self, cycle: int, c: int, rnd: torch.Tensor, probs: torch.Tensor,
                        idx: torch.Tensor, q: Optional[torch.Tensor]) -> None:
        """One record per candidate; ``q`` holds the reference classifier's probabilities, if any."""
        P = probs.size(0)
        top2 = probs.topk(2, dim=1).values
        admitted = np.zeros(P, dtype=bool)
        admitted[idx.numpy()] = True
        rec = {"cycle": np.full(P, cycle, np.int16), "cls": np.full(P, c, np.int8), "round": rnd.numpy().astype(np.int8),
               "p_c": probs[:, c].numpy().astype(np.float32), "pred": probs.argmax(1).numpy().astype(np.int8),
               "margin": (top2[:, 0] - top2[:, 1]).numpy().astype(np.float32), "admitted": admitted}
        if q is not None:
            rec["oracle_pred"] = q.argmax(1).numpy().astype(np.int8)
            rec["oracle_p_c"] = q[:, c].numpy().astype(np.float32)
        self._cand.append(rec)

    def _log_gallery(self, cycle: int, c: int, cand: torch.Tensor, probs: torch.Tensor, idx: torch.Tensor,
                     k: int = 8) -> None:
        """Images for the sample figure (R1.9): the first ``k`` admitted and the first ``k`` gate-failing
        candidates of this cycle and class, in candidate order, so the choice is not curated. Indexing
        only; no random numbers are drawn."""
        fails = torch.nonzero(~((probs.argmax(1) == c) & (probs[:, c] >= self.min_confidence))).flatten()[:k]
        for kind, sel in ((0, idx.sort().values[:k]), (1, fails)):
            if sel.numel():
                self._gallery.append({
                    "x": ((cand[sel].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy(),
                    "kind": np.full(sel.numel(), kind, np.int8), "cycle": np.full(sel.numel(), cycle, np.int16),
                    "cls": np.full(sel.numel(), c, np.int8), "p_c": probs[sel, c].numpy().astype(np.float32),
                    "pred": probs[sel].argmax(1).numpy().astype(np.int8)})

    def gallery_arrays(self) -> Dict[str, np.ndarray]:
        """``kind`` 0 = admitted, 1 = failed the gate."""
        if not getattr(self, "_gallery", None):
            return {}
        return {k: np.concatenate([r[k] for r in self._gallery]) for k in self._gallery[0]}

    def candidate_arrays(self) -> Dict[str, np.ndarray]:
        if not self._cand:
            return {}
        return {k: np.concatenate([r[k] for r in self._cand]) for k in self._cand[0]}

    # -- the loop -------------------------------------------------------------------
    def _build_and_train(self, deficits: Dict[int, int]) -> None:
        device, K, T = self.device, self.num_classes, self.timings
        bb = backbone_cache.get_backbone(self.regime, self.seed, self.X_real, self.y_real, self.config,
                                         self.channels, self.latent_dim, K, device)
        G, D, prior = bb.G, bb.D, bb.prior
        self._real_latents = bb.real_latents
        T.update({"backbone_cached": not bb.trained_here, **bb.timings})
        G_ref = copy.deepcopy(G).eval()
        for p in G_ref.parameters():
            p.requires_grad_(False)

        set_seed(stream_seed(self.seed, "loop"))
        if self.internal_split:
            Xtr, ytr, Xva, yva = self._stratified_val_split()
        else:
            Xtr, ytr, Xva, yva = self.X_real, self.y_real, None, None
        _sync(); t0 = time.perf_counter()
        C = _CoCNN(self.channels, K).to(device)
        init_weights(C)
        optC_cls = torch.optim.Adam(C.parameters(), lr=self.c_lr)
        self._train_classifier(C, Xtr, ytr, optC_cls, epochs=self.c_warmup_epochs, device=device)
        _sync(); T["internal_clf_warmup_s"] = time.perf_counter() - t0

        n_cycles = max(1, math.ceil(max(deficits.values()) / self.samples_per_cycle))
        regen = self.fallback == "regenerate" and self.selection_mode in selection_v2.GATED
        max_cycles = 2 * n_cycles if regen else n_cycles
        optD = torch.optim.Adam(D.parameters(), lr=self.base_lr * self.lr_d_factor, betas=self.base_betas)
        optG = torch.optim.Adam(G.parameters(), lr=self.base_lr * self.lr_g_factor, betas=self.base_betas)
        ce = nn.CrossEntropyLoss()
        selected: Dict[int, List[torch.Tensor]] = {c: [] for c in deficits}
        have = lambda c: sum(t.size(0) for t in selected[c])
        for k in ("acquisition_s", "cotrain_s", "cstep_s"):
            T[k] = 0.0

        cycle = 0
        while cycle < max_cycles:
            if regen and cycle >= n_cycles and all(have(c) >= d for c, d in deficits.items()):
                break
            # (a) acquisition, per minority class
            _sync(); ta = time.perf_counter()
            for c, deficit in deficits.items():
                n_add = min(self.samples_per_cycle, deficit - have(c))
                if n_add <= 0:
                    continue
                n_cand = max(n_add, n_add * self.pool_factor)
                cands, probs, feats, rounds = [], [], [], []
                n_pass = 0
                for rnd in range(1 + (self.max_regen_rounds if regen else 0)):
                    gen = torch.Generator().manual_seed(stream_seed(self.seed, f"cand:{cycle}:{c}:{rnd}"))
                    z = self._latents(c, n_cand, gen, prior).to(device)
                    with torch.no_grad():
                        cand = G(z).clamp(-1.0, 1.0).cpu()
                    p, h = selection_v2.score(C, cand, device)
                    cands.append(cand); probs.append(p); feats.append(h)
                    rounds.append(torch.full((cand.size(0),), rnd, dtype=torch.int64))
                    n_pass += int(((p.argmax(1) == c) & (p[:, c] >= self.min_confidence)).sum())
                    if not regen or n_pass >= n_add:
                        break
                cand, p, h, rnd_t = torch.cat(cands), torch.cat(probs), torch.cat(feats), torch.cat(rounds)
                q = self.oracle(cand) if self.oracle is not None else None
                idx, st = selection_v2.select(self.selection_mode, p, h, n_add, c, self.min_confidence,
                                              seed=self.seed, fallback="none" if regen else "top_confidence",
                                              oracle_agree=(q.argmax(1) == c) if q is not None else None)
                if idx.numel():
                    selected[c].append(cand[idx])
                st.update(cycle=cycle, cls=int(c), rounds=len(cands))
                self.gate_log.append(st)
                self._log_candidates(cycle, c, rnd_t, p, idx, q)
                self._log_gallery(cycle, c, cand, p, idx)
            _sync(); T["acquisition_s"] += time.perf_counter() - ta

            syn_x = [t for ts in selected.values() for t in ts]
            syn_y = [torch.full((t.size(0),), c, dtype=torch.long) for c, ts in selected.items() for t in ts]
            S_x = torch.cat(syn_x) if syn_x else None
            S_y = torch.cat(syn_y) if syn_y else None

            # (b) critic and (c) generator steps, exactly as in Ab8cMethod
            if self.cotrain_g:
                _sync(); tc = time.perf_counter()
                G.eval(); D.train()
                n_real = self.X_real.size(0)
                for _ in range(self.k_d):
                    bidx = torch.randint(0, n_real, (min(self.config.batch_size, n_real),))
                    real = self.X_real[bidx].to(device)
                    real_y = self.y_real[bidx].to(device)
                    zz, _ = self._sample_balanced_z(prior, real.size(0), device)
                    with torch.no_grad():
                        fake = G(zz)
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
                G.train(); D.eval()
                for q in D.parameters():
                    q.requires_grad_(False)
                for _ in range(self.k_g):
                    zz, y_z = self._sample_balanced_z(prior, self.config.batch_size, device)
                    fake = G(zz)
                    s_fake, a_fake = D(fake)
                    adv = -s_fake.mean()
                    aux_g = ce(a_fake, y_z)
                    C.eval()
                    cls_l = ce(C(fake), y_z)
                    with torch.no_grad():
                        phi_ref = D.trunk(G_ref(zz)).detach()
                    phi = D.trunk(fake)
                    anchor = ((phi - phi_ref) ** 2).mean()
                    loss_g = (self.lambda_adv * adv + self.lambda_aux_g * aux_g
                              + self.lambda_cls * cls_l + self.lambda_anchor * anchor)
                    optG.zero_grad(); loss_g.backward(); optG.step()
                for q in D.parameters():
                    q.requires_grad_(True)
                _sync(); T["cotrain_s"] += time.perf_counter() - tc

            # (d) classifier step on the training portion plus everything admitted so far
            _sync(); ts = time.perf_counter()
            if S_x is not None and S_x.size(0) > 0:
                grow_x, grow_y = torch.cat([Xtr, S_x]), torch.cat([ytr, S_y])
            else:
                grow_x, grow_y = Xtr, ytr
            self._train_classifier(C, grow_x, grow_y, optC_cls, self.c_epochs_per_cycle, device)
            _sync(); T["cstep_s"] += time.perf_counter() - ts
            if Xva is not None:
                self.val_curve.append(self._val_f2(C, Xva, yva, device))
            cycle += 1

        T["cycles"] = cycle
        G.eval()
        self._G, self._prior, self._D, self._C = G, prior, D, C
        empty = torch.empty(0, self.channels, self.image_size, self.image_size)
        self._selected = {c: (torch.cat(ts) if ts else empty) for c, ts in selected.items()}

    def synthesize(self, target_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        deficits = self._deficits(target_counts)
        if not deficits:
            return self._empty()
        if self._G is None:
            self._build_and_train(deficits)
        empty = torch.empty(0, self.channels, self.image_size, self.image_size)
        xs, ys = [], []
        for c, n in deficits.items():
            picked = self._selected.get(c, empty)[:n]
            if picked.size(0) < n:
                self.shortfall[int(c)] = int(n - picked.size(0))
            xs.append(picked)
            ys.append(torch.full((picked.size(0),), c, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)
