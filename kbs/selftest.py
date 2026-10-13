"""kbs.selftest - checks of the v2 code before any real GPU run (acceptance of S1-S3).

    python -m kbs.selftest units     # pool sampler, selection rules, backbone recipe (about a minute)
    python -m kbs.selftest smoke     # every method and variant end to end at a tiny budget

``smoke`` writes to ``saved_kbs_smoke/`` (never to ``saved_kbs/``) and uses a 2-epoch generative
budget, so its backbones carry a different digest and can never be loaded by a real run.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch


def units() -> bool:
    from casestudy.data.loaders import _apply_imbalance
    from casestudy.methods import selection_v2
    from casestudy.methods.backbone_cache import train_backbone
    from casestudy.methods.m09_bagan_gp import BAGANGPMethod
    from casestudy.methods.m10_ab8c import Ab8cMethod, _CoCNN
    from casestudy.protocol_v2 import make_config, regime_spec, sample_pool
    from casestudy.reproducibility import set_seed

    ok = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. the pool sampler draws exactly what the submitted loader draws
    labels = np.repeat([0, 1, 2], [3000, 2500, 2500])
    for seed in (42, 43, 142):
        old_idx, _ = _apply_imbalance(np.arange(len(labels)), labels, 800, 0.05, seed, False)
        same = np.array_equal(old_idx, sample_pool(labels, 800, 0.05, seed))
        ok &= same
        print(f"[pool]      seed {seed}: sample_pool == _apply_imbalance: {same}")

    # 2. selection rules equal the submitted implementation on identical inputs
    torch.manual_seed(0)
    C = _CoCNN(1, 3).to(device)
    for p_ in C.parameters():
        p_.data.normal_(0, 0.3)
    cand = torch.randn(200, 1, 64, 64).clamp(-1, 1)
    probs, h = selection_v2.score(C, cand, device)
    for c in (1, 2):
        for tau in (0.0, 0.34, 0.5, 0.9, 0.999):
            old = Ab8cMethod._quality_select(C, cand, 20, c, tau, 7, device)
            new, st = selection_v2.select("gate_diverse", probs, h, 20, c, tau, 7, fallback="top_confidence")
            same = torch.equal(old.sort().values, new.sort().values)
            ok &= same
            print(f"[select]    gate_diverse c={c} tau={tau:<5}: same as _quality_select: {same} "
                  f"(passers {st['n_pass']}, fell back {st['fell_back']})")
    old = Ab8cMethod._badge_select(C, cand, 20, True, 7, device)
    new, _ = selection_v2.select("badge", probs, h, 20, 1, 0.6, 7)
    same = torch.equal(old.sort().values, new.sort().values)
    ok &= same
    print(f"[select]    badge: same as _badge_select: {same}")
    for mode in selection_v2.MODES:
        idx, st = selection_v2.select(mode, probs, h, 20, 1, 0.6, 7, fallback="none")
        print(f"[select]    {mode:15s} admitted {st['n_admitted']:2d} of {st['n_pool']}, violating gate {st['n_violating']}")

    # 3. the cached backbone is the BAGAN-GP recipe (m09) given the same seed
    cfg = make_config({"gan_epochs": 2})
    spec = regime_spec("R1_confusable", cfg)
    g = torch.Generator().manual_seed(1)
    X = (torch.rand(150, 1, 64, 64, generator=g) * 2 - 1)
    y = torch.tensor([0] * 110 + [1] * 20 + [2] * 20)
    m09 = BAGANGPMethod(cfg, spec, device, seed=11)
    m09.fit(X, y)
    m09._build_and_train()
    set_seed(11)
    G2, _, prior2, _, _ = train_backbone(X, y, cfg, 1, cfg.latent_dim, 3, device)
    sd1, sd2 = m09._G.state_dict(), G2.state_dict()
    same = all(torch.equal(sd1[k].cpu(), sd2[k].cpu()) for k in sd1)
    same_prior = all(torch.equal(m09._prior.means[c], prior2.means[c]) for c in prior2.means)
    ok &= same and same_prior
    print(f"[backbone]  generator weights identical to m09: {same}; prior identical: {same_prior}")
    print("UNITS", "PASSED" if ok else "FAILED")
    return ok


def smoke() -> bool:
    os.environ["KBS_SAVE_ROOT"] = "saved_kbs_smoke"
    from casestudy.protocol_v2 import FIELD, Job, run_one_v2
    from kbs.verify import hashes

    tiny = {"gan_epochs": 2, "clf_max_epochs": 3, "clf_patience": 2}
    jobs = [Job("T_smoke", "R1_confusable", m, 42, config_overrides=tiny, save_generator=True,
                variant="submitted" if m == "p_proposed" else "",
                overrides={"fallback": "top_confidence", "cotrain_g": True} if m == "p_proposed" else {})
            for m in FIELD]
    jobs += [Job("T_smoke", "R1_confusable", m, 42, config_overrides=tiny) for m in ("m11_naive", "m08_bagan_ls")]
    for fb in ("top_confidence", "regenerate"):
        for co in (True, False):
            jobs.append(Job("T_smoke", "R1_confusable", "p_proposed", 142, variant=f"fallback={fb},cotrain={'on' if co else 'off'}",
                            overrides={"fallback": fb, "cotrain_g": co, "min_confidence": 0.9}, eval_split="dev",
                            config_overrides=tiny))
    for arm in ("random", "margin", "badge", "agree_margin", "top_confidence", "gate_random", "diversity_only", "gate_diverse"):
        jobs.append(Job("T_smoke", "S2_stl_mod", "p_proposed", 42, variant=arm,
                        overrides={"selection_mode": arm, "cotrain_g": False}, config_overrides=tiny))
    jobs += [Job("T_smoke", "OCT2_pathology_mod", m, 42, config_overrides=tiny) for m in ("b1_weighted", "m13_stdaug")]
    jobs += [Job("T_smoke", "G2_signs_mod", m, 42, config_overrides=tiny) for m in ("m13_stdaug", "m14_stdaug_online")]
    jobs += [Job("T_smoke", "C2_cifar_paper", m, 42, config_overrides=tiny) for m in ("b1_weighted", "m09_bagan_gp")]

    ok = True
    for job in jobs:
        t0 = time.time()
        try:
            r = run_one_v2(job)
        except Exception as exc:
            import traceback
            ok = False
            print(f"FAIL {job.key}: {exc}")
            traceback.print_exc()
            continue
        m = r["metrics"][r["primary"]]
        finite = all(math.isfinite(m[k]) for k in ("macro_f2", "gmean", "macro_recall", "macro_specificity"))
        files = sorted(p.name for p in job.run_dir.iterdir())
        ok &= finite and "result.json" in files
        print(f"ok   {job.key:58s} {r['primary']:9s} F2={m['macro_f2']:.3f} synth={r['n_synth']:4d}/{r['n_synth_target']:<4d} "
              f"gate={r['gate'].get('fallback_cycles', '-')}/{r['gate'].get('class_cycles', '-')} "
              f"{time.time() - t0:5.1f}s  {files}")
    ok &= hashes("T_smoke")
    print("SMOKE", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "units"
    raise SystemExit(0 if {"units": units, "smoke": smoke}[what]() else 1)
