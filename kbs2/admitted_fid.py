"""kbs2.admitted_fid - FID of the images the Proposed method admitted on the ablation regime (Table 4, R1.9).

    python -m kbs2.admitted_fid      # writes saved_kbs2/analysis/AB/admitted_fid.csv

Authors 13 Oct 2026: the co-trained generator of the panel A Proposed runs was not saved, so the Proposed row of
Table 4 reports the FID of the images it admitted instead: the synthetic sets of experiment AB (Table 3 panel B
data) for the proposed rule with co-training on (gate_diverse_cot), seeds 42-56, 684 images per minority class.
Same FID code (ablation.fid / ablation.shift_metrics) and the same per-seed real reference statistics
(saved/fid_ref_class_<c>_133J_s<seed>.npz) as the backbone rows, whose fake side used 2000 generated images per
class; a smaller sample inflates FID, so this row is conservative relative to the others.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ablation.fid import _load_ref_stats, _stats_from_images  # noqa: E402
from ablation.shift_metrics import _frechet_distance, _get_inception  # noqa: E402

SYNTH = os.path.join(ROOT, "saved_kbs2", "AB", "synth", "m10v2_ab8c_gate_diverse_cot__A1_ablation__s{seed}.pt")
REF = os.path.join(ROOT, "saved", "fid_ref_class_{c}_133J_s{seed}.npz")
OUT = os.path.join(ROOT, "saved_kbs2", "analysis", "AB", "admitted_fid.csv")


def main(device: str = "cpu") -> None:
    inception = _get_inception(device)
    rows = []
    for seed in range(42, 57):
        rec = torch.load(SYNTH.format(seed=seed), weights_only=False)  # written by kbs2.runner.generate
        X, y = rec["X"], rec["y"]
        for c in (1, 2):
            mu_r, sig_r, n_r = _load_ref_stats(REF.format(c=c, seed=seed))
            mu_f, sig_f, n_f = _stats_from_images(X[y == c], inception, device, 32)
            fid = float(_frechet_distance(mu_r, sig_r, mu_f, sig_f))
            rows.append({"seed": seed, "class": c, "fid": fid, "n_fake": n_f, "n_real": n_r})
            print(seed, c, round(fid, 3), n_f, flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cpu")
