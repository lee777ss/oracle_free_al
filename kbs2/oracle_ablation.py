"""kbs2.oracle_ablation — reference classifier for the label agreement of Table 3 panel B.

    python -m kbs2.oracle_ablation

Same recipe as the reference classifiers of Appendix A8 (``kbs.oracle.train_dataset``): three
ImageNet-initialised ResNet-18 members fine-tuned for 8 epochs on every official Fashion-MNIST training
image of the three classes (here T-shirt/top, Pullover, Dress, the classes of Table 3), class-weighted
cross-entropy, standard augmentation; scored on every official test image of the three classes.
Output: ``saved_kbs2/oracle/fashion_mnist_A1/``. Inference only afterwards (``load``).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = Path(ROOT) / "saved_kbs2" / "oracle" / "fashion_mnist_A1"
CLASSES = [0, 2, 3]
MEMBERS, EPOCHS, BATCH, LR, WD = 3, 8, 128, 3e-4, 0.05


def _data(cfg):
    from casestudy.data.loaders import _load_raw, _remap
    from casestudy.data.transforms import to_normalized_tensor
    out = []
    for train in (True, False):
        x, y = _load_raw("fmnist", train, cfg.data_root)
        keep = np.isin(y, CLASSES)
        out.append((to_normalized_tensor(x[keep], 1, cfg.image_size, cfg.normalize_range),
                    torch.from_numpy(_remap(y[keep], CLASSES)).long()))
    return out


def train() -> dict:
    from casestudy.data.augment import random_affine
    from casestudy.eval.oracle import Oracle, RefNet
    from casestudy.reproducibility import set_seed
    from kbs.oracle import _reload, _report, _scores
    from kbs2 import protocol as P

    cfg = P.submitted_config()
    device = cfg.resolve_device()
    (x_tr, y_tr), (x_te, y_te) = _data(cfg)
    K = len(CLASSES)
    counts = torch.bincount(y_tr, minlength=K).float()
    weights = (counts.sum() / (K * counts)).to(device)
    OUT.mkdir(parents=True, exist_ok=True)
    members, reports, t_start = [], [], time.time()
    for k in range(MEMBERS):
        set_seed(1000 + k)
        model = RefNet(K, pretrained=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        steps = EPOCHS * ((len(y_tr) + BATCH - 1) // BATCH)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
        ce = nn.CrossEntropyLoss(weight=weights)
        g_cpu = torch.Generator().manual_seed(2000 + k)
        g_dev = torch.Generator(device=device).manual_seed(3000 + k)
        model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(len(y_tr), generator=g_cpu)
            for i in range(0, len(perm), BATCH):
                b = perm[i:i + BATCH]
                xb, yb = x_tr[b].to(device), y_tr[b].to(device)
                with torch.no_grad():
                    xb = random_affine(xb, g_dev, hflip=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss = ce(model(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        model.eval()
        name = f"member{k}.pt"
        torch.save(model.state_dict(), OUT / name)
        members.append(name)
        reports.append(_report(_scores(model, x_te, device), y_te, K))
        print(f"  member {k}: accuracy {reports[-1]['accuracy']:.4f} ({time.time() - t_start:.0f} s)", flush=True)
    ens = Oracle([_reload(OUT / n, K, device) for n in members], device)
    ens_report = _report(_scores(ens, x_te, device), y_te, K)
    summary = {"dataset": "fashion_mnist", "classes": CLASSES, "class_names": ["T-shirt/top", "Pullover", "Dress"],
               "num_classes": K, "members": members, "member_reports": reports, "ensemble": ens_report,
               "n_train": int(len(y_tr)), "train_counts": counts.int().tolist(), "n_test": int(len(y_te)),
               "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WD, "hflip": True,
               "seconds": round(time.time() - t_start, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"ensemble accuracy {ens_report['accuracy']:.4f}, recall per class "
          f"{np.round(ens_report['recall_per_class'], 3).tolist()}", flush=True)
    return summary


def load(device: str):
    from casestudy.eval.oracle import Oracle, RefNet
    info = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    members = []
    for name in info["members"]:
        m = RefNet(info["num_classes"], pretrained=False)
        m.load_state_dict(torch.load(OUT / name, map_location="cpu", weights_only=True))
        members.append(m.to(device).eval())
    return Oracle(members, device)


if __name__ == "__main__":
    train()
