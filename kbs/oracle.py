"""kbs.oracle - train the reference classifiers of the label audit (E1, S5; Reviewer 3, comment 1).

    python -m kbs.oracle train [--datasets fashion_mnist cifar10 ...]
    python -m kbs.oracle status

Per dataset: three ImageNet-pretrained ResNet-18 members (``casestudy.eval.oracle.RefNet``),
fine-tuned on every official training image of the three classes (development images included,
images that also occur in a test split excluded), with class-weighted cross-entropy and standard
augmentation. Each member and the ensemble are scored on every official test image of the three
classes (OCTMNIST: validation and test splits); that accuracy is the noise floor reported next to
every agreement rate. The plan admits a dataset to the gate decision only if the ensemble reaches
0.85 accuracy.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from casestudy.data.augment import random_affine
from casestudy.data.transforms import to_normalized_tensor
from casestudy.eval.oracle import Oracle, RefNet, oracle_dir
from casestudy.protocol_v2 import _raw, make_config, regime_spec
from casestudy.reproducibility import set_seed

DATASETS = {  # dataset: (an anchor regime that fixes the class trio and order, epochs)
    "fashion_mnist": ("R1_confusable", 8),
    "cifar10": ("C2_cifar_paper", 8),
    "stl10": ("S2_stl_mod", 25),
    "octmnist": ("OCT2_pathology_mod", 4),
    "gtsrb": ("G2_signs_mod", 10),
}
MEMBERS, BATCH, LR, WD = 3, 128, 3e-4, 0.05


def _data(dataset: str, cfg):
    spec = regime_spec(DATASETS[dataset][0], cfg)
    raw = _raw(dataset, spec, cfg)
    to_t = lambda x: to_normalized_tensor(x, spec.channels, cfg.image_size, cfg.normalize_range)
    x_tr = to_t(np.concatenate([raw["res_x"], raw["dev_x"]]))
    y_tr = torch.from_numpy(np.concatenate([raw["res_y"], raw["dev_y"]])).long()
    return spec, x_tr, y_tr, to_t(raw["full_x"]), torch.from_numpy(raw["full_y"]).long()


@torch.no_grad()
def _scores(model_or_oracle, x: torch.Tensor, device: str) -> torch.Tensor:
    if isinstance(model_or_oracle, Oracle):
        return model_or_oracle(x)
    model_or_oracle.eval()
    out = [torch.softmax(model_or_oracle(x[i:i + 512].to(device)), 1).cpu() for i in range(0, len(x), 512)]
    return torch.cat(out)


def _report(probs: torch.Tensor, y: torch.Tensor, K: int) -> dict:
    pred = probs.argmax(1)
    recall = [float((pred[y == c] == c).float().mean()) for c in range(K)]
    return {"accuracy": float((pred == y).float().mean()), "balanced_accuracy": float(np.mean(recall)),
            "recall_per_class": recall}


def train_dataset(dataset: str, device: str) -> dict:
    cfg = make_config()
    spec, x_tr, y_tr, x_te, y_te = _data(dataset, cfg)
    K, epochs = len(spec.selected_classes), DATASETS[dataset][1]
    hflip = spec.loader_key != "gtsrb"
    counts = torch.bincount(y_tr, minlength=K).float()
    weights = (counts.sum() / (K * counts)).to(device)
    out_dir = oracle_dir(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    members, reports = [], []
    t_start = time.time()
    for k in range(MEMBERS):
        set_seed(1000 + k)
        model = RefNet(K, pretrained=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        steps = epochs * ((len(y_tr) + BATCH - 1) // BATCH)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
        ce = nn.CrossEntropyLoss(weight=weights)
        g_cpu = torch.Generator().manual_seed(2000 + k)
        g_dev = torch.Generator(device=device).manual_seed(3000 + k)
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(y_tr), generator=g_cpu)
            for i in range(0, len(perm), BATCH):
                b = perm[i:i + BATCH]
                xb, yb = x_tr[b].to(device), y_tr[b].to(device)
                with torch.no_grad():
                    xb = random_affine(xb, g_dev, hflip=hflip)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss = ce(model(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        model.eval()
        name = f"member{k}.pt"
        torch.save(model.state_dict(), out_dir / name)
        members.append(name)
        reports.append(_report(_scores(model, x_te, device), y_te, K))
        print(f"  {dataset} member {k}: accuracy {reports[-1]['accuracy']:.4f}  "
              f"balanced {reports[-1]['balanced_accuracy']:.4f}  ({time.time() - t_start:.0f} s)", flush=True)
    ens = Oracle([m.eval() for m in [_reload(out_dir / n, K, device) for n in members]], device)
    ens_report = _report(_scores(ens, x_te, device), y_te, K)
    summary = {"dataset": dataset, "classes": list(spec.selected_classes), "class_names": list(spec.class_names),
               "num_classes": K, "members": members, "member_reports": reports, "ensemble": ens_report,
               "n_train": int(len(y_tr)), "train_counts": counts.int().tolist(), "n_test": int(len(y_te)),
               "test_counts": torch.bincount(y_te, minlength=K).tolist(), "epochs": epochs, "batch": BATCH,
               "lr": LR, "weight_decay": WD, "hflip": hflip, "eligible_for_gate": ens_report["accuracy"] >= 0.85,
               "seconds": round(time.time() - t_start, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{dataset}: ensemble accuracy {ens_report['accuracy']:.4f}, balanced {ens_report['balanced_accuracy']:.4f}, "
          f"recall per class {np.round(ens_report['recall_per_class'], 3).tolist()}", flush=True)
    return summary


def _reload(path, K, device):
    m = RefNet(K, pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.to(device)


def status() -> None:
    for ds in DATASETS:
        f = oracle_dir(ds) / "summary.json"
        if not f.exists():
            print(f"{ds:14s} not trained")
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        e = s["ensemble"]
        print(f"{ds:14s} accuracy {e['accuracy']:.4f}  balanced {e['balanced_accuracy']:.4f}  "
              f"recall {np.round(e['recall_per_class'], 3).tolist()}  eligible {s['eligible_for_gate']}  ({s['finished']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--datasets", nargs="*", default=list(DATASETS))
    sub.add_parser("status")
    args = ap.parse_args()
    if args.cmd == "status":
        status()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds in args.datasets:
        print(f"== {ds} ({time.strftime('%H:%M:%S')})", flush=True)
        train_dataset(ds, device)
    status()


if __name__ == "__main__":
    main()
