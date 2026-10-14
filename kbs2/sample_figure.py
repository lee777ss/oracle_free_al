"""kbs2.sample_figure - Figure A5 of the revision: case-study samples at native resolution (R1.9).

    python -m kbs2.sample_figure      # writes Journal_paper_02_kbs_revise2/figures/final/samples_casestudy.png

Authors 13 Oct 2026 (second decision): an appendix figure, Fashion-MNIST only, the two minority classes, eight
images per class. Seed 50 (authors' choice; Figure 2 uses seed 44). Figure 2 stays the ablation figure
(kbs2.ablation_samples).
Rule: for each dataset, the first eight images of each minority class, in the order in which they are
stored, of the run (seed 50) behind Table 6 (proposed method: E1; generative rivals: E8, arm W runs), beside
the first eight real images of the class in the training portion.
Authors 14 Oct 2026: the three curated "Proposed (brighter)" rows are removed (one row per method, seven rows);
the seed is not mentioned in the manuscript or the reply.
Each image is shown at its native 64 x 64
pixels, enlarged only by pixel repetition (no interpolation).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kbs2 import protocol as P  # noqa: E402
from kbs2.runner import load_regime, split  # noqa: E402

OUT = os.environ.get("KBS2_SAMPLE_OUT") or os.path.join(ROOT, "Journal_paper_02_kbs_revise2", "figures", "final", "samples_casestudy.png")
DATASETS = ("fashion_mnist",)
SEED, N, ZOOM, GAP, BIG = int(os.environ.get("KBS2_SAMPLE_SEED", 50)), 8, 3, 4, 24
ROWS = [("Real", None, None), ("Proposed", "E1", "m10_ab8c"), ("DCGAN", "E8", "m04_dcgan"),
        ("cGAN", "E8", "m05_cgan"), ("ACGAN", "E8", "m06_acgan"), ("WGAN-GP", "E8", "m07_wgangp"),
        ("BAGAN", "E8", "m08_bagan")]
MINORITY = {"fashion_mnist": ["Coat", "Shirt"], "cifar10": ["automobile", "ship"], "stl10": ["dog", "deer"]}
DS_NAME = {"fashion_mnist": "Fashion-MNIST", "cifar10": "CIFAR-10", "stl10": "STL-10"}


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensor [C, H, W] -> uint8 [H, W, 3]."""
    a = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
    a = np.repeat(a, 3, axis=0) if a.shape[0] == 1 else a
    return a.transpose(1, 2, 0)


def first_n(X: torch.Tensor, y: torch.Tensor, c: int, band=None, skip: int = 0) -> list:
    """First N images of class c in stored order; with band=(lo, hi), only those whose mean intensity in [0, 1] lies in it."""
    idx = torch.nonzero(y == c).flatten()
    if band is not None:
        mu = ((X[idx] + 1) / 2).mean(dim=(1, 2, 3))
        idx = idx[(mu >= band[0]) & (mu <= band[1])]
    idx = idx[skip * N:(skip + 1) * N]
    if idx.numel() < N:
        raise SystemExit(f"fewer than {N} images of class {c}")
    return [to_uint8(X[i]) for i in idx]


def main() -> None:
    font = ImageFont.truetype("arial.ttf", 34)
    bold = ImageFont.truetype("arialbd.ttf", 38)
    cell = 64 * ZOOM
    label_w, head_h = 300, 60
    block_w = label_w + 2 * (N * (cell + GAP)) + BIG
    blocks = []
    for ds, (rid, _r) in P.ANCHORS.items():
        if ds not in DATASETS:
            continue
        D = load_regime(rid)
        tr, _ho = split(D["pool"], D["config"])
        Xtr, ytr = D["pool"]["images"][tr], D["pool"]["labels"][tr]
        # brighter row: mean intensity within the 10th-90th percentile of the real class (rule fixed before looking)
        bands = {}
        for k in (1, 2):
            mu = ((Xtr[ytr == k] + 1) / 2).mean(dim=(1, 2, 3))
            bands[k] = tuple(torch.quantile(mu, torch.tensor([.1, .9])).tolist())
        h = head_h * 2 + len(ROWS) * (cell + GAP)
        img = Image.new("RGB", (block_w, h), "white")
        d = ImageDraw.Draw(img)
        d.text((0, 8), DS_NAME[ds], fill="black", font=bold)
        for k, name in enumerate(MINORITY[ds]):
            x0 = label_w + k * (N * (cell + GAP) + BIG)
            d.text((x0 + (N * (cell + GAP)) // 2, head_h + 30), name, fill="black", font=font, anchor="mm")
        for r, (label, exp, method, *extra) in enumerate(ROWS):
            skip = extra[0] if extra else 0
            bright = "brighter" in label or bool(extra)
            y0 = head_h * 2 + r * (cell + GAP)
            d.multiline_text((label_w - 20, y0 + cell // 2), label, fill="black", font=font, anchor="rm", align="right")
            if exp is None:
                X, y = Xtr, ytr
            else:
                rec = torch.load(os.path.join(ROOT, "saved_kbs2", exp, "synth", f"{method}__{rid}__s{SEED}.pt"),
                                 weights_only=False)  # written by kbs2.runner.generate
                X, y = rec["X"], rec["y"]
            for k in (1, 2):
                x0 = label_w + (k - 1) * (N * (cell + GAP) + BIG)
                band = bands[k] if bright else None
                for j, a in enumerate(first_n(X, y, k, band, skip)):
                    tile = Image.fromarray(a).resize((cell, cell), Image.NEAREST)
                    img.paste(tile, (x0 + j * (cell + GAP), y0))
        blocks.append(img)
    total = Image.new("RGB", (block_w, sum(b.height for b in blocks) + 40 * (len(blocks) - 1)), "white")
    y = 0
    for b in blocks:
        total.paste(b, (0, y))
        y += b.height + 40
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    total.save(OUT, optimize=True)
    print(OUT, total.size)


if __name__ == "__main__":
    main()
