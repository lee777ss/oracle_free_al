"""kbs2.ablation_samples - Figure 2 of the revision: samples along the ablation chain at native resolution (R1.9).

    python -m kbs2.ablation_samples      # writes Journal_paper_02_kbs_revise2/figures/final/ablation_samples.png

Authors 13 Oct 2026: Figure 2 stays the ablation figure, rebuilt on the basis of Table 3 panel A, seed 44 (as
submitted). Rows (authors' order): real images, the autoencoder of each backbone, then the generator of each
backbone (BAGAN: Variants 1-2; BAGAN-GP: Variants 3-4 and the proposed method's starting point), from the sample
dumps of the panel A runs (saved/samples_133J_s44_<backbone>.npz, three fixed images per class).
Every image at its native 64 x 64 pixels, enlarged by pixel repetition only.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Journal_paper_02_kbs_revise2", "figures", "final", "ablation_samples.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)  # public release: create the output folder
SEED, PER, ZOOM, GAP, BIG = 44, 3, 3, 4, 28
CLASSES = ["T-shirt/top (majority)", "Pullover", "Dress"]


def u8(x) -> np.ndarray:
    a = ((np.clip(np.asarray(x, dtype=np.float32), -1, 1) + 1) * 127.5).round().astype(np.uint8)
    return np.repeat(a, 3, axis=0).transpose(1, 2, 0) if a.shape[0] == 1 else a.transpose(1, 2, 0)


def by_class(arr: np.ndarray) -> list:
    """Sample dumps hold PER images per class, classes in order 0, 1, 2."""
    return [[u8(arr[c * PER + j]) for j in range(PER)] for c in range(3)]


def main() -> None:
    bg = np.load(os.path.join(ROOT, "saved", f"samples_133J_s{SEED}_bagan.npz"))
    gp = np.load(os.path.join(ROOT, "saved", f"samples_133J_s{SEED}_bagan_gp.npz"))
    if not np.array_equal(bg["real"], gp["real"]) or list(bg["classes"]) != [0, 1, 2]:
        raise SystemExit("unexpected sample dumps")
    # Authors 13 Oct (second revision of R1.9): this order and these names; no Proposed row.
    # Authors 13 Oct (third revision): the names of the submitted figure; Autoencoder = BAGAN backbone's
    # autoencoder, Generative model = BAGAN generator, Proposed = BAGAN-GP generator before the acquisition loop.
    rows = [("Real", by_class(bg["real"])),
            ("Autoencoder", by_class(bg["ae"])),
            ("Generative model", by_class(bg["gen"])),
            ("Proposed", by_class(gp["gen"]))]
    font = ImageFont.truetype("arial.ttf", 30)
    cell, label_w, head_h = 64 * ZOOM, 470, 70
    W = label_w + 3 * (PER * (cell + GAP)) + 2 * BIG
    H = head_h + len(rows) * (cell + GAP)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for c, name in enumerate(CLASSES):
        x0 = label_w + c * (PER * (cell + GAP) + BIG)
        d.text((x0 + PER * (cell + GAP) // 2, head_h // 2), name, fill="black", font=font, anchor="mm")
    for r, (label, cols) in enumerate(rows):
        y0 = head_h + r * (cell + GAP)
        d.text((label_w - 20, y0 + cell // 2), label, fill="black", font=font, anchor="rm")
        for c in range(3):
            x0 = label_w + c * (PER * (cell + GAP) + BIG)
            if cols[c] is None:
                d.text((x0 + PER * (cell + GAP) // 2, y0 + cell // 2), "not generated", fill="gray",
                       font=font, anchor="mm")
                continue
            for j, a in enumerate(cols[c]):
                img.paste(Image.fromarray(a).resize((cell, cell), Image.NEAREST), (x0 + j * (cell + GAP), y0))
    img.save(OUT, optimize=True)
    print(OUT, img.size)


if __name__ == "__main__":
    main()
