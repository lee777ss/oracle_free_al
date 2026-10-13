"""kbs.sample_figure - uncurated samples at the generators' native resolution (R1.9).

    python -m kbs.sample_figure --all             # one figure per dataset, seed 42
    python -m kbs.sample_figure --all --check     # report what is missing; write nothing
    python -m kbs.sample_figure --regimes R1_confusable --seed 43

Each figure is assembled as a pixel array and written as a 300 dpi PNG in which every 64 x 64
sample occupies exactly 128 x 128 pixels (nearest-neighbour factor 2), so no sample is resampled.
For each minority class the rows are: real training images of the run; images the proposed
method admitted in its first and in its last full acquisition cycle (the final cycle may admit only
a remainder of the target), and candidates its gate rejected in that cycle, in candidate order
(``gallery.npz``); the first images of six generative rivals; and the nearest real training image
(pixel space) of each of those admitted images, as a check against memorisation. An admitted image that failed the gate (it entered through the fallback) gets
a red frame, drawn in the gap around the tile so that no sample pixel changes. No image is chosen
by looking at it: every row takes the first eight in a fixed order.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from kbs import PAPER_DIR
from kbs.audit import TAU
from casestudy.data.transforms import to_normalized_tensor
from casestudy.protocol_v2 import LABELS, REGIMES_V2, Job, _to_uint8, get_regime, make_config, regime_spec, sample_pool

FIGS = PAPER_DIR / "figures" / "final"
FIGURE_REGIMES = {"fashion_mnist": "R1_confusable", "cifar10": "C2_cifar_paper", "stl10": "S2_stl_mod",
                  "octmnist": "OCT2_pathology_mod", "gtsrb": "G2_signs_mod"}
RIVALS = ["m09_bagan_gp", "m08_bagan_ls", "m06_acgan", "m05_cgan", "m07_wgangp", "m04_dcgan"]
N, SCALE, GAP, LABEL_W, HEAD_H, BLOCK_GAP, DPI = 8, 2, 6, 720, 56, 28, 300
FRAME = 3                                           # px; fits in the gap between tiles


def _stage(regime: str) -> str:
    return "E6_domain" if REGIMES_V2[regime]["role"] == "domain" else "E3_anchors"


def _run(regime: str, method: str, seed: int) -> Path:
    variant = "frozen" if method == "p_proposed" else ""
    return Job(_stage(regime), regime, method, seed, variant=variant).run_dir


def training_images(regime: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """The training portion of the run's pool, as the methods saw it (64 x 64, uint8)."""
    cfg = make_config()
    rd = get_regime(regime, cfg)
    idx = sample_pool(rd.res_y, rd.spec.majority_cap, rd.rho, seed)
    X = to_normalized_tensor(rd.res_x[idx], rd.spec.channels, cfg.image_size, cfg.normalize_range)
    y = rd.res_y[idx]
    tr, _ = next(StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed).split(np.zeros(len(y)), y))
    return _to_uint8(X[tr]), y[tr]


def missing(regime: str, seed: int) -> List[str]:
    out = []
    for m in ["p_proposed"] + RIVALS:
        d = _run(regime, m, seed)
        need = ["result.json", "synth.npz"] + (["gallery.npz"] if m == "p_proposed" else [])
        out += [f"{d}/{f}" for f in need if not (d / f).exists()]
    return out


Row = Tuple[str, List[np.ndarray], List[bool]]     # label, images (C x H x W uint8), red frame per image


def _gallery_row(gal, sel, what: str) -> Row:
    x, p, pred, c = gal["x"][sel][:N], gal["p_c"][sel][:N], gal["pred"][sel][:N], gal["cls"][sel][:N]
    label = f"{what} (p {p.min():.2f}-{p.max():.2f})" if len(p) else what
    failed = (pred != c) | (p < TAU)
    return label, list(x), [bool(f) for f in failed] if what.endswith("admitted") else [False] * len(x)


def rows_for(regime: str, seed: int) -> List[Tuple[str, List[Row]]]:
    """[(class header, [(row label, images, frames)])] for each minority class."""
    names = list(regime_spec(regime, make_config()).class_names)
    xr, yr = training_images(regime, seed)
    gal = np.load(_run(regime, "p_proposed", seed) / "gallery.npz")
    synth = {m: np.load(_run(regime, m, seed) / "synth.npz") for m in RIVALS}
    blocks = []
    for c in range(1, len(names)):
        real_c = xr[yr == c]
        sel = gal["cls"] == c
        n_cyc = int(gal["cycle"][sel].max()) + 1
        full = [cy for cy in np.unique(gal["cycle"][sel & (gal["kind"] == 0)])
                if (sel & (gal["kind"] == 0) & (gal["cycle"] == cy)).sum() >= N]   # the final cycle may admit a remainder
        first = int(gal["cycle"][sel].min())
        last = int(max(full)) if full else n_cyc - 1
        adm_first = _gallery_row(gal, sel & (gal["kind"] == 0) & (gal["cycle"] == first),
                                 f"Proposed, cycle {first + 1} of {n_cyc}: admitted")
        adm_last = _gallery_row(gal, sel & (gal["kind"] == 0) & (gal["cycle"] == last),
                                f"Proposed, cycle {last + 1} of {n_cyc}: admitted")
        rej_last = _gallery_row(gal, sel & (gal["kind"] == 1) & (gal["cycle"] == last),
                                f"Proposed, cycle {last + 1} of {n_cyc}: rejected")
        flat_real = real_c.reshape(len(real_c), -1).astype(np.float32)
        nn_idx = [int(np.argmin(((flat_real - a.reshape(1, -1).astype(np.float32)) ** 2).sum(1))) for a in adm_last[1]]
        rows = [(f"Real (training, {len(real_c)} in all)", list(real_c[:N]), [False] * N),
                adm_first, adm_last, rej_last]
        for m in RIVALS:
            z = synth[m]
            rows.append((LABELS[m], list(z["x"][z["y"] == c][:N]), [False] * N))
        rows.append((f"Nearest real to each cycle-{last + 1} admitted", [real_c[i] for i in nn_idx], [False] * N))
        blocks.append((f"{names[c]} (minority class)", rows))
    return blocks


def render(blocks, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    side = 64 * SCALE
    width = LABEL_W + N * (side + GAP)
    height = sum(HEAD_H + len(rows) * (side + GAP) + BLOCK_GAP for _, rows in blocks)
    canvas = np.full((height, width, 3), 255, np.uint8)
    labels = []
    y = 0
    for head, rows in blocks:
        labels.append((8, y + HEAD_H // 2, head, True))
        y += HEAD_H
        for label, imgs, frames in rows:
            labels.append((8, y + side // 2, label, False))
            for k, (img, framed) in enumerate(zip(imgs[:N], frames)):
                tile = np.repeat(np.repeat(np.moveaxis(img, 0, -1), SCALE, 0), SCALE, 1)
                if tile.shape[-1] == 1:
                    tile = np.repeat(tile, 3, -1)
                x0 = LABEL_W + k * (side + GAP)
                if framed:                      # around the tile, never on it
                    canvas[y - FRAME:y + side + FRAME, x0 - FRAME:x0 + side + FRAME] = (200, 0, 0)
                canvas[y:y + side, x0:x0 + side] = tile
            y += side + GAP
        y += BLOCK_GAP
    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    fig.figimage(canvas, 0, 0, origin="upper", resize=False)
    for x, yy, text, bold in labels:
        fig.text(x / width, 1 - yy / height, text, fontsize=6.5 if not bold else 7, va="center",
                 fontweight="bold" if bold else "normal")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="one figure per dataset (anchors; OCTMNIST and GTSRB at rho 0.10)")
    ap.add_argument("--regimes", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check", action="store_true", help="report missing inputs; write nothing")
    ap.add_argument("--out", type=Path, default=FIGS)
    args = ap.parse_args()
    regimes = list(FIGURE_REGIMES.values()) if args.all else args.regimes
    for regime in regimes:
        ds = REGIMES_V2[regime]["dataset"]
        miss = missing(regime, args.seed)
        if miss:
            print(f"{regime:22s} waiting for {len(miss)} file(s), e.g. {miss[0]}")
            continue
        path = args.out / f"fig_samples_{ds}.png"
        if args.check:
            print(f"{regime:22s} complete (not written: --check)")
            continue
        render(rows_for(regime, args.seed), path)      # dataset, seed and "uncurated" go into the LaTeX caption
        print(f"{regime:22s} written: {path}")


if __name__ == "__main__":
    main()
