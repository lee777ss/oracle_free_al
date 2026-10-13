"""kbs.export_config - the configuration table of the revised paper (R1.4), read from the code.

    python -m kbs.export_config            # print
    python -m kbs.export_config --write    # also write src/generated/tbl_config.tex and tbl_config.json

Every value comes from the classes and the configuration that the runs use, and every parameter
count from instantiating the networks, so the table cannot disagree with the code. The proposed
method's fallback and co-training entries come from saved_kbs/frozen.json once the pilot has fixed them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch.nn as nn

from kbs import PAPER_DIR
from casestudy.classifiers.cnn import SmallCNN
from casestudy.classifiers.resnet import resnet32
from casestudy.methods._backbones import Generator
from casestudy.methods._bagan_common import AutoEncoder
from casestudy.methods.backbone_cache import BASE_BETAS, BASE_LR, LAMBDA_GP, N_CRITIC
from casestudy.methods.m04_dcgan import _DCDiscriminator
from casestudy.methods.m05_cgan import _CondDiscriminator
from casestudy.methods.m06_acgan import _ACDiscriminator
from casestudy.methods.m07_wgangp import _Critic
from casestudy.methods.m08_bagan import _NPlus1Discriminator
from casestudy.methods.m09_bagan_gp import _AuxCritic
from casestudy.methods.m10_ab8c import _CoCNN
from casestudy.methods.m10v2_ab8c import Ab8cV2
from casestudy.protocol_v2 import make_config, save_root

PAPER = PAPER_DIR


def n(m: nn.Module) -> str:
    return f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M"


def rows():
    cfg = make_config()
    P = Ab8cV2
    frozen_path = save_root() / "frozen.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    fb = {"top_confidence": "fill the batch with the most confident remaining candidates (as submitted)",
          "regenerate": "draw new candidates; the gate is never bypassed"}.get((frozen or {}).get("fallback"),
                                                                              "set by the pilot (Part 5.1)")
    co = "on" if (frozen or {}).get("cotrain_g") else ("off" if frozen else "set by the pilot (Part 5.1)")
    d, K = cfg.latent_dim, 3
    R = []
    R += [("Protocol", "seeds", "42-61 (anchors, selection study, domain); 42-51 (grid, sweeps)"),
          ("", "hold-out", "stratified 10% of the pool, split before any training"),
          ("", "augmentation target", r"$\min(N_{\max}, rN_c)$ per minority class; $r$ in the data table"),
          ("", "image size", f"{cfg.image_size} x {cfg.image_size}, scaled to [-1, 1]")]
    R += [("Downstream classifier", "Fashion-MNIST, STL-10, OCTMNIST, GTSRB",
           f"4-block CNN (16-32-64-128), FC 2048-64-3, dropout {cfg.dropout_rate}; {n(SmallCNN(1, K))} (grey) / {n(SmallCNN(3, K))} (colour)"),
          ("", "CIFAR-10", f"ResNet-32; {n(resnet32(in_channels=3, num_classes=K))}"),
          ("", "optimiser", f"Adam, learning rate {cfg.learning_rate:g}, weight decay {cfg.weight_decay:g}, batch {cfg.batch_size}"),
          ("", "training", f"up to {cfg.clf_max_epochs} epochs, early stopping after {cfg.clf_patience} without improvement"),
          ("", "loss and selection", "class-weighted cross-entropy; best hold-out macro-F2 (plain CE for the reference row)")]
    R += [("Autoencoder", "encoder / decoder", f"4 conv blocks (64-128-256-512) + linear to $d$ / generator architecture; {n(AutoEncoder(1, d))} (grey)"),
          ("", "training", f"MSE, Adam 1e-3, betas (0.5, 0.999), {cfg.gan_epochs} epochs, batch {cfg.batch_size}"),
          ("", "latent dimension $d$", f"{d}")]
    R += [("Generator", "architecture", f"linear $d$ to 512x4x4, 4 transposed-conv blocks (512-256-128-64), tanh; {n(Generator(d, 1))} (grey)"),
          ("Backbone warm-up", "critic", f"conv trunk without BatchNorm, score head and auxiliary class head; {n(_AuxCritic(1, K))} (grey)"),
          ("", "objective", f"WGAN-GP, $\\lambda_{{gp}}$ = {LAMBDA_GP:g}, $n_{{critic}}$ = {N_CRITIC}, auxiliary CE"),
          ("", "optimiser", f"Adam {BASE_LR:g}, betas {BASE_BETAS}, {cfg.gan_epochs} epochs")]
    R += [("Proposed: acquisition", "B / R / $\\tau$", f"{P.samples_per_cycle} / {P.pool_factor} / {P.min_confidence}"),
          ("", "seeding", f"latent SMOTE between two encoded real images, jitter $\\eta$ = {P.latent_smote_noise}"),
          ("", "cycles $T$", r"$\lceil \max_c \Delta_c / B \rceil$"),
          ("", "fallback when fewer than B pass", str(fb)),
          ("Proposed: internal classifier", "architecture", f"3 conv blocks (32-64-128) with BatchNorm, global pooling, linear; {n(_CoCNN(1, K))} (grey)"),
          ("", "training", f"Adam {P.c_lr:g}; {P.c_warmup_epochs} warm-up epochs on the training portion; {P.c_epochs_per_cycle} epoch per cycle on real + admitted"),
          ("Proposed: co-training", "generator co-training", co),
          ("", "critic / generator steps per cycle", f"{P.k_d} / {P.k_g}"),
          ("", "fine-tuning learning rates", f"{BASE_LR * P.lr_d_factor:g} / {BASE_LR * P.lr_g_factor:g}"),
          ("", "critic weights", f"aux real {P.lambda_aux_real}, aux synthetic {P.lambda_aux_synth}"),
          ("", "generator weights", f"adversarial {P.lambda_adv}, classifier {P.lambda_cls}, anchor {P.lambda_anchor}, aux {P.lambda_aux_g}")]
    R += [("DCGAN", "per minority class", f"BCE; Adam 2e-4, betas (0.5, 0.999); critic {n(_DCDiscriminator(1))}"),
          ("cGAN", "label-conditioned", f"BCE, one-hot planes; Adam 2e-4; discriminator {n(_CondDiscriminator(1, K))}"),
          ("ACGAN", "auxiliary head", f"BCE + CE; Adam 2e-4; discriminator {n(_ACDiscriminator(1, K))}"),
          ("WGAN-GP", "per minority class", f"$\\lambda_{{gp}}$ 10, $n_{{critic}}$ 5; Adam 1e-4, betas (0, 0.9); critic {n(_Critic(1))}"),
          ("BAGAN", "(n+1)-class discriminator", f"AE warm start; Adam 2e-4; label smoothing 0.1 as submitted (without it, appendix row); {n(_NPlus1Discriminator(1, K))}"),
          ("BAGAN-GP", "shared backbone", "the backbone above, sampled from the class prior"),
          ("All GANs", "budget", f"{cfg.gan_epochs} epochs, batch {cfg.batch_size}, latent dimension {d}"),
          ("SMOTE family", "pixel space", r"$k = \min(5,\ n_{\min} - 1)$ neighbours; Borderline-SMOTE kind 1"),
          ("Standard augmentation", "per image", "shift up to 6 px with reflection, rotation up to 10 degrees, scale 0.9-1.1, horizontal flip (not GTSRB)")]
    return R


def to_latex(R) -> str:
    out = [r"% Generated by kbs/export_config.py - do not edit by hand.",
           r"\providecommand{\revtable}{}  % the manuscript colours tables added in the revision",
           r"\begin{table}[htbp]", r"\revtable", r"\centering",
           r"\caption{Configuration of every method and of the protocol. Parameter counts are for greyscale input.}",
           r"\label{tbl:config}",
           r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{\scalebox{0.62}{",   # fit if still too wide
           r"\begin{tabular}{lll}", r"\toprule",
           r"Component & Setting & Value \\", r"\midrule"]
    esc = lambda s: re.sub(r"(?<!\\)%", r"\\%", str(s))         # a bare % would comment out the rest of the row
    for comp, setting, value in R:
        out.append(f"{esc(comp)} & {esc(setting)} & {esc(value)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}}}", r"\end{table}", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    R = rows()
    for comp, setting, value in R:
        print(f"{comp:30s} {setting:38s} {value}")
    if args.write:
        out = PAPER / "src" / "generated"
        out.mkdir(parents=True, exist_ok=True)
        (out / "tbl_config.tex").write_text(to_latex(R), encoding="utf-8")
        (out / "tbl_config.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
        print("written:", out / "tbl_config.tex")


if __name__ == "__main__":
    main()
