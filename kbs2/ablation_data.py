"""kbs2.ablation_data — the data of Table 3 (panel A) in the case-study format, for panel B (R1.6, R3.2).

Panel A was produced by the ``ablation`` package: Fashion-MNIST, classes T-shirt/top, Pullover, Dress
(800 : 40 : 40 pool, 1000 : 50 : 50 test), drawn with ``ablation.data.create_imbalanced_pool`` and
``build_test_set`` at ``random_seed = 42``. Panel B uses exactly those images: the indices come from the
ablation package, the pixels from the case-study loader (checked equal), and the preprocessing is the
case-study one (64 x 64, scaled to [-1, 1]), so that the rest of the case-study protocol of Section 4.2
(hold-out, downstream classifier, selection) applies unchanged.

Author decision 13 Oct 2026 (option 1): panel A's data and target, case-study training and evaluation.
"""

from __future__ import annotations

import numpy as np
import torch

REGIME = "A1_ablation"            # regime id used in file names and the backbone cache
CLASSES = [0, 2, 3]               # T-shirt/top (majority), Pullover, Dress
FULL_BALANCE_RATIO = 20           # min(N_max, 20 * 36) = 720 per class -> Delta_c = 684, as panel A


def load(config) -> dict:
    """Pool and test split of Table 3, as ``kbs2.runner.load_regime`` returns them."""
    import cs3j_runner as r
    from ablation.config import AblationConfig
    from ablation.data import build_test_set, create_imbalanced_pool
    from casestudy.data.loaders import _load_raw, _remap
    from casestudy.data.transforms import to_normalized_tensor

    base = config.datasets["fashion_mnist"]
    spec = r.regime_spec(base, REGIME, {"classes": CLASSES, "rho": 0.05, "majority_cap": 800})
    ac = AblationConfig()
    if list(ac.selected_classes) != CLASSES or ac.random_seed != 42 or ac.max_samples_majority != 800:
        raise SystemExit(f"ablation configuration changed: {ac.selected_classes}, {ac.random_seed}")
    _, pool_idx = create_imbalanced_pool(ac)
    test_idx = np.asarray(build_test_set(ac).indices)
    pool_idx = np.asarray(pool_idx)

    out = {}
    for split, idx, train in (("trainpool", pool_idx, True), ("test", test_idx, False)):
        x, y = _load_raw(spec.loader_key, train, config.data_root)
        imgs, lbls = x[idx], _remap(y[idx], CLASSES)
        out[split] = {"images": to_normalized_tensor(imgs, spec.channels, config.image_size, config.normalize_range),
                      "labels": torch.from_numpy(np.asarray(lbls)).long(), "classes": CLASSES,
                      "class_names": list(base.class_names), "source_index": idx}
    counts = {s: torch.bincount(out[s]["labels"], minlength=3).tolist() for s in out}
    if counts != {"trainpool": [800, 40, 40], "test": [1000, 50, 50]}:
        raise SystemExit(f"unexpected class counts {counts}")
    return {"config": config, "spec": spec, "pool": out["trainpool"], "test": out["test"], "cache": []}
