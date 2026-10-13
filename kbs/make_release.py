"""kbs.make_release - the curated public export of the revision (R1.11, D5).

    python -m kbs.make_release --out ../kbs_release            # build the export folder (nothing is pushed)
    python -m kbs.make_release --out ../kbs_release --zenodo ../kbs_zenodo   # also stage the large files
    python -m kbs.make_release --out ../kbs_release --check    # list what would be exported; write nothing

The working repository is never published: it holds private material (the decision letter, cover letters, a
signature image, earlier drafts) and tracks compiled files. This script copies an allow-list into a new folder:

* code from the current commit (``git archive``): ``casestudy/``, ``kbs/``, the dataset preparation scripts;
* ``configs/``: every job list, the regimes, the seeds, the frozen configuration, the full run configuration;
* ``results/runs.zip``: every finished run's ``result.json`` and gate log (``kbs.tables`` rebuilds all tables
  from it after ``python -m kbs.make_release unpack``), plus ``runs.csv`` and ``hashes.csv`` for direct reading;
* ``submitted_v1/``: the code and result files of the first submission, from the tag ``kbs-submitted``;
* ``README.md``, ``LICENSE`` (MIT), ``environment.yml`` with the versions of this machine.

With ``--zenodo`` it stages the large files for the archive (predictions, candidate records, generator
checkpoints of the proposed method for all seeds and of the rivals for seeds 42-46, reference classifiers).

Before anything is written, and again after, the export is scanned for private patterns; one hit aborts.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

from casestudy.protocol_v2 import REGIMES_V2, make_config, save_root

REPO = Path(__file__).resolve().parents[1]
CODE = ["casestudy", "kbs", "cs3j_runner.py", "_prep_stl10.py", "_prep_gtsrb.py"]   # protocol_v2 reads cs3j_runner
SUBMITTED_TAG = "kbs-submitted"
SUBMITTED_FILES = ["cs3j_runner.py", "_cs3j_rho3x3.py", "casestudy", "ablation_15_seeds.csv"]
SUBMITTED_RESULTS = re.compile(r"^saved_casestudy/cs3j_[a-z0-9]+_[A-Za-z0-9_]+\.(csv|manifest\.json)$")
# Files whose path matches are never published (decision letter, signature, cover letters, title page, the
# reviewers' material, reference scratch files, manuscript folders).
PRIVATE_PATH = re.compile(r"kbs_revision|signature|cover ?letter|titlepage|User sources|\.kbsref|Journal_paper_", re.I)
# Text lines that must not be published: any e-mail address, any user-folder path, the account name of this
# machine (read at run time, never written down here), credentials.
_USER = os.environ.get("USERNAME") or os.environ.get("USER") or ""
PRIVATE_TEXT = re.compile("|".join([r"[\w.+-]+@[\w-]+\.[\w.-]+", r"[A-Za-z]:[\\/]+Users[\\/]", r"/home/\w+",
                                    r"(?:api[_-]?key|password|secret|token)\s*[:=]\s*['\"][^'\"]{4,}['\"]"]
                                   + ([re.escape(_USER.split(".")[0])] if len(_USER) >= 3 else [])), re.I)
SKIP = re.compile(r"(^|/)(__pycache__|\.ipynb_checkpoints)(/|$)|\.pyc$")
TEXT = {".py", ".md", ".txt", ".json", ".csv", ".yml", ".yaml", ".tex", ".cfg", ".toml"}
GENERATIVE = {"m04_dcgan", "m05_cgan", "m06_acgan", "m07_wgangp", "m08_bagan", "m08_bagan_ls", "m09_bagan_gp"}


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def archive(ref: str, paths: Iterable[str]) -> Dict[str, bytes]:
    """{path: bytes} of the tracked files under ``paths`` at ``ref`` (compiled files dropped)."""
    tar = tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", ref, *paths)))
    return {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile() and not SKIP.search(m.name)}


def private_hits(files: Dict[str, bytes]) -> List[str]:
    hits = []
    for path, data in files.items():
        if PRIVATE_PATH.search(path):
            hits.append(f"path: {path}")
            continue
        if Path(path).suffix.lower() in TEXT:
            for i, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                m = PRIVATE_TEXT.search(line)
                if m:
                    hits.append(f"{path}:{i}: ...{line[max(0, m.start() - 30):m.end() + 30].strip()}...")
        elif path.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for name in z.namelist():
                    text = z.read(name).decode("utf-8", errors="replace")
                    m = PRIVATE_PATH.search(name) or PRIVATE_TEXT.search(text)
                    if m:
                        hits.append(f"{path}!{name}: ...{m.group(0)}...")
    return hits


def job_lists() -> Dict[str, bytes]:
    from kbs.jobs import STAGES, frozen, stage_jobs

    out = {}
    for st in STAGES:
        jobs = [{"stage": j.stage, "regime": j.regime, "method": j.method, "variant": j.variant, "seed": j.seed,
                 "overrides": j.overrides, "eval_split": j.eval_split, "oracle": j.oracle, "ratio": j.ratio,
                 "reuse": j.reuse, "weighted": j.weighted} for j in stage_jobs(st)]
        out[f"configs/jobs_{st}.json"] = json.dumps(jobs, indent=1).encode()
    import dataclasses
    out["configs/regimes.json"] = json.dumps(REGIMES_V2, indent=1).encode()
    out["configs/frozen.json"] = json.dumps(frozen(), indent=1).encode()
    out["configs/run_config.json"] = json.dumps(dataclasses.asdict(make_config()), indent=1, default=str).encode()
    return out


def results() -> Dict[str, bytes]:
    """runs.zip (result.json and gate.json of every finished run), runs.csv and hashes.csv."""
    import pandas as pd

    root = save_root() / "runs"
    buf, rows, hashes = io.BytesIO(), [], []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob("result.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r.get("status") != "ok" or r.get("stage", "").startswith("T_"):
                continue
            rel = f.relative_to(save_root()).as_posix()
            r.pop("env", None)                               # machine details stay out of the public record
            z.writestr(rel, json.dumps(r, indent=1, default=str))
            if (f.parent / "gate.json").exists():
                z.write(f.parent / "gate.json", (f.parent / "gate.json").relative_to(save_root()).as_posix())
            base = {k: r.get(k) for k in ("stage", "regime", "dataset", "rho", "method", "variant", "seed", "n_synth")}
            for split, m in r["metrics"].items():
                rows.append({**base, "split": split, "primary": split == r["primary"],
                             **{k: v for k, v in m.items() if isinstance(v, (int, float))}})
            hashes.append({**{k: base[k] for k in ("stage", "regime", "method", "variant", "seed")}, **r["hashes"]})
    return {"results/runs.zip": buf.getvalue(),
            "results/runs.csv": pd.DataFrame(rows).to_csv(index=False).encode(),
            "results/hashes.csv": pd.DataFrame(hashes).to_csv(index=False).encode()}


def submitted() -> Dict[str, bytes]:
    files = archive(SUBMITTED_TAG, SUBMITTED_FILES)
    names = git("ls-tree", "-r", "--name-only", SUBMITTED_TAG, "saved_casestudy").decode().splitlines()
    res = [n for n in names if SUBMITTED_RESULTS.match(n)]
    if res:
        files.update(archive(SUBMITTED_TAG, res))
    return {f"submitted_v1/{k}": v for k, v in files.items()}


def environment() -> bytes:
    import importlib
    pins = []
    for mod, name in (("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"),
                      ("imblearn", "imbalanced-learn"), ("medmnist", "medmnist"), ("matplotlib", "matplotlib"),
                      ("psutil", "psutil"), ("PIL", "pillow")):
        pins.append(f"      - {name}=={importlib.import_module(mod).__version__}")
    import torch
    import torchvision
    tv = torch.__version__.split("+")[0]
    return "\n".join([
        "name: kbs-revision", "channels:", "  - conda-forge", "dependencies:",
        f"  - python={sys.version_info.major}.{sys.version_info.minor}", "  - pip", "  - pip:",
        f"      - torch=={tv}", f"      - torchvision=={torchvision.__version__.split('+')[0]}", *pins,
        f"# measured with CUDA {torch.version.cuda}; install the torch build that matches your CUDA from pytorch.org",
        ""]).encode()


README = """# Confidence-gated generative augmentation: code, configurations and results

Companion repository of the revised manuscript KNOSYS-D-26-20956 (Knowledge-Based Systems). It contains the
implementation of the proposed method and of the fourteen methods it is compared with, the exact job lists and
seeds of every experiment, the per-run results behind every table and figure, and the code and results of the first
submission for provenance. Large files (predictions, candidate records, generator checkpoints, reference
classifiers) are archived on Zenodo: DOI {doi}.

## Install

    conda env create -f environment.yml
    conda activate kbs-revision

## Data

The datasets are not redistributed. Fashion-MNIST, CIFAR-10 and OCTMNIST are downloaded on first use into
`./data`; STL-10 and GTSRB are prepared once with `python _prep_stl10.py` and `python _prep_gtsrb.py`.

## Regenerate the tables and figures from the published results

    python -m kbs.make_release unpack          # results/runs.zip -> saved_kbs/runs/
    python -m kbs.tables                       # every table, figure and quoted number

## Rerun an experiment

    python -m kbs.jobs list                    # stages and job counts
    python -m kbs.runq run --stage E3_anchors  # resumable queue; one worker per GPU
    python -m kbs.runq status

Every run records the hashes of its pool, hold-out split, initial classifier weights and test set
(`results/hashes.csv`); `python -m kbs.verify hashes --stage E3_anchors` checks that all methods of a seed saw
the same data. Seeds: 42-61 (benchmark anchors, selection study, domain datasets), 42-51 (imbalance grid,
sensitivity), 142-151 (development pilot). Hardware used: one NVIDIA RTX 5090; run times are in the cost table.

## Layout

| Folder | Content |
|---|---|
| `casestudy/` | data, methods, classifiers, metrics; `protocol_v2.py` is the protocol of the revised paper |
| `kbs/` | experiment stages, queue, label audit, realism, tables and figures |
| `cs3j_runner.py` | regime definitions and run configuration shared with the first submission |
| `configs/` | job list of every stage, regimes, seeds, the frozen configuration of the proposed method |
| `results/` | per-run results (`runs.zip`), flat tables (`runs.csv`, `hashes.csv`) |
| `submitted_v1/` | code and result files behind the first submission (tag `kbs-submitted`) |

Source commit: {commit}.
"""

MIT = """MIT License

Copyright (c) 2026 {holders}

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def zenodo_files() -> List[Path]:
    """Large files for the archive: predictions and candidate records of every run; generator checkpoints of the
    proposed method for all seeds and of the generative rivals for seeds 42-46; reference classifiers."""
    root, out = save_root(), []
    for f in sorted((root / "runs").rglob("*")):
        if not f.is_file() or "/T_" in f.as_posix():
            continue
        if f.name in ("preds.npz", "candidates.npz", "gallery.npz", "synth.npz"):
            out.append(f)
        elif f.name == "generator.pt":
            method, seed = f.parent.parent.name.split("+")[0], int(f.parent.name[1:])
            if method == "p_proposed" or (method in GENERATIVE and 42 <= seed <= 46):
                out.append(f)
    out += sorted((root / "oracle").rglob("*")) if (root / "oracle").exists() else []
    return [f for f in out if f.is_file()]


def unpack() -> None:
    with zipfile.ZipFile(REPO / "results" / "runs.zip") as z:
        z.extractall(save_root())
    print("unpacked into", save_root() / "runs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="build", choices=["build", "unpack"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--zenodo", type=Path)
    ap.add_argument("--check", action="store_true", help="list and scan only; write nothing")
    ap.add_argument("--force", action="store_true", help="replace a non-empty --out folder")
    ap.add_argument("--doi", default="(to be reserved)")
    ap.add_argument("--holders", default="the authors")
    args = ap.parse_args()
    if args.cmd == "unpack":
        unpack()
        return
    if args.out is None:
        ap.error("--out is required")
    out = args.out.resolve()
    if REPO in out.parents or out == REPO:
        sys.exit("refusing: the export must be outside the working repository")
    commit = git("rev-parse", "HEAD").decode().strip()
    files: Dict[str, bytes] = {}
    files.update(archive("HEAD", CODE))
    files.update(job_lists())
    files.update(results())
    files.update(submitted())
    files["environment.yml"] = environment()
    files["README.md"] = README.format(doi=args.doi, commit=commit).encode()
    files["LICENSE"] = MIT.format(holders=args.holders).encode()
    hits = private_hits(files)
    size = sum(len(v) for v in files.values())
    print(f"export: {len(files)} files, {size / 2 ** 20:.1f} MB, from commit {commit[:8]}")
    for top in sorted({k.split("/")[0] for k in files}):
        n = sum(1 for k in files if k.split("/")[0] == top)
        print(f"  {top:24s} {n:5d} files")
    if hits:
        print(f"\nREFUSED: {len(hits)} private match(es); nothing written:")
        for h in hits[:40]:
            print("  ", h)
        sys.exit(1)
    if args.check:
        print("\nprivacy scan clean; --check: nothing written")
        return
    if out.exists() and any(out.iterdir()):
        if not args.force:
            sys.exit(f"refusing: {out} is not empty (use --force to replace it)")
        shutil.rmtree(out)
    for rel, data in files.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    print("written:", out)
    if args.zenodo:
        z = args.zenodo.resolve()
        if REPO in z.parents:
            sys.exit("refusing: the Zenodo staging folder must be outside the working repository")
        big = zenodo_files()
        for f in big:
            dst = z / f.relative_to(save_root())
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        print(f"zenodo staging: {len(big)} files, {sum(f.stat().st_size for f in big) / 2 ** 30:.2f} GB -> {z}")


if __name__ == "__main__":
    main()
