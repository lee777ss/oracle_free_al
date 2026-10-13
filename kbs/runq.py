"""kbs.runq - a resumable, lock-based job queue for the KBS revision (S7).

    python -m kbs.runq run --stage E3_anchors --workers 3     # start detached workers and return
    python -m kbs.runq run --stage E3_anchors --workers 1 --first-wid 3   # add a worker later
    python -m kbs.runq worker --stage E3_anchors --wid 0       # one worker (what `run` starts)
    python -m kbs.runq status [--stage E3_anchors]
    python -m kbs.runq drain --wid 2                           # worker 2 exits after its current job
    python -m kbs.runq stop --stage E3_anchors

A job counts as done when its ``result.json`` exists with ``"status": "ok"``. A worker claims a
job by creating its lock file atomically, so two workers never run the same job; a lock whose
process has died is stale and is reclaimed. Killing a worker loses at most the job it was
running; draining it loses nothing. Failed jobs record their traceback and are retried only with
``--retry-errors``.

On Windows (WDDM) several processes share the GPU by time-slicing, which costs throughput for
small, launch-bound jobs; measure before running more than one worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

from casestudy.locks import read_lock, release, try_lock
from casestudy.protocol_v2 import Job, run_one_v2, save_root, write_error
from kbs.jobs import STAGES, stage_jobs


def lock_path(job: Job) -> Path:
    return save_root() / "locks" / (hashlib.sha1(job.key.encode()).hexdigest()[:16] + ".lock")


def ready(job: Job) -> bool:
    """False while a prerequisite is missing: audit jobs need the dataset's reference classifier (E1),
    reuse jobs the finished run whose synthetic set they retrain on (E12)."""
    if job.oracle:
        from casestudy.eval.oracle import oracle_dir
        from casestudy.protocol_v2 import REGIMES_V2
        if not (oracle_dir(REGIMES_V2[job.regime]["dataset"]) / "summary.json").exists():
            return False
    if job.reuse:
        f = save_root() / "runs" / job.reuse / "result.json"
        try:
            return json.loads(f.read_text(encoding="utf-8")).get("status") == "ok"
        except (OSError, ValueError):
            return False
    return True


def job_state(job: Job) -> str:
    f = job.run_dir / "result.json"
    if not f.exists():
        return "pending"
    try:
        return "done" if json.loads(f.read_text(encoding="utf-8")).get("status") == "ok" else "error"
    except (OSError, ValueError):
        return "pending"


def worker(stage: str, wid: int, retry_errors: bool) -> None:
    """Run jobs until none is left. ``stage`` may list several stages, comma-separated, in priority
    order: the worker always takes the first runnable job of the earliest stage, so a later stage
    fills the GPU while an earlier one has nothing left to claim."""
    stages = [s for s in stage.split(",") if s]
    print(f"[w{wid}] start stages={stages} pid={os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    attempted, done = set(), set()      # a finished job stays finished: skip re-reading its result

    def state(j: Job) -> str:
        if j.key in done:
            return "done"
        s_ = job_state(j)
        if s_ == "done":
            done.add(j.key)
        return s_

    while True:
        if _drain_file(wid).exists():
            _drain_file(wid).unlink(missing_ok=True)
            print(f"[w{wid}] drained; exiting {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            return
        wanted = ("pending", "error") if retry_errors else ("pending",)
        jobs = [j for st in stages for j in stage_jobs(st) if j.key not in attempted and state(j) in wanted]
        if not jobs:
            print(f"[w{wid}] nothing left to run; exiting {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            return
        ran = False
        for job in jobs:
            if not ready(job):
                continue
            lp = lock_path(job)
            if not try_lock(lp, key=job.key, worker=wid):
                continue
            try:
                if job_state(job) not in wanted:
                    continue
                attempted.add(job.key)
                t0 = time.time()
                print(f"[w{wid}] RUN {job.key}  {time.strftime('%H:%M:%S')}", flush=True)
                try:
                    res = run_one_v2(job)
                    m = res["metrics"].get(res["primary"], {})
                    print(f"[w{wid}] OK  {job.key}  macro-F2={m.get('macro_f2', float('nan')):.4f}  "
                          f"({time.time() - t0:.0f} s)", flush=True)
                except Exception:
                    err = traceback.format_exc()
                    write_error(job, err)
                    print(f"[w{wid}] ERR {job.key}\n{err}", flush=True)
                ran = True
            finally:
                release(lp)
            break                       # re-read the stage after every job
        if not ran:
            time.sleep(30)              # everything left is locked by other workers


def _workers_file(stage: str) -> Path:
    return save_root() / "logs" / f"{stage}.workers.json"


def _drain_file(wid: int) -> Path:
    return save_root() / "logs" / f"drain.w{wid}"


def run(stage: str, workers: int, retry_errors: bool, first_wid: int = 0) -> None:
    logs = save_root() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | 0x01000000   # CREATE_BREAKAWAY_FROM_JOB
    pids = []
    for wid in range(first_wid, first_wid + workers):
        _drain_file(wid).unlink(missing_ok=True)
        cmd = [sys.executable, "-u", "-m", "kbs.runq", "worker", "--stage", stage, "--wid", str(wid)]
        if stage == "E1_oracle":                      # not a protocol stage: one training process
            cmd, workers = [sys.executable, "-u", "-m", "kbs.oracle", "train"], 1
        if retry_errors:
            cmd.append("--retry-errors")
        log = open(logs / f"{stage}.w{wid}.log", "a", encoding="utf-8")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        try:
            p = subprocess.Popen(cmd, cwd=os.getcwd(), stdout=log, stderr=subprocess.STDOUT, creationflags=flags, env=env)
        except OSError:                 # the parent's job object forbids breakaway: start without it
            p = subprocess.Popen(cmd, cwd=os.getcwd(), stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=flags & ~0x01000000, env=env)
        pids.append(p.pid)
        if stage == "E1_oracle":
            break
        time.sleep(2)                   # stagger the start so that workers claim different jobs
    wf = _workers_file(stage)
    if wf.exists():                     # keep the workers of an earlier launch that are still alive
        old = json.loads(wf.read_text(encoding="utf-8"))["pids"]
        pids = [p for p in old if psutil.pid_exists(p)] + pids
    wf.write_text(json.dumps({"pids": pids, "started": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
    print(f"started {workers} worker(s) for {stage}: pids {pids[-workers:]}; logs in {logs}")


def _alive_workers(stage: str) -> list:
    """Live workers serving ``stage``, including workers started for a comma-separated chain containing it."""
    pids = []
    for f in (save_root() / "logs").glob("*.workers.json"):
        chain = f.name[: -len(".workers.json")].split(",")
        if stage in chain or f.name == _workers_file(stage).name:
            pids += json.loads(f.read_text(encoding="utf-8"))["pids"]
    alive = []
    for p in set(pids):
        try:
            if psutil.pid_exists(p) and "python" in psutil.Process(p).name().lower():
                alive.append(p)
        except psutil.Error:
            pass
    return alive


def status(stage: str | None = None) -> None:
    if stage in (None, "E1_oracle"):
        from kbs.oracle import status as oracle_status
        print(f"E1_oracle      workers alive: {len(_alive_workers('E1_oracle'))}")
        oracle_status()
        if stage == "E1_oracle":
            return
    for st in ([stage] if stage else STAGES):
        jobs = stage_jobs(st)
        if not jobs:
            print(f"{st:14s} no jobs (blocked until the freeze)" if st in ("E4_selection",) else f"{st:14s} no jobs")
            continue
        states = {"done": 0, "error": 0, "pending": 0}
        secs = []
        for j in jobs:
            s_ = job_state(j)
            states[s_] += 1
            if s_ == "done":
                try:
                    secs.append(json.loads((j.run_dir / "result.json").read_text(encoding="utf-8"))["timings_s"]["total_s"])
                except (OSError, ValueError, KeyError):
                    pass
        running = sum(1 for j in jobs if lock_path(j).exists() and read_lock(lock_path(j)))
        alive = _alive_workers(st)
        mean = sum(secs) / len(secs) if secs else float("nan")
        left = states["pending"] + (0 if not alive else 0)
        eta_h = (left * mean / max(1, len(alive))) / 3600 if secs and alive else float("nan")
        print(f"{st:14s} {states['done']:5d} done  {states['error']:3d} failed  {states['pending']:5d} pending  "
              f"{running:2d} running  workers alive: {len(alive)}  mean {mean:6.0f} s/job  ETA {eta_h:5.1f} h")


def stop(stage: str) -> None:
    for pid in _alive_workers(stage):
        try:
            proc = psutil.Process(pid)
            for child in proc.children(recursive=True):
                child.terminate()
            proc.terminate()
            print("stopped", pid)
        except psutil.Error as exc:
            print("could not stop", pid, exc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--stage", required=True); r.add_argument("--workers", type=int, default=1)
    r.add_argument("--retry-errors", action="store_true"); r.add_argument("--first-wid", type=int, default=0)
    w = sub.add_parser("worker"); w.add_argument("--stage", required=True); w.add_argument("--wid", type=int, default=0)
    w.add_argument("--retry-errors", action="store_true")
    s = sub.add_parser("status"); s.add_argument("--stage")
    d = sub.add_parser("drain"); d.add_argument("--wid", type=int, required=True)
    k = sub.add_parser("stop"); k.add_argument("--stage", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.stage, args.workers, args.retry_errors, args.first_wid)
    elif args.cmd == "worker":
        worker(args.stage, args.wid, args.retry_errors)
    elif args.cmd == "status":
        status(args.stage)
    elif args.cmd == "drain":
        _drain_file(args.wid).parent.mkdir(parents=True, exist_ok=True)
        _drain_file(args.wid).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        print(f"worker {args.wid} will exit after its current job")
    else:
        stop(args.stage)


if __name__ == "__main__":
    main()
