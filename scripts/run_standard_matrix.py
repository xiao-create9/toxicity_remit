"""Run the frozen 3-split x 3-seed prediction matrix on one or more GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

MODEL_CONFIGS = {
    "ecfp_rf": "configs/train_ecfp_rf.yaml",
    "gine": "configs/train_gine.yaml",
    "attentivefp": "configs/train_attentivefp.yaml",
}
SPLIT_IDS = (0, 1, 2)
MODEL_SEEDS = (2026, 2027, 2028)


@dataclass(frozen=True)
class Job:
    model: str
    split_id: int
    seed: int

    @property
    def name(self) -> str:
        return f"{self.model}-split{self.split_id}-seed{self.seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_CONFIGS),
        default=list(MODEL_CONFIGS),
    )
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-dir", default="scheduler_logs")
    return parser.parse_args()


def command(job: Job) -> list[str]:
    return [
        sys.executable,
        "-m",
        "remit.cli",
        "train",
        "--config",
        MODEL_CONFIGS[job.model],
        "--split-id",
        str(job.split_id),
        "--seed",
        str(job.seed),
    ]


def execute(job: Job, gpu_id: int | None, log_dir: Path, dry_run: bool) -> int:
    job_command = command(job)
    device = "cpu" if gpu_id is None else f"GPU {gpu_id}"
    print(f"[{device}] {job.name}: {' '.join(job_command)}", flush=True)
    if dry_run:
        return 0
    environment = os.environ.copy()
    if gpu_id is None:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    else:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{job.name}.log").open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            job_command,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    print(f"[{device}] {job.name}: exit={completed.returncode}", flush=True)
    return completed.returncode


def gpu_worker(
    gpu_id: int,
    jobs: list[Job],
    log_dir: Path,
    dry_run: bool,
    failures: list[str],
    lock: threading.Lock,
) -> None:
    for job in jobs:
        if execute(job, gpu_id, log_dir, dry_run):
            with lock:
                failures.append(job.name)


def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir)
    jobs = [
        Job(model, split_id, seed)
        for model in args.models
        for split_id in SPLIT_IDS
        for seed in MODEL_SEEDS
    ]
    failures: list[str] = []

    # RF is deliberately sequential on CPU because each forest already uses all CPU cores.
    for job in [item for item in jobs if item.model == "ecfp_rf"]:
        if execute(job, None, log_dir, args.dry_run):
            failures.append(job.name)

    gpu_jobs = [item for item in jobs if item.model != "ecfp_rf"]
    if gpu_jobs and not args.gpu_ids:
        raise SystemExit("At least one --gpu-ids value is required for GNN jobs")
    queues: dict[int, list[Job]] = defaultdict(list)
    for index, job in enumerate(gpu_jobs):
        queues[args.gpu_ids[index % len(args.gpu_ids)]].append(job)
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=gpu_worker,
            args=(gpu_id, assigned, log_dir, args.dry_run, failures, lock),
            daemon=False,
        )
        for gpu_id, assigned in queues.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        print(f"Failed jobs ({len(failures)}): {', '.join(sorted(failures))}", file=sys.stderr)
        return 1
    if not args.dry_run and set(args.models) == set(MODEL_CONFIGS):
        report = subprocess.run(
            [sys.executable, "-m", "remit.cli", "report", "prediction"],
            check=False,
        )
        if report.returncode:
            return report.returncode
    print(f"Completed {len(jobs)} jobs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
