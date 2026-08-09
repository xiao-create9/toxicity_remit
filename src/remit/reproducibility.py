"""Randomness controls and machine-readable environment capture."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROCESS_START_PYTHONHASHSEED = os.environ.get("PYTHONHASHSEED")


def seed_everything(
    seed: int, deterministic_algorithms: bool = True, warn_only: bool = False
) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def git_state(project_root: Path) -> dict[str, Any]:
    commit = _command_output(["git", "rev-parse", "HEAD"], cwd=project_root)
    status = _command_output(["git", "status", "--porcelain"], cwd=project_root)
    branch = _command_output(["git", "branch", "--show-current"], cwd=project_root)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
        "status": status.splitlines() if status else [],
    }


def gpu_inventory() -> list[dict[str, str]]:
    query = "index,name,uuid,memory.total,driver_version"
    output = _command_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    )
    if not output:
        return []
    keys = ["index", "name", "uuid", "memory_total_mib", "driver_version"]
    return [
        dict(zip(keys, (part.strip() for part in line.split(",")), strict=True))
        for line in output.splitlines()
    ]


def dependency_versions(names: list[str] | None = None) -> dict[str, str | None]:
    selected = names or ["numpy", "pandas", "pyarrow", "pyyaml", "rdkit", "torch"]
    versions: dict[str, str | None] = {}
    for name in selected:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_manifest(project_root: Path) -> dict[str, Any]:
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": dependency_versions(),
        "gpu": gpu_inventory(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_hash_seed": {
            "at_process_start": PROCESS_START_PYTHONHASHSEED,
            "for_child_processes": os.environ.get("PYTHONHASHSEED"),
            "effective_for_current_process": os.environ.get("PYTHONHASHSEED")
            == PROCESS_START_PYTHONHASHSEED,
        },
        "git": git_state(project_root),
    }
