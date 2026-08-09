"""Server preflight for the CUDA training environment."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Any

from remit.reproducibility import gpu_inventory


def collect_runtime() -> dict[str, Any]:
    """Collect the runtime facts needed to audit a server experiment."""
    import torch

    cuda_available = torch.cuda.is_available()
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gib": round(properties.total_memory / 1024**3, 2),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "device_count": len(devices),
        "devices": devices,
        "nvidia_smi": gpu_inventory(),
    }


def validate_runtime(
    runtime: dict[str, Any], *, require_cuda: bool, expected_cuda: str | None
) -> list[str]:
    """Return actionable validation errors without depending on CUDA hardware in tests."""
    errors: list[str] = []
    if require_cuda and not runtime["cuda_available"]:
        errors.append("CUDA is required, but torch.cuda.is_available() returned False")
    if require_cuda and runtime["device_count"] == 0:
        errors.append("CUDA is required, but PyTorch reported zero visible devices")
    if expected_cuda and runtime["torch_cuda_runtime"] != expected_cuda:
        errors.append(
            "Expected the PyTorch CUDA runtime "
            f"{expected_cuda}, got {runtime['torch_cuda_runtime']!r}"
        )
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--expected-cuda",
        help="Exact CUDA runtime bundled with PyTorch, for example 12.8",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = collect_runtime()
    errors = validate_runtime(
        runtime,
        require_cuda=args.require_cuda,
        expected_cuda=args.expected_cuda,
    )
    payload = {**runtime, "status": "ok" if not errors else "error", "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
