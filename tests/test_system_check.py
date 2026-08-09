from pathlib import Path

import yaml

from remit.system_check import validate_runtime


def test_validate_runtime_accepts_expected_cuda() -> None:
    runtime = {
        "cuda_available": True,
        "device_count": 2,
        "torch_cuda_runtime": "12.8",
    }

    assert validate_runtime(runtime, require_cuda=True, expected_cuda="12.8") == []


def test_validate_runtime_reports_missing_or_wrong_cuda() -> None:
    runtime = {
        "cuda_available": False,
        "device_count": 0,
        "torch_cuda_runtime": None,
    }

    errors = validate_runtime(runtime, require_cuda=True, expected_cuda="12.8")

    assert len(errors) == 3
    assert "returned False" in errors[0]
    assert "zero visible devices" in errors[1]
    assert "got None" in errors[2]


def test_server_environment_is_conda_managed_without_uv() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = yaml.safe_load((project_root / "environment.server.yml").read_text())
    pip_dependencies = next(
        dependency["pip"]
        for dependency in environment["dependencies"]
        if isinstance(dependency, dict) and "pip" in dependency
    )

    assert environment["name"] == "toxicity-remit"
    assert "python=3.11" in environment["dependencies"]
    assert "-r requirements-server-cu128.txt" in pip_dependencies
    assert any("/cu128" in dependency for dependency in pip_dependencies)
    assert all("uv" not in dependency.lower() for dependency in pip_dependencies)
