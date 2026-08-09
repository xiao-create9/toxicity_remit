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
