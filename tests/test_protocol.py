from __future__ import annotations

import json
from pathlib import Path

from remit.config import ResolvedConfig
from remit.protocol import RunContext


def test_run_context_records_config_environment_and_completion(
    project_config: tuple[ResolvedConfig, Path],
) -> None:
    config, _ = project_config
    with RunContext(config, split_id=0, seed=101, run_id="unit-test") as run:
        run.write_metrics("validation", {"pr_auc": 0.5})

    manifest = json.loads((run.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["config_hash"] == config.config_hash
    assert manifest["split_id"] == 0
    assert manifest["seed"] == 101
    assert (run.run_dir / "config.yaml").is_file()
    assert (run.run_dir / "train.log").is_file()
    assert (run.run_dir / "metrics_validation.json").is_file()
