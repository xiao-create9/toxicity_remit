from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from remit.config import ResolvedConfig, load_config
from remit.data.splits import generate_scaffold_splits
from remit.training.runner import run_prediction
from tests.test_splits import _prepare


@pytest.mark.parametrize("model_name", ["ecfp_rf", "gine", "attentivefp"])
def test_prediction_training_smoke(
    project_config: tuple[ResolvedConfig, Path], model_name: str
) -> None:
    base_config, root = project_config
    _prepare(base_config, root)
    generate_scaffold_splits(base_config)
    overrides = []
    if model_name != "ecfp_rf":
        overrides = [
            f"model.name={model_name}",
            "model.hidden_dim=32",
            "model.num_layers=2",
            "model.num_timesteps=2",
            "model.dropout=0.0",
        ]
    config = load_config(root / "configs" / "default.yaml", overrides)
    run = run_prediction(config, split_id=0, seed=101)

    manifest = json.loads((run.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["model"] == model_name
    assert (run.run_dir / "metrics_validation.json").is_file()
    assert (run.run_dir / "metrics_test.json").is_file()
    predictions = pd.read_parquet(run.run_dir / "predictions_test.parquet")
    assert set(predictions["endpoint"]) == {"endpoint_a", "endpoint_b"}
    assert len(predictions) == 2 * len(predictions["sample_id"].unique())
    access_events = (run.run_dir / "data_access.jsonl").read_text(encoding="utf-8")
    assert '"phase": "final_evaluation"' in access_events
