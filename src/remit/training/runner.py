"""Prediction baseline dispatch with protocol preflight checks."""

from __future__ import annotations

from typing import Any

from remit.data.splits import verify_split_files
from remit.data.standardize import verify_processed_dataset
from remit.protocol import RunContext
from remit.training.common import TrainingError
from remit.training.gnn import train_gnn
from remit.training.rf import train_ecfp_rf


def run_prediction(config: Any, split_id: int, seed: int) -> RunContext:
    split_seeds = config.section("split")["seeds"]
    model_seeds = config.section("reproducibility")["model_seeds"]
    if split_id not in range(len(split_seeds)):
        raise TrainingError(f"split_id must be one of {list(range(len(split_seeds)))}")
    if seed not in model_seeds:
        raise TrainingError(f"seed must be one of the frozen model seeds: {model_seeds}")
    verify_processed_dataset(config)
    verify_split_files(config)
    model_name = config.section("model").get("name")
    if model_name == "ecfp_rf":
        return train_ecfp_rf(config, split_id, seed)
    if model_name in {"gine", "attentivefp"}:
        return train_gnn(config, split_id, seed)
    raise TrainingError(f"Unknown prediction model: {model_name}")
