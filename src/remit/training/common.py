"""Shared prediction-run outputs and masked multi-task helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from remit.evaluation.metrics import classification_metrics, prediction_table
from remit.protocol import RunContext


class TrainingError(RuntimeError):
    """Raised when a prediction run cannot satisfy the fixed protocol."""


def positive_class_weights(labels: np.ndarray, endpoints: Sequence[str], cap: float) -> np.ndarray:
    weights = np.ones(len(endpoints), dtype=np.float32)
    for column, endpoint in enumerate(endpoints):
        known = ~np.isnan(labels[:, column])
        positives = int((labels[known, column] == 1).sum())
        negatives = int((labels[known, column] == 0).sum())
        if not positives or not negatives:
            raise TrainingError(
                f"Training partition endpoint {endpoint} requires both classes; "
                f"positive={positives}, negative={negatives}"
            )
        weights[column] = min(negatives / positives, cap)
    return weights


def save_partition_outputs(
    run: RunContext,
    frame: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    endpoints: Sequence[str],
    thresholds: dict[str, float],
    partition: str,
    model_name: str,
) -> dict[str, object]:
    metrics = classification_metrics(labels, probabilities, endpoints, thresholds)
    if partition not in {"validation", "test"}:
        raise ValueError(f"Unsupported reported partition: {partition}")
    run.write_metrics(partition, metrics)
    predictions = prediction_table(
        frame=frame,
        labels=labels,
        probabilities=probabilities,
        endpoints=endpoints,
        thresholds=thresholds,
        dataset=run.config.section("data")["name"],
        model=model_name,
        partition=partition,
        split_id=run.split_id,
        seed=run.seed,
    )
    predictions.to_parquet(run.run_dir / f"predictions_{partition}.parquet", index=False)
    return metrics


def existing_path(path: Path, description: str) -> Path:
    if not path.is_file():
        raise TrainingError(f"{description} does not exist: {path}")
    return path
