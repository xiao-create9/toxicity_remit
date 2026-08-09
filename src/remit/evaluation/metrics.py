"""Masked multi-task classification metrics and validation-only thresholds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if selected.any():
            error += float(selected.mean()) * abs(
                float(probabilities[selected].mean()) - float(labels[selected].mean())
            )
    return error if total else float("nan")


def _macro_average(endpoint_metrics: dict[str, dict[str, Any]], metric: str) -> float | None:
    values = [
        float(result[metric])
        for result in endpoint_metrics.values()
        if result.get(metric) is not None and np.isfinite(result[metric])
    ]
    return float(np.mean(values)) if values else None


def select_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    endpoints: Sequence[str],
    grid_size: int = 91,
) -> dict[str, float]:
    """Choose endpoint thresholds on validation labels by MCC, with deterministic ties."""
    if grid_size < 3:
        raise ValueError("threshold_grid_size must be at least 3")
    grid = np.linspace(0.05, 0.95, grid_size)
    thresholds: dict[str, float] = {}
    for column, endpoint in enumerate(endpoints):
        known = ~np.isnan(labels[:, column])
        y_true = labels[known, column].astype(int)
        y_probability = probabilities[known, column]
        if len(np.unique(y_true)) < 2:
            thresholds[endpoint] = 0.5
            continue
        candidates = []
        for threshold in grid:
            score = matthews_corrcoef(y_true, (y_probability >= threshold).astype(int))
            candidates.append((float(score), -abs(float(threshold) - 0.5), -float(threshold)))
        best_index = max(range(len(grid)), key=lambda index: candidates[index])
        thresholds[endpoint] = float(grid[best_index])
    return thresholds


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    endpoints: Sequence[str],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if labels.shape != probabilities.shape or labels.shape[1] != len(endpoints):
        raise ValueError("Labels, probabilities, and endpoint names have incompatible shapes")
    endpoint_metrics: dict[str, dict[str, Any]] = {}
    for column, endpoint in enumerate(endpoints):
        known = ~np.isnan(labels[:, column])
        y_true = labels[known, column].astype(int)
        y_probability = np.clip(probabilities[known, column], 0.0, 1.0)
        threshold = float(thresholds[endpoint])
        y_pred = (y_probability >= threshold).astype(int)
        has_both_classes = len(np.unique(y_true)) == 2
        negatives = int((y_true == 0).sum())
        positives = int((y_true == 1).sum())
        specificity = (
            float(((y_pred == 0) & (y_true == 0)).sum() / negatives) if negatives else None
        )
        endpoint_metrics[endpoint] = {
            "known": int(len(y_true)),
            "positive": positives,
            "negative": negatives,
            "threshold": threshold,
            "pr_auc": float(average_precision_score(y_true, y_probability))
            if has_both_classes
            else None,
            "roc_auc": float(roc_auc_score(y_true, y_probability)) if has_both_classes else None,
            "mcc": float(matthews_corrcoef(y_true, y_pred)) if has_both_classes else None,
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))
            if has_both_classes
            else None,
            "recall": float(recall_score(y_true, y_pred, zero_division=0)) if positives else None,
            "specificity": specificity,
            "ece": expected_calibration_error(y_true, y_probability) if len(y_true) else None,
            "brier": float(brier_score_loss(y_true, y_probability)) if len(y_true) else None,
        }
    metric_names = (
        "pr_auc",
        "roc_auc",
        "mcc",
        "balanced_accuracy",
        "recall",
        "specificity",
        "ece",
        "brier",
    )
    return {
        "samples": int(labels.shape[0]),
        "known_labels": int((~np.isnan(labels)).sum()),
        "macro": {metric: _macro_average(endpoint_metrics, metric) for metric in metric_names},
        "endpoints": endpoint_metrics,
    }


def prediction_table(
    frame: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    endpoints: Sequence[str],
    thresholds: dict[str, float],
    dataset: str,
    model: str,
    partition: str,
    split_id: int,
    seed: int,
) -> pd.DataFrame:
    """Return one auditable row per molecule-endpoint prediction."""
    records: list[dict[str, Any]] = []
    for row_index, row in frame.reset_index(drop=True).iterrows():
        for column, endpoint in enumerate(endpoints):
            probability = float(probabilities[row_index, column])
            label = labels[row_index, column]
            records.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "partition": partition,
                    "sample_id": row["sample_id"],
                    "canonical_smiles": row["canonical_smiles"],
                    "endpoint": endpoint,
                    "split_id": split_id,
                    "seed": seed,
                    "y_true": None if np.isnan(label) else int(label),
                    "y_probability": probability,
                    "y_pred": int(probability >= thresholds[endpoint]),
                    "threshold": thresholds[endpoint],
                    "scaffold": row["scaffold"],
                }
            )
    output = pd.DataFrame(records)
    output["y_true"] = output["y_true"].astype("Int8")
    output["y_pred"] = output["y_pred"].astype("int8")
    return output
