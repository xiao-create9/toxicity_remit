from __future__ import annotations

import numpy as np

from remit.evaluation.metrics import classification_metrics, select_thresholds


def test_masked_metrics_and_validation_thresholds() -> None:
    labels = np.array([[0.0, 1.0], [1.0, np.nan], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    probabilities = np.array([[0.1, 0.8], [0.9, 0.2], [0.7, 0.3], [0.4, 0.9]], dtype=np.float32)
    endpoints = ["a", "b"]
    thresholds = select_thresholds(labels, probabilities, endpoints, grid_size=11)
    metrics = classification_metrics(labels, probabilities, endpoints, thresholds)
    assert metrics["known_labels"] == 7
    assert metrics["endpoints"]["a"]["pr_auc"] == 1.0
    assert metrics["endpoints"]["b"]["known"] == 3
    assert set(thresholds) == set(endpoints)
