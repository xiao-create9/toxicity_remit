"""ECFP4 plus per-endpoint Random Forest baseline."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from remit.chemistry.features import ecfp_matrix
from remit.data.prediction import label_matrix
from remit.evaluation.metrics import select_thresholds
from remit.protocol import RunContext
from remit.training.common import save_partition_outputs
from remit.utils import atomic_write_json


def _predict_positive(model: RandomForestClassifier, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = model.classes_.tolist()
    if 1 not in classes:
        return np.zeros(len(features), dtype=np.float64)
    return probabilities[:, classes.index(1)]


def train_ecfp_rf(config: Any, split_id: int, seed: int) -> RunContext:
    data_config = config.section("data")
    model_config = config.section("model")
    training_config = config.section("training")
    endpoints = list(data_config["label_columns"])
    with RunContext(config, split_id=split_id, seed=seed) as run:
        started = perf_counter()
        guard = run.split_guard()
        train_frame = guard.load("train", "training")
        validation_frame = guard.load("validation", "model_selection")
        fingerprint_config = model_config["fingerprint"]
        feature_options = {
            "radius": int(fingerprint_config.get("radius", 2)),
            "size": int(fingerprint_config.get("size", 2048)),
            "include_chirality": bool(fingerprint_config.get("include_chirality", True)),
        }
        run.logger.info("Building ECFP features for train and validation")
        train_features = ecfp_matrix(train_frame["canonical_smiles"].tolist(), **feature_options)
        validation_features = ecfp_matrix(
            validation_frame["canonical_smiles"].tolist(), **feature_options
        )
        train_labels = label_matrix(train_frame, endpoints)
        validation_labels = label_matrix(validation_frame, endpoints)

        forest_config = model_config["random_forest"]
        models: dict[str, RandomForestClassifier] = {}
        validation_probabilities = np.empty_like(validation_labels, dtype=np.float64)
        for column, endpoint in enumerate(endpoints):
            known = ~np.isnan(train_labels[:, column])
            model = RandomForestClassifier(
                n_estimators=int(forest_config.get("n_estimators", 500)),
                max_depth=forest_config.get("max_depth"),
                min_samples_leaf=int(forest_config.get("min_samples_leaf", 1)),
                max_features=forest_config.get("max_features", "sqrt"),
                class_weight=forest_config.get("class_weight", "balanced"),
                n_jobs=int(forest_config.get("n_jobs", -1)),
                random_state=seed + column,
            )
            model.fit(train_features[known], train_labels[known, column].astype(int))
            models[endpoint] = model
            validation_probabilities[:, column] = _predict_positive(model, validation_features)
            run.logger.info(
                "Fitted RF endpoint=%s known=%d positive=%d",
                endpoint,
                int(known.sum()),
                int((train_labels[known, column] == 1).sum()),
            )

        thresholds = select_thresholds(
            validation_labels,
            validation_probabilities,
            endpoints,
            grid_size=int(training_config.get("threshold_grid_size", 91)),
        )
        atomic_write_json(run.run_dir / "thresholds.json", thresholds)
        validation_metrics = save_partition_outputs(
            run,
            validation_frame,
            validation_labels,
            validation_probabilities,
            endpoints,
            thresholds,
            "validation",
            "ecfp_rf",
        )
        joblib.dump(
            {
                "models": models,
                "endpoints": endpoints,
                "thresholds": thresholds,
                "feature_options": feature_options,
                "config_hash": config.config_hash,
            },
            run.run_dir / "checkpoint.joblib",
            compress=3,
        )

        # The test partition is intentionally loaded only after every model and threshold is frozen.
        test_frame = guard.load("test", "final_evaluation")
        test_features = ecfp_matrix(test_frame["canonical_smiles"].tolist(), **feature_options)
        test_labels = label_matrix(test_frame, endpoints)
        test_probabilities = np.column_stack(
            [_predict_positive(models[endpoint], test_features) for endpoint in endpoints]
        )
        test_metrics = save_partition_outputs(
            run,
            test_frame,
            test_labels,
            test_probabilities,
            endpoints,
            thresholds,
            "test",
            "ecfp_rf",
        )
        duration = perf_counter() - started
        run.record_summary(
            model="ecfp_rf",
            best_validation_epoch=None,
            best_validation_macro_pr_auc=validation_metrics["macro"]["pr_auc"],
            test_macro_pr_auc=test_metrics["macro"]["pr_auc"],
            duration_seconds=duration,
            fingerprint=feature_options,
        )
        run.logger.info(
            "RF completed validation_macro_pr_auc=%.6f test_macro_pr_auc=%.6f duration=%.1fs",
            validation_metrics["macro"]["pr_auc"],
            test_metrics["macro"]["pr_auc"],
            duration,
        )
    return run
