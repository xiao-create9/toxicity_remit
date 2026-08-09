"""Unified masked multi-task training loop for GINE and AttentiveFP."""

from __future__ import annotations

import random
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import nn
from torch_geometric.loader import DataLoader

from remit.data.prediction import MoleculeGraphDataset, label_matrix
from remit.evaluation.metrics import classification_metrics, select_thresholds
from remit.models.gnn import build_gnn_model, trainable_parameter_count
from remit.protocol import RunContext
from remit.training.common import positive_class_weights, save_partition_outputs
from remit.utils import atomic_write_json


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _resolve_device(runtime_config: dict[str, Any]) -> torch.device:
    requested = str(runtime_config.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
    return torch.device(requested)


def _make_loader(
    dataset: MoleculeGraphDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=False,
    )


def _masked_loss(
    logits: torch.Tensor, labels: torch.Tensor, positive_weights: torch.Tensor
) -> torch.Tensor:
    known = ~torch.isnan(labels)
    targets = torch.nan_to_num(labels, nan=0.0)
    losses = functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weights, reduction="none"
    )
    return losses[known].mean()


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    positive_weights: torch.Tensor,
    amp_enabled: bool,
) -> tuple[np.ndarray, float]:
    model.eval()
    probabilities: list[np.ndarray] = []
    weighted_loss = 0.0
    known_labels = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = _masked_loss(logits, batch.y, positive_weights)
        known = int((~torch.isnan(batch.y)).sum().item())
        weighted_loss += float(loss.item()) * known
        known_labels += known
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probabilities, axis=0), weighted_loss / max(known_labels, 1)


def train_gnn(config: Any, split_id: int, seed: int) -> RunContext:
    data_config = config.section("data")
    model_config = config.section("model")
    training_config = config.section("training")
    runtime_config = config.section("runtime")
    endpoints = list(data_config["label_columns"])
    model_name = str(model_config["name"])
    if model_name not in {"gine", "attentivefp"}:
        raise ValueError(f"train_gnn cannot train model={model_name}")

    with RunContext(config, split_id=split_id, seed=seed) as run:
        started = perf_counter()
        device = _resolve_device(runtime_config)
        guard = run.split_guard()
        train_frame = guard.load("train", "training")
        validation_frame = guard.load("validation", "model_selection")
        run.logger.info("Building molecular graphs for train and validation")
        train_dataset = MoleculeGraphDataset(train_frame, endpoints)
        validation_dataset = MoleculeGraphDataset(validation_frame, endpoints)
        train_labels = label_matrix(train_frame, endpoints)
        validation_labels = label_matrix(validation_frame, endpoints)
        positive_weights_array = positive_class_weights(
            train_labels,
            endpoints,
            cap=float(training_config.get("pos_weight_cap", 20.0)),
        )
        positive_weights = torch.tensor(positive_weights_array, device=device)

        batch_size = int(training_config.get("batch_size", 128))
        workers = int(runtime_config.get("num_workers", 0))
        train_loader = _make_loader(
            train_dataset,
            batch_size,
            shuffle=True,
            seed=seed,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        validation_loader = _make_loader(
            validation_dataset,
            batch_size,
            shuffle=False,
            seed=seed,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        model = build_gnn_model(model_config, len(endpoints)).to(device)
        parameter_count = trainable_parameter_count(model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config.get("learning_rate", 1e-3)),
            weight_decay=float(training_config.get("weight_decay", 1e-5)),
        )
        amp_enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
        amp_dtype = str(training_config.get("amp_dtype", "bfloat16"))
        if amp_enabled and amp_dtype != "bfloat16":
            raise ValueError(
                "The reproducible Stage A trainer currently supports amp_dtype=bfloat16"
            )
        run.logger.info(
            "Training model=%s device=%s parameters=%d amp=%s",
            model_name,
            device,
            parameter_count,
            amp_enabled,
        )

        max_epochs = int(training_config.get("max_epochs", 100))
        min_epochs = int(training_config.get("min_epochs", 10))
        patience_limit = int(training_config.get("early_stopping_patience", 20))
        min_delta = float(training_config.get("early_stopping_min_delta", 1e-4))
        gradient_clip = float(training_config.get("gradient_clip_norm", 5.0))
        checkpoint_path = run.run_dir / "checkpoint.pt"
        history: list[dict[str, Any]] = []
        best_score = -float("inf")
        best_epoch = 0
        patience = 0
        default_thresholds = dict.fromkeys(endpoints, 0.5)

        for epoch in range(1, max_epochs + 1):
            model.train()
            total_loss = 0.0
            total_known = 0
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    loss = _masked_loss(logits, batch.y, positive_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                known = int((~torch.isnan(batch.y)).sum().item())
                total_loss += float(loss.item()) * known
                total_known += known

            validation_probabilities, validation_loss = _evaluate(
                model,
                validation_loader,
                device,
                positive_weights,
                amp_enabled,
            )
            validation_metrics = classification_metrics(
                validation_labels,
                validation_probabilities,
                endpoints,
                default_thresholds,
            )
            score = validation_metrics["macro"]["pr_auc"]
            if score is None:
                raise RuntimeError("Validation macro PR-AUC is undefined")
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(total_known, 1),
                    "validation_loss": validation_loss,
                    "validation_macro_pr_auc": score,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            improved = score > best_score + min_delta
            if improved:
                best_score = float(score)
                best_epoch = epoch
                patience = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": model_name,
                        "model_config": model_config,
                        "endpoints": endpoints,
                        "epoch": epoch,
                        "validation_macro_pr_auc": best_score,
                        "config_hash": config.config_hash,
                    },
                    checkpoint_path,
                )
            else:
                patience += 1
            run.logger.info(
                "epoch=%d train_loss=%.6f val_loss=%.6f val_macro_pr_auc=%.6f best=%.6f",
                epoch,
                history[-1]["train_loss"],
                validation_loss,
                score,
                best_score,
            )
            if epoch >= min_epochs and patience >= patience_limit:
                run.logger.info("Early stopping at epoch=%d", epoch)
                break

        pd.DataFrame(history).to_csv(run.run_dir / "training_history.csv", index=False)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        validation_probabilities, _ = _evaluate(
            model,
            validation_loader,
            device,
            positive_weights,
            amp_enabled,
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
            model_name,
        )

        # Test data is materialized exactly once, after checkpoint and thresholds are frozen.
        test_frame = guard.load("test", "final_evaluation")
        test_dataset = MoleculeGraphDataset(test_frame, endpoints)
        test_loader = _make_loader(
            test_dataset,
            batch_size,
            shuffle=False,
            seed=seed,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        test_probabilities, _ = _evaluate(
            model,
            test_loader,
            device,
            positive_weights,
            amp_enabled,
        )
        test_labels = label_matrix(test_frame, endpoints)
        test_metrics = save_partition_outputs(
            run,
            test_frame,
            test_labels,
            test_probabilities,
            endpoints,
            thresholds,
            "test",
            model_name,
        )
        duration = perf_counter() - started
        run.record_summary(
            model=model_name,
            device=str(device),
            trainable_parameters=parameter_count,
            best_validation_epoch=best_epoch,
            best_validation_macro_pr_auc=validation_metrics["macro"]["pr_auc"],
            test_macro_pr_auc=test_metrics["macro"]["pr_auc"],
            epochs_completed=len(history),
            duration_seconds=duration,
            positive_class_weights=dict(
                zip(endpoints, positive_weights_array.tolist(), strict=True)
            ),
        )
        run.logger.info(
            "%s completed best_epoch=%d validation_macro_pr_auc=%.6f "
            "test_macro_pr_auc=%.6f duration=%.1fs",
            model_name,
            best_epoch,
            validation_metrics["macro"]["pr_auc"],
            test_metrics["macro"]["pr_auc"],
            duration,
        )
    return run
