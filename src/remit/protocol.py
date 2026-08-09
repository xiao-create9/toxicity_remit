"""Unified run directories, manifests, logging, and test-access safeguards."""

from __future__ import annotations

import json
import logging
import traceback
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import pandas as pd

from remit.config import ResolvedConfig
from remit.reproducibility import environment_manifest, seed_everything
from remit.utils import atomic_write_json, atomic_write_text, resolve_path, sha256_file


class ProtocolViolation(RuntimeError):
    """Raised on test leakage or a run-protocol violation."""


RunPhase = Literal["training", "model_selection", "final_evaluation"]


class SplitAccessGuard:
    """Load explicit split partitions while enforcing the test-access policy."""

    def __init__(self, molecules_path: Path, split_path: Path, events_path: Path | None = None):
        self.molecules_path = molecules_path
        self.split_path = split_path
        self.events_path = events_path

    def load(self, partition: str, phase: RunPhase) -> pd.DataFrame:
        if partition not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown partition: {partition}")
        if partition == "test" and phase != "final_evaluation":
            self._record(partition, phase, "denied")
            raise ProtocolViolation(
                "Test data may only be accessed during final_evaluation, never during "
                "training or model_selection"
            )
        molecules = pd.read_parquet(self.molecules_path)
        split = pd.read_csv(self.split_path, dtype={"sample_id": str})
        selected = split.loc[split["partition"] == partition, ["sample_id"]]
        result = selected.merge(molecules, on="sample_id", how="left", validate="one_to_one")
        if len(result) != len(selected) or result["canonical_smiles"].isna().any():
            raise ProtocolViolation("Split index and processed dataset do not match")
        self._record(partition, phase, "allowed")
        return result

    def _record(self, partition: str, phase: RunPhase, outcome: str) -> None:
        if self.events_path is None:
            return
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "partition": partition,
            "phase": phase,
            "outcome": outcome,
            "molecules_sha256": sha256_file(self.molecules_path),
            "split_sha256": sha256_file(self.split_path),
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


class RunContext(AbstractContextManager["RunContext"]):
    """Context manager that makes every run self-describing, including failures."""

    def __init__(
        self,
        config: ResolvedConfig,
        split_id: int,
        seed: int,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.split_id = split_id
        self.seed = seed
        data = config.section("data")
        experiment = config.section("experiment")
        runtime = config.section("runtime")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = run_id or f"{timestamp}-{config.config_hash[:8]}"
        self.run_dir = (
            resolve_path(config.project_root, runtime["runs_dir"])
            / str(experiment["name"])
            / str(data["name"])
            / str(split_id)
            / str(seed)
            / self.run_id
        )
        self.manifest_path = self.run_dir / "manifest.json"
        self.logger = logging.getLogger(f"remit.run.{self.run_id}")
        self._started_at: str | None = None
        self.summary: dict[str, Any] = {}

    def __enter__(self) -> Self:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._started_at = datetime.now(UTC).isoformat()
        atomic_write_text(self.run_dir / "config.yaml", self.config.to_yaml())
        self._configure_logging()
        reproducibility = self.config.section("reproducibility")
        seed_everything(
            self.seed,
            deterministic_algorithms=bool(reproducibility.get("deterministic_algorithms", True)),
            warn_only=bool(reproducibility.get("warn_only", False)),
        )
        manifest = self._base_manifest(status="running")
        atomic_write_json(self.manifest_path, manifest)
        self.logger.info("Run started: %s", self.run_dir)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback_object: TracebackType | None,
    ) -> bool:
        if exc is None:
            manifest = self._base_manifest(status="completed")
            manifest["completed_at"] = datetime.now(UTC).isoformat()
            atomic_write_json(self.manifest_path, manifest)
            self.logger.info("Run completed")
        else:
            failure = {
                "type": exc_type.__name__ if exc_type else type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc_type, exc, traceback_object)),
                "failed_at": datetime.now(UTC).isoformat(),
            }
            atomic_write_json(self.run_dir / "failure.json", failure)
            manifest = self._base_manifest(status="failed")
            manifest["failed_at"] = failure["failed_at"]
            atomic_write_json(self.manifest_path, manifest)
            self.logger.exception("Run failed")
        self._close_logging()
        return False

    def split_guard(self) -> SplitAccessGuard:
        data = self.config.section("data")
        split = self.config.section("split")
        molecules_path = (
            resolve_path(self.config.project_root, data["output_dir"]) / "molecules.parquet"
        )
        split_path = (
            resolve_path(self.config.project_root, split["output_dir"])
            / f"split_{self.split_id}.csv"
        )
        return SplitAccessGuard(molecules_path, split_path, self.run_dir / "data_access.jsonl")

    def write_metrics(
        self, partition: Literal["validation", "test"], metrics: dict[str, Any]
    ) -> None:
        atomic_write_json(self.run_dir / f"metrics_{partition}.json", metrics)

    def record_summary(self, **values: Any) -> None:
        """Add model/training facts that persist in the final run manifest."""
        self.summary.update(values)

    def _base_manifest(self, status: str) -> dict[str, Any]:
        manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": status,
            "started_at": self._started_at,
            "experiment": self.config.section("experiment")["name"],
            "dataset": self.config.section("data")["name"],
            "split_id": self.split_id,
            "seed": self.seed,
            "config_hash": self.config.config_hash,
            "config_source": str(self.config.source),
            "environment": environment_manifest(self.config.project_root),
            "inputs": self._input_hashes(),
        }
        manifest.update(self.summary)
        return manifest

    def _input_hashes(self) -> dict[str, Any]:
        data = self.config.section("data")
        split = self.config.section("split")
        processed = resolve_path(self.config.project_root, data["output_dir"]) / "molecules.parquet"
        split_index = (
            resolve_path(self.config.project_root, split["output_dir"])
            / f"split_{self.split_id}.csv"
        )
        return {
            "processed_data": {
                "path": str(processed),
                "sha256": sha256_file(processed) if processed.is_file() else None,
            },
            "split_index": {
                "path": str(split_index),
                "sha256": sha256_file(split_index) if split_index.is_file() else None,
            },
        }

    def _configure_logging(self) -> None:
        self.logger.setLevel(self.config.section("runtime").get("log_level", "INFO"))
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for handler in [logging.FileHandler(self.run_dir / "train.log"), logging.StreamHandler()]:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _close_logging(self) -> None:
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
