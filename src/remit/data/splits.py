"""Fixed, leakage-checked scaffold split generation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from remit.config import ResolvedConfig
from remit.data.standardize import molecule_table_content_hash
from remit.utils import atomic_write_json, resolve_path, sha256_file, stable_hash


class SplitError(RuntimeError):
    """Raised when a split cannot satisfy the Stage A protocol."""


PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitArtifact:
    split_id: int
    seed: int
    index_path: Path
    metadata_path: Path
    counts: dict[str, int]


def _group_statistics(frame: pd.DataFrame, label_columns: list[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for scaffold, group in frame.groupby("scaffold", sort=True):
        labels: dict[str, dict[str, int]] = {}
        for label in label_columns:
            known = group[label].notna()
            labels[label] = {
                "known": int(known.sum()),
                "positive": int((group.loc[known, label] == 1).sum()),
            }
        groups.append(
            {
                "scaffold": scaffold,
                "indices": group.index.tolist(),
                "size": int(len(group)),
                "labels": labels,
            }
        )
    return groups


def _assignment_score(
    candidate: str,
    group: dict[str, Any],
    sizes: dict[str, int],
    label_stats: dict[str, dict[str, dict[str, int]]],
    targets: dict[str, float],
    global_rates: dict[str, float],
    label_weight: float,
) -> float:
    next_sizes = dict(sizes)
    next_sizes[candidate] += group["size"]
    size_cost = sum(
        ((next_sizes[partition] - targets[partition]) / max(targets[partition], 1.0)) ** 2
        for partition in PARTITIONS
    )

    # Penalize label prevalence drift only for partitions that already contain known labels.
    label_cost = 0.0
    comparisons = 0
    for partition in PARTITIONS:
        for endpoint, global_rate in global_rates.items():
            known = label_stats[partition][endpoint]["known"]
            positive = label_stats[partition][endpoint]["positive"]
            if partition == candidate:
                known += group["labels"][endpoint]["known"]
                positive += group["labels"][endpoint]["positive"]
            if known:
                label_cost += (positive / known - global_rate) ** 2
                comparisons += 1
    normalized_label_cost = label_cost / comparisons if comparisons else 0.0
    return size_cost + label_weight * normalized_label_cost


def _assign_scaffold_groups(
    frame: pd.DataFrame,
    label_columns: list[str],
    fractions: dict[str, float],
    seed: int,
    label_weight: float,
) -> dict[str, str]:
    groups = _group_statistics(frame, label_columns)
    if len(groups) < 3:
        raise SplitError("At least three distinct scaffold groups are required")

    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda item: item["size"], reverse=True)
    targets = {partition: len(frame) * fractions[partition] for partition in PARTITIONS}
    sizes = dict.fromkeys(PARTITIONS, 0)
    label_stats = {
        partition: {endpoint: {"known": 0, "positive": 0} for endpoint in label_columns}
        for partition in PARTITIONS
    }
    global_rates = {}
    for endpoint in label_columns:
        known = frame[endpoint].notna()
        global_rates[endpoint] = (
            float((frame.loc[known, endpoint] == 1).mean()) if known.any() else 0.0
        )

    scaffold_to_partition: dict[str, str] = {}
    for group in groups:
        candidates = list(PARTITIONS)
        rng.shuffle(candidates)
        partition = min(
            candidates,
            key=lambda candidate: _assignment_score(
                candidate,
                group,
                sizes,
                label_stats,
                targets,
                global_rates,
                label_weight,
            ),
        )
        scaffold_to_partition[group["scaffold"]] = partition
        sizes[partition] += group["size"]
        for endpoint in label_columns:
            label_stats[partition][endpoint]["known"] += group["labels"][endpoint]["known"]
            label_stats[partition][endpoint]["positive"] += group["labels"][endpoint]["positive"]

    empty = [partition for partition, size in sizes.items() if size == 0]
    if empty:
        raise SplitError(
            "Scaffold constraints produced empty partitions "
            f"{empty}; use a larger dataset or inspect oversized scaffold groups"
        )
    return scaffold_to_partition


def validate_split(frame: pd.DataFrame, split_frame: pd.DataFrame) -> dict[str, Any]:
    """Fail on missing/duplicate samples or molecule/scaffold leakage."""
    required = {"sample_id", "partition", "scaffold"}
    if not required.issubset(split_frame.columns):
        missing = sorted(required - set(split_frame.columns))
        raise SplitError(f"Split file is missing columns: {missing}")
    if split_frame["sample_id"].duplicated().any():
        raise SplitError("A sample_id occurs more than once in the split")
    expected_ids = set(frame["sample_id"])
    actual_ids = set(split_frame["sample_id"])
    if expected_ids != actual_ids:
        raise SplitError(
            f"Split coverage mismatch: missing={len(expected_ids - actual_ids)}, "
            f"unexpected={len(actual_ids - expected_ids)}"
        )
    unknown_partitions = set(split_frame["partition"]) - set(PARTITIONS)
    if unknown_partitions:
        raise SplitError(f"Unknown partitions: {sorted(unknown_partitions)}")
    if set(split_frame["partition"]) != set(PARTITIONS):
        raise SplitError("All train, validation, and test partitions must be non-empty")

    joined = frame[["sample_id", "canonical_smiles", "scaffold"]].merge(
        split_frame[["sample_id", "partition", "scaffold"]],
        on="sample_id",
        suffixes=("_data", "_split"),
        validate="one_to_one",
    )
    if not (joined["scaffold_data"] == joined["scaffold_split"]).all():
        raise SplitError("Split scaffold values do not match processed data")
    molecule_leaks = joined.groupby("canonical_smiles")["partition"].nunique()
    scaffold_leaks = joined.groupby("scaffold_data")["partition"].nunique()
    if (molecule_leaks > 1).any():
        raise SplitError("A canonical molecule crosses partitions")
    if (scaffold_leaks > 1).any():
        raise SplitError("A scaffold group crosses partitions")

    counts = split_frame["partition"].value_counts().to_dict()
    return {
        "sample_count": int(len(split_frame)),
        "partition_counts": {partition: int(counts.get(partition, 0)) for partition in PARTITIONS},
        "scaffold_counts": {
            partition: int(
                split_frame.loc[split_frame["partition"] == partition, "scaffold"].nunique()
            )
            for partition in PARTITIONS
        },
        "canonical_molecule_leakage": False,
        "scaffold_leakage": False,
    }


def generate_scaffold_splits(config: ResolvedConfig) -> list[SplitArtifact]:
    data_config = config.section("data")
    split_config = config.section("split")
    processed_dir = resolve_path(config.project_root, data_config["output_dir"])
    molecules_path = processed_dir / "molecules.parquet"
    if not molecules_path.is_file():
        raise SplitError(f"Processed molecules do not exist: {molecules_path}")
    frame = pd.read_parquet(molecules_path)
    expected_columns = {"sample_id", "canonical_smiles", "scaffold", *data_config["label_columns"]}
    if missing := expected_columns.difference(frame.columns):
        raise SplitError(f"Processed data is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any() or frame["canonical_smiles"].duplicated().any():
        raise SplitError("Processed dataset must contain unique sample IDs and canonical molecules")

    output_dir = resolve_path(config.project_root, split_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fractions = {key: float(value) for key, value in split_config["fractions"].items()}
    artifacts: list[SplitArtifact] = []
    for split_id, seed in enumerate(split_config["seeds"]):
        assignments = _assign_scaffold_groups(
            frame=frame,
            label_columns=list(data_config["label_columns"]),
            fractions=fractions,
            seed=int(seed),
            label_weight=float(split_config.get("label_balance_weight", 0.0)),
        )
        split_frame = frame[["sample_id", "scaffold"]].copy()
        split_frame["partition"] = split_frame["scaffold"].map(assignments)
        split_frame = split_frame[["sample_id", "partition", "scaffold"]].sort_values(
            ["partition", "sample_id"]
        )
        audit = validate_split(frame, split_frame)
        index_path = output_dir / f"split_{split_id}.csv"
        metadata_path = output_dir / f"split_{split_id}.json"
        split_frame.to_csv(index_path, index=False)
        metadata = {
            "schema_version": 1,
            "dataset": data_config["name"],
            "strategy": "scaffold",
            "split_id": split_id,
            "seed": int(seed),
            "fractions": fractions,
            "label_balance_weight": float(split_config.get("label_balance_weight", 0.0)),
            "processed_data_path": str(Path(data_config["output_dir"]) / "molecules.parquet"),
            "processed_data_content_sha256": molecule_table_content_hash(
                frame, list(data_config["label_columns"])
            ),
            "split_config_hash": stable_hash(split_config),
            "index_sha256": sha256_file(index_path),
            "audit": audit,
        }
        atomic_write_json(metadata_path, metadata)
        artifacts.append(
            SplitArtifact(
                split_id=split_id,
                seed=int(seed),
                index_path=index_path,
                metadata_path=metadata_path,
                counts=audit["partition_counts"],
            )
        )
    return artifacts


def verify_split_files(config: ResolvedConfig) -> list[dict[str, Any]]:
    """Revalidate all configured split artifacts and their hashes."""
    data_config = config.section("data")
    split_config = config.section("split")
    molecules_path = (
        resolve_path(config.project_root, data_config["output_dir"]) / "molecules.parquet"
    )
    if not molecules_path.is_file():
        raise SplitError(f"Processed molecules do not exist: {molecules_path}")
    frame = pd.read_parquet(molecules_path)
    output_dir = resolve_path(config.project_root, split_config["output_dir"])
    audits: list[dict[str, Any]] = []
    for split_id, expected_seed in enumerate(split_config["seeds"]):
        index_path = output_dir / f"split_{split_id}.csv"
        metadata_path = output_dir / f"split_{split_id}.json"
        if not index_path.is_file() or not metadata_path.is_file():
            raise SplitError(f"Missing split artifact for split_id={split_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("seed") != expected_seed:
            raise SplitError(f"Seed mismatch for split_id={split_id}")
        content_hash = molecule_table_content_hash(frame, list(data_config["label_columns"]))
        if metadata.get("processed_data_content_sha256") != content_hash:
            raise SplitError(f"Processed data hash mismatch for split_id={split_id}")
        if metadata.get("index_sha256") != sha256_file(index_path):
            raise SplitError(f"Split index hash mismatch for split_id={split_id}")
        audit = validate_split(frame, pd.read_csv(index_path, dtype=str))
        audits.append({"split_id": split_id, "seed": expected_seed, **audit})
    return audits


def summarize_split_files(config: ResolvedConfig) -> dict[str, Any]:
    """Return paper-ready partition, endpoint, and cross-seed split statistics."""
    data_config = config.section("data")
    split_config = config.section("split")
    molecules_path = (
        resolve_path(config.project_root, data_config["output_dir"]) / "molecules.parquet"
    )
    frame = pd.read_parquet(molecules_path)
    output_dir = resolve_path(config.project_root, split_config["output_dir"])
    assignments: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for split_id, seed in enumerate(split_config["seeds"]):
        split_frame = pd.read_csv(output_dir / f"split_{split_id}.csv", dtype=str)
        validate_split(frame, split_frame)
        assignments.append(split_frame.set_index("sample_id").sort_index())
        joined = frame.merge(split_frame, on=["sample_id", "scaffold"], validate="one_to_one")
        partition_summary: dict[str, Any] = {}
        for partition in PARTITIONS:
            subset = joined.loc[joined["partition"] == partition]
            partition_summary[partition] = {
                "samples": int(len(subset)),
                "fraction": float(len(subset) / len(joined)),
                "scaffolds": int(subset["scaffold"].nunique()),
                "endpoints": {
                    endpoint: {
                        "known": int(subset[endpoint].notna().sum()),
                        "positive": int((subset[endpoint] == 1).sum()),
                        "negative": int((subset[endpoint] == 0).sum()),
                        "positive_rate": (
                            float((subset[endpoint] == 1).sum() / subset[endpoint].notna().sum())
                            if subset[endpoint].notna().any()
                            else None
                        ),
                    }
                    for endpoint in data_config["label_columns"]
                },
            }
        summaries.append({"split_id": split_id, "seed": seed, "partitions": partition_summary})

    comparisons: list[dict[str, Any]] = []
    for left_id in range(len(assignments)):
        for right_id in range(left_id + 1, len(assignments)):
            left = assignments[left_id]
            right = assignments[right_id]
            if not left.index.equals(right.index):
                raise SplitError("Configured splits do not reference identical sample IDs")
            left_test = set(left.index[left["partition"] == "test"])
            right_test = set(right.index[right["partition"] == "test"])
            comparisons.append(
                {
                    "left_split_id": left_id,
                    "right_split_id": right_id,
                    "partition_agreement": float((left["partition"] == right["partition"]).mean()),
                    "changed_partition_fraction": float(
                        (left["partition"] != right["partition"]).mean()
                    ),
                    "test_jaccard": float(
                        len(left_test & right_test) / len(left_test | right_test)
                    ),
                }
            )
    return {
        "dataset": data_config["name"],
        "processed_data_content_sha256": molecule_table_content_hash(
            frame, list(data_config["label_columns"])
        ),
        "sample_count": int(len(frame)),
        "split_config_hash": stable_hash(split_config),
        "splits": summaries,
        "cross_seed_comparisons": comparisons,
    }
