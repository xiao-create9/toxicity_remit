"""Deterministic molecular standardization and duplicate-label auditing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdchem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from remit.config import ResolvedConfig
from remit.utils import atomic_write_json, resolve_path, sha256_file, stable_hash


class DataStandardizationError(RuntimeError):
    """Raised when input data cannot satisfy the fixed processing protocol."""


@dataclass(frozen=True)
class StandardizationResult:
    output_dir: Path
    molecules_path: Path
    manifest_path: Path
    valid_source_rows: int
    invalid_source_rows: int
    unique_molecules: int
    conflict_count: int


def _sample_id(canonical_smiles: str) -> str:
    digest = hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()[:16]
    return f"mol_{digest}"


def _parse_label(value: Any, missing_values: set[str]) -> int | None:
    if value is None or pd.isna(value):
        return None
    rendered = str(value).strip()
    if rendered in missing_values:
        return None
    try:
        numeric = float(rendered)
    except ValueError as exc:
        raise ValueError(f"label is neither 0, 1, nor missing: {value!r}") from exc
    if numeric not in {0.0, 1.0}:
        raise ValueError(f"label is neither 0, 1, nor missing: {value!r}")
    return int(numeric)


class MoleculeStandardizer:
    """Apply the exact configured RDKit normalization pipeline."""

    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options
        self.fragment_chooser = rdMolStandardize.LargestFragmentChooser(
            preferOrganic=bool(options.get("prefer_organic", True))
        )
        self.uncharger = rdMolStandardize.Uncharger()
        self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

    def standardize(self, smiles: str) -> tuple[str, str]:
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("SMILES is empty")
        mol = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
        if mol is None:
            raise ValueError("RDKit failed to parse or sanitize SMILES")
        try:
            if self.options.get("cleanup", True):
                mol = rdMolStandardize.Cleanup(mol)
            if self.options.get("largest_fragment", True):
                mol = self.fragment_chooser.choose(mol)
            if self.options.get("uncharge", True):
                mol = self.uncharger.uncharge(mol)
            if self.options.get("canonical_tautomer", False):
                mol = self.tautomer_enumerator.Canonicalize(mol)
            Chem.SanitizeMol(mol)
        except (ValueError, RuntimeError, rdchem.KekulizeException) as exc:
            raise ValueError(f"RDKit standardization failed: {exc}") from exc

        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=bool(self.options.get("isomeric_smiles", True)),
        )
        if not canonical:
            raise ValueError("standardization produced an empty canonical SMILES")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        # An empty Murcko scaffold is shared by all acyclic molecules. Treating it as one
        # group makes a scaffold split degenerate, so each acyclic molecule is its own
        # explicit scaffold group while duplicate canonical molecules remain grouped.
        scaffold_group = scaffold or f"ACYCLIC:{canonical}"
        return canonical, scaffold_group


def _validate_columns(frame: pd.DataFrame, data_config: dict[str, Any]) -> None:
    required = {data_config["smiles_column"], *data_config["label_columns"]}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataStandardizationError(f"Input CSV is missing required columns: {missing}")


def standardize_dataset(config: ResolvedConfig) -> StandardizationResult:
    """Standardize a CSV and emit molecule, invalid, conflict, and manifest artifacts."""
    data_config = config.section("data")
    input_path = resolve_path(config.project_root, data_config["input_path"])
    output_dir = resolve_path(config.project_root, data_config["output_dir"])
    if not input_path.is_file():
        raise DataStandardizationError(f"Raw dataset does not exist: {input_path}")

    frame = pd.read_csv(input_path, dtype=object, keep_default_na=False)
    _validate_columns(frame, data_config)
    standardizer = MoleculeStandardizer(data_config.get("standardization", {}))
    smiles_column = data_config["smiles_column"]
    configured_id_column = data_config.get("id_column")
    id_column = configured_id_column if configured_id_column in frame.columns else None
    label_columns = list(data_config["label_columns"])
    missing_values = {str(value) for value in data_config.get("missing_values", [])}

    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        source_id = str(row[id_column]) if id_column else str(row_index)
        original_smiles = str(row[smiles_column])
        try:
            canonical, scaffold = standardizer.standardize(original_smiles)
            labels = {label: _parse_label(row[label], missing_values) for label in label_columns}
        except (ValueError, TypeError) as exc:
            invalid_rows.append(
                {
                    "source_row_index": int(row_index),
                    "source_id": source_id,
                    "original_smiles": original_smiles,
                    "reason": str(exc),
                }
            )
            continue
        valid_rows.append(
            {
                "source_row_index": int(row_index),
                "source_id": source_id,
                "original_smiles": original_smiles,
                "canonical_smiles": canonical,
                "sample_id": _sample_id(canonical),
                "scaffold": scaffold,
                **labels,
            }
        )

    if not valid_rows:
        raise DataStandardizationError("No valid molecules remained after standardization")
    source_rows = pd.DataFrame(valid_rows)
    collisions = source_rows.groupby("sample_id")["canonical_smiles"].nunique()
    if (collisions > 1).any():
        raise DataStandardizationError("Stable sample ID collision detected")

    conflicts: list[dict[str, Any]] = []
    molecules: list[dict[str, Any]] = []
    conflict_policy = data_config.get("conflict_policy", "set_missing")
    for canonical, group in source_rows.groupby("canonical_smiles", sort=True):
        first = group.iloc[0]
        molecule: dict[str, Any] = {
            "sample_id": first["sample_id"],
            "canonical_smiles": canonical,
            "scaffold": first["scaffold"],
            "source_ids": json.dumps(sorted(group["source_id"].astype(str).tolist())),
            "source_row_count": int(len(group)),
        }
        for label in label_columns:
            observed = sorted({int(value) for value in group[label].dropna().tolist()})
            if len(observed) > 1:
                conflict = {
                    "sample_id": first["sample_id"],
                    "canonical_smiles": canonical,
                    "endpoint": label,
                    "observed_labels": json.dumps(observed),
                    "source_ids": molecule["source_ids"],
                }
                conflicts.append(conflict)
                if conflict_policy == "error":
                    raise DataStandardizationError(
                        f"Conflicting duplicate labels for {first['sample_id']} endpoint {label}"
                    )
                molecule[label] = pd.NA
            else:
                molecule[label] = observed[0] if observed else pd.NA
        molecules.append(molecule)

    molecules_frame = pd.DataFrame(molecules).sort_values("sample_id").reset_index(drop=True)
    for label in label_columns:
        molecules_frame[label] = molecules_frame[label].astype("Int8")

    output_dir.mkdir(parents=True, exist_ok=True)
    molecules_path = output_dir / "molecules.parquet"
    invalid_path = output_dir / "invalid_molecules.csv"
    conflicts_path = output_dir / "duplicate_conflicts.csv"
    source_rows_path = output_dir / "source_rows.csv"
    report_path = output_dir / "standardization_report.json"
    manifest_path = output_dir / "manifest.json"

    molecules_frame.to_parquet(molecules_path, index=False)
    pd.DataFrame(
        invalid_rows,
        columns=["source_row_index", "source_id", "original_smiles", "reason"],
    ).to_csv(invalid_path, index=False)
    pd.DataFrame(
        conflicts,
        columns=["sample_id", "canonical_smiles", "endpoint", "observed_labels", "source_ids"],
    ).to_csv(conflicts_path, index=False)
    source_rows.to_csv(source_rows_path, index=False)

    report = {
        "dataset": data_config["name"],
        "source_rows": int(len(frame)),
        "valid_source_rows": int(len(source_rows)),
        "invalid_source_rows": len(invalid_rows),
        "unique_molecules": int(len(molecules_frame)),
        "duplicate_source_rows": int(len(source_rows) - len(molecules_frame)),
        "conflicting_endpoint_labels": len(conflicts),
        "endpoint_counts": {
            label: {
                "known": int(molecules_frame[label].notna().sum()),
                "positive": int((molecules_frame[label] == 1).sum()),
                "negative": int((molecules_frame[label] == 0).sum()),
            }
            for label in label_columns
        },
    }
    atomic_write_json(report_path, report)
    artifacts = {
        path.name: sha256_file(path)
        for path in [molecules_path, invalid_path, conflicts_path, source_rows_path, report_path]
    }
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": data_config["name"],
        "source_path": str(input_path),
        "source_sha256": sha256_file(input_path),
        "processing_config": data_config,
        "processing_config_hash": stable_hash(data_config),
        "rdkit_version": rdBase.rdkitVersion,
        "artifacts": artifacts,
        "report": report,
    }
    atomic_write_json(manifest_path, manifest)
    return StandardizationResult(
        output_dir=output_dir,
        molecules_path=molecules_path,
        manifest_path=manifest_path,
        valid_source_rows=len(source_rows),
        invalid_source_rows=len(invalid_rows),
        unique_molecules=len(molecules_frame),
        conflict_count=len(conflicts),
    )


def verify_processed_dataset(config: ResolvedConfig) -> dict[str, Any]:
    """Verify that every processed artifact still matches its recorded digest."""
    data_config = config.section("data")
    output_dir = resolve_path(config.project_root, data_config["output_dir"])
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DataStandardizationError(f"Processing manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("processing_config_hash") != stable_hash(data_config):
        raise DataStandardizationError(
            "Current data configuration differs from the processing manifest"
        )
    mismatches: list[str] = []
    for filename, expected_digest in manifest.get("artifacts", {}).items():
        artifact = output_dir / filename
        if not artifact.is_file() or sha256_file(artifact) != expected_digest:
            mismatches.append(filename)
    if mismatches:
        raise DataStandardizationError(f"Processed artifact hash mismatch: {mismatches}")
    return {
        "dataset": manifest["dataset"],
        "source_sha256": manifest["source_sha256"],
        "processing_config_hash": manifest["processing_config_hash"],
        "verified_artifacts": sorted(manifest["artifacts"]),
        "report": manifest["report"],
    }
