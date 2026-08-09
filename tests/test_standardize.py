from __future__ import annotations

from pathlib import Path

import pandas as pd

from remit.config import ResolvedConfig
from remit.data.standardize import standardize_dataset, verify_processed_dataset


def test_standardization_audits_invalid_duplicates_and_conflicts(
    project_config: tuple[ResolvedConfig, Path],
) -> None:
    config, root = project_config
    pd.DataFrame(
        [
            {"sample_id": "a", "smiles": "CCO.[Na+]", "endpoint_a": 1, "endpoint_b": 0},
            {"sample_id": "b", "smiles": "CCO", "endpoint_a": 0, "endpoint_b": 0},
            {"sample_id": "c", "smiles": "c1ccccc1", "endpoint_a": 1, "endpoint_b": ""},
            {"sample_id": "bad", "smiles": "not-a-smiles", "endpoint_a": 1, "endpoint_b": 1},
        ]
    ).to_csv(root / "data" / "raw" / "molecules.csv", index=False)

    result = standardize_dataset(config)

    assert result.valid_source_rows == 3
    assert result.invalid_source_rows == 1
    assert result.unique_molecules == 2
    assert result.conflict_count == 1
    molecules = pd.read_parquet(result.molecules_path)
    ethanol = molecules.loc[molecules["canonical_smiles"] == "CCO"].iloc[0]
    assert pd.isna(ethanol["endpoint_a"])
    assert ethanol["endpoint_b"] == 0
    invalid = pd.read_csv(result.output_dir / "invalid_molecules.csv")
    assert invalid.loc[0, "source_id"] == "bad"
    verification = verify_processed_dataset(config)
    assert len(verification["verified_artifacts"]) == 5
