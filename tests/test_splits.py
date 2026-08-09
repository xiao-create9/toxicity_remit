from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from remit.config import ResolvedConfig
from remit.data.splits import generate_scaffold_splits, verify_split_files
from remit.data.standardize import standardize_dataset
from remit.protocol import ProtocolViolation, SplitAccessGuard

MOLECULES = [
    "c1ccccc1",
    "c1ccncc1",
    "C1CCCCC1",
    "C1CCNCC1",
    "c1ccc2ccccc2c1",
    "c1ccc2[nH]ccc2c1",
    "O=C1CCCCC1",
    "O=C1NCCCC1",
    "CCO",
    "CCN",
    "CCC(=O)O",
    "CC(C)O",
    "c1ccc(cc1)Cl",
    "c1ccc(cc1)F",
    "C1COCCO1",
    "C1CSCCS1",
    "c1ncc[nH]1",
    "c1ncn[nH]1",
    "O=C1OC=CC1",
    "O=C1NC=CC1",
]


def _prepare(config: ResolvedConfig, root: Path) -> None:
    rows = [
        {
            "sample_id": f"sample_{index}",
            "smiles": smiles,
            "endpoint_a": index % 2,
            "endpoint_b": (index // 2) % 2,
        }
        for index, smiles in enumerate(MOLECULES)
    ]
    pd.DataFrame(rows).to_csv(root / "data" / "raw" / "molecules.csv", index=False)
    standardize_dataset(config)


def test_three_scaffold_splits_are_explicit_reproducible_and_leak_free(
    project_config: tuple[ResolvedConfig, Path],
) -> None:
    config, root = project_config
    _prepare(config, root)
    first = generate_scaffold_splits(config)
    first_contents = [artifact.index_path.read_text(encoding="utf-8") for artifact in first]
    second = generate_scaffold_splits(config)
    second_contents = [artifact.index_path.read_text(encoding="utf-8") for artifact in second]
    assert first_contents == second_contents
    audits = verify_split_files(config)
    assert len(audits) == 3
    assert all(not audit["canonical_molecule_leakage"] for audit in audits)
    assert all(not audit["scaffold_leakage"] for audit in audits)
    assert all(set(artifact.counts) == {"train", "validation", "test"} for artifact in first)


def test_split_access_guard_blocks_test_during_selection(
    project_config: tuple[ResolvedConfig, Path],
) -> None:
    config, root = project_config
    _prepare(config, root)
    artifact = generate_scaffold_splits(config)[0]
    molecules_path = root / "data" / "processed" / "fixture" / "molecules.parquet"
    events_path = root / "events.jsonl"
    guard = SplitAccessGuard(molecules_path, artifact.index_path, events_path)

    with pytest.raises(ProtocolViolation, match="final_evaluation"):
        guard.load("test", "model_selection")
    test_frame = guard.load("test", "final_evaluation")
    assert not test_frame.empty
    assert len(events_path.read_text(encoding="utf-8").splitlines()) == 2
