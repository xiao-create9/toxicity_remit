from __future__ import annotations

from pathlib import Path

import pandas as pd

from remit.cli import main
from tests.test_splits import MOLECULES


def test_prepare_verify_and_protocol_smoke_cli(project_config: tuple[object, Path]) -> None:
    _, root = project_config
    rows = [
        {
            "sample_id": f"cli_{index}",
            "smiles": smiles,
            "endpoint_a": index % 2,
            "endpoint_b": (index // 2) % 2,
        }
        for index, smiles in enumerate(MOLECULES)
    ]
    pd.DataFrame(rows).to_csv(root / "data" / "raw" / "molecules.csv", index=False)
    config_path = root / "configs" / "default.yaml"

    assert main(["data", "prepare", "--config", str(config_path)]) == 0
    assert main(["data", "verify", "--config", str(config_path)]) == 0
    assert main(["data", "summary", "--config", str(config_path)]) == 0
    assert (
        main(
            [
                "protocol",
                "smoke",
                "--config",
                str(config_path),
                "--split-id",
                "0",
                "--seed",
                "101",
            ]
        )
        == 0
    )
