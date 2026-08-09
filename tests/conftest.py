from __future__ import annotations

from pathlib import Path

import pytest

from remit.config import ResolvedConfig, load_config


@pytest.fixture
def project_config(tmp_path: Path) -> tuple[ResolvedConfig, Path]:
    configs = tmp_path / "configs"
    for folder in ["data", "experiment", "runtime"]:
        (configs / folder).mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (configs / "default.yaml").write_text(
        """defaults:
  - data: fixture
  - experiment: stage_a
  - runtime: standard
project:
  name: test_remit
  root: .
""",
        encoding="utf-8",
    )
    (configs / "data" / "fixture.yaml").write_text(
        """data:
  name: fixture
  input_path: data/raw/molecules.csv
  output_dir: data/processed/fixture
  smiles_column: smiles
  id_column: sample_id
  label_columns: [endpoint_a, endpoint_b]
  missing_values: ["", "NA"]
  conflict_policy: set_missing
  standardization:
    cleanup: true
    largest_fragment: true
    prefer_organic: true
    uncharge: true
    canonical_tautomer: false
    isomeric_smiles: true
""",
        encoding="utf-8",
    )
    (configs / "experiment" / "stage_a.yaml").write_text(
        """experiment:
  name: stage_a_test
split:
  strategy: scaffold
  output_dir: data/splits/fixture/scaffold
  seeds: [13, 37, 73]
  fractions: {train: 0.8, validation: 0.1, test: 0.1}
  label_balance_weight: 0.25
reproducibility:
  model_seeds: [101, 102, 103]
  deterministic_algorithms: true
  warn_only: false
""",
        encoding="utf-8",
    )
    (configs / "runtime" / "standard.yaml").write_text(
        """runtime:
  runs_dir: runs
  reports_dir: reports
  log_level: INFO
  gpu_ids: [0, 1]
  num_workers: 0
""",
        encoding="utf-8",
    )
    config = load_config(configs / "default.yaml")
    return config, tmp_path
