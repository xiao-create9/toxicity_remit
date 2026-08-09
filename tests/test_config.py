from __future__ import annotations

from pathlib import Path

import pytest

from remit.config import ConfigError, load_config


def test_config_composition_and_override(project_config: tuple[object, Path]) -> None:
    _, project_root = project_config
    resolved = load_config(
        project_root / "configs" / "default.yaml",
        ["runtime.num_workers=4", "data.standardization.uncharge=false"],
    )
    assert resolved.project_root == project_root
    assert resolved.values["runtime"]["num_workers"] == 4
    assert resolved.values["data"]["standardization"]["uncharge"] is False
    assert len(resolved.config_hash) == 64


def test_standard_protocol_rejects_wrong_seed_count(project_config: tuple[object, Path]) -> None:
    _, project_root = project_config
    with pytest.raises(ConfigError, match="exactly three unique split seeds"):
        load_config(project_root / "configs" / "default.yaml", ["split.seeds=[1,2]"])
