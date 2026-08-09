"""Composable YAML configuration with strict Stage A validation."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from remit.utils import stable_hash


class ConfigError(ValueError):
    """Raised when configuration composition or validation fails."""


_ENV_PATTERN = re.compile(r"^\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?}$")


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    return payload


def _parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ConfigError(f"Override must use key=value syntax: {raw!r}")
    key, value = raw.split("=", 1)
    path = [part for part in key.split(".") if part]
    if not path:
        raise ConfigError(f"Override path cannot be empty: {raw!r}")
    try:
        parsed_value = yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid override value in {raw!r}: {exc}") from exc
    return path, parsed_value


def _set_nested(config: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = config
    for part in path[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"Cannot assign below non-mapping key: {'.'.join(path)}")
        cursor = child
    cursor[path[-1]] = value


def _resolve_env(value: Any) -> Any:
    import os

    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    if isinstance(value, str) and (match := _ENV_PATTERN.match(value)):
        name, default = match.groups()
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ConfigError(f"Required environment variable is missing: {name}")
    return value


@dataclass(frozen=True)
class ResolvedConfig:
    """Resolved configuration plus source metadata."""

    values: dict[str, Any]
    source: Path
    project_root: Path
    config_hash: str

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"Missing or invalid configuration section: {name}")
        return value

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.values, allow_unicode=True, sort_keys=False)


def load_config(
    path: str | Path = "configs/default.yaml", overrides: list[str] | None = None
) -> ResolvedConfig:
    """Compose defaults, apply overrides, resolve environment values, and validate."""
    source = Path(path).expanduser().resolve()
    root_payload = _load_yaml(source)
    defaults = root_payload.pop("defaults", [])
    if not isinstance(defaults, list):
        raise ConfigError("defaults must be a list")

    config: dict[str, Any] = {}
    for item in defaults:
        if not isinstance(item, dict) or len(item) != 1:
            raise ConfigError(f"Each defaults entry must contain exactly one group: {item!r}")
        group, name = next(iter(item.items()))
        include_path = source.parent / str(group) / f"{name}.yaml"
        config = _deep_merge(config, _load_yaml(include_path))
    config = _deep_merge(config, root_payload)

    for raw_override in overrides or []:
        override_path, override_value = _parse_override(raw_override)
        _set_nested(config, override_path, override_value)
    config = _resolve_env(config)

    configured_root = Path(str(config.get("project", {}).get("root", "."))).expanduser()
    project_root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (source.parent.parent / configured_root).resolve()
    )
    validate_config(config)
    return ResolvedConfig(
        values=config,
        source=source,
        project_root=project_root,
        config_hash=stable_hash(config),
    )


def validate_config(config: dict[str, Any]) -> None:
    """Validate the cross-section invariants required by the fixed protocol."""
    required_sections = {"project", "data", "experiment", "split", "reproducibility", "runtime"}
    missing = sorted(required_sections.difference(config))
    if missing:
        raise ConfigError(f"Missing required sections: {', '.join(missing)}")

    data = config["data"]
    labels = data.get("label_columns")
    if not isinstance(labels, list) or not labels or len(set(labels)) != len(labels):
        raise ConfigError("data.label_columns must be a non-empty list of unique names")
    if data.get("conflict_policy") not in {"error", "set_missing"}:
        raise ConfigError("data.conflict_policy must be 'error' or 'set_missing'")

    split = config["split"]
    if split.get("strategy") != "scaffold":
        raise ConfigError("Stage A main protocol currently requires split.strategy=scaffold")
    fractions = split.get("fractions", {})
    expected = {"train", "validation", "test"}
    if set(fractions) != expected:
        raise ConfigError(f"split.fractions must have exactly: {sorted(expected)}")
    if any(not 0 < float(value) < 1 for value in fractions.values()):
        raise ConfigError("Every split fraction must be in (0, 1)")
    if abs(sum(float(value) for value in fractions.values()) - 1.0) > 1e-9:
        raise ConfigError("split.fractions must sum to 1")
    seeds = split.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ConfigError("The standard protocol requires exactly three unique split seeds")
    if any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ConfigError("split.seeds must contain non-negative integers")

    model_seeds = config["reproducibility"].get("model_seeds")
    if not isinstance(model_seeds, list) or len(model_seeds) != 3 or len(set(model_seeds)) != 3:
        raise ConfigError("The standard protocol requires exactly three unique model seeds")


def config_as_json(config: ResolvedConfig) -> str:
    return json.dumps(config.values, ensure_ascii=False, sort_keys=True, indent=2)
