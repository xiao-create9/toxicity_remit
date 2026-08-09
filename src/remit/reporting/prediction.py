"""Aggregate the frozen 3-split x 3-seed prediction matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from remit.config import load_config
from remit.utils import atomic_write_json, atomic_write_text

STANDARD_CONFIGS = {
    "ecfp_rf": "configs/train_ecfp_rf.yaml",
    "gine": "configs/train_gine.yaml",
    "attentivefp": "configs/train_attentivefp.yaml",
}
METRICS = (
    "pr_auc",
    "roc_auc",
    "mcc",
    "balanced_accuracy",
    "recall",
    "specificity",
    "ece",
    "brier",
)


def _flatten_summary(grouped: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    summary = grouped[list(METRICS)].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary


def _markdown_macro_table(summary: pd.DataFrame) -> str:
    test = summary.loc[summary["partition"] == "test"].sort_values("model")
    lines = [
        "| Model | Runs | PR-AUC | ROC-AUC | MCC | ECE | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in test.iterrows():
        lines.append(
            "| {model} | {runs:.0f} | {pr:.4f} ± {pr_std:.4f} | "
            "{roc:.4f} ± {roc_std:.4f} | {mcc:.4f} ± {mcc_std:.4f} | "
            "{ece:.4f} ± {ece_std:.4f} | {brier:.4f} ± {brier_std:.4f} |".format(
                model=row["model"],
                runs=row["pr_auc_count"],
                pr=row["pr_auc_mean"],
                pr_std=row["pr_auc_std"],
                roc=row["roc_auc_mean"],
                roc_std=row["roc_auc_std"],
                mcc=row["mcc_mean"],
                mcc_std=row["mcc_std"],
                ece=row["ece_mean"],
                ece_std=row["ece_std"],
                brier=row["brier_mean"],
                brier_std=row["brier_std"],
            )
        )
    return "\n".join(lines)


def aggregate_prediction_runs(
    project_root: Path, output_dir: Path, require_complete: bool = True
) -> dict[str, Any]:
    expected_configs = {
        model: load_config(project_root / path) for model, path in STANDARD_CONFIGS.items()
    }
    experiment = next(iter(expected_configs.values())).section("experiment")["name"]
    dataset = next(iter(expected_configs.values())).section("data")["name"]
    runs_root = project_root / "runs" / experiment / dataset
    candidates: dict[tuple[str, int, int], list[tuple[str, Path, dict[str, Any]]]] = {}
    excluded: list[dict[str, Any]] = []
    for manifest_path in runs_root.rglob("manifest.json") if runs_root.exists() else []:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = manifest.get("model")
        if model not in expected_configs:
            excluded.append({"run_dir": str(manifest_path.parent), "reason": "unknown_model"})
            continue
        if manifest.get("config_hash") != expected_configs[model].config_hash:
            excluded.append({"run_dir": str(manifest_path.parent), "reason": "nonstandard_config"})
            continue
        if manifest.get("status") != "completed":
            excluded.append({"run_dir": str(manifest_path.parent), "reason": "not_completed"})
            continue
        key = (model, int(manifest["split_id"]), int(manifest["seed"]))
        candidates.setdefault(key, []).append(
            (str(manifest.get("completed_at", "")), manifest_path.parent, manifest)
        )

    expected_keys = {
        (model, split_id, seed)
        for model, config in expected_configs.items()
        for split_id in range(len(config.section("split")["seeds"]))
        for seed in config.section("reproducibility")["model_seeds"]
    }
    selected: dict[tuple[str, int, int], tuple[Path, dict[str, Any]]] = {}
    for key, runs in candidates.items():
        runs.sort(key=lambda item: item[0])
        _, run_dir, manifest = runs[-1]
        selected[key] = (run_dir, manifest)
        for _, duplicate_dir, _ in runs[:-1]:
            excluded.append({"run_dir": str(duplicate_dir), "reason": "superseded"})
    missing = sorted(expected_keys - set(selected))
    if require_complete and missing:
        raise RuntimeError(
            f"Prediction matrix is incomplete; missing {len(missing)} runs: {missing}"
        )

    run_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    for (model, split_id, seed), (run_dir, manifest) in sorted(selected.items()):
        for partition in ("validation", "test"):
            metrics = json.loads(
                (run_dir / f"metrics_{partition}.json").read_text(encoding="utf-8")
            )
            run_rows.append(
                {
                    "model": model,
                    "partition": partition,
                    "split_id": split_id,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "config_hash": manifest["config_hash"],
                    **metrics["macro"],
                }
            )
            for endpoint, values in metrics["endpoints"].items():
                endpoint_rows.append(
                    {
                        "model": model,
                        "partition": partition,
                        "split_id": split_id,
                        "seed": seed,
                        "endpoint": endpoint,
                        **values,
                    }
                )

    runs_frame = pd.DataFrame(run_rows)
    endpoints_frame = pd.DataFrame(endpoint_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_frame.to_csv(output_dir / "included_runs.csv", index=False)
    pd.DataFrame(excluded, columns=["run_dir", "reason"]).to_csv(
        output_dir / "excluded_runs.csv", index=False
    )
    macro_summary = _flatten_summary(runs_frame.groupby(["model", "partition"], dropna=False))
    endpoint_summary = _flatten_summary(
        endpoints_frame.groupby(["model", "partition", "endpoint"], dropna=False)
    )
    macro_summary.to_csv(output_dir / "prediction_summary.csv", index=False)
    endpoint_summary.to_csv(output_dir / "endpoint_summary.csv", index=False)
    protocol = {
        "dataset": dataset,
        "experiment": experiment,
        "required_runs": len(expected_keys),
        "included_runs": len(selected),
        "missing_runs": [list(key) for key in missing],
        "config_hashes": {model: config.config_hash for model, config in expected_configs.items()},
    }
    atomic_write_json(output_dir / "protocol.json", protocol)
    report = (
        "# Stage A prediction summary\n\n"
        f"Included {len(selected)}/{len(expected_keys)} frozen runs.\n\n"
        + _markdown_macro_table(macro_summary)
        + "\n"
    )
    atomic_write_text(output_dir / "report.md", report)
    return protocol
