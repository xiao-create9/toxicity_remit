"""Command-line interface for the Stage A data and protocol infrastructure."""

from __future__ import annotations

import argparse
import json
import sys

from remit.config import ConfigError, load_config
from remit.data.splits import (
    SplitError,
    generate_scaffold_splits,
    summarize_split_files,
    verify_split_files,
)
from remit.data.standardize import (
    DataStandardizationError,
    standardize_dataset,
    verify_processed_dataset,
)
from remit.protocol import RunContext
from remit.training.common import TrainingError


def _shared_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="Root YAML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config key; repeat as needed",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="remit", description="REMIT experiment infrastructure")
    groups = parser.add_subparsers(dest="group", required=True)

    config_group = groups.add_parser("config", help="Resolve and inspect configuration")
    config_commands = config_group.add_subparsers(dest="command", required=True)
    config_show = config_commands.add_parser("show", help="Print the fully resolved configuration")
    _shared_config_arguments(config_show)

    data_group = groups.add_parser("data", help="Prepare and verify molecular data")
    data_commands = data_group.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("standardize", "Standardize and deduplicate the raw CSV"),
        ("split", "Generate the three fixed scaffold splits"),
        ("prepare", "Standardize data and generate all scaffold splits"),
        ("verify", "Verify artifact hashes and leakage gates"),
        ("summary", "Report partition and endpoint statistics"),
    ]:
        command = data_commands.add_parser(name, help=help_text)
        _shared_config_arguments(command)

    protocol_group = groups.add_parser("protocol", help="Exercise the unified run protocol")
    protocol_commands = protocol_group.add_subparsers(dest="command", required=True)
    smoke = protocol_commands.add_parser("smoke", help="Create a completed smoke run")
    _shared_config_arguments(smoke)
    smoke.add_argument("--split-id", type=int, default=0)
    smoke.add_argument("--seed", type=int, default=2026)

    train = groups.add_parser("train", help="Train and evaluate one frozen prediction run")
    _shared_config_arguments(train)
    train.set_defaults(config="configs/train_gine.yaml")
    train.add_argument("--split-id", type=int, required=True)
    train.add_argument("--seed", type=int, required=True)

    report_group = groups.add_parser("report", help="Aggregate completed experiment runs")
    report_commands = report_group.add_subparsers(dest="command", required=True)
    prediction_report = report_commands.add_parser(
        "prediction", help="Aggregate the frozen prediction matrix"
    )
    prediction_report.add_argument("--output", default="reports/stage_a_prediction")
    prediction_report.add_argument("--allow-incomplete", action="store_true")
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.group == "report" and args.command == "prediction":
            from pathlib import Path

            from remit.reporting.prediction import aggregate_prediction_runs

            protocol = aggregate_prediction_runs(
                project_root=Path.cwd(),
                output_dir=Path(args.output),
                require_complete=not args.allow_incomplete,
            )
            _print_json(protocol)
            return 0
        config = load_config(args.config, args.overrides)
        if args.group == "config":
            print(config.to_yaml(), end="")
            print(f"# config_hash: {config.config_hash}")
            return 0

        if args.group == "data" and args.command in {"standardize", "prepare"}:
            result = standardize_dataset(config)
            _print_json(
                {
                    "molecules_path": str(result.molecules_path),
                    "valid_source_rows": result.valid_source_rows,
                    "invalid_source_rows": result.invalid_source_rows,
                    "unique_molecules": result.unique_molecules,
                    "conflict_count": result.conflict_count,
                }
            )
            if args.command == "standardize":
                return 0

        if args.group == "data" and args.command in {"split", "prepare"}:
            artifacts = generate_scaffold_splits(config)
            _print_json(
                [
                    {
                        "split_id": artifact.split_id,
                        "seed": artifact.seed,
                        "index_path": str(artifact.index_path),
                        "counts": artifact.counts,
                    }
                    for artifact in artifacts
                ]
            )
            return 0

        if args.group == "data" and args.command == "verify":
            _print_json(
                {
                    "processed": verify_processed_dataset(config),
                    "splits": verify_split_files(config),
                }
            )
            return 0

        if args.group == "data" and args.command == "summary":
            _print_json(summarize_split_files(config))
            return 0

        if args.group == "protocol" and args.command == "smoke":
            with RunContext(config, split_id=args.split_id, seed=args.seed) as run:
                run.logger.info("Stage A protocol smoke test")
                run.write_metrics("validation", {"status": "smoke", "metric": None})
                run.write_metrics("test", {"status": "not_evaluated", "metric": None})
            _print_json({"run_dir": str(run.run_dir), "status": "completed"})
            return 0
        if args.group == "train":
            from remit.training.runner import run_prediction

            run = run_prediction(config, split_id=args.split_id, seed=args.seed)
            _print_json({"run_dir": str(run.run_dir), "status": "completed"})
            return 0
    except (
        ConfigError,
        DataStandardizationError,
        SplitError,
        TrainingError,
        FileNotFoundError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
