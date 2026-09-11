from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from . import __version__
from .analyzer import ConfigurationError
from .chat import CHAT_SCHEMAS, SCHEMAS, adapt_chat, analyze_chat
from .comparison import compare_splits
from .models import ScanConfig, SplitInput
from .readers import DatasetReadError, read_dataset
from .report import (
    build_affected_records,
    build_comparison_report,
    build_report,
    write_report,
)

_SPLIT_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _add_similarity_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Near-duplicate Jaccard threshold (default: 0.95).",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=40,
        help="Minimum normalized text length for near-duplicate checks (default: 40).",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finelint",
        description="Facts for fine-tuning data. No models, no verdicts.",
    )
    parser.add_argument(
        "--version", action="version", version=f"FineLint {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Inspect one CSV or JSONL dataset.")
    inspect.add_argument("dataset", type=Path, help="Path to a CSV or JSONL file.")
    inspect.add_argument(
        "--format", choices=("csv", "jsonl"), help="Override format detection."
    )
    inspect.add_argument(
        "--schema",
        choices=SCHEMAS,
        default="auto",
        help="Dataset schema (default: auto).",
    )
    inspect.add_argument(
        "--fields", action="append", help="Generic text field to inspect; repeatable."
    )
    inspect.add_argument(
        "--input-fields",
        action="append",
        default=[],
        help="Generic input field; repeatable.",
    )
    inspect.add_argument(
        "--output-fields",
        action="append",
        default=[],
        help="Generic output field; repeatable.",
    )
    inspect.add_argument(
        "--required-fields",
        action="append",
        default=[],
        help="Required field; repeatable.",
    )
    inspect.add_argument("--id-field", help="Field included as a record identifier.")
    _add_similarity_options(inspect)
    inspect.add_argument("--report", type=Path, help="Main JSON report path.")
    inspect.add_argument(
        "--affected-records", type=Path, help="Affected-records JSON path."
    )

    compare = subparsers.add_parser(
        "compare", help="Find leakage across named dataset splits."
    )
    compare.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named split; repeat at least twice.",
    )
    compare.add_argument(
        "--schema",
        action="append",
        default=[],
        metavar="NAME=SCHEMA",
        help="Override one split schema; repeatable.",
    )
    compare.add_argument("--id-field", help="Field included as a record identifier.")
    _add_similarity_options(compare)
    compare.add_argument("--report", type=Path, help="Comparison JSON report path.")
    compare.add_argument(
        "--affected-records", type=Path, help="Affected-records JSON path."
    )
    return parser


def _inspect_paths(
    dataset: Path, report: Path | None, affected: Path | None
) -> tuple[Path, Path]:
    report_path = report or Path.cwd() / f"{dataset.stem}.finelint-report.json"
    affected_path = affected or Path.cwd() / f"{dataset.stem}.finelint-affected.json"
    _validate_output_paths([dataset], report_path, affected_path)
    return report_path, affected_path


def _comparison_paths(report: Path | None, affected: Path | None) -> tuple[Path, Path]:
    report_path = report or Path.cwd() / "finelint-comparison-report.json"
    affected_path = affected or Path.cwd() / "finelint-comparison-affected.json"
    if report_path.resolve() == affected_path.resolve():
        raise ConfigurationError("The report and affected-records paths must differ.")
    return report_path, affected_path


def _validate_output_paths(sources: list[Path], report: Path, affected: Path) -> None:
    source_paths = {path.resolve() for path in sources}
    if report.resolve() in source_paths or affected.resolve() in source_paths:
        raise ConfigurationError("Output paths cannot overwrite a source dataset.")
    if report.resolve() == affected.resolve():
        raise ConfigurationError("The report and affected-records paths must differ.")


def _parse_assignment(value: str, label: str) -> tuple[str, str]:
    if "=" not in value:
        raise ConfigurationError(f"{label} must use NAME=VALUE syntax: {value}")
    name, assigned = value.split("=", 1)
    if not name or not assigned or not _SPLIT_NAME.fullmatch(name):
        raise ConfigurationError(f"Invalid {label}: {value}")
    return name, assigned


def _validate_similarity(threshold: float, min_length: int) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ConfigurationError("--similarity-threshold must be between 0 and 1.")
    if min_length < 1:
        raise ConfigurationError("--min-text-length must be at least 1.")


def _print_summary(
    title: str, schemas: str, report_path: Path, affected_path: Path, findings: list
) -> None:
    counts = Counter(finding.category for finding in findings)
    print(title)
    print(f"Schema: {schemas}")
    print(f"Findings: {len(findings)}")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")
    if findings:
        print("Examples:")
        for finding in findings[:5]:
            references = ", ".join(
                f"{ref.get('split') + ':' if ref.get('split') else ''}row {ref['row']}"
                if "row" in ref
                else f"line {ref['line']}"
                for ref in finding.records[:4]
            )
            print(
                f"  [{finding.code}] {finding.message}{f' ({references})' if references else ''}"
            )
        if len(findings) > 5:
            print(f"  ... {len(findings) - 5} more finding(s) in the JSON report")
    print(f"Report: {report_path.resolve()}")
    print(f"Affected records: {affected_path.resolve()}")


def _run_inspect(args: argparse.Namespace) -> None:
    _validate_similarity(args.similarity_threshold, args.min_text_length)
    report_path, affected_path = _inspect_paths(
        args.dataset, args.report, args.affected_records
    )
    config = ScanConfig(
        fields=args.fields,
        input_fields=args.input_fields,
        output_fields=args.output_fields,
        required_fields=args.required_fields,
        id_field=args.id_field,
        similarity_threshold=args.similarity_threshold,
        min_text_length=args.min_text_length,
        schema=args.schema,
    )
    result = read_dataset(args.dataset, args.format)
    chat = adapt_chat(result, args.schema, args.id_field)
    fields, findings = analyze_chat(result, config, chat)
    report = build_report(result, config, fields, findings, chat.schema)
    affected = build_affected_records(findings)
    write_report(report, report_path)
    write_report(affected, affected_path)
    _print_summary(
        f"Dataset: {args.dataset}", chat.schema, report_path, affected_path, findings
    )


def _run_compare(args: argparse.Namespace) -> None:
    _validate_similarity(args.similarity_threshold, args.min_text_length)
    split_values = [_parse_assignment(value, "--split") for value in args.split]
    if len(split_values) < 2:
        raise ConfigurationError("compare requires at least two --split values.")
    names = [name for name, _ in split_values]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ConfigurationError(f"Duplicate split name(s): {', '.join(duplicates)}")
    schema_values = [_parse_assignment(value, "--schema") for value in args.schema]
    schema_map: dict[str, str] = {}
    for name, schema in schema_values:
        if name in schema_map:
            raise ConfigurationError(f"Duplicate schema override for split '{name}'.")
        if name not in names:
            raise ConfigurationError(
                f"Schema override references unknown split '{name}'."
            )
        if schema not in CHAT_SCHEMAS:
            raise ConfigurationError(
                f"Compare schema must be one of: {', '.join(CHAT_SCHEMAS)}."
            )
        schema_map[name] = schema

    source_paths = [Path(value) for _, value in split_values]
    report_path, affected_path = _comparison_paths(args.report, args.affected_records)
    _validate_output_paths(source_paths, report_path, affected_path)
    splits: list[SplitInput] = []
    structural_findings = []
    for name, path_value in split_values:
        result = read_dataset(Path(path_value))
        chat = adapt_chat(result, schema_map.get(name, "auto"), args.id_field)
        if chat.schema == "generic":
            raise ConfigurationError(
                f"Split '{name}' is not a recognized chat dataset; pass an explicit chat --schema."
            )
        splits.append(SplitInput(name, result, chat.schema, chat.examples))
        for finding in result.findings + chat.findings:
            for ref in finding.records:
                ref.setdefault("dataset", str(result.path.resolve()))
                ref.setdefault("split", name)
                ref.setdefault("schema", chat.schema)
            structural_findings.append(finding)

    findings = structural_findings + compare_splits(
        splits,
        args.similarity_threshold,
        args.min_text_length,
        args.id_field,
    )
    report = build_comparison_report(
        splits,
        findings,
        args.similarity_threshold,
        args.min_text_length,
        args.id_field,
    )
    affected = build_affected_records(findings)
    write_report(report, report_path)
    write_report(affected, affected_path)
    schemas = ", ".join(f"{split.name}={split.schema}" for split in splits)
    _print_summary(
        f"Splits: {', '.join(names)}", schemas, report_path, affected_path, findings
    )


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            _run_inspect(args)
        elif args.command == "compare":
            _run_compare(args)
        else:
            raise ConfigurationError(f"Unknown command: {args.command}")
    except (DatasetReadError, ConfigurationError, OSError) as exc:
        print(f"finelint: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
