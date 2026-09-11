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
from .config import ProjectConfig, load_project_config
from .models import ScanConfig, SplitInput
from .readers import DatasetReadError, read_dataset
from .report import (
    build_affected_records,
    build_comparison_report,
    build_report,
    apply_baseline,
    load_baseline,
    report_content_sha256,
    write_report,
)

_SPLIT_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _add_similarity_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Near-duplicate Jaccard threshold (default: 0.95).",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=None,
        help="Minimum normalized text length for near-duplicate checks (default: 40).",
    )


def _add_project_options(parser: argparse.ArgumentParser) -> None:
    config = parser.add_mutually_exclusive_group()
    config.add_argument("--config", type=Path, help="FineLint JSON configuration path.")
    config.add_argument(
        "--no-config", action="store_true", help="Do not load .finelint.json."
    )
    parser.add_argument(
        "--baseline", type=Path, help="Previous report schema v3 JSON report."
    )
    parser.add_argument(
        "--fail-on",
        choices=("never", "error", "warning"),
        default=None,
        help="Return exit code 2 at or above this severity (default: never).",
    )
    parser.add_argument(
        "--fail-scope",
        choices=("all", "new"),
        default=None,
        help="Apply the CI gate to all or only new findings (default: all).",
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
        default=None,
        help="Dataset schema (default: auto).",
    )
    inspect.add_argument(
        "--fields", action="append", help="Generic text field to inspect; repeatable."
    )
    inspect.add_argument(
        "--input-fields",
        action="append",
        default=None,
        help="Generic input field; repeatable.",
    )
    inspect.add_argument(
        "--output-fields",
        action="append",
        default=None,
        help="Generic output field; repeatable.",
    )
    inspect.add_argument(
        "--required-fields",
        action="append",
        default=None,
        help="Required field; repeatable.",
    )
    inspect.add_argument("--id-field", help="Field included as a record identifier.")
    _add_similarity_options(inspect)
    _add_project_options(inspect)
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
        default=None,
        metavar="NAME=SCHEMA",
        help="Override one split schema; repeatable.",
    )
    compare.add_argument("--id-field", help="Field included as a record identifier.")
    _add_similarity_options(compare)
    _add_project_options(compare)
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


def _option(
    cli_value: object,
    config: ProjectConfig,
    section: str,
    key: str,
    default: object,
) -> object:
    if cli_value is not None:
        return cli_value
    return config.section(section).get(key, default)


def _project_options(
    args: argparse.Namespace, config: ProjectConfig
) -> tuple[str, str, list[str]]:
    fail_on = args.fail_on or config.values.get("fail_on", "never")
    fail_scope = args.fail_scope or config.values.get("fail_scope", "all")
    disabled_rules = list(config.values.get("disabled_rules", []))
    if fail_scope == "new" and args.baseline is None:
        raise ConfigurationError("--fail-scope new requires --baseline.")
    return fail_on, fail_scope, disabled_rules


def _apply_baseline_and_hash(
    report: dict, baseline_path: Path | None
) -> None:
    if baseline_path is not None:
        try:
            baseline, digest = load_baseline(baseline_path)
            apply_baseline(report, baseline, baseline_path, digest)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
    report["content_sha256"] = report_content_sha256(report)


def _gate_failed(report: dict, fail_on: str, fail_scope: str) -> bool:
    if fail_on == "never":
        return False
    severities = {"error"} if fail_on == "error" else {"error", "warning"}
    return any(
        finding["severity"] in severities
        and (fail_scope == "all" or finding.get("baseline_status") == "new")
        for finding in report["findings"]
    )


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
    project = load_project_config(args.config, args.no_config)
    fail_on, fail_scope, disabled_rules = _project_options(args, project)
    schema = _option(args.schema, project, "inspect", "schema", "auto")
    fields = _option(args.fields, project, "inspect", "fields", None)
    input_fields = _option(
        args.input_fields, project, "inspect", "input_fields", []
    )
    output_fields = _option(
        args.output_fields, project, "inspect", "output_fields", []
    )
    required_fields = _option(
        args.required_fields, project, "inspect", "required_fields", []
    )
    id_field = _option(args.id_field, project, "inspect", "id_field", None)
    similarity_threshold = _option(
        args.similarity_threshold,
        project,
        "inspect",
        "similarity_threshold",
        0.95,
    )
    min_text_length = _option(
        args.min_text_length, project, "inspect", "min_text_length", 40
    )
    assert isinstance(schema, str)
    assert fields is None or isinstance(fields, list)
    assert isinstance(input_fields, list)
    assert isinstance(output_fields, list)
    assert isinstance(required_fields, list)
    assert id_field is None or isinstance(id_field, str)
    assert isinstance(similarity_threshold, (int, float))
    assert isinstance(min_text_length, int)
    _validate_similarity(float(similarity_threshold), min_text_length)
    report_path, affected_path = _inspect_paths(
        args.dataset, args.report, args.affected_records
    )
    config = ScanConfig(
        fields=fields,
        input_fields=input_fields,
        output_fields=output_fields,
        required_fields=required_fields,
        id_field=id_field,
        similarity_threshold=float(similarity_threshold),
        min_text_length=min_text_length,
        schema=schema,
    )
    result = read_dataset(args.dataset, args.format)
    chat = adapt_chat(result, schema, id_field)
    fields, findings = analyze_chat(result, config, chat)
    findings = [finding for finding in findings if finding.code not in disabled_rules]
    report = build_report(
        result,
        config,
        fields,
        findings,
        chat.schema,
        project.path,
        disabled_rules,
    )
    _apply_baseline_and_hash(report, args.baseline)
    affected = build_affected_records(findings)
    write_report(report, report_path)
    write_report(affected, affected_path)
    _print_summary(
        f"Dataset: {args.dataset}", chat.schema, report_path, affected_path, findings
    )
    if _gate_failed(report, fail_on, fail_scope):
        raise SystemExit(2)


def _run_compare(args: argparse.Namespace) -> None:
    project = load_project_config(args.config, args.no_config)
    fail_on, fail_scope, disabled_rules = _project_options(args, project)
    id_field = _option(args.id_field, project, "compare", "id_field", None)
    similarity_threshold = _option(
        args.similarity_threshold,
        project,
        "compare",
        "similarity_threshold",
        0.95,
    )
    min_text_length = _option(
        args.min_text_length, project, "compare", "min_text_length", 40
    )
    assert id_field is None or isinstance(id_field, str)
    assert isinstance(similarity_threshold, (int, float))
    assert isinstance(min_text_length, int)
    _validate_similarity(float(similarity_threshold), min_text_length)
    split_values = [_parse_assignment(value, "--split") for value in args.split]
    if len(split_values) < 2:
        raise ConfigurationError("compare requires at least two --split values.")
    names = [name for name, _ in split_values]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ConfigurationError(f"Duplicate split name(s): {', '.join(duplicates)}")
    schema_values = [
        _parse_assignment(value, "--schema") for value in (args.schema or [])
    ]
    schema_map = dict(project.section("compare").get("schemas", {}))
    unknown_config_schemas = sorted(set(schema_map) - set(names))
    if unknown_config_schemas:
        raise ConfigurationError(
            "Configured schema references unknown split(s): "
            + ", ".join(unknown_config_schemas)
        )
    for name, schema in schema_values:
        if sum(1 for item_name, _ in schema_values if item_name == name) > 1:
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
        chat = adapt_chat(result, schema_map.get(name, "auto"), id_field)
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
        float(similarity_threshold),
        min_text_length,
        id_field,
    )
    findings = [finding for finding in findings if finding.code not in disabled_rules]
    report = build_comparison_report(
        splits,
        findings,
        float(similarity_threshold),
        min_text_length,
        id_field,
        project.path,
        disabled_rules,
    )
    _apply_baseline_and_hash(report, args.baseline)
    affected = build_affected_records(findings)
    write_report(report, report_path)
    write_report(affected, affected_path)
    schemas = ", ".join(f"{split.name}={split.schema}" for split in splits)
    _print_summary(
        f"Splits: {', '.join(names)}", schemas, report_path, affected_path, findings
    )
    if _gate_failed(report, fail_on, fail_scope):
        raise SystemExit(2)


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
