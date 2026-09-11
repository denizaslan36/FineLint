from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .analyzer import ConfigurationError, analyze
from .models import ScanConfig
from .readers import DatasetReadError, read_dataset
from .report import build_report, write_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset-doctor", description="Deterministic, AI-free checks for text datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="Inspect a CSV or JSONL dataset.")
    inspect.add_argument("dataset", type=Path, help="Path to a CSV or JSONL file.")
    inspect.add_argument("--format", choices=("csv", "jsonl"), help="Override format detection.")
    inspect.add_argument("--fields", action="append", help="Text field to inspect; repeat for multiple fields.")
    inspect.add_argument("--input-fields", action="append", default=[], help="Input field used for contradiction checks; repeatable.")
    inspect.add_argument("--output-fields", action="append", default=[], help="Output field used for contradiction checks; repeatable.")
    inspect.add_argument("--required-fields", action="append", default=[], help="Required field; repeatable.")
    inspect.add_argument("--id-field", help="Field to include as a record identifier in findings.")
    inspect.add_argument("--similarity-threshold", type=float, default=0.95, help="Near-duplicate Jaccard threshold (default: 0.95).")
    inspect.add_argument("--min-text-length", type=int, default=40, help="Minimum normalized text length for near-duplicate checks (default: 40).")
    inspect.add_argument("--report", type=Path, help="JSON report path (default: ./<dataset-stem>.doctor-report.json).")
    return parser


def _default_report_path(dataset: Path) -> Path:
    return Path.cwd() / f"{dataset.stem}.doctor-report.json"


def _print_summary(dataset: Path, fields: list[str], report_path: Path, findings: list) -> None:
    counts = Counter(finding.category for finding in findings)
    print(f"Dataset: {dataset}")
    print(f"Text fields: {', '.join(fields)}")
    print(f"Findings: {len(findings)}")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")
    if findings:
        print("Examples:")
        for finding in findings[:5]:
            references = ", ".join(
                f"row {ref['row']}" if "row" in ref else f"line {ref['line']}"
                for ref in finding.records[:4]
            )
            suffix = f" ({references})" if references else ""
            print(f"  [{finding.code}] {finding.message}{suffix}")
        if len(findings) > 5:
            print(f"  ... {len(findings) - 5} more finding(s) in the JSON report")
    print(f"Report: {report_path.resolve()}")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command != "inspect":
        raise SystemExit(2)
    config = ScanConfig(
        fields=args.fields,
        input_fields=args.input_fields,
        output_fields=args.output_fields,
        required_fields=args.required_fields,
        id_field=args.id_field,
        similarity_threshold=args.similarity_threshold,
        min_text_length=args.min_text_length,
    )
    report_path = args.report or _default_report_path(args.dataset)
    try:
        if args.dataset.resolve() == report_path.resolve():
            raise ConfigurationError("The report path cannot overwrite the source dataset.")
        result = read_dataset(args.dataset, args.format)
        fields, findings = analyze(result, config)
        report = build_report(result, config, fields, findings)
        write_report(report, report_path)
    except (DatasetReadError, ConfigurationError, OSError) as exc:
        print(f"dataset-doctor: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    _print_summary(args.dataset, fields, report_path, findings)


if __name__ == "__main__":
    main()
