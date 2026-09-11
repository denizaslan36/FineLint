from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .models import Finding, ReadResult, Record


class DatasetReadError(Exception):
    pass


def _record_ref(index: int | None, line: int) -> dict[str, int]:
    ref = {"line": line}
    if index is not None:
        ref["row"] = index
    return ref


def read_dataset(path: Path, explicit_format: str | None = None) -> ReadResult:
    if not path.exists():
        raise DatasetReadError(f"Dataset does not exist: {path}")
    if not path.is_file():
        raise DatasetReadError(f"Dataset is not a file: {path}")

    selected_format = explicit_format or path.suffix.lower().lstrip(".")
    if selected_format == "csv":
        return _read_csv(path)
    if selected_format in {"jsonl", "ndjson"}:
        return _read_jsonl(path)
    raise DatasetReadError("Unsupported format. Use a .csv or .jsonl file, or pass --format.")


def _read_csv(path: Path) -> ReadResult:
    findings: list[Finding] = []
    records: list[Record] = []
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except (OSError, UnicodeError) as exc:
        raise DatasetReadError(f"Could not open dataset: {exc}") from exc

    try:
        with handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = next(reader)
            except StopIteration:
                raise DatasetReadError("CSV file is empty.")
            except csv.Error as exc:
                raise DatasetReadError(f"Could not parse CSV header: {exc}") from exc

            if not header or all(not name.strip() for name in header):
                raise DatasetReadError("CSV header is empty.")
            duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
            if duplicates:
                raise DatasetReadError(f"CSV contains duplicate headers: {', '.join(duplicates)}")
            if any(not name.strip() for name in header):
                raise DatasetReadError("CSV contains an empty header name.")

            try:
                for row in reader:
                    index = len(records) + 1
                    line = reader.line_num
                    if len(row) != len(header):
                        findings.append(
                            Finding(
                                code="CSV_ROW_WIDTH_MISMATCH",
                                category="structural",
                                severity="error",
                                message=f"Expected {len(header)} values but found {len(row)}.",
                                records=[_record_ref(index, line)],
                            )
                        )
                    values: list[Any] = row[: len(header)]
                    values.extend([None] * (len(header) - len(values)))
                    records.append(Record(index=index, line=line, data=dict(zip(header, values))))
            except csv.Error as exc:
                findings.append(
                    Finding(
                        code="MALFORMED_CSV",
                        category="structural",
                        severity="error",
                        message=str(exc),
                        records=[_record_ref(None, reader.line_num)],
                    )
                )
    except UnicodeDecodeError as exc:
        raise DatasetReadError(f"Dataset is not valid UTF-8: {exc}") from exc

    return ReadResult(path, "csv", records, header, findings)


def _read_jsonl(path: Path) -> ReadResult:
    findings: list[Finding] = []
    records: list[Record] = []
    fields: list[str] = []
    known_fields: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except OSError as exc:
        raise DatasetReadError(f"Could not open dataset: {exc}") from exc

    try:
        with handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    findings.append(
                        Finding(
                            code="EMPTY_JSONL_LINE",
                            category="structural",
                            severity="error",
                            message="JSONL line is empty.",
                            records=[_record_ref(None, line_number)],
                        )
                    )
                    continue
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    findings.append(
                        Finding(
                            code="MALFORMED_JSON",
                            category="structural",
                            severity="error",
                            message=f"{exc.msg} at column {exc.colno}.",
                            records=[_record_ref(None, line_number)],
                        )
                    )
                    continue
                if not isinstance(value, dict):
                    findings.append(
                        Finding(
                            code="JSONL_RECORD_NOT_OBJECT",
                            category="structural",
                            severity="error",
                            message="JSONL record must be an object.",
                            records=[_record_ref(None, line_number)],
                        )
                    )
                    continue
                for key in value:
                    if key not in known_fields:
                        known_fields.add(key)
                        fields.append(key)
                records.append(Record(index=len(records) + 1, line=line_number, data=value))
    except UnicodeDecodeError as exc:
        raise DatasetReadError(f"Dataset is not valid UTF-8: {exc}") from exc

    return ReadResult(path, "jsonl", records, fields, findings)
