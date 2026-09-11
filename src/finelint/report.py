from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .models import Finding, ReadResult, ScanConfig, SplitInput

REPORT_SCHEMA_VERSION = 2


def _summary(findings: list[Finding]) -> dict[str, Any]:
    return {
        "total_findings": len(findings),
        "by_category": dict(
            sorted(Counter(finding.category for finding in findings).items())
        ),
        "by_code": dict(sorted(Counter(finding.code for finding in findings).items())),
        "by_severity": dict(
            sorted(Counter(finding.severity for finding in findings).items())
        ),
    }


def _finding_identity(finding: Finding) -> str:
    stable_refs = []
    for ref in finding.records:
        stable_refs.append(
            {key: value for key, value in ref.items() if key != "dataset"}
        )
    payload = {
        "code": finding.code,
        "fields": finding.fields,
        "records": sorted(
            stable_refs,
            key=lambda value: json.dumps(value, sort_keys=True, default=str),
        ),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return "FL-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def prepare_findings(findings: list[Finding], dataset: Path | None = None) -> None:
    for finding in findings:
        if dataset is not None:
            for ref in finding.records:
                ref.setdefault("dataset", str(dataset.resolve()))
        finding.finding_id = _finding_identity(finding)


def build_report(
    result: ReadResult,
    config: ScanConfig,
    fields: list[str],
    findings: list[Finding],
    detected_schema: str = "generic",
) -> dict[str, Any]:
    prepare_findings(findings, result.path)
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "metadata": {
            "tool": "finelint",
            "version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(result.path.resolve()),
            "format": result.format,
            "detected_schema": detected_schema,
            "records_read": len(result.records),
        },
        "configuration": {
            "schema": config.schema,
            "fields": fields,
            "input_fields": config.input_fields,
            "output_fields": config.output_fields,
            "required_fields": config.required_fields,
            "id_field": config.id_field,
            "similarity_threshold": config.similarity_threshold,
            "min_text_length": config.min_text_length,
        },
        "summary": _summary(findings),
        "findings": [finding.to_dict() for finding in findings],
    }


def build_comparison_report(
    splits: list[SplitInput],
    findings: list[Finding],
    similarity_threshold: float,
    min_text_length: int,
    id_field: str | None,
) -> dict[str, Any]:
    prepare_findings(findings)
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "metadata": {
            "tool": "finelint",
            "version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "operation": "compare",
            "splits": [
                {
                    "name": split.name,
                    "dataset": str(split.result.path.resolve()),
                    "format": split.result.format,
                    "detected_schema": split.schema,
                    "records_read": len(split.result.records),
                    "examples_adapted": len(split.examples),
                }
                for split in splits
            ],
        },
        "configuration": {
            "similarity_threshold": similarity_threshold,
            "min_text_length": min_text_length,
            "id_field": id_field,
        },
        "summary": _summary(findings),
        "findings": [finding.to_dict() for finding in findings],
    }


def build_affected_records(findings: list[Finding]) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for finding in findings:
        for ref in finding.records:
            key = json.dumps(
                ref,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            entry = records.setdefault(key, {**ref, "reasons": []})
            entry["reasons"].append(
                {
                    "finding_id": finding.finding_id,
                    "code": finding.code,
                    "category": finding.category,
                    "severity": finding.severity,
                }
            )
    ordered = sorted(
        records.values(),
        key=lambda value: (
            str(value.get("split", "")),
            str(value.get("dataset", "")),
            int(value.get("row", 0)),
            int(value.get("turn", -1)),
        ),
    )
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "metadata": {
            "tool": "finelint",
            "version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "affected_record_references",
        },
        "records": ordered,
    }


def write_report(report: dict[str, Any], destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
