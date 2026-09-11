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

REPORT_SCHEMA_VERSION = 3


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


def _stable_value(value: Any) -> Any:
    if isinstance(value, dict):
        stable = {
            key: _stable_value(item)
            for key, item in value.items()
            if key != "dataset"
        }
        if "id" in stable:
            stable.pop("row", None)
            stable.pop("line", None)
        elif "row" in stable:
            stable.pop("line", None)
        return stable
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    return value


def _finding_identity(finding: Finding) -> str:
    stable_refs = [_stable_value(ref) for ref in finding.records]
    payload = {
        "code": finding.code,
        "category": finding.category,
        "severity": finding.severity,
        "message": finding.message,
        "fields": finding.fields,
        "evidence": _stable_value(finding.evidence),
        "records": sorted(
            stable_refs,
            key=lambda value: json.dumps(value, sort_keys=True, default=str),
        ),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return "FL-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _finding_sort_key(finding: Finding) -> tuple[Any, ...]:
    first = finding.records[0] if finding.records else {}
    return (
        str(first.get("split", "")),
        str(first.get("dataset", "")),
        int(first.get("row", 0)),
        int(first.get("line", 0)),
        int(first.get("turn", -1)),
        finding.code,
        finding.finding_id or "",
    )


def prepare_findings(findings: list[Finding], dataset: Path | None = None) -> None:
    occurrences: Counter[str] = Counter()
    for finding in findings:
        if dataset is not None:
            for ref in finding.records:
                ref.setdefault("dataset", str(dataset.resolve()))
        identity = _finding_identity(finding)
        occurrence = occurrences[identity]
        occurrences[identity] += 1
        if occurrence:
            encoded = f"{identity}:{occurrence}".encode("utf-8")
            identity = "FL-" + hashlib.sha256(encoded).hexdigest()[:16]
        finding.finding_id = identity


def _ordered(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=_finding_sort_key)


def build_report(
    result: ReadResult,
    config: ScanConfig,
    fields: list[str],
    findings: list[Finding],
    detected_schema: str = "generic",
    config_path: Path | None = None,
    disabled_rules: list[str] | None = None,
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
            "source": str(config_path) if config_path else None,
            "schema": config.schema,
            "fields": fields,
            "input_fields": config.input_fields,
            "output_fields": config.output_fields,
            "required_fields": config.required_fields,
            "id_field": config.id_field,
            "similarity_threshold": config.similarity_threshold,
            "min_text_length": config.min_text_length,
            "disabled_rules": disabled_rules or [],
        },
        "summary": _summary(findings),
        "findings": [finding.to_dict() for finding in _ordered(findings)],
    }


def build_comparison_report(
    splits: list[SplitInput],
    findings: list[Finding],
    similarity_threshold: float,
    min_text_length: int,
    id_field: str | None,
    config_path: Path | None = None,
    disabled_rules: list[str] | None = None,
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
            "source": str(config_path) if config_path else None,
            "similarity_threshold": similarity_threshold,
            "min_text_length": min_text_length,
            "id_field": id_field,
            "disabled_rules": disabled_rules or [],
        },
        "summary": _summary(findings),
        "findings": [finding.to_dict() for finding in _ordered(findings)],
    }


def load_baseline(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read baseline report: {exc}") from exc
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse baseline report: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Baseline report must be a JSON object.")
    version = value.get("report_schema_version")
    if version != REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"Baseline report schema must be {REPORT_SCHEMA_VERSION}; found {version!r}. "
            "Create a fresh baseline with this FineLint version."
        )
    findings = value.get("findings")
    if not isinstance(findings, list) or not all(
        isinstance(finding, dict)
        and isinstance(finding.get("finding_id"), str)
        and finding["finding_id"]
        for finding in findings
    ):
        raise ValueError("Baseline report contains invalid findings.")
    ids = [finding["finding_id"] for finding in findings]
    if len(ids) != len(set(ids)):
        raise ValueError("Baseline report contains duplicate finding IDs.")
    return value, hashlib.sha256(raw).hexdigest()


def apply_baseline(
    report: dict[str, Any], baseline: dict[str, Any], baseline_path: Path, digest: str
) -> None:
    current_operation = report.get("metadata", {}).get("operation", "inspect")
    baseline_operation = baseline.get("metadata", {}).get("operation", "inspect")
    if current_operation != baseline_operation:
        raise ValueError(
            f"Baseline operation '{baseline_operation}' does not match "
            f"current operation '{current_operation}'."
        )
    current = report["findings"]
    baseline_findings = baseline["findings"]
    current_ids = {finding["finding_id"] for finding in current}
    baseline_ids = {finding["finding_id"] for finding in baseline_findings}
    for finding in current:
        finding["baseline_status"] = (
            "unchanged" if finding["finding_id"] in baseline_ids else "new"
        )
    resolved = sorted(baseline_ids - current_ids)
    report["baseline"] = {
        "path": str(baseline_path.resolve()),
        "sha256": digest,
        "new": len(current_ids - baseline_ids),
        "unchanged": len(current_ids & baseline_ids),
        "resolved": len(resolved),
        "resolved_finding_ids": resolved,
    }


def report_content_sha256(report: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in report.items()
        if key not in {"content_sha256", "baseline"}
    }
    metadata = dict(stable.get("metadata", {}))
    metadata.pop("generated_at", None)
    stable["metadata"] = metadata
    configuration = dict(stable.get("configuration", {}))
    configuration.pop("source", None)
    stable["configuration"] = configuration
    stable["findings"] = [
        {key: value for key, value in finding.items() if key != "baseline_status"}
        for finding in stable.get("findings", [])
    ]
    encoded = json.dumps(
        _stable_value(stable),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
