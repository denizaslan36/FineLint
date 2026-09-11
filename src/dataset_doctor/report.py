from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .models import Finding, ReadResult, ScanConfig


def build_report(result: ReadResult, config: ScanConfig, fields: list[str], findings: list[Finding]) -> dict[str, Any]:
    category_counts = Counter(finding.category for finding in findings)
    code_counts = Counter(finding.code for finding in findings)
    severity_counts = Counter(finding.severity for finding in findings)
    return {
        "metadata": {
            "tool": "dataset-doctor",
            "version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(result.path.resolve()),
            "format": result.format,
            "records_read": len(result.records),
        },
        "configuration": {
            "fields": fields,
            "input_fields": config.input_fields,
            "output_fields": config.output_fields,
            "required_fields": config.required_fields,
            "id_field": config.id_field,
            "similarity_threshold": config.similarity_threshold,
            "min_text_length": config.min_text_length,
        },
        "summary": {
            "total_findings": len(findings),
            "by_category": dict(sorted(category_counts.items())),
            "by_code": dict(sorted(code_counts.items())),
            "by_severity": dict(sorted(severity_counts.items())),
        },
        "findings": [finding.to_dict() for finding in findings],
    }


def write_report(report: dict[str, Any], destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
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
