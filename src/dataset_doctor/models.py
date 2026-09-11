from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Record:
    index: int
    line: int
    data: dict[str, Any]


@dataclass(slots=True)
class Finding:
    code: str
    category: str
    severity: str
    message: str
    records: list[dict[str, Any]] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.evidence is None:
            value.pop("evidence")
        if not self.fields:
            value.pop("fields")
        if not self.records:
            value.pop("records")
        return value


@dataclass(slots=True)
class ReadResult:
    path: Path
    format: str
    records: list[Record]
    fields: list[str]
    findings: list[Finding]


@dataclass(slots=True)
class ScanConfig:
    fields: list[str] | None = None
    input_fields: list[str] = field(default_factory=list)
    output_fields: list[str] = field(default_factory=list)
    required_fields: list[str] = field(default_factory=list)
    id_field: str | None = None
    similarity_threshold: float = 0.95
    min_text_length: int = 40
