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
    finding_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.finding_id is None:
            value.pop("finding_id")
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
    schema: str = "auto"


@dataclass(slots=True)
class ConversationMessage:
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass(slots=True)
class TrainingUnit:
    context: str
    target: str
    turn: int


@dataclass(slots=True)
class ConversationExample:
    record: Record
    schema: str
    messages: list[ConversationMessage]
    units: list[TrainingUnit]
    raw_text: str


@dataclass(slots=True)
class ChatResult:
    schema: str
    examples: list[ConversationExample]
    findings: list[Finding]


@dataclass(slots=True)
class SplitInput:
    name: str
    result: ReadResult
    schema: str
    examples: list[ConversationExample]
