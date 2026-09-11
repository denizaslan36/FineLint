from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from functools import lru_cache
from typing import Any

from .models import Finding, ReadResult, Record, ScanConfig
from .normalize import (
    character_shingles,
    jaccard,
    minhash_bands,
    minhash_signature,
    normalize_text,
)

PREVIEW_LENGTH = 160
LSH_BUCKET_WINDOW = 64
MAX_NEAR_CANDIDATES = 32
MIN_SHARED_MINHASH_VALUES = 2


class ConfigurationError(Exception):
    pass


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _is_identifier_field(name: str) -> bool:
    lowered = name.casefold()
    return lowered in {"id", "uuid", "guid"} or lowered.endswith("_id")


def choose_text_fields(result: ReadResult, requested: list[str] | None) -> list[str]:
    if requested:
        missing = [field for field in requested if field not in result.fields]
        if missing:
            raise ConfigurationError(f"Unknown field(s): {', '.join(missing)}")
        return list(dict.fromkeys(requested))

    counters: dict[str, Counter[str]] = {field: Counter() for field in result.fields}
    for record in result.records:
        for field in result.fields:
            if field in record.data and record.data[field] is not None:
                counters[field][_type_name(record.data[field])] += 1

    selected = []
    for field in result.fields:
        counts = counters[field]
        observed = sum(counts.values())
        if (
            observed
            and counts["string"] / observed >= 0.8
            and not _is_identifier_field(field)
        ):
            selected.append(field)
    if not selected:
        raise ConfigurationError(
            "No text fields were detected. Select fields explicitly with --fields."
        )
    return selected


def _ref(record: Record, config: ScanConfig) -> dict[str, Any]:
    value: dict[str, Any] = {"row": record.index, "line": record.line}
    if config.id_field and config.id_field in record.data:
        value["id"] = record.data[config.id_field]
    return value


def _preview(value: Any) -> str:
    if value is None:
        return "null"
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    return text if len(text) <= PREVIEW_LENGTH else text[: PREVIEW_LENGTH - 1] + "…"


def _raw_key(record: Record, fields: list[str]) -> str:
    values = [[field, record.data.get(field)] for field in fields]
    return json.dumps(
        values, ensure_ascii=False, sort_keys=False, separators=(",", ":")
    )


def _normalized_values(record: Record, fields: list[str]) -> tuple[str, ...]:
    values: list[str] = []
    for field in fields:
        value = record.data.get(field)
        if isinstance(value, str):
            values.append(normalize_text(value))
        elif value is None:
            values.append("\0null")
        else:
            values.append(
                "\0"
                + json.dumps(
                    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            )
    return tuple(values)


def _combined_text(record: Record, fields: list[str]) -> str:
    return "\n".join(
        f"{field}: {value}"
        for field, value in zip(fields, _normalized_values(record, fields))
    )


def _validate_config(result: ReadResult, config: ScanConfig, fields: list[str]) -> None:
    known = set(result.fields)
    role_fields = config.input_fields + config.output_fields + config.required_fields
    if config.id_field:
        role_fields.append(config.id_field)
    missing = sorted(set(role_fields) - known)
    if missing:
        raise ConfigurationError(f"Unknown field(s): {', '.join(missing)}")
    if bool(config.input_fields) != bool(config.output_fields):
        raise ConfigurationError(
            "--input-fields and --output-fields must be used together."
        )
    if not 0.0 <= config.similarity_threshold <= 1.0:
        raise ConfigurationError("--similarity-threshold must be between 0 and 1.")
    if config.min_text_length < 1:
        raise ConfigurationError("--min-text-length must be at least 1.")
    if not fields:
        raise ConfigurationError("At least one text field is required.")


def _structural_findings(
    result: ReadResult, config: ScanConfig, fields: list[str]
) -> list[Finding]:
    findings: list[Finding] = []
    required = set(config.required_fields)
    presence = Counter(field for record in result.records for field in record.data)
    common = {
        field
        for field, count in presence.items()
        if result.records and count / len(result.records) >= 0.95
    }

    for record in result.records:
        if not record.data or all(
            value is None or (isinstance(value, str) and not value.strip())
            for value in record.data.values()
        ):
            findings.append(
                Finding(
                    "EMPTY_RECORD",
                    "structural",
                    "error",
                    "Record has no usable values.",
                    [_ref(record, config)],
                )
            )
        for field in sorted(required):
            value = record.data.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                findings.append(
                    Finding(
                        "MISSING_REQUIRED_FIELD",
                        "structural",
                        "error",
                        f"Required field '{field}' is missing or empty.",
                        [_ref(record, config)],
                        [field],
                    )
                )
        for field in sorted(common - required):
            if field not in record.data:
                findings.append(
                    Finding(
                        "MISSING_COMMON_FIELD",
                        "structural",
                        "warning",
                        f"Field '{field}' exists in at least 95% of records but is missing here.",
                        [_ref(record, config)],
                        [field],
                    )
                )
        for field in fields:
            value = record.data.get(field)
            if value is None:
                if field not in required and field in record.data:
                    findings.append(
                        Finding(
                            "NULL_TEXT_FIELD",
                            "structural",
                            "warning",
                            f"Text field '{field}' is null.",
                            [_ref(record, config)],
                            [field],
                        )
                    )
            elif not isinstance(value, str):
                findings.append(
                    Finding(
                        "NON_STRING_TEXT_FIELD",
                        "structural",
                        "error",
                        f"Text field '{field}' contains {_type_name(value)}.",
                        [_ref(record, config)],
                        [field],
                        {"preview": _preview(value)},
                    )
                )
            elif not value.strip() and field not in required:
                findings.append(
                    Finding(
                        "BLANK_TEXT_FIELD",
                        "structural",
                        "warning",
                        f"Text field '{field}' is blank.",
                        [_ref(record, config)],
                        [field],
                    )
                )

    if result.format == "jsonl":
        for field in result.fields:
            typed = [
                (record, _type_name(record.data[field]))
                for record in result.records
                if field in record.data and record.data[field] is not None
            ]
            counts = Counter(value_type for _, value_type in typed)
            if len(counts) <= 1:
                continue
            dominant = min(counts.items(), key=lambda item: (-item[1], item[0]))[0]
            for record, value_type in typed:
                if value_type != dominant and not (
                    field in fields and value_type != "string"
                ):
                    findings.append(
                        Finding(
                            "TYPE_MISMATCH",
                            "structural",
                            "warning",
                            f"Field '{field}' is usually {dominant} but this value is {value_type}.",
                            [_ref(record, config)],
                            [field],
                            {"preview": _preview(record.data[field])},
                        )
                    )
    return findings


def _duplicate_findings(
    result: ReadResult, config: ScanConfig, fields: list[str]
) -> tuple[list[Finding], dict[tuple[str, ...], list[Record]]]:
    findings: list[Finding] = []
    raw_groups: dict[str, list[Record]] = defaultdict(list)
    normalized_groups: dict[tuple[str, ...], list[Record]] = defaultdict(list)
    for record in result.records:
        raw_groups[_raw_key(record, fields)].append(record)
        normalized_groups[_normalized_values(record, fields)].append(record)

    for group in raw_groups.values():
        if len(group) > 1:
            findings.append(
                Finding(
                    "EXACT_DUPLICATE",
                    "exact_duplicate",
                    "warning",
                    f"{len(group)} records contain exactly the same selected field values.",
                    [_ref(record, config) for record in group],
                    fields,
                    {"preview": _preview(_combined_text(group[0], fields))},
                )
            )

    for normalized_key, group in normalized_groups.items():
        if len(group) <= 1:
            continue
        raw_variants = {_raw_key(record, fields) for record in group}
        if len(raw_variants) > 1:
            findings.append(
                Finding(
                    "NORMALIZED_DUPLICATE",
                    "normalized_duplicate",
                    "warning",
                    f"{len(group)} records match after Unicode, casing, and whitespace normalization.",
                    [_ref(record, config) for record in group],
                    fields,
                    {"normalized_preview": _preview("\n".join(normalized_key))},
                )
            )
    return findings, normalized_groups


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _near_duplicate_findings(
    result: ReadResult,
    config: ScanConfig,
    fields: list[str],
    normalized_groups: dict[tuple[str, ...], list[Record]],
) -> list[Finding]:
    representatives = [
        group[0]
        for key, group in normalized_groups.items()
        if any(key)
        and all(isinstance(group[0].data.get(field), str) for field in fields)
    ]
    texts = [_combined_text(record, fields) for record in representatives]
    eligible = [
        index for index, text in enumerate(texts) if len(text) >= config.min_text_length
    ]
    buckets: dict[tuple[int, int], deque[int]] = defaultdict(
        lambda: deque(maxlen=LSH_BUCKET_WINDOW)
    )
    union = _UnionFind(len(representatives))
    matches: list[tuple[int, int, float]] = []

    @lru_cache(maxsize=1024)
    def shingles(index: int) -> set[str]:
        return character_shingles(texts[index])

    for index in eligible:
        signature = minhash_signature(shingles(index))
        keys = minhash_bands(signature)
        candidate_counts: Counter[int] = Counter()
        for key in keys:
            candidate_counts.update(buckets[key])
        candidates = [
            candidate
            for candidate, shared_values in candidate_counts.most_common(
                MAX_NEAR_CANDIDATES
            )
            if shared_values >= MIN_SHARED_MINHASH_VALUES
        ]
        current_shingles = shingles(index)
        for other in sorted(candidates):
            score = jaccard(current_shingles, shingles(other))
            if score >= config.similarity_threshold:
                union.union(index, other)
                matches.append((other, index, score))
        for key in keys:
            buckets[key].append(index)

    components: dict[int, set[int]] = defaultdict(set)
    component_matches: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for left, right, score in matches:
        root = union.find(left)
        components[root].update((left, right))
        component_matches[root].append((left, right, score))

    findings: list[Finding] = []
    for root in sorted(components, key=lambda item: min(components[item])):
        member_indexes = sorted(components[root])
        records: list[Record] = []
        for index in member_indexes:
            key = _normalized_values(representatives[index], fields)
            records.extend(normalized_groups[key])
        edge_values = [
            {
                "left_row": representatives[left].index,
                "right_row": representatives[right].index,
                "similarity": round(score, 6),
            }
            for left, right, score in sorted(component_matches[root])
        ]
        findings.append(
            Finding(
                "NEAR_DUPLICATE",
                "near_duplicate",
                "warning",
                f"{len(records)} records form a highly similar group.",
                [
                    _ref(record, config)
                    for record in sorted(records, key=lambda record: record.index)
                ],
                fields,
                {"matches": edge_values, "preview": _preview(texts[member_indexes[0]])},
            )
        )
    return findings


def _conflict_findings(result: ReadResult, config: ScanConfig) -> list[Finding]:
    if not config.input_fields:
        return []
    grouped: dict[tuple[str, ...], list[Record]] = defaultdict(list)
    for record in result.records:
        role_fields = config.input_fields + config.output_fields
        if not all(isinstance(record.data.get(field), str) for field in role_fields):
            continue
        input_key = _normalized_values(record, config.input_fields)
        if not any(input_key):
            continue
        grouped[input_key].append(record)

    findings: list[Finding] = []
    for input_key, records in grouped.items():
        if len(records) <= 1:
            continue
        outputs: dict[tuple[str, ...], list[Record]] = defaultdict(list)
        for record in records:
            outputs[_normalized_values(record, config.output_fields)].append(record)
        if len(outputs) > 1:
            findings.append(
                Finding(
                    "CONFLICTING_OUTPUT",
                    "conflicting_output",
                    "error",
                    "The same normalized input has different normalized outputs.",
                    [_ref(record, config) for record in records],
                    config.input_fields + config.output_fields,
                    {
                        "input_preview": _preview("\n".join(input_key)),
                        "distinct_outputs": [
                            _preview("\n".join(output)) for output in outputs
                        ],
                    },
                )
            )
    return findings


def analyze(result: ReadResult, config: ScanConfig) -> tuple[list[str], list[Finding]]:
    fields = choose_text_fields(result, config.fields)
    _validate_config(result, config, fields)
    findings = list(result.findings)
    findings.extend(_structural_findings(result, config, fields))
    duplicates, normalized_groups = _duplicate_findings(result, config, fields)
    findings.extend(duplicates)
    findings.extend(_near_duplicate_findings(result, config, fields, normalized_groups))
    findings.extend(_conflict_findings(result, config))
    return fields, findings
