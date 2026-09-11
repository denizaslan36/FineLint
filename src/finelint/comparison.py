from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .analyzer import LSH_BUCKET_WINDOW, MAX_NEAR_CANDIDATES, MIN_SHARED_MINHASH_VALUES
from .models import ConversationExample, Finding, SplitInput
from .normalize import (
    character_shingles,
    jaccard,
    minhash_bands,
    minhash_signature,
    normalize_text,
)


@dataclass(slots=True)
class _CompareItem:
    split: SplitInput
    example: ConversationExample
    text: str
    turn: int | None = None
    target: str | None = None


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


def _ref(item: _CompareItem, id_field: str | None) -> dict[str, Any]:
    record = item.example.record
    value: dict[str, Any] = {
        "split": item.split.name,
        "dataset": str(item.split.result.path.resolve()),
        "schema": item.split.schema,
        "row": record.index,
        "line": record.line,
    }
    if id_field and id_field in record.data:
        value["id"] = record.data[id_field]
    if item.turn is not None:
        value["turn"] = item.turn
    return value


def _cross_split(items: list[_CompareItem]) -> bool:
    return len({item.split.name for item in items}) > 1


def _group_findings(
    items: list[_CompareItem],
    id_field: str | None,
    level: str,
) -> list[Finding]:
    raw_groups: dict[str, list[_CompareItem]] = defaultdict(list)
    normalized_groups: dict[str, list[_CompareItem]] = defaultdict(list)
    for item in items:
        raw_groups[item.text].append(item)
        normalized_groups[normalize_text(item.text)].append(item)

    prefix = "EXAMPLE" if level == "example" else "CONTEXT"
    findings: list[Finding] = []
    for raw, group in raw_groups.items():
        if len(group) > 1 and _cross_split(group):
            findings.append(
                Finding(
                    f"CROSS_SPLIT_EXACT_{prefix}",
                    "split_leakage",
                    "warning",
                    f"{len(group)} {level}s are exactly equal across splits.",
                    [_ref(item, id_field) for item in group],
                    evidence={"preview": raw[:160]},
                )
            )
    for normalized, group in normalized_groups.items():
        if (
            len(group) > 1
            and _cross_split(group)
            and len({item.text for item in group}) > 1
        ):
            findings.append(
                Finding(
                    f"CROSS_SPLIT_NORMALIZED_{prefix}",
                    "split_leakage",
                    "warning",
                    f"{len(group)} {level}s match across splits after normalization.",
                    [_ref(item, id_field) for item in group],
                    evidence={"normalized_preview": normalized[:160]},
                )
            )
    return findings


def _near_findings(
    items: list[_CompareItem],
    id_field: str | None,
    level: str,
    threshold: float,
    min_text_length: int,
) -> list[Finding]:
    unique: list[_CompareItem] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item.split.name, normalize_text(item.text))
        if key not in seen and len(key[1]) >= min_text_length:
            seen.add(key)
            unique.append(item)
    texts = [normalize_text(item.text) for item in unique]
    buckets: dict[tuple[int, int], deque[int]] = defaultdict(
        lambda: deque(maxlen=LSH_BUCKET_WINDOW)
    )
    union = _UnionFind(len(unique))
    matches: list[tuple[int, int, float]] = []

    @lru_cache(maxsize=1024)
    def shingles(index: int) -> set[str]:
        return character_shingles(texts[index])

    for index, item in enumerate(unique):
        keys = minhash_bands(minhash_signature(shingles(index)))
        candidate_counts: Counter[int] = Counter()
        for key in keys:
            candidate_counts.update(buckets[key])
        candidates = [
            candidate
            for candidate, shared in candidate_counts.most_common(MAX_NEAR_CANDIDATES)
            if shared >= MIN_SHARED_MINHASH_VALUES
        ]
        for other in sorted(candidates):
            if (
                item.split.name == unique[other].split.name
                or texts[index] == texts[other]
            ):
                continue
            score = jaccard(shingles(index), shingles(other))
            if score >= threshold:
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

    prefix = "EXAMPLE" if level == "example" else "CONTEXT"
    findings: list[Finding] = []
    for root in sorted(components, key=lambda value: min(components[value])):
        indexes = sorted(components[root])
        members = [unique[index] for index in indexes]
        if not _cross_split(members):
            continue
        edges = [
            {
                "left": _ref(unique[left], id_field),
                "right": _ref(unique[right], id_field),
                "similarity": round(score, 6),
            }
            for left, right, score in component_matches[root]
        ]
        findings.append(
            Finding(
                f"CROSS_SPLIT_NEAR_{prefix}",
                "split_leakage",
                "warning",
                f"{len(members)} highly similar {level}s occur across splits.",
                [_ref(item, id_field) for item in members],
                evidence={"matches": edges, "preview": members[0].text[:160]},
            )
        )
    return findings


def _conflicting_targets(
    items: list[_CompareItem], id_field: str | None
) -> list[Finding]:
    grouped: dict[str, list[_CompareItem]] = defaultdict(list)
    for item in items:
        grouped[normalize_text(item.text)].append(item)
    findings: list[Finding] = []
    for context, group in grouped.items():
        if not _cross_split(group):
            continue
        targets = {normalize_text(item.target or "") for item in group}
        if len(targets) > 1:
            findings.append(
                Finding(
                    "CROSS_SPLIT_CONFLICTING_TARGET",
                    "split_leakage",
                    "error",
                    "The same normalized context has different assistant targets across splits.",
                    [_ref(item, id_field) for item in group],
                    evidence={
                        "context_preview": context[:160],
                        "distinct_targets": len(targets),
                    },
                )
            )
    return findings


def compare_splits(
    splits: list[SplitInput],
    similarity_threshold: float,
    min_text_length: int,
    id_field: str | None = None,
) -> list[Finding]:
    example_items: list[_CompareItem] = []
    context_items: list[_CompareItem] = []
    for split in splits:
        for example in split.examples:
            if example.raw_text.strip():
                example_items.append(_CompareItem(split, example, example.raw_text))
            for unit in example.units:
                if unit.context.strip():
                    context_items.append(
                        _CompareItem(
                            split, example, unit.context, unit.turn, unit.target
                        )
                    )

    findings: list[Finding] = []
    findings.extend(_group_findings(example_items, id_field, "example"))
    findings.extend(
        _near_findings(
            example_items, id_field, "example", similarity_threshold, min_text_length
        )
    )
    findings.extend(_group_findings(context_items, id_field, "context"))
    findings.extend(
        _near_findings(
            context_items, id_field, "context", similarity_threshold, min_text_length
        )
    )
    findings.extend(_conflicting_targets(context_items, id_field))
    return findings
