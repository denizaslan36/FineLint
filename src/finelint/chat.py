from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .analyzer import ConfigurationError, analyze
from .models import (
    ChatResult,
    ConversationExample,
    ConversationMessage,
    Finding,
    ReadResult,
    Record,
    ScanConfig,
    TrainingUnit,
)

CHAT_SCHEMAS = ("openai", "sharegpt", "alpaca")
SCHEMAS = ("auto", "generic", *CHAT_SCHEMAS)


def source_ref(
    result: ReadResult, record: Record, id_field: str | None, turn: int | None = None
) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "dataset": str(result.path.resolve()),
        "row": record.index,
        "line": record.line,
    }
    if id_field and id_field in record.data:
        ref["id"] = record.data[id_field]
    if turn is not None:
        ref["turn"] = turn
    return ref


def _schema_matches(data: dict[str, Any]) -> set[str]:
    matches: set[str] = set()
    if "messages" in data:
        matches.add("openai")
    if "conversations" in data:
        matches.add("sharegpt")
    if "instruction" in data and "output" in data:
        matches.add("alpaca")
    return matches


def detect_schema(result: ReadResult) -> str:
    detected: set[str] = set()
    ambiguous_rows: list[int] = []
    for record in result.records:
        matches = _schema_matches(record.data)
        if len(matches) > 1:
            ambiguous_rows.append(record.index)
        detected.update(matches)
    if ambiguous_rows:
        preview = ", ".join(str(row) for row in ambiguous_rows[:5])
        raise ConfigurationError(
            f"Ambiguous chat schema at row(s) {preview}. Select one with --schema."
        )
    if len(detected) > 1:
        raise ConfigurationError(
            f"Mixed chat schemas detected ({', '.join(sorted(detected))}). Select one with --schema."
        )
    return next(iter(detected), "generic")


def _message_text(message: ConversationMessage) -> str:
    parts = [f"role={message.role}"]
    if message.content is not None:
        parts.append(f"content={message.content}")
    if message.tool_calls:
        parts.append(
            "tool_calls="
            + json.dumps(
                message.tool_calls,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    if message.tool_call_id:
        parts.append(f"tool_call_id={message.tool_call_id}")
    return "\n".join(parts)


def _canonical_conversation(messages: list[ConversationMessage]) -> str:
    return "\n\n".join(_message_text(message) for message in messages)


def _training_units(messages: list[ConversationMessage]) -> list[TrainingUnit]:
    context: list[str] = []
    units: list[TrainingUnit] = []
    for index, message in enumerate(messages):
        rendered = _message_text(message)
        if message.role == "assistant" and (
            message.content is not None or message.tool_calls
        ):
            target_parts: list[str] = []
            if message.content is not None:
                target_parts.append(message.content)
            if message.tool_calls:
                target_parts.append(
                    json.dumps(
                        message.tool_calls,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            units.append(
                TrainingUnit("\n\n".join(context), "\n".join(target_parts), index)
            )
        context.append(rendered)
    return units


def _flow_findings(
    result: ReadResult,
    record: Record,
    messages: list[ConversationMessage],
    id_field: str | None,
) -> list[Finding]:
    findings: list[Finding] = []
    seen: dict[str, int] = {}
    for index, message in enumerate(messages):
        ref = source_ref(result, record, id_field, index)
        if index and messages[index - 1].role == message.role:
            findings.append(
                Finding(
                    "CHAT_CONSECUTIVE_ROLE",
                    "chat_flow",
                    "warning",
                    f"Two consecutive messages use role '{message.role}'.",
                    [ref],
                )
            )
        if message.role == "system" and index > 0:
            findings.append(
                Finding(
                    "CHAT_SYSTEM_POSITION",
                    "chat_flow",
                    "warning",
                    "A system message appears after the first turn.",
                    [ref],
                )
            )
        rendered = _message_text(message)
        if rendered in seen:
            findings.append(
                Finding(
                    "CHAT_DUPLICATE_TURN",
                    "chat_flow",
                    "warning",
                    f"Turn duplicates turn {seen[rendered]} in the same example.",
                    [ref],
                )
            )
        else:
            seen[rendered] = index
    if not any(
        message.role == "assistant"
        and (message.content is not None or message.tool_calls)
        for message in messages
    ):
        findings.append(
            Finding(
                "CHAT_NO_ASSISTANT_TARGET",
                "chat_schema",
                "error",
                "Conversation contains no usable assistant target.",
                [source_ref(result, record, id_field)],
            )
        )
    return findings


def _invalid_record(
    result: ReadResult,
    record: Record,
    id_field: str | None,
    code: str,
    message: str,
    turn: int | None = None,
) -> Finding:
    return Finding(
        code,
        "chat_schema",
        "error",
        message,
        [source_ref(result, record, id_field, turn)],
    )


def _validate_tool_calls(
    result: ReadResult,
    record: Record,
    turn: int,
    value: Any,
    tool_names: set[str],
    id_field: str | None,
) -> tuple[list[dict[str, Any]], list[Finding], set[str]]:
    findings: list[Finding] = []
    call_ids: set[str] = set()
    if not isinstance(value, list) or not value:
        return (
            [],
            [
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "TOOL_CALLS_INVALID",
                    "tool_calls must be a non-empty array.",
                    turn,
                )
            ],
            call_ids,
        )
    valid: list[dict[str, Any]] = []
    for call in value:
        if not isinstance(call, dict):
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "TOOL_CALLS_INVALID",
                    "Each tool call must be an object.",
                    turn,
                )
            )
            continue
        call_id = call.get("id")
        function = call.get("function")
        if (
            not isinstance(call_id, str)
            or not call_id
            or call.get("type") != "function"
            or not isinstance(function, dict)
        ):
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "TOOL_CALLS_INVALID",
                    "Tool call requires id, type='function', and a function object.",
                    turn,
                )
            )
            continue
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name or not isinstance(arguments, str):
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "TOOL_CALLS_INVALID",
                    "Tool function requires string name and arguments.",
                    turn,
                )
            )
            continue
        try:
            json.loads(arguments)
        except json.JSONDecodeError:
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "TOOL_ARGUMENTS_INVALID_JSON",
                    f"Arguments for tool '{name}' are not valid JSON.",
                    turn,
                )
            )
        if not tool_names or name not in tool_names:
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "TOOL_DEFINITION_MISSING",
                    f"Tool '{name}' is called but not defined in tools.",
                    turn,
                )
            )
        if call_id in call_ids:
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "TOOL_CALL_ID_DUPLICATE",
                    f"Tool call id '{call_id}' is duplicated.",
                    turn,
                )
            )
        call_ids.add(call_id)
        valid.append(call)
    return valid, findings, call_ids


def _openai_tools(
    result: ReadResult, record: Record, id_field: str | None
) -> tuple[set[str], list[Finding]]:
    findings: list[Finding] = []
    tools = record.data.get("tools")
    if tools is None:
        return set(), findings
    if not isinstance(tools, list):
        return set(), [
            _invalid_record(
                result, record, id_field, "TOOLS_INVALID", "tools must be an array."
            )
        ]
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if (
            not isinstance(tool, dict)
            or tool.get("type") != "function"
            or not isinstance(name, str)
            or not name
        ):
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "TOOLS_INVALID",
                    "Each tool requires type='function' and a named function object.",
                )
            )
            continue
        if name in names:
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "TOOL_NAME_DUPLICATE",
                    f"Tool name '{name}' is duplicated.",
                )
            )
        names.add(name)
    return names, findings


def _adapt_openai(
    result: ReadResult, record: Record, id_field: str | None
) -> tuple[ConversationExample | None, list[Finding]]:
    findings: list[Finding] = []
    raw_messages = record.data.get("messages")
    if not isinstance(raw_messages, list):
        return None, [
            _invalid_record(
                result,
                record,
                id_field,
                "CHAT_MESSAGES_MISSING",
                "messages must be an array.",
            )
        ]
    if not raw_messages:
        return None, [
            _invalid_record(
                result, record, id_field, "CHAT_MESSAGES_EMPTY", "messages is empty."
            )
        ]
    if "parallel_tool_calls" in record.data and not isinstance(
        record.data["parallel_tool_calls"], bool
    ):
        findings.append(
            _invalid_record(
                result,
                record,
                id_field,
                "PARALLEL_TOOL_CALLS_INVALID",
                "parallel_tool_calls must be boolean.",
            )
        )
    tool_names, tool_findings = _openai_tools(result, record, id_field)
    findings.extend(tool_findings)
    messages: list[ConversationMessage] = []
    known_call_ids: set[str] = set()
    for turn, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "CHAT_MESSAGE_NOT_OBJECT",
                    "Each message must be an object.",
                    turn,
                )
            )
            continue
        role = raw.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "CHAT_ROLE_INVALID",
                    "Message role must be system, user, assistant, or tool.",
                    turn,
                )
            )
            continue
        content = raw.get("content")
        raw_calls = raw.get("tool_calls")
        calls: list[dict[str, Any]] = []
        if raw_calls is not None:
            if role != "assistant":
                findings.append(
                    _invalid_record(
                        result,
                        record,
                        id_field,
                        "TOOL_CALLS_INVALID",
                        "Only assistant messages may contain tool_calls.",
                        turn,
                    )
                )
            calls, call_findings, call_ids = _validate_tool_calls(
                result, record, turn, raw_calls, tool_names, id_field
            )
            findings.extend(call_findings)
            for duplicate_id in sorted(known_call_ids & call_ids):
                findings.append(
                    _invalid_record(
                        result,
                        record,
                        id_field,
                        "TOOL_CALL_ID_DUPLICATE",
                        f"Tool call id '{duplicate_id}' is reused in the conversation.",
                        turn,
                    )
                )
            known_call_ids.update(call_ids)
        if content is not None and not isinstance(content, str):
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "CHAT_CONTENT_INVALID",
                    "Message content must be a string or null.",
                    turn,
                )
            )
            content = None
        if (content is None or not content.strip()) and not (
            role == "assistant" and calls
        ):
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "CHAT_EMPTY_TURN",
                    "Message has no usable content.",
                    turn,
                )
            )
        tool_call_id = raw.get("tool_call_id")
        if role == "tool" and (
            not isinstance(tool_call_id, str) or tool_call_id not in known_call_ids
        ):
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "TOOL_RESULT_ORPHANED",
                    "Tool message does not match an earlier tool call id.",
                    turn,
                )
            )
            tool_call_id = tool_call_id if isinstance(tool_call_id, str) else None
        messages.append(ConversationMessage(role, content, calls, tool_call_id))
    findings.extend(_flow_findings(result, record, messages, id_field))
    example = ConversationExample(
        record,
        "openai",
        messages,
        _training_units(messages),
        _canonical_conversation(messages),
    )
    return example, findings


def _adapt_sharegpt(
    result: ReadResult, record: Record, id_field: str | None
) -> tuple[ConversationExample | None, list[Finding]]:
    raw_messages = record.data.get("conversations")
    if not isinstance(raw_messages, list):
        return None, [
            _invalid_record(
                result,
                record,
                id_field,
                "CHAT_MESSAGES_MISSING",
                "conversations must be an array.",
            )
        ]
    if not raw_messages:
        return None, [
            _invalid_record(
                result,
                record,
                id_field,
                "CHAT_MESSAGES_EMPTY",
                "conversations is empty.",
            )
        ]
    aliases = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "bot": "assistant",
        "system": "system",
        "tool": "tool",
    }
    messages: list[ConversationMessage] = []
    findings: list[Finding] = []
    for turn, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "CHAT_MESSAGE_NOT_OBJECT",
                    "Each conversation turn must be an object.",
                    turn,
                )
            )
            continue
        source = raw.get("from")
        role = aliases.get(source.casefold()) if isinstance(source, str) else None
        if role is None:
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "CHAT_ROLE_INVALID",
                    "ShareGPT turn has an unsupported 'from' value.",
                    turn,
                )
            )
            continue
        value = raw.get("value")
        if not isinstance(value, str):
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "CHAT_CONTENT_INVALID",
                    "ShareGPT 'value' must be a string.",
                    turn,
                )
            )
            value = None
        elif not value.strip():
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "CHAT_EMPTY_TURN",
                    "Conversation turn is blank.",
                    turn,
                )
            )
        messages.append(ConversationMessage(role, value))
    findings.extend(_flow_findings(result, record, messages, id_field))
    example = ConversationExample(
        record,
        "sharegpt",
        messages,
        _training_units(messages),
        _canonical_conversation(messages),
    )
    return example, findings


def _adapt_alpaca(
    result: ReadResult, record: Record, id_field: str | None
) -> tuple[ConversationExample | None, list[Finding]]:
    findings: list[Finding] = []
    instruction = record.data.get("instruction")
    input_value = record.data.get("input", "")
    output = record.data.get("output")
    for field, value, required in (
        ("instruction", instruction, True),
        ("input", input_value, False),
        ("output", output, True),
    ):
        if value is not None and not isinstance(value, str):
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "CHAT_CONTENT_INVALID",
                    f"Alpaca '{field}' must be a string.",
                )
            )
        elif required and (value is None or not value.strip()):
            findings.append(
                _invalid_record(
                    result,
                    record,
                    id_field,
                    "CHAT_EMPTY_TURN",
                    f"Alpaca '{field}' is missing or blank.",
                )
            )
    user_parts = [instruction] if isinstance(instruction, str) else []
    if isinstance(input_value, str) and input_value.strip():
        user_parts.append(input_value)
    messages = [ConversationMessage("user", "\n\n".join(user_parts))]
    if isinstance(output, str):
        messages.append(ConversationMessage("assistant", output))
    findings.extend(_flow_findings(result, record, messages, id_field))
    example = ConversationExample(
        record,
        "alpaca",
        messages,
        _training_units(messages),
        _canonical_conversation(messages),
    )
    return example, findings


def adapt_chat(
    result: ReadResult, schema: str, id_field: str | None = None
) -> ChatResult:
    if schema not in SCHEMAS:
        raise ConfigurationError(f"Unsupported schema: {schema}")
    resolved = detect_schema(result) if schema == "auto" else schema
    if resolved == "generic":
        return ChatResult("generic", [], [])
    adapters = {
        "openai": _adapt_openai,
        "sharegpt": _adapt_sharegpt,
        "alpaca": _adapt_alpaca,
    }
    examples: list[ConversationExample] = []
    findings: list[Finding] = []
    for record in result.records:
        example, record_findings = adapters[resolved](result, record, id_field)
        findings.extend(record_findings)
        if example is not None:
            examples.append(example)
    return ChatResult(resolved, examples, findings)


def analyze_chat(
    result: ReadResult, config: ScanConfig, chat: ChatResult
) -> tuple[list[str], list[Finding]]:
    if chat.schema == "generic":
        return analyze(result, config)
    synthetic = ReadResult(
        result.path,
        result.format,
        [
            Record(
                example.record.index,
                example.record.line,
                {"conversation": example.raw_text},
            )
            for example in chat.examples
        ],
        ["conversation"],
        [],
    )
    generic_config = ScanConfig(
        fields=["conversation"],
        id_field=None,
        similarity_threshold=config.similarity_threshold,
        min_text_length=config.min_text_length,
        schema=chat.schema,
    )
    _, duplicate_findings = analyze(synthetic, generic_config)
    for finding in duplicate_findings:
        for ref in finding.records:
            ref["dataset"] = str(result.path.resolve())
            original = next(
                (
                    example.record
                    for example in chat.examples
                    if example.record.index == ref.get("row")
                ),
                None,
            )
            if (
                original is not None
                and config.id_field
                and config.id_field in original.data
            ):
                ref["id"] = original.data[config.id_field]
    findings = list(result.findings) + chat.findings + duplicate_findings
    findings.extend(_unit_conflicts(result, config, chat.examples))
    return ["conversation"], findings


def _unit_conflicts(
    result: ReadResult, config: ScanConfig, examples: list[ConversationExample]
) -> list[Finding]:
    from .normalize import normalize_text

    grouped: dict[str, list[tuple[ConversationExample, TrainingUnit]]] = defaultdict(
        list
    )
    for example in examples:
        for unit in example.units:
            if unit.context.strip():
                grouped[normalize_text(unit.context)].append((example, unit))
    findings: list[Finding] = []
    for context, entries in grouped.items():
        targets = {normalize_text(unit.target) for _, unit in entries}
        if len(targets) > 1:
            findings.append(
                Finding(
                    "CONFLICTING_OUTPUT",
                    "conflicting_output",
                    "error",
                    "The same normalized conversation context has different assistant targets.",
                    [
                        source_ref(result, example.record, config.id_field, unit.turn)
                        for example, unit in entries
                    ],
                    evidence={
                        "context_preview": context[:160],
                        "distinct_targets": len(targets),
                    },
                )
            )
    return findings
