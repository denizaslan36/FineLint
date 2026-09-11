from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from finelint.analyzer import ConfigurationError
from finelint.chat import adapt_chat, analyze_chat
from finelint.cli import main
from finelint.models import ScanConfig
from finelint.readers import read_dataset
from finelint.report import build_report


class ChatAndCompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_jsonl(self, name: str, rows: list[dict]) -> Path:
        path = self.root / name
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        return path

    def test_auto_detects_all_supported_chat_schemas(self) -> None:
        cases = {
            "openai": {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            },
            "sharegpt": {
                "conversations": [
                    {"from": "human", "value": "Hi"},
                    {"from": "gpt", "value": "Hello"},
                ]
            },
            "alpaca": {"instruction": "Say hi", "input": "Politely", "output": "Hello"},
        }
        for expected, row in cases.items():
            with self.subTest(schema=expected):
                result = read_dataset(self.write_jsonl(f"{expected}.jsonl", [row]))
                chat = adapt_chat(result, "auto")
                self.assertEqual(chat.schema, expected)
                self.assertEqual(len(chat.examples), 1)
                self.assertEqual(len(chat.examples[0].units), 1)

    def test_multiturn_chat_creates_one_unit_per_assistant_target(self) -> None:
        path = self.write_jsonl(
            "multiturn.jsonl",
            [
                {
                    "messages": [
                        {"role": "system", "content": "Be concise"},
                        {"role": "user", "content": "First"},
                        {"role": "assistant", "content": "One"},
                        {"role": "user", "content": "Second"},
                        {"role": "assistant", "content": "Two"},
                    ]
                }
            ],
        )
        chat = adapt_chat(read_dataset(path), "auto")
        self.assertEqual([unit.turn for unit in chat.examples[0].units], [2, 4])
        self.assertIn("content=One", chat.examples[0].units[1].context)

    def test_ambiguous_and_mixed_schema_detection_stops(self) -> None:
        ambiguous = self.write_jsonl(
            "ambiguous.jsonl",
            [{"messages": [], "conversations": []}],
        )
        with self.assertRaises(ConfigurationError):
            adapt_chat(read_dataset(ambiguous), "auto")

        mixed = self.write_jsonl(
            "mixed.jsonl",
            [
                {"messages": [{"role": "assistant", "content": "A"}]},
                {"instruction": "B", "output": "C"},
            ],
        )
        with self.assertRaises(ConfigurationError):
            adapt_chat(read_dataset(mixed), "auto")

    def test_valid_tool_call_can_have_null_content(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{"city":"Istanbul"}',
                            },
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}},
                }
            ],
            "parallel_tool_calls": False,
        }
        result = read_dataset(self.write_jsonl("tools.jsonl", [row]))
        chat = adapt_chat(result, "openai")
        codes = {finding.code for finding in chat.findings}
        self.assertNotIn("CHAT_EMPTY_TURN", codes)
        self.assertNotIn("TOOL_CALLS_INVALID", codes)
        self.assertEqual(len(chat.examples[0].units), 1)

    def test_invalid_tool_relationships_are_reported(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "Use a tool"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "missing", "arguments": "not-json"},
                        }
                    ],
                },
                {"role": "tool", "content": "result", "tool_call_id": "other_call"},
            ]
        }
        result = read_dataset(self.write_jsonl("bad-tools.jsonl", [row]))
        codes = {finding.code for finding in adapt_chat(result, "openai").findings}
        self.assertIn("TOOL_ARGUMENTS_INVALID_JSON", codes)
        self.assertIn("TOOL_DEFINITION_MISSING", codes)
        self.assertIn("TOOL_RESULT_ORPHANED", codes)

    def test_chat_flow_uses_error_and_warning_certainty(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "One"},
                {"role": "user", "content": "Two"},
                {"role": "system", "content": "Late"},
            ]
        }
        result = read_dataset(self.write_jsonl("flow.jsonl", [row]))
        findings = adapt_chat(result, "openai").findings
        by_code = {finding.code: finding.severity for finding in findings}
        self.assertEqual(by_code["CHAT_CONSECUTIVE_ROLE"], "warning")
        self.assertEqual(by_code["CHAT_SYSTEM_POSITION"], "warning")
        self.assertEqual(by_code["CHAT_NO_ASSISTANT_TARGET"], "error")

    def test_compare_finds_cross_schema_context_leakage_and_conflict(self) -> None:
        train = self.write_jsonl(
            "train.jsonl",
            [
                {
                    "id": "t1",
                    "messages": [
                        {"role": "user", "content": "Shared prompt"},
                        {"role": "assistant", "content": "Answer A"},
                    ],
                }
            ],
        )
        validation = self.write_jsonl(
            "validation.jsonl",
            [
                {
                    "id": "v1",
                    "conversations": [
                        {"from": "human", "value": "Shared prompt"},
                        {"from": "gpt", "value": "Answer B"},
                    ],
                }
            ],
        )
        report = self.root / "comparison.json"
        affected = self.root / "affected.json"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            main(
                [
                    "compare",
                    "--split",
                    f"train={train}",
                    "--split",
                    f"validation={validation}",
                    "--id-field",
                    "id",
                    "--report",
                    str(report),
                    "--affected-records",
                    str(affected),
                ]
            )
        payload = json.loads(report.read_text(encoding="utf-8"))
        codes = payload["summary"]["by_code"]
        self.assertIn("CROSS_SPLIT_EXACT_CONTEXT", codes)
        self.assertIn("CROSS_SPLIT_CONFLICTING_TARGET", codes)
        self.assertEqual(
            {item["detected_schema"] for item in payload["metadata"]["splits"]},
            {"openai", "sharegpt"},
        )
        selected = json.loads(affected.read_text(encoding="utf-8"))["records"]
        self.assertEqual({item["split"] for item in selected}, {"train", "validation"})
        self.assertTrue(all(item["reasons"] for item in selected))

    def test_compare_finds_exact_examples_across_schemas(self) -> None:
        train = self.write_jsonl(
            "same-train.jsonl",
            [
                {
                    "messages": [
                        {"role": "user", "content": "Same prompt"},
                        {"role": "assistant", "content": "Same answer"},
                    ]
                }
            ],
        )
        test = self.write_jsonl(
            "same-test.jsonl",
            [
                {
                    "conversations": [
                        {"from": "human", "value": "Same prompt"},
                        {"from": "gpt", "value": "Same answer"},
                    ]
                }
            ],
        )
        report = self.root / "same-report.json"
        affected = self.root / "same-affected.json"
        with redirect_stdout(io.StringIO()):
            main(
                [
                    "compare",
                    "--split",
                    f"train={train}",
                    "--split",
                    f"test={test}",
                    "--report",
                    str(report),
                    "--affected-records",
                    str(affected),
                ]
            )
        codes = json.loads(report.read_text(encoding="utf-8"))["summary"]["by_code"]
        self.assertIn("CROSS_SPLIT_EXACT_EXAMPLE", codes)
        self.assertIn("CROSS_SPLIT_EXACT_CONTEXT", codes)

    def test_compare_finds_verified_near_contexts(self) -> None:
        first = "The quick brown fox jumps over the lazy dog beside the quiet river every morning."
        second = "The quick brown fox jumps over the lazy dog beside the quiet river each morning."
        train = self.write_jsonl(
            "near-train.jsonl",
            [{"instruction": first, "output": "A"}],
        )
        test = self.write_jsonl(
            "near-test.jsonl",
            [{"instruction": second, "output": "B"}],
        )
        report = self.root / "near-report.json"
        affected = self.root / "near-affected.json"
        with redirect_stdout(io.StringIO()):
            main(
                [
                    "compare",
                    "--split",
                    f"train={train}",
                    "--split",
                    f"test={test}",
                    "--similarity-threshold",
                    "0.80",
                    "--report",
                    str(report),
                    "--affected-records",
                    str(affected),
                ]
            )
        findings = json.loads(report.read_text(encoding="utf-8"))["findings"]
        near = [
            finding
            for finding in findings
            if finding["code"] == "CROSS_SPLIT_NEAR_CONTEXT"
        ]
        self.assertEqual(len(near), 1)
        self.assertGreaterEqual(near[0]["evidence"]["matches"][0]["similarity"], 0.80)

    def test_compare_rejects_duplicate_split_names(self) -> None:
        path = self.write_jsonl(
            "data.jsonl",
            [
                {
                    "messages": [
                        {"role": "user", "content": "Q"},
                        {"role": "assistant", "content": "A"},
                    ]
                }
            ],
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(["compare", "--split", f"train={path}", "--split", f"train={path}"])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Duplicate split", stderr.getvalue())

    def test_finding_ids_are_stable_and_affected_records_are_automatic(self) -> None:
        path = self.write_jsonl(
            "duplicates.jsonl",
            [
                {
                    "messages": [
                        {"role": "user", "content": "Q"},
                        {"role": "assistant", "content": "A"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "Q"},
                        {"role": "assistant", "content": "A"},
                    ]
                },
            ],
        )
        result = read_dataset(path)
        config = ScanConfig(schema="auto")
        chat = adapt_chat(result, "auto")
        fields, first_findings = analyze_chat(result, config, chat)
        first = build_report(result, config, fields, first_findings, chat.schema)
        _, second_findings = analyze_chat(result, config, chat)
        second = build_report(result, config, fields, second_findings, chat.schema)
        self.assertEqual(
            [finding["finding_id"] for finding in first["findings"]],
            [finding["finding_id"] for finding in second["findings"]],
        )
        self.assertTrue(
            all(
                value.startswith("FL-")
                for value in [finding["finding_id"] for finding in first["findings"]]
            )
        )


if __name__ == "__main__":
    unittest.main()
