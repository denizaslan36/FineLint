from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from finelint.chat import adapt_chat, analyze_chat
from finelint.cli import main
from finelint.models import ScanConfig
from finelint.readers import read_dataset
from finelint.report import build_report


class CiAndConfigTests(unittest.TestCase):
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

    def run_inspect(self, dataset: Path, report: Path, *extra: str) -> None:
        with redirect_stdout(io.StringIO()):
            main(
                [
                    "inspect",
                    str(dataset),
                    "--report",
                    str(report),
                    "--affected-records",
                    str(report.with_name(report.stem + "-affected.json")),
                    *extra,
                ]
            )

    def test_config_is_applied_and_cli_values_take_precedence(self) -> None:
        dataset = self.write_jsonl(
            "rows.jsonl", [{"prompt": "same"}, {"prompt": "same"}]
        )
        config = self.root / "config.json"
        config.write_text(
            json.dumps(
                {
                    "version": 1,
                    "disabled_rules": ["EXACT_DUPLICATE"],
                    "inspect": {
                        "schema": "generic",
                        "fields": ["prompt"],
                        "similarity_threshold": 0.75,
                    },
                }
            ),
            encoding="utf-8",
        )
        report = self.root / "report.json"
        self.run_inspect(
            dataset,
            report,
            "--config",
            str(config),
            "--similarity-threshold",
            "0.88",
        )
        payload = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(payload["report_schema_version"], 3)
        self.assertEqual(payload["configuration"]["similarity_threshold"], 0.88)
        self.assertEqual(payload["configuration"]["source"], str(config.resolve()))
        self.assertEqual(payload["configuration"]["disabled_rules"], ["EXACT_DUPLICATE"])
        self.assertNotIn("EXACT_DUPLICATE", payload["summary"]["by_code"])

    def test_invalid_config_is_a_configuration_error(self) -> None:
        dataset = self.write_jsonl("row.jsonl", [{"text": "value"}])
        config = self.root / "bad.json"
        config.write_text('{"version":1,"surprise":true}', encoding="utf-8")
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            self.run_inspect(dataset, self.root / "report.json", "--config", str(config))
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Unknown top-level", stderr.getvalue())

    def test_compare_uses_configured_schema_map(self) -> None:
        ambiguous = {
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
            "conversations": [
                {"from": "human", "value": "Q"},
                {"from": "gpt", "value": "A"},
            ],
        }
        train = self.write_jsonl("train.jsonl", [ambiguous])
        validation = self.write_jsonl("validation.jsonl", [ambiguous])
        config = self.root / "compare-config.json"
        config.write_text(
            json.dumps(
                {
                    "version": 1,
                    "compare": {
                        "schemas": {"train": "openai", "validation": "sharegpt"}
                    },
                }
            ),
            encoding="utf-8",
        )
        report = self.root / "compare.json"
        with redirect_stdout(io.StringIO()):
            main(
                [
                    "compare",
                    "--split",
                    f"train={train}",
                    "--split",
                    f"validation={validation}",
                    "--config",
                    str(config),
                    "--report",
                    str(report),
                    "--affected-records",
                    str(self.root / "compare-affected.json"),
                ]
            )
        payload = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(
            [item["detected_schema"] for item in payload["metadata"]["splits"]],
            ["openai", "sharegpt"],
        )

    def test_ci_gate_writes_outputs_then_returns_two(self) -> None:
        dataset = self.write_jsonl("duplicate.jsonl", [{"text": "x"}, {"text": "x"}])
        report = self.root / "report.json"
        with self.assertRaises(SystemExit) as raised:
            self.run_inspect(
                dataset, report, "--fields", "text", "--fail-on", "warning"
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertTrue(report.is_file())
        self.assertTrue(report.with_name(report.stem + "-affected.json").is_file())

    def test_baseline_marks_unchanged_and_new_findings(self) -> None:
        dataset = self.write_jsonl("baseline.jsonl", [{"text": "x"}, {"text": "x"}])
        baseline = self.root / "baseline-report.json"
        self.run_inspect(dataset, baseline, "--fields", "text")

        unchanged_report = self.root / "unchanged.json"
        self.run_inspect(
            dataset,
            unchanged_report,
            "--fields",
            "text",
            "--baseline",
            str(baseline),
            "--fail-on",
            "warning",
            "--fail-scope",
            "new",
        )
        unchanged = json.loads(unchanged_report.read_text(encoding="utf-8"))
        self.assertEqual(unchanged["baseline"]["new"], 0)
        original = json.loads(baseline.read_text(encoding="utf-8"))
        self.assertEqual(unchanged["content_sha256"], original["content_sha256"])
        self.assertTrue(
            all(item["baseline_status"] == "unchanged" for item in unchanged["findings"])
        )

        dataset.write_text('{"text":"x"}\n{"text":"y"}\n{"text":""}\n', encoding="utf-8")
        changed_report = self.root / "changed.json"
        with self.assertRaises(SystemExit) as raised:
            self.run_inspect(
                dataset,
                changed_report,
                "--fields",
                "text",
                "--baseline",
                str(baseline),
                "--fail-on",
                "warning",
                "--fail-scope",
                "new",
            )
        self.assertEqual(raised.exception.code, 2)
        changed = json.loads(changed_report.read_text(encoding="utf-8"))
        self.assertGreater(changed["baseline"]["new"], 0)
        self.assertGreater(changed["baseline"]["resolved"], 0)

    def test_finding_id_ignores_line_shifts_for_parsed_rows(self) -> None:
        dataset = self.root / "lines.jsonl"
        dataset.write_text('{"text":"x"}\n{"text":"x"}\n', encoding="utf-8")
        first_report = self.root / "first.json"
        self.run_inspect(dataset, first_report, "--fields", "text")
        first = json.loads(first_report.read_text(encoding="utf-8"))
        first_id = next(
            item["finding_id"]
            for item in first["findings"]
            if item["code"] == "EXACT_DUPLICATE"
        )

        dataset.write_text('\n{"text":"x"}\n{"text":"x"}\n', encoding="utf-8")
        second_report = self.root / "second.json"
        self.run_inspect(dataset, second_report, "--fields", "text")
        second = json.loads(second_report.read_text(encoding="utf-8"))
        second_id = next(
            item["finding_id"]
            for item in second["findings"]
            if item["code"] == "EXACT_DUPLICATE"
        )
        self.assertEqual(first_id, second_id)

    def test_v2_baseline_is_rejected_with_migration_message(self) -> None:
        dataset = self.write_jsonl("row.jsonl", [{"text": "value"}])
        baseline = self.root / "v2.json"
        baseline.write_text(
            json.dumps({"report_schema_version": 2, "findings": []}), encoding="utf-8"
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            self.run_inspect(
                dataset,
                self.root / "report.json",
                "--fields",
                "text",
                "--baseline",
                str(baseline),
            )
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Create a fresh baseline", stderr.getvalue())

    def test_same_row_findings_receive_distinct_stable_ids(self) -> None:
        dataset = self.write_jsonl(
            "alpaca.jsonl", [{"instruction": "", "output": ""}]
        )
        result = read_dataset(dataset)
        config = ScanConfig(schema="alpaca")
        chat = adapt_chat(result, "alpaca")
        fields, findings = analyze_chat(result, config, chat)
        report = build_report(result, config, fields, findings, chat.schema)
        empty_turn_ids = {
            finding["finding_id"]
            for finding in report["findings"]
            if finding["code"] == "CHAT_EMPTY_TURN"
        }
        self.assertEqual(len(empty_turn_ids), 2)

    def test_chat_id_field_must_exist(self) -> None:
        dataset = self.write_jsonl(
            "chat.jsonl",
            [{"messages": [{"role": "assistant", "content": "answer"}]}],
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            self.run_inspect(
                dataset, self.root / "report.json", "--id-field", "missing"
            )
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Unknown field", stderr.getvalue())

    def test_tool_result_lifecycle_findings(self) -> None:
        dataset = self.write_jsonl(
            "tools.jsonl",
            [
                {
                    "tools": [
                        {"type": "function", "function": {"name": "lookup"}}
                    ],
                    "messages": [
                        {"role": "user", "content": "Find it"},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                },
                            ],
                        },
                        {"role": "tool", "tool_call_id": "call_1", "content": "one"},
                        {"role": "tool", "tool_call_id": "call_1", "content": "again"},
                        {"role": "assistant", "content": "done"},
                    ],
                }
            ],
        )
        chat = adapt_chat(read_dataset(dataset), "openai")
        codes = {finding.code for finding in chat.findings}
        self.assertIn("TOOL_RESULT_DUPLICATE", codes)
        self.assertIn("TOOL_RESULT_SEQUENCE", codes)
        self.assertIn("TOOL_RESULT_MISSING", codes)


if __name__ == "__main__":
    unittest.main()
