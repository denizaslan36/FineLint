from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from dataset_doctor.analyzer import ConfigurationError, analyze
from dataset_doctor.cli import main
from dataset_doctor.models import ScanConfig
from dataset_doctor.readers import read_dataset
from dataset_doctor.report import build_report


class DatasetDoctorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, name: str, content: str) -> Path:
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def scan(self, path: Path, **kwargs):
        result = read_dataset(path)
        config = ScanConfig(**kwargs)
        fields, findings = analyze(result, config)
        return result, config, fields, findings

    def test_jsonl_structural_errors_are_recoverable(self) -> None:
        path = self.write(
            "broken.jsonl",
            '{"text":"valid row","kind":"a"}\n'
            '{not json}\n'
            '\n'
            '[1,2,3]\n'
            '{"text":12,"kind":"b"}\n',
        )
        result, _, _, findings = self.scan(path, fields=["text"], required_fields=["text"])
        self.assertEqual(len(result.records), 2)
        codes = [finding.code for finding in findings]
        self.assertIn("MALFORMED_JSON", codes)
        self.assertIn("EMPTY_JSONL_LINE", codes)
        self.assertIn("JSONL_RECORD_NOT_OBJECT", codes)
        self.assertIn("NON_STRING_TEXT_FIELD", codes)

    def test_csv_width_and_bom(self) -> None:
        path = self.write("rows.csv", "\ufefftext,label\nhello,a\ntoo,many,values\n")
        result, _, fields, findings = self.scan(path)
        self.assertEqual(result.fields, ["text", "label"])
        self.assertEqual(fields, ["text", "label"])
        self.assertIn("CSV_ROW_WIDTH_MISMATCH", [finding.code for finding in findings])

    def test_exact_and_normalized_duplicates_are_grouped(self) -> None:
        path = self.write(
            "duplicates.jsonl",
            '\n'.join(
                [
                    json.dumps({"text": "Hello   World"}),
                    json.dumps({"text": "Hello   World"}),
                    json.dumps({"text": " hello world "}),
                ]
            ),
        )
        _, _, _, findings = self.scan(path, fields=["text"])
        exact = [finding for finding in findings if finding.code == "EXACT_DUPLICATE"]
        normalized = [finding for finding in findings if finding.code == "NORMALIZED_DUPLICATE"]
        self.assertEqual(len(exact), 1)
        self.assertEqual(len(exact[0].records), 2)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(len(normalized[0].records), 3)

    def test_near_duplicates_include_verified_similarity(self) -> None:
        first = "The quick brown fox jumps over the lazy dog beside the quiet river every morning."
        second = "The quick brown fox jumps over the lazy dog beside the quiet river each morning."
        path = self.write(
            "near.jsonl",
            json.dumps({"text": first}) + "\n" + json.dumps({"text": second}) + "\n",
        )
        _, _, _, findings = self.scan(path, fields=["text"], similarity_threshold=0.80)
        near = [finding for finding in findings if finding.code == "NEAR_DUPLICATE"]
        self.assertEqual(len(near), 1)
        self.assertEqual(len(near[0].records), 2)
        self.assertGreaterEqual(near[0].evidence["matches"][0]["similarity"], 0.80)

    def test_conflicting_outputs(self) -> None:
        rows = [
            {"prompt": "Say hello", "answer": "Hello"},
            {"prompt": " say   HELLO ", "answer": "Goodbye"},
        ]
        path = self.write("conflicts.jsonl", "\n".join(json.dumps(row) for row in rows))
        _, _, _, findings = self.scan(
            path,
            fields=["prompt", "answer"],
            input_fields=["prompt"],
            output_fields=["answer"],
        )
        conflict = [finding for finding in findings if finding.code == "CONFLICTING_OUTPUT"]
        self.assertEqual(len(conflict), 1)
        self.assertEqual([record["row"] for record in conflict[0].records], [1, 2])

    def test_roles_must_be_paired(self) -> None:
        path = self.write("roles.jsonl", '{"prompt":"hello"}\n')
        with self.assertRaises(ConfigurationError):
            self.scan(path, fields=["prompt"], input_fields=["prompt"])

    def test_report_has_no_quality_verdict(self) -> None:
        path = self.write("clean.jsonl", '{"text":"A sufficiently long and unique example for this dataset record."}\n')
        result, config, fields, findings = self.scan(path, fields=["text"])
        report = build_report(result, config, fields, findings)
        self.assertEqual(set(report), {"metadata", "configuration", "summary", "findings"})
        encoded = json.dumps(report).casefold()
        self.assertNotIn("quality_score", encoded)
        self.assertNotIn("verdict", encoded)

    def test_cli_writes_full_report_and_returns_success_with_findings(self) -> None:
        path = self.write("cli.jsonl", '{"text":"same"}\n{"text":"same"}\n')
        report_path = self.root / "report.json"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            main(["inspect", str(path), "--fields", "text", "--report", str(report_path)])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertIn("EXACT_DUPLICATE", report["summary"]["by_code"])
        self.assertIn("Report:", stdout.getvalue())

    def test_cli_fatal_error_returns_nonzero(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(["inspect", str(self.root / "missing.jsonl")])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("does not exist", stderr.getvalue())

    def test_report_cannot_overwrite_source_dataset(self) -> None:
        path = self.write("protected.jsonl", '{"text":"keep this source record"}\n')
        original = path.read_text(encoding="utf-8")
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(["inspect", str(path), "--fields", "text", "--report", str(path)])
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_invalid_values_do_not_create_false_near_duplicates(self) -> None:
        path = self.write("types.jsonl", '{"text":null}\n{"text":12}\n')
        _, _, _, findings = self.scan(path, fields=["text"])
        self.assertNotIn("NEAR_DUPLICATE", [finding.code for finding in findings])


if __name__ == "__main__":
    unittest.main()
