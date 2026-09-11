<div align="center">

# Dataset-Doctor

**Find duplicate, suspicious, and structurally broken examples before they reach your training pipeline.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![AI free](https://img.shields.io/badge/AI-free-2ea44f)](#what-it-checks)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Terminal only · Deterministic · Local · No quality score

</div>

Dataset-Doctor is a focused inspector for CSV and JSONL text datasets. Give it a
file and it returns an actionable list of malformed records, exact copies,
normalization-hidden duplicates, highly similar examples, and conflicting outputs.

It does **not** call a model, create embeddings, use the network, change your source
file, or decide whether your dataset is “good” or “bad.”

## Quick start

```bash
git clone https://github.com/denizaslan36/Dataset-Doctor.git
cd Dataset-Doctor
pipx install .

dataset-doctor inspect data.jsonl
```

No runtime dependencies are required. Python 3.10 or newer is enough.

```text
Dataset: data.jsonl
Text fields: prompt, answer
Findings: 3
  conflicting_output: 1
  exact_duplicate: 1
  structural: 1
Examples:
  [MISSING_REQUIRED_FIELD] Required field 'prompt' is missing or empty. (row 5)
  [EXACT_DUPLICATE] 2 records contain exactly the same selected field values. (row 3, row 4)
  [CONFLICTING_OUTPUT] The same normalized input has different normalized outputs. (row 1, row 2, row 3, row 4)
Report: /path/to/data.doctor-report.json
```

The terminal stays compact. The generated JSON report contains the complete list,
including every affected row and line number.

## What it checks

| Check | What it catches |
| --- | --- |
| Structural errors | Malformed JSONL, non-object records, empty lines, malformed CSV, and wrong CSV row widths |
| Missing data | Required or commonly present fields that are missing, null, blank, or the wrong type |
| Exact duplicates | Records with identical selected field values |
| Normalized duplicates | Copies hidden by Unicode forms, casing, or whitespace differences |
| Near duplicates | Highly similar records verified with character 5-gram Jaccard similarity |
| Conflicting outputs | The same normalized input paired with different normalized outputs |

Near-duplicate candidates are selected with a bounded, deterministic MinHash/LSH
index. Candidate matches are then verified with exact Jaccard calculation. This keeps
the scan practical without using embeddings or an all-pairs comparison.

## Usage

Dataset-Doctor detects top-level text fields automatically:

```bash
dataset-doctor inspect conversations.jsonl
dataset-doctor inspect examples.csv
```

For a known schema, select fields and assign their roles explicitly:

```bash
dataset-doctor inspect conversations.jsonl \
  --fields instruction \
  --fields response \
  --input-fields instruction \
  --output-fields response \
  --required-fields instruction \
  --id-field id \
  --similarity-threshold 0.95 \
  --report ./conversations.doctor-report.json
```

Field options are repeatable. `--input-fields` and `--output-fields` must be used
together.

### Options

| Option | Purpose | Default |
| --- | --- | --- |
| `--format csv\|jsonl` | Override extension-based format detection | File extension |
| `--fields FIELD` | Choose a text field to inspect | Auto-detected |
| `--input-fields FIELD` | Define input fields for conflict checks | Disabled |
| `--output-fields FIELD` | Define output fields for conflict checks | Disabled |
| `--required-fields FIELD` | Require a non-empty value | Disabled |
| `--id-field FIELD` | Add a dataset ID to record references | Row and line only |
| `--similarity-threshold FLOAT` | Set the near-duplicate Jaccard threshold | `0.95` |
| `--min-text-length INTEGER` | Minimum text length for near-duplicate checks | `40` |
| `--report PATH` | Choose the JSON report path | `./<name>.doctor-report.json` |

## Report format

Reports have four stable top-level sections:

```json
{
  "metadata": {
    "tool": "dataset-doctor",
    "version": "0.1.0",
    "dataset": "/path/to/data.jsonl",
    "format": "jsonl",
    "records_read": 5000
  },
  "configuration": {
    "fields": ["prompt", "answer"],
    "similarity_threshold": 0.95,
    "min_text_length": 40
  },
  "summary": {
    "total_findings": 12,
    "by_category": {"exact_duplicate": 3, "structural": 9}
  },
  "findings": []
}
```

Each finding includes a stable code, category, severity, explanation, affected
records, relevant fields, and evidence where applicable. Similar records are grouped
instead of producing a noisy list of every possible pair.

A completed scan exits with code `0`, even when findings exist. Invalid arguments,
unreadable input, and fatal processing errors return a nonzero code.

## Development

Run from source:

```bash
PYTHONPATH=src python3 -m dataset_doctor inspect data.jsonl
```

Run the test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The project uses only the Python standard library at runtime. The test suite covers
CSV and JSONL parsing, structural errors, duplicate grouping, similarity evidence,
conflicting outputs, report safety, and CLI exit behavior.

## License

[MIT](LICENSE)
