<div align="center">

# FineLint

**Facts for fine-tuning data. No models, no verdicts.**

[![Test](https://github.com/denizaslan36/FineLint/actions/workflows/test.yml/badge.svg)](https://github.com/denizaslan36/FineLint/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![AI free](https://img.shields.io/badge/AI-free-2ea44f)](#what-finelint-reports)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Terminal only · Deterministic · Local · Vendor neutral

</div>

FineLint inspects text datasets before fine-tuning. It identifies malformed chat
records, duplicate examples, suspicious conversation flows, conflicting targets,
and leakage between train, validation, and test splits.

FineLint does **not** call a model, create embeddings, use an API, change your source
files, or label a dataset as good or bad. Every result points back to concrete rows,
lines, turns, and deterministic evidence.

## Install

```bash
pipx install finelint
```

Or install the current source version:

```bash
git clone https://github.com/denizaslan36/FineLint.git
cd FineLint
pipx install .
```

FineLint has no runtime dependencies and supports Python 3.10 through 3.14.

## Inspect one dataset

```bash
finelint inspect conversations.jsonl
```

FineLint detects these chat schemas automatically:

| Schema | Recognized shape |
| --- | --- |
| OpenAI chat | `messages` with `role`, `content`, and optional tool calls |
| ShareGPT | `conversations` with `from` and `value` |
| Alpaca | `instruction`, optional `input`, and `output` |
| Generic text | Top-level string fields in CSV or JSONL |

Choose a schema explicitly when needed:

```bash
finelint inspect conversations.jsonl --schema openai
finelint inspect sharegpt.jsonl --schema sharegpt
finelint inspect alpaca.jsonl --schema alpaca
finelint inspect custom.csv --schema generic --fields prompt --fields response
```

If a file contains conflicting chat shapes, auto-detection stops and asks for an
explicit schema instead of guessing.

## Compare dataset splits

```bash
finelint compare \
  --split train=train.jsonl \
  --split validation=validation.jsonl \
  --split test=test.jsonl
```

`compare` finds cross-split overlap at two levels:

- whole examples: exact, normalized, and highly similar conversations;
- training contexts: repeated prompts or histories, including contexts paired with
  different assistant targets.

Each split is adapted independently, so an OpenAI-formatted train file can be
compared with a ShareGPT validation file. Override one split when auto-detection is
not enough:

```bash
finelint compare \
  --split train=train.jsonl \
  --split validation=validation.jsonl \
  --schema train=openai \
  --schema validation=sharegpt
```

## What FineLint reports

| Area | Examples |
| --- | --- |
| File structure | Malformed JSONL, non-object rows, empty lines, malformed CSV, wrong row widths |
| Chat schema | Missing messages, invalid roles or content, empty turns, missing assistant targets |
| Tool calling | Invalid tool definitions, malformed arguments JSON, undefined tools, orphaned results |
| Conversation flow | Consecutive roles, late system messages, repeated turns |
| Duplication | Exact copies, Unicode/case/whitespace-hidden copies, near duplicates |
| Split leakage | Repeated examples or contexts across named splits, conflicting targets |

Definite schema violations are `error` findings. Patterns that may be intentional are
`warning` findings. Neither severity is an overall dataset score or verdict.

Near-duplicate candidates come from a bounded deterministic MinHash/LSH index and
are verified with character 5-gram Jaccard similarity. Adjust the conservative
default threshold when needed:

```bash
finelint inspect data.jsonl --similarity-threshold 0.92 --min-text-length 60
```

## Outputs

Every successful command writes two JSON files:

```text
conversations.finelint-report.json
conversations.finelint-affected.json
```

The report contains metadata, the effective configuration, counts, and complete
findings. Every finding has a deterministic ID such as `FL-a62db0194f2e9c11`.

The affected-records file is a compact index of every referenced record and its
reason codes. It deliberately contains no `keep`, `delete`, or `fix` decision.

```json
{
  "report_schema_version": 3,
  "records": [
    {
      "split": "validation",
      "dataset": "/data/validation.jsonl",
      "row": 204,
      "line": 204,
      "turn": 1,
      "reasons": [
        {
          "finding_id": "FL-a62db0194f2e9c11",
          "code": "CROSS_SPLIT_EXACT_CONTEXT",
          "category": "split_leakage",
          "severity": "warning"
        }
      ]
    }
  ]
}
```

Use custom output paths with `--report` and `--affected-records`. FineLint refuses
to overwrite an input dataset.

A completed scan exits with code `0`, even when it reports findings. Invalid input,
configuration problems, and fatal processing errors return exit code `1`. CI policy
failures use exit code `2` after both report files have been written.

## Project configuration

FineLint automatically reads `.finelint.json` from the current working directory.
Use `--config PATH` to select another file or `--no-config` to disable discovery.
Command-line values override project configuration.

```json
{
  "version": 1,
  "fail_on": "error",
  "fail_scope": "all",
  "disabled_rules": ["CHAT_CONSECUTIVE_ROLE"],
  "inspect": {
    "schema": "auto",
    "fields": ["prompt", "response"],
    "input_fields": ["prompt"],
    "output_fields": ["response"],
    "id_field": "id",
    "similarity_threshold": 0.92,
    "min_text_length": 60
  },
  "compare": {
    "id_field": "id",
    "similarity_threshold": 0.92,
    "min_text_length": 60,
    "schemas": {
      "train": "openai",
      "validation": "sharegpt"
    }
  }
}
```

Split paths deliberately stay on the command line. Unknown configuration keys and
invalid values are rejected instead of being silently ignored.

## CI gates and baselines

Use `--fail-on error` or `--fail-on warning` when findings should fail a CI job.
The default remains `never`, preserving FineLint's report-only behavior.

To fail only for regressions, first keep a FineLint report as the baseline and then
compare later scans against it:

```bash
finelint compare \
  --split train=train.jsonl \
  --split validation=validation.jsonl \
  --baseline baseline-report.json \
  --fail-on error \
  --fail-scope new
```

The report classifies current findings as `new` or `unchanged` and lists resolved
finding IDs in its `baseline` section. Baselines must use report schema version 3;
older reports should be regenerated with FineLint 0.3.

In GitHub Actions, the CLI can be used directly as a required check:

```yaml
- uses: actions/checkout@v7
- uses: actions/setup-python@v7
  with:
    python-version: "3.12"
- run: python -m pip install .
- run: finelint inspect data/train.jsonl --fail-on error
```

## Generic datasets

The original CSV/JSONL field inspector remains available:

```bash
finelint inspect data.jsonl \
  --schema generic \
  --fields instruction \
  --fields response \
  --input-fields instruction \
  --output-fields response \
  --required-fields instruction \
  --id-field id
```

Field options are repeatable. `--input-fields` and `--output-fields` must be supplied
together.

## Development

```bash
git clone https://github.com/denizaslan36/FineLint.git
cd FineLint

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 -m unittest discover -s tests -v

PYTHONPATH=src python3 -m finelint --help
```

The runtime uses only the Python standard library. Pull requests should preserve
deterministic output, source-file safety, and the no-model/no-verdict boundary.

## Roadmap

- dataset facts: length, role, source, Unicode script, and template distributions;
- SARIF/JUnit output and large-dataset performance benchmarks;
- DPO, evaluation, multimodal, and deterministic PII/secret checks;
- a stable adapter and rule extension interface.

## License

[MIT](LICENSE)
