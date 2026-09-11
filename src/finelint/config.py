from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analyzer import ConfigurationError


CONFIG_FILENAME = ".finelint.json"
CONFIG_VERSION = 1

_TOP_LEVEL_KEYS = {
    "version",
    "fail_on",
    "fail_scope",
    "disabled_rules",
    "inspect",
    "compare",
}
_INSPECT_KEYS = {
    "schema",
    "fields",
    "input_fields",
    "output_fields",
    "required_fields",
    "id_field",
    "similarity_threshold",
    "min_text_length",
}
_COMPARE_KEYS = {
    "schemas",
    "id_field",
    "similarity_threshold",
    "min_text_length",
}


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    path: Path | None
    values: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name, {})
        return value if isinstance(value, dict) else {}


def _unknown_keys(value: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigurationError(
            f"Unknown {location} configuration key(s): {', '.join(unknown)}"
        )


def _string_list(value: Any, location: str) -> None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ConfigurationError(f"{location} must be an array of non-empty strings.")
    if len(value) != len(set(value)):
        raise ConfigurationError(f"{location} must not contain duplicate values.")


def _validate_optional_string(value: Any, location: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ConfigurationError(f"{location} must be a non-empty string or null.")


def _validate_number(value: Any, location: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{location} must be a number.")


def _validate_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError("FineLint configuration must be a JSON object.")
    _unknown_keys(value, _TOP_LEVEL_KEYS, "top-level")
    if value.get("version") != CONFIG_VERSION:
        raise ConfigurationError(
            f"FineLint configuration version must be {CONFIG_VERSION}."
        )
    if "fail_on" in value and value["fail_on"] not in {"never", "error", "warning"}:
        raise ConfigurationError("fail_on must be never, error, or warning.")
    if "fail_scope" in value and value["fail_scope"] not in {"all", "new"}:
        raise ConfigurationError("fail_scope must be all or new.")
    if "disabled_rules" in value:
        _string_list(value["disabled_rules"], "disabled_rules")

    for section_name, allowed in (
        ("inspect", _INSPECT_KEYS),
        ("compare", _COMPARE_KEYS),
    ):
        section = value.get(section_name, {})
        if not isinstance(section, dict):
            raise ConfigurationError(f"{section_name} must be a JSON object.")
        _unknown_keys(section, allowed, section_name)
        for key in (
            "fields",
            "input_fields",
            "output_fields",
            "required_fields",
        ):
            if key in section:
                _string_list(section[key], f"{section_name}.{key}")
        if "id_field" in section:
            _validate_optional_string(section["id_field"], f"{section_name}.id_field")
        if "similarity_threshold" in section:
            _validate_number(
                section["similarity_threshold"],
                f"{section_name}.similarity_threshold",
            )
        if "min_text_length" in section and (
            isinstance(section["min_text_length"], bool)
            or not isinstance(section["min_text_length"], int)
        ):
            raise ConfigurationError(f"{section_name}.min_text_length must be an integer.")

    inspect = value.get("inspect", {})
    if "schema" in inspect and inspect["schema"] not in {
        "auto",
        "generic",
        "openai",
        "sharegpt",
        "alpaca",
    }:
        raise ConfigurationError("inspect.schema is not a supported schema.")

    compare = value.get("compare", {})
    if "schemas" in compare:
        schemas = compare["schemas"]
        if not isinstance(schemas, dict) or not all(
            isinstance(name, str)
            and name
            and schema in {"openai", "sharegpt", "alpaca"}
            for name, schema in schemas.items()
        ):
            raise ConfigurationError(
                "compare.schemas must map split names to supported chat schemas."
            )
    return value


def load_project_config(
    explicit_path: Path | None, disabled: bool, cwd: Path | None = None
) -> ProjectConfig:
    if disabled:
        return ProjectConfig(None, {})
    path = explicit_path or (cwd or Path.cwd()) / CONFIG_FILENAME
    if explicit_path is None and not path.is_file():
        return ProjectConfig(None, {})
    if not path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    if not path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Could not parse configuration: {exc.msg} at line {exc.lineno}, column {exc.colno}."
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"Could not read configuration: {exc}") from exc
    return ProjectConfig(path.resolve(), _validate_config(value))
