#!/usr/bin/env python3
"""Validate a Lingua parse-validator report JSON shape without external deps."""

from __future__ import annotations

import json
import sys
from pathlib import Path


TASKS = {"validate", "parse", "batch_validate"}
VERDICTS = {"valid", "minor_error", "major_error", "invalid", "not_applicable"}
SEVERITIES = {"minor", "major", "critical"}
REQUIRED_TOP_LEVEL = {
    "task",
    "sentence",
    "verdict",
    "confidence",
    "summary",
    "errors",
    "corrected_annotation",
    "parse",
    "notes",
}
REQUIRED_ERROR = {
    "error_type",
    "severity",
    "location",
    "violated_rule",
    "explanation",
    "suggested_fix",
    "confidence",
}


def fail(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_report(report: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["$: report must be a JSON object"]

    missing = REQUIRED_TOP_LEVEL - set(report)
    for key in sorted(missing):
        fail(errors, "$", f"missing required key {key!r}")

    extra = set(report) - REQUIRED_TOP_LEVEL
    for key in sorted(extra):
        fail(errors, "$", f"unexpected key {key!r}")

    if report.get("task") not in TASKS:
        fail(errors, "$.task", f"must be one of {sorted(TASKS)}")
    if not isinstance(report.get("sentence"), str):
        fail(errors, "$.sentence", "must be a string")
    if report.get("verdict") not in VERDICTS:
        fail(errors, "$.verdict", f"must be one of {sorted(VERDICTS)}")

    confidence = report.get("confidence")
    if not is_number(confidence) or not 0 <= float(confidence) <= 1:
        fail(errors, "$.confidence", "must be a number in [0, 1]")

    if not isinstance(report.get("summary"), str):
        fail(errors, "$.summary", "must be a string")

    report_errors = report.get("errors")
    if not isinstance(report_errors, list):
        fail(errors, "$.errors", "must be an array")
    else:
        for index, item in enumerate(report_errors):
            path = f"$.errors[{index}]"
            if not isinstance(item, dict):
                fail(errors, path, "must be an object")
                continue
            missing_error_keys = REQUIRED_ERROR - set(item)
            for key in sorted(missing_error_keys):
                fail(errors, path, f"missing required key {key!r}")
            extra_error_keys = set(item) - REQUIRED_ERROR
            for key in sorted(extra_error_keys):
                fail(errors, path, f"unexpected key {key!r}")
            if item.get("severity") not in SEVERITIES:
                fail(errors, f"{path}.severity", f"must be one of {sorted(SEVERITIES)}")
            err_conf = item.get("confidence")
            if not is_number(err_conf) or not 0 <= float(err_conf) <= 1:
                fail(errors, f"{path}.confidence", "must be a number in [0, 1]")
            for key in ["error_type", "location", "violated_rule", "explanation", "suggested_fix"]:
                if not isinstance(item.get(key), str):
                    fail(errors, f"{path}.{key}", "must be a string")

    if report.get("corrected_annotation") is not None and not isinstance(
        report.get("corrected_annotation"), dict
    ):
        fail(errors, "$.corrected_annotation", "must be an object or null")
    if report.get("verdict") in {"minor_error", "major_error"} and report.get(
        "corrected_annotation"
    ) is None:
        fail(
            errors,
            "$.corrected_annotation",
            "must contain a complete corrected annotation for minor_error or major_error",
        )
    if report.get("parse") is not None and not isinstance(report.get("parse"), dict):
        fail(errors, "$.parse", "must be an object or null")
    if not isinstance(report.get("notes"), list) or not all(
        isinstance(note, str) for note in report.get("notes", [])
    ):
        fail(errors, "$.notes", "must be an array of strings")

    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_output.py <report.json>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    try:
        report = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "errors": [f"failed to read JSON: {exc}"]}, indent=2))
        return 1

    errors = validate_report(report)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
