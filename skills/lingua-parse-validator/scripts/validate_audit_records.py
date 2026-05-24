#!/usr/bin/env python3
"""Validate Lingua per-line audit records without external dependencies."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from validate_graph import validate_graph


REQUIRED_KEYS = {
    "line_number",
    "sentence",
    "structural_status",
    "structural_errors",
    "linguistic_verdict",
    "confidence",
    "errors",
    "suggested_fix",
    "proposed_corrected_sample",
    "review_status",
    "notes",
}
STRUCTURAL_STATUSES = {"pass", "fail"}
VERDICTS = {
    "not_reviewed",
    "valid",
    "minor_error",
    "major_error",
    "invalid",
    "not_applicable",
}
REVIEW_STATUSES = {
    "pending_review",
    "screened",
    "reviewed",
    "needs_human_decision",
    "fix_proposed",
    "fixed",
    "skipped",
}


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def fail(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def validate_record(record: Any, index: int) -> list[str]:
    errors: list[str] = []
    path = f"$[{index}]"
    if not isinstance(record, dict):
        return [f"{path}: record must be an object"]

    missing = REQUIRED_KEYS - set(record)
    for key in sorted(missing):
        fail(errors, path, f"missing required key {key!r}")

    extra = set(record) - REQUIRED_KEYS
    for key in sorted(extra):
        fail(errors, path, f"unexpected key {key!r}")

    if not isinstance(record.get("line_number"), int) or record.get("line_number", 0) < 1:
        fail(errors, f"{path}.line_number", "must be a positive integer")
    if not isinstance(record.get("sentence"), str):
        fail(errors, f"{path}.sentence", "must be a string")
    if record.get("structural_status") not in STRUCTURAL_STATUSES:
        fail(errors, f"{path}.structural_status", "must be pass or fail")
    if not isinstance(record.get("structural_errors"), list):
        fail(errors, f"{path}.structural_errors", "must be an array")
    if record.get("linguistic_verdict") not in VERDICTS:
        fail(errors, f"{path}.linguistic_verdict", f"must be one of {sorted(VERDICTS)}")
    confidence = record.get("confidence")
    if not is_number(confidence) or not 0 <= float(confidence) <= 1:
        fail(errors, f"{path}.confidence", "must be a number in [0, 1]")
    if not isinstance(record.get("errors"), list):
        fail(errors, f"{path}.errors", "must be an array")
    if record.get("review_status") not in REVIEW_STATUSES:
        fail(errors, f"{path}.review_status", f"must be one of {sorted(REVIEW_STATUSES)}")
    if not isinstance(record.get("notes"), list) or not all(
        isinstance(note, str) for note in record.get("notes", [])
    ):
        fail(errors, f"{path}.notes", "must be an array of strings")

    proposed = record.get("proposed_corrected_sample")
    if proposed is not None and not isinstance(proposed, dict):
        fail(errors, f"{path}.proposed_corrected_sample", "must be an object or null")

    if record.get("review_status") in {"fix_proposed", "fixed"}:
        if proposed is None:
            fail(
                errors,
                f"{path}.proposed_corrected_sample",
                "is required for fix_proposed or fixed records",
            )
        else:
            structural_errors = validate_graph(record.get("line_number", index), proposed)
            for structural_error in structural_errors:
                fail(
                    errors,
                    f"{path}.proposed_corrected_sample",
                    json.dumps(structural_error, ensure_ascii=False),
                )

    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_audit_records.py <audit.jsonl>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    errors: list[str] = []
    count = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                count += 1
                try:
                    record = json.loads(line)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"$[{index}]: failed to parse JSON: {exc}")
                    continue
                errors.extend(validate_record(record, index))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"$: failed to read {path}: {exc}")

    print(json.dumps({"ok": not errors, "records": count, "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
