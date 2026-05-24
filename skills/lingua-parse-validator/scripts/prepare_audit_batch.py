#!/usr/bin/env python3
"""Create resumable per-line audit skeletons for a Lingua JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validate_graph import validate_graph


def structural_status(errors: list[dict[str, Any]]) -> str:
    return "pass" if not errors else "fail"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--start", type=int, default=1, help="1-based inclusive start line")
    parser.add_argument("--end", type=int, default=None, help="1-based inclusive end line")
    parser.add_argument("--output", type=Path, default=None, help="output JSONL path; defaults to stdout")
    parser.add_argument(
        "--mark-screened",
        action="store_true",
        help="mark review_status as screened instead of pending_review",
    )
    args = parser.parse_args()

    records: list[str] = []
    with args.dataset.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number < args.start:
                continue
            if args.end is not None and line_number > args.end:
                break
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                sentence = sample.get("sentence", "")
                structural_errors = validate_graph(line_number, sample)
            except Exception as exc:  # noqa: BLE001
                sentence = ""
                structural_errors = [
                    {
                        "code": "FORM_ERROR",
                        "location": str(line_number),
                        "message": f"failed to parse sample: {exc}",
                    }
                ]

            record = {
                "line_number": line_number,
                "sentence": sentence,
                "structural_status": structural_status(structural_errors),
                "structural_errors": structural_errors,
                "linguistic_verdict": "not_reviewed",
                "confidence": 0.0,
                "errors": [],
                "suggested_fix": None,
                "proposed_corrected_sample": None,
                "review_status": "screened" if args.mark_screened else "pending_review",
                "notes": [],
            }
            records.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))

    output = "\n".join(records) + ("\n" if records else "")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
