#!/usr/bin/env python3
"""Batch structural validation for Lingua JSONL datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from validate_graph import extract_graph, is_blocking_finding, validate_graph


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-errors", type=int, default=200)
    args = parser.parse_args()

    code_counts: Counter[str] = Counter()
    warning_code_counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    samples = 0
    node_count = 0
    edge_count = 0

    with args.dataset.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            samples += 1
            try:
                sample = json.loads(line)
                _, nodes, edges = extract_graph(sample)
                node_count += len(nodes)
                edge_count += len(edges)
                sample_errors = validate_graph(line_number, sample)
            except Exception as exc:  # noqa: BLE001
                sample_errors = [
                    {
                        "code": "FORM_ERROR",
                        "severity": "error",
                        "location": str(line_number),
                        "message": f"failed to parse sample: {exc}",
                    }
                ]
            for error in sample_errors:
                if is_blocking_finding(error):
                    code_counts[error["code"]] += 1
                    errors.append(error)
                else:
                    warning_code_counts[error["code"]] += 1
                    warnings.append(error)

    result = {
        "ok": not errors,
        "samples": samples,
        "nodes": node_count,
        "edges": edge_count,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "finding_count": len(errors) + len(warnings),
        "error_code_counts": dict(sorted(code_counts.items())),
        "warning_code_counts": dict(sorted(warning_code_counts.items())),
        "errors": (errors + warnings)[: args.max_errors],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
