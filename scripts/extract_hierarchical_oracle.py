#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lingua.learn.hierarchical import extract_hierarchical_oracle


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_number, json.loads(line)


def write_jsonl_row(handle: TextIO, row: dict, pretty: bool) -> None:
    if pretty:
        json.dump(row, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    else:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract recursive one-layer expansion targets from Lingua JSONL data."
    )
    parser.add_argument("data_path", type=Path, help="Input Lingua JSONL file")
    parser.add_argument("-o", "--output", type=Path, help="Output JSONL path; defaults to stdout")
    parser.add_argument("--limit", type=int, help="Maximum number of samples to process")
    parser.add_argument(
        "--format",
        choices=("sample", "unit"),
        default="sample",
        help="Write one oracle row per sample or one flattened row per unit",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON rows")
    args = parser.parse_args()

    output_handle: TextIO
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("w", encoding="utf-8")
    else:
        output_handle = sys.stdout

    processed = 0
    try:
        for line_number, sample in iter_jsonl(args.data_path):
            if args.limit is not None and processed >= args.limit:
                break

            oracle = extract_hierarchical_oracle(sample).to_dict()
            sample_id = sample.get("id") or f"{args.data_path.name}:{line_number}"

            if args.format == "sample":
                row = {"sample_id": sample_id, **oracle}
                write_jsonl_row(output_handle, row, args.pretty)
            else:
                base = {
                    "sample_id": sample_id,
                    "sentence": oracle["sentence"],
                    "words": oracle["words"],
                    "root_ids": oracle["root_ids"],
                }
                for unit_index, unit in enumerate(oracle["units"]):
                    row = {**base, "unit_index": unit_index, "unit": unit}
                    write_jsonl_row(output_handle, row, args.pretty)

            processed += 1
    finally:
        if args.output:
            output_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
