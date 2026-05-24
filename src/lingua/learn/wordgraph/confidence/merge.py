"""Merge n-fold confidence records."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _flatten_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    flat = dict(record)
    component_nll = flat.pop("component_nll", {}) or {}
    component_weights = flat.pop("component_weights", {}) or {}
    for key, value in component_nll.items():
        flat[f"component_nll.{key}"] = value
    for key, value in component_weights.items():
        flat[f"component_weight.{key}"] = value
    unknown_edge_labels = flat.get("unknown_edge_labels")
    if isinstance(unknown_edge_labels, list):
        flat["unknown_edge_labels"] = "|".join(str(x) for x in unknown_edge_labels)
    return flat


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [_flatten_for_csv(row) for row in rows]
    fieldnames: List[str] = []
    for row in flat_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def merge_fold_results(
    *,
    folds_dir: Path | str,
    output_jsonl: Path | str,
    output_csv: Optional[Path | str] = None,
    summary_path: Optional[Path | str] = None,
    low_confidence_path: Optional[Path | str] = None,
    low_confidence_limit: int = 200,
) -> Dict[str, Any]:
    """Merge `fold_*/confidence/detailed_results.jsonl` into one dataset."""

    folds_dir = Path(folds_dir)
    records: List[Dict[str, Any]] = []

    for fold_dir in sorted(folds_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1])):
        detailed_path = fold_dir / "confidence" / "detailed_results.jsonl"
        if detailed_path.exists():
            records.extend(read_jsonl(detailed_path))

    records.sort(key=lambda row: (row.get("global_idx") is None, row.get("global_idx", 0)))
    output_jsonl = Path(output_jsonl)
    write_jsonl(output_jsonl, records)

    if output_csv:
        write_csv(Path(output_csv), records)

    valid = [row for row in records if row.get("confidence") is not None]
    errors = [row for row in records if row.get("confidence") is None]
    sorted_valid = sorted(valid, key=lambda row: float(row["confidence"]))

    if low_confidence_path:
        write_jsonl(Path(low_confidence_path), sorted_valid[:low_confidence_limit])

    confidences = [float(row["confidence"]) for row in valid]
    summary = {
        "total_records": len(records),
        "scored_records": len(valid),
        "error_records": len(errors),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
        "min_confidence": min(confidences) if confidences else None,
        "max_confidence": max(confidences) if confidences else None,
        "lowest_confidence_count": min(low_confidence_limit, len(sorted_valid)),
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv) if output_csv else None,
        "low_confidence_path": str(low_confidence_path) if low_confidence_path else None,
    }

    if summary_path:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
