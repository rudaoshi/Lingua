#!/usr/bin/env python3
"""Split selected larger apostrophe+s predicates into explicit subject-predicate structure."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
APOSTROPHES = {"'", "’", "‘", "ʼ", "`"}
TARGET_FULL_PHRASES = {
    "she 's got",
    "it 's called",
    "it 's got",
    "it 's been",
    "what 's happening",
    "it 's worth",
}


def clean_token(token: str) -> str:
    return token.strip()


def node_token_indexes(node: dict[str, Any]) -> set[int]:
    spans = node.get("spans")
    if not isinstance(spans, list):
        return set()
    indexes: set[int] = set()
    for start, end in spans:
        indexes.update(range(start, end + 1))
    return indexes


def node_text(words: list[str], node: dict[str, Any]) -> str:
    spans = node.get("spans")
    if isinstance(spans, str):
        return spans
    return "".join(
        token
        for start, end in spans
        for token in words[start : end + 1]
    ).strip()


def exact_s_node(nodes: list[dict[str, Any]], apos_index: int) -> int | None:
    for node_index, node in enumerate(nodes):
        if node.get("spans") == [[apos_index, apos_index + 1]]:
            return node_index
    return None


def covering_factual_nodes(nodes: list[dict[str, Any]], apos_index: int) -> list[int]:
    covering: list[int] = []
    for node_index, node in enumerate(nodes):
        tokens = node_token_indexes(node)
        if (
            node.get("type") == "FactualPredicator"
            and apos_index in tokens
            and apos_index + 1 in tokens
            and len(tokens - {apos_index, apos_index + 1}) > 0
        ):
            covering.append(node_index)
    return covering


def remove_tokens_from_spans(spans: list[list[int]], remove: set[int]) -> list[list[int]]:
    kept_spans: list[list[int]] = []
    for start, end in spans:
        run_start: int | None = None
        prev: int | None = None
        for token_index in range(start, end + 1):
            if token_index in remove:
                if run_start is not None and prev is not None:
                    kept_spans.append([run_start, prev])
                run_start = None
                prev = None
                continue
            if run_start is None:
                run_start = token_index
            prev = token_index
        if run_start is not None and prev is not None:
            kept_spans.append([run_start, prev])
    return kept_spans


def find_target_occurrences(sample: dict[str, Any]) -> list[dict[str, Any]]:
    words = sample["lingua"]["words"]
    nodes = sample["lingua"]["graph"]["nodes"]
    occurrences: list[dict[str, Any]] = []
    for apos_index in range(len(words) - 1):
        if clean_token(words[apos_index]) not in APOSTROPHES:
            continue
        if clean_token(words[apos_index + 1]).lower() != "s":
            continue
        if exact_s_node(nodes, apos_index) is not None:
            continue
        for cover_index in covering_factual_nodes(nodes, apos_index):
            left = clean_token(words[apos_index - 1]).lower() if apos_index > 0 else "<bos>"
            full_phrase = f"{left} {node_text(words, nodes[cover_index]).lower()}"
            if full_phrase in TARGET_FULL_PHRASES:
                occurrences.append(
                    {
                        "apos_index": apos_index,
                        "cover_index": cover_index,
                        "full_phrase": full_phrase,
                    }
                )
    return occurrences


def compact_graph(sample: dict[str, Any]) -> str:
    words = sample["lingua"]["words"]
    nodes = sample["lingua"]["graph"]["nodes"]
    lines = [sample["sentence"], "NODES"]
    for index, node in enumerate(nodes):
        lines.append(f"{index}: {node_text(words, node)!r} {node.get('spans')!r} {node.get('type')}")
    lines.append("EDGES")
    for edge in sample["lingua"]["graph"]["edges"]:
        lines.append(
            f"{edge['start']}->{edge['end']} {edge['label']} | "
            f"{node_text(words, nodes[edge['start']])!r}->{node_text(words, nodes[edge['end']])!r}"
        )
    return "\n".join(lines)


def validate_graph(sample: dict[str, Any]) -> None:
    words = sample["lingua"]["words"]
    nodes = sample["lingua"]["graph"]["nodes"]
    for node_index, node in enumerate(nodes):
        spans = node.get("spans")
        if isinstance(spans, list):
            for start, end in spans:
                assert 0 <= start <= end < len(words), (node_index, node, len(words))
    for edge in sample["lingua"]["graph"]["edges"]:
        assert 0 <= edge["start"] < len(nodes), edge
        assert 0 <= edge["end"] < len(nodes), edge


def repair_occurrence(sample: dict[str, Any], occurrence: dict[str, Any]) -> str:
    words = sample["lingua"]["words"]
    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    apos_index = occurrence["apos_index"]
    cover_index = occurrence["cover_index"]
    cover_node = nodes[cover_index]
    old_surface = node_text(words, cover_node)

    assert exact_s_node(nodes, apos_index) is None
    assert cover_node.get("type") == "FactualPredicator"
    assert apos_index in node_token_indexes(cover_node)
    assert apos_index + 1 in node_token_indexes(cover_node)

    remaining_spans = remove_tokens_from_spans(cover_node["spans"], {apos_index, apos_index + 1})
    assert remaining_spans, (sample["sentence"], cover_node)
    cover_node["spans"] = remaining_spans
    new_s_index = len(nodes)
    nodes.append({"spans": [[apos_index, apos_index + 1]], "type": "FactualPredicator"})

    new_edges: list[dict[str, Any]] = []
    subject_edges: list[dict[str, Any]] = []
    for edge in edges:
        if edge["start"] == cover_index and edge["label"] == "pred.arg.1":
            subject_edges.append(deepcopy(edge))
            continue
        copied = deepcopy(edge)
        if copied["end"] == cover_index:
            copied["end"] = new_s_index
        new_edges.append(copied)

    assert subject_edges, (sample["sentence"], old_surface)
    for edge in subject_edges:
        edge["start"] = new_s_index
        new_edges.append(edge)
    new_edges.append({"start": new_s_index, "end": cover_index, "label": "pred.arg.2"})
    sample["lingua"]["graph"]["edges"] = new_edges
    validate_graph(sample)

    return f"{occurrence['full_phrase']}|{old_surface}->{node_text(words, cover_node)} + 's"


def dot_quote(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def write_dot(sample: dict[str, Any], output_path: Path) -> None:
    words = sample["lingua"]["words"]
    nodes = sample["lingua"]["graph"]["nodes"]
    lines = [
        "digraph G {",
        "  graph [rankdir=TB, labelloc=t, fontsize=18];",
        '  node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#64748b", fontname="Helvetica"];',
        '  edge [fontname="Helvetica", color="#334155", fontsize=10];',
        f"  label={dot_quote(sample['sentence'])};",
    ]
    for index, node in enumerate(nodes):
        label = f"{index}: {node_text(words, node)}\\n{node.get('type')}\\n{node.get('spans')}"
        lines.append(f"  n{index} [label={dot_quote(label)}];")
    for edge in sample["lingua"]["graph"]["edges"]:
        lines.append(f"  n{edge['start']} -> n{edge['end']} [label={dot_quote(edge['label'])}];")
    lines.append("}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dot(dot_path: Path, svg_path: Path) -> None:
    import subprocess

    subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)


def process_split(split: str, apply: bool, report_dir: Path) -> list[dict[str, Any]]:
    path = DATA_DIR / f"{split}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    changes: list[dict[str, Any]] = []
    new_lines = list(lines)
    for line_number, line in enumerate(lines, 1):
        sample = json.loads(new_lines[line_number - 1])
        occurrences = find_target_occurrences(sample)
        if not occurrences:
            continue
        before = deepcopy(sample)
        sample_changes: list[str] = []
        for occurrence in occurrences:
            sample_changes.append(repair_occurrence(sample, occurrence))
        after = sample
        new_lines[line_number - 1] = json.dumps(after, ensure_ascii=False, separators=(",", ":"))
        stem = f"{split}_{line_number:05d}"
        (report_dir / f"{stem}_before.json").write_text(
            json.dumps(before, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (report_dir / f"{stem}_after.json").write_text(
            json.dumps(after, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        before_dot = report_dir / f"{stem}_before.dot"
        after_dot = report_dir / f"{stem}_after.dot"
        write_dot(before, before_dot)
        write_dot(after, after_dot)
        run_dot(before_dot, report_dir / f"{stem}_before.svg")
        run_dot(after_dot, report_dir / f"{stem}_after.svg")
        changes.append(
            {
                "split": split,
                "line": line_number,
                "sentence": before["sentence"],
                "changes": sample_changes,
                "before": compact_graph(before),
                "after": compact_graph(after),
            }
        )
    if apply and changes:
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    all_changes: list[dict[str, Any]] = []
    for split in ["train", "dev", "test"]:
        all_changes.extend(process_split(split, args.apply, args.report_dir))

    with (args.report_dir / "changes.jsonl").open("w", encoding="utf-8") as out:
        for change in all_changes:
            out.write(json.dumps(change, ensure_ascii=False) + "\n")

    by_phrase = Counter()
    by_split = Counter()
    for change in all_changes:
        by_split[change["split"]] += 1
        for item in change["changes"]:
            by_phrase[item.split("|", 1)[0]] += 1

    summary = {
        "apply": args.apply,
        "changed_samples": len(all_changes),
        "changed_items": sum(len(change["changes"]) for change in all_changes),
        "by_split": dict(by_split),
        "by_phrase": dict(by_phrase),
    }
    (args.report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# larger apostrophe+s predicate repair",
        "",
        f"- apply: {args.apply}",
        f"- changed samples: {summary['changed_samples']}",
        f"- changed items: {summary['changed_items']}",
        "",
        "## By Phrase",
    ]
    for phrase, count in by_phrase.most_common():
        lines.append(f"- {phrase}: {count}")
    (args.report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
