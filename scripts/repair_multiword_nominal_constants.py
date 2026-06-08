#!/usr/bin/env python3
"""Split multi-word NominalConstant nodes in compact Lingua JSONL data.

The repair is intentionally local: external edges that used to touch the
multi-word node are rewired to the inferred internal head/root, while the
covered word pieces are represented as single-word nodes.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DET_WORDS = {
    "a",
    "an",
    "another",
    "each",
    "either",
    "every",
    "no",
    "some",
    "that",
    "the",
    "these",
    "this",
    "those",
}

FORMAT_SEPARATORS = {
    "-",
    "--",
    "/",
    "\\",
    "_",
    ".",
    ":",
    ";",
    "@",
    "#",
    "+",
    "&",
    "|",
    "'",
    "`",
    "\u2018",
    "\u2019",
}

PUNCT_ONLY_RE = re.compile(r"^[^\w\s]+$", re.UNICODE)
ALNUM_RE = re.compile(r"[A-Za-z0-9]", re.UNICODE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def positions_from_spans(spans: Any) -> list[int]:
    positions: list[int] = []
    if not isinstance(spans, list):
        return positions
    for span in spans:
        if (
            isinstance(span, list)
            and len(span) == 2
            and isinstance(span[0], int)
            and isinstance(span[1], int)
        ):
            positions.extend(range(span[0], span[1] + 1))
    return sorted(set(positions))


def spans_for_pos(pos: int) -> list[list[int]]:
    return [[int(pos), int(pos)]]


def token_text(words: list[str], pos: int) -> str:
    if 0 <= pos < len(words):
        return words[pos].strip()
    return ""


def surface_groups(words: list[str], positions: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for ordinal, pos in enumerate(positions):
        raw = words[pos] if 0 <= pos < len(words) else ""
        starts_new_surface = ordinal == 0 or raw[:1].isspace()
        if starts_new_surface or not groups:
            groups.append([])
        groups[-1].append(pos)
    return groups


def surface_group_text(words: list[str], group: list[int]) -> str:
    return "".join(token_text(words, pos) for pos in group)


def is_mixed_alnum_surface_group(words: list[str], group: list[int]) -> bool:
    text = surface_group_text(words, group)
    return len(group) > 1 and any(ch.isalpha() for ch in text) and any(ch.isdigit() for ch in text)


def node_text(words: list[str], node: dict[str, Any]) -> str:
    parts = []
    for pos in positions_from_spans(node.get("spans")):
        if 0 <= pos < len(words):
            parts.append(words[pos])
    return "".join(parts).strip()


def classify_pattern(words: list[str], positions: list[int]) -> str:
    tokens = [token_text(words, pos) for pos in positions]
    tokens_nonempty = [tok for tok in tokens if tok]
    if not tokens_nonempty:
        return "empty"

    stripped = "".join(tokens_nonempty)
    lower_tokens = [tok.lower() for tok in tokens_nonempty]
    has_punct = any(PUNCT_ONLY_RE.fullmatch(tok) for tok in tokens_nonempty)
    has_separator = any(tok in FORMAT_SEPARATORS for tok in tokens_nonempty)
    has_alnum = any(ALNUM_RE.search(tok) for tok in tokens_nonempty)
    has_digit = any(any(ch.isdigit() for ch in tok) for tok in tokens_nonempty)

    if len(tokens_nonempty) >= 2 and tokens_nonempty[-1] == ".":
        return "abbreviation_or_number_period"
    if stripped.endswith(".") and len(tokens_nonempty) == 1:
        return "abbreviation_or_number_period"
    if has_separator and has_alnum:
        return "format_expression"
    if any("/" in tok or "\\" in tok or "@" in tok or "_" in tok or "'" in tok for tok in tokens_nonempty):
        return "format_expression"
    if has_punct and has_alnum:
        return "format_expression"
    if lower_tokens[0] in DET_WORDS:
        return "determiner_head"
    if has_digit and has_alnum and any(
        is_mixed_alnum_surface_group(words, group) for group in surface_groups(words, positions)
    ):
        return "number_unit_or_code"
    if has_punct:
        return "punctuated_nominal"
    if all(tok[:1].isupper() or tok.isupper() for tok in tokens_nonempty):
        return "proper_or_title_name"
    return "nominal_compound"


def infer_leaf_type(token: str, pattern: str, ordinal: int) -> str:
    low = token.lower()
    if PUNCT_ONLY_RE.fullmatch(token):
        if pattern == "format_expression" and token in FORMAT_SEPARATORS:
            return "ExpressionFunctor"
        return "PunctuationalConstant"
    if pattern == "determiner_head" and ordinal == 0 and low in DET_WORDS:
        return "DeterminerConstant"
    return "NominalConstant"


def add_edge_once(edges: list[dict[str, Any]], start: int, end: int, label: str) -> None:
    if start == end:
        return
    if any(e["start"] == start and e["end"] == end and e["label"] == label for e in edges):
        return
    edges.append({"start": int(start), "end": int(end), "label": label})


def build_format_expression_edges(
    token_ids: list[int],
    token_types: dict[int, str],
) -> tuple[int, list[dict[str, Any]]]:
    """Return expression root and internal edges for simple separator expressions."""
    edges: list[dict[str, Any]] = []
    separators = [idx for idx in token_ids if token_types[idx] == "ExpressionFunctor"]
    if not separators:
        return build_right_head_modifier_edges(token_ids)

    current_root: int | None = None
    consumed: set[int] = set()
    for sep in separators:
        left_candidates = [idx for idx in token_ids if idx < sep and idx not in consumed]
        right_candidates = [idx for idx in token_ids if idx > sep and idx not in consumed]
        left = current_root if current_root is not None else (left_candidates[-1] if left_candidates else None)
        right = right_candidates[0] if right_candidates else None
        if left is not None:
            add_edge_once(edges, sep, left, "func.arg")
            consumed.add(left)
        if right is not None:
            add_edge_once(edges, sep, right, "func.arg")
            consumed.add(right)
        current_root = sep
        consumed.add(sep)

    if current_root is None:
        return build_right_head_modifier_edges(token_ids)

    for idx in token_ids:
        if idx == current_root or idx in consumed:
            continue
        add_edge_once(edges, current_root, idx, "func.arg")
    return current_root, edges


def build_right_head_modifier_edges(token_ids: list[int]) -> tuple[int, list[dict[str, Any]]]:
    if not token_ids:
        raise ValueError("empty token_ids")
    if len(token_ids) == 1:
        return token_ids[0], []
    edges: list[dict[str, Any]] = []
    for left, right in zip(token_ids, token_ids[1:]):
        add_edge_once(edges, left, right, "as:modification")
    return token_ids[0], edges


def build_number_code_expression_edges(
    words: list[str],
    positions: list[int],
    new_node_ids: list[int],
    new_nodes: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]], list[int]]:
    pos_to_node = dict(zip(positions, new_node_ids))
    group_roots: list[int] = []
    expression_roots: list[int] = []
    edges: list[dict[str, Any]] = []

    for group in surface_groups(words, positions):
        group_node_ids = [pos_to_node[pos] for pos in group]
        if is_mixed_alnum_surface_group(words, group):
            expr_id = len(new_nodes)
            new_nodes.append({"spans": "Expression", "type": "ExpressionFunctor"})
            expression_roots.append(expr_id)
            for child_id in group_node_ids:
                add_edge_once(edges, expr_id, child_id, "func.arg")
            group_roots.append(expr_id)
        else:
            group_root, group_edges = build_right_head_modifier_edges(group_node_ids)
            edges.extend(group_edges)
            group_roots.append(group_root)

    if not group_roots:
        raise ValueError("empty number/code expression")
    if len(group_roots) == 1:
        return group_roots[0], edges, expression_roots

    root, phrase_edges = build_right_head_modifier_edges(group_roots)
    edges.extend(phrase_edges)
    return root, edges, expression_roots


def build_internal_structure(
    words: list[str],
    positions: list[int],
    new_node_ids: list[int],
    pattern: str,
    token_types: dict[int, str],
) -> tuple[int, list[dict[str, Any]]]:
    if pattern == "abbreviation_or_number_period":
        content = [idx for idx in new_node_ids if token_types[idx] != "PunctuationalConstant"]
        punct = [idx for idx in new_node_ids if token_types[idx] == "PunctuationalConstant"]
        root = content[0] if content else new_node_ids[0]
        edges: list[dict[str, Any]] = []
        if len(content) > 1:
            sub_root, sub_edges = build_right_head_modifier_edges(content)
            root = sub_root
            edges.extend(sub_edges)
        if punct:
            punct_root = punct[0]
            add_edge_once(edges, punct_root, root, "as:punctuation")
            for punct_id in punct[1:]:
                add_edge_once(edges, punct_root, punct_id, "as:punctuation")
            return punct_root, edges
        return root, edges

    if pattern == "format_expression":
        return build_format_expression_edges(new_node_ids, token_types)

    content = [idx for idx in new_node_ids if token_types[idx] != "PunctuationalConstant"]
    punct = [idx for idx in new_node_ids if token_types[idx] == "PunctuationalConstant"]
    head, edges = build_right_head_modifier_edges(content or new_node_ids)
    if punct and content:
        punct_root = punct[0]
        add_edge_once(edges, punct_root, head, "as:punctuation")
        for punct_id in punct[1:]:
            add_edge_once(edges, punct_root, punct_id, "as:punctuation")
        return punct_root, edges
    return head, edges


def repair_sample(sample: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    words = sample["lingua"]["words"]
    graph = sample["lingua"]["graph"]
    nodes = graph["nodes"]
    old_edges = graph["edges"]

    new_nodes: list[dict[str, Any]] = []
    old_to_rep: dict[int, int] = {}
    internal_edges: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []

    for old_idx, node in enumerate(nodes):
        positions = positions_from_spans(node.get("spans"))
        if node.get("type") != "NominalConstant" or len(positions) < 2:
            old_to_rep[old_idx] = len(new_nodes)
            new_nodes.append(dict(node))
            continue

        pattern = classify_pattern(words, positions)
        new_ids: list[int] = []
        token_types: dict[int, str] = {}
        for ordinal, pos in enumerate(positions):
            tok = token_text(words, pos)
            node_type = infer_leaf_type(tok, pattern, ordinal)
            new_id = len(new_nodes)
            new_nodes.append({"spans": spans_for_pos(pos), "type": node_type})
            new_ids.append(new_id)
            token_types[new_id] = node_type

        if pattern == "number_unit_or_code":
            rep, added_edges, report_extra_ids = build_number_code_expression_edges(
                words,
                positions,
                new_ids,
                new_nodes,
            )
        else:
            rep, added_edges = build_internal_structure(words, positions, new_ids, pattern, token_types)
            report_extra_ids = []
        report_new_ids = new_ids + report_extra_ids
        old_to_rep[old_idx] = rep
        internal_edges.extend(added_edges)
        repairs.append(
            {
                "old_node": old_idx,
                "text": node_text(words, node),
                "spans": node.get("spans"),
                "word_count": len(positions),
                "pattern": pattern,
                "replacement_root": rep,
                "new_nodes": report_new_ids,
                "new_node_types": [new_nodes[idx]["type"] for idx in report_new_ids],
                "internal_edges": added_edges,
            }
        )

    new_edges: list[dict[str, Any]] = []
    for edge in old_edges:
        start = old_to_rep[int(edge["start"])]
        end = old_to_rep[int(edge["end"])]
        if start == end:
            continue
        new_edge = dict(edge)
        new_edge["start"] = start
        new_edge["end"] = end
        if not any(
            e["start"] == new_edge["start"]
            and e["end"] == new_edge["end"]
            and e["label"] == new_edge["label"]
            for e in new_edges
        ):
            new_edges.append(new_edge)
    for edge in internal_edges:
        add_edge_once(new_edges, edge["start"], edge["end"], edge["label"])

    repaired = dict(sample)
    repaired_lingua = dict(sample["lingua"])
    repaired_graph = {"nodes": new_nodes, "edges": new_edges}
    repaired_lingua["graph"] = repaired_graph
    repaired["lingua"] = repaired_lingua
    return repaired, repairs


def count_residual_multiword_nominals(sample: dict[str, Any]) -> int:
    return sum(
        1
        for node in sample["lingua"]["graph"]["nodes"]
        if node.get("type") == "NominalConstant"
        and len(positions_from_spans(node.get("spans"))) >= 2
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir) if args.report_dir else output_dir / "reports" / "multiword_nominal_split"
    data_out = output_dir / "data"
    data_out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "splits": {},
    }
    all_patterns = Counter()

    for split in args.splits:
        rows = read_jsonl(input_dir / f"{split}.jsonl")
        repaired_rows: list[dict[str, Any]] = []
        split_repairs: list[dict[str, Any]] = []
        counters = Counter()
        patterns = Counter()
        old_node_count = 0
        new_node_count = 0
        old_edge_count = 0
        new_edge_count = 0

        for idx, sample in enumerate(rows):
            old_node_count += len(sample["lingua"]["graph"]["nodes"])
            old_edge_count += len(sample["lingua"]["graph"]["edges"])
            repaired, repairs = repair_sample(sample)
            repaired_rows.append(repaired)
            new_node_count += len(repaired["lingua"]["graph"]["nodes"])
            new_edge_count += len(repaired["lingua"]["graph"]["edges"])
            if repairs:
                counters["examples_changed"] += 1
            counters["multiword_nominal_nodes_repaired"] += len(repairs)
            residual = count_residual_multiword_nominals(repaired)
            counters["residual_multiword_nominals"] += residual
            for repair in repairs:
                patterns[repair["pattern"]] += 1
                split_repairs.append(
                    {
                        "split": split,
                        "idx": idx,
                        "line": idx + 1,
                        "sentence": sample.get("sentence"),
                        **repair,
                    }
                )

        write_jsonl(data_out / f"{split}.jsonl", repaired_rows)
        write_jsonl(report_dir / f"{split}_repairs.jsonl", split_repairs)
        all_patterns.update(patterns)
        summary["splits"][split] = {
            "examples": len(rows),
            "old_nodes": old_node_count,
            "new_nodes": new_node_count,
            "node_delta": new_node_count - old_node_count,
            "old_edges": old_edge_count,
            "new_edges": new_edge_count,
            "edge_delta": new_edge_count - old_edge_count,
            "counters": dict(counters),
            "patterns": dict(patterns.most_common()),
        }

    summary["pattern_totals"] = dict(all_patterns.most_common())
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_dir", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
