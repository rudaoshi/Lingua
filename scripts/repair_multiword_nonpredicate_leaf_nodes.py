#!/usr/bin/env python3
"""Split reviewed non-predicate multiword leaf nodes.

This pass targets compact Lingua JSONL nodes that:

- are leaves in the current graph (no outgoing edges),
- cover multiple tokens whose reconstructed text contains whitespace, and
- are not FactualPredicator multiword predicate leaves.

The goal is to remove compressed multiword constants/fragments while preserving
the compact graph convention that the replacement root's descendants recover
the old node surface.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repair_be_copula_rules import compact_graph, read_jsonl, write_jsonl  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


DET_WORDS = {
    "a",
    "an",
    "another",
    "each",
    "either",
    "every",
    "few",
    "many",
    "my",
    "no",
    "our",
    "several",
    "some",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "two",
    "us",
}

ORDINAL_WORDS = {
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
}

WH_WORDS = {"what", "which", "who", "whom", "whose", "where", "when", "why", "how"}

WH_MODIFIERS = {"exactly", "else", "not"}

EXPRESSION_OPERATOR_TOKENS = {
    "-",
    "--",
    "/",
    "\\",
    "_",
    ".",
    ",",
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
HAS_SPACE_RE = re.compile(r"\s")
ALNUM_RE = re.compile(r"[A-Za-z0-9]", re.UNICODE)


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def node_token_indexes(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if isinstance(spans, str):
        return []
    indexes: list[int] = []
    for span in spans:
        if (
            isinstance(span, list)
            and len(span) == 2
            and isinstance(span[0], int)
            and isinstance(span[1], int)
        ):
            indexes.extend(range(span[0], span[1] + 1))
    return sorted(set(indexes))


def spans_for_tokens(tokens: list[int]) -> list[list[int]]:
    tokens = sorted(set(tokens))
    if not tokens:
        return []
    spans: list[list[int]] = []
    start = prev = tokens[0]
    for token in tokens[1:]:
        if token == prev + 1:
            prev = token
            continue
        spans.append([start, prev])
        start = prev = token
    spans.append([start, prev])
    return spans


def spans_for_token(token: int) -> list[list[int]]:
    return [[int(token), int(token)]]


def token_text(words: list[str], token: int) -> str:
    return words[token].strip() if 0 <= token < len(words) else ""


def text_for_tokens(sample: dict[str, Any], tokens: list[int]) -> str:
    words = sample["lingua"]["words"]
    return " ".join(token_text(words, token) for token in sorted(set(tokens)) if token_text(words, token))


def own_text(sample: dict[str, Any], node_index: int) -> str:
    return text_for_tokens(sample, node_token_indexes(graph(sample)["nodes"][node_index]))


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return int(edge["start"]), int(edge["end"]), str(edge["label"])


def add_edge_once(edges: list[dict[str, Any]], start: int, end: int, label: str) -> None:
    if start == end:
        return
    new_edge = {"start": int(start), "end": int(end), "label": label}
    if edge_key(new_edge) in {edge_key(edge) for edge in edges}:
        return
    edges.append(new_edge)


def is_leaf_node(outgoing: dict[int, list[dict[str, Any]]], node_index: int) -> bool:
    return not outgoing.get(node_index)


def has_multiword_surface(sample: dict[str, Any], node_index: int) -> bool:
    node = graph(sample)["nodes"][node_index]
    tokens = node_token_indexes(node)
    if len(tokens) < 2:
        return False
    compact = "".join(sample["lingua"]["words"][token] for token in tokens).strip()
    return bool(HAS_SPACE_RE.search(compact))


def normalize_surface(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def token_constant_type(token: str, fallback: str = "NominalConstant") -> str:
    low = token.lower()
    if PUNCT_ONLY_RE.fullmatch(token):
        return "PunctuationalConstant"
    if low in DET_WORDS:
        return "DeterminerConstant"
    if low in ORDINAL_WORDS:
        return "ModificationalConstant"
    return fallback


def make_token_nodes(
    words: list[str],
    tokens: list[int],
    fallback: str,
    *,
    operator_tokens: set[str] | None = None,
    wh_as_general: bool = False,
) -> tuple[list[dict[str, Any]], dict[int, int], dict[int, str]]:
    nodes: list[dict[str, Any]] = []
    pos_to_local: dict[int, int] = {}
    local_types: dict[int, str] = {}
    operator_tokens = operator_tokens or set()
    for pos in tokens:
        text = token_text(words, pos)
        low = text.lower()
        if text in operator_tokens:
            node_type = "ExpressionFunctor"
        elif wh_as_general and low in WH_WORDS:
            node_type = "GeneralFunctor"
        else:
            node_type = token_constant_type(text, fallback)
        local_index = len(nodes)
        nodes.append({"spans": spans_for_token(pos), "type": node_type})
        pos_to_local[pos] = local_index
        local_types[local_index] = node_type
    return nodes, pos_to_local, local_types


def right_head_modifier_structure(local_ids: list[int]) -> tuple[int, list[dict[str, Any]]]:
    if not local_ids:
        raise ValueError("empty local_ids")
    if len(local_ids) == 1:
        return local_ids[0], []
    edges: list[dict[str, Any]] = []
    for left, right in zip(local_ids, local_ids[1:]):
        add_edge_once(edges, left, right, "as:modification")
    return local_ids[0], edges


def expression_aux_structure(local_ids: list[int], nodes: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    expr = len(nodes)
    nodes.append({"spans": "Expression", "type": "ExpressionFunctor"})
    edges: list[dict[str, Any]] = []
    for child in local_ids:
        add_edge_once(edges, expr, child, "func.arg")
    return expr, edges


def operator_expression_structure(
    local_ids: list[int],
    local_types: dict[int, str],
    nodes: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    operators = [idx for idx in local_ids if local_types[idx] == "ExpressionFunctor"]
    if not operators:
        return expression_aux_structure(local_ids, nodes)

    edges: list[dict[str, Any]] = []
    root = operators[-1]
    for child in local_ids:
        if child != root:
            add_edge_once(edges, root, child, "func.arg")
    return root, edges


def looks_like_expression(words: list[str], tokens: list[int], node_type: str) -> bool:
    texts = [token_text(words, token) for token in tokens]
    if node_type in {"PunctuationalConstant", "OtherConstant", "InterjectionConstant", "ConjunctionalFunctor"}:
        return True
    if any(text in EXPRESSION_OPERATOR_TOKENS for text in texts):
        return True
    if any(PUNCT_ONLY_RE.fullmatch(text) for text in texts) and any(ALNUM_RE.search(text) for text in texts):
        return True
    if any(any(ch.isdigit() for ch in text) for text in texts):
        return True
    if node_type == "NominalConstant":
        nonempty = [text for text in texts if text]
        if nonempty and all(
            text[:1].isupper() or text.isupper() or text.lower() in {"de", "of", "la", "le", "van", "von", "da"}
            for text in nonempty
        ):
            return True
    return False


def build_nominal_replacement(
    sample: dict[str, Any],
    tokens: list[int],
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]], str]:
    words = sample["lingua"]["words"]
    texts = [token_text(words, token) for token in tokens]
    operator_tokens = {text for text in texts if text in EXPRESSION_OPERATOR_TOKENS}
    nodes, _pos_to_local, local_types = make_token_nodes(
        words,
        tokens,
        "NominalConstant",
        operator_tokens=operator_tokens,
    )
    local_ids = list(range(len(tokens)))
    if operator_tokens:
        leading_modifiers: list[int] = []
        for local_id, text in enumerate(texts):
            low = text.lower()
            if low in DET_WORDS or low in ORDINAL_WORDS:
                leading_modifiers.append(local_id)
                continue
            break
        if leading_modifiers and len(leading_modifiers) < len(local_ids) - 1:
            tail_ids = [idx for idx in local_ids if idx not in set(leading_modifiers)]
            tail_root, edges = operator_expression_structure(tail_ids, local_types, nodes)
            root = leading_modifiers[0]
            chain = leading_modifiers + [tail_root]
            for left, right in zip(chain, chain[1:]):
                add_edge_once(edges, left, right, "as:modification")
            return nodes, root, edges, "operator_expression_with_leading_modifiers"
        root, edges = operator_expression_structure(local_ids, local_types, nodes)
        return nodes, root, edges, "operator_expression"
    if looks_like_expression(words, tokens, "NominalConstant"):
        root, edges = expression_aux_structure(local_ids, nodes)
        return nodes, root, edges, "aux_expression"
    root, edges = right_head_modifier_structure(local_ids)
    return nodes, root, edges, "nominal_modifier_chain"


def build_modifier_replacement(
    sample: dict[str, Any],
    tokens: list[int],
    original_type: str,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]], str]:
    words = sample["lingua"]["words"]
    fallback = (
        "ModificationalConstant"
        if original_type in {"ModificationalConstant", "ModificationalFunctor"}
        else "NominalConstant"
    )
    nodes, _pos_to_local, local_types = make_token_nodes(words, tokens, fallback)
    texts = [token_text(words, token) for token in tokens]
    lower = [text.lower() for text in texts]

    if original_type == "ModificationalFunctor":
        for local_index, text in enumerate(lower):
            if text in {"as", "than", "to", "for", "of"}:
                nodes[local_index]["type"] = "ModificationalFunctor"
        local_types = {idx: node["type"] for idx, node in enumerate(nodes)}

    local_ids = list(range(len(tokens)))
    if looks_like_expression(words, tokens, original_type):
        root, edges = expression_aux_structure(local_ids, nodes)
        return nodes, root, edges, "non_nominal_aux_expression"
    root, edges = right_head_modifier_structure(local_ids)
    return nodes, root, edges, "modifier_chain"


def build_general_functor_replacement(
    sample: dict[str, Any],
    tokens: list[int],
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]], str]:
    words = sample["lingua"]["words"]
    texts = [token_text(words, token) for token in tokens]
    lower = [text.lower() for text in texts]
    operator_tokens = {text for text in texts if text in EXPRESSION_OPERATOR_TOKENS}
    nodes, _pos_to_local, local_types = make_token_nodes(
        words,
        tokens,
        "NominalConstant",
        operator_tokens=operator_tokens,
        wh_as_general=True,
    )
    local_ids = list(range(len(tokens)))
    wh_ids = [idx for idx, text in enumerate(lower) if text in WH_WORDS]

    if operator_tokens or len(wh_ids) != 1:
        root, edges = operator_expression_structure(local_ids, local_types, nodes)
        return nodes, root, edges, "general_expression"

    wh = wh_ids[0]
    edges: list[dict[str, Any]] = []
    after_nominals = [
        idx
        for idx in local_ids
        if idx > wh and lower[idx] not in WH_MODIFIERS and nodes[idx]["type"] != "PunctuationalConstant"
    ]
    if after_nominals:
        nominal_root, nominal_edges = right_head_modifier_structure(after_nominals)
        edges.extend(nominal_edges)
        add_edge_once(edges, wh, nominal_root, "func.arg")

    root = wh
    modifiers = [
        idx
        for idx in local_ids
        if idx != wh and (lower[idx] in WH_MODIFIERS or idx < wh)
    ]
    for modifier in reversed(modifiers):
        if nodes[modifier]["type"] == "DeterminerConstant":
            nodes[modifier]["type"] = "ModificationalConstant"
        add_edge_once(edges, modifier, root, "as:modification")
        root = modifier

    return nodes, root, edges, "general_functor_phrase"


def build_replacement(
    sample: dict[str, Any],
    node_index: int,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]], str]:
    node = graph(sample)["nodes"][node_index]
    node_type = node.get("type")
    tokens = node_token_indexes(node)
    if node_type == "GeneralFunctor":
        return build_general_functor_replacement(sample, tokens)
    if node_type == "NominalConstant":
        return build_nominal_replacement(sample, tokens)
    return build_modifier_replacement(sample, tokens, str(node_type))


def target_leaf_indexes(sample: dict[str, Any]) -> list[int]:
    outgoing: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph(sample).get("edges", []):
        outgoing[int(edge["start"])].append(edge)
    targets: list[int] = []
    for node_index, node in enumerate(graph(sample)["nodes"]):
        if node.get("type") == "FactualPredicator":
            continue
        if not is_leaf_node(outgoing, node_index):
            continue
        if not has_multiword_surface(sample, node_index):
            continue
        targets.append(node_index)
    return targets


def repair_sample(sample: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    targets = set(target_leaf_indexes(sample))
    if not targets:
        return sample, []

    old_nodes = graph(sample)["nodes"]
    old_edges = graph(sample)["edges"]
    new_nodes: list[dict[str, Any]] = []
    old_to_new: dict[int, int] = {}
    internal_edges: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []

    for old_index, old_node in enumerate(old_nodes):
        if old_index not in targets:
            old_to_new[old_index] = len(new_nodes)
            new_nodes.append(dict(old_node))
            continue

        base = len(new_nodes)
        replacement_nodes, local_root, replacement_edges, action = build_replacement(sample, old_index)
        for replacement_node in replacement_nodes:
            new_nodes.append(replacement_node)
        root = base + local_root
        old_to_new[old_index] = root
        for edge in replacement_edges:
            internal_edges.append(
                {
                    "start": base + int(edge["start"]),
                    "end": base + int(edge["end"]),
                    "label": edge["label"],
                }
            )
        repairs.append(
            {
                "old_node": old_index,
                "old_type": old_node.get("type"),
                "text": own_text(sample, old_index),
                "spans": old_node.get("spans"),
                "tokens": node_token_indexes(old_node),
                "action": action,
                "replacement_root": root,
                "new_nodes": list(range(base, base + len(replacement_nodes))),
                "new_node_types": [node["type"] for node in replacement_nodes],
                "internal_edges": internal_edges[-len(replacement_edges) :] if replacement_edges else [],
            }
        )

    new_edges: list[dict[str, Any]] = []
    for edge in old_edges:
        start = old_to_new[int(edge["start"])]
        end = old_to_new[int(edge["end"])]
        add_edge_once(new_edges, start, end, edge["label"])
    for edge in internal_edges:
        add_edge_once(new_edges, edge["start"], edge["end"], edge["label"])

    repaired = deepcopy(sample)
    repaired["lingua"] = deepcopy(sample["lingua"])
    repaired["lingua"]["graph"] = {"nodes": new_nodes, "edges": new_edges}
    return repaired, repairs


def validate_graph_shape(sample: dict[str, Any]) -> None:
    nodes = graph(sample)["nodes"]
    adjacency = [[] for _ in nodes]
    for edge in graph(sample)["edges"]:
        start = int(edge["start"])
        end = int(edge["end"])
        if not (0 <= start < len(nodes) and 0 <= end < len(nodes)):
            raise ValueError(f"dangling edge: {edge}")
        adjacency[start].append(end)

    color = [0] * len(nodes)

    def visit(node: int) -> None:
        if color[node] == 1:
            raise ValueError("cycle detected")
        if color[node] == 2:
            return
        color[node] = 1
        for child in adjacency[node]:
            visit(child)
        color[node] = 2

    for node in range(len(nodes)):
        visit(node)


def residual_nonpredicate_multiword_leaf_count(sample: dict[str, Any]) -> int:
    return len(target_leaf_indexes(sample))


def run(apply: bool, data_dir: Path, report_dir: Path) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    case_dir = report_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    changed_records: list[dict[str, Any]] = []
    skipped_predicates = Counter()
    residual_counts = Counter()
    by_action = Counter()
    by_old_type = Counter()

    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        new_rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(rows, 1):
            for node_index, node in enumerate(graph(row)["nodes"]):
                if node.get("type") != "FactualPredicator":
                    continue
                outgoing = defaultdict(list)
                for edge in graph(row).get("edges", []):
                    outgoing[int(edge["start"])].append(edge)
                if is_leaf_node(outgoing, node_index) and has_multiword_surface(row, node_index):
                    skipped_predicates[split] += 1

            before = deepcopy(row)
            after, repairs = repair_sample(deepcopy(row))
            validate_graph_shape(after)
            residual_counts[split] += residual_nonpredicate_multiword_leaf_count(after)
            if not repairs or before == after:
                new_rows.append(row)
                continue

            for repair in repairs:
                by_action[repair["action"]] += 1
                by_old_type[repair["old_type"]] += 1
            record = {
                "split": split,
                "line": line_number,
                "sample_id": line_number - 1,
                "sentence": row.get("sentence"),
                "repairs": repairs,
                "before": compact_graph(before),
                "after": compact_graph(after),
            }
            changed_records.append(record)
            (case_dir / f"{split}_{line_number:05d}.txt").write_text(
                f"{split}:{line_number - 1}\n{row.get('sentence')}\n\n"
                f"REPAIRS: {json.dumps(repairs, ensure_ascii=False)}\n\n"
                f"BEFORE\n{record['before']}\n\nAFTER\n{record['after']}\n",
                encoding="utf-8",
            )
            new_rows.append(after)
        if apply:
            write_jsonl(path, new_rows)

    summary = {
        "apply": apply,
        "changed_samples": len(changed_records),
        "changed_items": sum(len(record["repairs"]) for record in changed_records),
        "by_action": dict(by_action.most_common()),
        "by_old_type": dict(by_old_type.most_common()),
        "skipped_factual_predicate_leaf_nodes": dict(skipped_predicates),
        "residual_nonpredicate_multiword_leaf_nodes": dict(residual_counts),
        "report_dir": str(report_dir),
    }
    (report_dir / "changes.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in changed_records)
        + ("\n" if changed_records else ""),
        encoding="utf-8",
    )
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Multiword Non-Predicate Leaf Repair",
        "",
        f"- apply: `{apply}`",
        f"- changed samples: `{summary['changed_samples']}`",
        f"- changed items: `{summary['changed_items']}`",
        f"- skipped FactualPredicator leaf nodes: `{sum(skipped_predicates.values())}`",
        f"- residual non-predicate multiword leaf nodes: `{sum(residual_counts.values())}`",
        "",
        "## By Old Type",
    ]
    for old_type, count in by_old_type.most_common():
        lines.append(f"- `{old_type}`: {count}")
    lines.append("")
    lines.append("## By Action")
    for action, count in by_action.most_common():
        lines.append(f"- `{action}`: {count}")
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORTS / "multiword_nonpredicate_leaf_repair",
    )
    args = parser.parse_args()
    return run(args.apply, args.data_dir, args.report_dir)


if __name__ == "__main__":
    raise SystemExit(main())
