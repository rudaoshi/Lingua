#!/usr/bin/env python3
"""Repair reviewed expression leaf-granularity conflicts.

The targets are the 29 surfaces in the `expression_leaf_granularity` bucket
from the hierarchical surface-conflict audit.  Repairs are exact and local:
multi-token opaque names become Expression units, explicit format operators
(`.`, `-`, `&`, `,`, `/`) become ExpressionFunctor roots, and a few reviewed
shape conflicts are normalized to the structure required by docs/standard.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
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


AUX_EXPRESSION_LEAVES = {
    "united states": "opaque country name; split multi-token leaf into Expression(United, States)",
    "new york": "opaque place-name fragment; split into Expression(New, York)",
    "los angeles": "opaque place-name fragment; split into Expression(Los, Angeles)",
    "south africa": "opaque country name; split into Expression(South, Africa)",
    "world war ii": "opaque historical-event name; split into Expression(World, War, II)",
    "1940 s": "decade format; split into Expression(1940, s)",
    "achille lauro": "ship name; split into Expression(Achille, Lauro)",
}

DOT_OPERATOR_LEAVES = {
    "2 .": "list/rank marker; use dot as ExpressionFunctor root",
    "5 .": "list/rank marker; use dot as ExpressionFunctor root",
    "10 .": "list/rank marker; use dot as ExpressionFunctor root",
    "12 .": "list/rank marker; use dot as ExpressionFunctor root",
    "c .": "abbreviation/initial marker; use dot as ExpressionFunctor root",
    "s .": "abbreviation/initial marker; use dot as ExpressionFunctor root",
    "b .": "abbreviation/initial marker; use dot as ExpressionFunctor root",
    "d .": "abbreviation/initial marker; use dot as ExpressionFunctor root",
    "m .": "abbreviation/initial marker; use dot as ExpressionFunctor root",
}

HYPHEN_OPERATOR_LEAVES = {
    "one - day": "hyphenated measure/format compound; use hyphen ExpressionFunctor",
    "all - trans": "biochemical format compound; use hyphen ExpressionFunctor",
    "delta - domain": "biological name fragment; use hyphen ExpressionFunctor",
    "number - one": "hyphenated ranking format; use hyphen ExpressionFunctor",
    "one - quarter": "fraction format; use hyphen ExpressionFunctor",
    "runners - up": "lexicalized hyphenated format; use hyphen ExpressionFunctor",
    "semi - finals": "hyphenated competition-stage format; use hyphen ExpressionFunctor",
}

AMPERSAND_OPERATOR_LEAVES = {
    "b & b": "brand/name abbreviation; use ampersand ExpressionFunctor",
}

SPECIAL_DECISIONS = {
    "vs .": "match-up marker; build Expression(Qualifier, vs., opponent) instead of modifier chain",
    "14 1 / 2": "mixed fraction; group whole number and slash fraction with Expression aux",
    "2 , 300": "thousands separator; comma is the ExpressionFunctor root, no aux wrapper",
    "decor and you": "brand name; 'and' is part of an Expression, not semantic conjunction",
    "mild signs shee gave": "remove redundant Expression wrapper and keep the FactualPredicator unit",
}


def normalize_surface(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def node_token_indexes(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if isinstance(spans, str):
        return []
    indexes: list[int] = []
    for start, end in spans:
        indexes.extend(range(start, end + 1))
    return sorted(indexes)


def spans_for_token(token_index: int) -> list[list[int]]:
    return [[int(token_index), int(token_index)]]


def text_for_tokens(sample: dict[str, Any], tokens: list[int]) -> str:
    words = sample["lingua"]["words"]
    return " ".join(words[index].strip() for index in sorted(set(tokens)) if words[index].strip())


def own_text(sample: dict[str, Any], node_index: int) -> str:
    return text_for_tokens(sample, node_token_indexes(graph(sample)["nodes"][node_index]))


def descendant_nodes(sample: dict[str, Any], node_index: int) -> set[int]:
    edges = graph(sample)["edges"]
    seen: set[int] = set()

    def visit(index: int) -> None:
        if index in seen:
            return
        seen.add(index)
        for edge in edges:
            if edge["start"] == index:
                visit(edge["end"])

    visit(node_index)
    return seen


def descendant_text(sample: dict[str, Any], node_index: int) -> str:
    tokens: list[int] = []
    for index in descendant_nodes(sample, node_index):
        tokens.extend(node_token_indexes(graph(sample)["nodes"][index]))
    return text_for_tokens(sample, tokens)


def add_node(sample: dict[str, Any], spans: Any, node_type: str) -> int:
    index = len(graph(sample)["nodes"])
    graph(sample)["nodes"].append({"spans": spans, "type": node_type})
    return index


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return edge["start"], edge["end"], edge["label"]


def add_edge_once(sample: dict[str, Any], start: int, end: int, label: str) -> None:
    if start == end:
        return
    new_edge = {"start": int(start), "end": int(end), "label": label}
    if edge_key(new_edge) in {edge_key(edge) for edge in graph(sample)["edges"]}:
        return
    graph(sample)["edges"].append(new_edge)


def dedupe_edges(sample: dict[str, Any]) -> None:
    seen: set[tuple[int, int, str]] = set()
    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        key = edge_key(edge)
        if key in seen:
            continue
        seen.add(key)
        new_edges.append(edge)
    graph(sample)["edges"] = new_edges


def remove_nodes(sample: dict[str, Any], remove: set[int]) -> None:
    if not remove:
        return
    old_nodes = graph(sample)["nodes"]
    remap: dict[int, int] = {}
    new_nodes: list[dict[str, Any]] = []
    for old_index, node in enumerate(old_nodes):
        if old_index in remove:
            continue
        remap[old_index] = len(new_nodes)
        new_nodes.append(node)

    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        if edge["start"] in remove or edge["end"] in remove:
            continue
        new_edges.append(
            {
                "start": remap[edge["start"]],
                "end": remap[edge["end"]],
                "label": edge["label"],
            }
        )
    graph(sample)["nodes"] = new_nodes
    graph(sample)["edges"] = new_edges
    dedupe_edges(sample)


def split_leaf_node(
    sample: dict[str, Any],
    node_index: int,
    *,
    operator_token: str | None = None,
) -> tuple[list[int], int | None]:
    nodes = graph(sample)["nodes"]
    positions = node_token_indexes(nodes[node_index])
    if len(positions) < 2:
        raise ValueError("split_leaf_node called on a non-multi-token node")

    operator_node: int | None = None
    token_nodes: list[int] = []
    for ordinal, token_index in enumerate(positions):
        token = sample["lingua"]["words"][token_index].strip()
        node_type = "ExpressionFunctor" if operator_token is not None and token == operator_token else "NominalConstant"
        if ordinal == 0:
            nodes[node_index] = {"spans": spans_for_token(token_index), "type": node_type}
            new_index = node_index
        else:
            new_index = add_node(sample, spans_for_token(token_index), node_type)
        token_nodes.append(new_index)
        if node_type == "ExpressionFunctor" and operator_node is None:
            operator_node = new_index

    return token_nodes, operator_node


def outgoing_as_modification_heads(sample: dict[str, Any], node_index: int) -> list[int]:
    return [
        edge["end"]
        for edge in graph(sample)["edges"]
        if edge["start"] == node_index and edge["label"] == "as:modification"
    ]


def rewire_after_leaf_split(sample: dict[str, Any], old_index: int, replacement_root: int) -> int:
    """Redirect external users of a split leaf to its new recoverable root.

    If the original leaf was itself a compressed modifier source, the new
    structured body cannot remain an `as:modification` source.  In that case
    create an explicit Modification(variable=head, body=replacement_root) and
    redirect old incoming edges to that wrapper.
    """
    heads = outgoing_as_modification_heads(sample, old_index)
    if len(heads) > 1:
        raise ValueError(f"multiple outgoing as:modification heads for node {old_index}")

    incoming_replacement = replacement_root
    if heads:
        mod_index = add_node(sample, "Modification", "ModificationalFunctor")
        incoming_replacement = mod_index
        add_edge_once(sample, mod_index, heads[0], "variable")
        add_edge_once(sample, mod_index, replacement_root, "body")

    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        if edge["start"] == old_index and edge["label"] == "as:modification":
            continue
        new_edge = dict(edge)
        if new_edge["end"] == old_index:
            new_edge["end"] = incoming_replacement
        new_edges.append(new_edge)
    graph(sample)["edges"] = new_edges
    dedupe_edges(sample)
    return incoming_replacement


def convert_leaf_to_aux_expression(sample: dict[str, Any], node_index: int, surface_key: str) -> str | None:
    node = graph(sample)["nodes"][node_index]
    if len(node_token_indexes(node)) < 2 or normalize_surface(own_text(sample, node_index)) != surface_key:
        return None
    if node.get("type") != "NominalConstant":
        return None

    old_text = own_text(sample, node_index)
    token_nodes, _operator = split_leaf_node(sample, node_index)
    expr_index = add_node(sample, "Expression", "ExpressionFunctor")
    rewire_after_leaf_split(sample, node_index, expr_index)
    for child in token_nodes:
        add_edge_once(sample, expr_index, child, "func.arg")
    dedupe_edges(sample)
    return f"leaf_to_aux_expression|{surface_key}|{old_text}"


def convert_leaf_to_operator_expression(
    sample: dict[str, Any],
    node_index: int,
    surface_key: str,
    operator_token: str,
    allowed_types: set[str],
) -> str | None:
    node = graph(sample)["nodes"][node_index]
    if len(node_token_indexes(node)) < 2 or normalize_surface(own_text(sample, node_index)) != surface_key:
        return None
    if node.get("type") not in allowed_types:
        return None

    old_text = own_text(sample, node_index)
    token_nodes, operator_node = split_leaf_node(sample, node_index, operator_token=operator_token)
    if operator_node is None:
        return None
    rewire_after_leaf_split(sample, node_index, operator_node)
    for child in token_nodes:
        if child != operator_node:
            add_edge_once(sample, operator_node, child, "func.arg")
    dedupe_edges(sample)
    return f"leaf_to_operator_expression|{surface_key}|{old_text}"


def repair_vs_dot_matchup(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    for node_index, node in list(enumerate(graph(sample)["nodes"])):
        if normalize_surface(own_text(sample, node_index)) != "vs ." or len(node_token_indexes(node)) < 2:
            continue
        if node.get("type") != "ExpressionFunctor":
            continue

        incoming_as = [
            edge for edge in graph(sample)["edges"] if edge["end"] == node_index and edge["label"] == "as:modification"
        ]
        outgoing_as = [
            edge for edge in graph(sample)["edges"] if edge["start"] == node_index and edge["label"] == "as:modification"
        ]
        if len(incoming_as) != 1 or len(outgoing_as) != 1:
            continue

        left = incoming_as[0]["start"]
        right = outgoing_as[0]["end"]
        left_text = own_text(sample, left) or descendant_text(sample, left)
        right_text = descendant_text(sample, right)
        token_nodes, dot_node = split_leaf_node(sample, node_index, operator_token=".")
        if dot_node is None:
            continue
        vs_content = [child for child in token_nodes if child != dot_node]
        for child in vs_content:
            add_edge_once(sample, dot_node, child, "func.arg")

        expr_index = add_node(sample, "Expression", "ExpressionFunctor")
        new_edges: list[dict[str, Any]] = []
        for edge in graph(sample)["edges"]:
            if edge_key(edge) in {edge_key(incoming_as[0]), edge_key(outgoing_as[0])}:
                continue
            new_edge = dict(edge)
            if new_edge["end"] == left and new_edge["label"] == "pred.arg.1":
                new_edge["end"] = expr_index
            new_edges.append(new_edge)
        graph(sample)["edges"] = new_edges
        add_edge_once(sample, expr_index, left, "func.arg")
        add_edge_once(sample, expr_index, dot_node, "func.arg")
        add_edge_once(sample, expr_index, right, "func.arg")
        dedupe_edges(sample)
        changes.append(f"vs_dot_matchup_to_expression|vs .|{left_text} vs . {right_text}")
        break
    return changes


def repair_mixed_fraction(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    for slash_index, node in list(enumerate(graph(sample)["nodes"])):
        if own_text(sample, slash_index) != "/" or node.get("type") != "ExpressionFunctor":
            continue
        if normalize_surface(descendant_text(sample, slash_index)) != "14 1 / 2":
            continue
        args = [
            edge["end"]
            for edge in graph(sample)["edges"]
            if edge["start"] == slash_index and edge["label"] == "func.arg"
        ]
        whole = [index for index in args if own_text(sample, index) == "14"]
        if len(whole) != 1:
            continue

        expr_index = add_node(sample, "Expression", "ExpressionFunctor")
        new_edges: list[dict[str, Any]] = []
        for edge in graph(sample)["edges"]:
            if edge["start"] == slash_index and edge["end"] == whole[0] and edge["label"] == "func.arg":
                continue
            new_edge = dict(edge)
            if new_edge["end"] == slash_index:
                new_edge["end"] = expr_index
            new_edges.append(new_edge)
        graph(sample)["edges"] = new_edges
        add_edge_once(sample, expr_index, whole[0], "func.arg")
        add_edge_once(sample, expr_index, slash_index, "func.arg")
        dedupe_edges(sample)
        changes.append("mixed_fraction_aux_expression|14 1 / 2|14 1 / 2")
        break
    return changes


def repair_comma_thousands(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    for expr_index, node in list(enumerate(graph(sample)["nodes"])):
        if node.get("spans") != "Expression" or node.get("type") != "ExpressionFunctor":
            continue
        if normalize_surface(descendant_text(sample, expr_index)) != "2 , 300":
            continue
        children = [
            edge["end"]
            for edge in graph(sample)["edges"]
            if edge["start"] == expr_index and edge["label"] == "func.arg"
        ]
        comma = [index for index in children if own_text(sample, index) == ","]
        operands = [index for index in children if own_text(sample, index) in {"2", "300"}]
        if len(comma) != 1 or len(operands) != 2:
            continue

        comma_index = comma[0]
        graph(sample)["nodes"][comma_index]["type"] = "ExpressionFunctor"
        new_edges: list[dict[str, Any]] = []
        for edge in graph(sample)["edges"]:
            if edge["start"] == expr_index:
                continue
            new_edge = dict(edge)
            if new_edge["end"] == expr_index:
                new_edge["end"] = comma_index
            new_edges.append(new_edge)
        graph(sample)["edges"] = new_edges
        for operand in sorted(operands, key=lambda index: node_token_indexes(graph(sample)["nodes"][index])):
            add_edge_once(sample, comma_index, operand, "func.arg")
        remove_nodes(sample, {expr_index})
        changes.append("comma_thousands_operator_root|2 , 300|2 , 300")
        break
    return changes


def repair_decor_and_you(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    for node_index, node in enumerate(graph(sample)["nodes"]):
        if node.get("type") != "ConjunctionalFunctor":
            continue
        if normalize_surface(descendant_text(sample, node_index)) != "decor and you":
            continue
        node["type"] = "ExpressionFunctor"
        changes.append("brand_conjunction_to_expression|decor and you|Decor and You")
        break
    return changes


def repair_mild_signs_shared_predicate(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    for expr_index, node in list(enumerate(graph(sample)["nodes"])):
        if node.get("spans") != "Expression" or node.get("type") != "ExpressionFunctor":
            continue
        if normalize_surface(descendant_text(sample, expr_index)) != "mild signs shee gave":
            continue
        children = [
            edge["end"]
            for edge in graph(sample)["edges"]
            if edge["start"] == expr_index and edge["label"] == "func.arg"
        ]
        predicate_children = [
            index for index in children if graph(sample)["nodes"][index].get("type") == "FactualPredicator"
        ]
        if len(predicate_children) != 1:
            continue
        predicate = predicate_children[0]
        for edge in graph(sample)["edges"]:
            if edge["end"] == expr_index:
                edge["end"] = predicate
        remove_nodes(sample, {expr_index})
        changes.append("remove_redundant_expression_keep_predicate|mild signs shee gave|mild signs Shee gave")
        break
    return changes


def repair_reviewed_leaf_nodes(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    made_change = True
    while made_change:
        made_change = False
        for node_index in range(len(graph(sample)["nodes"])):
            if node_index >= len(graph(sample)["nodes"]):
                break
            surface_key = normalize_surface(own_text(sample, node_index))
            change: str | None = None
            if surface_key in AUX_EXPRESSION_LEAVES:
                change = convert_leaf_to_aux_expression(sample, node_index, surface_key)
            elif surface_key in DOT_OPERATOR_LEAVES:
                change = convert_leaf_to_operator_expression(
                    sample,
                    node_index,
                    surface_key,
                    ".",
                    {"NominalConstant", "OtherConstant"},
                )
            elif surface_key in HYPHEN_OPERATOR_LEAVES:
                change = convert_leaf_to_operator_expression(
                    sample,
                    node_index,
                    surface_key,
                    "-",
                    {"NominalConstant"},
                )
            elif surface_key in AMPERSAND_OPERATOR_LEAVES:
                change = convert_leaf_to_operator_expression(
                    sample,
                    node_index,
                    surface_key,
                    "&",
                    {"NominalConstant"},
                )
            if change is None:
                continue
            changes.append(change)
            made_change = True
            break
    return changes


def repair_sample(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    changes.extend(repair_vs_dot_matchup(sample))
    changes.extend(repair_mixed_fraction(sample))
    changes.extend(repair_comma_thousands(sample))
    changes.extend(repair_decor_and_you(sample))
    changes.extend(repair_mild_signs_shared_predicate(sample))
    changes.extend(repair_reviewed_leaf_nodes(sample))
    return changes


def write_decision_report(report_dir: Path) -> None:
    decision_rows = []
    for surface, decision in sorted(AUX_EXPRESSION_LEAVES.items()):
        decision_rows.append((surface, "aux_expression_leaf", decision))
    for surface, decision in sorted(DOT_OPERATOR_LEAVES.items()):
        decision_rows.append((surface, "dot_operator_leaf", decision))
    for surface, decision in sorted(HYPHEN_OPERATOR_LEAVES.items()):
        decision_rows.append((surface, "hyphen_operator_leaf", decision))
    for surface, decision in sorted(AMPERSAND_OPERATOR_LEAVES.items()):
        decision_rows.append((surface, "ampersand_operator_leaf", decision))
    for surface, decision in sorted(SPECIAL_DECISIONS.items()):
        decision_rows.append((surface, "special_shape", decision))

    lines = [
        "# Expression Leaf Granularity Decisions",
        "",
        "| surface | action | decision |",
        "| --- | --- | --- |",
    ]
    for surface, action, decision in decision_rows:
        lines.append(f"| `{surface}` | `{action}` | {decision} |")
    (report_dir / "decisions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(apply: bool, data_dir: Path, report_dir: Path) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    case_dir = report_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    write_decision_report(report_dir)

    changed_records: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        new_rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(rows, 1):
            before = deepcopy(row)
            after = deepcopy(row)
            changes = repair_sample(after)
            if not changes or before == after:
                new_rows.append(row)
                continue
            record = {
                "split": split,
                "line": line_number,
                "sentence": row.get("sentence"),
                "changes": changes,
                "before": compact_graph(before),
                "after": compact_graph(after),
            }
            changed_records.append(record)
            case_path = case_dir / f"{split}_{line_number:05d}.txt"
            case_path.write_text(
                f"{split}:{line_number}\n{row.get('sentence')}\n\n"
                f"CHANGES: {', '.join(changes)}\n\n"
                f"BEFORE\n{record['before']}\n\nAFTER\n{record['after']}\n",
                encoding="utf-8",
            )
            new_rows.append(after)
        if apply:
            write_jsonl(path, new_rows)

    by_change = Counter()
    for record in changed_records:
        for change in record["changes"]:
            parts = change.split("|", 2)
            surface = parts[1] if len(parts) == 3 else change
            by_change[surface] += 1
    summary = {
        "apply": apply,
        "changed_samples": len(changed_records),
        "changed_items": sum(len(record["changes"]) for record in changed_records),
        "by_surface": dict(by_change.most_common()),
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
        "# Expression Leaf Granularity Repair",
        "",
        f"- apply: `{apply}`",
        f"- changed samples: `{summary['changed_samples']}`",
        f"- changed items: `{summary['changed_items']}`",
        "",
        "## By Surface",
    ]
    for surface, count in by_change.most_common():
        lines.append(f"- `{surface}`: {count}")
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORTS / "expression_leaf_granularity_repair",
    )
    args = parser.parse_args()
    return run(args.apply, args.data_dir, args.report_dir)


if __name__ == "__main__":
    raise SystemExit(main())
