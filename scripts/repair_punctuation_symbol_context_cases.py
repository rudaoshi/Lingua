#!/usr/bin/env python3
"""Repair reviewed punctuation/symbol context annotation drift.

The cases here come from the hierarchical surface conflict audit's
`punctuation_or_symbol` cluster.  Repairs are exact split/line/token edits, not
lexical surface rewrites.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


TYPE_REPAIRS = [
    # OtherConstant pure punctuation/symbol drift.
    ("test", 755, (5,), "OtherConstant", "PunctuationalConstant", "other_comma_to_punctuation"),
    ("test", 882, (4,), "OtherConstant", "PunctuationalConstant", "other_comma_to_punctuation"),
    ("train", 938, (11,), "OtherConstant", "PunctuationalConstant", "other_comma_to_punctuation"),
    ("train", 1118, (1,), "OtherConstant", "PunctuationalConstant", "other_comma_to_punctuation"),
    ("train", 7332, (5,), "OtherConstant", "PunctuationalConstant", "other_comma_to_punctuation"),
    ("train", 7527, (2,), "OtherConstant", "PunctuationalConstant", "other_comma_to_punctuation"),
    ("train", 7527, (18,), "OtherConstant", "PunctuationalConstant", "other_comma_to_punctuation"),
    ("train", 7702, (5,), "OtherConstant", "PunctuationalConstant", "other_comma_to_punctuation"),
    ("train", 8114, (6,), "OtherConstant", "PunctuationalConstant", "other_comma_to_punctuation"),
    ("train", 8891, (3,), "OtherConstant", "PunctuationalConstant", "other_comma_to_punctuation"),
    ("train", 9637, (3,), "OtherConstant", "PunctuationalConstant", "other_comma_to_punctuation"),
    ("train", 10478, (5,), "OtherConstant", "PunctuationalConstant", "other_comma_to_punctuation"),
    ("train", 5136, (15,), "OtherConstant", "PunctuationalConstant", "other_dot_to_punctuation"),
    ("train", 9458, (14,), "OtherConstant", "PunctuationalConstant", "other_dot_to_punctuation"),
    ("train", 4730, (4,), "OtherConstant", "PunctuationalConstant", "other_quote_to_punctuation"),
    ("train", 5514, (12,), "OtherConstant", "PunctuationalConstant", "other_quote_to_punctuation"),
    ("train", 1480, (26,), "OtherConstant", "PunctuationalConstant", "other_quote_to_punctuation"),
    ("train", 354, (19,), "OtherConstant", "PunctuationalConstant", "other_quote_to_punctuation"),
    ("train", 11107, (16,), "OtherConstant", "PunctuationalConstant", "other_ellipsis_to_punctuation"),
    ("dev", 600, (4,), "OtherConstant", "PunctuationalConstant", "other_colon_to_punctuation"),
    # SymbolConstant delimiter drift.  The ratio colon in train:3745 is left
    # untouched because it behaves like a symbol/operator in context.
    ("train", 4461, (4,), "SymbolConstant", "PunctuationalConstant", "symbol_colon_delimiter_to_punctuation"),
    ("train", 4672, (10,), "SymbolConstant", "PunctuationalConstant", "symbol_colon_delimiter_to_punctuation"),
    ("train", 4951, (6,), "SymbolConstant", "PunctuationalConstant", "symbol_colon_delimiter_to_punctuation"),
    ("train", 5997, (16,), "SymbolConstant", "PunctuationalConstant", "symbol_colon_delimiter_to_punctuation"),
    # A comma that had become an expression leaf inside a list-like fragment.
    ("train", 128, (6,), "ExpressionFunctor", "PunctuationalConstant", "expression_comma_leaf_to_punctuation"),
]

PUNCT_AS_MOD_RELABELS = [
    ("train", 354, (19,), "punctuation_source_as_modification_to_as_punctuation"),
    ("train", 1480, (26,), "punctuation_source_as_modification_to_as_punctuation"),
    ("train", 4672, (10,), "punctuation_source_as_modification_to_as_punctuation"),
    ("train", 4730, (4,), "punctuation_source_as_modification_to_as_punctuation"),
    ("train", 5514, (12,), "punctuation_source_as_modification_to_as_punctuation"),
    ("train", 5997, (16,), "punctuation_source_as_modification_to_as_punctuation"),
    ("train", 8114, (6,), "punctuation_source_as_modification_to_as_punctuation"),
    ("train", 8891, (3,), "punctuation_source_as_modification_to_as_punctuation"),
    ("train", 11107, (16,), "punctuation_source_as_modification_to_as_punctuation"),
    ("test", 882, (4,), "punctuation_source_as_modification_to_as_punctuation"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def node_token_indexes(node: dict[str, Any]) -> tuple[int, ...]:
    spans = node["spans"]
    if isinstance(spans, str):
        return ()
    indexes: list[int] = []
    for start, end in spans:
        indexes.extend(range(start, end + 1))
    return tuple(indexes)


def node_text(sample: dict[str, Any], node_index: int) -> str:
    words = sample["lingua"]["words"]
    spans = graph(sample)["nodes"][node_index]["spans"]
    if isinstance(spans, str):
        return spans
    return "".join(words[index] for start, end in spans for index in range(start, end + 1)).strip()


def find_node(sample: dict[str, Any], tokens: tuple[int, ...], node_type: str | None = None) -> int:
    matches = [
        index
        for index, node in enumerate(graph(sample)["nodes"])
        if node_token_indexes(node) == tokens and (node_type is None or node.get("type") == node_type)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one node for tokens={tokens}, type={node_type}; got {matches}")
    return matches[0]


def add_edge_once(edges: list[dict[str, Any]], start: int, end: int, label: str) -> None:
    item = {"start": start, "end": end, "label": label}
    if item not in edges:
        edges.append(item)


def remove_edge(edges: list[dict[str, Any]], start: int, end: int, label: str) -> None:
    before = len(edges)
    edges[:] = [edge for edge in edges if not (edge["start"] == start and edge["end"] == end and edge["label"] == label)]
    if len(edges) == before:
        raise ValueError(f"missing edge {start}->{end} {label}")


def relabel_edge_once(
    edges: list[dict[str, Any]],
    start: int,
    end: int,
    from_label: str,
    to_label: str,
) -> bool:
    source = [edge for edge in edges if edge["start"] == start and edge["end"] == end and edge["label"] == from_label]
    target = [edge for edge in edges if edge["start"] == start and edge["end"] == end and edge["label"] == to_label]
    if source:
        if len(source) != 1:
            raise ValueError(f"expected one edge {start}->{end} {from_label}; got {source}")
        source[0]["label"] = to_label
        return True
    if target:
        return False
    raise ValueError(f"missing edge {start}->{end} {from_label}/{to_label}")


def rewire_edge(edges: list[dict[str, Any]], start: int, old_end: int, label: str, new_end: int) -> None:
    matches = [edge for edge in edges if edge["start"] == start and edge["end"] == old_end and edge["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"expected one edge {start}->{old_end} {label}; got {matches}")
    matches[0]["end"] = new_end


def add_aux_node(sample: dict[str, Any], spans: str, node_type: str) -> int:
    nodes = graph(sample)["nodes"]
    nodes.append({"spans": spans, "type": node_type})
    return len(nodes) - 1


def dedupe_edges(sample: dict[str, Any]) -> None:
    seen = set()
    unique = []
    for edge in graph(sample)["edges"]:
        key = (edge["start"], edge["end"], edge["label"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    graph(sample)["edges"] = unique


def apply_type_repairs(sample: dict[str, Any], split: str, line_number: int) -> list[str]:
    changes = []
    for case_split, case_line, tokens, from_type, to_type, change in TYPE_REPAIRS:
        if split != case_split or line_number != case_line:
            continue
        node_index = find_node(sample, tokens)
        node = graph(sample)["nodes"][node_index]
        if node.get("type") == from_type:
            node["type"] = to_type
            changes.append(f"{change}|node={node_index}|text={node_text(sample, node_index)!r}")
        elif node.get("type") != to_type:
            raise ValueError(
                f"unexpected node type for {split}:{line_number} tokens={tokens}: "
                f"{node.get('type')} not in {{{from_type}, {to_type}}}"
            )

    # In U.S. abbreviation fragments the final period was incorrectly modeled
    # as a modifier of the inner abbreviation expression; it is a punctuation
    # close marker.
    if (split, line_number) in {("train", 5136), ("train", 9458)}:
        dot = find_node(sample, (15,) if line_number == 5136 else (14,), "PunctuationalConstant")
        inner_dot = find_node(sample, (13,) if line_number == 5136 else (12,), "ExpressionFunctor")
        edges = graph(sample)["edges"]
        if relabel_edge_once(edges, dot, inner_dot, "as:modification", "as:punctuation"):
            changes.append(f"abbrev_final_dot_as_punctuation|node={dot}")

    return changes


def apply_punctuation_as_mod_relabels(sample: dict[str, Any], split: str, line_number: int) -> list[str]:
    changes = []
    for case_split, case_line, tokens, change in PUNCT_AS_MOD_RELABELS:
        if split != case_split or line_number != case_line:
            continue
        node_index = find_node(sample, tokens, "PunctuationalConstant")
        outgoing = [edge for edge in graph(sample)["edges"] if edge["start"] == node_index]
        as_mod = [edge for edge in outgoing if edge["label"] == "as:modification"]
        if as_mod:
            if len(as_mod) != 1:
                raise ValueError(f"expected one as:modification from punctuation {node_index}; got {as_mod}")
            as_mod[0]["label"] = "as:punctuation"
            changes.append(f"{change}|node={node_index}|text={node_text(sample, node_index)!r}")
        elif not any(edge["label"] == "as:punctuation" for edge in outgoing):
            raise ValueError(f"expected as:modification/as:punctuation from punctuation {node_index}; got {outgoing}")
    return changes


def repair_us_based(sample: dict[str, Any]) -> list[str]:
    hyphen = find_node(sample, (8,))
    us = find_node(sample, (7,), "NominalConstant")
    based = find_node(sample, (9,), "ModificationalConstant")
    supporter = find_node(sample, (10,), "NominalConstant")
    edges = graph(sample)["edges"]

    if graph(sample)["nodes"][hyphen]["type"] == "PunctuationalConstant" and {
        "start": hyphen,
        "end": us,
        "label": "as:punctuation",
    } in edges:
        return []

    outer = [
        index
        for index, node in enumerate(graph(sample)["nodes"])
        if node.get("spans") == "Modification"
        and {"start": index, "end": supporter, "label": "variable"} in edges
        and {"start": index, "end": hyphen, "label": "body"} in edges
    ]
    if len(outer) != 1:
        raise ValueError(f"expected one outer Modification for US-based supporter; got {outer}")
    outer_mod = outer[0]

    graph(sample)["nodes"][hyphen]["type"] = "PunctuationalConstant"
    remove_edge(edges, hyphen, us, "func.arg")
    remove_edge(edges, hyphen, based, "func.arg")
    add_edge_once(edges, hyphen, us, "as:punctuation")

    inner_mod = add_aux_node(sample, "Modification", "ModificationalFunctor")
    add_edge_once(edges, inner_mod, based, "variable")
    add_edge_once(edges, inner_mod, hyphen, "body")
    rewire_edge(edges, outer_mod, hyphen, "body", inner_mod)
    return [f"us_based_expression_to_hyphen_punctuation_modification|hyphen={hyphen}|inner_mod={inner_mod}"]


def repair_train_9810_dash(sample: dict[str, Any]) -> list[str]:
    dash = find_node(sample, (1,))
    year = find_node(sample, (0,), "NominalConstant")
    edges = graph(sample)["edges"]
    if graph(sample)["nodes"][dash]["type"] == "PunctuationalConstant" and {
        "start": dash,
        "end": year,
        "label": "as:punctuation",
    } in edges:
        return []

    graph(sample)["nodes"][dash]["type"] = "PunctuationalConstant"
    remove_edge(edges, year, dash, "as:modification")
    add_edge_once(edges, dash, year, "as:punctuation")

    wrappers = [
        index
        for index, node in enumerate(graph(sample)["nodes"])
        if node.get("spans") == "Modification" and {"start": index, "end": year, "label": "body"} in edges
    ]
    if len(wrappers) != 1:
        raise ValueError(f"expected one year-body Modification; got {wrappers}")
    rewire_edge(edges, wrappers[0], year, "body", dash)
    return [f"year_dash_to_punctuation_body|dash={dash}|wrapper={wrappers[0]}"]


def repair_dev_922_dash(sample: dict[str, Any]) -> list[str]:
    dash = find_node(sample, (1,))
    year = find_node(sample, (0,), "NominalConstant")
    brazil = find_node(sample, (2,), "NominalConstant")
    proclaimed = find_node(sample, (3,), "FactualPredicator")
    content = find_node(sample, (7,), "ConjunctionalFunctor")
    final_punct = find_node(sample, (18,), "PunctuationalConstant")
    edges = graph(sample)["edges"]

    if graph(sample)["nodes"][dash]["type"] == "PunctuationalConstant" and any(
        node.get("spans") == "Discourse"
        and {"start": index, "end": content, "label": "variable"} in edges
        and {"start": index, "end": dash, "label": "body"} in edges
        for index, node in enumerate(graph(sample)["nodes"])
    ):
        return []

    graph(sample)["nodes"][dash]["type"] = "PunctuationalConstant"
    remove_edge(edges, year, dash, "as:modification")
    remove_edge(edges, dash, brazil, "as:modification")
    add_edge_once(edges, dash, year, "as:punctuation")
    rewire_edge(edges, proclaimed, year, "pred.arg.1", brazil)

    discourse = add_aux_node(sample, "Discourse", "ModificationalFunctor")
    rewire_edge(edges, final_punct, content, "as:punctuation", discourse)
    add_edge_once(edges, discourse, content, "variable")
    add_edge_once(edges, discourse, dash, "body")
    return [f"year_dash_header_to_discourse|dash={dash}|discourse={discourse}"]


def repair_train_2092_dash(sample: dict[str, Any]) -> list[str]:
    country = find_node(sample, (0,), "NominalConstant")
    dash = find_node(sample, (1,))
    name_expr = [
        index
        for index, node in enumerate(graph(sample)["nodes"])
        if node.get("spans") == "Expression" and set(node_token_indexes(node)) == set()
    ]
    # There is one Expression aux for the player name in this fragment.
    if len(name_expr) != 1:
        raise ValueError(f"expected one name Expression aux; got {name_expr}")
    name = name_expr[0]
    attr = [
        index for index, node in enumerate(graph(sample)["nodes"]) if node.get("spans") == "Attribute"
    ]
    if len(attr) != 1:
        raise ValueError(f"expected one Attribute aux; got {attr}")

    edges = graph(sample)["edges"]
    if graph(sample)["nodes"][dash]["type"] == "PunctuationalConstant" and {
        "start": dash,
        "end": country,
        "label": "as:punctuation",
    } in edges:
        return []

    graph(sample)["nodes"][dash]["type"] = "PunctuationalConstant"
    remove_edge(edges, country, dash, "as:modification")
    remove_edge(edges, dash, name, "as:modification")
    add_edge_once(edges, dash, country, "as:punctuation")

    country_name = add_aux_node(sample, "Modification", "ModificationalFunctor")
    add_edge_once(edges, country_name, name, "variable")
    add_edge_once(edges, country_name, dash, "body")
    rewire_edge(edges, attr[0], country, "pred.arg.1", country_name)
    return [f"country_dash_name_to_modification|dash={dash}|mod={country_name}"]


STRUCTURAL_REPAIRS = {
    ("train", 7871): repair_us_based,
    ("train", 9810): repair_train_9810_dash,
    ("dev", 922): repair_dev_922_dash,
    ("train", 2092): repair_train_2092_dash,
}


def repair_sample(sample: dict[str, Any], split: str, line_number: int) -> list[str]:
    changes = apply_type_repairs(sample, split, line_number)
    changes.extend(apply_punctuation_as_mod_relabels(sample, split, line_number))
    repair = STRUCTURAL_REPAIRS.get((split, line_number))
    if repair is not None:
        changes.extend(repair(sample))
    if changes:
        dedupe_edges(sample)
    return changes


def compact_graph(sample: dict[str, Any]) -> str:
    lines = [sample["sentence"], "NODES"]
    for index, node in enumerate(graph(sample)["nodes"]):
        lines.append(f"{index}: {node_text(sample, index)!r} {node.get('spans')!r} {node.get('type')}")
    lines.append("EDGES")
    for edge in graph(sample)["edges"]:
        lines.append(
            f"{edge['start']}->{edge['end']} {edge['label']} | "
            f"{node_text(sample, edge['start'])!r}->{node_text(sample, edge['end'])!r}"
        )
    return "\n".join(lines)


def run(apply: bool, data_dir: Path, report_dir: Path) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    all_changes: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for split in ["train", "dev", "test"]:
        path = data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        new_rows = []
        for line_number, sample in enumerate(rows, 1):
            before = deepcopy(sample)
            changes = repair_sample(sample, split, line_number)
            if changes:
                for change in changes:
                    counts[change.split("|", 1)[0]] += 1
                all_changes.append(
                    {
                        "split": split,
                        "line": line_number,
                        "sentence": sample["sentence"],
                        "changes": changes,
                        "before": compact_graph(before),
                        "after": compact_graph(sample),
                    }
                )
            new_rows.append(sample)
        if apply:
            write_jsonl(path, new_rows)

    (report_dir / "changes.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in all_changes),
        encoding="utf-8",
    )
    summary = {
        "applied": apply,
        "changed_samples": len(all_changes),
        "changed_operations": sum(len(item["changes"]) for item in all_changes),
        "counts": dict(counts),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (report_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Punctuation/Symbol Context Repair",
                "",
                f"- applied: `{apply}`",
                f"- changed samples: `{summary['changed_samples']}`",
                f"- changed operations: `{summary['changed_operations']}`",
                "",
                "## Counts",
                "",
                *[f"- `{key}`: {value}" for key, value in sorted(counts.items())],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="rewrite data/train.jsonl, data/dev.jsonl, and data/test.jsonl")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--report-dir", type=Path, default=REPORTS / "punctuation_symbol_context_repair")
    args = parser.parse_args()
    return run(args.apply, args.data_dir, args.report_dir)


if __name__ == "__main__":
    raise SystemExit(main())
