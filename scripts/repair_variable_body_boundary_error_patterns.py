#!/usr/bin/env python3
"""Repair reviewed gold patterns behind BODY/VARIABLE boundary confusions.

The script is intentionally conservative:

* date expressions such as ``March 28, 2005`` and ``26 May 1974`` should be
  Expression structures, not month/day ``as:modification`` chains;
* ``Name, an expert ...`` is Apposition, not Modification.

Cases that need sentence-level scope judgment are written to the report and
left unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = [
    ROOT / "data" / "train.jsonl",
    ROOT / "data" / "dev.jsonl",
    ROOT / "data" / "test.jsonl",
]

MONTHS = {
    item.lower().rstrip(".")
    for item in (
        "January February March April May June July August September October November December "
        "Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec"
    ).split()
}
DAY_RE = re.compile(r"\d{1,2}(?:st|nd|rd|th)?$", re.IGNORECASE)
YEAR_RE = re.compile(r"\d{4}$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def graph(row: dict[str, Any]) -> dict[str, Any]:
    return row["lingua"]["graph"]


def node_tokens(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if not isinstance(spans, list):
        return []
    return [index for start, end in spans for index in range(start, end + 1)]


def node_text(row: dict[str, Any], index: int) -> str:
    nodes = graph(row)["nodes"]
    words = row["lingua"]["words"]
    indexes = node_tokens(nodes[index])
    if not indexes:
        return str(nodes[index].get("spans"))
    return "".join(words[token] for token in indexes).strip()


def descendant_text(row: dict[str, Any], start: int) -> str:
    nodes = graph(row)["nodes"]
    edges = graph(row)["edges"]
    outgoing: dict[int, list[int]] = {}
    for edge in edges:
        outgoing.setdefault(int(edge["start"]), []).append(int(edge["end"]))
    seen: set[int] = set()
    stack = [start]
    tokens: list[int] = []
    while stack:
        index = stack.pop()
        if index in seen or index < 0 or index >= len(nodes):
            continue
        seen.add(index)
        tokens.extend(node_tokens(nodes[index]))
        stack.extend(outgoing.get(index, []))
    return "".join(row["lingua"]["words"][token] for token in sorted(set(tokens))).strip()


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return int(edge["start"]), int(edge["end"]), str(edge["label"])


def dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, str]] = set()
    out: list[dict[str, Any]] = []
    for edge in edges:
        key = edge_key(edge)
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def add_edge(edges: list[dict[str, Any]], start: int, end: int, label: str) -> None:
    key = (start, end, label)
    if key not in {edge_key(edge) for edge in edges}:
        edges.append({"start": start, "end": end, "label": label})


def remove_edge_object(edges: list[dict[str, Any]], target: dict[str, Any]) -> None:
    key = edge_key(target)
    edges[:] = [edge for edge in edges if edge_key(edge) != key]


def redirect_incoming(edges: list[dict[str, Any]], old: int, new: int) -> None:
    for edge in edges:
        if int(edge["end"]) == old:
            edge["end"] = new


def is_month_text(text: str) -> bool:
    return text.lower().rstrip(".") in MONTHS


def is_day_text(text: str) -> bool:
    return bool(DAY_RE.fullmatch(text))


def is_year_node(row: dict[str, Any], index: int) -> bool:
    return bool(YEAR_RE.fullmatch(node_text(row, index)))


def expression_parents_of(row: dict[str, Any], node: int) -> list[int]:
    nodes = graph(row)["nodes"]
    parents: list[int] = []
    for edge in graph(row)["edges"]:
        if int(edge["end"]) != node or edge["label"] != "func.arg":
            continue
        parent = int(edge["start"])
        if nodes[parent].get("type") == "ExpressionFunctor":
            parents.append(parent)
    return parents


def month_year_edge(row: dict[str, Any], month: int) -> dict[str, Any] | None:
    matches = [
        edge
        for edge in graph(row)["edges"]
        if int(edge["start"]) == month and edge["label"] == "as:modification" and is_year_node(row, int(edge["end"]))
    ]
    return matches[0] if len(matches) == 1 else None


def nonyear_month_modifiers(row: dict[str, Any], month: int) -> list[dict[str, Any]]:
    return [
        edge
        for edge in graph(row)["edges"]
        if int(edge["start"]) == month and edge["label"] == "as:modification" and not is_year_node(row, int(edge["end"]))
    ]


def comma_year_root(row: dict[str, Any], day: int) -> tuple[int, int, dict[str, Any], dict[str, Any]] | None:
    nodes = graph(row)["nodes"]
    edges = graph(row)["edges"]
    for punctuation_edge in edges:
        if punctuation_edge["label"] != "as:punctuation" or int(punctuation_edge["end"]) != day:
            continue
        comma = int(punctuation_edge["start"])
        if nodes[comma].get("type") != "PunctuationalConstant" or node_text(row, comma) != ",":
            continue
        for year_edge in edges:
            if year_edge["label"] == "as:modification" and int(year_edge["end"]) == comma:
                year = int(year_edge["start"])
                if is_year_node(row, year):
                    return comma, year, punctuation_edge, year_edge
    return None


def new_expression_node(row: dict[str, Any]) -> int:
    nodes = graph(row)["nodes"]
    nodes.append({"spans": "Expression", "type": "ExpressionFunctor"})
    return len(nodes) - 1


def repair_date_modification_chains(row: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    edges = graph(row)["edges"]
    candidates = [
        edge
        for edge in list(edges)
        if edge["label"] == "as:modification"
        and is_day_text(node_text(row, int(edge["start"])))
        and is_month_text(node_text(row, int(edge["end"])))
    ]
    for day_month in candidates:
        if day_month not in edges:
            continue
        day = int(day_month["start"])
        month = int(day_month["end"])
        skipped = nonyear_month_modifiers(row, month)
        if skipped:
            changes.append(
                {
                    "action": "skip",
                    "pattern": "month_day_nonyear_modifier",
                    "day": node_text(row, day),
                    "month": node_text(row, month),
                    "nonyear_targets": [descendant_text(row, int(edge["end"])) for edge in skipped],
                    "reason": "date expression also modifies a following noun; needs NP-scope review",
                }
            )
            continue

        year_from_month = month_year_edge(row, month)
        year = int(year_from_month["end"]) if year_from_month else None
        root: int | None = None
        mode = "new_expression"

        existing_expression_parents = expression_parents_of(row, day)
        if len(existing_expression_parents) == 1:
            root = existing_expression_parents[0]
            mode = "existing_expression"

        comma_year = comma_year_root(row, day)
        if root is None and comma_year is not None:
            comma, comma_year_node, punctuation_edge, year_edge = comma_year
            root = comma
            year = comma_year_node
            mode = "punctuation_comma_to_expression"
            graph(row)["nodes"][comma]["type"] = "ExpressionFunctor"
            redirect_incoming(edges, comma_year_node, comma)
            remove_edge_object(edges, punctuation_edge)
            remove_edge_object(edges, year_edge)

        if root is None:
            root = new_expression_node(row)
            redirect_incoming(edges, day, root)

        remove_edge_object(edges, day_month)
        if year_from_month is not None:
            remove_edge_object(edges, year_from_month)

        add_edge(edges, root, month, "func.arg")
        add_edge(edges, root, day, "func.arg")
        if year is not None:
            add_edge(edges, root, year, "func.arg")

        changes.append(
            {
                "action": "repair",
                "pattern": "date_expression_not_modification",
                "mode": mode,
                "root": root,
                "day": node_text(row, day),
                "month": node_text(row, month),
                "year": node_text(row, year) if year is not None else "",
            }
        )

    if changes:
        graph(row)["edges"] = dedupe_edges(edges)
    return changes


def repair_expert_apposition(row: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    nodes = graph(row)["nodes"]
    edges = graph(row)["edges"]
    for index, node in enumerate(nodes):
        if node.get("spans") != "Modification" or node.get("type") != "ModificationalFunctor":
            continue
        outgoing = [edge for edge in edges if int(edge["start"]) == index]
        variable = [edge for edge in outgoing if edge["label"] == "variable"]
        body = [edge for edge in outgoing if edge["label"] == "body"]
        if len(outgoing) != 2 or len(variable) != 1 or len(body) != 1:
            continue
        variable_text = descendant_text(row, int(variable[0]["end"]))
        body_text = descendant_text(row, int(body[0]["end"]))
        if not variable_text.endswith(","):
            continue
        if not re.match(r"(?i)^(a|an|the)\s+expert\b", body_text):
            continue

        node["spans"] = "Apposition"
        node["type"] = "AppositionalPredicator"
        variable[0]["label"] = "pred.arg.1"
        body[0]["label"] = "pred.arg.2"
        changes.append(
            {
                "action": "repair",
                "pattern": "comma_expert_apposition",
                "root": index,
                "arg1": variable_text,
                "arg2": body_text,
            }
        )
    if changes:
        graph(row)["edges"] = dedupe_edges(edges)
    return changes


def audit_measurement_parentheticals(row: dict[str, Any]) -> list[dict[str, Any]]:
    sentence = row["sentence"]
    if "(" not in sentence or ")" not in sentence:
        return []
    unit_re = re.compile(
        r"\b(?:cm|mm|m|km|kg|g|lb|lbs|ft|feet|inch|inches|mi|miles|sq|gmt|utc|p\.m\.|a\.m\.)\b",
        re.IGNORECASE,
    )
    if not unit_re.search(sentence):
        return []
    out: list[dict[str, Any]] = []
    for index, node in enumerate(graph(row)["nodes"]):
        if node.get("spans") == "Modification" and node.get("type") == "ModificationalFunctor":
            text = descendant_text(row, index)
            if "(" in text and ")" in text and unit_re.search(text):
                out.append({"node": index, "text": text})
    return out


def process_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    changes: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    changes.extend(repair_date_modification_chains(row))
    changes.extend(repair_expert_apposition(row))
    review.extend(
        {"pattern": "measurement_parenthetical_scope_review", **item}
        for item in audit_measurement_parentheticals(row)
    )
    review.extend(item for item in changes if item.get("action") == "skip")
    changes = [item for item in changes if item.get("action") == "repair"]
    return changes, review


def write_report(report_dir: Path, changes: list[dict[str, Any]], review: list[dict[str, Any]], apply: bool) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "changes.jsonl").open("w", encoding="utf-8") as handle:
        for row in changes:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (report_dir / "manual_review.jsonl").open("w", encoding="utf-8") as handle:
        for row in review:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    change_counts = Counter(row["pattern"] for row in changes)
    review_counts = Counter(row["pattern"] for row in review)
    summary = {
        "apply": apply,
        "num_changes": len(changes),
        "change_counts": dict(change_counts),
        "num_manual_review": len(review),
        "manual_review_counts": dict(review_counts),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Variable/body boundary pattern repair",
        "",
        f"- apply: `{apply}`",
        f"- changes: `{len(changes)}`",
        f"- manual_review: `{len(review)}`",
        "",
        "## Change counts",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in sorted(change_counts.items()))
    lines.extend(["", "## Manual review counts"])
    lines.extend(f"- `{key}`: `{value}`" for key, value in sorted(review_counts.items()))
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--report_dir",
        type=Path,
        default=ROOT / "reports" / "variable_body_boundary_pattern_repair",
    )
    args = parser.parse_args()

    all_changes: list[dict[str, Any]] = []
    all_review: list[dict[str, Any]] = []
    changed_files: dict[Path, list[dict[str, Any]]] = {}

    for path in DATA_FILES:
        split = path.stem
        rows = read_jsonl(path)
        output_rows = rows if args.apply else copy.deepcopy(rows)
        for lineno, row in enumerate(output_rows, start=1):
            changes, review = process_row(row)
            for item in changes:
                item.update({"split": split, "lineno": lineno, "sentence": row["sentence"]})
            for item in review:
                item.update({"split": split, "lineno": lineno, "sentence": row["sentence"]})
            all_changes.extend(changes)
            all_review.extend(review)
        if args.apply:
            changed_files[path] = output_rows

    if args.apply:
        for path, rows in changed_files.items():
            write_jsonl(path, rows)
    write_report(args.report_dir, all_changes, all_review, args.apply)
    print(json.dumps({"changes": len(all_changes), "manual_review": len(all_review)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
