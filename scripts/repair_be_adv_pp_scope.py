#!/usr/bin/env python3
"""Repair copular `be + adverb/negation + PP` scope.

Standard implemented here:

1. For predicative copular PPs, the PP relation is the semantic predicate
   complement of `be`.
2. Negation and ordinary scope adverbs between `be` and the preposition modify
   the PP relation, not the bare copula.
3. `no longer` is one modifier phrase. In `is no longer in business`, `no
   longer` scopes over `in business`.
4. Locative predicate material such as `here/there` is not treated as a scope
   adverb by this repair.
5. Passive/progressive/main-verb predicates and `be + adjective + PP` cases are
   excluded by requiring a pure copular predicate with no `pred.arg.2`
   complement before the PP repair.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repair_be_copula_rules import (  # noqa: E402
    BE_FORMS,
    DIRECT_PREPOSITIONS,
    MODALS_AND_AUX,
    add_edge,
    add_node,
    compact_graph,
    edge_out,
    graph,
    node_text,
    node_tokens,
    read_jsonl,
    remove_edges,
    set_node_token_indexes,
    spans_from_token_indexes,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

LOCATIVE_OR_PARTICLE_COMPLEMENTS = {
    "about",
    "above",
    "across",
    "ahead",
    "along",
    "apart",
    "around",
    "aside",
    "away",
    "back",
    "down",
    "forward",
    "here",
    "in",
    "indoors",
    "inside",
    "off",
    "on",
    "out",
    "outdoors",
    "outside",
    "over",
    "there",
    "through",
    "together",
    "under",
    "up",
    "upstairs",
}

RESIDUAL_SCOPE_ADVERBS = {
    "all",
    "longer",
}

ALL_SCOPE_PREPOSITIONS = {
    "about",
    "for",
}


def token_indexes(sample: dict[str, Any], node_index: int) -> list[int]:
    return [index for index, _surface, _normalized in node_tokens(sample, node_index)]


def token_norms(sample: dict[str, Any], node_index: int) -> list[str]:
    return [normalized for _index, _surface, normalized in node_tokens(sample, node_index)]


def node_type(sample: dict[str, Any], node_index: int) -> str | None:
    return graph(sample)["nodes"][node_index].get("type")


def is_factual(sample: dict[str, Any], node_index: int) -> bool:
    return node_type(sample, node_index) == "FactualPredicator"


def is_modifier(sample: dict[str, Any], node_index: int) -> bool:
    return node_type(sample, node_index) == "ModificationalConstant"


def is_be_apostrophe(tokens: list[tuple[int, str, str]], position: int) -> bool:
    if tokens[position][2] != "'":
        return False
    return position + 1 < len(tokens) and tokens[position + 1][2] in {"m", "re", "s"}


def is_core_be_token(tokens: list[tuple[int, str, str]], position: int) -> bool:
    normalized = tokens[position][2]
    return normalized in BE_FORMS or normalized in MODALS_AND_AUX or is_be_apostrophe(tokens, position)


def partition_be_node(
    sample: dict[str, Any],
    node_index: int,
) -> tuple[list[int], list[int], list[str]] | None:
    tokens = node_tokens(sample, node_index)
    if not tokens:
        return None
    has_be = any(normalized in BE_FORMS for _index, _surface, normalized in tokens)
    if not has_be:
        return None

    core_indexes: list[int] = []
    residual_indexes: list[int] = []
    residual_norms: list[str] = []
    for position, (index, _surface, normalized) in enumerate(tokens):
        if is_core_be_token(tokens, position):
            core_indexes.append(index)
        elif normalized in RESIDUAL_SCOPE_ADVERBS:
            residual_indexes.append(index)
            residual_norms.append(normalized)
        else:
            return None

    if not core_indexes:
        return None
    return sorted(core_indexes), sorted(residual_indexes), residual_norms


def modifier_chain_to_factual(
    sample: dict[str, Any],
    start: int,
) -> tuple[list[int], int] | None:
    chain: list[int] = []
    current = start
    seen: set[int] = set()
    while current not in seen:
        seen.add(current)
        if not is_modifier(sample, current):
            return None
        chain.append(current)
        outgoing = edge_out(sample, current, "as:modification")
        if len(outgoing) != 1:
            return None
        target = outgoing[0]["end"]
        if is_factual(sample, target):
            return chain, target
        current = target
    return None


def incoming_modifier_chain_to_target(
    sample: dict[str, Any],
    target: int,
) -> tuple[list[int], int] | None:
    incoming = [
        edge
        for edge in graph(sample)["edges"]
        if edge["end"] == target and edge["label"] == "as:modification" and is_modifier(sample, edge["start"])
    ]
    if len(incoming) != 1:
        return None

    chain = [incoming[0]["start"]]
    current = chain[0]
    seen = {target, current}
    while True:
        parents = [
            edge
            for edge in graph(sample)["edges"]
            if edge["end"] == current
            and edge["label"] == "as:modification"
            and is_modifier(sample, edge["start"])
        ]
        if not parents:
            break
        if len(parents) != 1:
            return None
        parent = parents[0]["start"]
        if parent in seen:
            return None
        chain.insert(0, parent)
        seen.add(parent)
        current = parent
    return chain, chain[-1]


def chain_token_indexes(sample: dict[str, Any], chain: list[int]) -> list[int]:
    indexes: list[int] = []
    for node_index in chain:
        indexes.extend(token_indexes(sample, node_index))
    return sorted(indexes)


def chain_norms(sample: dict[str, Any], chain: list[int]) -> list[str]:
    norms: list[str] = []
    for node_index in chain:
        norms.extend(token_norms(sample, node_index))
    return norms


def is_allowed_modifier_chain(
    sample: dict[str, Any],
    chain: list[int],
    prep_norm: str,
) -> bool:
    norms = [normalized for normalized in chain_norms(sample, chain) if normalized != "'"]
    if not norms:
        return False
    if any(normalized in LOCATIVE_OR_PARTICLE_COMPLEMENTS for normalized in norms):
        return False
    if "all" in norms and prep_norm not in ALL_SCOPE_PREPOSITIONS:
        return False
    return True


def residuals_allowed_for_prep(residual_norms: list[str], prep_norm: str) -> bool:
    if not residual_norms:
        return True
    if any(normalized == "all" for normalized in residual_norms) and prep_norm not in ALL_SCOPE_PREPOSITIONS:
        return False
    return all(normalized in RESIDUAL_SCOPE_ADVERBS for normalized in residual_norms)


def indexes_between(middle: list[int], left: list[int], right: list[int]) -> bool:
    if not middle or not left or not right:
        return False
    lower = max(left)
    upper = min(right)
    return all(lower < index < upper for index in middle)


def set_or_add_modifier_edge(sample: dict[str, Any], start: int, end: int) -> None:
    for edge in graph(sample)["edges"]:
        if edge["start"] == start and edge["end"] == end and edge["label"] == "as:modification":
            return
    add_edge(sample, start, end, "as:modification")


def transfer_incoming(
    sample: dict[str, Any],
    old: int,
    new: int,
    excluded_starts: set[int] | None = None,
) -> None:
    excluded_starts = excluded_starts or set()
    for edge in graph(sample)["edges"]:
        if edge["end"] == old and edge["start"] != new and edge["start"] not in excluded_starts:
            edge["end"] = new


def make_modifier_chain(
    sample: dict[str, Any],
    modifier_indexes: list[int],
    target: int,
) -> tuple[int, int, set[int]]:
    spans = spans_from_token_indexes(modifier_indexes)
    current_target = target
    new_nodes: set[int] = set()
    tail = -1
    for start, end in reversed(spans):
        node_index = add_node(sample, [[start, end]], "ModificationalConstant")
        if tail < 0:
            tail = node_index
        new_nodes.add(node_index)
        add_edge(sample, node_index, current_target, "as:modification")
        current_target = node_index
    return current_target, tail, new_nodes


def extend_chain_root_with_residual(
    sample: dict[str, Any],
    chain: list[int],
    residual_indexes: list[int],
    residual_norms: list[str],
) -> bool:
    if not residual_indexes:
        return True
    root = chain[0]
    root_norms = [normalized for normalized in token_norms(sample, root) if normalized != "'"]
    if root_norms == ["no"] and residual_norms == ["longer"]:
        set_node_token_indexes(sample, root, sorted(token_indexes(sample, root) + residual_indexes))
        return True
    return False


def repair_existing_prep_with_modifier_chain(sample: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    original_node_count = len(graph(sample)["nodes"])
    for prep_index in range(original_node_count):
        if node_type(sample, prep_index) != "ModificationalFunctor":
            continue
        prep_tokens = node_tokens(sample, prep_index)
        if len(prep_tokens) != 1 or prep_tokens[0][2] not in DIRECT_PREPOSITIONS:
            continue
        prep_norm = prep_tokens[0][2]
        variable_edges = edge_out(sample, prep_index, "variable")
        body_edges = edge_out(sample, prep_index, "body")
        if len(variable_edges) != 1 or len(body_edges) != 1:
            continue

        chain_result = modifier_chain_to_factual(sample, variable_edges[0]["end"])
        if chain_result is None:
            continue
        chain, be_index = chain_result
        if edge_out(sample, be_index, "pred.arg.2"):
            continue
        partition = partition_be_node(sample, be_index)
        if partition is None:
            continue
        be_indexes, residual_indexes, residual_norms = partition
        if not residuals_allowed_for_prep(residual_norms, prep_norm):
            continue
        modifier_indexes = chain_token_indexes(sample, chain) + residual_indexes
        if not indexes_between(modifier_indexes, be_indexes, [prep_tokens[0][0]]):
            continue
        if not is_allowed_modifier_chain(sample, chain, prep_norm):
            continue
        if not extend_chain_root_with_residual(sample, chain, residual_indexes, residual_norms):
            continue

        before = {
            "prep": node_text(sample, prep_index),
            "modifier": " > ".join(node_text(sample, node_index) for node_index in chain),
            "be": node_text(sample, be_index),
        }
        tail = chain[-1]
        root = chain[0]
        set_node_token_indexes(sample, be_index, be_indexes)
        variable_edges[0]["end"] = be_index
        remove_edges(
            sample,
            lambda edge, t=tail, b=be_index: (
                edge["start"] == t and edge["end"] == b and edge["label"] == "as:modification"
            ),
        )
        transfer_incoming(sample, prep_index, root, excluded_starts=set(chain))
        set_or_add_modifier_edge(sample, tail, prep_index)
        changes.append(
            {
                "kind": "existing_prep_modifier_scoped_to_pp",
                "prep_index": prep_index,
                "modifier_root": root,
                "modifier_tail": tail,
                "be_index": be_index,
                "before": before,
                "after": {
                    "prep": node_text(sample, prep_index),
                    "modifier": " > ".join(node_text(sample, node_index) for node_index in chain),
                    "be": node_text(sample, be_index),
                },
            }
        )
    return changes


def repair_existing_prep_with_residual_be_modifier(sample: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    original_node_count = len(graph(sample)["nodes"])
    for prep_index in range(original_node_count):
        if node_type(sample, prep_index) != "ModificationalFunctor":
            continue
        prep_tokens = node_tokens(sample, prep_index)
        if len(prep_tokens) != 1 or prep_tokens[0][2] not in DIRECT_PREPOSITIONS:
            continue
        prep_norm = prep_tokens[0][2]
        variable_edges = edge_out(sample, prep_index, "variable")
        body_edges = edge_out(sample, prep_index, "body")
        if len(variable_edges) != 1 or len(body_edges) != 1:
            continue
        be_index = variable_edges[0]["end"]
        if not is_factual(sample, be_index) or edge_out(sample, be_index, "pred.arg.2"):
            continue
        partition = partition_be_node(sample, be_index)
        if partition is None:
            continue
        be_indexes, residual_indexes, residual_norms = partition
        if not residual_indexes or not residuals_allowed_for_prep(residual_norms, prep_norm):
            continue
        if not indexes_between(residual_indexes, be_indexes, [prep_tokens[0][0]]):
            continue

        before = {
            "prep": node_text(sample, prep_index),
            "modifier": " ".join(sample["lingua"]["words"][index].strip() for index in residual_indexes),
            "be": node_text(sample, be_index),
        }
        set_node_token_indexes(sample, be_index, be_indexes)
        modifier_root, modifier_tail, new_nodes = make_modifier_chain(sample, residual_indexes, prep_index)
        transfer_incoming(sample, prep_index, modifier_root, excluded_starts=new_nodes)
        changes.append(
            {
                "kind": "existing_prep_residual_be_modifier_scoped_to_pp",
                "prep_index": prep_index,
                "modifier_root": modifier_root,
                "modifier_tail": modifier_tail,
                "be_index": be_index,
                "before": before,
                "after": {
                    "prep": node_text(sample, prep_index),
                    "modifier": node_text(sample, modifier_root),
                    "be": node_text(sample, be_index),
                },
            }
        )
    return changes


def partition_merged_be_prep_node(
    sample: dict[str, Any],
    node_index: int,
) -> tuple[list[int], list[int], list[int], str] | None:
    tokens = node_tokens(sample, node_index)
    if not tokens:
        return None
    be_positions = [position for position, (_index, _surface, normalized) in enumerate(tokens) if normalized in BE_FORMS]
    if not be_positions:
        return None
    prep_positions = [
        position
        for position, (_index, _surface, normalized) in enumerate(tokens)
        if normalized in DIRECT_PREPOSITIONS and position > be_positions[-1]
    ]
    if not prep_positions:
        return None
    prep_position = prep_positions[-1]
    if prep_position != len(tokens) - 1:
        return None

    prep_index, _prep_surface, prep_norm = tokens[prep_position]
    be_indexes: list[int] = []
    residual_indexes: list[int] = []
    residual_norms: list[str] = []
    for position, (index, _surface, normalized) in enumerate(tokens[:prep_position]):
        if is_core_be_token(tokens, position):
            be_indexes.append(index)
        elif normalized in RESIDUAL_SCOPE_ADVERBS:
            residual_indexes.append(index)
            residual_norms.append(normalized)
        else:
            return None
    if not be_indexes:
        return None
    if not residuals_allowed_for_prep(residual_norms, prep_norm):
        return None
    return sorted(be_indexes), sorted(residual_indexes), [prep_index], prep_norm


def repair_merged_be_prep_with_scope_modifier(sample: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    original_node_count = len(graph(sample)["nodes"])
    for node_index in range(original_node_count):
        if not is_factual(sample, node_index):
            continue
        body_edges = edge_out(sample, node_index, "pred.arg.2")
        if len(body_edges) != 1:
            continue
        partition = partition_merged_be_prep_node(sample, node_index)
        if partition is None:
            continue
        be_indexes, residual_indexes, prep_indexes, prep_norm = partition
        chain_result = incoming_modifier_chain_to_target(sample, node_index)
        chain: list[int] | None = None
        tail: int | None = None
        if chain_result is not None:
            chain, tail = chain_result
            modifier_indexes = chain_token_indexes(sample, chain) + residual_indexes
            if not indexes_between(modifier_indexes, be_indexes, prep_indexes):
                continue
            if not is_allowed_modifier_chain(sample, chain, prep_norm):
                continue
            if not extend_chain_root_with_residual(sample, chain, residual_indexes, [sample["lingua"]["words"][i].strip().lower() for i in residual_indexes]):
                continue
            modifier_root = chain[0]
            excluded_starts = set(chain)
        else:
            if not residual_indexes:
                continue
            if not indexes_between(residual_indexes, be_indexes, prep_indexes):
                continue
            modifier_root = -1
            excluded_starts = set()

        object_node = body_edges[0]["end"]
        before = {
            "predicate": node_text(sample, node_index),
            "modifier": " > ".join(node_text(sample, item) for item in chain) if chain else "",
            "object": node_text(sample, object_node),
        }

        if chain is not None and tail is not None:
            remove_edges(
                sample,
                lambda edge, t=tail, n=node_index: (
                    edge["start"] == t and edge["end"] == n and edge["label"] == "as:modification"
                ),
            )
        set_node_token_indexes(sample, node_index, be_indexes)
        remove_edges(
            sample,
            lambda edge, n=node_index, obj=object_node: (
                edge["start"] == n and edge["end"] == obj and edge["label"] == "pred.arg.2"
            ),
        )
        prep_node = add_node(sample, spans_from_token_indexes(prep_indexes), "ModificationalFunctor")
        if chain is None:
            modifier_root, tail, new_nodes = make_modifier_chain(sample, residual_indexes, prep_node)
            excluded_starts = new_nodes
        else:
            assert tail is not None
            set_or_add_modifier_edge(sample, tail, prep_node)

        transfer_incoming(sample, node_index, modifier_root, excluded_starts=excluded_starts)
        add_edge(sample, prep_node, node_index, "variable")
        add_edge(sample, prep_node, object_node, "body")
        changes.append(
            {
                "kind": "merged_be_prep_scope_modifier_split",
                "predicate_index": node_index,
                "prep_index": prep_node,
                "modifier_root": modifier_root,
                "before": before,
                "after": {
                    "predicate": node_text(sample, node_index),
                    "modifier": node_text(sample, modifier_root),
                    "prep": node_text(sample, prep_node),
                    "object": node_text(sample, object_node),
                },
            }
        )
    return changes


def repair_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    changes.extend(repair_existing_prep_with_modifier_chain(sample))
    changes.extend(repair_existing_prep_with_residual_be_modifier(sample))
    changes.extend(repair_merged_be_prep_with_scope_modifier(sample))
    return changes


def write_standard(report_dir: Path) -> None:
    lines = [
        "# Be + Scope Modifier + PP Standard",
        "",
        "## Scope Rule",
        "",
        "- In a copular PP predication, the preposition is the relation head and `be` supplies the variable/event support.",
        "- A modifier between `be` and the preposition scopes over the PP relation: `is not in danger` is `not -> in`, and `in -> is` by `variable`.",
        "- `no longer` is a single `ModificationalConstant`; in `is no longer in business`, `no longer -> in`, `in -> is`, and `in -> business`.",
        "- This is different from ordinary verbal/passive clauses: in `were no longer expressed in monocytes`, `no longer` modifies `were expressed`, while `in monocytes` remains a PP adjunct of that predicate.",
        "",
        "## Automatic Repair Boundary",
        "",
        "- Auto-fix only when the target predicate is pure `be` plus optional modal/aux material, with no `pred.arg.2` adjectival/state complement already present.",
        "- Auto-fix split bad shapes: `PP.variable -> modifier -> be` becomes `modifier -> PP`, `PP.variable -> be`.",
        "- Auto-fix residual bad shapes where `all/longer` remained inside the be node, e.g. `is all about` and `is no longer in`.",
        "- Do not auto-fix `be + adjective/participle + PP` (`was clearly pleased at...`) or passive/progressive predicates (`was not inhibited by...`).",
        "- Do not treat `here/there/back/away/in/out/up` as scope adverbs here; they are locative/particle predicate material and need a separate standard.",
        "",
        "## Examples",
        "",
        "```text",
        "is no longer in business",
        "no longer -> in          as:modification",
        "in -> is                 variable",
        "in -> business           body",
        "is -> location           pred.arg.1",
        "```",
        "",
        "```text",
        "I was just in Oklahoma",
        "just -> in               as:modification",
        "in -> was                variable",
        "in -> Oklahoma           body",
        "was -> I                 pred.arg.1",
        "```",
        "",
    ]
    (report_dir / "standard.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(
    report_dir: Path,
    args: argparse.Namespace,
    all_changes: list[dict[str, Any]],
    counts: Counter[str],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "changes.jsonl").write_text(
        "\n".join(json.dumps(change, ensure_ascii=False) for change in all_changes)
        + ("\n" if all_changes else ""),
        encoding="utf-8",
    )
    write_standard(report_dir)
    summary_lines = [
        "# Be + Scope Modifier + PP Repair Summary",
        "",
        f"- input_dir: `{args.input_dir}`",
        f"- output_dir: `{args.output_dir}`",
        f"- report_dir: `{args.report_dir}`",
        f"- changed samples: {len(all_changes)}",
        "",
        "## Counts",
        "",
        "| key | count |",
        "|---|---:|",
    ]
    for key, count in sorted(counts.items()):
        summary_lines.append(f"| {key} | {count} |")
    summary_lines.extend(["", "## First 80 Changes", ""])
    for change in all_changes[:80]:
        kinds = ", ".join(item["kind"] for item in change["changes"])
        summary_lines.extend(
            [
                f"### {change['split']}:{change['line_number']} `{kinds}`",
                "",
                change["sentence"],
                "",
                "```text",
                change["before"],
                "",
                "--- after ---",
                change["after"],
                "```",
                "",
            ]
        )
    (report_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (report_dir / "summary.json").write_text(
        json.dumps(
            {
                "input_dir": str(args.input_dir),
                "output_dir": str(args.output_dir),
                "report_dir": str(args.report_dir),
                "changed_samples": len(all_changes),
                "counts": dict(counts),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    all_changes: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for split in ("train", "dev", "test"):
        rows = read_jsonl(input_dir / f"{split}.jsonl")
        new_rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(rows, 1):
            before = deepcopy(row)
            after = deepcopy(row)
            changes = repair_sample(after)
            if changes:
                for change in changes:
                    counts[change["kind"]] += 1
                    counts[f"{split}:{change['kind']}"] += 1
                all_changes.append(
                    {
                        "split": split,
                        "line_number": line_number,
                        "sentence": row.get("sentence", ""),
                        "changes": changes,
                        "before": compact_graph(before),
                        "after": compact_graph(after),
                    }
                )
                new_rows.append(after)
            else:
                new_rows.append(row)
        write_jsonl(output_dir / f"{split}.jsonl", new_rows)

    write_report(report_dir, args, all_changes, counts)
    return {
        "changed_samples": len(all_changes),
        "counts": dict(counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
