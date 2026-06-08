#!/usr/bin/env python3
"""Split non-particle postmodifier material out of FactualPredicator nodes.

The input is the audit produced by
`audit_factual_predicator_particle_filter.py`. Only rows in the
`split_non_particle` bucket are considered.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repair_be_copula_rules import (  # noqa: E402
    BE_FORMS,
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
    transfer_incoming_to_new_node,
    write_jsonl,
)


FUNCTOR_TOKENS = {
    "about",
    "across",
    "around",
    "behind",
    "by",
    "down",
    "in",
    "inside",
    "off",
    "on",
    "out",
    "outside",
    "over",
    "through",
    "under",
    "up",
}

LOCATIVE_BEFORE_BE_MODIFIERS = {"here"}
EXISTENTIAL_BEFORE_BE_NOMINALS = {"there"}
BE_COMPLEMENT_TOKENS = {
    "away",
    "back",
    "close",
    "down",
    "here",
    "home",
    "inside",
    "north",
    "off",
    "open",
    "out",
    "outside",
    "over",
    "south",
    "there",
    "west",
}
FUNCTION_JOINERS = {"-", "and"}
DIRECTIONAL_RELATION_TOKENS = {"east", "north", "south", "west"}
SMALL_DEGREE_TOKENS = {"one", "two", "three"}
PREFIXAL_HYPHEN_MODIFIERS = {"off", "over"}


def token_indexes(sample: dict[str, Any], node_index: int) -> list[int]:
    return [index for index, _surface, _norm in node_tokens(sample, node_index)]


def token_norms(sample: dict[str, Any], node_index: int) -> list[str]:
    return [norm for _index, _surface, norm in node_tokens(sample, node_index)]


def words(sample: dict[str, Any]) -> list[str]:
    return sample["lingua"]["words"]


def descendants(sample: dict[str, Any], node_index: int) -> list[int]:
    seen: set[int] = set()
    stack = [node_index]
    indexes: list[int] = []
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        indexes.extend(token_indexes(sample, current))
        stack.extend(edge["end"] for edge in edge_out(sample, current))
    return indexes


def has_be(norms: list[str]) -> bool:
    return any(norm in BE_FORMS for norm in norms)


def be_positions(norms: list[str]) -> list[int]:
    return [position for position, norm in enumerate(norms) if norm in BE_FORMS]


def core_predicate_token(norm: str) -> bool:
    return norm in BE_FORMS or norm in MODALS_AND_AUX or norm in {"'", "d", "ll", "ve", "wo", "ca"}


def contiguous_indexes(tokens: list[tuple[int, str, str]], start_pos: int, end_pos: int) -> list[int]:
    return [tokens[position][0] for position in range(start_pos, end_pos + 1)]


def target_positions(tokens: list[tuple[int, str, str]], targets: set[str]) -> list[int]:
    return [position for position, (_index, _surface, norm) in enumerate(tokens) if norm in targets]


def add_constant(
    sample: dict[str, Any],
    indexes: list[int],
    node_type: str = "ModificationalConstant",
) -> int:
    return add_node(sample, spans_from_token_indexes(indexes), node_type)


def add_hyphen_expression_or_constant(sample: dict[str, Any], indexes: list[int]) -> int:
    token_by_index = {
        index: norm
        for index, _surface, norm in [
            item
            for node_index in range(len(graph(sample)["nodes"]))
            for item in node_tokens(sample, node_index)
        ]
    }
    hyphen_indexes = [index for index in indexes if token_by_index.get(index) == "-"]
    if len(hyphen_indexes) != 1:
        return add_constant(sample, indexes)
    hyphen = hyphen_indexes[0]
    left_indexes = [index for index in indexes if index < hyphen]
    right_indexes = [index for index in indexes if index > hyphen]
    if not left_indexes or not right_indexes:
        return add_constant(sample, indexes)
    left_norms = [token_by_index.get(index, "") for index in left_indexes]
    if len(left_norms) == 1 and left_norms[0] in PREFIXAL_HYPHEN_MODIFIERS:
        left = add_constant(sample, left_indexes)
        dash = add_node(sample, [[hyphen, hyphen]], "PunctuationalConstant")
        right = add_constant(sample, right_indexes)
        mod = add_node(sample, "Modification", "ModificationalFunctor")
        add_edge(sample, dash, left, "as:punctuation")
        add_edge(sample, mod, right, "variable")
        add_edge(sample, mod, dash, "body")
        return mod
    expression = add_node(sample, [[hyphen, hyphen]], "ExpressionFunctor")
    left = add_constant(sample, left_indexes)
    right = add_constant(sample, right_indexes)
    add_edge(sample, expression, left, "func.arg")
    add_edge(sample, expression, right, "func.arg")
    return expression


def relabel_postposed_arg1_to_arg2(sample: dict[str, Any], node_index: int, predicate_indexes: list[int]) -> None:
    if edge_out(sample, node_index, "pred.arg.2") or not predicate_indexes:
        return
    predicate_min = min(predicate_indexes)
    predicate_max = max(predicate_indexes)
    arg1_edges = edge_out(sample, node_index, "pred.arg.1")
    if not arg1_edges:
        return
    has_preposed_subject = False
    for edge in arg1_edges:
        child_indexes = descendants(sample, edge["end"])
        if child_indexes and min(child_indexes) < predicate_min:
            has_preposed_subject = True
            break
    if has_preposed_subject:
        return
    for edge in arg1_edges:
        child_indexes = descendants(sample, edge["end"])
        if child_indexes and min(child_indexes) > predicate_max:
            edge["label"] = "pred.arg.2"


def split_as_constant_modifier(
    sample: dict[str, Any],
    node_index: int,
    modifier_indexes: list[int],
    keep_indexes: list[int],
) -> int:
    modifier = add_constant(sample, modifier_indexes)
    set_node_token_indexes(sample, node_index, keep_indexes)
    add_edge(sample, modifier, node_index, "as:modification")
    transfer_incoming_to_new_node(sample, node_index, modifier, excluded_starts={modifier})
    return modifier


def split_as_be_complement(
    sample: dict[str, Any],
    node_index: int,
    complement_indexes: list[int],
    keep_indexes: list[int],
) -> int:
    complement = add_hyphen_expression_or_constant(sample, complement_indexes)
    set_node_token_indexes(sample, node_index, keep_indexes)
    add_edge(sample, node_index, complement, "pred.arg.2")
    return complement


def split_as_functor(
    sample: dict[str, Any],
    node_index: int,
    functor_indexes: list[int],
    keep_indexes: list[int],
    body_node: int,
    variable_node: int | None = None,
    excluded_starts: set[int] | None = None,
) -> int:
    functor = add_node(sample, spans_from_token_indexes(functor_indexes), "ModificationalFunctor")
    set_node_token_indexes(sample, node_index, keep_indexes)
    remove_edges(
        sample,
        lambda edge, n=node_index, b=body_node: (
            edge["start"] == n and edge["end"] == b and edge["label"] == "pred.arg.2"
        ),
    )
    add_edge(sample, functor, variable_node if variable_node is not None else node_index, "variable")
    add_edge(sample, functor, body_node, "body")
    transfer_incoming_to_new_node(
        sample,
        node_index,
        functor,
        excluded_starts={functor} | (excluded_starts or set()),
    )
    return functor


def convert_single_predicator_relation(sample: dict[str, Any], node_index: int) -> str | None:
    out_labels = {edge["label"] for edge in edge_out(sample, node_index)}
    if not out_labels <= {"pred.arg.1", "pred.arg.2"}:
        return None
    graph(sample)["nodes"][node_index]["type"] = "ModificationalFunctor"
    for edge in edge_out(sample, node_index):
        if edge["label"] == "pred.arg.1":
            edge["label"] = "variable"
        elif edge["label"] == "pred.arg.2":
            edge["label"] = "body"
    return "single_relation_predicator_to_functor"


def repair_before_be_locative(
    sample: dict[str, Any],
    node_index: int,
    tokens: list[tuple[int, str, str]],
    targets: set[str],
) -> str | None:
    norms = [norm for _index, _surface, norm in tokens]
    be_pos = be_positions(norms)
    if not be_pos:
        return None
    first_be = min(be_pos)
    before_targets = [
        position
        for position, (_index, _surface, norm) in enumerate(tokens[:first_be])
        if norm in targets and norm in LOCATIVE_BEFORE_BE_MODIFIERS | EXISTENTIAL_BEFORE_BE_NOMINALS
    ]
    if not before_targets:
        return None
    target_pos = before_targets[0]
    target_norm = tokens[target_pos][2]
    target_index = tokens[target_pos][0]
    keep_indexes = [index for index, _surface, _norm in tokens if index != target_index]
    if not keep_indexes:
        return None
    relabel_postposed_arg1_to_arg2(sample, node_index, keep_indexes)
    set_node_token_indexes(sample, node_index, keep_indexes)
    if target_norm in EXISTENTIAL_BEFORE_BE_NOMINALS:
        existing_arg2 = edge_out(sample, node_index, "pred.arg.2")
        if not existing_arg2:
            relabel_postposed_arg1_to_arg2(sample, node_index, keep_indexes)
        remove_edges(
            sample,
            lambda edge, n=node_index: edge["start"] == n and edge["label"] == "pred.arg.1",
        )
        nominal = add_node(sample, [[target_index, target_index]], "NominalConstant")
        add_edge(sample, node_index, nominal, "pred.arg.1")
        return "before_be_existential_split"
    modifier = add_constant(sample, [target_index])
    add_edge(sample, modifier, node_index, "as:modification")
    transfer_incoming_to_new_node(sample, node_index, modifier, excluded_starts={modifier})
    return "before_be_locative_modifier_split"


def repair_fixed_pp_with_internal_body(
    sample: dict[str, Any],
    node_index: int,
    tokens: list[tuple[int, str, str]],
    prep_norm: str,
    body_norm: str,
) -> str | None:
    norms = [norm for _index, _surface, norm in tokens]
    for position, norm in enumerate(norms[:-1]):
        if norm == prep_norm and norms[position + 1] == body_norm:
            prep_index = tokens[position][0]
            body_index = tokens[position + 1][0]
            keep_indexes = [index for index, _surface, _norm in tokens if index not in {prep_index, body_index}]
            if not keep_indexes:
                return None
            body = add_node(sample, [[body_index, body_index]], "NominalConstant")
            split_as_functor(sample, node_index, [prep_index], keep_indexes, body)
            return f"fixed_{prep_norm}_{body_norm}_split"
    return None


def repair_directional_coordination(
    sample: dict[str, Any],
    node_index: int,
    tokens: list[tuple[int, str, str]],
    targets: set[str],
) -> str | None:
    norms = [norm for _index, _surface, norm in tokens]
    positions = target_positions(tokens, targets)
    if not positions:
        return None
    start = min(positions)
    end = max(positions)
    while start > 0 and norms[start - 1] in FUNCTION_JOINERS:
        start -= 1
    while end + 1 < len(tokens) and norms[end + 1] in FUNCTION_JOINERS | targets:
        end += 1
    modifier_indexes = contiguous_indexes(tokens, start, end)
    keep_indexes = [index for index, _surface, _norm in tokens if index not in set(modifier_indexes)]
    if not keep_indexes:
        return None
    arg2_edges = edge_out(sample, node_index, "pred.arg.2")
    if len(arg2_edges) == 1:
        split_as_functor(sample, node_index, modifier_indexes, keep_indexes, arg2_edges[0]["end"])
        return "directional_coordination_functor_split"
    split_as_constant_modifier(sample, node_index, modifier_indexes, keep_indexes)
    return "directional_coordination_constant_split"


def repair_fixed_up_close(
    sample: dict[str, Any],
    node_index: int,
    tokens: list[tuple[int, str, str]],
) -> str | None:
    norms = [norm for _index, _surface, norm in tokens]
    if "up" not in norms or "close" not in norms:
        return None
    start = norms.index("up")
    modifier_indexes = contiguous_indexes(tokens, start, len(tokens) - 1)
    keep_indexes = [index for index, _surface, _norm in tokens if index not in set(modifier_indexes)]
    if not keep_indexes:
        return None
    split_as_constant_modifier(sample, node_index, modifier_indexes, keep_indexes)
    return "fixed_adverbial_up_close_split"


def repair_body_functor(
    sample: dict[str, Any],
    node_index: int,
    tokens: list[tuple[int, str, str]],
    targets: set[str],
) -> str | None:
    arg2_edges = edge_out(sample, node_index, "pred.arg.2")
    if len(arg2_edges) != 1:
        return None
    norms = [norm for _index, _surface, norm in tokens]
    functor_positions = [
        position
        for position, norm in enumerate(norms)
        if norm in targets and norm in FUNCTOR_TOKENS
    ]
    if not functor_positions:
        return None
    functor_pos = max(functor_positions)
    prefix_positions = [
        position
        for position in range(functor_pos)
        if norms[position] in targets and norms[position] not in FUNCTOR_TOKENS
    ]
    if not prefix_positions:
        prefix_positions = [
            position
            for position in range(functor_pos)
            if norms[position] in targets and norms[position] in BE_COMPLEMENT_TOKENS
        ]
    removed_indexes = [tokens[functor_pos][0]] + [tokens[position][0] for position in prefix_positions]
    keep_indexes = [index for index, _surface, _norm in tokens if index not in set(removed_indexes)]
    if not keep_indexes:
        return None

    variable_node: int | None = None
    prefix_nodes: set[int] = set()
    if prefix_positions:
        current_target = node_index
        for position in reversed(prefix_positions):
            modifier = add_constant(sample, [tokens[position][0]])
            add_edge(sample, modifier, current_target, "as:modification")
            prefix_nodes.add(modifier)
            current_target = modifier
        variable_node = current_target
    split_as_functor(
        sample,
        node_index,
        [tokens[functor_pos][0]],
        keep_indexes,
        arg2_edges[0]["end"],
        variable_node=variable_node,
        excluded_starts=prefix_nodes,
    )
    return "body_functor_split"


def repair_be_complement(
    sample: dict[str, Any],
    node_index: int,
    tokens: list[tuple[int, str, str]],
    targets: set[str],
) -> str | None:
    norms = [norm for _index, _surface, norm in tokens]
    be_pos = be_positions(norms)
    if not be_pos or edge_out(sample, node_index, "pred.arg.2"):
        return None
    last_be = max(be_pos)
    complement_positions = [
        position
        for position in range(last_be + 1, len(tokens))
        if not core_predicate_token(norms[position])
    ]
    if not complement_positions:
        return None
    if not any(norms[position] in targets for position in complement_positions):
        return None
    start = min(complement_positions)
    complement_indexes = contiguous_indexes(tokens, start, len(tokens) - 1)
    keep_indexes = [index for index, _surface, _norm in tokens if index not in set(complement_indexes)]
    if not keep_indexes:
        return None
    split_as_be_complement(sample, node_index, complement_indexes, keep_indexes)
    return "be_complement_split"


def repair_directional_relation_without_be(
    sample: dict[str, Any],
    node_index: int,
    tokens: list[tuple[int, str, str]],
    targets: set[str],
) -> str | None:
    norms = [norm for _index, _surface, norm in tokens]
    direction_positions = [
        position
        for position, norm in enumerate(norms)
        if norm in targets and norm in DIRECTIONAL_RELATION_TOKENS
    ]
    if len(direction_positions) != 1:
        return None
    direction_pos = direction_positions[0]
    direction_index = tokens[direction_pos][0]
    degree_indexes = [index for index, _surface, norm in tokens if index != direction_index and norm != "'"]
    degree_norms = [words(sample)[index].strip().lower().replace("’", "'") for index in degree_indexes]
    if not degree_indexes or not all(norm.isdigit() or norm in SMALL_DEGREE_TOKENS for norm in degree_norms):
        return None

    graph(sample)["nodes"][node_index]["type"] = "ModificationalFunctor"
    set_node_token_indexes(sample, node_index, [direction_index])
    for edge in edge_out(sample, node_index):
        if edge["label"] == "pred.arg.1":
            edge["label"] = "variable"
        elif edge["label"] == "pred.arg.2":
            edge["label"] = "body"
    degree = add_constant(sample, degree_indexes)
    add_edge(sample, degree, node_index, "as:modification")
    transfer_incoming_to_new_node(sample, node_index, degree, excluded_starts={degree})
    return "directional_relation_without_be_split"


def repair_constant_modifier(
    sample: dict[str, Any],
    node_index: int,
    tokens: list[tuple[int, str, str]],
    targets: set[str],
) -> str | None:
    positions = target_positions(tokens, targets)
    if not positions:
        return None
    start = min(positions)
    end = max(positions)
    norms = [norm for _index, _surface, norm in tokens]
    if len(positions) > 1:
        while start > 0 and norms[start - 1] in FUNCTION_JOINERS:
            start -= 1
        while end + 1 < len(tokens) and norms[end + 1] in FUNCTION_JOINERS | targets:
            end += 1
    modifier_indexes = contiguous_indexes(tokens, start, end)
    keep_indexes = [index for index, _surface, _norm in tokens if index not in set(modifier_indexes)]
    if not keep_indexes:
        return convert_single_predicator_relation(sample, node_index)
    if len(keep_indexes) == 1:
        kept_norm = words(sample)[keep_indexes[0]].strip().lower().replace("’", "'")
        if kept_norm.isdigit() or kept_norm in {"one", "two", "three"}:
            repaired = repair_directional_relation_without_be(sample, node_index, tokens, targets)
            if repaired:
                return repaired
            return None
    split_as_constant_modifier(sample, node_index, modifier_indexes, keep_indexes)
    return "constant_modifier_split"


def repair_node(sample: dict[str, Any], node_index: int, rows: list[dict[str, str]]) -> tuple[str | None, str | None]:
    if node_index >= len(graph(sample)["nodes"]):
        return None, "node_id_out_of_range"
    if graph(sample)["nodes"][node_index].get("type") != "FactualPredicator":
        return None, "node_not_factual_predicator"
    tokens = node_tokens(sample, node_index)
    if not tokens:
        return None, "node_has_no_token_spans"
    targets = {row["token"] for row in rows}
    reasons = {row["reason"] for row in rows}

    all_indexes = [index for index, _surface, _norm in tokens]
    target_idx = [index for index, _surface, norm in tokens if norm in targets]
    if set(all_indexes) == set(target_idx):
        converted = convert_single_predicator_relation(sample, node_index)
        if converted:
            return converted, None

    if "fixed_pp_on_board_not_particle" in reasons:
        repaired = repair_fixed_pp_with_internal_body(sample, node_index, tokens, "on", "board")
        if repaired:
            return repaired, None

    if "locative_or_state_word_not_particle" in reasons and "under" in targets:
        repaired = repair_fixed_pp_with_internal_body(sample, node_index, tokens, "under", "way")
        if repaired:
            return repaired, None

    if "fixed_adverbial_up_close_not_particle" in reasons:
        repaired = repair_fixed_up_close(sample, node_index, tokens)
        if repaired:
            return repaired, None

    if "directional_coordination_not_particle" in reasons:
        repaired = repair_directional_coordination(sample, node_index, tokens, targets)
        if repaired:
            return repaired, None

    repaired = repair_before_be_locative(sample, node_index, tokens, targets)
    if repaired:
        return repaired, None

    repaired = repair_body_functor(sample, node_index, tokens, targets)
    if repaired:
        return repaired, None

    repaired = repair_be_complement(sample, node_index, tokens, targets)
    if repaired:
        return repaired, None

    repaired = repair_constant_modifier(sample, node_index, tokens, targets)
    if repaired:
        return repaired, None

    return None, "no_safe_repair_rule"


def load_audit_groups(path: Path) -> dict[tuple[str, int, int], list[dict[str, str]]]:
    groups: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("bucket") != "split_non_particle":
                continue
            groups[(row["split"], int(row["line_no"]), int(row["node_id"]))].append(row)
    return groups


def write_report(
    report_dir: Path,
    args: argparse.Namespace,
    changes: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    counts: Counter[str],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "changes.jsonl").write_text(
        "\n".join(json.dumps(change, ensure_ascii=False) for change in changes)
        + ("\n" if changes else ""),
        encoding="utf-8",
    )
    (report_dir / "reviews.jsonl").write_text(
        "\n".join(json.dumps(review, ensure_ascii=False) for review in reviews)
        + ("\n" if reviews else ""),
        encoding="utf-8",
    )
    summary = {
        "data_dir": str(args.data_dir),
        "audit_tsv": str(args.audit_tsv),
        "report_dir": str(report_dir),
        "apply": args.apply,
        "changed_samples": len(changes),
        "review_samples": len(reviews),
        "counts": dict(counts),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# FactualPredicator Non-Particle Split Repair",
        "",
        f"- data_dir: `{args.data_dir}`",
        f"- audit_tsv: `{args.audit_tsv}`",
        f"- apply: `{args.apply}`",
        f"- changed samples: {len(changes)}",
        f"- review samples: {len(reviews)}",
        "",
        "## Counts",
        "",
        "| key | count |",
        "|---|---:|",
    ]
    for key, count in sorted(counts.items()):
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## First 40 Changes", ""])
    for change in changes[:40]:
        lines.extend(
            [
                f"### {change['split']}:{change['line_number']}",
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
    lines.extend(["", "## Reviews", ""])
    for review in reviews[:80]:
        lines.extend(
            [
                f"### {review['split']}:{review['line_number']} node {review['node_id']} `{review['reason']}`",
                "",
                review["sentence"],
                "",
                "```text",
                review["graph"],
                "```",
                "",
            ]
        )
    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run(args: argparse.Namespace) -> dict[str, Any]:
    groups = load_audit_groups(args.audit_tsv)
    by_sample: dict[tuple[str, int], dict[int, list[dict[str, str]]]] = defaultdict(dict)
    for (split, line_no, node_id), rows in groups.items():
        by_sample[(split, line_no)][node_id] = rows

    counts: Counter[str] = Counter()
    changes: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []

    for split in ("train", "dev", "test"):
        rows = read_jsonl(args.data_dir / f"{split}.jsonl")
        output_rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(rows, 1):
            node_groups = by_sample.get((split, line_number))
            if not node_groups:
                output_rows.append(row)
                continue
            before = deepcopy(row)
            after = deepcopy(row)
            sample_changes: list[dict[str, Any]] = []
            sample_reviews: list[dict[str, Any]] = []
            for node_id in sorted(node_groups):
                node_before = node_text(after, node_id) if node_id < len(graph(after)["nodes"]) else "<missing>"
                kind, reason = repair_node(after, node_id, node_groups[node_id])
                if kind:
                    counts[kind] += 1
                    counts[f"{split}:{kind}"] += 1
                    sample_changes.append(
                        {
                            "node_id": node_id,
                            "before": node_before,
                            "after": node_text(after, node_id) if node_id < len(graph(after)["nodes"]) else "<missing>",
                            "kind": kind,
                            "tokens": sorted({row["token"] for row in node_groups[node_id]}),
                            "reasons": sorted({row["reason"] for row in node_groups[node_id]}),
                        }
                    )
                else:
                    counts["review"] += 1
                    counts[f"{split}:review"] += 1
                    counts[f"review_reason:{reason}"] += 1
                    sample_reviews.append(
                        {
                            "node_id": node_id,
                            "before": node_before,
                            "reason": reason,
                            "tokens": sorted({row["token"] for row in node_groups[node_id]}),
                            "reasons": sorted({row["reason"] for row in node_groups[node_id]}),
                        }
                    )
            if sample_changes:
                changes.append(
                    {
                        "split": split,
                        "line_number": line_number,
                        "sentence": row.get("sentence", ""),
                        "changes": sample_changes,
                        "before": compact_graph(before),
                        "after": compact_graph(after),
                    }
                )
            if sample_reviews:
                reviews.append(
                    {
                        "split": split,
                        "line_number": line_number,
                        "sentence": row.get("sentence", ""),
                        "reviews": sample_reviews,
                        "node_id": ",".join(str(item["node_id"]) for item in sample_reviews),
                        "reason": ",".join(str(item["reason"]) for item in sample_reviews),
                        "graph": compact_graph(after),
                    }
                )
            output_rows.append(after if sample_changes else row)
        if args.apply:
            write_jsonl(args.data_dir / f"{split}.jsonl", output_rows)

    write_report(args.report_dir, args, changes, reviews, counts)
    return {
        "changed_samples": len(changes),
        "review_samples": len(reviews),
        "counts": dict(counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument(
        "--audit-tsv",
        type=Path,
        default=REPORTS / "factual_predicator_particle_filter" / "split_non_particle.tsv",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORTS / "factual_predicator_split_non_particle_repair_dryrun",
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
