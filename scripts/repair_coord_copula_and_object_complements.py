#!/usr/bin/env python3
"""Repair coordinated copular states and object-complement state frames.

This script normalizes two recoverable historical annotation patterns:

1. In `be + adj and adj`, the copular predicate points to a conjunction as its
   single `pred.arg.2`; all state conjuncts are parallel modification constants.
2. In `make/keep/leave/have + object + state`, the matrix predicate keeps the
   object as `pred.arg.2` and points to the state complement with `pred.arg.4`.

The repairs are intentionally graph-local and lexical-gated. Ambiguous mixed
coordination is fixed only when the existing graph already exposes the shared
subject/object and the state predicates to be demoted.
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
    ADVERBIAL_MODIFIERS,
    BE_FORMS,
    MODALS_AND_AUX,
    PARTICIPLE_STATE_WHITELIST,
    STATE_ADJECTIVES,
    add_edge,
    compact_graph,
    edge_in,
    edge_out,
    graph,
    is_factual,
    node_text,
    node_tokens,
    read_jsonl,
    remove_edges,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

MOD_CONST = "ModificationalConstant"
MOD_FUNCTOR = "ModificationalFunctor"
CONJ = "ConjunctionalFunctor"
FACTUAL = "FactualPredicator"

EXTRA_STATE_WORDS = {
    "accessible",
    "advised",
    "abrasive",
    "affordable",
    "alike",
    "attentive",
    "available",
    "barren",
    "better",
    "blank",
    "caring",
    "cold",
    "confidential",
    "conscientious",
    "conscious",
    "cooked",
    "courteous",
    "cozy",
    "creative",
    "cruel",
    "dark",
    "dead",
    "deep",
    "delicious",
    "delicous",
    "dingy",
    "dire",
    "dirty",
    "eager",
    "entertaining",
    "extreme",
    "fast",
    "fit",
    "fresh",
    "frustrated",
    "gracious",
    "healthy",
    "hot",
    "immediate",
    "ignorant",
    "informative",
    "inoperable",
    "insulting",
    "intended",
    "intoxicating",
    "involved",
    "knowledgeable",
    "limp",
    "loose",
    "marvelous",
    "mediocre",
    "monotonous",
    "neat",
    "nonjudgmental",
    "outdated",
    "overpriced",
    "patient",
    "personable",
    "prideful",
    "progressive",
    "relaxed",
    "rude",
    "polite",
    "running",
    "shedding",
    "smooth",
    "social",
    "stranded",
    "tasting",
    "tasty",
    "tasteless",
    "tolerant",
    "unsure",
    "up",
    "upset",
    "valid",
    "varied",
    "void",
    "welcoming",
    "worse",
}

STATE_WORDS = STATE_ADJECTIVES | PARTICIPLE_STATE_WHITELIST | EXTRA_STATE_WORDS

STATE_MODIFIER_WORDS = ADVERBIAL_MODIFIERS | {
    "all",
    "completely",
    "extremely",
    "highly",
    "incredibly",
    "little",
    "more",
    "most",
    "much",
    "not",
    "quite",
    "really",
    "so",
    "super",
    "too",
    "very",
}

MULTIWORD_STATE_EXCEPTIONS = {
    ("messed", "up"),
}

MATRIX_COMPLEMENT_VERBS = {
    "had",
    "has",
    "have",
    "having",
    "keep",
    "keeping",
    "keeps",
    "kept",
    "leave",
    "leaves",
    "leaving",
    "left",
    "made",
    "make",
    "makes",
    "making",
}

PRED_ARGS = {"pred.arg.1", "pred.arg.2", "pred.arg.3", "pred.arg.4"}


def node_type(sample: dict[str, Any], node_index: int) -> str:
    return graph(sample)["nodes"][node_index].get("type", "")


def set_node_type(sample: dict[str, Any], node_index: int, new_type: str) -> None:
    graph(sample)["nodes"][node_index]["type"] = new_type


def token_norms(sample: dict[str, Any], node_index: int) -> list[str]:
    return [norm for _index, _surface, norm in node_tokens(sample, node_index) if norm != "'"]


def token_indexes(sample: dict[str, Any], node_index: int) -> list[int]:
    return [index for index, _surface, _norm in node_tokens(sample, node_index)]


def min_token_index(sample: dict[str, Any], node_index: int) -> int:
    indexes = token_indexes(sample, node_index)
    return min(indexes) if indexes else 10**9


def add_edge_once(sample: dict[str, Any], start: int, end: int, label: str) -> bool:
    for edge in graph(sample)["edges"]:
        if edge["start"] == start and edge["end"] == end and edge["label"] == label:
            return False
    add_edge(sample, start, end, label)
    return True


def remove_exact_edges(sample: dict[str, Any], start: int, end: int, label: str) -> bool:
    before = len(graph(sample)["edges"])
    remove_edges(sample, lambda edge: edge["start"] == start and edge["end"] == end and edge["label"] == label)
    return len(graph(sample)["edges"]) != before


def dedupe_edges(sample: dict[str, Any]) -> None:
    seen: set[tuple[int, int, str]] = set()
    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        key = (edge["start"], edge["end"], edge["label"])
        if key in seen:
            continue
        seen.add(key)
        new_edges.append(edge)
    graph(sample)["edges"] = new_edges


def single_arg_target(sample: dict[str, Any], node_index: int, label: str) -> int | None:
    edges = edge_out(sample, node_index, label)
    if len(edges) != 1:
        return None
    return edges[0]["end"]


def has_same_arg1(sample: dict[str, Any], node_index: int, subject: int) -> bool:
    return any(edge["end"] == subject for edge in edge_out(sample, node_index, "pred.arg.1"))


def is_copular_be_predicate(sample: dict[str, Any], node_index: int) -> bool:
    if not is_factual(sample, node_index):
        return False
    norms = token_norms(sample, node_index)
    return bool(norms) and any(token in BE_FORMS for token in norms) and all(
        token in BE_FORMS or token in MODALS_AND_AUX for token in norms
    )


def is_state_predicate(sample: dict[str, Any], node_index: int, subject: int) -> bool:
    if node_type(sample, node_index) != FACTUAL:
        return False
    if not has_same_arg1(sample, node_index, subject):
        return False
    norms = token_norms(sample, node_index)
    if not norms:
        return False
    if len(norms) > 3:
        return False
    other_pred_args = [
        edge
        for edge in edge_out(sample, node_index)
        if edge["label"] in PRED_ARGS and not (edge["label"] == "pred.arg.1" and edge["end"] == subject)
    ]
    if any(edge["label"] in {"pred.arg.2", "pred.arg.3"} for edge in other_pred_args):
        return False
    head = norms[-1]
    if tuple(norms) not in MULTIWORD_STATE_EXCEPTIONS and head not in STATE_WORDS:
        return False
    if len(norms) > 1 and tuple(norms) not in MULTIWORD_STATE_EXCEPTIONS:
        if not all(token in STATE_WORDS or token in STATE_MODIFIER_WORDS for token in norms):
            return False
    return True


def modifier_target(sample: dict[str, Any], node_index: int) -> int | None:
    targets = edge_out(sample, node_index, "as:modification")
    if len(targets) != 1:
        return None
    return targets[0]["end"]


def child_nodes(sample: dict[str, Any], conj: int) -> list[int]:
    return [edge["end"] for edge in edge_out(sample, conj, "func.arg")]


def rewire_modifier_target(sample: dict[str, Any], modifier: int, old_target: int, new_target: int) -> bool:
    changed = False
    for edge in graph(sample)["edges"]:
        if edge["start"] == modifier and edge["end"] == old_target and edge["label"] == "as:modification":
            edge["end"] = new_target
            changed = True
    return changed


def transfer_incoming_to_copula(sample: dict[str, Any], conj: int, be_node: int) -> bool:
    """Move parents of an internal complement conjunction to the copula."""
    changed = False
    for edge in graph(sample)["edges"]:
        if edge["end"] == conj and edge["start"] != be_node and edge["label"] in {"func.arg", "as:punctuation"}:
            edge["end"] = be_node
            changed = True
    return changed


def collect_copular_state_items(
    sample: dict[str, Any],
    conj: int,
    be_node: int,
    subject: int,
) -> tuple[dict[int, int], set[int], int | None]:
    """Return state->root, roots to remove from conj, and base modifier root."""
    children = child_nodes(sample, conj)
    state_roots: dict[int, int] = {}
    direct_state_children: set[int] = set()
    remove_child_roots: set[int] = set()
    base_modifier_root: int | None = None

    for child in children:
        child_type = node_type(sample, child)
        if is_state_predicate(sample, child, subject):
            state_roots.setdefault(child, child)
            direct_state_children.add(child)
            continue

        if child_type == MOD_CONST:
            target = modifier_target(sample, child)
            if target == be_node:
                base_modifier_root = child
            elif target is not None and is_state_predicate(sample, target, subject):
                state_roots[target] = child
                if target in direct_state_children:
                    remove_child_roots.add(target)
            continue

        if child_type == MOD_FUNCTOR:
            variables = edge_out(sample, child, "variable")
            if len(variables) != 1:
                continue
            variable = variables[0]["end"]
            if is_state_predicate(sample, variable, subject):
                state_roots[variable] = child
                if variable in direct_state_children:
                    remove_child_roots.add(variable)
                continue
            if node_type(sample, variable) == MOD_CONST:
                target = modifier_target(sample, variable)
                if target is not None and is_state_predicate(sample, target, subject):
                    state_roots[target] = child
                    if target in direct_state_children:
                        remove_child_roots.add(target)
                    continue
            if variable == be_node:
                direct_states = [
                    state
                    for state in direct_state_children
                    if min_token_index(sample, state) < min_token_index(sample, child)
                ]
                if direct_states:
                    state = max(direct_states, key=lambda item: min_token_index(sample, item))
                    state_roots[state] = child
                    remove_child_roots.add(state)
                    variables[0]["end"] = state

    return state_roots, remove_child_roots, base_modifier_root


def demote_state_predicate(sample: dict[str, Any], state: int, subject: int) -> bool:
    changed = node_type(sample, state) != MOD_CONST
    set_node_type(sample, state, MOD_CONST)
    removed = remove_exact_edges(sample, state, subject, "pred.arg.1")
    return changed or removed


def repair_copular_coordination(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    original_node_count = len(graph(sample)["nodes"])

    for be_node in range(original_node_count):
        if not is_copular_be_predicate(sample, be_node):
            continue
        subject = single_arg_target(sample, be_node, "pred.arg.1")
        base = single_arg_target(sample, be_node, "pred.arg.2")
        if subject is None or base is None:
            continue
        if node_type(sample, base) == CONJ:
            continue

        best: tuple[int, dict[int, int], set[int], int | None, int] | None = None
        for conj, node in enumerate(graph(sample)["nodes"][:original_node_count]):
            if node.get("type") != CONJ:
                continue
            state_roots, remove_roots, base_modifier = collect_copular_state_items(sample, conj, be_node, subject)
            if not state_roots:
                continue
            children = set(child_nodes(sample, conj))
            score = len(state_roots)
            if be_node in children:
                score += 2
            if base_modifier is not None and base_modifier in children:
                score += 2
            if any(root in children for root in state_roots.values()):
                score += 1
            if best is None or score > best[4]:
                best = (conj, state_roots, remove_roots, base_modifier, score)

        if best is None:
            continue

        conj, state_roots, remove_roots, base_modifier, _score = best
        base_root = base
        if base_modifier is not None:
            if rewire_modifier_target(sample, base_modifier, be_node, base):
                changes.append("copular_coord_base_modifier_scope")
            base_root = base_modifier

        state_root_set = set(state_roots.values())
        allowed_children = state_root_set | {base_root, be_node}
        children = set(child_nodes(sample, conj))
        other_children = children - allowed_children - remove_roots

        if other_children:
            inner_conj = len(graph(sample)["nodes"])
            graph(sample)["nodes"].append({"spans": "Parataxis", "type": CONJ})
            for root in state_root_set | {base_root} | remove_roots:
                remove_exact_edges(sample, conj, root, "func.arg")
            add_edge_once(sample, conj, be_node, "func.arg")
            target_conj = inner_conj
            changes.append("copular_coord_split_inner_conjunction")
        else:
            target_conj = conj
            remove_exact_edges(sample, conj, be_node, "func.arg")
            if transfer_incoming_to_copula(sample, conj, be_node):
                changes.append("copular_coord_parent_to_copula")

        for edge in edge_out(sample, be_node, "pred.arg.2"):
            if edge["end"] == base:
                edge["end"] = target_conj
        add_edge_once(sample, be_node, target_conj, "pred.arg.2")
        add_edge_once(sample, target_conj, base_root, "func.arg")

        for state, root in state_roots.items():
            demote_state_predicate(sample, state, subject)
            add_edge_once(sample, target_conj, root, "func.arg")
            if root != state:
                remove_exact_edges(sample, target_conj, state, "func.arg")
            for remove_root in remove_roots:
                if remove_root != root:
                    remove_exact_edges(sample, target_conj, remove_root, "func.arg")

        changes.append("copular_coord_state_conjuncts_demoted")

    if changes:
        dedupe_edges(sample)
    return changes


def cleanup_copular_arg2_conjunctions(sample: dict[str, Any]) -> list[str]:
    """Repair older runs of this script and normalize existing copula parents."""
    changes: list[str] = []
    seen_spans: dict[str, int] = {}
    for node_index, node in enumerate(graph(sample)["nodes"]):
        spans = node.get("spans")
        if isinstance(spans, str):
            continue
        key = json.dumps(spans, separators=(",", ":"))
        if key in seen_spans and node.get("type") == CONJ:
            incoming_be = [
                edge["start"]
                for edge in edge_in(sample, node_index, "pred.arg.2")
                if is_copular_be_predicate(sample, edge["start"])
            ]
            if incoming_be:
                node["spans"] = "Parataxis"
                changes.append("cleanup_inner_conjunction_aux_span")
        else:
            seen_spans[key] = node_index

    for be_node in range(len(graph(sample)["nodes"])):
        if not is_copular_be_predicate(sample, be_node):
            continue
        arg2 = single_arg_target(sample, be_node, "pred.arg.2")
        if arg2 is None or node_type(sample, arg2) != CONJ:
            continue
        if transfer_incoming_to_copula(sample, arg2, be_node):
            changes.append("cleanup_copular_parent_to_copula")

    if changes:
        dedupe_edges(sample)
    return changes


def cleanup_object_complement_wrappers(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    indegree = [0] * len(graph(sample)["nodes"])
    for edge in graph(sample)["edges"]:
        indegree[edge["end"]] += 1

    for matrix in range(len(graph(sample)["nodes"])):
        if not is_matrix_object_complement_predicate(sample, matrix):
            continue
        for arg4_edge in edge_out(sample, matrix, "pred.arg.4"):
            state = arg4_edge["end"]
            if node_type(sample, state) != MOD_CONST:
                continue
            wrappers = [
                wrapper
                for wrapper, node in enumerate(graph(sample)["nodes"])
                if node.get("type") == MOD_FUNCTOR
                and indegree[wrapper] == 0
                and any(edge["end"] == state for edge in edge_out(sample, wrapper, "variable"))
            ]
            if not wrappers:
                continue
            wrapper = min(wrappers, key=lambda item: min_token_index(sample, item))
            arg4_edge["end"] = wrapper
            changes.append("cleanup_object_complement_wrapper_arg4_root")

    if changes:
        dedupe_edges(sample)
    return changes


def is_matrix_object_complement_predicate(sample: dict[str, Any], node_index: int) -> bool:
    if not is_factual(sample, node_index):
        return False
    return any(token in MATRIX_COMPLEMENT_VERBS for token in token_norms(sample, node_index))


def demote_object_state(sample: dict[str, Any], state: int, obj: int) -> bool:
    if not is_state_predicate(sample, state, obj):
        return False
    demote_state_predicate(sample, state, obj)
    return True


def repair_matrix_arg2_state(
    sample: dict[str, Any],
    matrix: int,
    state: int,
    object_node: int,
) -> bool:
    if not demote_object_state(sample, state, object_node):
        return False
    for edge in edge_out(sample, matrix, "pred.arg.2"):
        if edge["end"] == state:
            edge["end"] = object_node
    add_edge_once(sample, matrix, object_node, "pred.arg.2")
    add_edge_once(sample, matrix, state, "pred.arg.4")
    return True


def collect_conj_object_states(
    sample: dict[str, Any],
    conj: int,
) -> tuple[int, dict[int, int], set[int]] | None:
    state_roots: dict[int, int] = {}
    remove_roots: set[int] = set()
    object_node: int | None = None

    for child in child_nodes(sample, conj):
        candidates: list[tuple[int, int]] = []
        if node_type(sample, child) == FACTUAL:
            candidates.append((child, child))
        elif node_type(sample, child) == MOD_CONST:
            target = modifier_target(sample, child)
            if target is not None:
                candidates.append((target, child))
        elif node_type(sample, child) == MOD_FUNCTOR:
            variables = edge_out(sample, child, "variable")
            if len(variables) == 1:
                variable = variables[0]["end"]
                candidates.append((variable, child))
                if node_type(sample, variable) == MOD_CONST:
                    target = modifier_target(sample, variable)
                    if target is not None:
                        candidates.append((target, child))

        for state, root in candidates:
            arg1s = [edge["end"] for edge in edge_out(sample, state, "pred.arg.1")]
            if len(arg1s) != 1:
                continue
            obj = arg1s[0]
            if not is_state_predicate(sample, state, obj):
                continue
            if object_node is None:
                object_node = obj
            elif object_node != obj:
                return None
            state_roots[state] = root
            if root != state:
                remove_roots.add(state)
            break

    if object_node is None or not state_roots:
        return None
    return object_node, state_roots, remove_roots


def repair_object_complements(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    original_node_count = len(graph(sample)["nodes"])

    for matrix in range(original_node_count):
        if not is_matrix_object_complement_predicate(sample, matrix):
            continue
        matrix_subjects = {edge["end"] for edge in edge_out(sample, matrix, "pred.arg.1")}

        for arg2_edge in list(edge_out(sample, matrix, "pred.arg.2")):
            arg2 = arg2_edge["end"]
            if node_type(sample, arg2) == FACTUAL:
                object_node = single_arg_target(sample, arg2, "pred.arg.1")
                if object_node is None or object_node in matrix_subjects:
                    continue
                if repair_matrix_arg2_state(sample, matrix, arg2, object_node):
                    changes.append("object_complement_arg2_state_to_arg4")
                continue

            if node_type(sample, arg2) == CONJ:
                collected = collect_conj_object_states(sample, arg2)
                if collected is None:
                    continue
                object_node, state_roots, remove_roots = collected
                if object_node in matrix_subjects:
                    continue
                arg2_edge["end"] = object_node
                add_edge_once(sample, matrix, object_node, "pred.arg.2")
                add_edge_once(sample, matrix, arg2, "pred.arg.4")
                for state in state_roots:
                    demote_object_state(sample, state, object_node)
                for remove_root in remove_roots:
                    remove_exact_edges(sample, arg2, remove_root, "func.arg")
                changes.append("object_complement_arg2_conjunction_to_arg4")

        for arg4_edge in list(edge_out(sample, matrix, "pred.arg.4")):
            arg4 = arg4_edge["end"]
            if node_type(sample, arg4) != MOD_FUNCTOR:
                continue
            variables = edge_out(sample, arg4, "variable")
            if len(variables) != 1:
                continue
            state = variables[0]["end"]
            object_node = single_arg_target(sample, state, "pred.arg.1")
            if object_node is None or object_node in matrix_subjects:
                continue
            if not demote_object_state(sample, state, object_node):
                continue
            if not any(edge["end"] == object_node for edge in edge_out(sample, matrix, "pred.arg.2")):
                add_edge_once(sample, matrix, object_node, "pred.arg.2")
            changes.append("object_complement_arg4_wrapper_state_demoted")

    if changes:
        dedupe_edges(sample)
    return changes


def repair_sample(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    changes.extend(repair_copular_coordination(sample))
    changes.extend(repair_object_complements(sample))
    changes.extend(cleanup_copular_arg2_conjunctions(sample))
    changes.extend(cleanup_object_complement_wrappers(sample))
    return changes


def run(apply: bool, data_dir: Path, report_dir: Path) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    case_dir = report_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    all_changes: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        new_rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(rows, 1):
            before = deepcopy(row)
            after = deepcopy(row)
            changes = repair_sample(after)
            if changes:
                for change in changes:
                    counts[f"{split}:{change}"] += 1
                    counts[change] += 1
                record = {
                    "split": split,
                    "line": line_number,
                    "sentence": row["sentence"],
                    "changes": changes,
                    "before": compact_graph(before),
                    "after": compact_graph(after),
                }
                all_changes.append(record)
                case_path = case_dir / f"{split}_{line_number:05d}.txt"
                case_path.write_text(
                    f"{split}:{line_number}\n{row['sentence']}\n\n"
                    f"CHANGES: {', '.join(changes)}\n\n"
                    f"BEFORE\n{record['before']}\n\nAFTER\n{record['after']}\n",
                    encoding="utf-8",
                )
            new_rows.append(after if changes else row)

        if apply:
            write_jsonl(path, new_rows)

    (report_dir / "changes.jsonl").write_text(
        "\n".join(json.dumps(change, ensure_ascii=False) for change in all_changes) + ("\n" if all_changes else ""),
        encoding="utf-8",
    )
    summary_lines = [
        "# Coordinated Copula and Object Complement Repair",
        "",
        f"- apply: {apply}",
        f"- changed samples: {len(all_changes)}",
        "",
        "## Counts",
    ]
    for key, value in sorted(counts.items()):
        summary_lines.append(f"- {key}: {value}")
    (report_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--report-dir", type=Path, default=REPORTS / "coord_copula_object_complement_repair")
    args = parser.parse_args()
    return run(args.apply, args.data_dir, args.report_dir)


if __name__ == "__main__":
    raise SystemExit(main())
