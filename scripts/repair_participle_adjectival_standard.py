#!/usr/bin/env python3
"""Repair strict participial adjective vs eventive participle standards.

This pass is intentionally narrow:

1. Split merged state predicates such as ``is interested`` when the state head
   is an adjectival ``-ing/-ed/-en`` participle.
2. Relabel leaf adjectival participles that were modeled as
   ``FactualPredicator(pred.arg.1=head)`` into direct
   ``ModificationalConstant --as:modification--> head``.

Eventive or complement-taking participles are reported for review instead of
being rewritten automatically.
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

from repair_be_copula_rules import (  # noqa: E402
    ADVERBIAL_MODIFIERS,
    BE_FORMS,
    MODALS_AND_AUX,
    add_edge,
    add_modifier_chain,
    add_node,
    compact_graph,
    edge_in,
    edge_out,
    grouped_modifier_spans,
    graph,
    node_tokens,
    node_text,
    read_jsonl,
    remove_nodes,
    set_node_token_indexes,
    spans_from_token_indexes,
    transfer_incoming_to_new_node,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
DEFAULT_REPORT_DIR = REPORTS / "participle_adjectival_standard_repair"

PARTICIPLE_RE = re.compile(r"(ing|ed|en)$", re.IGNORECASE)
ADJ_XPOS = {"JJ", "JJR", "JJS"}

# Common proper-name pieces that UD often tags as ADJ/JJ but should not be
# treated as participial adjectives by this pass.
PROPER_NAME_SKIP = {
    "alien",
    "armed",
    "civilized",
    "golden",
    "green",
    "open",
    "red",
    "reformed",
    "rotten",
    "united",
    "warped",
    "zen",
}

CONTENT_CHILD_DEPRELS = {
    "advcl",
    "ccomp",
    "csubj",
    "iobj",
    "nsubj",
    "nsubj:pass",
    "obj",
    "obl",
    "xcomp",
}

STATE_PREDICATE_WORDS = BE_FORMS | MODALS_AND_AUX | {
    "appear",
    "appeared",
    "appears",
    "became",
    "become",
    "becomes",
    "feel",
    "feels",
    "felt",
    "get",
    "gets",
    "got",
    "keep",
    "keeps",
    "kept",
    "look",
    "looked",
    "looks",
    "remain",
    "remained",
    "remains",
    "seem",
    "seemed",
    "seems",
    "sound",
    "sounded",
    "sounds",
    "stay",
    "stayed",
    "stays",
}

STATE_CHILD_REVIEW_DEPRELS = {
    "advcl",
    "ccomp",
    "csubj",
    "obl",
    "xcomp",
}


def node_token_indexes(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if isinstance(spans, str):
        return []
    indexes: list[int] = []
    for span in spans:
        if isinstance(span, list) and len(span) == 2:
            indexes.extend(range(int(span[0]), int(span[1]) + 1))
    return sorted(set(indexes))


def token_nodes(sample: dict[str, Any], token_index: int) -> tuple[list[int], list[int]]:
    exact: list[int] = []
    containing: list[int] = []
    for node_index, node in enumerate(graph(sample)["nodes"]):
        indexes = node_token_indexes(node)
        if token_index not in indexes:
            continue
        containing.append(node_index)
        if indexes == [token_index]:
            exact.append(node_index)
    return exact, containing


def is_participle_surface(text: str, xpos: str) -> bool:
    return xpos in {"VBG", "VBN"} or bool(PARTICIPLE_RE.search(text))


def is_adjectival_participle(ud_word: dict[str, Any]) -> bool:
    text = str(ud_word.get("text") or "")
    lower = text.lower()
    xpos = str(ud_word.get("xpos") or "")
    upos = str(ud_word.get("upos") or "")
    if lower in PROPER_NAME_SKIP:
        return False
    if not is_participle_surface(lower, xpos):
        return False
    return upos == "ADJ" or xpos in ADJ_XPOS


def ud_children_by_index(ud_row: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    children: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sentences = ud_row.get("sentences") or []
    if not sentences:
        return children
    for word in sentences[0].get("words") or []:
        head = word.get("head")
        if isinstance(head, int) and head > 0:
            children[head - 1].append(word)
    return children


def get_ud_words(ud_row: dict[str, Any]) -> list[dict[str, Any]]:
    sentences = ud_row.get("sentences") or []
    if not sentences:
        return []
    return list(sentences[0].get("words") or [])


def add_edge_once(sample: dict[str, Any], start: int, end: int, label: str, template: dict[str, Any] | None = None) -> None:
    if start == end:
        return
    edges = graph(sample).setdefault("edges", [])
    for edge in edges:
        if edge.get("start") == start and edge.get("end") == end and edge.get("label") == label:
            return
    new_edge = {"start": int(start), "end": int(end), "label": label}
    if template:
        for key, value in template.items():
            if key not in new_edge and key not in {"start", "end", "label"}:
                new_edge[key] = value
    edges.append(new_edge)


def remove_edges_by_indexes(sample: dict[str, Any], indexes: set[int]) -> None:
    if not indexes:
        return
    graph(sample)["edges"] = [
        edge for index, edge in enumerate(graph(sample).get("edges", [])) if index not in indexes
    ]


def simple_modification_aux_for_body(sample: dict[str, Any], body: int, variable: int) -> int | None:
    nodes = graph(sample)["nodes"]
    candidates: list[int] = []
    for node_index, node in enumerate(nodes):
        if node.get("type") != "ModificationalFunctor" or node.get("spans") != "Modification":
            continue
        outs = edge_out(sample, node_index)
        core = {(edge["label"], edge["end"]) for edge in outs}
        if core == {("body", body), ("variable", variable)}:
            candidates.append(node_index)
    if len(candidates) == 1:
        return candidates[0]
    return None


def wrapping_relation_for_adjective(sample: dict[str, Any], adjective_node: int, head_node: int) -> int | None:
    """Return a relation node that already wraps adjective_node as modifier body."""
    for incoming in edge_in(sample, adjective_node):
        if incoming.get("label") not in {"variable", "body"}:
            continue
        relation = int(incoming["start"])
        if relation == adjective_node:
            continue
        aux = simple_modification_aux_for_body(sample, relation, head_node)
        if aux is not None:
            return relation
    return None


def collapse_simple_modification_aux(sample: dict[str, Any], aux: int, body: int) -> dict[str, Any]:
    """Remove a simple Modification aux and rewire incoming edges to body."""
    rewired: list[dict[str, Any]] = []
    for edge in list(edge_in(sample, aux)):
        add_edge_once(sample, edge["start"], body, edge["label"], template=edge)
        rewired.append(dict(edge))
    removed_edges = {
        index
        for index, edge in enumerate(graph(sample).get("edges", []))
        if edge["start"] == aux or edge["end"] == aux
    }
    remove_edges_by_indexes(sample, removed_edges)
    remove_nodes(sample, {aux})
    return {"removed_aux": aux, "rewired_incoming": rewired}


def repair_leaf_adjectival_fp(
    sample: dict[str, Any],
    ud_words: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    changes: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    nodes = graph(sample)["nodes"]

    # Iterate by token rather than node so UD alignment gates every change.
    for token_index, ud_word in enumerate(ud_words):
        if not is_adjectival_participle(ud_word):
            continue
        exact, _containing = token_nodes(sample, token_index)
        fp_nodes = [node_index for node_index in exact if nodes[node_index].get("type") == "FactualPredicator"]
        if len(fp_nodes) != 1:
            continue
        node_index = fp_nodes[0]
        if node_index >= len(graph(sample)["nodes"]):
            continue
        pred_edges = [
            (edge_index, edge)
            for edge_index, edge in enumerate(graph(sample).get("edges", []))
            if edge["start"] == node_index and str(edge["label"]).startswith("pred.arg.")
        ]

        if not pred_edges:
            graph(sample)["nodes"][node_index]["type"] = "ModificationalConstant"
            changes.append(
                {
                    "kind": "leaf_adjectival_participle_fp_relabel_no_pred_edges",
                    "token_index": token_index,
                    "text": ud_word.get("text"),
                    "node_index": node_index,
                }
            )
            continue

        if len(pred_edges) == 1 and pred_edges[0][1]["label"] == "pred.arg.1":
            edge_index, pred_edge = pred_edges[0]
            variable = int(pred_edge["end"])
            aux = simple_modification_aux_for_body(sample, node_index, variable)
            before_text = node_text(sample, node_index)
            wrapping_relation = wrapping_relation_for_adjective(sample, node_index, variable)
            has_relation_complement = any(
                incoming.get("label") in {"variable", "body"}
                and graph(sample)["nodes"][int(incoming["start"])].get("type") == "ModificationalFunctor"
                for incoming in edge_in(sample, node_index)
            )
            if token_index == 0 and has_relation_complement and wrapping_relation is None and aux is None:
                reviews.append(
                    {
                        "kind": "preposed_adjectival_participle_phrase_review",
                        "token_index": token_index,
                        "text": ud_word.get("text"),
                        "node_index": node_index,
                        "node_text": before_text,
                        "pred_edges": [edge for _idx, edge in pred_edges],
                        "ud": ud_word,
                    }
                )
                continue
            graph(sample)["nodes"][node_index]["type"] = "ModificationalConstant"
            collapse_info = None
            if wrapping_relation is not None:
                # The adjective has an internal complement, e.g.
                # "Wahoos disenchanted with Pete": keep
                # Modification(variable=Wahoos, body=with(...)) and only remove
                # the stale predicate edge from the adjective itself.
                remove_edges_by_indexes(sample, {edge_index})
            else:
                graph(sample)["edges"][edge_index]["label"] = "as:modification"
            if aux is not None and wrapping_relation is None:
                collapse_info = collapse_simple_modification_aux(sample, aux, node_index)
            changes.append(
                {
                    "kind": "leaf_adjectival_participle_fp_arg1_to_modifier",
                    "token_index": token_index,
                    "text": ud_word.get("text"),
                    "node_index": node_index,
                    "before_text": before_text,
                    "head_node": variable,
                    "wrapping_relation": wrapping_relation,
                    "collapsed_aux": collapse_info,
                }
            )
            continue

        reviews.append(
            {
                "kind": "adjectival_participle_fp_has_complex_pred_edges",
                "token_index": token_index,
                "text": ud_word.get("text"),
                "node_index": node_index,
                "node_text": node_text(sample, node_index),
                "pred_edges": [edge for _idx, edge in pred_edges],
                "ud": ud_word,
            }
        )

    return changes, reviews


def norm_word(sample: dict[str, Any], token_index: int) -> str:
    return sample["lingua"]["words"][token_index].strip().lower().replace("’", "'")


def partition_merged_state_participle_ud(
    sample: dict[str, Any],
    ud_words: list[dict[str, Any]],
    ud_children: dict[int, list[dict[str, Any]]],
    node_index: int,
) -> tuple[list[int], list[int], list[int], str] | None:
    node = graph(sample)["nodes"][node_index]
    if node.get("type") != "FactualPredicator":
        return None
    if edge_out(sample, node_index, "pred.arg.2"):
        return None
    tokens = node_token_indexes(node)
    if len(tokens) < 2:
        return None

    state_positions = [
        pos
        for pos, token_index in enumerate(tokens)
        if 0 <= token_index < len(ud_words) and is_adjectival_participle(ud_words[token_index])
    ]
    if not state_positions:
        return None
    state_pos = state_positions[-1]
    if state_pos != len(tokens) - 1:
        return None

    state_indexes = [tokens[state_pos]]
    state_child_deps = {
        str(child.get("deprel") or "") for child in ud_children.get(tokens[state_pos], [])
    }
    if state_child_deps & STATE_CHILD_REVIEW_DEPRELS:
        return None

    predicate_indexes: list[int] = []
    modifier_indexes: list[int] = []
    for token_index in tokens[:state_pos]:
        normalized = norm_word(sample, token_index)
        if normalized in STATE_PREDICATE_WORDS:
            predicate_indexes.append(token_index)
            continue
        if normalized in ADVERBIAL_MODIFIERS or normalized in {"not", "n't", "nt", "n", "t"}:
            modifier_indexes.append(token_index)
            continue
        return None

    if not predicate_indexes:
        return None
    return sorted(predicate_indexes), sorted(modifier_indexes), state_indexes, "ud_adjectival_participle"


def ud_infos(ud_words: list[dict[str, Any]], indexes: list[int]) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for index in indexes:
        if 0 <= index < len(ud_words):
            word = ud_words[index]
            infos.append(
                {
                    "token_index": index,
                    "text": word.get("text"),
                    "upos": word.get("upos"),
                    "xpos": word.get("xpos"),
                    "deprel": word.get("deprel"),
                    "head": word.get("head"),
                }
            )
    return infos


def split_merged_state_participle_node(
    sample: dict[str, Any],
    ud_words: list[dict[str, Any]],
    node_index: int,
    predicate_indexes: list[int],
    modifier_indexes: list[int],
    state_indexes: list[int],
    gate_reason: str,
) -> dict[str, Any]:
    before_text = node_text(sample, node_index)
    original_tokens = node_tokens(sample, node_index)

    set_node_token_indexes(sample, node_index, predicate_indexes)
    state_node = add_node(sample, spans_from_token_indexes(state_indexes), "ModificationalConstant")
    add_edge(sample, node_index, state_node, "pred.arg.2")

    modifier_root: int | None = None
    change_kind = "merged_state_participle_split_ud"
    if modifier_indexes:
        modifier_spans = grouped_modifier_spans(original_tokens, modifier_indexes)
        if not modifier_spans:
            modifier_spans = spans_from_token_indexes(modifier_indexes)
        old_node_count = len(graph(sample)["nodes"])
        modifier_root = add_modifier_chain(sample, modifier_spans, node_index)
        new_nodes = set(range(old_node_count, len(graph(sample)["nodes"])))
        transfer_incoming_to_new_node(sample, node_index, modifier_root, excluded_starts=new_nodes)
        change_kind = "merged_state_participle_split_ud_with_modifier"

    return {
        "kind": change_kind,
        "node_index": node_index,
        "before_text": before_text,
        "predicate_indexes": predicate_indexes,
        "modifier_indexes": modifier_indexes,
        "state_indexes": state_indexes,
        "state_node": state_node,
        "modifier_root": modifier_root,
        "gate_reason": gate_reason,
        "state_ud": ud_infos(ud_words, state_indexes),
    }


def repair_merged_state_participles(
    sample: dict[str, Any],
    ud_words: list[dict[str, Any]],
    ud_row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    changes: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    ud_children = ud_children_by_index(ud_row)
    original_node_count = len(graph(sample)["nodes"])

    for node_index in range(original_node_count):
        if node_index >= len(graph(sample)["nodes"]):
            continue
        partition = partition_merged_state_participle_ud(sample, ud_words, ud_children, node_index)
        if partition is None:
            continue
        predicate_indexes, modifier_indexes, state_indexes, gate_reason = partition
        if not any(
            0 <= index < len(ud_words) and is_adjectival_participle(ud_words[index])
            for index in state_indexes
        ):
            continue
        changes.append(
            split_merged_state_participle_node(
                sample,
                ud_words,
                node_index,
                predicate_indexes,
                modifier_indexes,
                state_indexes,
                gate_reason,
            )
        )
    return changes, reviews


def audit_eventive_participle_single_modification(
    sample: dict[str, Any],
    ud_words: list[dict[str, Any]],
    ud_row: dict[str, Any],
) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    children = ud_children_by_index(ud_row)
    nodes = graph(sample)["nodes"]
    for token_index, ud_word in enumerate(ud_words):
        xpos = str(ud_word.get("xpos") or "")
        if xpos not in {"VBG", "VBN"} or is_adjectival_participle(ud_word):
            continue
        exact, _containing = token_nodes(sample, token_index)
        mc_nodes = [node_index for node_index in exact if nodes[node_index].get("type") == "ModificationalConstant"]
        if len(mc_nodes) != 1:
            continue
        node_index = mc_nodes[0]
        if not edge_out(sample, node_index, "as:modification"):
            continue
        child_deps = {str(child.get("deprel") or "") for child in children.get(token_index, [])}
        if not (child_deps & CONTENT_CHILD_DEPRELS):
            continue
        reviews.append(
            {
                "kind": "verb_participle_single_layer_modifier_with_ud_arguments",
                "token_index": token_index,
                "text": ud_word.get("text"),
                "node_index": node_index,
                "node_text": node_text(sample, node_index),
                "out_edges": edge_out(sample, node_index),
                "ud": ud_word,
                "ud_children": children.get(token_index, []),
            }
        )
    return reviews


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    all_changes: list[dict[str, Any]] = []
    all_reviews: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for split in ("train", "dev", "test"):
        data_path = data_dir / f"{split}.jsonl"
        ud_path = data_dir / f"{split}_ud.jsonl"
        rows = read_jsonl(data_path)
        ud_rows = read_jsonl(ud_path)
        if len(rows) != len(ud_rows):
            raise ValueError(f"{split}: data/UD row count mismatch {len(rows)} != {len(ud_rows)}")
        new_rows: list[dict[str, Any]] = []

        for line_number, (row, ud_row) in enumerate(zip(rows, ud_rows), 1):
            ud_words = get_ud_words(ud_row)
            words = row["lingua"]["words"]
            if len(ud_words) != len(words):
                new_rows.append(row)
                counts[f"{split}:skipped_ud_alignment"] += 1
                continue

            before = deepcopy(row)
            after = deepcopy(row)
            changes: list[dict[str, Any]] = []
            reviews: list[dict[str, Any]] = []

            state_changes, state_reviews = repair_merged_state_participles(after, ud_words, ud_row)
            changes.extend(state_changes)
            reviews.extend(state_reviews)

            leaf_changes, leaf_reviews = repair_leaf_adjectival_fp(after, ud_words)
            changes.extend(leaf_changes)
            reviews.extend(leaf_reviews)

            reviews.extend(audit_eventive_participle_single_modification(after, ud_words, ud_row))

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

            if reviews:
                for review in reviews:
                    counts[review["kind"]] += 1
                    counts[f"{split}:{review['kind']}"] += 1
                all_reviews.append(
                    {
                        "split": split,
                        "line_number": line_number,
                        "sentence": row.get("sentence", ""),
                        "reviews": reviews,
                        "graph": compact_graph(after),
                    }
                )

        if args.apply:
            write_jsonl(data_path, new_rows)

    (report_dir / "changes.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in all_changes)
        + ("\n" if all_changes else ""),
        encoding="utf-8",
    )
    (report_dir / "reviews.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in all_reviews)
        + ("\n" if all_reviews else ""),
        encoding="utf-8",
    )

    summary = {
        "data_dir": str(data_dir),
        "report_dir": str(report_dir),
        "external_syntax": "ud_cache",
        "apply": bool(args.apply),
        "changed_samples": len(all_changes),
        "review_samples": len(all_reviews),
        "counts": dict(counts),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_lines = [
        "# Participle Adjectival Standard Repair",
        "",
        f"- data_dir: `{data_dir}`",
        f"- apply: `{args.apply}`",
        f"- changed_samples: `{len(all_changes)}`",
        f"- review_samples: `{len(all_reviews)}`",
        "",
        "| key | count |",
        "|---|---:|",
    ]
    for key, count in sorted(counts.items()):
        summary_lines.append(f"| {key} | {count} |")
    summary_lines.extend(["", "## First 40 Changes", ""])
    for record in all_changes[:40]:
        summary_lines.extend(
            [
                f"### {record['split']}:{record['line_number']}",
                "",
                record["sentence"],
                "",
                "```text",
                record["before"],
                "",
                "--- after ---",
                record["after"],
                "```",
                "",
            ]
        )
    summary_lines.extend(["", "## First 40 Reviews", ""])
    for record in all_reviews[:40]:
        summary_lines.extend(
            [
                f"### {record['split']}:{record['line_number']}",
                "",
                record["sentence"],
                "",
                "```text",
                record["graph"],
                "```",
                "",
            ]
        )
    (report_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
