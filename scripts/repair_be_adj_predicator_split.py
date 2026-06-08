#!/usr/bin/env python3
"""Split merged `be + adjective/state` FactualPredicator nodes with spaCy.

The repair is intentionally context gated:

* merged predicates such as `is ready` become `is ->pred.arg.2-> ready`;
* adjective phrases such as `is 6 years old` keep the whole complement as one
  `ModificationalConstant`;
* passive/progressive forms whose complement is pure `VBN`/`VBD`/`VBG` are not
  split by default.

When given a previous change report, the script can also merge back the narrow
class of earlier script-created splits that spaCy now classifies as pure
`VBN`/`VBD` and whose state node is structurally isolated.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Doc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repair_be_copula_rules import (  # noqa: E402
    ADVERBIAL_MODIFIERS,
    BE_FORMS,
    DIRECT_PREPOSITIONS,
    MODALS_AND_AUX,
    NEGATION_PARTS,
    add_edge,
    add_modifier_chain,
    add_node,
    compact_graph,
    edge_in,
    edge_out,
    graph,
    grouped_modifier_spans,
    has_be,
    is_be_apostrophe,
    is_factual,
    is_negation_apostrophe,
    node_text,
    node_tokens,
    read_jsonl,
    remove_nodes,
    set_node_token_indexes,
    spans_from_token_indexes,
    transfer_incoming_to_new_node,
    write_jsonl,
)


ADJ_TAGS = {"JJ", "JJR", "JJS"}
PAST_PARTICIPLE_TAGS = {"VBN", "VBD"}
ALLOWED_LEADING_STATE_ADPOSITIONS = {"beyond"}
LEXICAL_STATE_EXCEPTIONS = {"alright", "fine", "fun", "ok", "okay", "right"}


def make_doc(nlp: Language, sample: dict[str, Any]) -> Doc:
    """Run spaCy on Lingua's existing tokenization so graph indexes stay valid."""
    words = [word.strip() or word for word in sample["lingua"]["words"]]
    spaces = [True] * max(0, len(words) - 1) + ([False] if words else [])
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    for _name, pipe in nlp.pipeline:
        doc = pipe(doc)
    return doc


def token_indexes_for_node(sample: dict[str, Any], node_index: int) -> list[int]:
    return [index for index, _surface, _norm in node_tokens(sample, node_index)]


def token_infos(doc: Doc, indexes: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "text": doc[index].text,
            "lemma": doc[index].lemma_,
            "pos": doc[index].pos_,
            "tag": doc[index].tag_,
            "dep": doc[index].dep_,
        }
        for index in indexes
        if 0 <= index < len(doc)
    ]


def node_tokens_for_indexes(sample: dict[str, Any], indexes: list[int]) -> list[tuple[int, str, str]]:
    wanted = set(indexes)
    tokens: list[tuple[int, str, str]] = []
    for word_index, word in enumerate(sample["lingua"]["words"]):
        if word_index in wanted:
            tokens.append((word_index, word, word.strip().lower().replace("’", "'")))
    return tokens


def has_spacy_adj(doc: Doc, indexes: list[int]) -> bool:
    return any(doc[index].pos_ == "ADJ" or doc[index].tag_ in ADJ_TAGS for index in indexes)


def is_pure_past_participle_state(doc: Doc, indexes: list[int]) -> bool:
    content = [doc[index] for index in indexes if doc[index].pos_ not in {"PUNCT", "SPACE"}]
    return bool(content) and all(token.tag_ in PAST_PARTICIPLE_TAGS for token in content)


def state_gate(
    sample: dict[str, Any],
    doc: Doc,
    state_indexes: list[int],
    *,
    max_state_tokens: int,
    allow_lexical_exceptions: bool,
) -> tuple[bool, str]:
    if not state_indexes:
        return False, "empty_state"
    if len(state_indexes) > max_state_tokens:
        return False, "state_too_long"

    state_tokens = node_tokens_for_indexes(sample, state_indexes)
    state_norms = [norm for _index, _surface, norm in state_tokens if norm != "'"]
    if any(norm in BE_FORMS or norm in MODALS_AND_AUX for norm in state_norms):
        return False, "state_contains_aux_or_be"

    for offset, (index, _surface, norm) in enumerate(state_tokens):
        if norm == "'":
            continue
        token = doc[index]
        is_adposition = token.pos_ == "ADP" or norm in DIRECT_PREPOSITIONS
        if is_adposition and not (offset == 0 and norm in ALLOWED_LEADING_STATE_ADPOSITIONS):
            return False, "state_contains_adposition"

    if has_spacy_adj(doc, state_indexes):
        return True, "spacy_adj"

    content_indexes = [
        index
        for index in state_indexes
        if doc[index].pos_ not in {"PUNCT", "SPACE"} and node_tokens_for_indexes(sample, [index])[0][2] != "'"
    ]
    if allow_lexical_exceptions and content_indexes:
        head_norm = sample["lingua"]["words"][content_indexes[-1]].strip().lower().replace("’", "'")
        if head_norm in LEXICAL_STATE_EXCEPTIONS:
            return True, f"lexical_exception:{head_norm}"

    if content_indexes:
        head = doc[content_indexes[-1]]
        return False, f"no_spacy_adj:{head.pos_}/{head.tag_}"
    return False, "no_content_state"


def partition_merged_be_adj(
    sample: dict[str, Any],
    node_index: int,
    doc: Doc,
    *,
    max_state_tokens: int,
    include_modal_aux: bool,
    allow_lexical_exceptions: bool,
) -> tuple[list[int], list[int], list[int], str] | None:
    """Return (predicate_indexes, modifier_indexes, state_indexes, reason)."""
    if not is_factual(sample, node_index) or not has_be(sample, node_index):
        return None
    if edge_out(sample, node_index, "pred.arg.2"):
        return None

    tokens = node_tokens(sample, node_index)
    if not tokens:
        return None

    normalized = [norm for _index, _surface, norm in tokens if norm != "'"]
    if "there" in normalized:
        return None

    predicate_indexes: list[int] = []
    modifier_indexes: list[int] = []
    state_indexes: list[int] = []
    saw_be = False

    for position, (index, _surface, norm) in enumerate(tokens):
        if is_be_apostrophe(tokens, position):
            predicate_indexes.append(index)
            saw_be = True
            continue
        if is_negation_apostrophe(tokens, position):
            modifier_indexes.append(index)
            continue
        if norm == "'":
            predicate_indexes.append(index)
            continue
        if norm in BE_FORMS:
            predicate_indexes.append(index)
            saw_be = True
            continue
        if include_modal_aux and norm in MODALS_AND_AUX and not state_indexes:
            predicate_indexes.append(index)
            continue
        if norm in NEGATION_PARTS or (
            norm in ADVERBIAL_MODIFIERS and not (doc[index].pos_ == "ADJ" or doc[index].tag_ in ADJ_TAGS)
        ):
            modifier_indexes.append(index)
            continue
        if not saw_be:
            return None
        state_indexes.append(index)

    if not predicate_indexes or not state_indexes:
        return None
    ok, reason = state_gate(
        sample,
        doc,
        sorted(state_indexes),
        max_state_tokens=max_state_tokens,
        allow_lexical_exceptions=allow_lexical_exceptions,
    )
    if not ok:
        return None
    return sorted(predicate_indexes), sorted(modifier_indexes), sorted(state_indexes), reason


def split_merged_be_adj_node(
    sample: dict[str, Any],
    doc: Doc,
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

    change_kind = "merged_be_adj_predicator_split_spacy"
    modifier_root = None
    if modifier_indexes:
        modifier_spans = grouped_modifier_spans(original_tokens, modifier_indexes)
        if not modifier_spans:
            modifier_spans = spans_from_token_indexes(modifier_indexes)
        old_node_count = len(graph(sample)["nodes"])
        modifier_root = add_modifier_chain(sample, modifier_spans, node_index)
        new_nodes = set(range(old_node_count, len(graph(sample)["nodes"])))
        transfer_incoming_to_new_node(sample, node_index, modifier_root, excluded_starts=new_nodes)
        change_kind = "merged_be_adj_predicator_split_spacy_with_modifier"

    return {
        "kind": change_kind,
        "node_index": node_index,
        "before_text": before_text,
        "predicate_text": node_text(sample, node_index),
        "state_node": state_node,
        "state_text": node_text(sample, state_node),
        "modifier_root": modifier_root,
        "gate_reason": gate_reason,
        "state_spacy": token_infos(doc, state_indexes),
    }


def repair_sample(
    sample: dict[str, Any],
    doc: Doc,
    *,
    max_state_tokens: int,
    include_modal_aux: bool,
    allow_lexical_exceptions: bool,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    original_node_count = len(graph(sample)["nodes"])

    for node_index in range(original_node_count):
        partition = partition_merged_be_adj(
            sample,
            node_index,
            doc,
            max_state_tokens=max_state_tokens,
            include_modal_aux=include_modal_aux,
            allow_lexical_exceptions=allow_lexical_exceptions,
        )
        if partition is None:
            continue
        predicate_indexes, modifier_indexes, state_indexes, gate_reason = partition
        changes.append(
            split_merged_be_adj_node(
                sample,
                doc,
                node_index,
                predicate_indexes,
                modifier_indexes,
                state_indexes,
                gate_reason,
            )
        )
    return changes


def load_previous_changes(path: Path | None) -> dict[tuple[str, int], list[dict[str, Any]]]:
    if path is None:
        return {}
    previous: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = (record["split"], int(record["line_number"]))
        previous[key].extend(record.get("changes", []))
    return dict(previous)


def can_merge_previous_split(sample: dict[str, Any], predicate_node: int, state_node: int) -> tuple[bool, str]:
    nodes = graph(sample)["nodes"]
    if predicate_node >= len(nodes) or state_node >= len(nodes):
        return False, "node_index_out_of_range"
    if nodes[predicate_node].get("type") != "FactualPredicator":
        return False, "predicate_node_not_factual"
    if nodes[state_node].get("type") != "ModificationalConstant":
        return False, "state_node_not_modificational"

    arg2_edges = [
        edge
        for edge in edge_out(sample, predicate_node, "pred.arg.2")
        if edge["end"] == state_node
    ]
    if len(arg2_edges) != 1:
        return False, "missing_or_duplicate_pred_arg2"
    incoming = edge_in(sample, state_node)
    if len(incoming) != 1 or incoming[0]["start"] != predicate_node or incoming[0]["label"] != "pred.arg.2":
        return False, "state_has_other_incoming_edges"
    if edge_out(sample, state_node):
        return False, "state_has_outgoing_edges"
    return True, "mergeable"


def merge_previous_vbn_vbd_splits(
    sample: dict[str, Any],
    doc: Doc,
    previous_changes: list[dict[str, Any]],
    *,
    merge: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    changes: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    merge_plan: list[tuple[int, int, dict[str, Any]]] = []
    seen_state_nodes: set[int] = set()

    for change in previous_changes:
        predicate_node = change.get("node_index")
        state_node = change.get("state_node")
        if not isinstance(predicate_node, int) or not isinstance(state_node, int):
            continue
        if state_node in seen_state_nodes:
            continue
        if predicate_node >= len(graph(sample)["nodes"]) or state_node >= len(graph(sample)["nodes"]):
            continue
        previous_predicate_text = change.get("predicate_text")
        previous_state_text = change.get("state_text")
        if previous_predicate_text is not None and node_text(sample, predicate_node) != previous_predicate_text:
            continue
        if previous_state_text is not None and node_text(sample, state_node) != previous_state_text:
            continue

        state_indexes = token_indexes_for_node(sample, state_node)
        state_info = token_infos(doc, state_indexes)
        has_adj = has_spacy_adj(doc, state_indexes)
        pure_vbn_vbd = is_pure_past_participle_state(doc, state_indexes)
        if has_adj:
            continue

        review = {
            "kind": "previous_split_non_spacy_adj_review",
            "node_index": predicate_node,
            "state_node": state_node,
            "predicate_text": node_text(sample, predicate_node)
            if predicate_node < len(graph(sample)["nodes"])
            else None,
            "state_text": node_text(sample, state_node),
            "state_spacy": state_info,
            "previous_kind": change.get("kind"),
        }

        if pure_vbn_vbd:
            ok, reason = can_merge_previous_split(sample, predicate_node, state_node)
            review["kind"] = "previous_split_pure_vbn_vbd_review"
            review["merge_check"] = reason
            if merge and ok:
                merge_plan.append((predicate_node, state_node, review))
                seen_state_nodes.add(state_node)
            else:
                reviews.append(review)
        else:
            reviews.append(review)

    if not merge_plan:
        return changes, reviews

    removed_state_nodes: set[int] = set()
    for predicate_node, state_node, review in merge_plan:
        predicate_indexes = token_indexes_for_node(sample, predicate_node)
        state_indexes = token_indexes_for_node(sample, state_node)
        before_text = f"{node_text(sample, predicate_node)} + {node_text(sample, state_node)}"
        set_node_token_indexes(sample, predicate_node, sorted(set(predicate_indexes + state_indexes)))
        removed_state_nodes.add(state_node)
        change = dict(review)
        change.update(
            {
                "kind": "previous_split_pure_vbn_vbd_merged_back",
                "before_text": before_text,
                "after_text": node_text(sample, predicate_node),
            }
        )
        changes.append(change)

    remove_nodes(sample, removed_state_nodes)
    return changes, reviews


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    nlp = spacy.load(args.spacy_model)
    previous_by_sample = load_previous_changes(args.previous_changes)

    all_changes: list[dict[str, Any]] = []
    all_reviews: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for split in ("train", "dev", "test"):
        rows = read_jsonl(input_dir / f"{split}.jsonl")
        new_rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(rows, 1):
            before = deepcopy(row)
            after = deepcopy(row)
            doc = make_doc(nlp, after)

            changes: list[dict[str, Any]] = []
            reviews: list[dict[str, Any]] = []
            previous_changes = previous_by_sample.get((split, line_number), [])
            if previous_changes:
                merge_changes, previous_reviews = merge_previous_vbn_vbd_splits(
                    after,
                    doc,
                    previous_changes,
                    merge=args.merge_previous_vbn_vbd,
                )
                changes.extend(merge_changes)
                reviews.extend(previous_reviews)

            changes.extend(
                repair_sample(
                    after,
                    doc,
                    max_state_tokens=args.max_state_tokens,
                    include_modal_aux=not args.no_modal_aux,
                    allow_lexical_exceptions=args.allow_lexical_exceptions,
                )
            )

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
                        "sample": after if args.include_samples_in_report else None,
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
        write_jsonl(output_dir / f"{split}.jsonl", new_rows)

    changes_path = report_dir / "changes.jsonl"
    changes_path.write_text(
        "\n".join(json.dumps(change, ensure_ascii=False) for change in all_changes)
        + ("\n" if all_changes else ""),
        encoding="utf-8",
    )
    reviews_path = report_dir / "reviews.jsonl"
    reviews_path.write_text(
        "\n".join(json.dumps(review, ensure_ascii=False) for review in all_reviews)
        + ("\n" if all_reviews else ""),
        encoding="utf-8",
    )

    summary_lines = [
        "# spaCy-gated Be + Adj FactualPredicator Split Summary",
        "",
        f"- input_dir: `{input_dir}`",
        f"- output_dir: `{output_dir}`",
        f"- report_dir: `{report_dir}`",
        f"- spacy_model: `{args.spacy_model}`",
        f"- changed samples: {len(all_changes)}",
        f"- review samples: {len(all_reviews)}",
        f"- changed nodes: {sum(count for key, count in counts.items() if ':' not in key and not key.endswith('_review'))}",
        f"- max_state_tokens: {args.max_state_tokens}",
        f"- include_modal_aux: {not args.no_modal_aux}",
        f"- allow_lexical_exceptions: {args.allow_lexical_exceptions}",
        f"- merge_previous_vbn_vbd: {args.merge_previous_vbn_vbd}",
        "",
        "## Counts",
        "",
        "| key | count |",
        "|---|---:|",
    ]
    for key, count in sorted(counts.items()):
        summary_lines.append(f"| {key} | {count} |")

    summary_lines.extend(["", "## First 50 Changes", ""])
    for change in all_changes[:50]:
        summary_lines.extend(
            [
                f"### {change['split']}:{change['line_number']} "
                f"`{', '.join(c['kind'] for c in change['changes'])}`",
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

    summary_lines.extend(["", "## First 50 Reviews", ""])
    for review in all_reviews[:50]:
        summary_lines.extend(
            [
                f"### {review['split']}:{review['line_number']} "
                f"`{', '.join(r['kind'] for r in review['reviews'])}`",
                "",
                review["sentence"],
                "",
                "```text",
                review["graph"],
                "```",
                "",
            ]
        )

    summary_path = report_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "spacy_model": args.spacy_model,
        "changed_samples": len(all_changes),
        "review_samples": len(all_reviews),
        "counts": dict(counts),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--previous-changes", type=Path)
    parser.add_argument("--max-state-tokens", type=int, default=5)
    parser.add_argument("--no-modal-aux", action="store_true")
    parser.add_argument("--allow-lexical-exceptions", action="store_true")
    parser.add_argument("--merge-previous-vbn-vbd", action="store_true")
    parser.add_argument("--include-samples-in-report", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
