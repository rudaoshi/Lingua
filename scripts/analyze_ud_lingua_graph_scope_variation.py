#!/usr/bin/env python3
"""Find UD-local patterns where Lingua-layer scope annotations diverge.

This is intentionally different from the word-lingua analysis: endpoints are
resolved against gold/pred Lingua graphs first. If an endpoint is an abstract
Lingua wrapper such as Modification, Expression, Ref, or Attribute, its
descendant token span is used as the UD alignment surface.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "test")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize_spans(spans: Any) -> str:
    return json.dumps(spans, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def node_token_indexes(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if not isinstance(spans, list):
        return []
    indexes: list[int] = []
    for span in spans:
        if (
            isinstance(span, list)
            and len(span) == 2
            and isinstance(span[0], int)
            and isinstance(span[1], int)
            and span[0] <= span[1]
        ):
            indexes.extend(range(span[0], span[1] + 1))
    return sorted(set(indexes))


def text_from_tokens(words: list[str], indexes: list[int]) -> str:
    return "".join(words[index] for index in sorted(set(indexes)) if 0 <= index < len(words)).strip()


def graph_payload(bundle: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    return bundle.get("words") or [], (bundle.get("graph") or {}).get("nodes") or [], (bundle.get("graph") or {}).get("edges") or []


def graph_node_info(words: list[str], node: dict[str, Any]) -> dict[str, Any]:
    spans = node.get("spans")
    if isinstance(spans, str):
        text = spans
    elif isinstance(spans, list):
        text = text_from_tokens(words, node_token_indexes(node))
    else:
        text = ""
    return {"text": text, "type": node.get("type"), "spans": spans}


def candidate_node_indexes(bundle: dict[str, Any], endpoint: dict[str, Any]) -> list[int]:
    words, nodes, _ = graph_payload(bundle)
    endpoint_text = str(endpoint.get("text") or "")
    endpoint_type = endpoint.get("type")
    endpoint_spans = endpoint.get("spans")
    endpoint_span_key = normalize_spans(endpoint_spans)

    scored: list[tuple[int, int]] = []
    for index, node in enumerate(nodes):
        info = graph_node_info(words, node)
        if endpoint_type is not None and info["type"] != endpoint_type:
            continue
        score = 0
        graph_spans = info["spans"]
        graph_span_key = normalize_spans(graph_spans)
        if endpoint_spans is not None and graph_span_key == endpoint_span_key:
            score += 10
        if endpoint_spans is None and isinstance(graph_spans, str) and endpoint_text == graph_spans:
            score += 10
        if endpoint_text and endpoint_text == info["text"]:
            score += 4
        if endpoint_type == info["type"]:
            score += 2
        if score:
            scored.append((score, index))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [index for _, index in scored]


def resolve_edge_indexes(
    bundle: dict[str, Any],
    source: dict[str, Any],
    label: str,
    target: dict[str, Any],
) -> tuple[int | None, int | None, str]:
    _, _, edges = graph_payload(bundle)
    source_candidates = candidate_node_indexes(bundle, source)
    target_candidates = candidate_node_indexes(bundle, target)
    source_set = set(source_candidates)
    target_set = set(target_candidates)
    matches = [
        edge
        for edge in edges
        if edge.get("label") == label
        and int(edge.get("start", -1)) in source_set
        and int(edge.get("end", -1)) in target_set
    ]
    if matches:
        edge = matches[0]
        return int(edge["start"]), int(edge["end"]), "edge_match"
    return (
        source_candidates[0] if source_candidates else None,
        target_candidates[0] if target_candidates else None,
        "node_match_only" if source_candidates or target_candidates else "unresolved",
    )


def descendant_tokens(bundle: dict[str, Any], node_index: int | None) -> list[int]:
    if node_index is None:
        return []
    _, nodes, edges = graph_payload(bundle)
    outgoing: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        outgoing[int(edge["start"])].append(int(edge["end"]))

    tokens: set[int] = set()
    seen: set[int] = set()
    stack = [node_index]
    while stack:
        index = stack.pop()
        if index in seen or index < 0 or index >= len(nodes):
            continue
        seen.add(index)
        tokens.update(node_token_indexes(nodes[index]))
        stack.extend(outgoing.get(index, []))
    return sorted(tokens)


def lingua_word_char_spans(words: list[str]) -> list[tuple[int, int, int, int]]:
    spans = []
    pos = 0
    for word in words:
        raw_start = pos
        raw_end = pos + len(word)
        leading = len(word) - len(word.lstrip())
        token_start = raw_start + leading
        token_end = raw_end
        spans.append((raw_start, raw_end, token_start, token_end))
        pos = raw_end
    return spans


def token_indexes_char_span(indexes: list[int], words: list[str]) -> tuple[int, int] | None:
    word_spans = lingua_word_char_spans(words)
    starts: list[int] = []
    ends: list[int] = []
    for index in indexes:
        if 0 <= index < len(word_spans):
            starts.append(word_spans[index][2])
            ends.append(word_spans[index][3])
    if not starts:
        return None
    return min(starts), max(ends)


def flatten_ud_words(ud_row: dict[str, Any]) -> list[dict[str, Any]]:
    words = []
    offset = 0
    for sent in ud_row.get("sentences") or []:
        sent_words = sent.get("words") or []
        for word in sent_words:
            item = dict(word)
            item["global_id"] = offset + int(word["id"])
            head = int(word["head"] or 0)
            item["global_head"] = 0 if head == 0 else offset + head
            words.append(item)
        offset += len(sent_words)
    return words


def overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def choose_ud_root(token_ids: list[int], words_by_id: dict[int, dict[str, Any]]) -> int | None:
    if not token_ids:
        return None
    token_set = set(token_ids)
    for token_id in token_ids:
        head = int(words_by_id[token_id].get("global_head") or 0)
        if head not in token_set:
            return token_id
    return token_ids[0]


def map_tokens_to_ud(indexes: list[int], ud_row: dict[str, Any]) -> dict[str, Any]:
    ud_words = flatten_ud_words(ud_row)
    words_by_id = {int(word["global_id"]): word for word in ud_words}
    char_span = token_indexes_char_span(indexes, ud_row.get("words") or [])
    if char_span is None:
        return {"token_ids": [], "root_id": None, "status": "no_desc_tokens"}
    token_ids = []
    for word in ud_words:
        start = word.get("start_char")
        end = word.get("end_char")
        if start is None or end is None:
            continue
        if overlap(char_span, (int(start), int(end))) > 0:
            token_ids.append(int(word["global_id"]))
    return {
        "token_ids": token_ids,
        "root_id": choose_ud_root(token_ids, words_by_id),
        "status": "ok" if token_ids else "no_ud_overlap",
    }


def token_summary(token_id: int | None, words_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if token_id is None or token_id not in words_by_id:
        return {"id": None, "text": "", "lemma": "", "upos": "NONE", "deprel": "NONE", "head": None, "head_upos": "ROOT"}
    word = words_by_id[token_id]
    head = int(word.get("global_head") or 0)
    head_word = words_by_id.get(head)
    return {
        "id": token_id,
        "text": word.get("text") or "",
        "lemma": word.get("lemma") or "",
        "upos": word.get("upos") or "NONE",
        "deprel": word.get("deprel") or "NONE",
        "head": head,
        "head_text": "" if head_word is None else head_word.get("text") or "",
        "head_upos": "ROOT" if head == 0 or head_word is None else head_word.get("upos") or "NONE",
    }


def ancestor_map(token_id: int, words_by_id: dict[int, dict[str, Any]]) -> dict[int, tuple[int, list[str]]]:
    result = {token_id: (0, [])}
    current = token_id
    path: list[str] = []
    depth = 0
    while current in words_by_id:
        head = int(words_by_id[current].get("global_head") or 0)
        if head == 0:
            result[0] = (depth + 1, path + [f"up:{words_by_id[current].get('deprel')}"])
            break
        path = path + [f"up:{words_by_id[current].get('deprel')}"]
        depth += 1
        current = head
        if current in result:
            break
        result[current] = (depth, list(path))
    return result


def ud_relation(src_id: int | None, tgt_id: int | None, words_by_id: dict[int, dict[str, Any]]) -> str:
    if src_id is None or tgt_id is None:
        return "unmapped"
    if src_id == tgt_id:
        return "same_ud_token"
    src = words_by_id.get(src_id)
    tgt = words_by_id.get(tgt_id)
    if not src or not tgt:
        return "unmapped"
    if int(src.get("global_head") or 0) == tgt_id:
        return f"src_dep_of_tgt:{src.get('deprel')}"
    if int(tgt.get("global_head") or 0) == src_id:
        return f"tgt_dep_of_src:{tgt.get('deprel')}"
    src_head = int(src.get("global_head") or 0)
    tgt_head = int(tgt.get("global_head") or 0)
    if src_head == tgt_head and src_head != 0:
        head = words_by_id.get(src_head)
        head_upos = "NONE" if head is None else head.get("upos")
        return f"siblings:{head_upos}:{src.get('deprel')}+{tgt.get('deprel')}"

    src_anc = ancestor_map(src_id, words_by_id)
    tgt_anc = ancestor_map(tgt_id, words_by_id)
    common = set(src_anc) & set(tgt_anc)
    if not common:
        return "disconnected"
    lca = min(common, key=lambda node: src_anc[node][0] + tgt_anc[node][0])
    distance = src_anc[lca][0] + tgt_anc[lca][0]
    lca_word = words_by_id.get(lca)
    lca_upos = "ROOT" if lca == 0 or lca_word is None else lca_word.get("upos")
    return f"path_len={distance}:lca={lca_upos}"


def ud_pattern(src: dict[str, Any], tgt: dict[str, Any], relation: str, *, lexical: bool = False) -> str:
    if lexical:
        return (
            f"{src['lemma'].lower()}/{src['upos']}[{src['deprel']}->{src['head_upos']}] "
            f"--{relation}--> "
            f"{tgt['lemma'].lower()}/{tgt['upos']}[{tgt['deprel']}->{tgt['head_upos']}]"
        )
    return (
        f"{src['upos']}[{src['deprel']}->{src['head_upos']}] "
        f"--{relation}--> "
        f"{tgt['upos']}[{tgt['deprel']}->{tgt['head_upos']}]"
    )


def load_ud(base: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for split in SPLITS:
        for row in read_jsonl(base / f"{split}.jsonl"):
            rows[(str(row["split"]), int(row["line"]))] = row
    return rows


def load_predictions(base: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for split in SPLITS:
        for row in read_jsonl(base / split / "predictions.jsonl"):
            rows[(split, int(row["line"]))] = row
    return rows


def label_group(label: str) -> str:
    if label in {"as:modification", "@modification_of", "body", "variable"}:
        return "modification"
    if label.startswith("pred.arg."):
        return "predicate_args"
    if label in {"as:punctuation", "punctuation"}:
        return "punctuation"
    if label in {"func.arg", "@expression", "@list"}:
        return "expression_list"
    return "other"


def compact_counter(counter: Counter[str], limit: int = 20) -> str:
    return " | ".join(f"{key}:{count}" for key, count in counter.most_common(limit))


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def endpoint_record(bundle: dict[str, Any], ud_row: dict[str, Any], node_index: int | None, endpoint: dict[str, Any], prefix: str) -> dict[str, Any]:
    words, nodes, _ = graph_payload(bundle)
    desc_indexes = descendant_tokens(bundle, node_index)
    own = nodes[node_index] if node_index is not None and 0 <= node_index < len(nodes) else {}
    own_indexes = node_token_indexes(own) if own else []
    anchor_indexes = own_indexes or desc_indexes
    mapped = map_tokens_to_ud(anchor_indexes, ud_row)
    ud_words = flatten_ud_words(ud_row)
    words_by_id = {int(word["global_id"]): word for word in ud_words}
    summary = token_summary(mapped["root_id"], words_by_id)
    own_info = graph_node_info(words, own) if own else endpoint
    return {
        f"{prefix}_node_index": "" if node_index is None else str(node_index),
        f"{prefix}_node_text": own_info.get("text") or endpoint.get("text") or "",
        f"{prefix}_node_type": own_info.get("type") or endpoint.get("type") or "",
        f"{prefix}_node_spans": normalize_spans(own_info.get("spans", endpoint.get("spans"))),
        f"{prefix}_anchor_text": text_from_tokens(words, anchor_indexes),
        f"{prefix}_anchor_tokens": json.dumps(anchor_indexes, ensure_ascii=False, separators=(",", ":")),
        f"{prefix}_desc_text": text_from_tokens(words, desc_indexes),
        f"{prefix}_desc_tokens": json.dumps(desc_indexes, ensure_ascii=False, separators=(",", ":")),
        f"{prefix}_map_status": mapped["status"],
        f"{prefix}_ud": summary,
    }


def token_set_relation(left: list[int], right: list[int]) -> str:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return "both_empty"
    if not left_set:
        return "source_empty"
    if not right_set:
        return "target_empty"
    if left_set == right_set:
        return "same_desc_tokens"
    if right_set < left_set:
        return "source_contains_target"
    if left_set < right_set:
        return "target_contains_source"
    if left_set & right_set:
        return "overlap_desc_tokens"
    if max(left_set) < min(right_set):
        return "source_before_target"
    if max(right_set) < min(left_set):
        return "target_before_source"
    return "disjoint_desc_tokens"


def annotate_pair(case: dict[str, Any], pair: dict[str, Any], pred_row: dict[str, Any], ud_row: dict[str, Any], pair_index: int) -> dict[str, Any] | None:
    gold_bundle = pred_row.get("gold_lingua_graph") or {}
    pred_bundle = pred_row.get("pred_lingua_graph") or {}
    label = str(pair["label"])
    gold_src_idx, gold_tgt_idx, gold_resolve = resolve_edge_indexes(gold_bundle, pair["gold_source"], label, pair["gold_target"])
    pred_src_idx, pred_tgt_idx, pred_resolve = resolve_edge_indexes(pred_bundle, pair["pred_source"], label, pair["pred_target"])

    gold_src = endpoint_record(gold_bundle, ud_row, gold_src_idx, pair["gold_source"], "gold_source")
    gold_tgt = endpoint_record(gold_bundle, ud_row, gold_tgt_idx, pair["gold_target"], "gold_target")
    pred_src = endpoint_record(pred_bundle, ud_row, pred_src_idx, pair["pred_source"], "pred_source")
    pred_tgt = endpoint_record(pred_bundle, ud_row, pred_tgt_idx, pair["pred_target"], "pred_target")
    gold_desc_relation = token_set_relation(
        json.loads(gold_src["gold_source_desc_tokens"]),
        json.loads(gold_tgt["gold_target_desc_tokens"]),
    )
    pred_desc_relation = token_set_relation(
        json.loads(pred_src["pred_source_desc_tokens"]),
        json.loads(pred_tgt["pred_target_desc_tokens"]),
    )

    ud_words = flatten_ud_words(ud_row)
    words_by_id = {int(word["global_id"]): word for word in ud_words}
    gold_relation = ud_relation(gold_src["gold_source_ud"]["id"], gold_tgt["gold_target_ud"]["id"], words_by_id)
    pred_relation = ud_relation(pred_src["pred_source_ud"]["id"], pred_tgt["pred_target_ud"]["id"], words_by_id)
    gold_pattern = ud_pattern(gold_src["gold_source_ud"], gold_tgt["gold_target_ud"], gold_relation)
    pred_pattern = ud_pattern(pred_src["pred_source_ud"], pred_tgt["pred_target_ud"], pred_relation)
    gold_lex_pattern = ud_pattern(gold_src["gold_source_ud"], gold_tgt["gold_target_ud"], gold_relation, lexical=True)
    pred_lex_pattern = ud_pattern(pred_src["pred_source_ud"], pred_tgt["pred_target_ud"], pred_relation, lexical=True)

    return {
        "split": case["split"],
        "line": case["line"],
        "idx": case["idx"],
        "pair_index": pair_index,
        "label": label,
        "label_group": label_group(label),
        "sentence": case["sentence"],
        "gold_resolve": gold_resolve,
        "pred_resolve": pred_resolve,
        "gold_desc_relation": gold_desc_relation,
        "pred_desc_relation": pred_desc_relation,
        **gold_src,
        **gold_tgt,
        **pred_src,
        **pred_tgt,
        "gold_ud_pattern": gold_pattern,
        "gold_cluster_pattern": f"{gold_pattern} | desc:{gold_desc_relation}",
        "pred_ud_pattern": pred_pattern,
        "gold_ud_lex_pattern": gold_lex_pattern,
        "pred_ud_lex_pattern": pred_lex_pattern,
        "lingua_gold_signature": f"{label}|{gold_src['gold_source_node_type']}->{gold_tgt['gold_target_node_type']}",
        "lingua_pred_signature": f"{label}|{pred_src['pred_source_node_type']}->{pred_tgt['pred_target_node_type']}",
    }


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "line",
        "idx",
        "pair_index",
        "label",
        "label_group",
        "gold_resolve",
        "pred_resolve",
        "gold_desc_relation",
        "pred_desc_relation",
        "gold_cluster_pattern",
        "gold_ud_pattern",
        "pred_ud_pattern",
        "gold_ud_lex_pattern",
        "pred_ud_lex_pattern",
        "lingua_gold_signature",
        "lingua_pred_signature",
        "gold_source_node_text",
        "gold_source_node_type",
        "gold_source_anchor_text",
        "gold_source_anchor_tokens",
        "gold_source_desc_text",
        "gold_target_node_text",
        "gold_target_node_type",
        "gold_target_anchor_text",
        "gold_target_anchor_tokens",
        "gold_target_desc_text",
        "pred_source_node_text",
        "pred_source_node_type",
        "pred_source_anchor_text",
        "pred_source_anchor_tokens",
        "pred_source_desc_text",
        "pred_target_node_text",
        "pred_target_node_type",
        "pred_target_anchor_text",
        "pred_target_anchor_tokens",
        "pred_target_desc_text",
        "sentence",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def build_cluster_rows(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if args.label_group != "all" and record["label_group"] != args.label_group:
            continue
        if args.skip_unmapped and "NONE[NONE->ROOT]" in record["gold_ud_pattern"]:
            continue
        clusters[record["gold_cluster_pattern"]].append(record)

    rows = []
    for pattern, items in clusters.items():
        label_counts = Counter(item["label"] for item in items)
        group_counts = Counter(item["label_group"] for item in items)
        signature_counts = Counter(item["lingua_gold_signature"] for item in items)
        pred_signature_counts = Counter(item["lingua_pred_signature"] for item in items)
        split_counts = Counter(str(item["split"]) for item in items)
        if len(items) < args.min_count or len(signature_counts) < args.min_signatures:
            continue
        score = len(signature_counts) * math.log2(len(items) + 1) * (1.0 + entropy(label_counts))
        rows.append(
            {
                "ud_pattern": pattern,
                "count": len(items),
                "divergence_score": score,
                "label_entropy": entropy(label_counts),
                "signature_entropy": entropy(signature_counts),
                "distinct_labels": len(label_counts),
                "distinct_label_groups": len(group_counts),
                "distinct_lingua_signatures": len(signature_counts),
                "split_counts": compact_counter(split_counts),
                "label_counts": compact_counter(label_counts),
                "label_group_counts": compact_counter(group_counts),
                "lingua_signature_counts": compact_counter(signature_counts),
                "pred_signature_counts": compact_counter(pred_signature_counts),
            }
        )
    rows.sort(key=lambda row: (-row["divergence_score"], -row["count"], row["ud_pattern"]))
    return rows, clusters


def write_cluster_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["ud_pattern"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], clusters: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> None:
    lines = [
        "# UD pattern consistent, Lingua graph divergent",
        "",
        f"Layer: `{args.layer}`",
        f"Label group: `{args.label_group}`",
        f"Cases: `{args.cases}`",
        f"Prediction base: `{args.prediction_base}`",
        f"UD cache: `{args.ud_base}`",
        "",
        "Endpoint mapping uses Lingua graph descendant tokens before UD alignment.",
        "",
        "## Top Divergent Patterns",
        "",
        "| rank | count | labels | signatures | UD + desc pattern | top Lingua signatures |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for index, row in enumerate(rows[: args.top_clusters], start=1):
        lines.append(
            f"| {index} | {row['count']} | {row['distinct_labels']} | {row['distinct_lingua_signatures']} | "
            f"`{row['ud_pattern']}` | {row['lingua_signature_counts']} |"
        )

    lines.extend(["", "## Examples"])
    for index, row in enumerate(rows[: args.top_clusters], start=1):
        pattern = row["ud_pattern"]
        items = clusters[pattern]
        sig_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            sig_groups[item["lingua_gold_signature"]].append(item)
        lines.extend(
            [
                "",
                f"### {index}. `{pattern}`",
                "",
                f"- count: {row['count']}; labels: {row['label_counts']}",
                f"- signatures: {row['lingua_signature_counts']}",
            ]
        )
        for sig, sig_items in sorted(sig_groups.items(), key=lambda item: (-len(item[1]), item[0]))[: args.examples_per_cluster]:
            lines.append(f"- `{sig}` ({len(sig_items)})")
            for item in sig_items[:2]:
                lines.append(f"  - {item['split']}:{item['line']} `{item['sentence']}`")
                lines.append(
                    "    - gold Lingua: "
                    f"{item['gold_source_node_text']}/{item['gold_source_node_type']}"
                    f" [{item['gold_source_desc_text']}] --{item['label']}--> "
                    f"{item['gold_target_node_text']}/{item['gold_target_node_type']}"
                    f" [{item['gold_target_desc_text']}]"
                )
                lines.append(f"    - UD lexical: `{item['gold_ud_lex_pattern']}`")
                lines.append(
                    "    - pred Lingua: "
                    f"{item['pred_source_node_text']}/{item['pred_source_node_type']}"
                    f" [{item['pred_source_desc_text']}] --> "
                    f"{item['pred_target_node_text']}/{item['pred_target_node_type']}"
                    f" [{item['pred_target_desc_text']}]"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ud_rows = load_ud(args.ud_base)
    pred_rows = load_predictions(args.prediction_base)
    records = []
    for case in read_jsonl(args.cases):
        layer_info = case.get(args.layer) or {}
        pairs = layer_info.get("same_label_scope_attachment") or []
        if not pairs:
            continue
        key = (str(case["split"]), int(case["line"]))
        pred_row = pred_rows.get(key)
        ud_row = ud_rows.get(key)
        if pred_row is None or ud_row is None or ud_row.get("error"):
            continue
        for pair_index, pair in enumerate(pairs):
            record = annotate_pair(case, pair, pred_row, ud_row, pair_index)
            if record is not None:
                records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "ud_lingua_graph_scope_pair_records.tsv"
    write_records(detail_path, records)
    cluster_rows, clusters = build_cluster_rows(records, args)
    cluster_path = args.output_dir / "ud_lingua_graph_divergent_patterns.tsv"
    report_path = args.output_dir / "ud_lingua_graph_divergent_patterns.md"
    write_cluster_tsv(cluster_path, cluster_rows)
    write_report(report_path, cluster_rows, clusters, args)
    summary = {
        "cases": str(args.cases),
        "prediction_base": str(args.prediction_base),
        "ud_base": str(args.ud_base),
        "output_dir": str(args.output_dir),
        "layer": args.layer,
        "label_group": args.label_group,
        "records": len(records),
        "clusters": len(cluster_rows),
        "detail_tsv": str(detail_path),
        "cluster_tsv": str(cluster_path),
        "report": str(report_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("reports/v13_overwrite_latest_nonexact_patterns/nonexact_cases.jsonl"))
    parser.add_argument("--prediction_base", type=Path, default=Path("reports/v13_overwrite_latest_prediction_dump"))
    parser.add_argument("--ud_base", type=Path, default=Path("reports/v13_overwrite_latest_ud_stanza"))
    parser.add_argument("--output_dir", type=Path, default=Path("reports/v13_overwrite_latest_ud_lingua_scope_variation/lingua_graph_desc"))
    parser.add_argument("--layer", choices=["lingua"], default="lingua")
    parser.add_argument("--label_group", choices=["all", "modification", "predicate_args", "punctuation", "expression_list", "other"], default="modification")
    parser.add_argument("--min_count", type=int, default=8)
    parser.add_argument("--min_signatures", type=int, default=3)
    parser.add_argument("--top_clusters", type=int, default=30)
    parser.add_argument("--examples_per_cluster", type=int, default=6)
    parser.add_argument("--skip_unmapped", action="store_true")
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
