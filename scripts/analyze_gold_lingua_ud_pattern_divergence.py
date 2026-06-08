#!/usr/bin/env python3
"""Analyze gold Lingua edges that share a UD pattern but differ in Lingua annotation."""

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
    indexes: set[int] = set()
    for span in spans:
        if (
            isinstance(span, list)
            and len(span) == 2
            and isinstance(span[0], int)
            and isinstance(span[1], int)
            and span[0] <= span[1]
        ):
            indexes.update(range(span[0], span[1] + 1))
    return sorted(indexes)


def text_from_tokens(words: list[str], indexes: list[int]) -> str:
    return "".join(words[index] for index in indexes if 0 <= index < len(words)).strip()


def graph_payload(bundle: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    graph = bundle.get("graph") or {}
    return bundle.get("words") or [], graph.get("nodes") or [], graph.get("edges") or []


def outgoing_map(edges: list[dict[str, Any]]) -> dict[int, list[int]]:
    outgoing: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        outgoing[int(edge["start"])].append(int(edge["end"]))
    return outgoing


def descendant_tokens(nodes: list[dict[str, Any]], outgoing: dict[int, list[int]], node_index: int) -> list[int]:
    seen: set[int] = set()
    tokens: set[int] = set()
    stack = [node_index]
    while stack:
        index = stack.pop()
        if index in seen or index < 0 or index >= len(nodes):
            continue
        seen.add(index)
        tokens.update(node_token_indexes(nodes[index]))
        stack.extend(outgoing.get(index, []))
    return sorted(tokens)


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


def map_tokens_to_ud(indexes: list[int], ud_row: dict[str, Any]) -> tuple[list[int], int | None]:
    ud_words = flatten_ud_words(ud_row)
    words_by_id = {int(word["global_id"]): word for word in ud_words}
    char_span = token_indexes_char_span(indexes, ud_row.get("words") or [])
    if char_span is None:
        return [], None
    token_ids = []
    for word in ud_words:
        start = word.get("start_char")
        end = word.get("end_char")
        if start is None or end is None:
            continue
        if overlap(char_span, (int(start), int(end))) > 0:
            token_ids.append(int(word["global_id"]))
    return token_ids, choose_ud_root(token_ids, words_by_id)


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


def ancestor_map(token_id: int, words_by_id: dict[int, dict[str, Any]]) -> dict[int, int]:
    result = {token_id: 0}
    current = token_id
    depth = 0
    while current in words_by_id:
        head = int(words_by_id[current].get("global_head") or 0)
        depth += 1
        if head == 0:
            result[0] = depth
            break
        if head in result:
            break
        result[head] = depth
        current = head
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
    lca = min(common, key=lambda node: src_anc[node] + tgt_anc[node])
    lca_word = words_by_id.get(lca)
    lca_upos = "ROOT" if lca == 0 or lca_word is None else lca_word.get("upos")
    return f"path_len={src_anc[lca] + tgt_anc[lca]}:lca={lca_upos}"


def ud_pattern(src: dict[str, Any], tgt: dict[str, Any], relation: str, lexical: bool = False) -> str:
    if lexical:
        return (
            f"{src['lemma'].lower()}/{src['upos']}[{src['deprel']}->{src['head_upos']}] "
            f"--{relation}--> {tgt['lemma'].lower()}/{tgt['upos']}[{tgt['deprel']}->{tgt['head_upos']}]"
        )
    return f"{src['upos']}[{src['deprel']}->{src['head_upos']}] --{relation}--> {tgt['upos']}[{tgt['deprel']}->{tgt['head_upos']}]"


def label_group(label: str) -> str:
    if label in {"as:modification", "@modification_of", "body", "variable"}:
        return "modification"
    if label.startswith("pred.arg."):
        return "predicate_args"
    if label in {"as:punctuation", "punctuation"}:
        return "punctuation"
    if label in {"func.arg", "@expression", "@list"}:
        return "expression_list"
    if label in {"as:discourse", "@discourse_of"}:
        return "discourse"
    return "other"


def load_ud(base: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    for split in SPLITS:
        for row in read_jsonl(base / f"{split}.jsonl"):
            rows[(split, int(row["line"]))] = row
    return rows


def edge_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    ud_rows = load_ud(args.ud_base)
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        for row in read_jsonl(args.prediction_base / split / "predictions.jsonl"):
            line = int(row["line"])
            ud_row = ud_rows.get((split, line))
            if not ud_row or ud_row.get("error"):
                continue
            bundle = row.get("gold_lingua_graph") or {}
            words, nodes, edges = graph_payload(bundle)
            outgoing = outgoing_map(edges)
            ud_words = flatten_ud_words(ud_row)
            words_by_id = {int(word["global_id"]): word for word in ud_words}
            for edge_index, edge in enumerate(edges):
                src_i = int(edge["start"])
                tgt_i = int(edge["end"])
                if src_i >= len(nodes) or tgt_i >= len(nodes):
                    continue
                src_node = nodes[src_i]
                tgt_node = nodes[tgt_i]
                src_own = node_token_indexes(src_node)
                tgt_own = node_token_indexes(tgt_node)
                src_desc = descendant_tokens(nodes, outgoing, src_i)
                tgt_desc = descendant_tokens(nodes, outgoing, tgt_i)
                src_anchor = src_own or src_desc
                tgt_anchor = tgt_own or tgt_desc
                src_ud_ids, src_root = map_tokens_to_ud(src_anchor, ud_row)
                tgt_ud_ids, tgt_root = map_tokens_to_ud(tgt_anchor, ud_row)
                src_ud = token_summary(src_root, words_by_id)
                tgt_ud = token_summary(tgt_root, words_by_id)
                relation = ud_relation(src_root, tgt_root, words_by_id)
                label = str(edge["label"])
                signature = f"{label}|{src_node.get('type')}->{tgt_node.get('type')}"
                pattern = ud_pattern(src_ud, tgt_ud, relation, lexical=False)
                desc_relation = token_set_relation(src_desc, tgt_desc)
                records.append(
                    {
                        "split": split,
                        "line": line,
                        "idx": row.get("idx"),
                        "edge_index": edge_index,
                        "sentence": row.get("sentence") or "",
                        "label": label,
                        "label_group": label_group(label),
                        "lingua_signature": signature,
                        "source_type": src_node.get("type"),
                        "target_type": tgt_node.get("type"),
                        "source_text": text_from_tokens(words, src_own) or str(src_node.get("spans") or ""),
                        "target_text": text_from_tokens(words, tgt_own) or str(tgt_node.get("spans") or ""),
                        "source_anchor_text": text_from_tokens(words, src_anchor),
                        "target_anchor_text": text_from_tokens(words, tgt_anchor),
                        "source_desc_text": text_from_tokens(words, src_desc),
                        "target_desc_text": text_from_tokens(words, tgt_desc),
                        "source_anchor_mode": "own" if src_own else "desc",
                        "target_anchor_mode": "own" if tgt_own else "desc",
                        "desc_relation": desc_relation,
                        "ud_pattern": pattern,
                        "ud_desc_pattern": f"{pattern} | desc:{desc_relation}",
                        "ud_lex_pattern": ud_pattern(src_ud, tgt_ud, relation, lexical=True),
                        "source_ud_token_ids": json.dumps(src_ud_ids, separators=(",", ":")),
                        "target_ud_token_ids": json.dumps(tgt_ud_ids, separators=(",", ":")),
                    }
                )
    return records


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def compact_counter(counter: Counter[str], limit: int = 20) -> str:
    return " | ".join(f"{key}:{count}" for key, count in counter.most_common(limit))


def cluster_records(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if args.label_group != "all" and record["label_group"] != args.label_group:
            continue
        if args.anchor_mode != "all" and (record["source_anchor_mode"] != args.anchor_mode or record["target_anchor_mode"] != args.anchor_mode):
            continue
        key = record["ud_desc_pattern"] if args.include_desc_relation else record["ud_pattern"]
        clusters[key].append(record)

    rows = []
    for pattern, items in clusters.items():
        label_counts = Counter(item["label"] for item in items)
        group_counts = Counter(item["label_group"] for item in items)
        sig_counts = Counter(item["lingua_signature"] for item in items)
        split_counts = Counter(item["split"] for item in items)
        if len(items) < args.min_count:
            continue
        if len(label_counts) < args.min_labels and len(sig_counts) < args.min_signatures:
            continue
        score = math.log2(len(items) + 1) * (1 + entropy(label_counts)) * (len(label_counts) + 0.25 * len(sig_counts))
        rows.append(
            {
                "pattern": pattern,
                "count": len(items),
                "divergence_score": score,
                "label_entropy": entropy(label_counts),
                "signature_entropy": entropy(sig_counts),
                "distinct_labels": len(label_counts),
                "distinct_label_groups": len(group_counts),
                "distinct_signatures": len(sig_counts),
                "split_counts": compact_counter(split_counts),
                "label_counts": compact_counter(label_counts),
                "label_group_counts": compact_counter(group_counts),
                "signature_counts": compact_counter(sig_counts),
            }
        )
    rows.sort(key=lambda row: (-row["divergence_score"], -row["distinct_labels"], -row["count"], row["pattern"]))
    return rows, clusters


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["pattern"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, cluster_rows: list[dict[str, Any]], clusters: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> None:
    lines = [
        "# UD 相同但 Lingua gold 标注分裂的模式",
        "",
        "本报告只看 gold Lingua graph，不使用 predicted graph。",
        "UD anchor 规则：Lingua 节点有自身 token 时用自身 token；只有无词面 wrapper 才用 DESC token。",
        "",
        f"- label_group: `{args.label_group}`",
        f"- anchor_mode: `{args.anchor_mode}`",
        f"- include_desc_relation: `{args.include_desc_relation}`",
        "",
        "## Top Patterns",
        "",
        "| rank | count | labels | signatures | pattern | label counts |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for i, row in enumerate(cluster_rows[: args.top_clusters], start=1):
        lines.append(
            f"| {i} | {row['count']} | {row['distinct_labels']} | {row['distinct_signatures']} | "
            f"`{row['pattern']}` | {row['label_counts']} |"
        )

    lines.extend(["", "## Examples"])
    for i, row in enumerate(cluster_rows[: args.top_clusters], start=1):
        pattern = row["pattern"]
        items = clusters[pattern]
        lines.extend(
            [
                "",
                f"### {i}. `{pattern}`",
                "",
                f"- count: {row['count']}",
                f"- labels: {row['label_counts']}",
                f"- signatures: {row['signature_counts']}",
            ]
        )
        sig_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            sig_groups[item["lingua_signature"]].append(item)
        for sig, sig_items in sorted(sig_groups.items(), key=lambda item: (-len(item[1]), item[0]))[: args.examples_per_signature]:
            lines.append(f"- `{sig}` ({len(sig_items)})")
            for item in sig_items[: args.examples_per_signature_item]:
                lines.append(f"  - {item['split']}:{item['line']} `{item['sentence']}`")
                lines.append(
                    "    - Lingua: "
                    f"{item['source_text'] or item['source_anchor_text']}/{item['source_type']}"
                    f" [{item['source_desc_text']}] --{item['label']}--> "
                    f"{item['target_text'] or item['target_anchor_text']}/{item['target_type']}"
                    f" [{item['target_desc_text']}]"
                )
                lines.append(f"    - UD: `{item['ud_lex_pattern']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = edge_records(args)
    record_fields = [
        "split",
        "line",
        "idx",
        "edge_index",
        "label",
        "label_group",
        "lingua_signature",
        "ud_pattern",
        "ud_desc_pattern",
        "ud_lex_pattern",
        "desc_relation",
        "source_type",
        "target_type",
        "source_text",
        "target_text",
        "source_anchor_text",
        "target_anchor_text",
        "source_desc_text",
        "target_desc_text",
        "source_anchor_mode",
        "target_anchor_mode",
        "sentence",
    ]
    detail_path = args.output_dir / "gold_lingua_ud_edge_records.tsv"
    write_tsv(detail_path, records, record_fields)
    cluster_rows, clusters = cluster_records(records, args)
    cluster_path = args.output_dir / "gold_lingua_ud_divergent_patterns.tsv"
    report_path = args.output_dir / "gold_lingua_ud_divergent_patterns.md"
    write_tsv(cluster_path, cluster_rows)
    write_report(report_path, cluster_rows, clusters, args)
    summary = {
        "records": len(records),
        "clusters": len(cluster_rows),
        "prediction_base": str(args.prediction_base),
        "ud_base": str(args.ud_base),
        "output_dir": str(args.output_dir),
        "detail_tsv": str(detail_path),
        "cluster_tsv": str(cluster_path),
        "report": str(report_path),
        "label_group": args.label_group,
        "anchor_mode": args.anchor_mode,
        "include_desc_relation": args.include_desc_relation,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_base", type=Path, default=Path("reports/v13_overwrite_latest_prediction_dump"))
    parser.add_argument("--ud_base", type=Path, default=Path("reports/v13_overwrite_latest_ud_stanza"))
    parser.add_argument("--output_dir", type=Path, default=Path("reports/v13_overwrite_latest_gold_lingua_ud_divergence"))
    parser.add_argument("--label_group", choices=["all", "modification", "predicate_args", "punctuation", "expression_list", "discourse", "other"], default="all")
    parser.add_argument("--anchor_mode", choices=["all", "own", "desc"], default="all")
    parser.add_argument("--include_desc_relation", action="store_true")
    parser.add_argument("--min_count", type=int, default=8)
    parser.add_argument("--min_labels", type=int, default=2)
    parser.add_argument("--min_signatures", type=int, default=4)
    parser.add_argument("--top_clusters", type=int, default=40)
    parser.add_argument("--examples_per_signature", type=int, default=8)
    parser.add_argument("--examples_per_signature_item", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
