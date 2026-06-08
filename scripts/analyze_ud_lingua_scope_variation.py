#!/usr/bin/env python3
"""Find local UD patterns with divergent Lingua annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_ud(base: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for split in ("train", "dev", "test"):
        path = base / f"{split}.jsonl"
        for row in read_jsonl(path):
            rows[(str(row["split"]), int(row["line"]))] = row
    return rows


def parse_spans(value: str) -> list[list[int]]:
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    spans = []
    for span in data:
        if isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span):
            spans.append([int(span[0]), int(span[1])])
    return spans


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


def node_char_span(spans: list[list[int]], words: list[str]) -> tuple[int, int] | None:
    word_spans = lingua_word_char_spans(words)
    starts = []
    ends = []
    for start, end in spans:
        if start < 0 or end >= len(word_spans) or start > end:
            continue
        starts.append(word_spans[start][2])
        ends.append(word_spans[end][3])
    if not starts or not ends:
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


def map_node_to_ud(node_spans: str, ud_row: dict[str, Any]) -> dict[str, Any]:
    spans = parse_spans(node_spans)
    words = ud_row.get("words") or []
    char_span = node_char_span(spans, words)
    ud_words = flatten_ud_words(ud_row)
    words_by_id = {int(word["global_id"]): word for word in ud_words}
    if char_span is None:
        return {"status": "no_lingua_span", "char_span": None, "token_ids": [], "root_id": None}

    token_ids = []
    for word in ud_words:
        start = word.get("start_char")
        end = word.get("end_char")
        if start is None or end is None:
            continue
        if overlap(char_span, (int(start), int(end))) > 0:
            token_ids.append(int(word["global_id"]))
    root_id = choose_ud_root(token_ids, words_by_id)
    return {"status": "ok" if root_id is not None else "no_ud_overlap", "char_span": char_span, "token_ids": token_ids, "root_id": root_id}


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
        "head_deprel": "root" if head == 0 or head_word is None else head_word.get("deprel") or "NONE",
    }


def ancestor_map(token_id: int, words_by_id: dict[int, dict[str, Any]]) -> dict[int, tuple[int, list[str]]]:
    result = {token_id: (0, [])}
    current = token_id
    path: list[str] = []
    depth = 0
    while current in words_by_id:
        head = int(words_by_id[current].get("global_head") or 0)
        if head == 0:
            result[0] = (depth + 1, path + [f"↑{words_by_id[current].get('deprel')}"])
            break
        path = path + [f"↑{words_by_id[current].get('deprel')}"]
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


def annotate_edge(prefix: str, row: dict[str, str], ud_row: dict[str, Any]) -> dict[str, Any]:
    ud_words = flatten_ud_words(ud_row)
    words_by_id = {int(word["global_id"]): word for word in ud_words}
    src_map = map_node_to_ud(row[f"{prefix}_source_spans"], ud_row)
    tgt_map = map_node_to_ud(row[f"{prefix}_target_spans"], ud_row)
    src = token_summary(src_map["root_id"], words_by_id)
    tgt = token_summary(tgt_map["root_id"], words_by_id)
    relation = ud_relation(src_map["root_id"], tgt_map["root_id"], words_by_id)
    return {
        f"{prefix}_ud_source": src,
        f"{prefix}_ud_target": tgt,
        f"{prefix}_ud_relation": relation,
        f"{prefix}_ud_pattern": ud_pattern(src, tgt, relation, lexical=False),
        f"{prefix}_ud_lex_pattern": ud_pattern(src, tgt, relation, lexical=True),
        f"{prefix}_ud_source_tokens": src_map["token_ids"],
        f"{prefix}_ud_target_tokens": tgt_map["token_ids"],
    }


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def compact_counter(counter: Counter[str], limit: int = 20) -> str:
    return " | ".join(f"{key}:{count}" for key, count in counter.most_common(limit))


def run(args: argparse.Namespace) -> dict[str, Any]:
    ud_rows = load_ud(args.ud_base)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with args.queue_tsv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["split"], int(row["line"]))
            ud_row = ud_rows.get(key)
            if ud_row is None or ud_row.get("error"):
                continue
            gold = annotate_edge("gold", row, ud_row)
            pred = annotate_edge("pred", row, ud_row)
            lingua_sig = f"{row['label']}|{row['gold_source_type']}->{row['gold_target_type']}"
            pred_sig = f"{row['label']}|{row['pred_source_type']}->{row['pred_target_type']}"
            records.append(
                {
                    **row,
                    **gold,
                    **pred,
                    "lingua_gold_signature": lingua_sig,
                    "lingua_pred_signature": pred_sig,
                }
            )

    detail_path = args.output_dir / "ud_scope_pair_records.tsv"
    fieldnames = [
        "split",
        "line",
        "pair_index",
        "label",
        "label_group",
        "pair_category",
        "endpoint_diff",
        "lingua_gold_signature",
        "lingua_pred_signature",
        "gold_ud_pattern",
        "pred_ud_pattern",
        "gold_ud_lex_pattern",
        "pred_ud_lex_pattern",
        "gold_source_text",
        "gold_target_text",
        "pred_source_text",
        "pred_target_text",
        "gold_source_type",
        "gold_target_type",
        "pred_source_type",
        "pred_target_type",
        "sentence",
    ]
    with detail_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pattern = record["gold_ud_pattern"]
        clusters[pattern].append(record)

    cluster_rows = []
    for pattern, rows in clusters.items():
        label_counts = Counter(row["label"] for row in rows)
        group_counts = Counter(row["label_group"] for row in rows)
        signature_counts = Counter(row["lingua_gold_signature"] for row in rows)
        pred_signature_counts = Counter(row["lingua_pred_signature"] for row in rows)
        split_counts = Counter(row["split"] for row in rows)
        if len(rows) < args.min_count:
            continue
        if len(signature_counts) < args.min_signatures:
            continue
        score = len(signature_counts) * math.log2(len(rows) + 1) * (1.0 + entropy(label_counts))
        cluster_rows.append(
            {
                "ud_pattern": pattern,
                "count": len(rows),
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
    cluster_rows.sort(key=lambda row: (-row["divergence_score"], -row["count"], row["ud_pattern"]))

    cluster_path = args.output_dir / "ud_lingua_divergent_patterns.tsv"
    with cluster_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cluster_rows[0].keys()) if cluster_rows else ["ud_pattern"], delimiter="\t")
        writer.writeheader()
        writer.writerows(cluster_rows)

    report_path = args.output_dir / "ud_lingua_divergent_patterns.md"
    write_report(report_path, cluster_rows, clusters, args)

    summary = {
        "queue_tsv": str(args.queue_tsv),
        "ud_base": str(args.ud_base),
        "output_dir": str(args.output_dir),
        "records": len(records),
        "clusters": len(cluster_rows),
        "detail_tsv": str(detail_path),
        "cluster_tsv": str(cluster_path),
        "report": str(report_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def write_report(path: Path, cluster_rows: list[dict[str, Any]], clusters: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> None:
    lines = [
        "# UD pattern 一致但 Lingua 标注分裂的 scope cases",
        "",
        f"输入队列：`{args.queue_tsv}`",
        f"UD 缓存：`{args.ud_base}`",
        "",
        "这里的 UD pattern 是 lexeme-free 的局部模式：source/token 的 UPOS、deprel、head UPOS，加上 source-target 在 UD 树里的局部关系。",
        "",
        "## Top Divergent Patterns",
        "",
        "| rank | count | distinct labels | distinct signatures | UD pattern | top Lingua signatures |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for i, row in enumerate(cluster_rows[: args.top_clusters], start=1):
        lines.append(
            f"| {i} | {row['count']} | {row['distinct_labels']} | {row['distinct_lingua_signatures']} | "
            f"`{row['ud_pattern']}` | {row['lingua_signature_counts']} |"
        )

    lines.extend(["", "## Examples"])
    for i, row in enumerate(cluster_rows[: args.top_clusters], start=1):
        pattern = row["ud_pattern"]
        rows = clusters[pattern]
        sig_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in rows:
            sig_groups[item["lingua_gold_signature"]].append(item)
        lines.extend(
            [
                "",
                f"### {i}. `{pattern}`",
                "",
                f"- count: {row['count']}; labels: {row['label_counts']}",
                f"- signatures: {row['lingua_signature_counts']}",
            ]
        )
        for sig, sig_rows in sorted(sig_groups.items(), key=lambda item: (-len(item[1]), item[0]))[: args.examples_per_cluster]:
            lines.append(f"- `{sig}` ({len(sig_rows)})")
            for item in sig_rows[:2]:
                lines.append(
                    f"  - {item['split']}:{item['line']} `{item['sentence']}`"
                )
                lines.append(
                    "    - Lingua gold: "
                    f"{item['gold_source_text']}/{item['gold_source_type']} --{item['label']}--> "
                    f"{item['gold_target_text']}/{item['gold_target_type']}"
                )
                lines.append(
                    "    - UD lexical: "
                    f"`{item['gold_ud_lex_pattern']}`"
                )
                lines.append(
                    "    - pred: "
                    f"{item['pred_source_text']}/{item['pred_source_type']} --> "
                    f"{item['pred_target_text']}/{item['pred_target_type']}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue_tsv",
        type=Path,
        default=Path("reports/v13_overwrite_latest_scope_error_queues/all_scope_attachment_pairs.tsv"),
    )
    parser.add_argument(
        "--ud_base",
        type=Path,
        default=Path("reports/v13_overwrite_latest_ud_stanza"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/v13_overwrite_latest_ud_lingua_scope_variation"),
    )
    parser.add_argument("--min_count", type=int, default=8)
    parser.add_argument("--min_signatures", type=int, default=4)
    parser.add_argument("--top_clusters", type=int, default=25)
    parser.add_argument("--examples_per_cluster", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
