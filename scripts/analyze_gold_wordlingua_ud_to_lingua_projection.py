#!/usr/bin/env python3
"""Find gold word-lingua UD patterns and project them back to Lingua graphs."""

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


def load_ud(base: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    for split in SPLITS:
        for row in read_jsonl(base / f"{split}.jsonl"):
            rows[(split, int(row["line"]))] = row
    return rows


def graph_payload(bundle: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    graph = bundle.get("graph") or {}
    return bundle.get("words") or [], graph.get("nodes") or [], graph.get("edges") or []


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
    return "".join(words[index] for index in sorted(set(indexes)) if 0 <= index < len(words)).strip()


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
    if label in {"as:punctuation", "punctuation", "next_word"}:
        return "punctuation"
    if label in {"func.arg", "@expression", "@list"}:
        return "expression_list"
    if label in {"as:discourse", "@discourse_of"}:
        return "discourse"
    return "other"


def node_infos(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    words, nodes, edges = graph_payload(bundle)
    outgoing = outgoing_map(edges)
    infos = []
    for index, node in enumerate(nodes):
        own = node_token_indexes(node)
        desc = descendant_tokens(nodes, outgoing, index)
        spans = node.get("spans")
        text = spans if isinstance(spans, str) else text_from_tokens(words, own)
        infos.append(
            {
                "index": index,
                "type": node.get("type") or "",
                "spans": spans,
                "own_tokens": own,
                "desc_tokens": desc,
                "text": str(text or ""),
                "own_text": text_from_tokens(words, own),
                "desc_text": text_from_tokens(words, desc),
            }
        )
    return infos


def project_node(infos: list[dict[str, Any]], token_ids: list[int]) -> dict[str, Any] | None:
    token_set = set(token_ids)
    if not token_set:
        return None
    scored: list[tuple[int, int, int, dict[str, Any]]] = []
    for info in infos:
        own = set(info["own_tokens"])
        desc = set(info["desc_tokens"])
        if own == token_set:
            scored.append((0, len(own), info["index"], info))
        elif desc == token_set:
            scored.append((1, len(desc), info["index"], info))
        elif token_set < own:
            scored.append((2, len(own), info["index"], info))
        elif token_set < desc:
            scored.append((3, len(desc), info["index"], info))
        elif token_set & own:
            scored.append((4, len(own), info["index"], info))
        elif token_set & desc:
            scored.append((5, len(desc), info["index"], info))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return scored[0][3]


def projection_kind(
    lingua_edges: list[dict[str, Any]],
    src_info: dict[str, Any] | None,
    tgt_info: dict[str, Any] | None,
) -> tuple[str, str]:
    if src_info is None or tgt_info is None:
        return "unresolved", ""
    src_i = int(src_info["index"])
    tgt_i = int(tgt_info["index"])
    if src_i == tgt_i:
        return "absorbed_same_lingua_node", src_info["type"]
    direct = [edge.get("label") for edge in lingua_edges if int(edge["start"]) == src_i and int(edge["end"]) == tgt_i]
    if direct:
        labels = ",".join(str(x) for x in sorted(set(direct)))
        return "direct_lingua_edge", labels
    reverse = [edge.get("label") for edge in lingua_edges if int(edge["start"]) == tgt_i and int(edge["end"]) == src_i]
    if reverse:
        labels = ",".join(str(x) for x in sorted(set(reverse)))
        return "reverse_lingua_edge", labels
    src_out = {int(edge["end"]): edge.get("label") for edge in lingua_edges if int(edge["start"]) == src_i}
    tgt_out = {int(edge["end"]): edge.get("label") for edge in lingua_edges if int(edge["start"]) == tgt_i}
    shared_children = sorted(set(src_out) & set(tgt_out))
    if shared_children:
        labels = "|".join(f"{src_out[i]}+{tgt_out[i]}" for i in shared_children[:3])
        return "shared_child", labels
    src_in = {int(edge["start"]): edge.get("label") for edge in lingua_edges if int(edge["end"]) == src_i}
    tgt_in = {int(edge["start"]): edge.get("label") for edge in lingua_edges if int(edge["end"]) == tgt_i}
    shared_parents = sorted(set(src_in) & set(tgt_in))
    if shared_parents:
        labels = "|".join(f"{src_in[i]}+{tgt_in[i]}" for i in shared_parents[:3])
        return "shared_parent", labels
    return "separate_lingua_nodes", ""


def projection_signature(
    kind: str,
    relation_labels: str,
    src_info: dict[str, Any] | None,
    tgt_info: dict[str, Any] | None,
) -> str:
    src_type = "NONE" if src_info is None else src_info["type"]
    tgt_type = "NONE" if tgt_info is None else tgt_info["type"]
    label_part = f":{relation_labels}" if relation_labels else ""
    return f"{kind}{label_part}|{src_type}->{tgt_type}"


def edge_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    ud_rows = load_ud(args.ud_base)
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        for row in read_jsonl(args.prediction_base / split / "predictions.jsonl"):
            line = int(row["line"])
            ud_row = ud_rows.get((split, line))
            if not ud_row or ud_row.get("error"):
                continue
            word_bundle = row.get("gold_word_lingua_graph") or {}
            lingua_bundle = row.get("gold_lingua_graph") or {}
            words, word_nodes, word_edges = graph_payload(word_bundle)
            _, _, lingua_edges = graph_payload(lingua_bundle)
            l_infos = node_infos(lingua_bundle)
            ud_words = flatten_ud_words(ud_row)
            words_by_id = {int(word["global_id"]): word for word in ud_words}
            for edge_index, edge in enumerate(word_edges):
                src_i = int(edge["start"])
                tgt_i = int(edge["end"])
                if src_i >= len(word_nodes) or tgt_i >= len(word_nodes):
                    continue
                src_node = word_nodes[src_i]
                tgt_node = word_nodes[tgt_i]
                src_tokens = node_token_indexes(src_node)
                tgt_tokens = node_token_indexes(tgt_node)
                if not src_tokens or not tgt_tokens:
                    continue
                src_ud_ids, src_root = map_tokens_to_ud(src_tokens, ud_row)
                tgt_ud_ids, tgt_root = map_tokens_to_ud(tgt_tokens, ud_row)
                src_ud = token_summary(src_root, words_by_id)
                tgt_ud = token_summary(tgt_root, words_by_id)
                relation = ud_relation(src_root, tgt_root, words_by_id)
                label = str(edge["label"])
                src_l = project_node(l_infos, src_tokens)
                tgt_l = project_node(l_infos, tgt_tokens)
                kind, rel_labels = projection_kind(lingua_edges, src_l, tgt_l)
                ling_sig = projection_signature(kind, rel_labels, src_l, tgt_l)
                word_sig = f"{label}|{src_node.get('type')}->{tgt_node.get('type')}"
                records.append(
                    {
                        "split": split,
                        "line": line,
                        "idx": row.get("idx"),
                        "edge_index": edge_index,
                        "sentence": row.get("sentence") or "",
                        "word_label": label,
                        "word_label_group": label_group(label),
                        "word_signature": word_sig,
                        "ud_pattern": ud_pattern(src_ud, tgt_ud, relation, lexical=False),
                        "ud_lex_pattern": ud_pattern(src_ud, tgt_ud, relation, lexical=True),
                        "source_text": text_from_tokens(words, src_tokens),
                        "target_text": text_from_tokens(words, tgt_tokens),
                        "source_type": src_node.get("type"),
                        "target_type": tgt_node.get("type"),
                        "source_tokens": json.dumps(src_tokens, separators=(",", ":")),
                        "target_tokens": json.dumps(tgt_tokens, separators=(",", ":")),
                        "lingua_projection_kind": kind,
                        "lingua_projection_relation": rel_labels,
                        "lingua_projection_signature": ling_sig,
                        "lingua_source_index": "" if src_l is None else src_l["index"],
                        "lingua_target_index": "" if tgt_l is None else tgt_l["index"],
                        "lingua_source_type": "" if src_l is None else src_l["type"],
                        "lingua_target_type": "" if tgt_l is None else tgt_l["type"],
                        "lingua_source_text": "" if src_l is None else (src_l["own_text"] or src_l["text"]),
                        "lingua_target_text": "" if tgt_l is None else (tgt_l["own_text"] or tgt_l["text"]),
                        "lingua_source_desc_text": "" if src_l is None else src_l["desc_text"],
                        "lingua_target_desc_text": "" if tgt_l is None else tgt_l["desc_text"],
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
        if args.label_group != "all" and record["word_label_group"] != args.label_group:
            continue
        if args.skip_unmapped and "NONE[NONE->ROOT]" in record["ud_pattern"]:
            continue
        clusters[record["ud_pattern"]].append(record)

    rows = []
    for pattern, items in clusters.items():
        word_label_counts = Counter(item["word_label"] for item in items)
        word_sig_counts = Counter(item["word_signature"] for item in items)
        ling_sig_counts = Counter(item["lingua_projection_signature"] for item in items)
        ling_kind_counts = Counter(item["lingua_projection_kind"] for item in items)
        split_counts = Counter(item["split"] for item in items)
        if len(items) < args.min_count:
            continue
        if len(word_sig_counts) < args.min_word_signatures and len(ling_sig_counts) < args.min_lingua_signatures:
            continue
        score = math.log2(len(items) + 1) * (1 + entropy(word_sig_counts)) * (
            len(word_sig_counts) + 0.5 * len(ling_sig_counts)
        )
        rows.append(
            {
                "ud_pattern": pattern,
                "count": len(items),
                "divergence_score": score,
                "word_signature_entropy": entropy(word_sig_counts),
                "lingua_projection_entropy": entropy(ling_sig_counts),
                "distinct_word_labels": len(word_label_counts),
                "distinct_word_signatures": len(word_sig_counts),
                "distinct_lingua_projection_signatures": len(ling_sig_counts),
                "split_counts": compact_counter(split_counts),
                "word_label_counts": compact_counter(word_label_counts),
                "word_signature_counts": compact_counter(word_sig_counts),
                "lingua_projection_kind_counts": compact_counter(ling_kind_counts),
                "lingua_projection_signature_counts": compact_counter(ling_sig_counts),
            }
        )
    rows.sort(key=lambda row: (-row["divergence_score"], -row["count"], row["ud_pattern"]))
    return rows, clusters


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["ud_pattern"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, cluster_rows: list[dict[str, Any]], clusters: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> None:
    lines = [
        "# Gold word-lingua UD pattern divergence projected to Lingua",
        "",
        "Pattern discovery is done on gold word-lingua edges aligned to UD.",
        "Each example is then projected to the final gold Lingua graph.",
        "",
        f"- label_group: `{args.label_group}`",
        f"- min_count: `{args.min_count}`",
        "",
        "## Top Patterns",
        "",
        "| rank | count | word sigs | Lingua projections | UD pattern | word labels | Lingua projection kinds |",
        "|---:|---:|---:|---:|---|---|---|",
    ]
    for i, row in enumerate(cluster_rows[: args.top_clusters], start=1):
        lines.append(
            f"| {i} | {row['count']} | {row['distinct_word_signatures']} | "
            f"{row['distinct_lingua_projection_signatures']} | `{row['ud_pattern']}` | "
            f"{row['word_label_counts']} | {row['lingua_projection_kind_counts']} |"
        )

    lines.extend(["", "## Examples"])
    for i, row in enumerate(cluster_rows[: args.top_clusters], start=1):
        pattern = row["ud_pattern"]
        items = clusters[pattern]
        lines.extend(
            [
                "",
                f"### {i}. `{pattern}`",
                "",
                f"- count: {row['count']}",
                f"- word labels: {row['word_label_counts']}",
                f"- word signatures: {row['word_signature_counts']}",
                f"- Lingua projections: {row['lingua_projection_signature_counts']}",
            ]
        )
        sig_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            key = f"{item['word_signature']} || {item['lingua_projection_signature']}"
            sig_groups[key].append(item)
        for sig, sig_items in sorted(sig_groups.items(), key=lambda item: (-len(item[1]), item[0]))[: args.examples_per_signature]:
            lines.append(f"- `{sig}` ({len(sig_items)})")
            for item in sig_items[: args.examples_per_signature_item]:
                lines.append(f"  - {item['split']}:{item['line']} `{item['sentence']}`")
                lines.append(
                    f"    - word-lingua: {item['source_text']}/{item['source_type']} "
                    f"--{item['word_label']}--> {item['target_text']}/{item['target_type']}"
                )
                lines.append(f"    - UD: `{item['ud_lex_pattern']}`")
                lines.append(
                    f"    - Lingua: {item['lingua_projection_kind']}"
                    f"({item['lingua_projection_relation']}) "
                    f"{item['lingua_source_text']}/{item['lingua_source_type']} "
                    f"=> {item['lingua_target_text']}/{item['lingua_target_type']}"
                )
                if item["lingua_source_desc_text"] != item["lingua_source_text"] or item["lingua_target_desc_text"] != item["lingua_target_text"]:
                    lines.append(
                        f"    - Lingua desc: `{item['lingua_source_desc_text']}` => `{item['lingua_target_desc_text']}`"
                    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = edge_records(args)
    record_fields = [
        "split",
        "line",
        "idx",
        "edge_index",
        "word_label",
        "word_label_group",
        "word_signature",
        "ud_pattern",
        "ud_lex_pattern",
        "source_text",
        "target_text",
        "source_type",
        "target_type",
        "source_tokens",
        "target_tokens",
        "lingua_projection_kind",
        "lingua_projection_relation",
        "lingua_projection_signature",
        "lingua_source_index",
        "lingua_target_index",
        "lingua_source_type",
        "lingua_target_type",
        "lingua_source_text",
        "lingua_target_text",
        "lingua_source_desc_text",
        "lingua_target_desc_text",
        "sentence",
    ]
    detail_path = args.output_dir / "word_lingua_ud_edge_records.tsv"
    write_tsv(detail_path, records, record_fields)
    cluster_rows, clusters = cluster_records(records, args)
    cluster_path = args.output_dir / "word_lingua_ud_divergent_patterns.tsv"
    report_path = args.output_dir / "word_lingua_ud_to_lingua_projection.md"
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
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_base", type=Path, default=Path("reports/v13_overwrite_latest_prediction_dump"))
    parser.add_argument("--ud_base", type=Path, default=Path("reports/v13_overwrite_latest_ud_stanza"))
    parser.add_argument("--output_dir", type=Path, default=Path("reports/v13_overwrite_latest_word_lingua_ud_to_lingua_projection"))
    parser.add_argument("--label_group", choices=["all", "modification", "predicate_args", "punctuation", "expression_list", "discourse", "other"], default="all")
    parser.add_argument("--min_count", type=int, default=8)
    parser.add_argument("--min_word_signatures", type=int, default=2)
    parser.add_argument("--min_lingua_signatures", type=int, default=2)
    parser.add_argument("--top_clusters", type=int, default=50)
    parser.add_argument("--examples_per_signature", type=int, default=8)
    parser.add_argument("--examples_per_signature_item", type=int, default=2)
    parser.add_argument("--skip_unmapped", action="store_true")
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
