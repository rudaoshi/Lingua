#!/usr/bin/env python3
"""Audit Lingua-level modification scope/layer annotation inconsistency."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_DIR = REPO_ROOT / "src"
for path in (str(SCRIPT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from lingua.structure.gpgraph import GPGraph, GPGPhraseNode  # noqa: E402
from lingua.utils.metric import get_node_info  # noqa: E402


SPLITS = ("train", "dev", "test")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def capish(text: str) -> bool:
    value = text.strip()
    return bool(value) and bool(re.search(r"[A-Za-z]", value)) and (value[0].isupper() or value.isupper())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def node_info(node: Any, graph: GPGraph) -> dict[str, Any]:
    info = get_node_info(node, graph, with_type=True, with_spans=True)
    info["id"] = getattr(node, "ID", None)
    info["text"] = clean_text(info.get("text", ""))
    return info


def token_indexes(node: Any) -> list[int]:
    if not isinstance(node, GPGPhraseNode):
        return []
    indexes: list[int] = []
    for span in getattr(node, "spans", ()) or ():
        if isinstance(span, str):
            continue
        start, end = span
        indexes.extend(range(int(start), int(end) + 1))
    return sorted(set(indexes))


def first_token_pair(node: Any, graph: GPGraph) -> tuple[str, str] | None:
    indexes = token_indexes(node)
    if len(indexes) < 2:
        return None
    words = [graph.words[index].strip() for index in indexes]
    if len(words) < 2:
        return None
    return normalize_text(words[0]), normalize_text(words[1])


def all_adjacent_token_pairs(node: Any, graph: GPGraph) -> list[tuple[str, str]]:
    indexes = token_indexes(node)
    pairs: list[tuple[str, str]] = []
    for left, right in zip(indexes, indexes[1:]):
        if right != left + 1:
            continue
        pairs.append((normalize_text(graph.words[left]), normalize_text(graph.words[right])))
    return pairs


def min_token(node: Any) -> int:
    indexes = token_indexes(node)
    return indexes[0] if indexes else 10**9


def edge_children_by_label(graph: GPGraph, parent: Any) -> dict[str, list[Any]]:
    children: dict[str, list[Any]] = defaultdict(list)
    for source, edge, target in graph.edges():
        if source == parent:
            children[str(edge.label)].append(target)
    return children


def is_abstract_modification(node: Any, graph: GPGraph) -> bool:
    info = node_info(node, graph)
    return info.get("type") == "ModificationalFunctor" and info.get("spans") is None


def is_expression_functor(node: Any, graph: GPGraph) -> bool:
    return node_info(node, graph).get("type") == "ExpressionFunctor"


def pair_key(left: str, right: str) -> tuple[str, str]:
    return normalize_text(left), normalize_text(right)


def example_record(
    split: str,
    line: int,
    sentence: str,
    representation: str,
    detail: str,
    properish: bool = False,
    left_type: str | None = None,
    right_type: str | None = None,
) -> dict[str, Any]:
    return {
        "split": split,
        "line": line,
        "sentence": sentence,
        "representation": representation,
        "detail": detail,
        "properish": properish,
        "left_type": left_type,
        "right_type": right_type,
    }


def is_token_fragment_pair(pair: tuple[str, str]) -> bool:
    left, right = pair
    if re.fullmatch(r"\d{2,4}", left) and right in {"s", "th", "st", "nd", "rd"}:
        return True
    if left.lower() in {"st", "mr", "mrs", "dr", "prof"} and right == ".":
        return True
    if re.fullmatch(r"[a-z]+", left) and re.fullmatch(r"[ivx]+", right):
        return True
    if left.lower() in {"stat", "nf", "il", "hla", "cd"} and re.fullmatch(r"[a-z0-9+-]+", right.lower()):
        return True
    return False


def is_noise_pair(pair: tuple[str, str]) -> bool:
    blocked = {
        "modification",
        "expression",
        "list",
        "ref",
        "attribute",
        "apposition",
        "discourse",
        "vocative",
    }
    if pair[0] in blocked or pair[1] in blocked:
        return True
    return any(re.fullmatch(r"\W+", part or "") for part in pair)


def has_lexical_wrapper_conflict(records: list[dict[str, Any]]) -> bool:
    lexical_types = {"ModificationalConstant", "NominalConstant"}
    has_wrapper = any(
        record["representation"] == "abstract_modification_wrapper"
        and record.get("left_type") in lexical_types
        and record.get("right_type") in lexical_types
        for record in records
    )
    has_direct = any(
        record["representation"] == "direct_as_modification"
        and record.get("left_type") in lexical_types
        and record.get("right_type") in lexical_types
        for record in records
    )
    has_properish = any(record.get("properish") for record in records)
    return has_wrapper and has_direct and not has_properish


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(field, "")).replace("\t", " ") for field in fields) + "\n")


def compact(counter: Counter[Any], limit: int = 30) -> list[dict[str, Any]]:
    return [{"key": str(key), "count": count} for key, count in counter.most_common(limit)]


def collect(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    reason_summary = json.loads(args.reason_summary.read_text(encoding="utf-8")) if args.reason_summary else {}

    pair_records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    node_surface_types: dict[str, Counter[str]] = defaultdict(Counter)
    node_surface_examples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    edge_label_counts: Counter[str] = Counter()
    mod_wrapper_counts: Counter[str] = Counter()
    mod_wrapper_type_counts: Counter[tuple[str, str]] = Counter()
    paren_counts: Counter[str] = Counter()
    paren_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conversion_errors: list[dict[str, Any]] = []

    for split in SPLITS:
        for idx, item in enumerate(read_jsonl(args.data_dir / f"{split}.jsonl")):
            line = idx + 1
            sentence = item["sentence"]
            try:
                graph = GPGraph.parse(item["lingua"])
            except Exception as exc:
                conversion_errors.append({"split": split, "line": line, "sentence": sentence, "error": str(exc)})
                continue

            for node in graph.nodes():
                info = node_info(node, graph)
                text = normalize_text(info["text"])
                node_type = str(info.get("type"))
                if text:
                    node_surface_types[text][node_type] += 1
                    key = (text, node_type)
                if len(node_surface_examples[key]) < 3:
                    node_surface_examples[key].append(
                            example_record(
                                split,
                                line,
                                sentence,
                                f"node:{node_type}",
                                f"{info['text']} {info.get('spans')}",
                                properish=capish(info["text"]),
                                left_type=node_type,
                                right_type=node_type,
                            )
                    )

                indexes = token_indexes(node)
                if len(indexes) >= 2:
                    for pair in all_adjacent_token_pairs(node, graph):
                        representation = f"merged_node:{node_type}"
                        pair_records[pair].append(
                            example_record(
                                split,
                                line,
                                sentence,
                                representation,
                                f"{info['text']}[{node_type}] spans={info.get('spans')}",
                                properish=any(capish(graph.words[index]) for index in indexes) or capish(info["text"]),
                                left_type=node_type,
                                right_type=node_type,
                            )
                        )

            for parent, edge, child in graph.edges():
                label = str(edge.label)
                edge_label_counts[label] += 1
                p_info = node_info(parent, graph)
                c_info = node_info(child, graph)

                if label == "as:modification":
                    pair = pair_key(p_info["text"], c_info["text"])
                    pair_records[pair].append(
                        example_record(
                            split,
                            line,
                            sentence,
                            "direct_as_modification",
                            f"{p_info['text']}[{p_info.get('type')}] --as:modification--> {c_info['text']}[{c_info.get('type')}]",
                            properish=capish(p_info["text"]) or capish(c_info["text"]),
                            left_type=str(p_info.get("type")),
                            right_type=str(c_info.get("type")),
                        )
                    )

                if (
                    p_info.get("type") == "PunctuationalConstant"
                    and p_info["text"] in {"(", "[", "{"}
                ) or (
                    c_info.get("type") == "PunctuationalConstant"
                    and c_info["text"] in {"(", "[", "{"}
                ):
                    paren_counts[label] += 1
                    if len(paren_examples[label]) < 5:
                        paren_examples[label].append(
                            example_record(
                                split,
                                line,
                                sentence,
                                f"paren:{label}",
                                f"{p_info['text']}[{p_info.get('type')}] --{label}--> {c_info['text']}[{c_info.get('type')}]",
                                properish=capish(p_info["text"]) or capish(c_info["text"]),
                                left_type=str(p_info.get("type")),
                                right_type=str(c_info.get("type")),
                            )
                        )

            for node in graph.nodes():
                if is_abstract_modification(node, graph):
                    children = edge_children_by_label(graph, node)
                    variables = children.get("variable", [])
                    bodies = children.get("body", [])
                    mod_wrapper_counts["abstract_modification_nodes"] += 1
                    if not variables:
                        mod_wrapper_counts["missing_variable"] += 1
                    if not bodies:
                        mod_wrapper_counts["missing_body"] += 1
                    if len(variables) > 1:
                        mod_wrapper_counts["multi_variable"] += 1
                    if len(bodies) > 1:
                        mod_wrapper_counts["multi_body"] += 1
                    for variable in variables[:1]:
                        for body in bodies[:1]:
                            v_info = node_info(variable, graph)
                            b_info = node_info(body, graph)
                            mod_wrapper_type_counts[(str(b_info.get("type")), str(v_info.get("type")))] += 1
                            pair = pair_key(b_info["text"], v_info["text"])
                            pair_records[pair].append(
                                example_record(
                                    split,
                                    line,
                                    sentence,
                                    "abstract_modification_wrapper",
                                    f"Modification --body--> {b_info['text']}[{b_info.get('type')}]; --variable--> {v_info['text']}[{v_info.get('type')}]",
                                    properish=capish(b_info["text"]) or capish(v_info["text"]),
                                    left_type=str(b_info.get("type")),
                                    right_type=str(v_info.get("type")),
                                )
                            )

                if is_expression_functor(node, graph):
                    children = edge_children_by_label(graph, node).get("func.arg", [])
                    ordered = sorted(children, key=min_token)
                    for left, right in zip(ordered, ordered[1:]):
                        left_info = node_info(left, graph)
                        right_info = node_info(right, graph)
                        pair = pair_key(left_info["text"], right_info["text"])
                        pair_records[pair].append(
                            example_record(
                                split,
                                line,
                                sentence,
                                "expression_functor",
                                f"Expression --func.arg--> {left_info['text']}[{left_info.get('type')}], {right_info['text']}[{right_info.get('type')}]",
                                properish=capish(left_info["text"]) or capish(right_info["text"]),
                                left_type=str(left_info.get("type")),
                                right_type=str(right_info.get("type")),
                            )
                        )

    conflict_rows: list[dict[str, Any]] = []
    simple_wrapper_conflicts: list[dict[str, Any]] = []
    lexical_simple_wrapper_conflicts: list[dict[str, Any]] = []
    proper_name_conflicts: list[dict[str, Any]] = []
    token_fragment_conflicts: list[dict[str, Any]] = []

    for pair, records in pair_records.items():
        rep_counts = Counter(record["representation"] for record in records)
        family_counts = Counter(record["representation"].split(":", 1)[0] for record in records)
        if len(family_counts) <= 1:
            continue
        labels = ",".join(f"{key}:{value}" for key, value in rep_counts.most_common())
        examples_by_rep: dict[str, list[str]] = defaultdict(list)
        for record in records:
            if len(examples_by_rep[record["representation"]]) < 2:
                examples_by_rep[record["representation"]].append(
                    f"{record['split']}:{record['line']} {record['detail']} || {record['sentence']}"
                )
        row = {
            "left": pair[0],
            "right": pair[1],
            "total": len(records),
            "representations": labels,
            "families": ",".join(f"{key}:{value}" for key, value in family_counts.most_common()),
            "examples": json.dumps(examples_by_rep, ensure_ascii=False),
        }
        conflict_rows.append(row)
        if "direct_as_modification" in rep_counts and "abstract_modification_wrapper" in rep_counts:
            simple_wrapper_conflicts.append(row)
            if not is_noise_pair(pair) and has_lexical_wrapper_conflict(records):
                lexical_simple_wrapper_conflicts.append(row)
        if any(record.get("properish") for record in records) and any(
            rep.startswith(("merged_node", "expression_functor", "direct_as_modification", "abstract_modification_wrapper"))
            for rep in rep_counts
        ):
            proper_name_conflicts.append(row)
        if is_token_fragment_pair(pair):
            token_fragment_conflicts.append(row)

    conflict_rows.sort(key=lambda row: (-int(row["total"]), row["left"], row["right"]))
    simple_wrapper_conflicts.sort(key=lambda row: (-int(row["total"]), row["left"], row["right"]))
    lexical_simple_wrapper_conflicts.sort(key=lambda row: (-int(row["total"]), row["left"], row["right"]))
    proper_name_conflicts.sort(key=lambda row: (-int(row["total"]), row["left"], row["right"]))
    token_fragment_conflicts.sort(key=lambda row: (-int(row["total"]), row["left"], row["right"]))

    node_type_conflict_rows = []
    for text, counts in node_surface_types.items():
        if len(counts) <= 1:
            continue
        if sum(counts.values()) < 2:
            continue
        if not any("Modification" in node_type or "Expression" in node_type for node_type in counts):
            continue
        examples = {}
        for node_type in counts:
            examples[node_type] = [
                f"{record['split']}:{record['line']} {record['detail']} || {record['sentence']}"
                for record in node_surface_examples.get((text, node_type), [])[:2]
            ]
        node_type_conflict_rows.append(
            {
                "surface": text,
                "total": sum(counts.values()),
                "types": ",".join(f"{key}:{value}" for key, value in counts.most_common()),
                "examples": json.dumps(examples, ensure_ascii=False),
            }
        )
    node_type_conflict_rows.sort(key=lambda row: (-int(row["total"]), row["surface"]))

    paren_rows = [
        {
            "label": label,
            "count": count,
            "examples": json.dumps(paren_examples[label], ensure_ascii=False),
        }
        for label, count in paren_counts.most_common()
    ]

    fields = ["left", "right", "total", "representations", "families", "examples"]
    write_tsv(report_dir / "lingua_pair_representation_conflicts.tsv", conflict_rows, fields)
    write_tsv(report_dir / "lingua_simple_modifier_wrapper_conflicts.tsv", simple_wrapper_conflicts, fields)
    write_tsv(report_dir / "lingua_lexical_simple_modifier_wrapper_conflicts.tsv", lexical_simple_wrapper_conflicts, fields)
    write_tsv(report_dir / "lingua_proper_name_representation_conflicts.tsv", proper_name_conflicts, fields)
    write_tsv(report_dir / "lingua_token_fragment_representation_conflicts.tsv", token_fragment_conflicts, fields)
    write_tsv(report_dir / "lingua_node_surface_type_conflicts.tsv", node_type_conflict_rows, ["surface", "total", "types", "examples"])
    write_tsv(report_dir / "lingua_parenthetical_layer_review.tsv", paren_rows, ["label", "count", "examples"])

    v13_mod = [
        row
        for row in reason_summary.get("reason_families", [])
        if row.get("category") == "modification"
    ]

    summary = {
        "data_dir": str(args.data_dir),
        "reason_summary": str(args.reason_summary) if args.reason_summary else None,
        "conversion_error_count": len(conversion_errors),
        "v13_modification_nonexact_counts": v13_mod,
        "edge_label_counts": compact(edge_label_counts, 30),
        "abstract_modification_summary": dict(mod_wrapper_counts),
        "abstract_modification_body_variable_type_counts": compact(mod_wrapper_type_counts, 30),
        "pair_representation_conflict_count": len(conflict_rows),
        "simple_modifier_wrapper_conflict_count": len(simple_wrapper_conflicts),
        "lexical_simple_modifier_wrapper_conflict_count": len(lexical_simple_wrapper_conflicts),
        "proper_name_representation_conflict_count": len(proper_name_conflicts),
        "token_fragment_representation_conflict_count": len(token_fragment_conflicts),
        "node_surface_type_conflict_count": len(node_type_conflict_rows),
        "parenthetical_label_counts": compact(paren_counts, 30),
        "outputs": {
            "pair_representation_conflicts": str(report_dir / "lingua_pair_representation_conflicts.tsv"),
            "simple_modifier_wrapper_conflicts": str(report_dir / "lingua_simple_modifier_wrapper_conflicts.tsv"),
            "lexical_simple_modifier_wrapper_conflicts": str(report_dir / "lingua_lexical_simple_modifier_wrapper_conflicts.tsv"),
            "proper_name_representation_conflicts": str(report_dir / "lingua_proper_name_representation_conflicts.tsv"),
            "token_fragment_representation_conflicts": str(report_dir / "lingua_token_fragment_representation_conflicts.tsv"),
            "node_surface_type_conflicts": str(report_dir / "lingua_node_surface_type_conflicts.tsv"),
            "parenthetical_layer_review": str(report_dir / "lingua_parenthetical_layer_review.tsv"),
        },
    }
    (report_dir / "lingua_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report_dir / "lingua_findings_zh.md", summary)
    return summary


def table(rows: list[dict[str, Any]], limit: int = 12) -> str:
    lines = ["| item | count |", "|---|---:|"]
    for row in rows[:limit]:
        lines.append(f"| `{row['key']}` | {row['count']} |")
    return "\n".join(lines)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    mod_counts = {}
    if summary["v13_modification_nonexact_counts"]:
        mod_counts = summary["v13_modification_nonexact_counts"][0].get("counts", {})
    lines = [
        "# Lingua 层级 Modification Scope/Layer 标注不一致分析",
        "",
        "## 口径",
        "",
        "这版直接读取 gold `item[\"lingua\"]` 的 `GPGraph`，不经过 `lingua2wordlingua`。因此这里分析的是 Lingua 原图里的表示层级，而不是 word-lingua 转换后的 `@expression`、`@modification_of`、`next_word` label。",
        "",
        "在 Lingua 层级，类似问题主要体现为同一 surface pair 在这些 representation 间摇摆：",
        "",
        "- merged node：多个 token 被合成一个 `NominalConstant` / `ModificationalFunctor` / 其他节点。",
        "- direct `as:modification`：词汇 modifier 直接挂到 head。",
        "- abstract `Modification` wrapper：无 span 的 `ModificationalFunctor` 通过 `body` 和 `variable` 表示修饰。",
        "- `ExpressionFunctor`：通过 `func.arg` 表示实体/固定表达内部结构。",
        "",
        "## V13 路由信号",
        "",
        f"V13 `Modification scope/layer` non-exact: train `{mod_counts.get('train', 0)}`, dev `{mod_counts.get('dev', 0)}`, test `{mod_counts.get('test', 0)}`。",
        "",
        "## Lingua 原图统计",
        "",
        "Edge label 分布：",
        "",
        table(summary["edge_label_counts"], 12),
        "",
        f"抽象 `Modification` wrapper 数：`{summary['abstract_modification_summary'].get('abstract_modification_nodes', 0)}`。",
        "",
        "抽象 wrapper 的 body/variable 类型分布：",
        "",
        table(summary["abstract_modification_body_variable_type_counts"], 12),
        "",
        "## 发现的不一致",
        "",
        f"1. Pair representation conflict：`{summary['pair_representation_conflict_count']}` 组，同一 pair 在 merged node / direct `as:modification` / abstract wrapper / expression 间摇摆。见 `lingua_pair_representation_conflicts.tsv`。",
        f"2. Broad simple modifier wrapper conflict：`{summary['simple_modifier_wrapper_conflict_count']}` 组，同时出现 direct `as:modification` 和 abstract `Modification(body, variable)`。这个队列较宽，包含 determiner/wrapper 外层结构。见 `lingua_simple_modifier_wrapper_conflicts.tsv`。",
        f"3. Lexical simple modifier wrapper conflict：`{summary['lexical_simple_modifier_wrapper_conflict_count']}` 组，过滤掉 proper-name、punctuation、`Modification/Expression` 元节点后，保留真正词汇 modifier 的 P0 队列。见 `lingua_lexical_simple_modifier_wrapper_conflicts.tsv`。",
        f"4. Proper-name representation conflict：`{summary['proper_name_representation_conflict_count']}` 组，专名/实体内部在 merged node、expression、direct modification、wrapper 之间摇摆。见 `lingua_proper_name_representation_conflicts.tsv`。",
        f"5. Token-fragment/domain-term conflict：`{summary['token_fragment_representation_conflict_count']}` 组。见 `lingua_token_fragment_representation_conflicts.tsv`。",
        f"6. 同 surface node type conflict：`{summary['node_surface_type_conflict_count']}` 组。见 `lingua_node_surface_type_conflicts.tsv`。",
        "",
        "Parenthetical 相关边分布：",
        "",
        table(summary["parenthetical_label_counts"], 12),
        "",
        "## Lingua 层级修复建议",
        "",
        "1. 普通词汇 modifier：如果只是 `last week`、`good quality`、`few days` 这种内部词汇组合，优先用 direct `as:modification`，不要创建抽象 `Modification` wrapper。",
        "2. 抽象 `Modification` wrapper：只用于完整短语修饰外部 variable 的情况；`body` 是完整修饰内容，`variable` 是被修饰对象。不要把 wrapper 用来表达 entity 内部 token 组合。",
        "3. 专名/实体：优先统一成 merged node 或 `ExpressionFunctor`，不要在同一实体内部混用 direct `as:modification` 和 abstract wrapper。若整个实体修饰外部 head，应先构造实体内部结构，再让整体修饰外部 head。",
        "4. token fragment：`1970s`、`20th`、`St.` 这类由 tokenizer 拆开的 orthographic token，应在 Lingua 层级统一为 merged node；不要和普通 `as:modification` 混用。",
        "5. parenthetical：括号补充语应稳定归入 attribute/apposition/list 或数量补充结构；括号标点不应直接成为 `variable/body/pred.arg.N` 的语义主体。",
        "",
        "## 输出文件",
        "",
    ]
    for name, output in summary["outputs"].items():
        lines.append(f"- `{name}`: `{output}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--reason_summary", type=Path, default=None)
    parser.add_argument("--report_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    summary = collect(parse_args())
    print(
        json.dumps(
            {
                "pair_representation_conflict_count": summary["pair_representation_conflict_count"],
                "simple_modifier_wrapper_conflict_count": summary["simple_modifier_wrapper_conflict_count"],
                "lexical_simple_modifier_wrapper_conflict_count": summary["lexical_simple_modifier_wrapper_conflict_count"],
                "proper_name_representation_conflict_count": summary["proper_name_representation_conflict_count"],
                "token_fragment_representation_conflict_count": summary["token_fragment_representation_conflict_count"],
                "node_surface_type_conflict_count": summary["node_surface_type_conflict_count"],
                "conversion_error_count": summary["conversion_error_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
