#!/usr/bin/env python3
"""Audit annotation inconsistency around Lingua modification scope/layer."""

from __future__ import annotations

import argparse
import copy
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

from lingua.learn.wordgraph.modeler.gp2word import lingua2wordlingua
from lingua.structure.gpgraph import GPGraph, GPGPhraseNode
from lingua.utils.metric import get_node_info


SPLITS = ("train", "dev", "test")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def capish(text: str) -> bool:
    value = text.strip()
    return bool(value) and bool(re.search(r"[A-Za-z]", value)) and (value[0].isupper() or value.isupper())


def single_word_idx(node: Any) -> int | None:
    if not isinstance(node, GPGPhraseNode):
        return None
    try:
        words = list(node.words(with_aux=False))
        if len(words) == 1:
            return int(words[0])
    except Exception:
        pass
    spans = getattr(node, "spans", None)
    if spans:
        span = spans[0]
        return int(span[0] if isinstance(span, (list, tuple)) else span)
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_word_edges(data_dir: Path):
    for split in SPLITS:
        for idx, item in enumerate(read_jsonl(data_dir / f"{split}.jsonl")):
            lingua_graph = GPGraph.parse(item["lingua"])
            try:
                word_graph, _, _ = lingua2wordlingua(copy.deepcopy(lingua_graph))
            except Exception as exc:
                yield {
                    "split": split,
                    "line": idx + 1,
                    "sentence": item["sentence"],
                    "conversion_error": str(exc),
                }
                continue
            for parent, edge, child in word_graph.edges():
                parent_info = get_node_info(parent, word_graph, with_type=True, with_spans=True)
                child_info = get_node_info(child, word_graph, with_type=True, with_spans=True)
                parent_idx = single_word_idx(parent)
                child_idx = single_word_idx(child)
                yield {
                    "split": split,
                    "line": idx + 1,
                    "sentence": item["sentence"],
                    "label": str(edge.label),
                    "parent_text": parent_info["text"],
                    "child_text": child_info["text"],
                    "parent_type": parent_info.get("type"),
                    "child_type": child_info.get("type"),
                    "parent_idx": parent_idx,
                    "child_idx": child_idx,
                    "adjacent": (
                        parent_idx is not None
                        and child_idx is not None
                        and abs(parent_idx - child_idx) == 1
                    ),
                }


def group_for_edge(row: dict[str, Any]) -> str | None:
    if not row.get("adjacent"):
        return None
    parent_type = row.get("parent_type")
    child_type = row.get("child_type")
    parent_text = row.get("parent_text", "")
    child_text = row.get("child_text", "")
    if parent_type == "ModificationalConstant" and child_type == "NominalConstant":
        return "adjacent_modconst_nominal"
    if parent_type == "NominalConstant" and child_type == "ModificationalConstant":
        return "adjacent_nominal_modconst"
    if parent_type == "NominalConstant" and child_type == "NominalConstant":
        if capish(parent_text) and capish(child_text):
            return "adjacent_cap_nominal"
        return "adjacent_noncap_nominal"
    return None


def pair_key(row: dict[str, Any]) -> tuple[str, str, str | None, str | None]:
    return (
        normalize_text(row["parent_text"]),
        normalize_text(row["child_text"]),
        row.get("parent_type"),
        row.get("child_type"),
    )


def compact_counter(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(limit)]


def first_examples(rows: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    examples = []
    for row in rows[:limit]:
        examples.append(
            {
                "split": row["split"],
                "line": row["line"],
                "sentence": row["sentence"],
                "edge": (
                    f"{row['parent_text']}[{row.get('parent_type')}] "
                    f"--{row['label']}--> {row['child_text']}[{row.get('child_type')}]"
                ),
            }
        )
    return examples


def is_token_fragment_pair(pair: tuple[str, str, str | None, str | None]) -> bool:
    left, right, _left_type, _right_type = pair
    if re.fullmatch(r"\d{2,4}", left) and right in {"s", "th", "st", "nd", "rd"}:
        return True
    if left in {"st", "mr", "mrs", "dr"} and right == ".":
        return True
    if re.fullmatch(r"[a-z]+", left) and re.fullmatch(r"[ivx]+", right):
        return True
    if left in {"stat", "nf", "il", "cd", "hla"} and re.fullmatch(r"[a-z0-9+-]+", right):
        return True
    return False


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(field, "")).replace("\t", " ") for field in fields) + "\n")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    reason_summary = json.loads(args.reason_summary.read_text(encoding="utf-8"))

    group_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_examples: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    pair_rows: dict[str, dict[tuple[str, str, str | None, str | None], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    open_paren_counts: Counter[str] = Counter()
    open_paren_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conversion_errors: list[dict[str, Any]] = []

    for row in iter_word_edges(data_dir):
        if row.get("conversion_error"):
            conversion_errors.append(row)
            continue
        group = group_for_edge(row)
        if group:
            group_label_counts[group][row["label"]] += 1
            pair_rows[group][pair_key(row)].append(row)
            if len(group_examples[group][row["label"]]) < 5:
                group_examples[group][row["label"]].append(row)
        if row["parent_text"].strip() in {"(", "[", "{"} or row["child_text"].strip() in {"(", "[", "{"}:
            open_paren_counts[row["label"]] += 1
            if len(open_paren_examples[row["label"]]) < 5:
                open_paren_examples[row["label"]].append(row)

    conflict_rows: list[dict[str, Any]] = []
    p0_simple_modifier_rows: list[dict[str, Any]] = []
    proper_name_review_rows: list[dict[str, Any]] = []
    token_fragment_rows: list[dict[str, Any]] = []

    for group, pairs in pair_rows.items():
        for pair, rows in pairs.items():
            label_counts = Counter(row["label"] for row in rows)
            if len(label_counts) <= 1:
                continue
            total = sum(label_counts.values())
            if total < 2:
                continue
            label_summary = ",".join(f"{label}:{count}" for label, count in label_counts.most_common())
            example_map = defaultdict(list)
            for row in rows:
                if len(example_map[row["label"]]) < 2:
                    example_map[row["label"]].append(f"{row['split']}:{row['line']} {row['sentence']}")
            conflict_record = {
                "group": group,
                "left": pair[0],
                "right": pair[1],
                "left_type": pair[2],
                "right_type": pair[3],
                "total": total,
                "labels": label_summary,
                "examples": json.dumps(example_map, ensure_ascii=False),
            }
            conflict_rows.append(conflict_record)

            if (
                group == "adjacent_modconst_nominal"
                and label_counts.get("as:modification", 0) >= 2
                and any(label in label_counts for label in ("@modification_of", "@expression", "@lower"))
            ):
                p0_simple_modifier_rows.append(conflict_record)
            if group == "adjacent_cap_nominal" and (
                "as:modification" in label_counts
                and ("@expression" in label_counts or "next_word" in label_counts)
            ):
                proper_name_review_rows.append(conflict_record)
            if is_token_fragment_pair(pair) and (
                "next_word" in label_counts
                and ("as:modification" in label_counts or "@expression" in label_counts)
            ):
                token_fragment_rows.append(conflict_record)

    conflict_rows.sort(key=lambda row: (-len(row["labels"].split(",")), -int(row["total"]), row["group"], row["left"], row["right"]))
    p0_simple_modifier_rows.sort(key=lambda row: (-int(row["total"]), row["left"], row["right"]))
    proper_name_review_rows.sort(key=lambda row: (-int(row["total"]), row["left"], row["right"]))
    token_fragment_rows.sort(key=lambda row: (-int(row["total"]), row["left"], row["right"]))

    paren_review_rows = [
        {
            "label": label,
            "count": count,
            "examples": json.dumps(first_examples(open_paren_examples[label], limit=4), ensure_ascii=False),
        }
        for label, count in open_paren_counts.most_common()
        if label
        in {
            "@attribute_of",
            "@modification_of",
            "variable",
            "as:modification",
            "body",
            "@appositive_of",
            "pred.arg.1",
            "pred.arg.2",
            "func.arg",
            "@list",
        }
    ]

    fields = ["group", "left", "right", "left_type", "right_type", "total", "labels", "examples"]
    write_tsv(report_dir / "surface_pair_conflicts.tsv", conflict_rows, fields)
    write_tsv(report_dir / "p0_simple_modifier_candidates.tsv", p0_simple_modifier_rows, fields)
    write_tsv(report_dir / "proper_name_expression_review.tsv", proper_name_review_rows, fields)
    write_tsv(report_dir / "token_fragment_review.tsv", token_fragment_rows, fields)
    write_tsv(report_dir / "paren_attribute_review.tsv", paren_review_rows, ["label", "count", "examples"])

    summary = {
        "data_dir": str(data_dir),
        "reason_summary": str(args.reason_summary),
        "conversion_error_count": len(conversion_errors),
        "v13_modification_nonexact_counts": [
            row
            for row in reason_summary.get("reason_families", [])
            if row.get("category") == "modification"
        ],
        "v13_top_label_confusions": reason_summary.get("case_level_label_confusions", [])[:30],
        "v13_top_scope_labels": reason_summary.get("case_level_scope_labels", [])[:30],
        "group_label_counts": {
            group: compact_counter(counter, 20) for group, counter in group_label_counts.items()
        },
        "open_paren_counts": compact_counter(open_paren_counts, 30),
        "surface_pair_conflict_count": len(conflict_rows),
        "p0_simple_modifier_candidate_count": len(p0_simple_modifier_rows),
        "proper_name_review_count": len(proper_name_review_rows),
        "token_fragment_review_count": len(token_fragment_rows),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report_dir / "summary.md", summary, group_examples)
    return summary


def counts_table(counter_rows: list[dict[str, Any]], limit: int = 10) -> str:
    lines = ["| label | count |", "|---|---:|"]
    for row in counter_rows[:limit]:
        lines.append(f"| `{row['key']}` | {row['count']} |")
    return "\n".join(lines)


def write_markdown(path: Path, summary: dict[str, Any], group_examples: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    mod_counts = summary["v13_modification_nonexact_counts"][0]["counts"]
    lines = [
        "# Modification Scope/Layer Annotation Inconsistency Audit",
        "",
        f"Data: `{summary['data_dir']}`",
        f"V13 reason summary: `{summary['reason_summary']}`",
        "",
        "This report treats the V13 modification non-exact cases as a priority signal, then checks the gold annotation data for same/near-same structures that receive different labels.",
        "",
        "## V13 priority signal",
        "",
        f"`Modification scope/layer` appears in train/dev/test non-exact cases as: train `{mod_counts.get('train', 0)}`, dev `{mod_counts.get('dev', 0)}`, test `{mod_counts.get('test', 0)}`.",
        "",
        "Top same-endpoint label confusions involving modification-like structures:",
        "",
        counts_table(summary["v13_top_label_confusions"], 12),
        "",
        "Top same-label wrong attachment/scope labels:",
        "",
        counts_table(summary["v13_top_scope_labels"], 12),
        "",
        "## Gold annotation inconsistency signals",
        "",
        "### 1. Simple adjacent modifiers are mostly `as:modification`, but minority labels encode the same surface relation differently",
        "",
        "Adjacent `ModificationalConstant -> NominalConstant` label distribution:",
        "",
        counts_table(summary["group_label_counts"].get("adjacent_modconst_nominal", []), 8),
        "",
        f"Exact surface-pair conflicts in this group: `{summary['p0_simple_modifier_candidate_count']}` candidate pairs written to `p0_simple_modifier_candidates.tsv`.",
        "",
        "Examples include `last week`, `prime minister`, `nuclear factor`, `regulatory region`, `good quality`, and `genomic DNA`, where one instance is `@modification_of` but most comparable instances are `as:modification`.",
        "",
        "Repair suggestion: for adjacent lexical/pre-nominal modifiers with no independent modifier argument structure, normalize to a direct `as:modification` chain. Reserve `Modification(...)` / `@modification_of` for phrase-level modifiers that truly have `variable` and `body` scope.",
        "",
        "### 2. Proper names and named entities mix `as:modification`, `@expression`, and `next_word`",
        "",
        "Adjacent capitalized `NominalConstant -> NominalConstant` label distribution:",
        "",
        counts_table(summary["group_label_counts"].get("adjacent_cap_nominal", []), 10),
        "",
        f"Exact surface-pair conflicts in this family requiring policy review: `{summary['proper_name_review_count']}` pairs written to `proper_name_expression_review.tsv`.",
        "",
        "High-signal conflicts include `United States`, `New York`, `Los Angeles`, and scientific/title names. These are not all automatic fixes: some are true entity-internal expression edges, while others are semantic modifiers of a following head.",
        "",
        "Repair suggestion: define a clear policy: fixed proper-name/entity-internal tokens should use `@expression`; a name used as a semantic modifier of an external head should be represented as an expression internally, then attach the whole expression to the head as modification. Avoid mixing `as:modification` and `@expression` for the same entity-internal pair.",
        "",
        "### 3. Tokenization fragments and domain terms are inconsistently split between `next_word`, `@expression`, and `as:modification`",
        "",
        "Adjacent non-capital nominal pair distribution:",
        "",
        counts_table(summary["group_label_counts"].get("adjacent_noncap_nominal", []), 10),
        "",
        f"Likely token-fragment conflicts written to `token_fragment_review.tsv`: `{summary['token_fragment_review_count']}`.",
        "",
        "Examples include decade suffixes (`1970 s` vs `1970s`), ordinal suffixes (`20 th`), abbreviations (`St .`), and biomedical fragments (`STAT 1`, `class II`, `kappa B`).",
        "",
        "Repair suggestion: use `next_word` only for tokenization fragments that form a single orthographic token or symbol; use `@expression` for fixed multi-token names/terms; use `as:modification` for genuinely compositional noun modification.",
        "",
        "### 4. Parenthetical attributes/appositions are mixed with modification-layer edges",
        "",
        "Open-parenthesis-related edge labels:",
        "",
        counts_table(summary["open_paren_counts"], 15),
        "",
        "`paren_attribute_review.tsv` lists examples for `@attribute_of`, `@modification_of`, `variable`, `as:modification`, `body`, `@appositive_of`, and argument edges around parentheticals.",
        "",
        "Repair suggestion: parenthetical nationality, dates, roles, and short descriptors should be annotated consistently as attribute/apposition/list structures according to their semantics. Do not use `variable/body` or direct `as:modification` for parentheticals unless the parenthetical itself is a real modifier functor with an explicit body.",
        "",
        "### 5. `variable/body` scope and compound `@modification_of+pred.arg.N` need a narrower standard",
        "",
        "V13 scope errors concentrate on `variable`, `body`, `pred.arg.1`, `pred.arg.2`, `as:punctuation`, and `func.arg`. These are often not label-vocabulary problems; the label is legal but attached to a different layer.",
        "",
        "Repair suggestion: state the invariant explicitly: `variable` is the semantic host being modified, `body` is the modifier content. Punctuation should attach to the complete modification wrapper when it scopes over the wrapper, not arbitrarily to the body or variable child. Use compound labels like `@modification_of+pred.arg.1` only when the predicate is both a modifier and has a recovered predicate argument; otherwise use the plain predicate role plus a separate modification relation.",
        "",
        "## Repair plan",
        "",
        "1. P0 automatic candidates: review/apply `p0_simple_modifier_candidates.tsv`; most minority `@modification_of` cases among simple adjacent modifiers should collapse to `as:modification`.",
        "2. P1 policy review: resolve `proper_name_expression_review.tsv` and `token_fragment_review.tsv`; these require a written distinction among entity expression, semantic modification, and tokenization fragments.",
        "3. P1 parenthetical review: audit `paren_attribute_review.tsv` and normalize parenthetical role/date/nationality structures away from modification wrappers where appropriate.",
        "4. P2 decoder/data cleanup: after gold repairs, rerun word-lingua conversion validation and retrain/evaluate; these changes directly target the top V13 modification error families.",
        "",
        "## Generated files",
        "",
        "- `summary.json`: machine-readable counters",
        "- `surface_pair_conflicts.tsv`: all exact surface-pair label conflicts",
        "- `p0_simple_modifier_candidates.tsv`: simple modifier candidates for likely normalization",
        "- `proper_name_expression_review.tsv`: proper-name/entity-internal policy review queue",
        "- `token_fragment_review.tsv`: tokenization/domain-term review queue",
        "- `paren_attribute_review.tsv`: parenthetical attachment review queue",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--reason_summary", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    summary = analyze(parse_args())
    print(
        json.dumps(
            {
                "surface_pair_conflict_count": summary["surface_pair_conflict_count"],
                "p0_simple_modifier_candidate_count": summary["p0_simple_modifier_candidate_count"],
                "proper_name_review_count": summary["proper_name_review_count"],
                "token_fragment_review_count": summary["token_fragment_review_count"],
                "conversion_error_count": summary["conversion_error_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
