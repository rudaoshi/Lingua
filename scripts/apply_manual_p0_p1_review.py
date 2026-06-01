#!/usr/bin/env python3
"""Apply explicit manual fixes from the P0/P1 review queue.

This script is intentionally sample-specific.  The P0/P1 queues are model-based
review priorities, not repair rules, so each change below corresponds to an
item that was manually inspected against the current Lingua standard.
"""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "manual_p0_p1_review"
REPAIR_SCRIPT = ROOT / "scripts" / "repair_be_copula_rules.py"

spec = importlib.util.spec_from_file_location("repair_be_copula_rules", REPAIR_SCRIPT)
repair = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(repair)


Fix = Callable[[dict[str, Any]], str]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def set_graph(sample: dict[str, Any], nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    sample["lingua"]["graph"] = {"nodes": nodes, "edges": edges}


def edge(start: int, end: int, label: str) -> dict[str, Any]:
    return {"start": start, "end": end, "label": label}


def node(spans: Any, node_type: str) -> dict[str, Any]:
    return {"spans": spans, "type": node_type}


def compact(sample: dict[str, Any]) -> str:
    return repair.compact_graph(sample)


def fix_dev_236(sample: dict[str, Any]) -> str:
    """`1 .` is a list marker, not the subject of a fabricated copula."""
    set_graph(
        sample,
        [
            node([[0, 1]], "NominalConstant"),
            node([[4, 4]], "PunctuationalConstant"),
            node([[2, 3]], "FactualPredicator"),
        ],
        [
            edge(0, 2, "as:discourse"),
            edge(1, 0, "as:punctuation"),
        ],
    )
    return "treat list marker as discourse metadata and leave the omitted imperative subject absent"


def fix_dev_751(sample: dict[str, Any]) -> str:
    """`so excited` is the selected/result state of `got`, not its object."""
    for item in graph(sample)["edges"]:
        if item == {"start": 2, "end": 3, "label": "pred.arg.2"}:
            item["label"] = "pred.arg.4"
            return "change got -> so excited from pred.arg.2 to pred.arg.4"
    raise ValueError("expected got -> so pred.arg.2 edge not found")


def fix_dev_684(sample: dict[str, Any]) -> str:
    """Split over-merged `mostly deals` and make `mostly` scope over the PP event."""
    set_graph(
        sample,
        [
            node([[3, 3]], "ModificationalFunctor"),
            node([[0, 0]], "NominalConstant"),
            node([[5, 5]], "NominalConstant"),
            node([[4, 4]], "ModificationalConstant"),
            node([[2, 2]], "FactualPredicator"),
            node([[1, 1]], "ModificationalConstant"),
        ],
        [
            edge(4, 1, "pred.arg.1"),
            edge(3, 2, "as:modification"),
            edge(0, 4, "variable"),
            edge(0, 3, "body"),
            edge(5, 0, "as:modification"),
        ],
    )
    return "split `mostly` from `deals`; attach it outside `deals with dating advice`"


def fix_train_8106(sample: dict[str, Any]) -> str:
    """`No` determines `need`; the infinitive modifies the no-need phrase."""
    graph(sample)["edges"] = [
        edge(1, 0, "as:modification"),
        edge(3, 1, "as:modification"),
        edge(3, 4, "pred.arg.1"),
        edge(2, 3, "as:punctuation"),
    ]
    return "make `No` modify `need`, with `to worry` closing over the no-need phrase"


def fix_test_765(sample: dict[str, Any]) -> str:
    """`until 1997` modifies the playing-with-Jura event, not the team name."""
    graph(sample)["edges"] = [
        edge(1, 8, "variable"),
        edge(1, 3, "body"),
        edge(3, 4, "as:modification"),
        edge(4, 2, "as:modification"),
        edge(5, 1, "variable"),
        edge(5, 6, "body"),
        edge(7, 5, "as:punctuation"),
        edge(8, 0, "pred.arg.1"),
    ]
    return "reattach `until 1997` to the playing-with event instead of `Jura Sud Foot`"


def fix_train_5339(sample: dict[str, Any]) -> str:
    """Pure key-value field: colon is punctuation, not a factual predicate."""
    set_graph(
        sample,
        [
            node([[0, 0]], "NominalConstant"),
            node([[1, 1]], "PunctuationalConstant"),
            node([[2, 2]], "NominalConstant"),
            node("Attribute", "AttributePredicator"),
        ],
        [
            edge(3, 0, "pred.arg.1"),
            edge(3, 2, "pred.arg.2"),
            edge(3, 1, "punctuation"),
        ],
    )
    return "convert passcode key-value record to Attribute"


def fix_test_106(sample: dict[str, Any]) -> str:
    """Pure key-value field: `Job Title` is the key and `Attorney` the value."""
    set_graph(
        sample,
        [
            node([[0, 0]], "NominalConstant"),
            node([[1, 1]], "NominalConstant"),
            node("Expression", "ExpressionFunctor"),
            node([[2, 2]], "PunctuationalConstant"),
            node([[3, 3]], "NominalConstant"),
            node("Attribute", "AttributePredicator"),
        ],
        [
            edge(2, 0, "func.arg"),
            edge(2, 1, "func.arg"),
            edge(5, 2, "pred.arg.1"),
            edge(5, 4, "pred.arg.2"),
            edge(5, 3, "punctuation"),
        ],
    )
    return "convert job-title key-value record to Attribute"


def fix_train_8145(sample: dict[str, Any]) -> str:
    """Pure key-value field: colon is punctuation and the room code is a value."""
    set_graph(
        sample,
        [
            node([[0, 0]], "DeterminerConstant"),
            node([[1, 1]], "NominalConstant"),
            node("Expression", "ExpressionFunctor"),
            node([[3, 3]], "NominalConstant"),
            node([[4, 5]], "NominalConstant"),
            node("Expression", "ExpressionFunctor"),
            node([[2, 2]], "PunctuationalConstant"),
            node("Attribute", "AttributePredicator"),
        ],
        [
            edge(2, 0, "func.arg"),
            edge(2, 1, "func.arg"),
            edge(5, 3, "func.arg"),
            edge(5, 4, "func.arg"),
            edge(7, 2, "pred.arg.1"),
            edge(7, 5, "pred.arg.2"),
            edge(7, 6, "punctuation"),
        ],
    )
    return "convert place key-value record to Attribute"


def fix_train_3720(sample: dict[str, Any]) -> str:
    """`there` locates the seeing event, not the object `anyone`."""
    graph(sample)["edges"] = [
        edge(4, 2, "pred.arg.1"),
        edge(4, 0, "pred.arg.2"),
        edge(1, 4, "as:modification"),
        edge(5, 1, "as:modification"),
        edge(3, 5, "as:punctuation"),
    ]
    return "reattach locative `there` to the seeing event; keep `never` outer"


def fix_train_10311(sample: dict[str, Any]) -> str:
    """Decorative stars close the update header; exclamation closes the whole parataxis."""
    graph(sample)["edges"] = [
        edge(1, 0, "as:punctuation"),
        edge(2, 1, "as:punctuation"),
        edge(5, 2, "func.arg"),
        edge(5, 4, "func.arg"),
        edge(4, 6, "pred.arg.1"),
        edge(3, 5, "as:punctuation"),
    ]
    return "rewire update header punctuation and parataxis arguments"


def fix_train_5824(sample: dict[str, Any]) -> str:
    """All profession nouns are complements of `is`, not a separate outer fact."""
    graph(sample)["edges"] = [
        edge(0, 2, "pred.arg.1"),
        edge(0, 9, "pred.arg.2"),
        edge(2, 1, "as:modification"),
        edge(4, 3, "as:modification"),
        edge(8, 0, "as:punctuation"),
        edge(9, 10, "func.arg"),
        edge(9, 7, "func.arg"),
        edge(10, 5, "func.arg"),
        edge(10, 6, "func.arg"),
        edge(10, 4, "func.arg"),
    ]
    return "make the outer coordination the copular complement list"


def fix_train_5597(sample: dict[str, Any]) -> str:
    """Sentence-initial `But` has an omitted left discourse argument."""
    return "leave sentence-initial `But` without an invented external pred.arg.1"


def fix_train_4304(sample: dict[str, Any]) -> str:
    """Initial `2009` is a dateline/header fragment, not a semantic modifier."""
    for item in graph(sample)["edges"]:
        if item == {"start": 1, "end": 6, "label": "as:modification"}:
            item["label"] = "as:discourse"
            return "change leading dateline year from modifier to discourse metadata"
    raise ValueError("expected leading 2009 modifier edge not found")


def fix_train_10869(sample: dict[str, Any]) -> str:
    """Zero relative over `story I try to tell...` should not introduce Ref."""
    nodes = graph(sample)["nodes"]
    nodes[18] = node("Modification", "ModificationalFunctor")
    graph(sample)["edges"] = [
        edge(item["start"], item["end"], item["label"])
        for item in graph(sample)["edges"]
        if item["start"] != 18
    ]
    graph(sample)["edges"].extend(
        [
            edge(18, 2, "variable"),
            edge(18, 15, "body"),
        ]
    )
    return "replace zero-relative Ref with Modification"


def fix_dev_650(sample: dict[str, Any]) -> str:
    """Zero relative over `everything they had been doing...` should not use Ref."""
    nodes = graph(sample)["nodes"]
    nodes[19] = node("Modification", "ModificationalFunctor")
    graph(sample)["edges"] = [
        edge(item["start"], item["end"], item["label"])
        for item in graph(sample)["edges"]
        if item["start"] not in {13, 18, 19}
    ]
    graph(sample)["edges"].extend(
        [
            edge(13, 16, "pred.arg.4"),
            edge(13, 15, "pred.arg.2"),
            edge(13, 11, "pred.arg.1"),
            edge(18, 11, "pred.arg.1"),
            edge(18, 10, "pred.arg.2"),
            edge(19, 10, "variable"),
            edge(19, 12, "body"),
        ]
    )
    return "replace zero-relative Ref and restore `they` as subject of `make`"


def fix_train_4955(sample: dict[str, Any]) -> str:
    """Identifier fragment: `-` is an expression separator, not a factual predicate."""
    nodes = graph(sample)["nodes"]
    nodes[0]["type"] = "ExpressionFunctor"
    graph(sample)["edges"] = [
        edge(0, 1, "func.arg"),
        edge(0, 2, "func.arg"),
    ]
    return "convert hyphen code separator from FactualPredicator to ExpressionFunctor"


def fix_train_8156(sample: dict[str, Any]) -> str:
    """Identifier fragment: leading `-` is an expression separator, not a predicate."""
    nodes = graph(sample)["nodes"]
    nodes[0]["type"] = "ExpressionFunctor"
    graph(sample)["edges"] = [
        edge(0, 1, "func.arg"),
        edge(0, 6, "func.arg"),
        edge(4, 3, "as:modification"),
        edge(4, 5, "punctuation"),
        edge(6, 2, "variable"),
        edge(6, 4, "body"),
    ]
    return "convert leading hyphen code separator from FactualPredicator to ExpressionFunctor"


FIXES: dict[tuple[str, int], Fix] = {
    ("dev", 236): fix_dev_236,
    ("dev", 751): fix_dev_751,
    ("dev", 684): fix_dev_684,
    ("dev", 650): fix_dev_650,
    ("train", 8106): fix_train_8106,
    ("test", 765): fix_test_765,
    ("train", 5339): fix_train_5339,
    ("test", 106): fix_test_106,
    ("train", 8145): fix_train_8145,
    ("train", 3720): fix_train_3720,
    ("train", 10311): fix_train_10311,
    ("train", 5824): fix_train_5824,
    ("train", 5597): fix_train_5597,
    ("train", 4304): fix_train_4304,
    ("train", 10869): fix_train_10869,
    ("train", 4955): fix_train_4955,
    ("train", 8156): fix_train_8156,
}


REVIEWED_LEFT: dict[tuple[str, int], str] = {
    ("train", 650): "phrasal imperative; modifier chain is DESC-recoverable",
    ("train", 7668): "`never -> again -> there -> back -> go` matches outer-to-inner scope",
    ("train", 7828): "`going` selects the non-finite complement; `again` wraps that complement by DESC",
    ("train", 1143): "`got it to blackline correctly` uses arg4 to a recoverable complement wrapper",
    ("train", 9496): "`below` correctly builds the object `link below` by DESC",
    ("test", 801): "ellipsis after `and nothing` is acceptable; model collapse is not enough to rewrite",
    ("dev", 456): "identity clause plus participial modifier is recoverable",
    ("dev", 152): "`from Orthalicoidea` modifies the also-removed event by DESC",
    ("train", 10486): "`on Thursday` modifies the already-have-plans event by DESC",
    ("train", 4602): "the non-finite complement and rhythmic/natural modifier remain recoverable",
    ("train", 10321): "purpose coordination under `come here` is recoverable",
    ("train", 6059): "question wrapper and purpose infinitive are recoverable",
    ("train", 2982): "clausal object of `think` is correct",
    ("train", 8487): "`the last place` is a result complement of `make`",
    ("train", 9305): "`back to Earth` is a recoverable transmission modifier",
    ("train", 9596): "copula-like `feel soft and soothing` is a lexical predicate choice",
    ("train", 3327): "`go hungry` is an acceptable idiomatic predicate-complement structure",
    ("train", 10067): "already normalized Attribute key-value record",
    ("train", 687): "already normalized Attribute key-value record",
    ("train", 5965): "title-like key-value fragment is already Attribute",
    ("train", 1458): "already normalized Attribute key-value record",
    ("train", 10663): "already normalized Attribute key-value record",
    ("test", 153): "OK and smiley punctuation close a DESC-recoverable clause",
    ("train", 11099): "`just` remains an explicit modifier of the simple copular predicate",
    ("train", 286): "list marker is discourse metadata over a recoverable clause",
    ("train", 8908): "`Why not` question wrapper and phrasal verb are recoverable",
}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_split = {split: read_jsonl(DATA / f"{split}.jsonl") for split in ("train", "dev", "test")}
    changes: list[dict[str, Any]] = []

    for (split, line_no), fix in FIXES.items():
        sample = rows_by_split[split][line_no - 1]
        before = deepcopy(sample)
        reason = fix(sample)
        if sample != before:
            changes.append(
                {
                    "sample_id": f"{split}:{line_no}",
                    "sentence": sample["sentence"],
                    "reason": reason,
                    "before": compact(before),
                    "after": compact(sample),
                }
            )

    for split, rows in rows_by_split.items():
        write_jsonl(DATA / f"{split}.jsonl", rows)

    (REPORT_DIR / "changes.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in changes) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "reviewed_left.json").write_text(
        json.dumps(
            [
                {"sample_id": f"{split}:{line_no}", "reason": reason}
                for (split, line_no), reason in sorted(REVIEWED_LEFT.items())
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "summary.md").write_text(
        "\n".join(
            [
                "# Manual P0/P1 Review",
                "",
                f"- Applied fixes: {len(changes)}",
                f"- Reviewed and left unchanged: {len(REVIEWED_LEFT)}",
                "",
                "## Applied Fixes",
                "",
                *[
                    f"- `{row['sample_id']}`: {row['reason']} — {row['sentence']}"
                    for row in changes
                ],
                "",
                "## Reviewed And Left",
                "",
                *[
                    f"- `{split}:{line_no}`: {reason}"
                    for (split, line_no), reason in sorted(REVIEWED_LEFT.items())
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"changed_samples": len(changes), "report_dir": str(REPORT_DIR)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
