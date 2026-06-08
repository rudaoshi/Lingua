#!/usr/bin/env python3
"""Classify particle/adverb-like tokens merged into FactualPredicator nodes.

This is a read-only audit. It separates merged post-verbal material into:

- keep_particle: likely lexical verb-particle predicates such as "find out".
- split_non_particle: locative/state/prepositional/directional material that is
  not acting as a verb particle.
- review: ambiguous cases that should be checked before an automatic repair.
- ignore_not_postmodifier: list pollution such as lexical "open"/"close" verbs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

CANDIDATE_TOKENS = {
    "about",
    "across",
    "ahead",
    "along",
    "around",
    "aside",
    "away",
    "back",
    "behind",
    "by",
    "close",
    "down",
    "forth",
    "forward",
    "here",
    "home",
    "in",
    "inside",
    "north",
    "off",
    "on",
    "open",
    "out",
    "outside",
    "over",
    "south",
    "there",
    "through",
    "together",
    "under",
    "up",
    "west",
}

BE_FORMS = {
    "'m",
    "'re",
    "'s",
    "am",
    "are",
    "be",
    "been",
    "being",
    "is",
    "m",
    "re",
    "s",
    "was",
    "were",
}

AUXILIARY_FORMS = {
    "'d",
    "'ll",
    "'ve",
    "can",
    "ca",
    "could",
    "d",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "may",
    "might",
    "must",
    "need",
    "needed",
    "shall",
    "should",
    "to",
    "will",
    "ll",
    "ve",
    "wo",
    "would",
}

FUNCTION_TOKENS = {
    "",
    "'",
    "-",
    "and",
    "n",
    "n't",
    "not",
    "nt",
    "t",
}

IRREGULAR_LEMMAS = {
    "blew": "blow",
    "blown": "blow",
    "bought": "buy",
    "broke": "break",
    "broken": "break",
    "brought": "bring",
    "came": "come",
    "chased": "chase",
    "closed": "close",
    "done": "do",
    "drove": "drive",
    "dropped": "drop",
    "fell": "fall",
    "found": "find",
    "gave": "give",
    "gone": "go",
    "got": "get",
    "had": "have",
    "hung": "hang",
    "laid": "lay",
    "laying": "lay",
    "led": "lead",
    "left": "leave",
    "made": "make",
    "paid": "pay",
    "put": "put",
    "ran": "run",
    "sent": "send",
    "set": "set",
    "shot": "shoot",
    "sped": "speed",
    "stood": "stand",
    "struck": "strike",
    "swept": "sweep",
    "taken": "take",
    "thought": "think",
    "took": "take",
    "torn": "tear",
    "ve": "have",
    "went": "go",
    "woke": "wake",
    "wo": "will",
}

# Pairs here are deliberately lexicalized. Directional or PP-like combinations
# such as "walk down", "focus on", and "be in" are not included.
KEEP_PARTICLE_LEMMAS: dict[str, set[str]] = {
    "about": {"set"},
    "across": {"come", "run"},
    "ahead": {"go"},
    "along": {"get", "go", "hum", "pass"},
    "around": {"get", "switch", "turn"},
    "aside": {"put", "push", "set"},
    "away": {
        "blow",
        "come",
        "get",
        "go",
        "lead",
        "move",
        "pass",
        "push",
        "run",
        "shy",
        "sweep",
        "take",
        "throw",
        "wash",
    },
    "back": {
        "back",
        "blow",
        "call",
        "check",
        "chase",
        "come",
        "get",
        "go",
        "head",
        "hear",
        "kick",
        "pull",
        "put",
        "rush",
        "strike",
        "switch",
        "turn",
    },
    "by": {"stand", "stop"},
    "down": {
        "break",
        "bring",
        "calm",
        "close",
        "crack",
        "cut",
        "get",
        "lay",
        "nail",
        "play",
        "put",
        "settle",
        "shoot",
        "strike",
        "take",
        "talk",
        "tear",
        "wind",
    },
    "forth": {"bring"},
    "forward": {"look"},
    "in": {
        "bring",
        "check",
        "come",
        "dine",
        "drive",
        "fill",
        "give",
        "log",
        "move",
        "send",
        "take",
        "turn",
        "weigh",
    },
    "off": {
        "call",
        "cut",
        "draw",
        "drop",
        "fall",
        "fend",
        "get",
        "haul",
        "knock",
        "leave",
        "lick",
        "pay",
        "put",
        "rip",
        "send",
        "slack",
        "step",
        "take",
        "twist",
        "ward",
        "walk",
        "write",
    },
    "on": {
        "carry",
        "come",
        "go",
        "move",
        "pass",
        "put",
        "take",
        "try",
    },
    "out": {
        "back",
        "break",
        "carry",
        "check",
        "come",
        "drop",
        "fan",
        "find",
        "get",
        "hang",
        "help",
        "make",
        "point",
        "put",
        "rule",
        "work",
        "shell",
        "sort",
        "strike",
        "try",
        "turn",
    },
    "over": {
        "carry",
        "come",
        "go",
        "hand",
        "head",
        "knock",
        "pull",
        "roll",
        "send",
        "take",
        "turn",
    },
    "through": {"blow", "come", "fall", "follow", "get", "push"},
    "together": {"get"},
    "up": {
        "back",
        "beef",
        "break",
        "bring",
        "build",
        "clean",
        "come",
        "divy",
        "draw",
        "dry",
        "edge",
        "end",
        "follow",
        "give",
        "grow",
        "hang",
        "jam",
        "line",
        "look",
        "make",
        "mess",
        "open",
        "pick",
        "rack",
        "screw",
        "set",
        "show",
        "sign",
        "step",
        "stir",
        "sum",
        "take",
        "tie",
        "wake",
        "wrap",
    },
}

KNOWN_LEMMA_VOCAB = set().union(*KEEP_PARTICLE_LEMMAS.values())

PREPOSITIONAL_OR_STATE_DEFAULT = {
    "about",
    "behind",
    "here",
    "home",
    "in",
    "inside",
    "north",
    "on",
    "outside",
    "south",
    "there",
    "under",
    "west",
}

DIRECTIONAL_DEFAULT = {
    "across",
    "ahead",
    "around",
    "by",
    "down",
    "forward",
    "over",
    "through",
    "together",
}

HEAD_VERB_CANDIDATES = {"close", "forward", "open"}


def norm(token: str) -> str:
    return token.strip().lower().replace("’", "'")


def lemma_like(token: str) -> str:
    token = norm(token)
    if token in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        stem = token[:-3]
        candidates = [stem]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            candidates.append(stem[:-1])
        candidates.append(stem + "e")
        for candidate in candidates:
            if candidate in KNOWN_LEMMA_VOCAB:
                return candidate
        return candidates[0]
    if len(token) > 4 and token.endswith("ied"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ed"):
        stem = token[:-2]
        candidates = [stem]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            candidates.append(stem[:-1])
        candidates.append(stem + "e")
        for candidate in candidates:
            if candidate in KNOWN_LEMMA_VOCAB:
                return candidate
        return candidates[0]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def span_pairs(spans: Any) -> list[tuple[int, int]]:
    if not isinstance(spans, list):
        return []
    pairs: list[tuple[int, int]] = []
    for span in spans:
        if isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span):
            pairs.append((span[0], span[1]))
    return pairs


def node_tokens(sample: dict[str, Any], node_index: int) -> list[tuple[int, str, str]]:
    node = graph(sample)["nodes"][node_index]
    words = sample["lingua"]["words"]
    tokens: list[tuple[int, str, str]] = []
    for start, end in span_pairs(node.get("spans")):
        for index in range(start, end + 1):
            if 0 <= index < len(words):
                raw = words[index].strip()
                tokens.append((index, raw, norm(raw)))
    return sorted(tokens)


def node_text(sample: dict[str, Any], node_index: int) -> str:
    text = " ".join(token for _, token, _ in node_tokens(sample, node_index)).strip()
    if text:
        return text
    return str(graph(sample)["nodes"][node_index].get("spans"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def content_lemmas_before(tokens: list[str], position: int) -> list[str]:
    ignored = BE_FORMS | AUXILIARY_FORMS | FUNCTION_TOKENS
    lemmas: list[str] = []
    for token in tokens[:position]:
        if token in ignored or token in CANDIDATE_TOKENS:
            continue
        lemmas.append(lemma_like(token))
    return lemmas


def has_be(tokens: list[str]) -> bool:
    return any(token in BE_FORMS for token in tokens)


def has_cmon_shape(tokens: list[str], target: str) -> bool:
    return target == "on" and tokens == ["c", "'", "m", "on"]


def is_directional_coordination(tokens: list[str], target: str) -> bool:
    direction_tokens = {"up", "down", "back", "forth"}
    return target in direction_tokens and "and" in tokens and len(direction_tokens & set(tokens)) >= 2


def classify(tokens: list[str], target: str, position: int) -> tuple[str, str]:
    lemmas_before = content_lemmas_before(tokens, position)
    keep_lemmas = KEEP_PARTICLE_LEMMAS.get(target, set())

    if has_cmon_shape(tokens, target):
        return "keep_particle", "tokenized_cmon_lexicalized_come_on"

    if is_directional_coordination(tokens, target):
        return "split_non_particle", "directional_coordination_not_particle"

    if target in {"up", "close"} and "up" in tokens and "close" in tokens:
        return "split_non_particle", "fixed_adverbial_up_close_not_particle"

    if target in HEAD_VERB_CANDIDATES and has_be(tokens) and not lemmas_before:
        return "split_non_particle", "copular_state_not_particle"

    if target in HEAD_VERB_CANDIDATES and not lemmas_before:
        return "ignore_not_postmodifier", "candidate_is_lexical_head_verb"

    if target == "close" and lemmas_before:
        return "split_non_particle", "locative_or_degree_close_not_particle"

    if target == "forward" and not lemmas_before:
        return "ignore_not_postmodifier", "candidate_is_lexical_head_verb"

    if target in {"here", "there", "home", "inside", "outside", "north", "south", "west", "under", "behind"}:
        return "split_non_particle", "locative_or_state_word_not_particle"

    if target == "on" and "board" in tokens[position + 1 :]:
        return "split_non_particle", "fixed_pp_on_board_not_particle"

    if any(lemma in keep_lemmas for lemma in lemmas_before):
        return "keep_particle", "known_lexical_verb_particle_pair"

    if has_be(tokens) and target in {
        "about",
        "around",
        "away",
        "back",
        "down",
        "in",
        "off",
        "on",
        "out",
        "over",
        "through",
    }:
        return "split_non_particle", "copular_location_or_state_not_particle"

    if target in PREPOSITIONAL_OR_STATE_DEFAULT:
        return "split_non_particle", "prepositional_or_state_default_not_particle"

    if target in DIRECTIONAL_DEFAULT:
        return "split_non_particle", "directional_or_adverbial_default_not_particle"

    if target in {"off", "out", "up", "away", "back", "aside", "forth", "along"}:
        return "keep_particle", "default_core_particle_after_verb"

    return "review", "ambiguous_particle_or_adverb"


def iter_cases(data_dir: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.jsonl"
        for line_number, sample in enumerate(read_jsonl(path), 1):
            nodes = graph(sample)["nodes"]
            for node_index, node in enumerate(nodes):
                if node.get("type") != "FactualPredicator":
                    continue
                token_records = node_tokens(sample, node_index)
                normalized = [record[2] for record in token_records]
                seen_in_node: set[str] = set()
                for position, target in enumerate(normalized):
                    if target not in CANDIDATE_TOKENS or target in seen_in_node:
                        continue
                    seen_in_node.add(target)
                    bucket, reason = classify(normalized, target, position)
                    cases.append(
                        {
                            "split": split,
                            "line_no": line_number,
                            "node_id": node_index,
                            "token": target,
                            "bucket": bucket,
                            "reason": reason,
                            "predicator": node_text(sample, node_index),
                            "tokens": " ".join(normalized),
                            "spans": json.dumps(node.get("spans"), ensure_ascii=False, separators=(",", ":")),
                            "sentence": sample.get("sentence", ""),
                        }
                    )
    return cases


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(cases: list[dict[str, Any]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "line_no",
        "node_id",
        "token",
        "bucket",
        "reason",
        "predicator",
        "tokens",
        "spans",
        "sentence",
    ]
    write_tsv(report_dir / "cases.tsv", cases, fields)

    for bucket in ("split_non_particle", "review", "keep_particle", "ignore_not_postmodifier"):
        bucket_rows = [case for case in cases if case["bucket"] == bucket]
        write_tsv(report_dir / f"{bucket}.tsv", bucket_rows, fields)

    bucket_counts = Counter(case["bucket"] for case in cases)
    token_bucket_counts = Counter((case["bucket"], case["token"]) for case in cases)
    reason_counts = Counter((case["bucket"], case["reason"]) for case in cases)

    summary_rows: list[dict[str, Any]] = []
    for bucket, count in bucket_counts.most_common():
        summary_rows.append({"section": "bucket", "bucket": bucket, "token": "", "reason": "", "count": count})
    for (bucket, token), count in sorted(token_bucket_counts.items()):
        summary_rows.append({"section": "bucket_token", "bucket": bucket, "token": token, "reason": "", "count": count})
    for (bucket, reason), count in sorted(reason_counts.items()):
        summary_rows.append({"section": "bucket_reason", "bucket": bucket, "token": "", "reason": reason, "count": count})
    write_tsv(report_dir / "summary.tsv", summary_rows, ["section", "bucket", "token", "reason", "count"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORTS / "factual_predicator_particle_filter",
    )
    args = parser.parse_args()

    cases = iter_cases(args.data_dir)
    write_reports(cases, args.report_dir)
    bucket_counts = Counter(case["bucket"] for case in cases)
    print(f"wrote {len(cases)} cases to {args.report_dir}")
    for bucket, count in bucket_counts.most_common():
        print(f"{bucket}\t{count}")


if __name__ == "__main__":
    main()
