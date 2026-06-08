#!/usr/bin/env python3
"""Run Stanza UD parsing for saved prediction dump samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "test")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def done_lines(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("line") is not None:
                done.add(int(row["line"]))
    return done


def word_to_json(word: Any) -> dict[str, Any]:
    return {
        "id": int(word.id) if isinstance(word.id, int) else word.id,
        "text": word.text,
        "lemma": word.lemma,
        "upos": word.upos,
        "xpos": word.xpos,
        "feats": word.feats,
        "head": int(word.head) if word.head is not None else None,
        "deprel": word.deprel,
        "start_char": getattr(word, "start_char", None),
        "end_char": getattr(word, "end_char", None),
    }


def parse_split(args: argparse.Namespace, nlp: Any, split: str) -> dict[str, Any]:
    input_path = args.prediction_base / split / "predictions.jsonl"
    output_path = args.output_dir / f"{split}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = done_lines(output_path) if args.resume else set()

    rows = [row for row in read_jsonl(input_path) if int(row["line"]) not in existing]
    if args.max_samples:
        rows = rows[: args.max_samples]

    total = len(rows)
    processed = 0
    errors = 0
    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as out:
        for start in range(0, total, args.batch_size):
            batch = rows[start : start + args.batch_size]
            sentences = [row["sentence"] for row in batch]
            try:
                docs = nlp.bulk_process(sentences)
            except Exception:
                docs = []
                for sentence in sentences:
                    try:
                        docs.append(nlp(sentence))
                    except Exception as exc:
                        docs.append(exc)
            for row, doc in zip(batch, docs):
                if isinstance(doc, Exception):
                    record = {
                        "split": split,
                        "idx": int(row["idx"]),
                        "line": int(row["line"]),
                        "sentence": row["sentence"],
                        "words": row.get("words") or [],
                        "error": str(doc),
                    }
                    errors += 1
                else:
                    sentences_json = []
                    for sent in doc.sentences:
                        sentences_json.append(
                            {
                                "tokens": [
                                    {
                                        "id": token.id,
                                        "text": token.text,
                                        "start_char": getattr(token, "start_char", None),
                                        "end_char": getattr(token, "end_char", None),
                                    }
                                    for token in sent.tokens
                                ],
                                "words": [word_to_json(word) for word in sent.words],
                            }
                        )
                    record = {
                        "split": split,
                        "idx": int(row["idx"]),
                        "line": int(row["line"]),
                        "sentence": row["sentence"],
                        "words": row.get("words") or [],
                        "sentences": sentences_json,
                    }
                out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                processed += 1
            out.flush()
            print(f"{split}: {processed}/{total} newly parsed", flush=True)
    return {
        "split": split,
        "input": str(input_path),
        "output": str(output_path),
        "skipped_existing": len(existing),
        "newly_processed": processed,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction_base",
        type=Path,
        default=Path("reports/v13_overwrite_latest_prediction_dump"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/v13_overwrite_latest_ud_stanza"),
    )
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import stanza

    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,pos,lemma,depparse",
        tokenize_no_ssplit=True,
        use_gpu=args.use_gpu,
        verbose=False,
    )
    summaries = [parse_split(args, nlp, split) for split in args.splits]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "summaries": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
