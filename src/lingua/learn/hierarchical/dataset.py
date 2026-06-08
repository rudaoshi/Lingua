from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from .oracle import extract_hierarchical_oracle


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
LEAF_DECISION = "leaf"
EXPAND_DECISION = "expand"
DECISION_LABELS = [LEAF_DECISION, EXPAND_DECISION]


@dataclass(frozen=True)
class HierarchicalTrainingExample:
    sample_id: str
    unit_index: int
    words: tuple[str, ...]
    input_token_ids: tuple[int, ...]
    unit_kind: str
    decision: str
    role_labels_by_token: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class HierarchicalVocab:
    word2id: dict[str, int]
    unit_kind2id: dict[str, int]
    role_label2id: dict[str, int]
    decision2id: dict[str, int]

    @property
    def pad_id(self) -> int:
        return self.word2id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.word2id[UNK_TOKEN]

    def to_dict(self) -> dict[str, Any]:
        return {
            "word2id": self.word2id,
            "unit_kind2id": self.unit_kind2id,
            "role_label2id": self.role_label2id,
            "decision2id": self.decision2id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HierarchicalVocab":
        return cls(
            word2id={str(key): int(value) for key, value in data["word2id"].items()},
            unit_kind2id={str(key): int(value) for key, value in data["unit_kind2id"].items()},
            role_label2id={str(key): int(value) for key, value in data["role_label2id"].items()},
            decision2id={str(key): int(value) for key, value in data["decision2id"].items()},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "HierarchicalVocab":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


class HierarchicalOracleDataset(Dataset):
    def __init__(self, examples: list[HierarchicalTrainingExample], vocab: HierarchicalVocab):
        self.examples = examples
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        word_ids = [
            self.vocab.word2id.get(_normalize_word(word), self.vocab.unk_id)
            for word in example.words
        ]
        unit_mask = [0] * len(example.words)
        role_targets = [
            [0.0] * len(self.vocab.role_label2id)
            for _ in example.words
        ]

        for token_id in example.input_token_ids:
            if 0 <= token_id < len(unit_mask):
                unit_mask[token_id] = 1

        for token_id, labels in example.role_labels_by_token:
            if token_id < 0 or token_id >= len(role_targets):
                continue
            for label in labels:
                if label == "O":
                    continue
                label_id = self.vocab.role_label2id.get(label)
                if label_id is not None:
                    role_targets[token_id][label_id] = 1.0

        return {
            "sample_id": example.sample_id,
            "unit_index": example.unit_index,
            "word_ids": word_ids,
            "unit_mask": unit_mask,
            "role_targets": role_targets,
            "unit_kind": self.vocab.unit_kind2id.get(example.unit_kind, self.vocab.unit_kind2id[UNK_TOKEN]),
            "decision": self.vocab.decision2id[example.decision],
            "length": len(word_ids),
        }


def build_hierarchical_examples(
    data_path: Path,
    max_samples: int | None = None,
    max_units: int | None = None,
) -> list[HierarchicalTrainingExample]:
    examples: list[HierarchicalTrainingExample] = []

    with data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_samples is not None and line_number > max_samples:
                break
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample_id = str(sample.get("id") or f"{data_path.name}:{line_number}")
            oracle = extract_hierarchical_oracle(sample).to_dict()

            for unit_index, unit in enumerate(oracle["units"]):
                labels_by_token = tuple(
                    (
                        int(token_label["token_id"]),
                        tuple(str(label) for label in token_label["labels"]),
                    )
                    for token_label in unit["token_labels"]
                )
                examples.append(
                    HierarchicalTrainingExample(
                        sample_id=sample_id,
                        unit_index=unit_index,
                        words=tuple(str(word) for word in oracle["words"]),
                        input_token_ids=tuple(int(token_id) for token_id in unit["input_token_ids"]),
                        unit_kind=str(unit["unit_kind"]),
                        decision=str(unit["decision"]),
                        role_labels_by_token=labels_by_token,
                    )
                )
                if max_units is not None and len(examples) >= max_units:
                    return examples

    return examples


def build_vocab(
    examples: Iterable[HierarchicalTrainingExample],
    min_word_freq: int = 1,
) -> HierarchicalVocab:
    word_counter: Counter[str] = Counter()
    unit_kinds = set()
    role_labels = set()

    for example in examples:
        word_counter.update(_normalize_word(word) for word in example.words)
        unit_kinds.add(example.unit_kind)
        for _, labels in example.role_labels_by_token:
            for label in labels:
                if label != "O":
                    role_labels.add(label)

    word2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, count in sorted(word_counter.items()):
        if count >= min_word_freq and word not in word2id:
            word2id[word] = len(word2id)

    return HierarchicalVocab(
        word2id=word2id,
        unit_kind2id={
            label: index
            for index, label in enumerate([UNK_TOKEN, *sorted(unit_kinds)])
        },
        role_label2id={label: index for index, label in enumerate(sorted(role_labels))},
        decision2id={label: index for index, label in enumerate(DECISION_LABELS)},
    )


def collate_hierarchical_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)
    num_role_labels = len(batch[0]["role_targets"][0]) if max_len else 0

    word_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    unit_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    role_targets = torch.zeros((batch_size, max_len, num_role_labels), dtype=torch.float)
    unit_kind = torch.zeros(batch_size, dtype=torch.long)
    decision = torch.zeros(batch_size, dtype=torch.long)

    for row_index, item in enumerate(batch):
        length = item["length"]
        word_ids[row_index, :length] = torch.tensor(item["word_ids"], dtype=torch.long)
        attention_mask[row_index, :length] = True
        unit_mask[row_index, :length] = torch.tensor(item["unit_mask"], dtype=torch.bool)
        role_targets[row_index, :length] = torch.tensor(item["role_targets"], dtype=torch.float)
        unit_kind[row_index] = int(item["unit_kind"])
        decision[row_index] = int(item["decision"])

    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "unit_indices": [item["unit_index"] for item in batch],
        "word_ids": word_ids,
        "attention_mask": attention_mask,
        "unit_mask": unit_mask,
        "role_targets": role_targets,
        "unit_kind": unit_kind,
        "decision": decision,
    }


def _normalize_word(word: str) -> str:
    normalized = word.strip().lower()
    return normalized if normalized else word.lower()
