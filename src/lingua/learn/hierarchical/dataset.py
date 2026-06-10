from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from .oracle import CONTENT_VIEW, DEFAULT_VIEW, extract_hierarchical_oracle


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
QUERY_ROOT_ROLE = "ROOT"
SHARED_NONE_ROLE = "O"
LEAF_DECISION = "leaf"
EXPAND_DECISION = "expand"
DECISION_LABELS = [LEAF_DECISION, EXPAND_DECISION]


@dataclass(frozen=True)
class HierarchicalTrainingExample:
    sample_id: str
    unit_index: int
    query_index: int
    parent_unit_index: int | None
    words: tuple[str, ...]
    input_token_ids: tuple[int, ...]
    view_kind: str
    unit_pos: str
    aux_kind: str
    decision: str
    role_labels_by_token: tuple[tuple[int, tuple[str, ...]], ...]
    query_role_by_token: tuple[tuple[int, str], ...]
    shared_role_by_token: tuple[tuple[int, tuple[str, ...]], ...]


@dataclass(frozen=True)
class HierarchicalVocab:
    word2id: dict[str, int]
    view_kind2id: dict[str, int]
    query_role2id: dict[str, int]
    shared_role2id: dict[str, int]
    unit_pos2id: dict[str, int]
    aux_kind2id: dict[str, int]
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
            "view_kind2id": self.view_kind2id,
            "query_role2id": self.query_role2id,
            "shared_role2id": self.shared_role2id,
            "unit_pos2id": self.unit_pos2id,
            "aux_kind2id": self.aux_kind2id,
            "role_label2id": self.role_label2id,
            "decision2id": self.decision2id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HierarchicalVocab":
        return cls(
            word2id={str(key): int(value) for key, value in data["word2id"].items()},
            view_kind2id={
                str(key): int(value)
                for key, value in data.get(
                    "view_kind2id",
                    {DEFAULT_VIEW: 0, CONTENT_VIEW: 1},
                ).items()
            },
            query_role2id={
                str(key): int(value)
                for key, value in data.get(
                    "query_role2id",
                    {PAD_TOKEN: 0, QUERY_ROOT_ROLE: 1},
                ).items()
            },
            shared_role2id={
                str(key): int(value)
                for key, value in data.get(
                    "shared_role2id",
                    {PAD_TOKEN: 0, SHARED_NONE_ROLE: 1},
                ).items()
            },
            unit_pos2id={str(key): int(value) for key, value in data["unit_pos2id"].items()},
            aux_kind2id={str(key): int(value) for key, value in data["aux_kind2id"].items()},
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
        query_role_ids = [self.vocab.query_role2id[PAD_TOKEN]] * len(example.words)
        shared_role_ids = [self.vocab.shared_role2id[PAD_TOKEN]] * len(example.words)
        role_targets = [
            [0.0] * len(self.vocab.role_label2id)
            for _ in example.words
        ]

        for token_id in example.input_token_ids:
            if 0 <= token_id < len(unit_mask):
                unit_mask[token_id] = 1

        for token_id, role in example.query_role_by_token:
            if 0 <= token_id < len(query_role_ids):
                query_role_ids[token_id] = self.vocab.query_role2id.get(
                    role,
                    self.vocab.query_role2id[QUERY_ROOT_ROLE],
                )

        for token_id, roles in example.shared_role_by_token:
            if 0 <= token_id < len(shared_role_ids):
                shared_role_ids[token_id] = self.vocab.shared_role2id.get(
                    _shared_role_key(roles),
                    self.vocab.shared_role2id[SHARED_NONE_ROLE],
                )

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
            "query_index": example.query_index,
            "parent_unit_index": example.parent_unit_index,
            "word_ids": word_ids,
            "unit_mask": unit_mask,
            "query_role_ids": query_role_ids,
            "shared_role_ids": shared_role_ids,
            "role_targets": role_targets,
            "unit_pos": self.vocab.unit_pos2id.get(example.unit_pos, self.vocab.unit_pos2id[UNK_TOKEN]),
            "aux_kind": self.vocab.aux_kind2id.get(example.aux_kind, self.vocab.aux_kind2id[UNK_TOKEN]),
            "view_kind": self.vocab.view_kind2id.get(example.view_kind, self.vocab.view_kind2id[DEFAULT_VIEW]),
            "decision": self.vocab.decision2id[example.decision],
            "force_role_per_token": not any(
                labels == ("O",)
                for token_id, labels in example.role_labels_by_token
                if token_id in set(example.input_token_ids)
            ),
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
            sample_examples = build_hierarchical_examples_from_oracle(sample_id, oracle)
            for example in sample_examples:
                examples.append(example)
                if max_units is not None and len(examples) >= max_units:
                    return examples

    return examples


def build_hierarchical_examples_from_oracle(
    sample_id: str,
    oracle: dict[str, Any],
) -> list[HierarchicalTrainingExample]:
    words = tuple(str(word) for word in oracle["words"])
    units = list(oracle["units"])
    node_to_unit_index = {
        int(unit["node_index"]): unit_index
        for unit_index, unit in enumerate(units)
    }
    examples: list[HierarchicalTrainingExample] = []

    for unit_index, unit in enumerate(units):
        if not unit.get("is_root", False):
            continue
        examples.append(
            _training_example_from_unit(
                sample_id=sample_id,
                query_index=len(examples),
                unit_index=unit_index,
                parent_unit_index=None,
                words=words,
                unit=unit,
                query_role_by_token=_root_query_role_by_token(unit),
                shared_role_by_token=_empty_shared_role_by_token(unit),
            )
        )

    for parent_unit_index, parent_unit in enumerate(units):
        for child in parent_unit.get("children", []):
            if not _should_enqueue_child_query(parent_unit, child):
                continue
            child_node_index = int(child["node_index"])
            child_unit_index = node_to_unit_index.get(child_node_index)
            if child_unit_index is None:
                continue
            child_unit = units[child_unit_index]
            examples.append(
                _training_example_from_unit(
                    sample_id=sample_id,
                    query_index=len(examples),
                    unit_index=child_unit_index,
                    parent_unit_index=parent_unit_index,
                    words=words,
                    unit=child_unit,
                    query_role_by_token=_child_query_role_by_token(child),
                    shared_role_by_token=_child_shared_role_by_token(child),
                )
            )

    return examples


def _training_example_from_unit(
    *,
    sample_id: str,
    query_index: int,
    unit_index: int,
    parent_unit_index: int | None,
    words: tuple[str, ...],
    unit: dict[str, Any],
    query_role_by_token: tuple[tuple[int, str], ...],
    shared_role_by_token: tuple[tuple[int, tuple[str, ...]], ...],
) -> HierarchicalTrainingExample:
    return HierarchicalTrainingExample(
        sample_id=sample_id,
        unit_index=unit_index,
        query_index=query_index,
        parent_unit_index=parent_unit_index,
        words=words,
        input_token_ids=tuple(int(token_id) for token_id in unit["input_token_ids"]),
        view_kind=_view_kind(str(unit.get("view_kind", DEFAULT_VIEW))),
        unit_pos=str(unit["unit_pos"]),
        aux_kind=str(unit["aux_kind"]),
        decision=str(unit["decision"]),
        role_labels_by_token=tuple(
            (
                int(token_label["token_id"]),
                tuple(str(label) for label in token_label["labels"]),
            )
            for token_label in unit["token_labels"]
        ),
        query_role_by_token=query_role_by_token,
        shared_role_by_token=shared_role_by_token,
    )


def _should_enqueue_child_query(parent_unit: dict[str, Any], child: dict[str, Any]) -> bool:
    parent_token_ids = tuple(int(token_id) for token_id in parent_unit["input_token_ids"])
    child_token_ids = tuple(int(token_id) for token_id in child["token_ids"])
    if child_token_ids == parent_token_ids and not str(child["role"]).endswith("_REF"):
        return False
    return True


def _root_query_role_by_token(unit: dict[str, Any]) -> tuple[tuple[int, str], ...]:
    return tuple(
        (int(token_id), QUERY_ROOT_ROLE)
        for token_id in unit["input_token_ids"]
    )


def _empty_shared_role_by_token(unit: dict[str, Any]) -> tuple[tuple[int, tuple[str, ...]], ...]:
    return tuple(
        (int(token_id), (SHARED_NONE_ROLE,))
        for token_id in unit["input_token_ids"]
    )


def _child_query_role_by_token(child: dict[str, Any]) -> tuple[tuple[int, str], ...]:
    role = _canonical_query_role(str(child["role"]))
    return tuple(
        (int(token_id), role)
        for token_id in child["token_ids"]
    )


def _child_shared_role_by_token(child: dict[str, Any]) -> tuple[tuple[int, tuple[str, ...]], ...]:
    token_ids = tuple(int(token_id) for token_id in child["token_ids"])
    shared_roles: dict[int, set[str]] = {token_id: set() for token_id in token_ids}
    for link in child.get("shared_arg_links", []):
        shared_role = _canonical_query_role(str(link["shared_role"]))
        for token_id in link.get("token_ids", []):
            token_id = int(token_id)
            if token_id in shared_roles:
                shared_roles[token_id].add(shared_role)
    return tuple(
        (
            token_id,
            tuple(sorted(shared_roles[token_id])) or (SHARED_NONE_ROLE,),
        )
        for token_id in token_ids
    )


def _canonical_query_role(role: str) -> str:
    return role[:-4] if role.endswith("_REF") else role


def _view_kind(view_kind: str) -> str:
    return DEFAULT_VIEW if view_kind == DEFAULT_VIEW else CONTENT_VIEW


def _shared_role_key(roles: tuple[str, ...]) -> str:
    clean_roles = tuple(sorted(role for role in roles if role != SHARED_NONE_ROLE))
    return "+".join(clean_roles) if clean_roles else SHARED_NONE_ROLE


def build_vocab(
    examples: Iterable[HierarchicalTrainingExample],
    min_word_freq: int = 1,
) -> HierarchicalVocab:
    word_counter: Counter[str] = Counter()
    view_kinds = set()
    query_roles = set()
    shared_roles = set()
    unit_pos = set()
    aux_kinds = set()
    role_labels = set()

    for example in examples:
        word_counter.update(_normalize_word(word) for word in example.words)
        view_kinds.add(_view_kind(example.view_kind))
        query_roles.update(role for _, role in example.query_role_by_token)
        shared_roles.update(_shared_role_key(roles) for _, roles in example.shared_role_by_token)
        unit_pos.add(example.unit_pos)
        aux_kinds.add(example.aux_kind)
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
        view_kind2id={
            label: index
            for index, label in enumerate(
                [
                    DEFAULT_VIEW,
                    CONTENT_VIEW,
                    *sorted(view_kinds - {DEFAULT_VIEW, CONTENT_VIEW}),
                ]
            )
        },
        query_role2id={
            label: index
            for index, label in enumerate(
                [
                    PAD_TOKEN,
                    QUERY_ROOT_ROLE,
                    *sorted(query_roles - {PAD_TOKEN, QUERY_ROOT_ROLE}),
                ]
            )
        },
        shared_role2id={
            label: index
            for index, label in enumerate(
                [
                    PAD_TOKEN,
                    SHARED_NONE_ROLE,
                    *sorted(shared_roles - {PAD_TOKEN, SHARED_NONE_ROLE}),
                ]
            )
        },
        unit_pos2id={
            label: index
            for index, label in enumerate([UNK_TOKEN, *sorted(unit_pos)])
        },
        aux_kind2id={
            label: index
            for index, label in enumerate([UNK_TOKEN, *sorted(aux_kinds)])
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
    query_role_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    shared_role_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    role_targets = torch.zeros((batch_size, max_len, num_role_labels), dtype=torch.float)
    unit_pos = torch.zeros(batch_size, dtype=torch.long)
    aux_kind = torch.zeros(batch_size, dtype=torch.long)
    view_kind = torch.zeros(batch_size, dtype=torch.long)
    decision = torch.zeros(batch_size, dtype=torch.long)
    force_role_per_token = torch.zeros(batch_size, dtype=torch.bool)

    for row_index, item in enumerate(batch):
        length = item["length"]
        word_ids[row_index, :length] = torch.tensor(item["word_ids"], dtype=torch.long)
        attention_mask[row_index, :length] = True
        unit_mask[row_index, :length] = torch.tensor(item["unit_mask"], dtype=torch.bool)
        query_role_ids[row_index, :length] = torch.tensor(item["query_role_ids"], dtype=torch.long)
        shared_role_ids[row_index, :length] = torch.tensor(item["shared_role_ids"], dtype=torch.long)
        role_targets[row_index, :length] = torch.tensor(item["role_targets"], dtype=torch.float)
        unit_pos[row_index] = int(item["unit_pos"])
        aux_kind[row_index] = int(item["aux_kind"])
        view_kind[row_index] = int(item["view_kind"])
        decision[row_index] = int(item["decision"])
        force_role_per_token[row_index] = bool(item.get("force_role_per_token", True))

    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "unit_indices": [item["unit_index"] for item in batch],
        "query_indices": [item["query_index"] for item in batch],
        "parent_unit_indices": [item["parent_unit_index"] for item in batch],
        "word_ids": word_ids,
        "attention_mask": attention_mask,
        "unit_mask": unit_mask,
        "query_role_ids": query_role_ids,
        "shared_role_ids": shared_role_ids,
        "role_targets": role_targets,
        "unit_pos": unit_pos,
        "aux_kind": aux_kind,
        "view_kind": view_kind,
        "decision": decision,
        "force_role_per_token": force_role_per_token,
    }


def _normalize_word(word: str) -> str:
    normalized = word.strip().lower()
    return normalized if normalized else word.lower()
