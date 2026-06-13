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
O_ROLE_LABEL = "O"
LEAF_DECISION = "leaf"
EXPAND_DECISION = "expand"
DECISION_LABELS = [LEAF_DECISION, EXPAND_DECISION]


@dataclass(frozen=True)
class HierarchicalTrainingExample:
    sample_id: str
    unit_index: int
    example_index: int
    parent_unit_index: int | None
    words: tuple[str, ...]
    input_token_ids: tuple[int, ...]
    view_kind: str
    upos_tags: tuple[str, ...]
    unit_pos: str
    aux_kind: str
    decision: str
    role_labels_by_token: tuple[tuple[int, tuple[str, ...]], ...]


@dataclass(frozen=True)
class HierarchicalVocab:
    word2id: dict[str, int]
    upos2id: dict[str, int]
    view_kind2id: dict[str, int]
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

    @property
    def upos_pad_id(self) -> int:
        return self.upos2id[PAD_TOKEN]

    @property
    def upos_unk_id(self) -> int:
        return self.upos2id[UNK_TOKEN]

    def to_dict(self) -> dict[str, Any]:
        return {
            "word2id": self.word2id,
            "upos2id": self.upos2id,
            "view_kind2id": self.view_kind2id,
            "unit_pos2id": self.unit_pos2id,
            "aux_kind2id": self.aux_kind2id,
            "role_label2id": self.role_label2id,
            "decision2id": self.decision2id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HierarchicalVocab":
        return cls(
            word2id={str(key): int(value) for key, value in data["word2id"].items()},
            upos2id={
                str(key): int(value)
                for key, value in data.get(
                    "upos2id",
                    {PAD_TOKEN: 0, UNK_TOKEN: 1},
                ).items()
            },
            view_kind2id={
                str(key): int(value)
                for key, value in data.get(
                    "view_kind2id",
                    {DEFAULT_VIEW: 0, CONTENT_VIEW: 1},
                ).items()
            },
            unit_pos2id={str(key): int(value) for key, value in data["unit_pos2id"].items()},
            aux_kind2id={str(key): int(value) for key, value in data["aux_kind2id"].items()},
            role_label2id={
                str(key): int(value)
                for key, value in data["role_label2id"].items()
            },
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
        upos_ids = [
            self.vocab.upos2id.get(_normalize_upos(tag), self.vocab.upos_unk_id)
            for tag in example.upos_tags
        ]
        unit_mask = [0] * len(example.words)
        role_target_ids = [self.vocab.role_label2id[O_ROLE_LABEL]] * len(example.words)

        for token_id in example.input_token_ids:
            if 0 <= token_id < len(unit_mask):
                unit_mask[token_id] = 1

        for token_id, labels in example.role_labels_by_token:
            if token_id < 0 or token_id >= len(role_target_ids):
                continue
            label = _single_role_label(labels)
            role_target_ids[token_id] = self.vocab.role_label2id.get(
                label,
                self.vocab.role_label2id[O_ROLE_LABEL],
            )

        active_unit_pos = self.vocab.unit_pos2id.get(
            example.unit_pos,
            self.vocab.unit_pos2id[UNK_TOKEN],
        )
        head_pos = active_unit_pos

        return {
            "sample_id": example.sample_id,
            "unit_index": example.unit_index,
            "example_index": example.example_index,
            "parent_unit_index": example.parent_unit_index,
            "words": [_tokenizer_word(word) for word in example.words],
            "word_ids": word_ids,
            "upos_ids": upos_ids,
            "active_unit_mask": unit_mask,
            "role_target_ids": role_target_ids,
            "active_unit_pos": active_unit_pos,
            "head_pos": head_pos,
            "head_aux_kind": self.vocab.aux_kind2id.get(
                example.aux_kind,
                self.vocab.aux_kind2id[UNK_TOKEN],
            ),
            "view_kind": self.vocab.view_kind2id.get(example.view_kind, self.vocab.view_kind2id[DEFAULT_VIEW]),
            "decision": self.vocab.decision2id[example.decision],
            "length": len(word_ids),
        }


def build_hierarchical_examples(
    data_path: Path,
    max_samples: int | None = None,
    max_units: int | None = None,
    ud_upos_by_line: dict[int, tuple[str, ...]] | None = None,
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
            sample_examples = build_hierarchical_examples_from_oracle(
                sample_id,
                oracle,
                upos_tags=(ud_upos_by_line or {}).get(line_number),
            )
            for example in sample_examples:
                examples.append(example)
                if max_units is not None and len(examples) >= max_units:
                    return examples

    return examples


def build_hierarchical_examples_from_oracle(
    sample_id: str,
    oracle: dict[str, Any],
    upos_tags: tuple[str, ...] | None = None,
) -> list[HierarchicalTrainingExample]:
    words = tuple(str(word) for word in oracle["words"])
    upos_tags = _coerce_upos_tags(upos_tags, len(words))
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
                example_index=len(examples),
                unit_index=unit_index,
                parent_unit_index=None,
                words=words,
                upos_tags=upos_tags,
                unit=unit,
            )
        )

    for parent_unit_index, parent_unit in enumerate(units):
        for child in parent_unit.get("children", []):
            child_node_index = int(child["node_index"])
            child_unit_index = node_to_unit_index.get(child_node_index)
            if child_unit_index is None:
                continue
            child_unit = units[child_unit_index]
            if not _should_enqueue_child_unit(parent_unit, child, child_unit):
                continue
            examples.append(
                _training_example_from_unit(
                    sample_id=sample_id,
                    example_index=len(examples),
                    unit_index=child_unit_index,
                    parent_unit_index=parent_unit_index,
                    words=words,
                    upos_tags=upos_tags,
                    unit=child_unit,
                )
            )

    return examples


def _training_example_from_unit(
    *,
    sample_id: str,
    example_index: int,
    unit_index: int,
    parent_unit_index: int | None,
    words: tuple[str, ...],
    upos_tags: tuple[str, ...],
    unit: dict[str, Any],
) -> HierarchicalTrainingExample:
    return HierarchicalTrainingExample(
        sample_id=sample_id,
        unit_index=unit_index,
        example_index=example_index,
        parent_unit_index=parent_unit_index,
        words=words,
        input_token_ids=tuple(int(token_id) for token_id in unit["input_token_ids"]),
        view_kind=_view_kind(str(unit.get("view_kind", DEFAULT_VIEW))),
        upos_tags=upos_tags,
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
    )


def _should_enqueue_child_unit(
    parent_unit: dict[str, Any],
    child: dict[str, Any],
    child_unit: dict[str, Any],
) -> bool:
    parent_token_ids = tuple(int(token_id) for token_id in parent_unit["input_token_ids"])
    child_token_ids = tuple(int(token_id) for token_id in child["token_ids"])
    if child_token_ids != parent_token_ids:
        return True

    parent_signature = (
        parent_token_ids,
        _view_kind(str(parent_unit.get("view_kind", DEFAULT_VIEW))),
        str(parent_unit["unit_pos"]),
    )
    child_signature = (
        tuple(int(token_id) for token_id in child_unit["input_token_ids"]),
        _view_kind(str(child_unit.get("view_kind", DEFAULT_VIEW))),
        str(child_unit["unit_pos"]),
    )
    if child_signature == parent_signature:
        return False
    return True


def _view_kind(view_kind: str) -> str:
    return DEFAULT_VIEW if view_kind == DEFAULT_VIEW else CONTENT_VIEW


def _single_role_label(labels: tuple[str, ...]) -> str:
    clean_labels = tuple(label for label in labels if label != O_ROLE_LABEL)
    if not clean_labels:
        return O_ROLE_LABEL
    if len(clean_labels) > 1:
        raise ValueError(f"expected one role-set label, found {clean_labels}")
    return clean_labels[0]


def build_vocab(
    examples: Iterable[HierarchicalTrainingExample],
    min_word_freq: int = 1,
) -> HierarchicalVocab:
    word_counter: Counter[str] = Counter()
    upos_tags = set()
    view_kinds = set()
    unit_pos = set()
    aux_kinds = set()
    role_labels = {O_ROLE_LABEL}

    for example in examples:
        word_counter.update(_normalize_word(word) for word in example.words)
        upos_tags.update(_normalize_upos(tag) for tag in example.upos_tags)
        view_kinds.add(_view_kind(example.view_kind))
        unit_pos.add(example.unit_pos)
        aux_kinds.add(example.aux_kind)
        for _, labels in example.role_labels_by_token:
            role_labels.add(_single_role_label(labels))

    word2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, count in sorted(word_counter.items()):
        if count >= min_word_freq and word not in word2id:
            word2id[word] = len(word2id)

    return HierarchicalVocab(
        word2id=word2id,
        upos2id={
            label: index
            for index, label in enumerate(
                [PAD_TOKEN, UNK_TOKEN, *sorted(upos_tags - {PAD_TOKEN, UNK_TOKEN})]
            )
        },
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


def collate_hierarchical_batch(
    batch: list[dict[str, Any]],
    pad_id: int,
    upos_pad_id: int = 0,
    tokenizer: Any | None = None,
    max_subword_length: int | None = 512,
) -> dict[str, Any]:
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)

    word_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    upos_ids = torch.full((batch_size, max_len), upos_pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    active_unit_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    role_target_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    active_unit_pos = torch.zeros(batch_size, dtype=torch.long)
    head_pos = torch.zeros(batch_size, dtype=torch.long)
    head_aux_kind = torch.zeros(batch_size, dtype=torch.long)
    view_kind = torch.zeros(batch_size, dtype=torch.long)
    decision = torch.zeros(batch_size, dtype=torch.long)

    for row_index, item in enumerate(batch):
        length = item["length"]
        word_ids[row_index, :length] = torch.tensor(item["word_ids"], dtype=torch.long)
        upos_ids[row_index, :length] = torch.tensor(item["upos_ids"], dtype=torch.long)
        attention_mask[row_index, :length] = True
        active_unit_mask[row_index, :length] = torch.tensor(
            item["active_unit_mask"],
            dtype=torch.bool,
        )
        role_target_ids[row_index, :length] = torch.tensor(
            item["role_target_ids"],
            dtype=torch.long,
        )
        active_unit_pos[row_index] = int(item["active_unit_pos"])
        head_pos[row_index] = int(item["head_pos"])
        head_aux_kind[row_index] = int(item["head_aux_kind"])
        view_kind[row_index] = int(item["view_kind"])
        decision[row_index] = int(item["decision"])

    output = {
        "sample_ids": [item["sample_id"] for item in batch],
        "unit_indices": [item["unit_index"] for item in batch],
        "example_indices": [item["example_index"] for item in batch],
        "parent_unit_indices": [item["parent_unit_index"] for item in batch],
        "word_ids": word_ids,
        "upos_ids": upos_ids,
        "attention_mask": attention_mask,
        "active_unit_mask": active_unit_mask,
        "role_target_ids": role_target_ids,
        "active_unit_pos": active_unit_pos,
        "head_pos": head_pos,
        "head_aux_kind": head_aux_kind,
        "view_kind": view_kind,
        "decision": decision,
    }
    if tokenizer is not None:
        output.update(
            _collate_bert_inputs(
                batch=batch,
                tokenizer=tokenizer,
                active_unit_mask=active_unit_mask,
                upos_ids=upos_ids,
                max_word_length=max_len,
                max_subword_length=max_subword_length,
            )
        )
    return output


def _normalize_word(word: str) -> str:
    normalized = word.strip().lower()
    return normalized if normalized else word.lower()


def _normalize_upos(tag: str) -> str:
    stripped = str(tag).strip()
    if stripped in {PAD_TOKEN, UNK_TOKEN}:
        return stripped
    normalized = stripped.upper()
    return normalized if normalized else UNK_TOKEN


def _coerce_upos_tags(
    upos_tags: tuple[str, ...] | None,
    word_count: int,
) -> tuple[str, ...]:
    if upos_tags is None or len(upos_tags) != word_count:
        return tuple(UNK_TOKEN for _ in range(word_count))
    return tuple(_normalize_upos(tag) for tag in upos_tags)


def _tokenizer_word(word: str) -> str:
    stripped = str(word).strip()
    return stripped if stripped else str(word)


def _collate_bert_inputs(
    *,
    batch: list[dict[str, Any]],
    tokenizer: Any,
    active_unit_mask: torch.Tensor,
    upos_ids: torch.Tensor,
    max_word_length: int,
    max_subword_length: int | None,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        [item["words"] for item in batch],
        is_split_into_words=True,
        padding=True,
        truncation=max_subword_length is not None,
        max_length=max_subword_length,
        return_tensors="pt",
    )
    if not hasattr(encoded, "word_ids"):
        raise TypeError("hierarchical BERT encoder requires a fast tokenizer with word_ids()")

    input_ids = encoded["input_ids"]
    bert_attention_mask = encoded["attention_mask"].bool()
    token_type_ids = torch.zeros_like(input_ids)
    bert_upos_ids = torch.zeros_like(input_ids)
    word_starts = torch.full(
        (len(batch), max_word_length),
        -1,
        dtype=torch.long,
    )

    for row_index, item in enumerate(batch):
        seen_word_ids: set[int] = set()
        word_ids = encoded.word_ids(batch_index=row_index)
        for piece_index, word_index in enumerate(word_ids):
            if word_index is None:
                continue
            bert_upos_ids[row_index, piece_index] = upos_ids[row_index, word_index]
            if bool(active_unit_mask[row_index, word_index]):
                token_type_ids[row_index, piece_index] = 1
            if word_index not in seen_word_ids:
                word_starts[row_index, word_index] = piece_index
                seen_word_ids.add(word_index)

        missing = [
            word_index
            for word_index in range(int(item["length"]))
            if int(word_starts[row_index, word_index]) < 0
        ]
        if missing:
            raise ValueError(
                "BERT tokenization truncated Lingua tokens for "
                f"{item['sample_id']} unit={item['unit_index']} missing={missing[:8]}; "
                "increase --max_subword_length or pre-filter long samples"
            )

    return {
        "bert_input_ids": input_ids,
        "bert_attention_mask": bert_attention_mask,
        "bert_token_type_ids": token_type_ids,
        "bert_upos_ids": bert_upos_ids,
        "bert_word_starts": word_starts,
    }


def load_ud_upos_by_line(path: Path) -> dict[int, tuple[str, ...]]:
    """Load Stanza UD cache rows and project word-level UPOS to Lingua tokens."""
    upos_by_line: dict[int, tuple[str, ...]] = {}
    if not path.exists():
        return upos_by_line

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            line_number = row.get("line")
            if line_number is None or row.get("error"):
                continue
            words = tuple(str(word) for word in row.get("words") or ())
            upos_by_line[int(line_number)] = align_ud_upos_to_lingua_words(
                words,
                row.get("sentences") or [],
            )
    return upos_by_line


def align_ud_upos_to_lingua_words(
    lingua_words: tuple[str, ...],
    ud_sentences: list[dict[str, Any]],
) -> tuple[str, ...]:
    """Assign each Lingua token the UPOS of the overlapping UD word span."""
    ud_words = [
        word
        for sentence in ud_sentences
        for word in sentence.get("words", [])
        if word.get("start_char") is not None and word.get("end_char") is not None
    ]
    if not ud_words:
        return tuple(UNK_TOKEN for _ in lingua_words)

    projected = []
    for token_start, token_end in _lingua_word_content_spans(lingua_words):
        best_upos = UNK_TOKEN
        best_overlap = 0
        for word in ud_words:
            ud_start = int(word["start_char"])
            ud_end = int(word["end_char"])
            overlap = max(0, min(token_end, ud_end) - max(token_start, ud_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_upos = str(word.get("upos") or UNK_TOKEN)
        projected.append(_normalize_upos(best_upos) if best_overlap > 0 else UNK_TOKEN)
    return tuple(projected)


def _lingua_word_content_spans(words: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    spans = []
    cursor = 0
    for word in words:
        raw = str(word)
        start = cursor
        end = cursor + len(raw)
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        content_start = start + leading
        content_end = end - trailing
        if content_start >= content_end:
            content_start = start
            content_end = end
        spans.append((content_start, content_end))
        cursor = end
    return tuple(spans)
