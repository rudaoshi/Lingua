from functools import partial
from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from lingua.learn.hierarchical import CONTENT_VIEW, DEFAULT_VIEW
from lingua.learn.hierarchical.dataset import (
    HierarchicalOracleDataset,
    align_ud_upos_to_lingua_words,
    build_hierarchical_examples,
    build_vocab,
    collate_hierarchical_batch,
)
from lingua.learn.hierarchical.model import HierarchicalTokenSetParser
from lingua.learn.hierarchical.training import (
    compute_hierarchical_loss,
    evaluate_hierarchical_model,
    train_one_epoch,
)


def test_builds_training_examples_from_lingua_jsonl():
    examples = build_hierarchical_examples(Path("data/dev.jsonl"), max_samples=1)
    vocab = build_vocab(examples)
    dataset = HierarchicalOracleDataset(examples, vocab)

    assert examples
    assert "FactualPredicator" in vocab.unit_pos2id
    assert vocab.view_kind2id[DEFAULT_VIEW] == 0
    assert CONTENT_VIEW in vocab.view_kind2id
    assert "<non_aux>" in vocab.aux_kind2id
    assert "O" in vocab.role_label2id
    assert "B-HEAD" in vocab.role_label2id
    assert vocab.upos2id["<unk>"] == 1
    assert dataset[0]["word_ids"]
    assert len(dataset[0]["upos_ids"]) == len(dataset[0]["word_ids"])
    assert sum(dataset[0]["active_unit_mask"]) > 0
    assert len(dataset[0]["role_target_ids"]) == len(dataset[0]["word_ids"])
    assert isinstance(dataset[0]["active_unit_pos"], int)
    assert isinstance(dataset[0]["head_pos"], int)
    assert isinstance(dataset[0]["head_aux_kind"], int)


def test_model_trains_and_evaluates_one_batch():
    examples = build_hierarchical_examples(Path("data/dev.jsonl"), max_samples=2, max_units=16)
    vocab = build_vocab(examples)
    dataset = HierarchicalOracleDataset(examples, vocab)
    loader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=partial(collate_hierarchical_batch, pad_id=vocab.pad_id),
    )
    model = HierarchicalTokenSetParser(
        vocab_size=len(vocab.word2id),
        num_unit_pos=len(vocab.unit_pos2id),
        num_aux_kinds=len(vocab.aux_kind2id),
        num_role_labels=len(vocab.role_label2id),
        embedding_dim=16,
        hidden_size=16,
        pad_id=vocab.pad_id,
        num_view_kinds=len(vocab.view_kind2id),
        role_label_names=[
            label
            for label, _index in sorted(vocab.role_label2id.items(), key=lambda item: item[1])
        ],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = next(iter(loader))
    output = model(
        batch["word_ids"],
        batch["attention_mask"],
        batch["active_unit_mask"],
        batch["view_kind"],
        batch["active_unit_pos"],
        batch["role_target_ids"],
    )
    losses = compute_hierarchical_loss(output, batch)

    assert model.role_crf is not None
    assert "role_crf_nll" in output
    assert model.decode_role_labels(
        output["role_logits"],
        batch["active_unit_mask"],
    ).shape == batch["role_target_ids"].shape
    assert losses["loss"].item() > 0
    train_metrics = train_one_epoch(model, loader, optimizer, torch.device("cpu"))
    eval_metrics = evaluate_hierarchical_model(model, loader, torch.device("cpu"))

    assert train_metrics["loss"] > 0
    assert 0.0 <= eval_metrics["role_f1"] <= 1.0
    assert eval_metrics["num_units"] == len(dataset)


def test_aligns_ud_upos_to_lingua_token_splits():
    upos = align_ud_upos_to_lingua_words(
        ("wo", " n", "'", "t"),
        [
            {
                "words": [
                    {"text": "wo", "upos": "AUX", "start_char": 0, "end_char": 2},
                    {"text": "n't", "upos": "PART", "start_char": 3, "end_char": 6},
                ]
            }
        ],
    )

    assert upos == ("AUX", "PART", "PART", "PART")


def test_structured_decode_uses_role_type_compatibility():
    unit_pos_names = ["<unk>", "FactualPredicator", "ModificationalFunctor"]
    aux_kind_names = ["<unk>", "<non_aux>", "Modification"]
    decision_names = ["leaf", "expand"]
    role_label_names = ["O", "B-HEAD", "B-ARG1", "B-ARG2", "B-VARIABLE", "B-BODY"]
    model = HierarchicalTokenSetParser(
        vocab_size=8,
        num_unit_pos=len(unit_pos_names),
        num_aux_kinds=len(aux_kind_names),
        num_role_labels=len(role_label_names),
        embedding_dim=8,
        hidden_size=8,
        num_decisions=len(decision_names),
        role_label_names=role_label_names,
        unit_pos_names=unit_pos_names,
        aux_kind_names=aux_kind_names,
        decision_names=decision_names,
    )
    output = {
        "head_pos_logits": torch.tensor([[0.0, 1.0, 3.0]]),
        "head_aux_kind_logits": torch.tensor([[0.0, 0.5, 3.0]]),
        "decision_logits": torch.tensor([[0.0, 3.0]]),
        "role_logits": torch.full((1, 3, len(role_label_names)), -5.0),
    }
    output["role_logits"][0, 0, role_label_names.index("B-ARG1")] = 5.0
    output["role_logits"][0, 1, role_label_names.index("B-HEAD")] = 5.0
    output["role_logits"][0, 2, role_label_names.index("B-ARG2")] = 5.0
    decoded = model.decode_structured(
        output,
        torch.tensor([[True, True, True]]),
        strategy="two_pass",
    )

    assert decoded["head_pos"].tolist() == [unit_pos_names.index("FactualPredicator")]
    assert decoded["head_aux_kind"].tolist() == [aux_kind_names.index("<non_aux>")]
    assert decoded["decision"].tolist() == [decision_names.index("expand")]
    assert decoded["role_labels"].tolist()[0] == [
        role_label_names.index("B-ARG1"),
        role_label_names.index("B-HEAD"),
        role_label_names.index("B-ARG2"),
    ]


def test_bert_encoder_path_uses_wordpiece_alignment(monkeypatch):
    class FakeBatchEncoding(dict):
        def __init__(self, payload, word_id_rows):
            super().__init__(payload)
            self._word_id_rows = word_id_rows

        def word_ids(self, batch_index: int):
            return self._word_id_rows[batch_index]

    class FakeTokenizer:
        is_fast = True

        def __call__(
            self,
            words_by_row,
            *,
            is_split_into_words,
            padding,
            truncation,
            max_length,
            return_tensors,
        ):
            assert is_split_into_words
            max_pieces = max(len(words) + 2 for words in words_by_row)
            input_ids = torch.zeros((len(words_by_row), max_pieces), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            word_id_rows = []
            for row_index, words in enumerate(words_by_row):
                word_ids = [None, *range(len(words)), None]
                word_id_rows.append(word_ids + [None] * (max_pieces - len(word_ids)))
                attention_mask[row_index, : len(word_ids)] = 1
                input_ids[row_index] = torch.arange(max_pieces)
            return FakeBatchEncoding(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                word_id_rows,
            )

    class FakeBertModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.embedding = nn.Embedding(64, 8)
            self.saw_inputs_embeds = False

        def get_input_embeddings(self):
            return self.embedding

        def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            token_type_ids=None,
        ):
            assert attention_mask is not None
            assert input_ids is None
            assert inputs_embeds is not None
            self.saw_inputs_embeds = True
            encoded = inputs_embeds
            if token_type_ids is not None:
                encoded = encoded + token_type_ids.unsqueeze(-1).float()
            return SimpleNamespace(last_hidden_state=encoded)

    import transformers

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: FakeBertModel(),
    )
    examples = build_hierarchical_examples(Path("data/dev.jsonl"), max_samples=1, max_units=8)
    vocab = build_vocab(examples)
    dataset = HierarchicalOracleDataset(examples, vocab)
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=partial(
            collate_hierarchical_batch,
            pad_id=vocab.pad_id,
            tokenizer=FakeTokenizer(),
        ),
    )
    model = HierarchicalTokenSetParser(
        vocab_size=len(vocab.word2id),
        num_unit_pos=len(vocab.unit_pos2id),
        num_aux_kinds=len(vocab.aux_kind2id),
        num_role_labels=len(vocab.role_label2id),
        pad_id=vocab.pad_id,
        num_view_kinds=len(vocab.view_kind2id),
        role_label_names=[
            label
            for label, _index in sorted(vocab.role_label2id.items(), key=lambda item: item[1])
        ],
        encoder_type="bert",
        bert_model_name="fake-bert",
    )
    batch = next(iter(loader))
    output = model(
        batch["word_ids"],
        batch["attention_mask"],
        batch["active_unit_mask"],
        batch["view_kind"],
        batch["active_unit_pos"],
        batch["role_target_ids"],
        batch["bert_input_ids"],
        batch["bert_attention_mask"],
        batch["bert_token_type_ids"],
        batch["bert_word_starts"],
        upos_ids=batch["upos_ids"],
        bert_upos_ids=batch["bert_upos_ids"],
    )
    losses = compute_hierarchical_loss(output, batch)

    assert batch["bert_word_starts"].shape == batch["word_ids"].shape
    assert batch["bert_upos_ids"].shape == batch["bert_input_ids"].shape
    assert model.bert.saw_inputs_embeds
    assert output["role_logits"].shape[:2] == batch["word_ids"].shape
    assert losses["loss"].item() > 0


def test_role_set_bio_labels_include_shared_span_composites():
    examples = build_hierarchical_examples(Path("data/train.jsonl"), max_samples=1564)
    assert any(
        labels == ("B-ARG2+ARG3",)
        for example in examples
        if example.sample_id == "train.jsonl:1564"
        for _, labels in example.role_labels_by_token
    )
