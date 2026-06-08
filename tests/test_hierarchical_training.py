from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lingua.learn.hierarchical.dataset import (
    HierarchicalOracleDataset,
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
    assert "<non_aux>" in vocab.aux_kind2id
    assert "B-HEAD" in vocab.role_label2id
    assert dataset[0]["word_ids"]
    assert sum(dataset[0]["unit_mask"]) > 0


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
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = next(iter(loader))
    output = model(batch["word_ids"], batch["attention_mask"], batch["unit_mask"])
    losses = compute_hierarchical_loss(output, batch)

    assert losses["loss"].item() > 0
    train_metrics = train_one_epoch(model, loader, optimizer, torch.device("cpu"))
    eval_metrics = evaluate_hierarchical_model(model, loader, torch.device("cpu"))

    assert train_metrics["loss"] > 0
    assert 0.0 <= eval_metrics["role_f1"] <= 1.0
    assert eval_metrics["num_units"] == len(dataset)
