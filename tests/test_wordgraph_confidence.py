import json

from lingua.learn.wordgraph.confidence.merge import merge_fold_results
from lingua.learn.wordgraph.confidence.splits import make_kfold_indices, prepare_nfold_splits


def test_make_kfold_indices_covers_each_sample_once():
    folds = make_kfold_indices(10, 3, random_seed=7)

    flattened = [idx for fold in folds for idx in fold]
    assert sorted(flattened) == list(range(10))
    assert len(flattened) == len(set(flattened))
    assert [len(fold) for fold in folds] == [4, 3, 3]


def test_prepare_nfold_splits_writes_trainer_layout(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [{"sentence": f"s{i}", "lingua": {"words": [], "graph": {"nodes": [], "edges": []}}} for i in range(5)]
    dev_rows = [{"sentence": "dev", "lingua": {"words": [], "graph": {"nodes": [], "edges": []}}}]

    for split, split_rows in [("train", rows), ("dev", dev_rows)]:
        with (data_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row) + "\n")

    specs = prepare_nfold_splits(data_dir=data_dir, output_dir=tmp_path / "out", n_folds=2)

    assert len(specs) == 2
    assert sum(spec.test_size for spec in specs) == 5
    for spec in specs:
        assert spec.train_path.exists()
        assert spec.dev_path.exists()
        assert spec.test_path.exists()
        assert spec.manifest_path.exists()

    heldout = []
    for spec in specs:
        with spec.manifest_path.open(encoding="utf-8") as f:
            heldout.extend(json.loads(line)["global_idx"] for line in f if line.strip())
    assert sorted(heldout) == list(range(5))


def test_merge_fold_results_orders_by_global_idx(tmp_path):
    folds_dir = tmp_path / "folds"
    for fold, records in [
        (0, [{"global_idx": 2, "confidence": 0.2}, {"global_idx": 0, "confidence": 0.4}]),
        (1, [{"global_idx": 1, "confidence": 0.3}]),
    ]:
        path = folds_dir / f"fold_{fold}" / "confidence"
        path.mkdir(parents=True)
        with (path / "detailed_results.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    summary = merge_fold_results(
        folds_dir=folds_dir,
        output_jsonl=tmp_path / "confidence.jsonl",
        output_csv=tmp_path / "confidence.csv",
        summary_path=tmp_path / "summary.json",
        low_confidence_path=tmp_path / "low.jsonl",
        low_confidence_limit=2,
    )

    with (tmp_path / "confidence.jsonl").open(encoding="utf-8") as f:
        merged = [json.loads(line) for line in f if line.strip()]
    assert [row["global_idx"] for row in merged] == [0, 1, 2]
    assert summary["total_records"] == 3
    assert summary["scored_records"] == 3
