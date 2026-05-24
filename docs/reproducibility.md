# Reproducibility

## Supported pipeline

1. Train with `scripts/train_wordgraph.py`
2. Export a Hugging Face style model bundle into `artifacts/...`
3. Evaluate with `scripts/eval_wordgraph.py`
4. Serve a local demo with `demos/gradio_app.py`

## Sample confidence

Held-out sample confidence is produced with the same wordgraph parser training
and evaluation path:

```bash
python scripts/run_nfold_confidence.py \
  --data_dir data \
  --source_split train \
  --output_dir artifacts/wordgraph-nfold-confidence \
  --n_folds 3
```

The runner creates fold-specific `train.jsonl`, `dev.jsonl`, and `test.jsonl`
files, trains each fold with `scripts/train_wordgraph.py`, evaluates the
held-out fold with `scripts/eval_wordgraph.py`, and then computes per-sample
confidence from the held-out model's supervised gold-graph NLL.

## Repo conventions

- Supported package code lives under `src/lingua`
- Public data lives under `data`
- Legacy and exploratory code lives under `research`

## Release checklist

- Confirm dataset documentation and license text
- Confirm model card and Hugging Face repo metadata
- Export a clean model directory from the training script
- Run `pytest` on the public test suite
- Verify `scripts/eval_wordgraph.py` against the released test split
