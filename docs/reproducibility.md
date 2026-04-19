# Reproducibility

## Supported pipeline

1. Train with `scripts/train_wordgraph.py`
2. Export a Hugging Face style model bundle into `artifacts/...`
3. Evaluate with `scripts/eval_wordgraph.py`
4. Serve a local demo with `demos/gradio_app.py`

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

