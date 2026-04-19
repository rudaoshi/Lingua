# Lingua

Lingua is the public research repository for the Lingua-Graph dataset, parser, and demo assets used in the paper.

## What is in this repo

- `src/lingua/`: installable Python package with graph structures, word-graph modeling code, decoders, and inference utilities
- `data/`: released `train/dev/test` Lingua-Graph splits
- `scripts/`: training and evaluation entrypoints for the public release layout
- `demos/`: Gradio demo app for local use or Hugging Face Spaces
- `tests/`: lightweight regression tests for decoder components
- `research/`: older research scripts and prototypes kept for reference, but not part of the main supported package

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Run a lightweight import check:

```bash
PYTHONPATH=src python -c "from lingua.learn.wordgraph.decoding.inference import WordLinguaInference"
```

## Training

The main public training entrypoint is:

```bash
python scripts/train_wordgraph.py \
  --data_dir data \
  --output_dir artifacts/wordgraph-baseline
```

This script trains the parser and exports a Hugging Face style model directory containing:

- `pytorch_model.bin`
- `config.json`
- tokenizer files
- `label2id.json`
- `test_results.json`

## Evaluation

```bash
python scripts/eval_wordgraph.py \
  --model_dir artifacts/wordgraph-baseline \
  --data_path data/test.jsonl \
  --output_path artifacts/wordgraph-baseline/eval.json
```

## Demo

```bash
python demos/gradio_app.py
```

By default the demo loads the model published on Hugging Face. You can also point it to a local exported model directory by changing `MODEL_ID` in the script or setting it before launch.

## Notes

- The main supported code path is the package under `src/lingua`.
- `research/` contains older exploratory assets preserved from the paper repo reorganization.
- Large model weights should live on Hugging Face rather than in Git history.
