# Dataset

The released Lingua-Graph dataset is stored as JSONL files:

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`

Each example has the following top-level fields:

- `sentence`: original input sentence
- `lingua`: serialized Lingua graph object

At a high level, the serialized graph contains:

- `meta`: graph metadata
- `words`: tokenized sentence with preserved spacing prefixes
- `graph.nodes`: node spans and node types
- `graph.edges`: directed labeled edges between nodes

Current split sizes in this repo:

- train: 11,129
- dev: 1,000
- test: 1,000

The repaired train split removes 15 source-unrecoverable examples whose text
was truncated, garbled, or semantically incomplete.

The main training script expects `--data_dir` to point to this directory.
