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

- train: 11,915
- dev: 1,324
- test: 1,324

These counts include the reviewed V15 UD-EWT long-sentence import
`reviewed_v15_ud_ewt_outside_unmerged73_20260624` and the subsequent UD-EWT
split realignment. Current rows found unambiguously in UD-EWT are assigned to
the official UD split; rows not found in UD-EWT, or found in multiple UD splits
by sentence text alone, keep their previous split.

The repaired train split removes 15 source-unrecoverable examples whose text
was truncated, garbled, or semantically incomplete.

The main training script expects `--data_dir` to point to this directory.
