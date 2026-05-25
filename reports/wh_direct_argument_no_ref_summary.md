# wh direct argument without Ref scan

Scope: `/Users/sun/Projects/oix/lingua-graph-paper/data/{train,dev,test}.jsonl`.

Definition used: wh-token node has an incoming `pred.arg.*` or `func.arg` edge, and the wh node is not covered by a direct `ref/as:ref`, an incident `Ref/ReferentialPredicator` node, or a `Ref.pred.arg.2` descendant structure.

## Counts

- Direct wh-argument nodes without Ref: 498
- Unique examples: 471
- By split: {'train': 414, 'dev': 50, 'test': 34}
- By wh word: {'when': 86, 'who': 33, 'what': 233, 'where': 37, 'how': 54, 'whatever': 7, 'which': 10, 'whenever': 3, 'why': 29, 'however': 3, 'whose': 1, 'whom': 1, 'whomever': 1}
- By incoming argument label: {'pred.arg.2': 305, 'pred.arg.1': 106, 'func.arg': 73, 'pred.arg.3': 17, 'pred.arg.4': 5}
- By wh node type: {'LogicalPredicator': 82, 'GeneralFunctor': 389, 'NominalConstant': 19, 'ModificationalConstant': 7, 'DeterminerConstant': 1}

## High-risk headed-relative subset

- Likely true missing-Ref nodes: 28
- By wh word: {'where': 13, 'when': 6, 'which': 4, 'whose': 1, 'who': 1, 'why': 3}

Most of the full set are embedded questions, free relatives, temporal adjunct clauses, or discourse fragments where no external antecedent exists, so a Ref node is not automatically required. The high-risk subset contains cases with an overt head such as location/time/reason/person/entity plus an explicit wh relative marker.
