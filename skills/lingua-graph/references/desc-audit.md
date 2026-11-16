# SELF/DESC audit and repair

## Core Standard

Treat Lingua's basic semantic unit as `SELF(v)` plus `DESC(v)`, not as an isolated token, node label, or dependency edge.

- `SELF(v)`: the node's own tokens, type, or symbol label.
- `DESC(v)`: the complete directed subgraph reachable from `v`, including child structure, edge labels, and descendant token coverage.
- When an edge points to node `v`, judge whether `DESC(v)` can recover the entity, fact, modifier, boundary, reference, expression, or logical unit required by that edge.
- Do not reject a target just because `SELF(v)` is a wrapper such as punctuation, determiner, modifier, `Modification`, `Expression`, `Ref`, or an aux symbol. If `DESC(v)` clearly recovers the intended local analytic unit, it can be a valid compact Lingua structure.
- Do not accept arbitrary nesting merely because the head tokens are reachable. Modifier order, semantic closeness, scope, boundary, and role assignment must still match the sentence.

Follow the standard selected by [SKILL.md](../SKILL.md#required-references). Start with the sections around `核心图语义`, `复杂论元`, `修饰层级`, `Ref`, `Apposition`, `标点`, `as:punctuation`, and the final checklist.

## Audit Workflow

1. Pin the exact artifact first.
   - Common corpus files: `data/train.jsonl`, `data/dev.jsonl`, `data/test.jsonl`
   - If the user gives a split/index, inspect that exact record and preserve the split/index in all notes.
2. Reconstruct the whole sentence graph before deciding.
   - List sentence tokens.
   - For each reviewed node, compute own tokens/type (`SELF`) and reachable descendant nodes/tokens/edges (`DESC`).
   - Include incoming edges into the reviewed node, because those edges define what local meaning `DESC(v)` must provide.
3. Ask the role question for each incoming edge.
   - `pred.arg.*`: Does `DESC(arg)` recover the complete participant, proposition, attribute value, controller, or complement required by the predicate role?
   - `variable`: Does `DESC(variable)` recover the head/event/entity being modified, including inner modifiers already in scope?
   - `body`: Does `DESC(body)` recover the modifier, content, punctuation-closed item, or relation body, with its internal arguments?
   - `func.arg`: Does `DESC(child)` recover a complete list/expression/conjunction/function argument rather than a stray leaf?
   - `as:punctuation`: Does the punctuation node close the precise local target, so `DESC(p)` is the punctuation-closed unit?
   - `as:modification`, `as:discourse`, `as:ref`, and related compressed edges: Does the compressed target still recover the intended aux relation and its scope?
4. Compare `DESC(v)` to the sentence meaning.
   - It should include the local head/relation, required core arguments, internal modifiers, boundaries, and reference links needed to understand that sentence segment.
   - It should exclude unrelated siblings, outer-scope material, and later modifiers that are not part of the local unit.
   - It should preserve complex structures such as relative clauses, attributes, appositions, parentheticals, lists, expressions, coordination, and shared arguments instead of flattening them into opaque phrase nodes.
5. Decide one case at a time when the user asks for audit/review.
   - Do not collapse the task into heuristic screening.
   - For each case, state the verdict, evidence from `SELF`/`DESC`, and the exact reason tied to the sentence.

## Verdicts

Use these labels consistently:

- `correct`: `DESC(v)` recovers the required local meaning and scope.
- `correct_variant`: `SELF(v)` looks indirect or wrapper-like, but `DESC(v)` cleanly recovers the required unit.
- `minor`: the structure is recoverable but stylistically noisy, redundant, or slightly noncanonical without changing local meaning.
- `major`: `DESC(v)` fails to recover the required head, argument, internal relation, boundary, or modifier scope; or it includes wrong-scope material that changes the local meaning.
- `blocker`: the graph is invalid or the reviewed subgraph destroys core sentence meaning, such as cycles, dangling nodes, missing main fact, or unrecoverable core arguments across the sample.
- `needs_human_decision`: the sentence is ambiguous, source text is noisy, or multiple Lingua-consistent interpretations remain after full graph inspection.

Prefer `major` for recoverability/scope failures. Reserve `minor` for differences that a downstream analytic unit could still use without semantic loss.

## What Counts As Complete Local Meaning

A descendant graph is complete enough when a reader can recover the sentence-local analytic unit from that subgraph alone plus the edge label that selected it.

Check these requirements:

- Head: the semantic head, predicate, relation, expression operator, or entity root is present.
- Roles: required arguments are present or responsibly recoverable through control, sharing, passive/raising, relative reference, wrapper context, or another explicit Lingua edge.
- Internal structure: determiners, attributes, nested entities, relative clauses, appositions, parentheticals, lists, and expressions are represented when they affect meaning.
- Scope: close modifiers are inside before wider modifiers; outer-scope material is not smuggled into an inner child.
- Boundary: punctuation wrappers close the local item they actually delimit, not a larger or smaller structure by accident.
- Compactness: wrappers and `as:*` edges are acceptable only when they improve compactness without hiding required structure.
- Non-redundancy: the same relation should not be represented twice by both a wrapper/aux node and a direct edge unless the standard explicitly allows the two views.

## Common Failure Patterns

- A `pred.arg.*`, `variable`, `body`, or `func.arg` edge points to a leaf whose `DESC` cannot recover the full entity/fact needed by the role.
- A modifier chain is recoverable in token coverage but has near/far scope reversed.
- A punctuation node is treated as a generic child instead of a local close wrapper, so `DESC(p)` cannot recover the closed item.
- A punctuation wrapper closes too much or too little, especially in quotes, parentheses, colons, dashes, sentence-final punctuation, and question marks.
- A complex noun phrase, title, relative clause, apposition, attribute, or parenthetical is flattened into one opaque lexical node, preventing downstream recovery of internal structure.
- A `Modification` wrapper is added around a relation that is already self-contained, creating duplicate recoverability targets.
- A `Ref`/relative construction points to an internal wh-word or leaf where the external relation needs the whole restricted referent, or vice versa.
- Shared descendants are simplified away; the graph becomes cleaner-looking but can no longer recover overlapping roles or shared arguments.
- A local repair changes graph cardinality, roots, or schema counts without a clear semantic reason.

## Repair Discipline

For review-only requests, report findings and candidate repairs. A request to correct a graph authorizes producing the corrected copy and its evidence; complete that work without another confirmation. Apply repairs to source files or a corpus when that scope is requested, preserving unrelated changes. Keep before/after artifacts in either case.

1. Prefer a focused repair script for a narrow family of cases.
2. Run dry-run first and inspect the full `AFTER` graph for representative and risky cases.
3. Apply only after the intended `AFTER` shape is clear.
4. Re-run the repair to confirm idempotency or an equivalent `changed samples: 0` result.
5. Validate the corrected samples with the local validator. Run corpus-wide checks or schema tests when the change affects that scope, not for every isolated review:

```bash
python3 /absolute/path/to/lingua-graph/scripts/validate_graph.py /absolute/corrected.json --max-errors 20
```

Do not revert or rewrite unrelated dirty worktree changes. If the repo is already dirty, report only the files touched by the current repair.

## Report Shape

For a focused subgraph request, return JSON records with the following fields (or prose when requested). State the reviewed scope; a targeted verdict does not certify the unreviewed remainder of the graph:

```text
split:index
sentence: ...
target: edge/node being audited
SELF(v): type + own text
DESC(v): descendant text + key child edges
verdict: correct | correct_variant | minor | major | blocker | needs_human_decision
reason: one concise semantic reason
repair_hint: only if a fix is requested or obvious
```

For batches, save a JSONL/TSV artifact when useful, but summarize counts by verdict and include representative examples. Always keep enough evidence for the user to audit decisions one by one.

When used within an existing-annotation or dataset review, include these local findings in the enclosing report's errors and notes. In online Mode B, put the same judgments in the existing per-node/per-edge review records; do not create a competing review or acceptance result. Preserve `needs_human_decision` for real ambiguity, and do not force it into `valid` merely to fit a legacy schema.
