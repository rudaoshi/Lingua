---
name: lingua-parse-validator
description: Validate linguistic parsing annotations and parse sentences according to the Lingua graph standard. Use when Codex needs to check annotation quality, validate train/dev/test JSONL samples, judge whether a Lingua parse is correct, produce a Lingua-style graph, explain parsing errors, compare data against the standard, propose corrected Lingua JSONL samples, or batch-review predicate arguments, modifier scope, references, punctuation roots, and legal node or edge labels.
---

# Lingua Parse Validator

## Purpose

Use this skill for two tasks:

1. Validate an existing Lingua annotation against the standard.
2. Parse a raw sentence into a Lingua-compliant graph and validate the result.

Keep this file as the workflow. Load detailed rules from `references/` only as needed.

## Required References

Before final judgments, consult the relevant reference files:

- `references/lingua-standard.md`: complete Lingua standard.
- `references/annotation-rules.md`: executable mandatory checks and rule IDs.
- `references/error-taxonomy.md`: error categories and severity guidance.
- `references/examples.md`: positive and negative examples.
- `references/audit-protocol.md`: sampled and full-dataset audit protocol.
- `references/session-lessons.md`: calibrated pitfalls from Lingua standard/data alignment work.
- `references/output-schema.json`: expected JSON report shape.
- `references/audit-record-schema.json`: per-line audit artifact shape.

## Mode A: Validate Existing Annotation

Use this mode when input includes a sentence plus nodes, edges, labels, JSONL, or a dataset file.

Procedure:

1. Identify the sample format:
   - Repo JSONL: `sentence`, `lingua.words`, `lingua.graph.nodes`, `lingua.graph.edges`.
   - Generic graph: `sentence`, `tokens` or `words`, `nodes`, `edges`.
2. Run deterministic checks when files are available:
   - `scripts/validate_graph.py <sample.json|sample.jsonl>`
   - `scripts/batch_validate.py <dataset.jsonl>` for datasets.
3. Check structural validity:
   - required fields exist;
   - lexical spans are valid and cover tokens exactly once when tokens are present;
   - node types, auxiliary labels, and edge labels are legal;
   - edge endpoints exist;
   - graph is a single-root reachable DAG unless the standard explicitly allows a fragment.
4. Check linguistic validity:
   - first decide whether the input is a normal sentence, a document/structured fragment, a title, or a pure proper-name/nominal fragment;
   - do not require a main factual predicate for pure fragments that are correctly represented as `List`, `Expression`, `Attribute`, or entity/title structures;
   - predicate-argument structure is semantically correct;
   - every predicate has recoverable `pred.arg.1` unless the sentence truly lacks one;
   - `pred.arg.4` is not used for ordinary purpose, reason, time, manner, or condition adjuncts;
   - ordinary adjuncts introduced by `to` or a preposition should use that overt token as the anchored `ModificationalFunctor`; fixed complex prepositions such as `in order to`, `so as to`, `because of`, `due to`, `according to`, `in front of`, `by means of`, and `in addition to` are kept as one functor, with `variable` pointing to the core unit and `body` pointing to the introduced content;
   - modifiers attach to the smallest complete semantic unit they modify;
   - multi-layer modifiers follow semantic proximity and scope;
   - explicit wh-words, relative `that`, or referring expressions use `ref/as:ref` or `Ref`;
   - zero relative clauses do not invent `Ref`;
   - punctuation roots close the correct local structure;
   - expression-internal format separators split into recoverable parts and use the separator/operator as `ExpressionFunctor`; field/header/discourse/syntactic close punctuation remains punctuation.
5. Produce a JSON report. Do not silently repair; report original errors and include a minimal corrected annotation in `corrected_annotation` whenever the verdict is `minor_error`, `major_error`, or `invalid` and a repair is possible.
6. If the user requests changes, edit the data and provide before/after diffs or diagrams.

## Mode B: Parse Raw Sentence

Use this mode when only a raw sentence is provided.

Procedure:

1. Segment the sentence into token spans and linguistically meaningful units.
2. Identify predicates, arguments, modifiers, clauses, coordination, discourse markers, punctuation scope, references, omissions, and expressions.
3. Construct a Lingua graph with legal node and edge labels.
4. Validate the graph structurally, then linguistically.
5. Return JSON with both the parse and a concise explanation of key decisions.

## Mode C: Audit A Dataset

Use this mode when the user asks to audit many JSONL samples, especially `train.jsonl`.

Procedure:

1. Ask or infer the audit scope: full dataset, contiguous range, fixed random sample, or high-risk subset.
2. State the method explicitly:
   - `batch screening`: scripts and heuristics over many samples;
   - `full per-sample audit`: every sample is expanded and semantically judged;
   - `targeted review`: only flagged or user-selected samples are deeply reviewed.
3. Never describe batch screening as a full per-sample audit.
4. Create resumable artifacts for full audits: audit JSONL records, summary, proposed corrected samples, and optional visualizations.
5. Use `scripts/prepare_audit_batch.py <dataset.jsonl> --start N --end M` to generate per-line audit skeletons.
6. For each record, run structural checks first, then perform linguistic judgment, then write the verdict.
7. For every record with `review_status: "fix_proposed"`, include `proposed_corrected_sample`: a complete corrected JSONL sample with `sentence`, `lingua.words`, and `lingua.graph`, not just a patch or prose note.
8. Validate each `proposed_corrected_sample` structurally before reporting it as a proposed repair. If a repair depends on an unresolved policy choice, set `review_status: "needs_human_decision"` and explain the blocked choice.
9. Do not apply fixes in bulk until the user confirms the repair policy for that batch.

## Output Contract

Always output JSON unless the user explicitly asks for prose or diagrams.

Use `references/output-schema.json` as the authoritative shape. The top-level object should include:

```json
{
  "task": "validate",
  "sentence": "...",
  "verdict": "valid",
  "confidence": 0.9,
  "summary": "...",
  "errors": [],
  "corrected_annotation": null,
  "parse": null,
  "notes": []
}
```

Verdicts:

- `valid`: no meaningful issue found.
- `minor_error`: recoverable issue that does not change core semantics.
- `major_error`: core structure, argument, reference, or modifier scope issue.
- `invalid`: graph is unusable, disconnected, schema-invalid, or semantically misleading.
- `not_applicable`: input is not a Lingua validation or parsing task.

Every error must include:

- `error_type`
- `severity`
- `location`
- `violated_rule`
- `explanation`
- `suggested_fix`
- `confidence`

For non-valid validation results, `corrected_annotation` should be the complete corrected annotation object, not a diff. For repo JSONL samples, preserve the original `sentence` and `lingua.words`, and change only the minimal `lingua.graph.nodes` / `lingua.graph.edges` needed to satisfy the standard. Use `null` only when no safe correction can be proposed; explain that in `notes`.

## Scripts

Use scripts for deterministic checks:

- `scripts/validate_output.py <report.json>` validates the report JSON shape.
- `scripts/validate_graph.py <sample.json|sample.jsonl>` validates graph structure and schema.
- `scripts/batch_validate.py <dataset.jsonl>` validates many samples and emits JSON summary.
- `scripts/prepare_audit_batch.py <dataset.jsonl> --start N --end M` creates resumable audit skeleton records.
- `scripts/validate_audit_records.py <audit.jsonl>` validates per-line audit records, including corrected-sample requirements for proposed fixes.

Principle: programmatic checks catch form errors; the model still performs linguistic judgment.

## Important Gotchas

- Do not accept a graph merely because it is valid JSON.
- A structurally valid graph can still be linguistically wrong.
- Do not call high-risk screening a per-sample audit. Report how many records were deeply reviewed.
- Prefer exact node, edge, line, or token locations over vague comments.
- Do not invent labels outside the standard.
- Treat `nonsense` as legal but low-information.
- Treat low model confidence as a review-priority signal, not proof of annotation error.
- For header/table/field/UI fragments, evaluate fragment policy before applying ordinary sentence PFA expectations.
- For proper names, titles, and nominal fragments, do not invent copulas or factual predicates merely to create a sentence-like graph.
- Treat `but` as `LogicalPredicator`.
- Treat `if` as `LogicalPredicator` for conditional "if", and `GeneralFunctor` for whether-clauses.
- Treat `that` by function: nominal-clause marker as `GeneralFunctor`, demonstrative entity as `NominalConstant`, determiner as `DeterminerConstant`, relative marker as `GeneralFunctor`.
- Do not convert an object to `pred.arg.1` merely because a predicate lacks an explicit subject. If the subject is omitted and cannot be recovered from sentence-internal structure, leave `pred.arg.1` absent; current data must not use `Missing` nodes.
- Pure punctuation or symbol tokens must not be any `*Predicator`; use punctuation/symbol/expression structure, or lexical/auxiliary relation nodes for the relation they delimit.
- If punctuation closes a local component, make it the wrapper of that component with `as:punctuation`; the outer relation should point to that wrapper, not merely attach the punctuation as an outer `punctuation` child.
- Format is a functor: in dates, times, addresses, person/proper-name abbreviations, URL/email/file paths, codes, formulas, calculations, currency/percent expressions, and similar expressions, split recoverable parts and make the internal separator/operator an `ExpressionFunctor`. Keep field/header/discourse delimiters and ordinary syntactic close punctuation as punctuation close markers.
- Treat punctuation-root warnings as low-value by default in this dataset; inspect punctuation only for local close-scope errors.
- When uncertain, report competing analyses and confidence.
