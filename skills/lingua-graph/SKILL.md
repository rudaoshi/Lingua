---
name: lingua-graph
description: Validate, audit and repair Lingua graphs through one workflow, including online Hugging Face parsing, existing annotations, datasets and focused SELF/DESC subgraph reviews. Check argument binding, recoverability, modifier and negation scope, references and schema compliance; return complete corrected graphs with evidence and unresolved issues.
---

# Lingua Graph

## Purpose

Use this single skill for all Lingua validation and repair tasks:

1. Validate an existing Lingua annotation against the standard.
2. Parse a raw sentence with the online Lingua service, review and repair it, and return the complete graph with evidence.
3. Audit and repair a selected node, edge or descendant subgraph for complete local meaning.
4. Screen or review a dataset with an explicit scope and per-sample evidence.

Keep this file as the workflow. Load detailed rules from `references/` only as needed.

## Required References

Use [the repository standard](../../docs/standard.md) when running from a Lingua checkout, including a symlink into the checkout. For a standalone copy of this folder, use the [bundled standard](references/lingua-standard.md), a snapshot of the same document. The selected standard takes precedence over examples. The online mode freezes its bytes and SHA-256 for the run. When updating the standard for a release, sync the bundled snapshot to the repository document.

Before final judgments, consult the relevant references:

- `references/annotation-rules.md`: executable mandatory checks and rule IDs.
- `references/error-taxonomy.md`: error categories and severity guidance.
- `references/examples.md`: positive and negative examples.
- `references/audit-protocol.md`: sampled and full-dataset audit protocol.
- `references/session-lessons.md`: calibrated pitfalls from Lingua standard/data alignment work.
- `references/output-schema.json`: JSON report shape for existing-annotation, offline, and dataset modes.
- `references/audit-record-schema.json`: per-line audit artifact shape.
- `references/online-workflow.md`: online parse, evidence preservation, review, repair and delivery; read for Mode B.
- `references/online-review-contract.md`: online review records and result states.
- `references/desc-audit.md`: focused SELF/DESC audit, incoming-role checks, scope, wrapper recoverability, severity and repair discipline; read for subgraph audits or difficult semantic repairs in any mode.

The scripts use Python 3.10+ and its standard library. The full workflow requires an agent that can read/write files and run those scripts. Mode B also needs HTTPS access to the public Hugging Face parser.

## Shared Semantic Review and Repair

All modes use `SELF(v)` plus the complete directed `DESC(v)`. An incoming edge determines the local unit its target must recover. Keep internal edges, argument roles, references and modifier scope in view; reachable tokens alone do not establish semantic correctness. A determiner, punctuation or auxiliary wrapper can be a valid target when its DESC recovers the required unit.

For a focused node/edge/descendant question, use `references/desc-audit.md` within this skill. Pin the exact record, inspect the whole sentence, review the requested local units and relevant incoming edges, then report the scope actually reviewed. Existing annotations do not need a fresh online call merely to perform this audit.

For requested repairs, produce a minimally changed complete graph and before/after evidence, recheck the entire affected graph, and verify idempotency. Preserve unresolved ambiguity and unrelated source edits. Programmatic structure checks and model semantic judgments remain separate.

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
   - apply the shared SELF/DESC review above; use `references/desc-audit.md` for local-unit, wrapper, scope or reference issues;
   - first decide whether the input is a normal sentence, a document/structured fragment, a title, or a pure proper-name/nominal fragment;
   - do not require a main factual predicate for pure fragments that are correctly represented as `List`, `Expression`, `Attribute`, or entity/title structures;
   - predicate-argument structure is semantically correct;
   - every predicate has recoverable `pred.arg.1` unless the sentence truly lacks one;
   - `pred.arg.4` is not used for ordinary purpose, reason, time, manner, or condition adjuncts;
   - ordinary adjuncts introduced by `to` or a preposition should use that overt token as the anchored `ModificationalFunctor`; fixed complex prepositions such as `in order to`, `so as to`, `because of`, `due to`, `according to`, `in front of`, `by means of`, and `in addition to` are kept as one functor, with `variable` pointing to the core unit and `body` pointing to the introduced content;
   - modifiers attach to the smallest complete semantic unit they modify;
   - multi-layer modifiers follow semantic proximity and scope;
   - remove redundant empty `Modification` wrappers when `variable` and `body` recover the same token set, when `body` already anchors the same `variable` as an overt `ModificationalFunctor` / inner `Modification`, or when `body` is already reachable from `variable`;
   - explicit wh-words, relative `that`, or referring expressions use `ref/as:ref` or `Ref`;
   - zero relative clauses do not invent `Ref`;
   - punctuation roots close the correct local structure;
   - expression-internal format separators split into recoverable parts and use the separator/operator as `ExpressionFunctor`; field/header/discourse/syntactic close punctuation remains punctuation.
   - list items are logical fields: keep flat table/stat columns as siblings, but group field-internal qualifiers, scores, dates, `vs.` matches, and domain notation such as cricket `b Harris` before inserting them into the outer `List`; never leave an `OtherConstant` as a sibling list item.
5. Produce a JSON report. Do not silently repair; report original errors and include a minimal corrected annotation in `corrected_annotation` whenever the verdict is `minor_error`, `major_error`, or `invalid` and a repair is possible.
6. If the user requests changes, edit the data and provide before/after diffs or diagrams.

## Mode B: Parse Raw Sentence

Use this mode when a raw sentence is provided or the user asks to check and repair an online Lingua parse. Follow `references/online-workflow.md` through delivery:

1. Call the real Hugging Face `rudaoshi/lang2logic` service with `scripts/review_online.py fetch`. Preserve the original response and standard snapshot.
2. Run `inspect`; read the complete sentence, every node's SELF/DESC, every edge, and all program findings.
3. Apply only sentence-specific, standard-backed minimal repairs to a copy. The executing model makes semantic judgments; Python does not infer them.
4. Inspect the complete candidate again, fill the review record, then run `finalize`. Return original/corrected graphs, exact changes, reasons and unresolved issues.

Do not replace a failed online call with a constructed graph labeled as service output. Keep genuine ambiguity unresolved. An explicit request for offline/manual parsing may use local construction and Mode A validation, with `source: agent_constructed` clearly stated and no online provenance claim.

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
6. For each record, run structural checks first, then perform linguistic judgment using SELF/DESC and incoming roles, then write the verdict. Use `references/desc-audit.md` for targeted or difficult cases.
7. For every record with `review_status: "fix_proposed"`, include `proposed_corrected_sample`: a complete corrected JSONL sample with `sentence`, `lingua.words`, and `lingua.graph`, not just a patch or prose note.
8. Validate each `proposed_corrected_sample` structurally before reporting it as a proposed repair. If a repair depends on an unresolved policy choice, set `review_status: "needs_human_decision"` and explain the blocked choice.
9. Do not apply fixes in bulk until the user confirms the repair policy for that batch.

## Output Contract

Always output JSON unless the user explicitly asks for prose or diagrams.

For a focused SELF/DESC audit, use the scoped target records in `references/desc-audit.md`. These local verdicts do not claim a full-graph pass. When the audit is part of Mode A/B/C, incorporate the findings into that mode's existing report or review records.

For the online Mode B, use the evidence-preserving contract in `references/online-review-contract.md`: deliver `result.json`, the complete `{sentence, lingua}` in `corrected.json` when accepted, and `REPORT.md`. Link these artifacts and summarize the outcome. An unresolved candidate has `corrected_graph: null`; structural checks and model semantic judgments are reported separately. Do not pass this result envelope to the legacy `validate_output.py` or silently collapse it into a legacy verdict.

For Modes A/C and explicit offline parsing, use `references/output-schema.json` as the authoritative shape. The top-level object should include:

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

- `scripts/review_online.py fetch|inspect|finalize`: online parsing, pinned evidence, SELF/DESC packets and reviewed result delivery. Follow `references/online-workflow.md` for arguments and semantic review steps.
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
- If `p/PunctuationalConstant --as:punctuation--> child` exists, then `p` is the close wrapper and `DESC(p)` is the closed component. Do not also treat the same `p` as an ordinary outer `punctuation` child; retarget the outer semantic edge to `p`, or delete the redundant plain punctuation edge when `p` already has the correct semantic parent.
- In `Apposition`, judge `pred.arg.1` by `DESC(arg1)`: a host-only close punctuation may be `arg1`, but punctuation whose `as:punctuation` target is the whole apposition stays outside the `Apposition` node and is not an alternate internal `arg1` form.
- Do not keep a tokenless outer `Modification` if its `body` already represents the full anchored modifier with the same `variable`; retarget incoming edges to that body root unless the outer wrapper adds a real wider scope. If the `body` is already reachable from `variable`, retarget incoming edges to `variable` instead.
- Numeric value-label statistics such as `42 - Number of ...` and `49 - Percentage of ...` are `ExpressionFunctor` dash structures with `func.arg` value and label; they are not `Apposition`.
- Format is a functor: in dates, times, addresses, person/proper-name abbreviations, URL/email/file paths, codes, formulas, calculations, currency/percent expressions, and similar expressions, split recoverable parts and make the internal separator/operator an `ExpressionFunctor`. Keep field/header/discourse delimiters and ordinary syntactic close punctuation as punctuation close markers.
- Treat punctuation-root warnings as low-value by default in this dataset; inspect punctuation only for local close-scope errors.
- When uncertain, report competing analyses and confidence.
