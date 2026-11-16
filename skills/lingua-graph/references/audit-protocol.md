# Audit Protocol

Use this protocol for sampled or full-dataset Lingua annotation review.

## Audit Levels

`batch screening`:

- Run structural scripts and heuristic checks over many samples.
- Suitable for finding high-risk records.
- Does not count as full semantic review.

`targeted review`:

- Deeply inspect selected records, usually high-risk flags or user-selected lines.
- Report exactly how many were inspected.
- When records are selected by model confidence, report the confidence source, threshold or top-N rule, and separate model uncertainty from confirmed annotation errors.

`full per-sample audit`:

- Expand and inspect every sample in scope.
- Create one audit record for every line, including valid lines.
- Save results in a resumable JSONL file.

## Required Full-Audit Record

Each reviewed line should have:

```json
{
  "line_number": 1,
  "sentence": "...",
  "structural_status": "pass",
  "linguistic_verdict": "valid",
  "confidence": 0.9,
  "errors": [],
  "suggested_fix": null,
  "proposed_corrected_sample": null,
  "review_status": "reviewed",
  "notes": []
}
```

Allowed `review_status` values:

- `screened`
- `reviewed`
- `needs_human_decision`
- `fix_proposed`
- `fixed`
- `skipped`

## Recommended Artifacts

For a batch such as lines 1-200:

- `reports/train_audit/audit_000001_000200.jsonl`
- `reports/train_audit/summary_000001_000200.md`
- `reports/train_audit/fixes_proposed_000001_000200.jsonl`
- `reports/train_audit/visualizations_000001_000200.md`

The fixes file should contain the full audit records for problem lines, including `proposed_corrected_sample`.

## Procedure

1. Generate audit skeletons with `prepare_audit_batch.py`.
2. Run structural validation for the same range.
3. For each record:
   - read sentence, words, nodes, and edges;
   - check deterministic structural status;
   - inspect predicate-argument structure;
   - inspect modifier scope and hierarchy;
   - inspect references, function words, punctuation scope, coordination, and omissions;
   - classify document fragments, table/header fields, titles, and proper-name fragments before requiring ordinary sentence PFA structure;
   - write verdict and errors;
   - for each fixable error record, write a complete `proposed_corrected_sample`.
4. After the batch, summarize:
   - number screened;
   - number deeply reviewed;
   - number valid;
   - number with minor/major/critical errors;
   - top error types;
   - repair policy questions.
5. Apply data fixes only after user confirmation.

## Proposed Corrected Samples

For `review_status: "fix_proposed"`, `proposed_corrected_sample` is required.

The corrected sample must be:

- a complete JSONL sample, not a patch;
- in the same shape as the input repo sample: `sentence`, `lingua.words`, `lingua.graph.nodes`, `lingua.graph.edges`;
- minimal: preserve tokenization, spans, metadata, and unaffected graph parts;
- structurally valid under `scripts/validate_graph.py`.

Use `proposed_corrected_sample: null` only for `valid`, `screened`, `pending_review`, or `needs_human_decision` records. If a repair is blocked by an unresolved policy choice, use `needs_human_decision` and explain the competing options in `notes`.

## Reporting Discipline

Always separate:

- what scripts checked;
- what the model semantically reviewed;
- what was actually changed.

If only high-risk samples were reviewed, say so. Do not imply the remaining samples were deeply audited.
