# Error Taxonomy

Use these categories in `errors[].error_type`.

## Structural Errors

`FORM_ERROR`: missing required fields, malformed JSON, or unsupported input shape.

`TOKEN_PARTITION_ERROR`: lexical spans do not cover tokens exactly once, overlap, or refer outside token bounds.

`GRAPH_FORM_ERROR`: root count, reachability, or DAG validity failure.

`SCHEMA_ERROR`: illegal node type, auxiliary label, edge label, or edge endpoint.

`AUX_STRUCTURE_ERROR`: required auxiliary node edges are missing, such as `Modification.variable/body` or `Ref.pred.arg.1/pred.arg.2`.

## Linguistic Errors

`BOUNDARY_ERROR`: node span includes too much, too little, or the wrong phrase.

`NODE_TYPE_ERROR`: linguistic unit has the wrong node type.

`PREDICATE_ARGUMENT_ERROR`: wrong, missing, reversed, or over-generated predicate arguments.

`PRED_ARG4_ADJUNCT_ERROR`: ordinary adjunct mislabeled as `pred.arg.4`.

`MODIFIER_SCOPE_ERROR`: modifier attaches to the wrong head or wrong semantic scope.

`MODIFIER_HIERARCHY_ERROR`: nested modifiers are in the wrong proximity order.

`REDUNDANT_STRUCTURE_ERROR`: wrapper node adds no recoverability, scope, or semantic value.

`REFERENCE_ERROR`: wrong, missing, or invented `Ref`/`ref` relation.

`FUNCTION_WORD_ERROR`: `but`, `if`, `that`, or similar functional word has wrong type or structure.

`PUNCTUATION_SCOPE_ERROR`: punctuation closes the wrong local structure.

`COORDINATION_ERROR`: conjunction/list/parataxis structure loses items, reverses scope, or misrepresents shared predicates.

`EXPRESSION_ERROR`: expression, range, formula, date, score, or symbol sequence is modeled as the wrong structure.

`ATTRIBUTE_APPOSITION_ERROR`: attribute or apposition relation is reversed or confused with ordinary modification.

`OMISSION_ERROR`: omitted or implicit element is missing, unjustified, or represented by the wrong explicit node.

`OVER_GENERATION_ERROR`: graph adds entities, facts, references, or relations not present in the sentence.

`UNDER_SPECIFICATION_ERROR`: graph is technically valid but loses important recoverable semantics.

`AUDIT_PROCESS_ERROR`: the review method is misstated, such as calling a high-risk targeted review a full per-sample audit.

## Severity Guidance

`minor`:

- Does not change core fact structure.
- Local label or wrapper choice remains recoverable by `DESC(v)`.
- Punctuation or discourse issue with little semantic effect.

`major`:

- Changes core predicate-argument structure.
- Wrong modifier scope or hierarchy changes meaning.
- Missing recoverable `pred.arg.1`.
- Reassigning an object to `pred.arg.1` merely to satisfy the `arg1` preference.
- Incorrect or invented reference.
- `pred.arg.4` used for an ordinary adjunct.

`critical`:

- Graph is invalid or disconnected.
- Edge endpoints or labels are illegal.
- Main fact is absent or seriously misleading.
- Annotation cannot be reliably interpreted as Lingua.

Verdict mapping:

- no errors: `valid`
- only minor errors: `minor_error`
- at least one major error and no critical errors: `major_error`
- any critical error or multiple interacting major errors making the graph unusable: `invalid`
