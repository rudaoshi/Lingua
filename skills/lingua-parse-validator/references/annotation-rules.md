# Annotation Rules

Use these rule IDs in validation reports. The complete discussion lives in `lingua-standard.md`.

## Object Model

Node:

- `spans`: either token span list `[[start, end], ...]` or an auxiliary label string.
- `type`: legal Lingua node type.
- Optional derived `text`: the token text covered by `spans`.

Edge:

- `start`: source node index.
- `end`: target node index.
- `label`: legal Lingua edge label.

Repo sample:

- `sentence`
- `lingua.words`
- `lingua.graph.nodes`
- `lingua.graph.edges`

## Inventories

Legal node types:

- `AttributePredicator`
- `AppositionalPredicator`
- `FactualPredicator`
- `LogicalPredicator`
- `ReferentialPredicator`
- `ConjunctionalFunctor`
- `ExpressionFunctor`
- `GeneralFunctor`
- `ListFunctor`
- `ModificationalFunctor`
- `DeterminerConstant`
- `InterjectionConstant`
- `ModificationalConstant`
- `NominalConstant`
- `OtherConstant`
- `PunctuationalConstant`
- `SymbolConstant`

Legal auxiliary node labels:

- `Apposition`
- `Attribute`
- `Copula`
- `Discourse`
- `Expression`
- `List`
- `Missing`
- `Modification`
- `Parataxis`
- `Ref`
- `Vocative`
- `Whether`

Legal base edge labels:

- `body`
- `variable`
- `pred.arg.1`
- `pred.arg.2`
- `pred.arg.3`
- `pred.arg.4`
- `func.arg`
- `appositive`
- `attribute`
- `discourse`
- `index`
- `modification`
- `nonsense`
- `punctuation`
- `ref`
- `repeat`
- `vocative`

Each base edge is also legal with `as:` prefix.

## Structural Rules

R-FORM-01: A sample must contain a sentence and a graph with nodes and edges.

R-FORM-02: Every lexical token must be covered exactly once by lexical nodes when tokenization is available.

R-SCHEMA-01: Every node type must be in the legal node type inventory.

R-SCHEMA-02: Every auxiliary node string must be in the legal auxiliary label inventory.

R-SCHEMA-03: Every edge label must be legal, including legal `as:` forms.

R-EDGE-01: Every edge source and target must refer to existing node IDs.

R-GRAPH-01: A complete Lingua graph should have exactly one root.

R-GRAPH-02: The graph must be reachable from the root.

R-GRAPH-03: The graph must be acyclic unless the standard explicitly allows a special cyclic encoding. Current JSONL data expects a DAG.

R-AUX-01: `Modification` nodes must have at least one `variable` and one `body`.

R-AUX-02: `Ref` nodes must have `pred.arg.1` and `pred.arg.2`.

## Linguistic Rules

R-PRED-01: Predicate nodes must cover predicate expressions, not ordinary nominal heads.

R-PRED-02: Core predicate arguments must use the correct semantic role numbers.

R-PRED-03: Every predicate should have `pred.arg.1` when the subject-like argument is explicit or structurally recoverable. Missing `arg1` is acceptable only when the sentence truly lacks one or it cannot be recovered.

R-PRED-04: Use `pred.arg.4` only for predicate-internal complements such as object complements, result complements, and selected nonfinite complements.

R-PRED-05: Do not use `pred.arg.4` for ordinary purpose, reason, time, manner, condition, or discourse adjuncts.

R-PRED-06: Do not repair missing `pred.arg.1` by reassigning an object or complement to `pred.arg.1`. If a subject is omitted, either leave `pred.arg.1` absent with justification or add a justified `Missing` node according to the repair policy.

R-PRED-07: In ditransitives like `give me a date`, the thing given is normally `pred.arg.2` and the recipient is `pred.arg.3`.

R-PRED-08: Rating or naming complements such as `rated it 5 stars` are complement/value roles, not indirect objects; prefer `pred.arg.4` or an attribute-like structure over `pred.arg.3`.

R-MOD-01: `Modification.variable` is the semantic unit being modified; `Modification.body` is the modifier.

R-MOD-02: A modifier must attach to the smallest complete semantic unit it modifies.

R-MOD-03: Multi-layer modifiers must follow semantic proximity: closer modifiers are inner, wider-scope modifiers are outer.

R-MOD-04: Fronted modifiers are high-risk but not special by themselves; check all modifier scope and hierarchy.

R-MOD-05: Remove redundant `Modification` wrappers when no recoverability or scope information is lost.

R-MOD-06: Purpose clauses headed by infinitives such as `to push through...` usually modify the complete event they explain; do not attach them as the main predicate's object or `pred.arg.4` unless the predicate selects that complement.

R-MOD-07: Reason expressions such as `because of...` are reason modifiers or logical relations, not predicate-internal complements.

R-REF-01: Explicit wh-words, relative `that`, or referring expressions require `ref/as:ref` or equivalent `Ref`.

R-REF-02: Zero relative clauses must not invent `Ref`; connect the head entity directly to the implicit gap or use justified `Missing`.

R-FUNC-01: `but` used for contrast is `LogicalPredicator`.

R-FUNC-02: `if` meaning "if" is `LogicalPredicator`; `if` meaning "whether" is `GeneralFunctor`.

R-FUNC-03: `that` is typed by function: nominal clause marker `GeneralFunctor`, demonstrative entity `NominalConstant`, determiner `DeterminerConstant`, relative marker `GeneralFunctor`.

R-PUNCT-01: If punctuation clearly closes a local structure, it should be the root of that local closed structure.

R-PUNCT-02: A punctuation root is common in the current JSONL representation and is not by itself an error. It becomes an error only when the punctuation closes the wrong structure.

R-DESC-01: Judge wrapper arguments by descendant subgraph recoverability, not by the wrapper node alone.

R-DESC-02: Descendant recoverability does not excuse wrong modifier scope or wrong inner/outer order.

R-SEM-01: Do not add information absent from the sentence.

R-SEM-02: Do not omit core semantic content.

R-SEM-03: Treat `nonsense` as legal but low-information, not as a high-confidence semantic relation.

R-FRAG-01: Before requiring a main factual predicate, decide whether the whole sample or a prefix/suffix is a document fragment, table/header field, UI fragment, title, or proper-name fragment.

R-FRAG-02: Pure structured fragments still need token coverage, legal labels, one recoverable root, and a DAG, but they do not need an invented `FactualPredicator`, subject, object, or ordinary proposition.

R-FRAG-03: Parallel headers, table columns, menu items, and list-like labels should use `ListFunctor`/`List` or equivalent list structure; do not invent predicate-argument relations between parallel fields.

R-FRAG-04: Page refs, scores, dates, times, time ranges, phone numbers, addresses, filenames, percentages, money, model codes, and similar symbol strings should be recoverable as `ExpressionFunctor`/`Expression` units when they are not ordinary sentences.

R-FRAG-05: Field or key-value records should distinguish key/field label from value. When the field label is explicit, the field label token(s) must be the lexical `AttributePredicator`; its value is `pred.arg.2`, and a missing external host is represented as `pred.arg.1 -> Missing`. Do not create `Attribute(spans="Attribute") --attribute--> field_label`, and do not represent a non-predicative field label such as `Email`, `Location`, `Attendance`, `Pin #`, `Job ID`, `DATE`, `Game N`, or `Preseason Game N` as a normal `FactualPredicator` merely to connect it to the value.

R-FRAG-06: Header, speaker label, field label, key-value label, or UI metadata must not enter the core predicate-argument frame of nearby body text unless the text explicitly makes it part of the fact.

R-FRAG-07: Use `nonsense` only for truly uninterpretable retained fragments, not as a substitute for recoverable `List`, `Expression`, `Attribute`, punctuation, or discourse structure.

R-FRAG-08: Low-information fragments such as greetings, signoffs, copyright notices, media boundary markers, decorative separators, clipped text, and malformed residues still have fragment semantics. Preserve them as low-information fragments unless an external dataset-cleaning policy explicitly requires removal.

R-FRAG-09: Do not flatten greetings, signoffs, thanks, clipped clauses, or malformed residues into `Expression` merely because they are low-information. If ordinary syntax or ellipsis is recoverable, parse that structure first and use `Missing` only for structurally justified omissions.

R-FRAG-10: Standalone date/time/date-time/time-range fragments such as `02/27/2001 08:23 AM`, `Feb 2005`, `September 21, 2004`, `Tuesday, March 20th.`, `September 20, 1888?`, or `4:00 pm - 6:00 pm` should be represented as one recoverable `Expression` unit, with date, time, meridiem, punctuation, timezone, and range boundary as expression parts when useful. Do not model internal date-time formatting as `Modification`, `as:modification`, or a normal `FactualPredicator`.

R-FRAG-11: Dateline/source rows such as `DUBLIN 1996-08-27` or `LOS ANGELES 1996-12-05` should be represented as `List(location-expression, date-expression)` or an equivalent recoverable two-field structure. The location and date are sibling metadata fields, not modifier/head, predicate/argument, or omitted-sentence relations.

R-FRAG-12: Standalone page references such as `p. 36` or `pp. 12-15` should be represented as a recoverable page-reference `Expression`, not as a parallel `List` and not as a normal predicate.

R-NAME-01: Multi-token proper names, titles, and nominal fragments without explicit predicates should be represented as entity/title/expression fragments; do not invent copulas or factual predicates.

R-NAME-02: Role labels, parenthetical disambiguators, geographic qualifiers, and title qualifiers should be represented as attribute, apposition, modification, or discourse metadata according to their semantic function.

R-NAME-03: Entity + qualifier strings such as `Jordan, New York` or `Beetham Tower, Manchester` are usually not flat lists; preserve the qualifier relation when recoverable.

## Audit Process Rules

R-AUDIT-01: Distinguish `batch screening`, `targeted review`, and `full per-sample audit` in all reports.

R-AUDIT-02: A full per-sample audit requires a saved record for every reviewed line, including lines judged valid.

R-AUDIT-03: For long audits, produce resumable JSONL artifacts with `line_number`, `sentence`, `structural_status`, `linguistic_verdict`, `errors`, `suggested_fix`, `confidence`, and `review_status`.

R-AUDIT-04: Do not claim that a dataset was fully audited if only high-risk samples were expanded and semantically reviewed.

R-AUDIT-05: For repairs, record the original error and minimal change. Do not silently rewrite a graph.

R-AUDIT-06: Low model confidence is a review-priority signal, not proof of a standard violation. Separate model uncertainty, standard coverage gaps, data noise, and confirmed annotation errors in reports.
