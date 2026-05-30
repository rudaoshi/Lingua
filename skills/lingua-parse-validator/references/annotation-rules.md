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

R-SCHEMA-04: `Missing` nodes are not allowed. A sample must not contain `spans: "Missing"`, symbol label `Missing`, or any node used as a placeholder for an implicit entity or omitted argument.

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

R-PRED-06: Do not repair missing `pred.arg.1` by reassigning an object or complement to `pred.arg.1`. If a subject is omitted and cannot be recovered from sentence-internal structure, leave `pred.arg.1` absent with justification; do not add a `Missing` node.

R-PRED-07: In ditransitives like `give me a date`, the thing given is normally `pred.arg.2` and the recipient is `pred.arg.3`.

R-PRED-08: Rating or naming complements such as `rated it 5 stars` are complement/value roles, not indirect objects; prefer `pred.arg.4` or an attribute-like structure over `pred.arg.3`.

R-PRED-09: A node whose visible token content is only punctuation or symbols must not be typed as `FactualPredicator`, `LogicalPredicator`, `AppositionalPredicator`, `AttributePredicator`, or `ReferentialPredicator`. Use `PunctuationalConstant`, `SymbolConstant`, `ExpressionFunctor`/`Expression`, or a lexical/auxiliary relation-bearing node instead.

R-MOD-01: `Modification.variable` is the semantic unit being modified; `Modification.body` is the modifier.

R-MOD-02: A modifier must attach to the smallest complete semantic unit it modifies.

R-MOD-03: Multi-layer modifiers must follow semantic proximity: closer modifiers are inner, wider-scope modifiers are outer.

R-MOD-04: Fronted modifiers are high-risk but not special by themselves; check all modifier scope and hierarchy.

R-MOD-05: Remove redundant `Modification` wrappers when no recoverability or scope information is lost.

R-MOD-06: Purpose clauses headed by infinitives such as `to push through...` usually modify the complete event they explain; do not attach them as the main predicate's object or `pred.arg.4` unless the predicate selects that complement.

R-MOD-07: When an ordinary purpose, time, manner, or similar adjunct is introduced by overt `to` or a preposition, prefer that token as the anchored `ModificationalFunctor`: `variable` points to the core event or wrapped core event, and `body` points to the adjunct content.

R-MOD-08: Fixed complex prepositions and modifier introducers are single `ModificationalFunctor` nodes when they function as modifiers. The mandatory list includes purpose/use (`in order to`, `so as to`, `in order for`, `for the purpose of`, `for purposes of`, `for the sake of`), reason/result (`because of`, `due to`, `owing to`, `thanks to`, `on account of`, `as a result of`, `by reason of`, `in view of`, `in light of`), concession/exception/replacement (`in spite of`, `regardless of`, `irrespective of`, `apart from`, `aside from`, `except for`, `instead of`, `rather than`), basis/constraint (`according to`, `pursuant to`, `in accordance with`, `in compliance with`, `in line with`, `consistent with`, `subject to`, `contrary to`), relation/topic (`with respect to`, `with regard to`, `in regard to`, `in relation to`, `relative to`, `in terms of`, `in connection with`, `with reference to`, `in reference to`, `as to`, `as for`), space/time (`in front of`, `in back of`, `on top of`, `at the top of`, `at the bottom of`, `at the end of`, `at the beginning of`, `in the middle of`, `next to`, `close to`, `far from`, `away from`, `out of`, `inside of`, `outside of`, `ahead of`, `prior to`, `subsequent to`), and means/exchange/addition (`by means of`, `by way of`, `in exchange for`, `in return for`, `in place of`, `on behalf of`, `in favor of`, `for lack of`, `in case of`, `in addition to`, `as well as`, `together with`, `along with`). Do not split these internally or add an extra unanchored `Modification` wrapper unless extra scope requires it.

R-MOD-08a: Apply R-MOD-08 by construction, not by raw substring. Do not force a fixed-modifier node for homographs that belong to another construction, such as scheduled `due + infinitive` (`is due to start`), phrasal/prepositional verbs (`back/drop/come/go out of`, `get/go along with`, `shy away from`, `come away from`), rating ratios (`5 out of 5`), or adjacent but separate `next` + purpose `to`.

R-MOD-09: Reason expressions such as `because of...` are reason modifiers or logical relations, not predicate-internal complements.

R-REF-01: Explicit wh-words, relative `that`, or referring expressions require `ref/as:ref` or equivalent `Ref`.

R-REF-02: Zero relative clauses must not invent `Ref`; connect the head entity directly to the implicit gap when recoverable, otherwise leave the gap unfilled. Do not use `Missing`.

R-FUNC-01: `but` used for contrast is `LogicalPredicator`.

R-FUNC-02: `if` meaning "if" is `LogicalPredicator`; `if` meaning "whether" is `GeneralFunctor`.

R-FUNC-03: `that` is typed by function: nominal clause marker `GeneralFunctor`, demonstrative entity `NominalConstant`, determiner `DeterminerConstant`, relative marker `GeneralFunctor`.

R-FUNC-04: Sentence-initial `but/however` that preserves an adversative or contrast relation remains `LogicalPredicator`; absence of an in-sample left argument is not by itself a reason to retag it as discourse or modification. The previous or left proposition is `pred.arg.1` when present or recoverable, and the current adversative proposition is normally `pred.arg.2`.

R-FUNC-05: `LogicalPredicator` argument numbers are syntactic relation labels: the main, matrix, host, or unmarked proposition is `pred.arg.1`; the clause or proposition directly marked by the logical connector is `pred.arg.2`. For `because/if/although/when/unless` and similar subordinators, the marked subordinate clause is `pred.arg.2` regardless of word order.

R-FUNC-06: Resultive or conclusive `so/therefore/thus` is `LogicalPredicator`, but it is not a subordinating main/subordinate-clause relation. In `A so/therefore/thus B`, the previous or left unmarked proposition `A` is `pred.arg.1`, and the marker-introduced posterior/current proposition `B` is `pred.arg.2`. In sentence-initial one-sided `So/Therefore/Thus B`, do not invent an external `A`; the visible current proposition `B` remains `pred.arg.2`.

R-FUNC-07: Additive or alternative sentence-initial `and/or` should preserve conjunction or alternative semantics as `ConjunctionalFunctor` with `func.arg`, rather than being changed to ordinary modification solely because one conjunct is discourse-external.

R-PUNCT-01: If punctuation clearly closes a local structure, it should be the root of that local closed structure.

R-PUNCT-02: A punctuation root is common in the current JSONL representation and is not by itself an error. It becomes an error only when the punctuation closes the wrong structure.

R-PUNCT-03: Delimiters such as `:`, `-`, `--`, `...`, parentheses, brackets, quotes, slashes, and repeated punctuation may mark field boundaries, parentheticals, discourse breaks, appositions, lists, or expressions, but the semantic relation must be carried by lexical tokens or auxiliary structures such as `Attribute`, `Discourse`, `Parataxis`, `Apposition`, or `Expression`; field/discourse delimiters remain punctuation/symbol unless they are explicitly serving as expression format functors.

R-PUNCT-04: `&` meaning "and" is `ConjunctionalFunctor`; `&` inside names, titles, codes, or symbol strings is an expression/symbol part. It should not be `LogicalPredicator`.

R-PUNCT-05: Before attaching a punctuation or symbol node to an outer relation with `punctuation`, check whether it closes a local component. In `A: B`, `A - B`, `A -- B`, or `A ... B`, the delimiter usually closes `A`, so the outer relation should point to the punctuation wrapper and the punctuation should have `as:punctuation -> A`. For paired parentheses/brackets/quotes around `B`, the wrapper closes `B`. Use outer `punctuation` only when the mark belongs to the whole relation or cannot be localized.

R-EXPR-01: Format is a functor. In dates, times, durations, scores, addresses, person/proper-name abbreviations, URL/email/file paths, model codes, chemical or biological markers, and calculation expressions, an internal separator or operator that combines recoverable parts of the expression should be an `ExpressionFunctor` with `func.arg` edges to the adjacent parts or to an inner expression functor. Do not merge the whole formatted string into one `NominalConstant` merely because it is lexicalized as a surface string.

R-EXPR-02: Common format functors include numeric time/duration/score colons (`08:23`, `08:36:00`, `1:18.52`), date slashes or dashes (`06/04/01`, `1996-12-06`), decimal/version dots (`0.25`, `2.3`), numeric thousands/date/address commas (`35,000`, `March 28, 2003`, `Houston, Texas`), hyphenated proper-name/code separators (`Epstein-Barr`, `ATF-1`), URL/email/path/query separators (`/`, `.`, `@`, `?`, `=`, `&`, `#`), calculation operators, compact formula parentheses (`25(OH)D3`), currency/percent/rank symbols when they form a numeric expression, and compact biomedical suffixes such as `CD4+`. Field/header colons, discourse dashes, ordinary parentheticals, list commas, and punctuation that closes a local syntactic component remain punctuation close markers rather than expression functors.

R-DESC-01: Judge wrapper arguments by descendant subgraph recoverability, not by the wrapper node alone.

R-DESC-02: Descendant recoverability does not excuse wrong modifier scope or wrong inner/outer order.

R-SEM-01: Do not add information absent from the sentence.

R-SEM-02: Do not omit core semantic content.

R-SEM-03: Treat `nonsense` as legal but low-information, not as a high-confidence semantic relation.

R-SEM-03a: Dummy or extraposition `it` in patterns such as `It is interesting/easy/fair to ...` or `it is possible that ...` is analyzable; use `Ref`, `Modification`, or an equivalent recoverable structure instead of `nonsense`.

R-FRAG-01: Before requiring a main factual predicate, decide whether the whole sample or a prefix/suffix is a document fragment, table/header field, UI fragment, title, or proper-name fragment.

R-FRAG-02: Pure structured fragments still need token coverage, legal labels, one recoverable root, and a DAG, but they do not need an invented `FactualPredicator`, subject, object, or ordinary proposition.

R-FRAG-03: Parallel headers, table columns, menu items, and list-like labels should use `ListFunctor`/`List` or equivalent list structure; do not invent predicate-argument relations between parallel fields.

R-FRAG-04: Page refs, scores, dates, times, time ranges, phone numbers, addresses, filenames, percentages, money, model codes, and similar symbol strings should be recoverable as `ExpressionFunctor`/`Expression` units when they are not ordinary sentences.

R-FRAG-05: Field or key-value records should distinguish key/field label from value. When the field label is explicit, the field label token(s) must be the lexical `AttributePredicator`; its value is `pred.arg.2`. If the external host is absent, leave `pred.arg.1` absent; do not add `Missing`. Do not create `Attribute(spans="Attribute") --attribute--> field_label`, and do not represent a non-predicative field label such as `Email`, `Location`, `Attendance`, `Pin #`, `Job ID`, `DATE`, `Game N`, or `Preseason Game N` as a normal `FactualPredicator` merely to connect it to the value.

R-FRAG-06: Header, speaker label, field label, key-value label, or UI metadata must not enter the core predicate-argument frame of nearby body text unless the text explicitly makes it part of the fact.

R-FRAG-07: Use `nonsense` only for truly uninterpretable retained fragments, not as a substitute for recoverable `List`, `Expression`, `Attribute`, punctuation, or discourse structure.

R-FRAG-08: Low-information fragments such as greetings, signoffs, copyright notices, media boundary markers, decorative separators, clipped text, and malformed residues still have fragment semantics. Preserve them as low-information fragments unless an external dataset-cleaning policy explicitly requires removal.

R-FRAG-09: Do not flatten greetings, signoffs, thanks, clipped clauses, or malformed residues into `Expression` merely because they are low-information. If ordinary syntax or ellipsis is recoverable, parse that structure first; unrecoverable omitted participants remain absent and must not be represented with `Missing`.

R-FRAG-10: Standalone date/time/date-time/time-range fragments such as `02/27/2001 08:23 AM`, `Feb 2005`, `September 21, 2004`, `Tuesday, March 20th.`, `September 20, 1888?`, or `4:00 pm - 6:00 pm` should be represented as one recoverable `Expression` unit, with date, time, meridiem, punctuation, timezone, and range boundary as expression parts when useful. Internal date/time/range separators inside these expressions follow R-EXPR-01 and R-EXPR-02. Do not model internal date-time formatting as `Modification`, `as:modification`, or a normal `FactualPredicator`.

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
