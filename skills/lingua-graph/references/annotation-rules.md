# Annotation Rules

Use these rule IDs in validation reports. Follow the standard selected by [SKILL.md](../SKILL.md#required-references) for the complete rules.

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

R-PRED-04: Use `pred.arg.4` for complements such as object complements, result complements, and nonfinite events controlled by a matrix argument such as `pred.arg.2` or `pred.arg.3`. Use `pred.arg.2` for a selected content event such as `want to play` or `decide to stay` (§6.1.1).

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

R-MOD-05a: A `Modification` whose `variable` and `body` recover the same descendant token set is a redundant empty wrapper; retarget incoming edges to the shared child and remove the aux node.

R-MOD-05b: If `Modification.body` is already a `ModificationalFunctor` or inner `Modification` whose own `variable` recovers the same semantic unit as the outer `Modification.variable`, the outer `Modification` is redundant. Retarget the outer incoming edge to the body root unless the outer node adds an independently recoverable scope boundary.

R-MOD-05c: If `Modification.body` is already reachable from `Modification.variable` through internal graph edges, the body is already part of the variable's recoverable unit. Retarget incoming edges to the variable root and remove the redundant aux node.

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

R-PUNCT-05: Before attaching a punctuation or symbol node to an outer relation with `punctuation`, check whether it closes a local component. In `A: B`, `A - B`, `A -- B`, or `A ... B`, the delimiter usually closes `A`, so the outer relation should point to the punctuation wrapper and the punctuation should have `as:punctuation -> A`. For parentheses, brackets, quotes, or other punctuation wrappers, use the actual `as:punctuation` target as the closed component; do not infer a special scope merely from surface punctuation count. Use outer `punctuation` only when the mark belongs to the whole relation or cannot be localized.

R-PUNCT-06: If `p/PunctuationalConstant --as:punctuation--> child` exists, `p` is the close wrapper and `DESC(p)` is the closed component. An outer predicate, functor, or aux relation that needs the closed component should point its semantic edge to `p`, not merely attach `p` as `punctuation`. The mixed pattern `parent --punctuation--> p` plus `p --as:punctuation--> child` is usually an error; retarget the outer semantic edge to `p`, or remove the redundant plain punctuation edge if `p` already has the correct semantic parent.

R-APPOS-01: `Apposition.pred.arg.1` is the host/base expression and should be judged by `DESC(arg1)`. In `Lopez, the secretary`, if the comma only closes `Lopez`, use `Apposition.pred.arg.1 -> comma` plus `comma --as:punctuation--> Lopez`. If a punctuation wrapper has `as:punctuation -> Apposition`, that wrapper is outside the `Apposition` and is not an alternate internal `pred.arg.1` form. Do not use the same punctuation wrapper both as an outer apposition close wrapper and as internal `pred.arg.1`.

R-APPOS-02: Numeric value-label statistic fragments such as `42 - Number of nationalities...`, `49 - Percentage of Americans...`, and `63 - Percentage of oil...` are not appositions: the number and the `Number/Percentage/...` label phrase are not the same referent. Use the dash as an `ExpressionFunctor` with `func.arg` edges to the numeric value and to the label phrase.

R-AS-01: A compact shorthand edge `A --as:*--> B` is allowed only when `A` has exactly one outgoing edge, namely that `as:*` edge. If `A` also has `func.arg`, `pred.arg.*`, `variable`, `body`, `repeat`, `punctuation`, or another `as:*` child, expand the relation into the corresponding auxiliary node (`Modification`, `Discourse`, `Vocative`, `Ref`, etc.) and put `A`'s full internal structure in the appropriate argument/body position.

R-EXPR-01: Format is a functor. In dates, times, durations, scores, addresses, person/proper-name abbreviations, URL/email/file paths, model codes, chemical or biological markers, and calculation expressions, an internal separator or operator that combines recoverable parts of the expression should be the `ExpressionFunctor` root with `func.arg` edges to the adjacent parts or to an inner expression functor. Do not add an extra `spans: "Expression"` aux merely to wrap that explicit operator, and do not merge the whole formatted string into one `NominalConstant` merely because it is lexicalized as a surface string.

R-EXPR-02: Common format functors include numeric time/duration/score colons (`08:23`, `08:36:00`, `1:18.52`), date slashes or dashes (`06/04/01`, `1996-12-06`), decimal/version dots (`0.25`, `2.3`), numeric thousands/date/address commas (`35,000`, `March 28, 2003`, `Houston, Texas`), hyphenated proper-name/code separators (`Epstein-Barr`, `ATF-1`), URL/email/path/query separators (`/`, `.`, `@`, `?`, `=`, `&`, `#`), calculation operators, compact formula parentheses (`25(OH)D3`), currency/percent/rank symbols when they form a numeric expression, and compact biomedical suffixes such as `CD4+`. Field/header colons, discourse dashes, ordinary parentheticals, and punctuation that closes a local syntactic component remain punctuation close markers rather than expression functors.

R-EXPR-03: Hyphenated compounds, proper names, codes, and measure modifiers such as `paddle-steamer`, `Al-Anbar`, `99-tonne`, `one-week`, and `NF-kappa` should use the visible hyphen as an `ExpressionFunctor` with `func.arg` edges to the left and right parts. Do not encode this as `Modification(variable=right_part, body=hyphen/PunctuationalConstant)` plus `hyphen --as:punctuation--> left_part`. Dashes that are field, headline, topic, or discourse delimiters are not compound operators; analyze them with punctuation close scope, `Discourse`, `Parataxis`, `Attribute`, or `List` as appropriate.

R-EXPR-04: In numeric statistic fragments of the form `value - Number/Percentage/... label`, the dash is an `ExpressionFunctor` combining the numeric value and the statistical label phrase. This is a value-label expression, not a compound hyphen, not a punctuation-close apposition, and not an `Attribute` with a fabricated host.

R-LIST-01: An explicit list separator can be the list functor. If `,`, `;`, `/`, `...`, or a list-separating dash connects list items, the separator node itself should be the `ListFunctor` root with `func.arg` edges to the items. A single `ListFunctor` may cover multiple separator tokens in one n-ary list. Do not add an extra `spans: "List"` aux merely to wrap that explicit operator; use a `List` aux only when there is no single visible list operator or when a higher-level grouping must combine already formed sublists or heterogeneous fields.

R-LIST-02: A `List.func.arg` item is a logical field, not an arbitrary surface token. In table, ranking, score, roster, and result rows, keep flat numeric/stat columns as sibling items, but first group field-internal structures such as `name (country/team/role)`, dates, scores, ranges, `player vs. player`, and cricket `b bowler` notation as `Attribute`, `Expression`, `ConjunctionalFunctor`/`ListFunctor`, or a punctuation close wrapper.

R-LIST-03: `OtherConstant` must not be used as a sibling list item. If the token is a recoverable field label, give it a concrete constant type; if it is a delimiter or notation marker, make it a list/expression/operator or local close wrapper inside the field.

R-DESC-01: Judge wrapper arguments by descendant subgraph recoverability, not by the wrapper node alone.

R-DESC-02: Descendant recoverability does not excuse wrong modifier scope or wrong inner/outer order.

R-SEM-01: Do not add information absent from the sentence.

R-SEM-02: Do not omit core semantic content.

R-SEM-03: Treat `nonsense` as legal but low-information, not as a high-confidence semantic relation.

R-SEM-03a: Dummy or extraposition `it` in patterns such as `It is interesting/easy/fair to ...` or `it is possible that ...` is analyzable; use `Ref`, `Modification`, or an equivalent recoverable structure instead of `nonsense`.

R-FRAG-01: Before requiring a main factual predicate, decide whether the whole sample or a prefix/suffix is a document fragment, table/header field, UI fragment, title, or proper-name fragment.

R-FRAG-02: Pure structured fragments still need token coverage, legal labels, one recoverable root, and a DAG, but they do not need an invented `FactualPredicator`, subject, object, or ordinary proposition.

R-FRAG-03: Parallel headers, table columns, menu items, and list-like labels should use `ListFunctor`/`List` or equivalent list structure; do not invent predicate-argument relations between parallel fields.

R-FRAG-03b: For structured rows, the outer `List` contains fields, while field-internal qualifiers or notations are analyzed before entering the list. Examples: `rank. name (country) result` is `List(rank, Attribute(name,country), result)`, `player vs. player` uses `vs` as the inner operator, and `b Harris` in a cricket scorecard is one notation field rather than separate `b` and `Harris` list items.

R-FRAG-03A: Topic/header/source/speaker metadata that provides utterance context but does not enter the core predicate-argument structure should use compact `as:discourse`: `context --as:discourse--> core`. The `context` node may be a punctuation close wrapper such as `BASEBALL -` or `DPA:`. Retarget outer edges that need the full contextualized unit to the context wrapper so `DESC(context)` recovers the complete utterance. Do not add new `core --discourse--> context` edges.

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

R-NAME-04: In table, result, or roster rows, structures such as `name ( country/team/role ) score` or `rank. name ( country ) result` should keep the parenthetical as a qualifier of the preceding entity, usually `Attribute(pred.arg.1=name, pred.arg.2=paren-wrapper)`. The outer `List` should contain that `Attribute` as one field, not split the entity and parenthetical into sibling `func.arg`s. Leave row-initial parentheticals without a recoverable host as list fields, and wrap whole-row notes such as `( halftime 2-0 )` or `( home team in CAPS )` around the full result/header rather than attaching them to the preceding entity.

R-NAME-05: Use `ExpressionFunctor` / `Expression` for proper names, titles, brands, works, person names, codes, abbreviations, URLs/emails/paths, and symbol strings whose internal relation is format/identity composition rather than recoverable head-modifier structure. First/last names, nobiliary particles, title fragments, code fragments, opaque brand/title words, and official organization names such as `World Bank`, `World Hats Mart`, or `Clean Ocean Action` should not be turned into predicate arguments or forced into `Modification` merely because one token looks like a common noun. Visible format separators inside names/codes (`ATF-1`, `Epstein-Barr`, `NF-kappa`, URL/email/path-like names) should be `ExpressionFunctor` roots with `func.arg` parts. A simple one-token proper name may remain a `NominalConstant` without an extra `Expression` wrapper.

R-NAME-06: Use hierarchical `Modification` / `as:modification` for proper names with recoverable heads, type words, or nested modifier scope. The head or already-modified head is the `variable`; place, institution, person, class, rank, field, type, and other qualifiers are the `body` or compressed modifier source. Typical recoverable heads include airport, school, university, hospital, river, mountain, province, department, district, league, final, promoter, and virus when those words still function as type/head words. Names such as `Los Angeles International Airport`, `Mojave Airport`, `World Cup quarter-final`, `NF-kappa B`, `Epstein-Barr virus`, and `BamHI W promoter` should preserve their internal head/modifier hierarchy instead of being flattened into one `Expression`.

R-NAME-07: `Expression` and `Modification` may be nested. Use `Expression` to build an inner formatted or opaque name part, then use that part as `Modification.body` when it qualifies a recoverable head. For example, `Epstein-Barr` can be a hyphen `ExpressionFunctor`, while `Epstein-Barr virus` is `Modification(variable=virus, body=Epstein-Barr-expression)`. In `Los Angeles International Airport`, `Los Angeles` can be an inner location-name expression that modifies the already modified head `International Airport`.

R-NAME-08: Do not use flat `List`/`Expression` for entity + qualifier strings when a qualifier relation is recoverable, and do not use `Modification` merely to split opaque names. If internal boundaries are genuinely uncertain but the root recovers the full entity/title and no invented fact is added, treat the variant as minor/acceptable; if a clear head/scope is flattened or an opaque name is over-split, treat it as an entity-structure error.

## Audit Process Rules

R-AUDIT-01: Distinguish `batch screening`, `targeted review`, and `full per-sample audit` in all reports.

R-AUDIT-02: A full per-sample audit requires a saved record for every reviewed line, including lines judged valid.

R-AUDIT-03: For long audits, produce resumable JSONL artifacts with `line_number`, `sentence`, `structural_status`, `linguistic_verdict`, `errors`, `suggested_fix`, `confidence`, and `review_status`.

R-AUDIT-04: Do not claim that a dataset was fully audited if only high-risk samples were expanded and semantically reviewed.

R-AUDIT-05: For repairs, record the original error and minimal change. Do not silently rewrite a graph.

R-AUDIT-06: Low model confidence is a review-priority signal, not proof of a standard violation. Separate model uncertainty, standard coverage gaps, data noise, and confirmed annotation errors in reports.
