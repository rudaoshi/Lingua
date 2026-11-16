# Session Lessons

These are calibrated lessons from aligning the Lingua standard with existing JSONL data.

## Standard Updates And Stable Decisions

- `nonsense` is a legal edge label and automatically allows `as:nonsense`; treat it as low-information, not schema-invalid.
- `but` used as contrast is always `LogicalPredicator`.
- `if` is `LogicalPredicator` when it means conditional "if"; it is `GeneralFunctor` when it means "whether".
- `that` is typed by function:
  - nominal-clause marker: `GeneralFunctor`;
  - relative marker: `GeneralFunctor` plus `Ref` or equivalent reference structure;
  - demonstrative entity: `NominalConstant`;
  - determiner: `DeterminerConstant`.
- Predicate nodes should have `pred.arg.1` when explicit or structurally recoverable, but missing `arg1` does not justify role reassignment.
- Use `pred.arg.4` for object complements, result complements, and nonfinite events controlled by an object or other matrix argument. A selected content event such as `decide to stay` is `pred.arg.2`. An event-level purpose is a modifier (§6.1.1).
- Descendant subgraph semantics are central: judge wrappers by `DESC(v)` recoverability, but still enforce modifier proximity and scope order.
- Explicit wh-words, relative `that`, or referring expressions require `Ref` or `ref/as:ref`; zero relatives must not invent `Ref`.
- Punctuation that clearly closes a local structure should root that local closed structure.
- Current data must not contain `Missing` nodes. Unrecoverable omitted subjects, field hosts, and implicit participants stay absent rather than being represented by `Missing`.
- Pure structured fragments such as table headers, scores, page refs, schedule rows, addresses, filenames, and UI labels are not ordinary sentences; they may be rooted as `List`, `Expression`, `Attribute`, or low-information fragment structures without an invented `FactualPredicator`.
- Greetings, signoffs, copyright notices, media boundary markers, decorative separators, clipped text, and malformed residues still have fragment semantics. Do not filter them merely for low semantic density; preserve them as low-information fragments unless a separate dataset-cleaning policy says otherwise.
- Low-information does not imply `Expression` flattening. For greetings, signoffs, thanks, clipped clauses, and malformed residues, first try an ordinary or elliptical parse; use `Expression` only when no more specific syntactic or semantic relation can be recovered.
- Proper names, titles, and nominal fragments without explicit predicates should not be forced into copula/factual-predicate graphs; preserve the entity/title expression and any qualifier relation.

## Repaired Pattern Library

Purpose adjunct:

- `To push through a sales tax increase, the state will have to suspend...`
- `To push through...` modifies the `suspend` event and has recoverable `pred.arg.1 -> the state`.
- It is not `suspend.pred.arg.4`.

Zero relative:

- `the collective ocean we all swim in`
- No `Ref`; model the relative body through `in.variable -> swim`, `in.body -> ocean`.

Modifier hierarchy:

- `Before July ... without focusing...`
- `without focusing...` is closer to the event; `Before July` is outer.

Temporal/event scope:

- `resigned as the captain in September 2007`
- Date modifies the `resigned as captain` event, not the noun `captain` alone.

Ditransitive roles:

- `give me a date`
- `a date` is `pred.arg.2`; `me` is `pred.arg.3`.

Rating complement:

- `rated it 5 stars`
- `5 stars` is a complement/value role such as `pred.arg.4`, not `pred.arg.3`.

## Important Boundary Case

`Broke out the activities of 1179.`

- Earlier attempted repair changed `the activities of 1179` from `pred.arg.2` to `pred.arg.1`.
- That was retracted.
- Surface order favors an omitted-subject transitive reading: `[someone/it] broke out the activities...`
- Current policy leaves the omitted `arg1` absent; do not add `Missing`, and do not promote the object to subject.

## Audit Process Lesson

Batch screening is not full audit.

In one 100-sample pass:

- 100 samples were batch-screened;
- 86 received some heuristic flag, mostly punctuation roots;
- about 16 were truly high-risk;
- 21 were deeply expanded because previous repair samples were also checked.

The honest description is: `100 screened + high-risk/selected samples deeply reviewed`, not `100 fully audited`.

## Confidence Screening Lesson

The lowest-confidence 100 samples from the 3-fold wordgraph run were dominated by non-sentential or low-distribution forms rather than obvious core Lingua contradictions:

- structured fragments / table fields: 33;
- proper-name, title, or nominal fragments: 26;
- noise, greetings, media markers, or clipped text: 17;
- complete complex sentences needing semantic review: 13;
- questions, ellipsis, or whether-like risk cases: 11.

The dominant loss component was `rel` for 79/100 records. Treat this as evidence that edge-relation policy for fragments and name/title structures needs standardization; do not treat low confidence alone as proof that the annotation is wrong.
