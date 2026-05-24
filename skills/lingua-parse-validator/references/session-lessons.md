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
- `pred.arg.4` is for predicate-selected complements, object complements, result complements, and similar internal complements, not ordinary adjuncts.
- Descendant subgraph semantics are central: judge wrappers by `DESC(v)` recoverability, but still enforce modifier proximity and scope order.
- Explicit wh-words, relative `that`, or referring expressions require `Ref` or `ref/as:ref`; zero relatives must not invent `Ref`.
- Punctuation that clearly closes a local structure should root that local closed structure.

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
- If a stricter policy requires explicit subjects, add or propose `Missing` for `arg1`; do not promote the object to subject.

## Audit Process Lesson

Batch screening is not full audit.

In one 100-sample pass:

- 100 samples were batch-screened;
- 86 received some heuristic flag, mostly punctuation roots;
- about 16 were truly high-risk;
- 21 were deeply expanded because previous repair samples were also checked.

The honest description is: `100 screened + high-risk/selected samples deeply reviewed`, not `100 fully audited`.
