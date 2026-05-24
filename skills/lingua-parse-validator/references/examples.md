# Examples

These examples are intentionally small. Use `lingua-standard.md` for full detail.

## Correct: Predicate With Recoverable Subject

Sentence:

```text
She used her knowledge to give private lessons.
```

Expected key structure:

```text
used.pred.arg.1 -> She
used.pred.arg.2 -> her knowledge
used.pred.arg.4 -> to give
to give.pred.arg.1 -> She
to give.pred.arg.2 -> private lessons
```

Reason: `to give` has a recoverable controller, so `pred.arg.1` should be explicit.

## Incorrect: Purpose Adjunct As `pred.arg.4`

Sentence:

```text
To pass the bill, the state suspended the limit.
```

Incorrect:

```text
suspended.pred.arg.4 -> To pass the bill
```

Correct:

```text
Modification.variable -> suspended
Modification.body -> To pass the bill
```

Reason: purpose adjuncts are modifiers, not predicate-internal complements.

## Correct: Zero Relative Without `Ref`

Sentence:

```text
the ocean we swim in
```

Expected key structure:

```text
Modification.variable -> ocean
Modification.body -> in
in.variable -> swim
in.body -> ocean
swim.pred.arg.1 -> we
```

Reason: no explicit wh-word or `that` appears, so no `Ref` should be invented.

## Incorrect: Invented `Ref` In Zero Relative

Sentence:

```text
the way the markets are regulated
```

Incorrect:

```text
Ref.pred.arg.1 -> the way
Ref.pred.arg.2 -> are regulated
```

Correct:

```text
Modification.variable -> the way
Modification.body -> are regulated
```

Reason: `Ref` requires an explicit referring expression or relation marker.

## Correct: Modifier Proximity

Sentence:

```text
On Monday winds will blow across the region.
```

Expected hierarchy:

```text
On Monday.variable -> across the region
across.variable -> will blow
```

Reason: direction/location is closer to `blow`; time frame is wider scope.

## Incorrect: Reversed Give Arguments

Sentence:

```text
No one will give me a date.
```

Incorrect:

```text
give.pred.arg.2 -> me
give.pred.arg.3 -> a date
```

Correct:

```text
give.pred.arg.2 -> a date
give.pred.arg.3 -> me
```

## Function Word Typing

`but`:

```text
A happened, but B happened.
but.type -> LogicalPredicator
```

Conditional `if`:

```text
If it rains, leave.
if.type -> LogicalPredicator
```

Whether `if`:

```text
Do you know if Bobby can see visitors?
if.type -> GeneralFunctor
```

Nominal-clause `that`:

```text
We determined that the region was within the fragment.
that.type -> GeneralFunctor
```

Relative `that`:

```text
the tool that can execute surveys
that.type -> GeneralFunctor
Ref connects the tool to the relative-clause structure.
```

Demonstrative `that`:

```text
I want that.
that.type -> NominalConstant
```

## Correct: Do Not Promote Object To `arg1`

Sentence:

```text
Broke out the activities of 1179.
```

Do not "fix" missing `arg1` like this:

```text
Broke out.pred.arg.1 -> the activities of 1179
```

Preferred if no explicit subject is present:

```text
Broke out.pred.arg.2 -> the activities of 1179
```

Possible stricter repair if the batch policy requires explicit omissions:

```text
Broke out.pred.arg.1 -> Missing
Broke out.pred.arg.2 -> the activities of 1179
```

Reason: the surface order is `Broke out [the activities...]`, not `[the activities...] broke out`. Missing `arg1` does not license reassigning the object.

## Incorrect: Reason Adjunct As `pred.arg.4`

Sentence:

```text
Generators were refusing to sell power in California because of the price caps.
```

Incorrect:

```text
refusing.pred.arg.4 -> because of the price caps
```

Better:

```text
because of.variable -> refusing
because of.body -> the price caps
refusing.pred.arg.4 -> to sell power in California
```

Reason: `because of` expresses a reason adjunct.

## Correct: Temporal Scope Outside Local Role

Sentence:

```text
who resigned as the captain in September 2007
```

Correct hierarchy:

```text
in September 2007.variable -> as the captain
as.variable -> resigned
as.body -> the captain
```

Reason: the date modifies the resignation-as-captain event, not the noun phrase `the captain` alone.

## Audit Method Honesty

If 100 samples are batch-screened and only 16 high-risk samples are deeply reviewed, report:

```text
100 samples screened; 16 samples deeply reviewed; 84 samples not fully semantically audited.
```

Do not report:

```text
100 samples fully audited.
```

## Audit Fix Records Include Corrected Samples

For a fixable problem, do not stop at prose. Include a complete proposed corrected sample.

Original issue:

```text
See.pred.arg.1 -> you
```

Reason: in `See you Sunday.`, `you` is the object, not the omitted subject.

Audit record excerpt:

```json
{
  "line_number": 83,
  "sentence": "See you Sunday.",
  "linguistic_verdict": "major_error",
  "errors": [
    {
      "error_type": "PREDICATE_ARGUMENT_ERROR",
      "severity": "major",
      "location": "edge See -[pred.arg.1]-> you",
      "violated_rule": "R-PRED-02; R-PRED-06",
      "explanation": "`you` is the object of `See`, not the subject.",
      "suggested_fix": "Use `See.pred.arg.2 -> you` and add an implicit subject only if the repair policy requires it.",
      "confidence": 0.93
    }
  ],
  "suggested_fix": "Minimal corrected sample supplied in `proposed_corrected_sample`.",
  "proposed_corrected_sample": {
    "sentence": "See you Sunday.",
    "lingua": {
      "words": ["See", " you", " Sunday", "."],
      "graph": {
        "nodes": [
          {"spans": [[0, 0]], "type": "FactualPredicator"},
          {"spans": [[1, 1]], "type": "NominalConstant"},
          {"spans": [[2, 2]], "type": "NominalConstant"},
          {"spans": [[3, 3]], "type": "PunctuationalConstant"},
          {"spans": "Missing", "type": "NominalConstant"}
        ],
        "edges": [
          {"start": 0, "end": 4, "label": "pred.arg.1"},
          {"start": 0, "end": 1, "label": "pred.arg.2"},
          {"start": 2, "end": 0, "label": "as:modification"},
          {"start": 3, "end": 2, "label": "as:punctuation"}
        ]
      }
    }
  },
  "review_status": "fix_proposed"
}
```
