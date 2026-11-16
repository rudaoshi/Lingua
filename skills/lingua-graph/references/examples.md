# Examples

These examples show selected edges. Follow the standard selected by [SKILL.md](../SKILL.md#required-references) for the complete rules.

## Correct: Predicate With Recoverable Subject

Sentence:

```text
She persuaded him to give private lessons.
```

Expected key structure:

```text
persuaded.pred.arg.1 -> She
persuaded.pred.arg.2 -> him
persuaded.pred.arg.4 -> to give
to give.pred.arg.1 -> him
to give.pred.arg.2 -> private lessons
```

Reason: the persuaded person controls the selected complement, so its predicate's `pred.arg.1` points to `him` (§6.1.1).

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
To/ModificationalFunctor.variable -> suspended
To/ModificationalFunctor.body -> pass
pass.pred.arg.1 -> the state
pass.pred.arg.2 -> the bill
```

Reason: the purpose adjunct modifies the suspension event. The overt `To` anchors the modifier (§6.1.1).

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

## Correct: Pure Header/List Fragment Without Invented Predicate

Sentence:

```text
LATEST PREVIOUS ACTUAL
```

Expected key structure:

```text
List/ListFunctor
  func.arg -> LATEST
  func.arg -> PREVIOUS
  func.arg -> ACTUAL
```

Reason: this is a table/header fragment, not a sentence. It should not be forced into a factual predicate or copula structure.

## Correct: Field / Schedule Fragment

Sentence:

```text
Game 11: Tuesday, July 24 @ 7:30 PM vs. Utah STARZZ
```

Expected key decisions:

```text
Game 11 -> field/key or record identifier
Tuesday, July 24 @ 7:30 PM -> date-time Expression
vs. Utah STARZZ -> opponent/value Expression
```

Reason: this is a structured schedule record. Use `Attribute`/`Expression`/punctuation boundaries as needed; do not invent a normal event predicate unless the text actually states one.

## Correct: Key-Value Metadata Fragment

Sentence:

```text
Location: Houston, Texas USA
```

Expected key structure:

```text
Attribute/AttributePredicator
  spans -> Location
  pred.arg.2 -> Expression(Houston, ,, Texas, USA)
  punctuation -> :
```

Reason: `Location` is the field label and therefore the lexical `AttributePredicator` itself, not `pred.arg.1`. The value is the location expression associated with that field; the external host is not represented with a `Missing` node.

## Correct: Attendance Field Fragment

Sentence:

```text
Attendance : 15,150
```

Expected key structure:

```text
Attribute/AttributePredicator
  spans -> Attendance
  pred.arg.2 -> Expression(15, ,, 150)
  punctuation -> :
```

Reason: `Attendance` is the field label and should be the lexical `AttributePredicator`; `15,150` is its value. Do not create an `Attribute` symbol node and then add `attribute -> Attendance`.

## Correct: Standalone Date-Time Fragment

Sentence:

```text
02/27/2001 08:23 AM
```

Expected key structure:

```text
Expression/ExpressionFunctor
  func.arg -> 02/27/2001
  func.arg -> 08:23
  func.arg -> AM
```

Reason: the date, time, and meridiem are parts of one formatted expression. They should not be represented as a modifier chain.

## Correct: Time Range Fragment

Sentence:

```text
4:00 pm - 6:00 pm
```

Expected key structure:

```text
Expression/ExpressionFunctor
  func.arg -> 4
  func.arg -> :
  func.arg -> 00
  func.arg -> pm
  func.arg -> -
  func.arg -> 6
  func.arg -> :
  func.arg -> 00
  func.arg -> pm
```

Reason: this is a formatted time range. The dash is a range boundary, not a factual predicate.

## Correct: Natural Date Fragment

Sentence:

```text
September 21, 2004
```

Expected key structure:

```text
Expression/ExpressionFunctor
  func.arg -> September
  func.arg -> 21
  func.arg -> ,
  func.arg -> 2004
```

Reason: this is a formatted date expression. The month, day, comma, and year are not semantic modifier layers.

## Correct: Page Reference Fragment

Sentence:

```text
p. 36
```

Expected key structure:

```text
Expression/ExpressionFunctor
  func.arg -> p
  func.arg -> .
  func.arg -> 36
```

Reason: this is one page-reference expression, not a list of parallel fields.

## Correct: Dateline Fragment

Sentence:

```text
LOS ANGELES 1996-12-05
```

Expected key structure:

```text
List/ListFunctor
  func.arg -> LOS ANGELES/Expression
  func.arg -> 1996-12-05/Expression
```

Reason: this is a two-field source/date row. The location and date are sibling metadata fields, not a predicate-argument structure and not a modification relation.

## Correct: Proper Name Fragment Without Copula

Sentence:

```text
Marquis de Condorcet
```

Expected key structure:

```text
Expression/ExpressionFunctor
  func.arg -> Marquis
  func.arg -> de
  func.arg -> Condorcet
```

Reason: the sample is a name, not a proposition. Do not add an implicit `is` or `be named` predicate.

## Correct: Entity With Qualifier

Sentence:

```text
Beetham Tower, Manchester
```

Expected key decision:

```text
Beetham Tower -> entity
Manchester -> location/qualifier attribute or appositive-like qualifier
comma -> local punctuation boundary
```

Reason: comma-separated entity + location is usually a qualified entity, not two unrelated list items.

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

Reason: the surface order is `Broke out [the activities...]`, not `[the activities...] broke out`. A missing `arg1` does not license reassigning the object, and current data must not add a `Missing` node.

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
refusing.pred.arg.2 -> to sell power in California
```

Reason: `because of` introduces the reason for the refusal. The sale is the content event selected by `refusing` (§6.1.1).

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
      "suggested_fix": "Use `See.pred.arg.2 -> you`; leave `pred.arg.1` absent because current data must not add `Missing` nodes.",
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
          {"spans": [[3, 3]], "type": "PunctuationalConstant"}
        ],
        "edges": [
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
