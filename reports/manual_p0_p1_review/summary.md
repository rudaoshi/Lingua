# Manual P0/P1 Review

- Applied fixes: 17
- Reviewed and left unchanged: 26

## Applied Fixes

- `dev:236`: treat list marker as discourse metadata and give imperative an implicit subject — 1 . Be responsible.
- `dev:751`: change got -> so excited from pred.arg.2 to pred.arg.4 — I got so excited!!!!
- `dev:684`: split `mostly` from `deals`; attach it outside `deals with dating advice` — I mostly deals with dating advice
- `train:8106`: make `No` modify `need`, with `to worry` closing over the no-need phrase — No need to worry.
- `test:765`: reattach `until 1997` to the playing-with event instead of `Jura Sud Foot` — He started playing with Jura Sud Foot until 1997 .
- `dev:650`: replace zero-relative Ref and restore `they` as subject of `make` — and this is really the behind the scenes story of everything they had been doing to make this country safer .
- `train:5339`: convert passcode key-value record to Attribute — Passcode: 4153030
- `test:106`: convert job-title key-value record to Attribute — Job Title: Attorney
- `train:8145`: convert place key-value record to Attribute — the place: EB 3143C
- `train:3720`: reattach locative `there` to the seeing event; keep `never` outer — I never saw anyone there.
- `train:4955`: convert hyphen code separator from FactualPredicator to ExpressionFunctor — 5765 - NTKB
- `train:10311`: rewire update header punctuation and parataxis arguments — *** update *** NEVER MIND!
- `train:5824`: make the outer coordination the copular complement list — Diane Stevenett is an artist , singer and actress , and producer .
- `train:8156`: convert leading hyphen code separator from FactualPredicator to ExpressionFunctor — 1691 - EPI - EBS Europe
- `train:5597`: add Missing pred.arg.1 for sentence-initial `But` — But That was Years ago.
- `train:4304`: change leading dateline year from modifier to discourse metadata — 2009 Between November 2009 and February 2010 at least 11 Siberian tiger s died of malnutrition .
- `train:10869`: replace zero-relative Ref with Modification — that ' s the story i try to to tell about the men and women who have brought us this point .

## Reviewed And Left

- `dev:152`: `from Orthalicoidea` modifies the also-removed event by DESC
- `dev:456`: identity clause plus participial modifier is recoverable
- `test:153`: OK and smiley punctuation close a DESC-recoverable clause
- `test:801`: ellipsis after `and nothing` is acceptable; model collapse is not enough to rewrite
- `train:286`: list marker is discourse metadata over a recoverable clause
- `train:650`: phrasal imperative; modifier chain is DESC-recoverable
- `train:687`: already normalized Attribute key-value record
- `train:1143`: `got it to blackline correctly` uses arg4 to a recoverable complement wrapper
- `train:1458`: already normalized Attribute key-value record
- `train:2982`: clausal object of `think` is correct
- `train:3327`: `go hungry` is an acceptable idiomatic predicate-complement structure
- `train:4602`: the non-finite complement and rhythmic/natural modifier remain recoverable
- `train:5965`: title-like key-value fragment is already Attribute
- `train:6059`: question wrapper and purpose infinitive are recoverable
- `train:7668`: `never -> again -> there -> back -> go` matches outer-to-inner scope
- `train:7828`: `going` selects the non-finite complement; `again` wraps that complement by DESC
- `train:8487`: `the last place` is a result complement of `make`
- `train:8908`: `Why not` question wrapper and phrasal verb are recoverable
- `train:9305`: `back to Earth` is a recoverable transmission modifier
- `train:9496`: `below` correctly builds the object `link below` by DESC
- `train:9596`: copula-like `feel soft and soothing` is a lexical predicate choice
- `train:10067`: already normalized Attribute key-value record
- `train:10321`: purpose coordination under `come here` is recoverable
- `train:10486`: `on Thursday` modifies the already-have-plans event by DESC
- `train:10663`: already normalized Attribute key-value record
- `train:11099`: `just` remains an explicit modifier of the simple copular predicate
