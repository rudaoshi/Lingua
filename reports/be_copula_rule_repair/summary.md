# Be/Copula Rule Repair Summary

- mode: apply
- changed samples: 384

## Change Counts

| key | count |
|---|---:|
| dev:merged_intervened_be_prep_split | 1 |
| dev:merged_simple_copula_adverb_split | 15 |
| dev:split_simple_copula_merged | 3 |
| dev:there_be_merged_split | 18 |
| merged_intervened_be_prep_split | 26 |
| merged_simple_copula_adverb_split | 146 |
| split_direct_be_prep_merged | 2 |
| split_simple_copula_merged | 43 |
| test:merged_intervened_be_prep_split | 2 |
| test:merged_simple_copula_adverb_split | 9 |
| test:split_simple_copula_merged | 5 |
| test:there_be_merged_split | 12 |
| there_be_arg4_to_arg2 | 2 |
| there_be_merged_split | 171 |
| train:merged_intervened_be_prep_split | 23 |
| train:merged_simple_copula_adverb_split | 122 |
| train:split_direct_be_prep_merged | 2 |
| train:split_simple_copula_merged | 35 |
| train:there_be_arg4_to_arg2 | 2 |
| train:there_be_merged_split | 141 |

## First 40 Changes

### train:21 `merged_simple_copula_adverb_split`

Karzai was clearly pleased at the results.

```text
Karzai was clearly pleased at the results.
NODES
0: 'at' [[4, 4]] ModificationalFunctor
1: 'Karzai' [[0, 0]] NominalConstant
2: 'results' [[6, 6]] NominalConstant
3: 'the' [[5, 5]] DeterminerConstant
4: '.' [[7, 7]] PunctuationalConstant
5: 'was clearly pleased' [[1, 3]] FactualPredicator
EDGES
0->3 body | 'at'->'the'
0->5 variable | 'at'->'was clearly pleased'
3->2 as:modification | 'the'->'results'
4->0 as:punctuation | '.'->'at'
5->1 pred.arg.1 | 'was clearly pleased'->'Karzai'

--- after ---
Karzai was clearly pleased at the results.
NODES
0: 'at' [[4, 4]] ModificationalFunctor
1: 'Karzai' [[0, 0]] NominalConstant
2: 'results' [[6, 6]] NominalConstant
3: 'the' [[5, 5]] DeterminerConstant
4: '.' [[7, 7]] PunctuationalConstant
5: 'was pleased' [[1, 1], [3, 3]] FactualPredicator
6: 'clearly' [[2, 2]] ModificationalConstant
EDGES
0->3 body | 'at'->'the'
0->6 variable | 'at'->'clearly'
3->2 as:modification | 'the'->'results'
4->0 as:punctuation | '.'->'at'
5->1 pred.arg.1 | 'was pleased'->'Karzai'
6->5 as:modification | 'clearly'->'was pleased'
```

### train:33 `merged_intervened_be_prep_split, merged_simple_copula_adverb_split`

We are very concerned with providing services that are relevant to that environment.

```text
We are very concerned with providing services that are relevant to that environment.
NODES
0: 'with' [[4, 4]] ModificationalFunctor
1: 'providing' [[5, 5]] FactualPredicator
2: 'services' [[6, 6]] NominalConstant
3: 'that' [[7, 7]] GeneralFunctor
4: 'We' [[0, 0]] NominalConstant
5: 'environment' [[12, 12]] NominalConstant
6: 'that' [[11, 11]] DeterminerConstant
7: '.' [[13, 13]] PunctuationalConstant
8: 'are very concerned' [[1, 3]] FactualPredicator
9: 'are relevant to' [[8, 10]] FactualPredicator
10: 'Ref' 'Ref' ReferentialPredicator
11: 'Modification' 'Modification' ModificationalFunctor
EDGES
0->1 body | 'with'->'providing'
0->8 variable | 'with'->'are very concerned'
1->4 pred.arg.1 | 'providing'->'We'
1->10 pred.arg.2 | 'providing'->'Ref'
6->5 as:modification | 'that'->'environment'
7->0 as:punctuation | '.'->'with'
8->4 pred.arg.1 | 'are very concerned'->'We'
9->3 pred.arg.1 | 'are relevant to'->'that'
9->6 pred.arg.2 | 'are relevant to'->'that'
10->2 pred.arg.1 | 'Ref'->'services'
10->11 pred.arg.2 | 'Ref'->'Modification'
11->3 variable | 'Modification'->'that'
11->9 body | 'Modification'->'are relevant to'

--- after ---
We are very concerned with providing services that are relevant to that environment.
NODES
0: 'with' [[4, 4]] ModificationalFunctor
1: 'providing' [[5, 5]] FactualPredicator
2: 'services' [[6, 6]] NominalConstant
3: 'that' [[7, 7]] GeneralFunctor
4: 'We' [[0, 0]] NominalConstant
5: 'environment' [[12, 12]] NominalConstant
6: 'that' [[11, 11]] DeterminerConstant
7: '.' [[13, 13]] PunctuationalConstant
8: 'are concerned' [[1, 1], [3, 3]] FactualPredicator
9: 'are relevant' [[8, 9]] FactualPredicator
10: 'Ref' 'Ref' ReferentialPredicator
11: 'Modification' 'Modification' ModificationalFunctor
12: 'to' [[10, 10]] ModificationalFunctor
13: 'very' [[2, 2]] ModificationalConstant
EDGES
0->1 body | 'with'->'providing'
0->13 variable | 'with'->'very'
1->4 pred.arg.1 | 'providing'->'We'
1->10 pred.arg.2 | 'providing'->'Ref'
6->5 as:modification | 'that'->'environment'
7->0 as:punctuation | '.'->'with'
8->4 pred.arg.1 | 'are concerned'->'We'
9->3 pred.arg.1 | 'are relevant'->'that'
10->2 pred.arg.1 | 'Ref'->'services'
10->11 pred.arg.2 | 'Ref'->'Modification'
11->3 variable | 'Modification'->'that'
11->12 body | 'Modification'->'to'
12->9 variable | 'to'->'are relevant'
12->6 body | 'to'->'that'
13->8 as:modification | 'very'->'are concerned'
```

### train:44 `split_simple_copula_merged`

He said no exact date had been set but it was possible the local elections would take place in the spring of 1997 .

```text
He said no exact date had been set but it was possible the local elections would take place in the spring of 1997 .
NODES
0: 'but' [[8, 8]] LogicalPredicator
1: 'said' [[1, 1]] FactualPredicator
2: 'He' [[0, 0]] NominalConstant
3: 'date' [[4, 4]] NominalConstant
4: 'no' [[2, 2]] DeterminerConstant
5: 'exact' [[3, 3]] ModificationalConstant
6: 'was' [[10, 10]] FactualPredicator
7: 'it' [[9, 9]] NominalConstant
8: 'possible' [[11, 11]] ModificationalConstant
9: 'elections' [[14, 14]] NominalConstant
10: 'the' [[12, 12]] DeterminerConstant
11: 'local' [[13, 13]] ModificationalConstant
12: 'in' [[18, 18]] ModificationalFunctor
13: 'spring' [[20, 20]] NominalConstant
14: 'the' [[19, 19]] DeterminerConstant
15: 'of' [[21, 21]] ModificationalFunctor
16: '1997' [[22, 22]] NominalConstant
17: '.' [[23, 23]] PunctuationalConstant
18: 'had been set' [[5, 7]] FactualPredicator
19: 'would take place' [[15, 17]] FactualPredicator
EDGES
0->1 pred.arg.1 | 'but'->'said'
0->6 pred.arg.2 | 'but'->'was'
1->2 pred.arg.1 | 'said'->'He'
1->18 pred.arg.2 | 'said'->'had been set'
4->5 as:modification | 'no'->'exact'
5->3 as:modification | 'exact'->'date'
6->7 pred.arg.1 | 'was'->'it'
6->8 pred.arg.2 | 'was'->'possible'
6->12 pred.arg.4 | 'was'->'in'
10->11 as:modification | 'the'->'local'
11->9 as:modification | 'local'->'elections'
12->15 body | 'in'->'of'
12->19 variable | 'in'->'would take place'
14->13 as:modification | 'the'->'spring'
15->16 body | 'of'->'1997'
15->14 variable | 'of'->'the'
17->0 as:punctuation | '.'->'but'
18->4 pred.arg.1 | 'had been set'->'no'
19->10 pred.arg.1 | 'would take place'->'the'

--- after ---
He said no exact date had been set but it was possible the local elections would take place in the spring of 1997 .
NODES
0: 'but' [[8, 8]] LogicalPredicator
1: 'said' [[1, 1]] FactualPredicator
2: 'He' [[0, 0]] NominalConstant
3: 'date' [[4, 4]] NominalConstant
4: 'no' [[2, 2]] DeterminerConstant
5: 'exact' [[3, 3]] ModificationalConstant
6: 'was possible' [[10, 11]] FactualPredicator
7: 'it' [[9, 9]] NominalConstant
8: 'elections' [[14, 14]] NominalConstant
9: 'the' [[12, 12]] DeterminerConstant
10: 'local' [[13, 13]] ModificationalConstant
11: 'in' [[18, 18]] ModificationalFunctor
12: 'spring' [[20, 20]] NominalConstant
13: 'the' [[19, 19]] DeterminerConstant
14: 'of' [[21, 21]] ModificationalFunctor
15: '1997' [[22, 22]] NominalConstant
16: '.' [[23, 23]] PunctuationalConstant
17: 'had been set' [[5, 7]] FactualPredicator
18: 'would take place' [[15, 17]] FactualPredicator
EDGES
0->1 pred.arg.1 | 'but'->'said'
0->6 pred.arg.2 | 'but'->'was possible'
1->2 pred.arg.1 | 'said'->'He'
1->17 pred.arg.2 | 'said'->'had been set'
4->5 as:modification | 'no'->'exact'
5->3 as:modification | 'exact'->'date'
6->7 pred.arg.1 | 'was possible'->'it'
6->11 pred.arg.4 | 'was possible'->'in'
9->10 as:modification | 'the'->'local'
10->8 as:modification | 'local'->'elections'
11->14 body | 'in'->'of'
11->18 variable | 'in'->'would take place'
13->12 as:modification | 'the'->'spring'
14->15 body | 'of'->'1997'
14->13 variable | 'of'->'the'
16->0 as:punctuation | '.'->'but'
17->4 pred.arg.1 | 'had been set'->'no'
18->9 pred.arg.1 | 'would take place'->'the'
```

### train:63 `there_be_merged_split`

There were occasional bears on the deck in the morning.

```text
There were occasional bears on the deck in the morning.
NODES
0: 'in' [[7, 7]] ModificationalFunctor
1: 'on' [[4, 4]] ModificationalFunctor
2: 'bears' [[3, 3]] NominalConstant
3: 'occasional' [[2, 2]] ModificationalConstant
4: 'morning' [[9, 9]] NominalConstant
5: 'the' [[8, 8]] DeterminerConstant
6: 'deck' [[6, 6]] NominalConstant
7: 'the' [[5, 5]] DeterminerConstant
8: '.' [[10, 10]] PunctuationalConstant
9: 'There were' [[0, 1]] FactualPredicator
EDGES
0->1 variable | 'in'->'on'
0->5 body | 'in'->'the'
1->7 body | 'on'->'the'
1->9 variable | 'on'->'There were'
3->2 as:modification | 'occasional'->'bears'
5->4 as:modification | 'the'->'morning'
7->6 as:modification | 'the'->'deck'
8->0 as:punctuation | '.'->'in'
9->3 pred.arg.1 | 'There were'->'occasional'

--- after ---
There were occasional bears on the deck in the morning.
NODES
0: 'in' [[7, 7]] ModificationalFunctor
1: 'on' [[4, 4]] ModificationalFunctor
2: 'bears' [[3, 3]] NominalConstant
3: 'occasional' [[2, 2]] ModificationalConstant
4: 'morning' [[9, 9]] NominalConstant
5: 'the' [[8, 8]] DeterminerConstant
6: 'deck' [[6, 6]] NominalConstant
7: 'the' [[5, 5]] DeterminerConstant
8: '.' [[10, 10]] PunctuationalConstant
9: 'were' [[1, 1]] FactualPredicator
10: 'There' [[0, 0]] NominalConstant
EDGES
0->1 variable | 'in'->'on'
0->5 body | 'in'->'the'
1->7 body | 'on'->'the'
1->9 variable | 'on'->'were'
3->2 as:modification | 'occasional'->'bears'
5->4 as:modification | 'the'->'morning'
7->6 as:modification | 'the'->'deck'
8->0 as:punctuation | '.'->'in'
9->3 pred.arg.2 | 'were'->'occasional'
9->10 pred.arg.1 | 'were'->'There'
```

### train:107 `merged_intervened_be_prep_split`

For me it isn't about fulfillment or finding my life's purpose in my work.

```text
For me it isn't about fulfillment or finding my life's purpose in my work.
NODES
0: 'or' [[9, 9]] ConjunctionalFunctor
1: 'For' [[0, 0]] ModificationalFunctor
2: 'finding' [[10, 10]] FactualPredicator
3: 'in' [[16, 16]] ModificationalFunctor
4: 'it' [[2, 2]] NominalConstant
5: 'fulfillment' [[8, 8]] NominalConstant
6: 'me' [[1, 1]] NominalConstant
7: 'purpose' [[15, 15]] NominalConstant
8: "' s" [[13, 14]] ModificationalFunctor
9: 'work' [[18, 18]] NominalConstant
10: 'my' [[17, 17]] NominalConstant
11: 'life' [[12, 12]] NominalConstant
12: 'my' [[11, 11]] NominalConstant
13: '.' [[19, 19]] PunctuationalConstant
14: "is n ' t about" [[3, 7]] FactualPredicator
EDGES
0->5 func.arg | 'or'->'fulfillment'
0->3 func.arg | 'or'->'in'
1->6 body | 'For'->'me'
1->14 variable | 'For'->"is n ' t about"
2->8 pred.arg.2 | 'finding'->"' s"
2->4 pred.arg.1 | 'finding'->'it'
3->2 variable | 'in'->'finding'
3->10 body | 'in'->'my'
8->7 variable | "' s"->'purpose'
8->12 body | "' s"->'my'
10->9 as:modification | 'my'->'work'
12->11 as:modification | 'my'->'life'
13->1 as:punctuation | '.'->'For'
14->0 pred.arg.2 | "is n ' t about"->'or'
14->4 pred.arg.1 | "is n ' t about"->'it'

--- after ---
For me it isn't about fulfillment or finding my life's purpose in my work.
NODES
0: 'or' [[9, 9]] ConjunctionalFunctor
1: 'For' [[0, 0]] ModificationalFunctor
2: 'finding' [[10, 10]] FactualPredicator
3: 'in' [[16, 16]] ModificationalFunctor
4: 'it' [[2, 2]] NominalConstant
5: 'fulfillment' [[8, 8]] NominalConstant
6: 'me' [[1, 1]] NominalConstant
7: 'purpose' [[15, 15]] NominalConstant
8: "' s" [[13, 14]] ModificationalFunctor
9: 'work' [[18, 18]] NominalConstant
10: 'my' [[17, 17]] NominalConstant
11: 'life' [[12, 12]] NominalConstant
12: 'my' [[11, 11]] NominalConstant
13: '.' [[19, 19]] PunctuationalConstant
14: 'is' [[3, 3]] FactualPredicator
15: 'about' [[7, 7]] ModificationalFunctor
16: "n ' t" [[4, 6]] ModificationalConstant
EDGES
0->5 func.arg | 'or'->'fulfillment'
0->3 func.arg | 'or'->'in'
1->6 body | 'For'->'me'
1->15 variable | 'For'->'about'
2->8 pred.arg.2 | 'finding'->"' s"
2->4 pred.arg.1 | 'finding'->'it'
3->2 variable | 'in'->'finding'
3->10 body | 'in'->'my'
8->7 variable | "' s"->'purpose'
8->12 body | "' s"->'my'
10->9 as:modification | 'my'->'work'
12->11 as:modification | 'my'->'life'
13->1 as:punctuation | '.'->'For'
14->4 pred.arg.1 | 'is'->'it'
16->14 as:modification | "n ' t"->'is'
15->16 variable | 'about'->"n ' t"
15->0 body | 'about'->'or'
```

### train:123 `split_simple_copula_merged`

The administration is still studying the new data , and officials have said that if experts think the information is significant , the agency could reopen its investigation .

```text
The administration is still studying the new data , and officials have said that if experts think the information is significant , the agency could reopen its investigation .
NODES
0: 'administration' [[1, 1]] NominalConstant
1: 'The' [[0, 0]] DeterminerConstant
2: 'data' [[7, 7]] NominalConstant
3: 'the' [[5, 5]] DeterminerConstant
4: 'new' [[6, 6]] ModificationalConstant
5: 'officials' [[10, 10]] NominalConstant
6: 'that' [[13, 13]] GeneralFunctor
7: 'agency' [[23, 23]] NominalConstant
8: 'the' [[22, 22]] DeterminerConstant
9: 'investigation' [[27, 27]] NominalConstant
10: 'its' [[26, 26]] ModificationalConstant
11: 'think' [[16, 16]] FactualPredicator
12: 'experts' [[15, 15]] NominalConstant
13: 'is' [[19, 19]] FactualPredicator
14: 'information' [[18, 18]] NominalConstant
15: 'the' [[17, 17]] DeterminerConstant
16: 'significant' [[20, 20]] ModificationalConstant
17: '.' [[28, 28]] PunctuationalConstant
18: ', and' [[8, 9]] ConjunctionalFunctor
19: 'is still studying' [[2, 4]] FactualPredicator
20: 'have said' [[11, 12]] FactualPredicator
21: 'if ,' [[14, 14], [21, 21]] LogicalPredicator
22: 'could reopen' [[24, 25]] FactualPredicator
EDGES
1->0 as:modification | 'The'->'administration'
3->4 as:modification | 'the'->'new'
4->2 as:modification | 'new'->'data'
6->21 func.arg | 'that'->'if ,'
8->7 as:modification | 'the'->'agency'
10->9 as:modification | 'its'->'investigation'
11->12 pred.arg.1 | 'think'->'experts'
11->13 pred.arg.2 | 'think'->'is'
13->16 pred.arg.2 | 'is'->'significant'
13->15 pred.arg.1 | 'is'->'the'
15->14 as:modification | 'the'->'information'
17->18 as:punctuation | '.'->', and'
18->19 func.arg | ', and'->'is still studying'
18->20 func.arg | ', and'->'have said'
19->1 pred.arg.1 | 'is still studying'->'The'
19->3 pred.arg.2 | 'is still studying'->'the'
20->5 pred.arg.1 | 'have said'->'officials'
20->6 pred.arg.2 | 'have said'->'that'
21->11 pred.arg.2 | 'if ,'->'think'
21->22 pred.arg.1 | 'if ,'->'could reopen'
22->8 pred.arg.1 | 'could reopen'->'the'
22->10 pred.arg.2 | 'could reopen'->'its'

--- after ---
The administration is still studying the new data , and officials have said that if experts think the information is significant , the agency could reopen its investigation .
NODES
0: 'administration' [[1, 1]] NominalConstant
1: 'The' [[0, 0]] DeterminerConstant
2: 'data' [[7, 7]] NominalConstant
3: 'the' [[5, 5]] DeterminerConstant
4: 'new' [[6, 6]] ModificationalConstant
5: 'officials' [[10, 10]] NominalConstant
6: 'that' [[13, 13]] GeneralFunctor
7: 'agency' [[23, 23]] NominalConstant
8: 'the' [[22, 22]] DeterminerConstant
9: 'investigation' [[27, 27]] NominalConstant
10: 'its' [[26, 26]] ModificationalConstant
11: 'think' [[16, 16]] FactualPredicator
12: 'experts' [[15, 15]] NominalConstant
13: 'is significant' [[19, 20]] FactualPredicator
14: 'information' [[18, 18]] NominalConstant
15: 'the' [[17, 17]] DeterminerConstant
16: '.' [[28, 28]] PunctuationalConstant
17: ', and' [[8, 9]] ConjunctionalFunctor
18: 'is still studying' [[2, 4]] FactualPredicator
19: 'have said' [[11, 12]] FactualPredicator
20: 'if ,' [[14, 14], [21, 21]] LogicalPredicator
21: 'could reopen' [[24, 25]] FactualPredicator
EDGES
1->0 as:modification | 'The'->'administration'
3->4 as:modification | 'the'->'new'
4->2 as:modification | 'new'->'data'
6->20 func.arg | 'that'->'if ,'
8->7 as:modification | 'the'->'agency'
10->9 as:modification | 'its'->'investigation'
11->12 pred.arg.1 | 'think'->'experts'
11->13 pred.arg.2 | 'think'->'is significant'
13->15 pred.arg.1 | 'is significant'->'the'
15->14 as:modification | 'the'->'information'
16->17 as:punctuation | '.'->', and'
17->18 func.arg | ', and'->'is still studying'
17->19 func.arg | ', and'->'have said'
18->1 pred.arg.1 | 'is still studying'->'The'
18->3 pred.arg.2 | 'is still studying'->'the'
19->5 pred.arg.1 | 'have said'->'officials'
19->6 pred.arg.2 | 'have said'->'that'
20->11 pred.arg.2 | 'if ,'->'think'
20->21 pred.arg.1 | 'if ,'->'could reopen'
21->8 pred.arg.1 | 'could reopen'->'the'
21->10 pred.arg.2 | 'could reopen'->'its'
```

### train:177 `merged_intervened_be_prep_split`

I ca nt speak for them but any tests or appointments they recommend are probably in the best interests of us (the patient) and you have the ability to decline anything that they suggest to you.

```text
I ca nt speak for them but any tests or appointments they recommend are probably in the best interests of us (the patient) and you have the ability to decline anything that they suggest to you.
NODES
0: 'but' [[6, 6]] LogicalPredicator
1: 'and' [[25, 25]] ConjunctionalFunctor
2: 'have' [[27, 27]] FactualPredicator
3: 'for' [[4, 4]] ModificationalFunctor
4: 'recommend' [[12, 12]] FactualPredicator
5: 'anything' [[32, 32]] NominalConstant
6: 'that' [[33, 33]] GeneralFunctor
7: 'suggest' [[35, 35]] FactualPredicator
8: 'to' [[36, 36]] ModificationalFunctor
9: 'I' [[0, 0]] NominalConstant
10: 'you' [[26, 26]] NominalConstant
11: 'them' [[5, 5]] NominalConstant
12: 'they' [[11, 11]] NominalConstant
13: 'they' [[34, 34]] NominalConstant
14: 'you' [[37, 37]] NominalConstant
15: 'tests' [[8, 8]] NominalConstant
16: 'appointments' [[10, 10]] NominalConstant
17: 'any' [[7, 7]] ModificationalConstant
18: 'or' [[9, 9]] ConjunctionalFunctor
19: 'of' [[19, 19]] ModificationalFunctor
20: 'us' [[20, 20]] NominalConstant
21: 'ability' [[29, 29]] NominalConstant
22: 'the' [[28, 28]] DeterminerConstant
23: 'interests' [[18, 18]] NominalConstant
24: 'best' [[17, 17]] ModificationalConstant
25: 'the' [[16, 16]] DeterminerConstant
26: 'patient' [[23, 23]] NominalConstant
27: 'the' [[22, 22]] DeterminerConstant
28: '.' [[38, 38]] PunctuationalConstant
29: 'are probably in' [[13, 15]] FactualPredicator
30: 'ca nt speak' [[1, 3]] FactualPredicator
31: '( )' [[21, 21], [24, 24]] AppositionalPredicator
32: 'to decline' [[30, 31]] FactualPredicator
33: 'Modification' 'Modification' ModificationalFunctor
34: 'Modification' 'Modification' ModificationalFunctor
35: 'Ref' 'Ref' ReferentialPredicator
36: 'Modification' 'Modification' ModificationalFunctor
37: 'Modification' 'Modification' ModificationalFunctor
EDGES
0->1 pred.arg.2 | 'but'->'and'
0->3 pred.arg.1 | 'but'->'for'
1->2 func.arg | 'and'->'have'
1->29 func.arg | 'and'->'are probably in'
2->10 pred.arg.1 | 'have'->'you'
2->33 pred.arg.2 | 'have'->'Modification'
3->11 body | 'for'->'them'
3->30 variable | 'for'->'ca nt speak'
4->12 pred.arg.1 | 'recommend'->'they'
4->18 pred.arg.2 | 'recommend'->'or'
7->13 pred.arg.1 | 'suggest'->'they'
7->6 pred.arg.2 | 'suggest'->'that'
8->14 body | 'to'->'you'
8->7 variable | 'to'->'suggest'
17->15 as:modification | 'any'->'tests'
18->17 func.arg | 'or'->'any'
18->36 func.arg | 'or'->'Modification'
19->20 body | 'of'->'us'
19->25 variable | 'of'->'the'
22->21 as:modification | 'the'->'ability'
24->23 as:modification | 'best'->'interests'
25->24 as:modification | 'the'->'best'
27->26 as:modification | 'the'->'patient'
28->0 as:punctuation | '.'->'but'
29->34 pred.arg.1 | 'are probably in'->'Modification'
29->31 pred.arg.2 | 'are probably in'->'( )'
30->9 pred.arg.1 | 'ca nt speak'->'I'
31->19 pred.arg.1 | '( )'->'of'
31->27 pred.arg.2 | '( )'->'the'
32->10 pred.arg.1 | 'to decline'->'you'
32->35 pred.arg.2 | 'to decline'->'Ref'
33->22 variable | 'Modification'->'the'
33->32 body | 'Modification'->'to decline'
34->18 variable | 'Modification'->'or'
34->4 body | 'Modification'->'recommend'
35->5 pred.arg.1 | 'Ref'->'anything'
35->37 pred.arg.2 | 'Ref'->'Modification'
36->16 variable | 'Modification'->'appointments'
36->17 body | 'Modification'->'any'
37->6 variable | 'Modification'->'that'
37->8 body | 'Modification'->'to'

--- after ---
I ca nt speak for them but any tests or appointments they recommend are probably in the best interests of us (the patient) and you have the ability to decline anything that they suggest to you.
NODES
0: 'but' [[6, 6]] LogicalPredicator
1: 'and' [[25, 25]] ConjunctionalFunctor
2: 'have' [[27, 27]] FactualPredicator
3: 'for' [[4, 4]] ModificationalFunctor
4: 'recommend' [[12, 12]] FactualPredicator
5: 'anything' [[32, 32]] NominalConstant
6: 'that' [[33, 33]] GeneralFunctor
7: 'suggest' [[35, 35]] FactualPredicator
8: 'to' [[36, 36]] ModificationalFunctor
9: 'I' [[0, 0]] NominalConstant
10: 'you' [[26, 26]] NominalConstant
11: 'them' [[5, 5]] NominalConstant
12: 'they' [[11, 11]] NominalConstant
13: 'they' [[34, 34]] NominalConstant
14: 'you' [[37, 37]] NominalConstant
15: 'tests' [[8, 8]] NominalConstant
16: 'appointments' [[10, 10]] NominalConstant
17: 'any' [[7, 7]] ModificationalConstant
18: 'or' [[9, 9]] ConjunctionalFunctor
19: 'of' [[19, 19]] ModificationalFunctor
20: 'us' [[20, 20]] NominalConstant
21: 'ability' [[29, 29]] NominalConstant
22: 'the' [[28, 28]] DeterminerConstant
23: 'interests' [[18, 18]] NominalConstant
24: 'best' [[17, 17]] ModificationalConstant
25: 'the' [[16, 16]] DeterminerConstant
26: 'patient' [[23, 23]] NominalConstant
27: 'the' [[22, 22]] DeterminerConstant
28: '.' [[38, 38]] PunctuationalConstant
29: 'are' [[13, 13]] FactualPredicator
30: 'ca nt speak' [[1, 3]] FactualPredicator
31: '( )' [[21, 21], [24, 24]] AppositionalPredicator
32: 'to decline' [[30, 31]] FactualPredicator
33: 'Modification' 'Modification' ModificationalFunctor
34: 'Modification' 'Modification' ModificationalFunctor
35: 'Ref' 'Ref' ReferentialPredicator
36: 'Modification' 'Modification' ModificationalFunctor
37: 'Modification' 'Modification' ModificationalFunctor
38: 'in' [[15, 15]] ModificationalFunctor
39: 'probably' [[14, 14]] ModificationalConstant
EDGES
0->1 pred.arg.2 | 'but'->'and'
0->3 pred.arg.1 | 'but'->'for'
1->2 func.arg | 'and'->'have'
1->38 func.arg | 'and'->'in'
2->10 pred.arg.1 | 'have'->'you'
2->33 pred.arg.2 | 'have'->'Modification'
3->11 body | 'for'->'them'
3->30 variable | 'for'->'ca nt speak'
4->12 pred.arg.1 | 'recommend'->'they'
4->18 pred.arg.2 | 'recommend'->'or'
7->13 pred.arg.1 | 'suggest'->'they'
7->6 pred.arg.2 | 'suggest'->'that'
8->14 body | 'to'->'you'
8->7 variable | 'to'->'suggest'
17->15 as:modification | 'any'->'tests'
18->17 func.arg | 'or'->'any'
18->36 func.arg | 'or'->'Modification'
19->20 body | 'of'->'us'
19->25 variable | 'of'->'the'
22->21 as:modification | 'the'->'ability'
24->23 as:modification | 'best'->'interests'
25->24 as:modification | 'the'->'best'
27->26 as:modification | 'the'->'patient'
28->0 as:punctuation | '.'->'but'
29->34 pred.arg.1 | 'are'->'Modification'
30->9 pred.arg.1 | 'ca nt speak'->'I'
31->19 pred.arg.1 | '( )'->'of'
31->27 pred.arg.2 | '( )'->'the'
32->10 pred.arg.1 | 'to decline'->'you'
32->35 pred.arg.2 | 'to decline'->'Ref'
33->22 variable | 'Modification'->'the'
33->32 body | 'Modification'->'to decline'
34->18 variable | 'Modification'->'or'
34->4 body | 'Modification'->'recommend'
35->5 pred.arg.1 | 'Ref'->'anything'
35->37 pred.arg.2 | 'Ref'->'Modification'
36->16 variable | 'Modification'->'appointments'
36->17 body | 'Modification'->'any'
37->6 variable | 'Modification'->'that'
37->8 body | 'Modification'->'to'
39->29 as:modification | 'probably'->'are'
38->39 variable | 'in'->'probably'
38->31 body | 'in'->'( )'
```

### train:210 `merged_simple_copula_adverb_split`

Meats are kept VERY cold, seafood smells fresh and the serving bar is VERY clean.

```text
Meats are kept VERY cold, seafood smells fresh and the serving bar is VERY clean.
NODES
0: 'seafood' [[6, 6]] NominalConstant
1: 'Meats' [[0, 0]] NominalConstant
2: 'bar' [[12, 12]] NominalConstant
3: 'serving' [[11, 11]] FactualPredicator
4: 'the' [[10, 10]] DeterminerConstant
5: '.' [[16, 16]] PunctuationalConstant
6: ', and' [[5, 5], [9, 9]] ConjunctionalFunctor
7: 'smells fresh' [[7, 8]] FactualPredicator
8: 'is VERY clean' [[13, 15]] FactualPredicator
9: 'VERY cold' [[3, 4]] ModificationalConstant
10: 'are kept' [[1, 2]] FactualPredicator
11: 'Modification' 'Modification' ModificationalFunctor
EDGES
3->2 pred.arg.1 | 'serving'->'bar'
4->11 as:modification | 'the'->'Modification'
5->6 as:punctuation | '.'->', and'
6->7 func.arg | ', and'->'smells fresh'
6->8 func.arg | ', and'->'is VERY clean'
6->9 func.arg | ', and'->'VERY cold'
7->0 pred.arg.1 | 'smells fresh'->'seafood'
8->4 pred.arg.1 | 'is VERY clean'->'the'
9->10 as:modification | 'VERY cold'->'are kept'
10->1 pred.arg.1 | 'are kept'->'Meats'
11->2 variable | 'Modification'->'bar'
11->3 body | 'Modification'->'serving'

--- after ---
Meats are kept VERY cold, seafood smells fresh and the serving bar is VERY clean.
NODES
0: 'seafood' [[6, 6]] NominalConstant
1: 'Meats' [[0, 0]] NominalConstant
2: 'bar' [[12, 12]] NominalConstant
3: 'serving' [[11, 11]] FactualPredicator
4: 'the' [[10, 10]] DeterminerConstant
5: '.' [[16, 16]] PunctuationalConstant
6: ', and' [[5, 5], [9, 9]] ConjunctionalFunctor
7: 'smells fresh' [[7, 8]] FactualPredicator
8: 'is clean' [[13, 13], [15, 15]] FactualPredicator
9: 'VERY cold' [[3, 4]] ModificationalConstant
10: 'are kept' [[1, 2]] FactualPredicator
11: 'Modification' 'Modification' ModificationalFunctor
12: 'VERY' [[14, 14]] ModificationalConstant
EDGES
3->2 pred.arg.1 | 'serving'->'bar'
4->11 as:modification | 'the'->'Modification'
5->6 as:punctuation | '.'->', and'
6->7 func.arg | ', and'->'smells fresh'
6->12 func.arg | ', and'->'VERY'
6->9 func.arg | ', and'->'VERY cold'
7->0 pred.arg.1 | 'smells fresh'->'seafood'
8->4 pred.arg.1 | 'is clean'->'the'
9->10 as:modification | 'VERY cold'->'are kept'
10->1 pred.arg.1 | 'are kept'->'Meats'
11->2 variable | 'Modification'->'bar'
11->3 body | 'Modification'->'serving'
12->8 as:modification | 'VERY'->'is clean'
```

### train:236 `there_be_merged_split`

Is there any cure for mouse poison?

```text
Is there any cure for mouse poison?
NODES
0: 'for' [[4, 4]] ModificationalFunctor
1: 'cure' [[3, 3]] NominalConstant
2: 'any' [[2, 2]] DeterminerConstant
3: 'poison' [[6, 6]] NominalConstant
4: 'mouse' [[5, 5]] NominalConstant
5: '?' [[7, 7]] PunctuationalConstant
6: 'Is there' [[1, 1], [0, 0]] FactualPredicator
7: 'Whether' 'Whether' GeneralFunctor
EDGES
0->2 variable | 'for'->'any'
0->4 body | 'for'->'mouse'
2->1 as:modification | 'any'->'cure'
4->3 as:modification | 'mouse'->'poison'
5->7 as:punctuation | '?'->'Whether'
6->0 pred.arg.1 | 'Is there'->'for'
7->6 func.arg | 'Whether'->'Is there'

--- after ---
Is there any cure for mouse poison?
NODES
0: 'for' [[4, 4]] ModificationalFunctor
1: 'cure' [[3, 3]] NominalConstant
2: 'any' [[2, 2]] DeterminerConstant
3: 'poison' [[6, 6]] NominalConstant
4: 'mouse' [[5, 5]] NominalConstant
5: '?' [[7, 7]] PunctuationalConstant
6: 'Is' [[0, 0]] FactualPredicator
7: 'Whether' 'Whether' GeneralFunctor
8: 'there' [[1, 1]] NominalConstant
EDGES
0->2 variable | 'for'->'any'
0->4 body | 'for'->'mouse'
2->1 as:modification | 'any'->'cure'
4->3 as:modification | 'mouse'->'poison'
5->7 as:punctuation | '?'->'Whether'
6->0 pred.arg.2 | 'Is'->'for'
7->6 func.arg | 'Whether'->'Is'
6->8 pred.arg.1 | 'Is'->'there'
```

### train:267 `split_simple_copula_merged`

The opposition motion against Banharn accuses him of being incompetent , lacking ethical leadership and alleges his administration is corrupt .

```text
The opposition motion against Banharn accuses him of being incompetent , lacking ethical leadership and alleges his administration is corrupt .
NODES
0: 'and' [[14, 14]] ConjunctionalFunctor
1: 'accuses' [[5, 5]] FactualPredicator
2: 'motion' [[2, 2]] NominalConstant
3: 'The' [[0, 0]] DeterminerConstant
4: 'opposition' [[1, 1]] ModificationalConstant
5: 'against' [[3, 3]] ModificationalFunctor
6: 'Banharn' [[4, 4]] NominalConstant
7: 'him' [[6, 6]] NominalConstant
8: 'of' [[7, 7]] ModificationalFunctor
9: ',' [[10, 10]] ConjunctionalFunctor
10: 'being' [[8, 8]] FactualPredicator
11: 'incompetent' [[9, 9]] ModificationalConstant
12: 'lacking' [[11, 11]] FactualPredicator
13: 'leadership' [[13, 13]] NominalConstant
14: 'ethical' [[12, 12]] ModificationalConstant
15: 'alleges' [[15, 15]] FactualPredicator
16: 'is' [[18, 18]] FactualPredicator
17: 'administration' [[17, 17]] NominalConstant
18: 'his' [[16, 16]] ModificationalConstant
19: 'corrupt' [[19, 19]] ModificationalConstant
20: '.' [[20, 20]] PunctuationalConstant
EDGES
0->15 func.arg | 'and'->'alleges'
0->8 func.arg | 'and'->'of'
1->7 pred.arg.2 | 'accuses'->'him'
1->5 pred.arg.1 | 'accuses'->'against'
3->4 as:modification | 'The'->'opposition'
4->2 as:modification | 'opposition'->'motion'
5->6 body | 'against'->'Banharn'
5->3 variable | 'against'->'The'
8->9 body | 'of'->','
8->1 variable | 'of'->'accuses'
9->10 func.arg | ','->'being'
9->12 func.arg | ','->'lacking'
10->11 pred.arg.2 | 'being'->'incompetent'
10->7 pred.arg.1 | 'being'->'him'
12->14 pred.arg.2 | 'lacking'->'ethical'
12->7 pred.arg.1 | 'lacking'->'him'
14->13 as:modification | 'ethical'->'leadership'
15->16 pred.arg.2 | 'alleges'->'is'
15->5 pred.arg.1 | 'alleges'->'against'
16->19 pred.arg.2 | 'is'->'corrupt'
16->18 pred.arg.1 | 'is'->'his'
18->17 as:modification | 'his'->'administration'
20->0 as:punctuation | '.'->'and'

--- after ---
The opposition motion against Banharn accuses him of being incompetent , lacking ethical leadership and alleges his administration is corrupt .
NODES
0: 'and' [[14, 14]] ConjunctionalFunctor
1: 'accuses' [[5, 5]] FactualPredicator
2: 'motion' [[2, 2]] NominalConstant
3: 'The' [[0, 0]] DeterminerConstant
4: 'opposition' [[1, 1]] ModificationalConstant
5: 'against' [[3, 3]] ModificationalFunctor
6: 'Banharn' [[4, 4]] NominalConstant
7: 'him' [[6, 6]] NominalConstant
8: 'of' [[7, 7]] ModificationalFunctor
9: ',' [[10, 10]] ConjunctionalFunctor
10: 'being' [[8, 8]] FactualPredicator
11: 'incompetent' [[9, 9]] ModificationalConstant
12: 'lacking' [[11, 11]] FactualPredicator
13: 'leadership' [[13, 13]] NominalConstant
14: 'ethical' [[12, 12]] ModificationalConstant
15: 'alleges' [[15, 15]] FactualPredicator
16: 'is corrupt' [[18, 19]] FactualPredicator
17: 'administration' [[17, 17]] NominalConstant
18: 'his' [[16, 16]] ModificationalConstant
19: '.' [[20, 20]] PunctuationalConstant
EDGES
0->15 func.arg | 'and'->'alleges'
0->8 func.arg | 'and'->'of'
1->7 pred.arg.2 | 'accuses'->'him'
1->5 pred.arg.1 | 'accuses'->'against'
3->4 as:modification | 'The'->'opposition'
4->2 as:modification | 'opposition'->'motion'
5->6 body | 'against'->'Banharn'
5->3 variable | 'against'->'The'
8->9 body | 'of'->','
8->1 variable | 'of'->'accuses'
9->10 func.arg | ','->'being'
9->12 func.arg | ','->'lacking'
10->11 pred.arg.2 | 'being'->'incompetent'
10->7 pred.arg.1 | 'being'->'him'
12->14 pred.arg.2 | 'lacking'->'ethical'
12->7 pred.arg.1 | 'lacking'->'him'
14->13 as:modification | 'ethical'->'leadership'
15->16 pred.arg.2 | 'alleges'->'is corrupt'
15->5 pred.arg.1 | 'alleges'->'against'
16->18 pred.arg.1 | 'is corrupt'->'his'
18->17 as:modification | 'his'->'administration'
19->0 as:punctuation | '.'->'and'
```

### train:271 `merged_simple_copula_adverb_split`

I am SO GLAD to have found Dr. Liau.

```text
I am SO GLAD to have found Dr. Liau.
NODES
0: 'I' [[0, 0]] NominalConstant
1: 'Liau' [[9, 9]] NominalConstant
2: 'Dr .' [[7, 8]] NominalConstant
3: '.' [[10, 10]] PunctuationalConstant
4: 'am SO GLAD' [[1, 3]] FactualPredicator
5: 'to have found' [[4, 6]] FactualPredicator
EDGES
2->1 as:modification | 'Dr .'->'Liau'
3->4 as:punctuation | '.'->'am SO GLAD'
4->0 pred.arg.1 | 'am SO GLAD'->'I'
4->5 pred.arg.4 | 'am SO GLAD'->'to have found'
5->0 pred.arg.1 | 'to have found'->'I'
5->2 pred.arg.2 | 'to have found'->'Dr .'

--- after ---
I am SO GLAD to have found Dr. Liau.
NODES
0: 'I' [[0, 0]] NominalConstant
1: 'Liau' [[9, 9]] NominalConstant
2: 'Dr .' [[7, 8]] NominalConstant
3: '.' [[10, 10]] PunctuationalConstant
4: 'am GLAD' [[1, 1], [3, 3]] FactualPredicator
5: 'to have found' [[4, 6]] FactualPredicator
6: 'SO' [[2, 2]] ModificationalConstant
EDGES
2->1 as:modification | 'Dr .'->'Liau'
3->6 as:punctuation | '.'->'SO'
4->0 pred.arg.1 | 'am GLAD'->'I'
4->5 pred.arg.4 | 'am GLAD'->'to have found'
5->0 pred.arg.1 | 'to have found'->'I'
5->2 pred.arg.2 | 'to have found'->'Dr .'
6->4 as:modification | 'SO'->'am GLAD'
```

### train:286 `split_simple_copula_merged`

#1 it s immaculately clean.

```text
#1 it s immaculately clean.
NODES
0: 'it' [[2, 2]] NominalConstant
1: '#' [[0, 0]] SymbolConstant
2: '1' [[1, 1]] NominalConstant
3: '.' [[6, 6]] PunctuationalConstant
4: 's' [[3, 3]] FactualPredicator
5: 'immaculately' [[4, 4]] ModificationalConstant
6: 'clean' [[5, 5]] ModificationalConstant
7: 'Expression' 'Expression' ExpressionFunctor
EDGES
7->1 func.arg | 'Expression'->'#'
7->2 func.arg | 'Expression'->'1'
4->0 pred.arg.1 | 's'->'it'
5->6 as:modification | 'immaculately'->'clean'
4->5 pred.arg.2 | 's'->'immaculately'
7->4 as:discourse | 'Expression'->'s'
3->7 as:punctuation | '.'->'Expression'

--- after ---
#1 it s immaculately clean.
NODES
0: 'it' [[2, 2]] NominalConstant
1: '#' [[0, 0]] SymbolConstant
2: '1' [[1, 1]] NominalConstant
3: '.' [[6, 6]] PunctuationalConstant
4: 's clean' [[3, 3], [5, 5]] FactualPredicator
5: 'immaculately' [[4, 4]] ModificationalConstant
6: 'Expression' 'Expression' ExpressionFunctor
EDGES
6->1 func.arg | 'Expression'->'#'
6->2 func.arg | 'Expression'->'1'
4->0 pred.arg.1 | 's clean'->'it'
5->4 as:modification | 'immaculately'->'s clean'
6->5 as:discourse | 'Expression'->'immaculately'
3->6 as:punctuation | '.'->'Expression'
```

### train:289 `there_be_merged_split`

(Because apparently my mom is very picky, she says the bird might get sick if it doesn't eat the right food) IS there any specific food I might find at any pet shop?

```text
(Because apparently my mom is very picky, she says the bird might get sick if it doesn't eat the right food) IS there any specific food I might find at any pet shop?
NODES
0: 'says' [[10, 10]] FactualPredicator
1: 'Because' [[1, 1]] ModificationalFunctor
2: 'if' [[16, 16]] LogicalPredicator
3: 'apparently' [[2, 2]] ModificationalConstant
4: 'at' [[35, 35]] ModificationalFunctor
5: 'she' [[9, 9]] NominalConstant
6: 'it' [[17, 17]] NominalConstant
7: 'I' [[32, 32]] NominalConstant
8: 'food' [[31, 31]] NominalConstant
9: 'specific' [[30, 30]] ModificationalConstant
10: 'any' [[29, 29]] DeterminerConstant
11: 'bird' [[12, 12]] NominalConstant
12: 'the' [[11, 11]] DeterminerConstant
13: 'mom' [[4, 4]] NominalConstant
14: 'my' [[3, 3]] NominalConstant
15: 'food' [[25, 25]] NominalConstant
16: 'right' [[24, 24]] ModificationalConstant
17: 'the' [[23, 23]] DeterminerConstant
18: 'shop' [[38, 38]] NominalConstant
19: 'pet' [[37, 37]] ModificationalConstant
20: 'any' [[36, 36]] DeterminerConstant
21: ',' [[8, 8]] PunctuationalConstant
22: '?' [[39, 39]] PunctuationalConstant
23: '( )' [[0, 0], [26, 26]] PunctuationalConstant
24: 'IS there' [[27, 28]] FactualPredicator
25: 'might find' [[33, 34]] FactualPredicator
26: 'is very picky' [[5, 7]] FactualPredicator
27: 'might get sick' [[13, 15]] FactualPredicator
28: "does n ' t eat" [[18, 22]] FactualPredicator
29: 'Parataxis' 'Parataxis' ConjunctionalFunctor
30: 'Modification' 'Modification' ModificationalFunctor
31: 'Whether' 'Whether' GeneralFunctor
32: 'Modification' 'Modification' ModificationalFunctor
EDGES
0->5 pred.arg.1 | 'says'->'she'
0->2 pred.arg.2 | 'says'->'if'
1->0 variable | 'Because'->'says'
1->21 body | 'Because'->','
2->27 pred.arg.1 | 'if'->'might get sick'
2->28 pred.arg.2 | 'if'->"does n ' t eat"
3->1 as:modification | 'apparently'->'Because'
4->20 body | 'at'->'any'
4->25 variable | 'at'->'might find'
9->8 as:modification | 'specific'->'food'
10->9 as:modification | 'any'->'specific'
12->11 as:modification | 'the'->'bird'
14->13 as:modification | 'my'->'mom'
16->15 as:modification | 'right'->'food'
17->16 as:modification | 'the'->'right'
19->18 as:modification | 'pet'->'shop'
20->19 as:modification | 'any'->'pet'
21->26 as:punctuation | ','->'is very picky'
22->29 as:punctuation | '?'->'Parataxis'
23->30 as:punctuation | '( )'->'Modification'
24->32 pred.arg.1 | 'IS there'->'Modification'
25->7 pred.arg.1 | 'might find'->'I'
25->10 pred.arg.2 | 'might find'->'any'
26->14 pred.arg.1 | 'is very picky'->'my'
27->12 pred.arg.1 | 'might get sick'->'the'
28->6 pred.arg.1 | "does n ' t eat"->'it'
28->17 pred.arg.2 | "does n ' t eat"->'the'
29->23 func.arg | 'Parataxis'->'( )'
29->31 func.arg | 'Parataxis'->'Whether'
30->0 variable | 'Modification'->'says'
30->3 body | 'Modification'->'apparently'
31->24 func.arg | 'Whether'->'IS there'
32->10 variable | 'Modification'->'any'
32->4 body | 'Modification'->'at'

--- after ---
(Because apparently my mom is very picky, she says the bird might get sick if it doesn't eat the right food) IS there any specific food I might find at any pet shop?
NODES
0: 'says' [[10, 10]] FactualPredicator
1: 'Because' [[1, 1]] ModificationalFunctor
2: 'if' [[16, 16]] LogicalPredicator
3: 'apparently' [[2, 2]] ModificationalConstant
4: 'at' [[35, 35]] ModificationalFunctor
5: 'she' [[9, 9]] NominalConstant
6: 'it' [[17, 17]] NominalConstant
7: 'I' [[32, 32]] NominalConstant
8: 'food' [[31, 31]] NominalConstant
9: 'specific' [[30, 30]] ModificationalConstant
10: 'any' [[29, 29]] DeterminerConstant
11: 'bird' [[12, 12]] NominalConstant
12: 'the' [[11, 11]] DeterminerConstant
13: 'mom' [[4, 4]] NominalConstant
14: 'my' [[3, 3]] NominalConstant
15: 'food' [[25, 25]] NominalConstant
16: 'right' [[24, 24]] ModificationalConstant
17: 'the' [[23, 23]] DeterminerConstant
18: 'shop' [[38, 38]] NominalConstant
19: 'pet' [[37, 37]] ModificationalConstant
20: 'any' [[36, 36]] DeterminerConstant
21: ',' [[8, 8]] PunctuationalConstant
22: '?' [[39, 39]] PunctuationalConstant
23: '( )' [[0, 0], [26, 26]] PunctuationalConstant
24: 'IS' [[27, 27]] FactualPredicator
25: 'might find' [[33, 34]] FactualPredicator
26: 'is very picky' [[5, 7]] FactualPredicator
27: 'might get sick' [[13, 15]] FactualPredicator
28: "does n ' t eat" [[18, 22]] FactualPredicator
29: 'Parataxis' 'Parataxis' ConjunctionalFunctor
30: 'Modification' 'Modification' ModificationalFunctor
31: 'Whether' 'Whether' GeneralFunctor
32: 'Modification' 'Modification' ModificationalFunctor
33: 'there' [[28, 28]] NominalConstant
EDGES
0->5 pred.arg.1 | 'says'->'she'
0->2 pred.arg.2 | 'says'->'if'
1->0 variable | 'Because'->'says'
1->21 body | 'Because'->','
2->27 pred.arg.1 | 'if'->'might get sick'
2->28 pred.arg.2 | 'if'->"does n ' t eat"
3->1 as:modification | 'apparently'->'Because'
4->20 body | 'at'->'any'
4->25 variable | 'at'->'might find'
9->8 as:modification | 'specific'->'food'
10->9 as:modification | 'any'->'specific'
12->11 as:modification | 'the'->'bird'
14->13 as:modification | 'my'->'mom'
16->15 as:modification | 'right'->'food'
17->16 as:modification | 'the'->'right'
19->18 as:modification | 'pet'->'shop'
20->19 as:modification | 'any'->'pet'
21->26 as:punctuation | ','->'is very picky'
22->29 as:punctuation | '?'->'Parataxis'
23->30 as:punctuation | '( )'->'Modification'
24->32 pred.arg.2 | 'IS'->'Modification'
25->7 pred.arg.1 | 'might find'->'I'
25->10 pred.arg.2 | 'might find'->'any'
26->14 pred.arg.1 | 'is very picky'->'my'
27->12 pred.arg.1 | 'might get sick'->'the'
28->6 pred.arg.1 | "does n ' t eat"->'it'
28->17 pred.arg.2 | "does n ' t eat"->'the'
29->23 func.arg | 'Parataxis'->'( )'
29->31 func.arg | 'Parataxis'->'Whether'
30->0 variable | 'Modification'->'says'
30->3 body | 'Modification'->'apparently'
31->24 func.arg | 'Whether'->'IS'
32->10 variable | 'Modification'->'any'
32->4 body | 'Modification'->'at'
24->33 pred.arg.1 | 'IS'->'there'
```

### train:323 `merged_simple_copula_adverb_split`

The repair went quickly and the price was extremely fair.

```text
The repair went quickly and the price was extremely fair.
NODES
0: 'and' [[4, 4]] ConjunctionalFunctor
1: 'repair' [[1, 1]] NominalConstant
2: 'The' [[0, 0]] DeterminerConstant
3: 'price' [[6, 6]] NominalConstant
4: 'the' [[5, 5]] DeterminerConstant
5: '.' [[10, 10]] PunctuationalConstant
6: 'went quickly' [[2, 3]] FactualPredicator
7: 'was extremely fair' [[7, 9]] FactualPredicator
EDGES
0->6 func.arg | 'and'->'went quickly'
0->7 func.arg | 'and'->'was extremely fair'
2->1 as:modification | 'The'->'repair'
4->3 as:modification | 'the'->'price'
5->0 as:punctuation | '.'->'and'
6->2 pred.arg.1 | 'went quickly'->'The'
7->4 pred.arg.1 | 'was extremely fair'->'the'

--- after ---
The repair went quickly and the price was extremely fair.
NODES
0: 'and' [[4, 4]] ConjunctionalFunctor
1: 'repair' [[1, 1]] NominalConstant
2: 'The' [[0, 0]] DeterminerConstant
3: 'price' [[6, 6]] NominalConstant
4: 'the' [[5, 5]] DeterminerConstant
5: '.' [[10, 10]] PunctuationalConstant
6: 'went quickly' [[2, 3]] FactualPredicator
7: 'was fair' [[7, 7], [9, 9]] FactualPredicator
8: 'extremely' [[8, 8]] ModificationalConstant
EDGES
0->6 func.arg | 'and'->'went quickly'
0->8 func.arg | 'and'->'extremely'
2->1 as:modification | 'The'->'repair'
4->3 as:modification | 'the'->'price'
5->0 as:punctuation | '.'->'and'
6->2 pred.arg.1 | 'went quickly'->'The'
7->4 pred.arg.1 | 'was fair'->'the'
8->7 as:modification | 'extremely'->'was fair'
```

### train:326 `merged_intervened_be_prep_split`

I'm not even in Iraq and I could have predicted to you the consequences of doing what the CPA has been doing.

```text
I'm not even in Iraq and I could have predicted to you the consequences of doing what the CPA has been doing.
NODES
0: 'and' [[7, 7]] ConjunctionalFunctor
1: 'to' [[12, 12]] ModificationalFunctor
2: 'doing' [[17, 17]] FactualPredicator
3: 'what' [[18, 18]] GeneralFunctor
4: 'I' [[8, 8]] NominalConstant
5: 'I' [[0, 0]] NominalConstant
6: 'Iraq' [[6, 6]] NominalConstant
7: 'you' [[13, 13]] NominalConstant
8: 'of' [[16, 16]] ModificationalFunctor
9: 'consequences' [[15, 15]] NominalConstant
10: 'the' [[14, 14]] DeterminerConstant
11: 'CPA' [[20, 20]] NominalConstant
12: 'the' [[19, 19]] DeterminerConstant
13: '.' [[24, 24]] PunctuationalConstant
14: "' m not even in" [[1, 5]] FactualPredicator
15: 'could have predicted' [[9, 11]] FactualPredicator
16: 'has been doing' [[21, 23]] FactualPredicator
17: 'Modification' 'Modification' ModificationalFunctor
EDGES
0->1 func.arg | 'and'->'to'
0->14 func.arg | 'and'->"' m not even in"
1->7 body | 'to'->'you'
1->15 variable | 'to'->'could have predicted'
2->17 pred.arg.2 | 'doing'->'Modification'
8->2 body | 'of'->'doing'
8->10 variable | 'of'->'the'
10->9 as:modification | 'the'->'consequences'
12->11 as:modification | 'the'->'CPA'
13->0 as:punctuation | '.'->'and'
14->5 pred.arg.1 | "' m not even in"->'I'
14->6 pred.arg.2 | "' m not even in"->'Iraq'
15->4 pred.arg.1 | 'could have predicted'->'I'
15->8 pred.arg.2 | 'could have predicted'->'of'
16->3 pred.arg.2 | 'has been doing'->'what'
16->12 pred.arg.1 | 'has been doing'->'the'
17->3 variable | 'Modification'->'what'
17->16 body | 'Modification'->'has been doing'

--- after ---
I'm not even in Iraq and I could have predicted to you the consequences of doing what the CPA has been doing.
NODES
0: 'and' [[7, 7]] ConjunctionalFunctor
1: 'to' [[12, 12]] ModificationalFunctor
2: 'doing' [[17, 17]] FactualPredicator
3: 'what' [[18, 18]] GeneralFunctor
4: 'I' [[8, 8]] NominalConstant
5: 'I' [[0, 0]] NominalConstant
6: 'Iraq' [[6, 6]] NominalConstant
7: 'you' [[13, 13]] NominalConstant
8: 'of' [[16, 16]] ModificationalFunctor
9: 'consequences' [[15, 15]] NominalConstant
10: 'the' [[14, 14]] DeterminerConstant
11: 'CPA' [[20, 20]] NominalConstant
12: 'the' [[19, 19]] DeterminerConstant
13: '.' [[24, 24]] PunctuationalConstant
14: "' m" [[1, 2]] FactualPredicator
15: 'could have predicted' [[9, 11]] FactualPredicator
16: 'has been doing' [[21, 23]] FactualPredicator
17: 'Modification' 'Modification' ModificationalFunctor
18: 'in' [[5, 5]] ModificationalFunctor
19: 'even' [[4, 4]] ModificationalConstant
20: 'not' [[3, 3]] ModificationalConstant
EDGES
0->1 func.arg | 'and'->'to'
0->18 func.arg | 'and'->'in'
1->7 body | 'to'->'you'
1->15 variable | 'to'->'could have predicted'
2->17 pred.arg.2 | 'doing'->'Modification'
8->2 body | 'of'->'doing'
8->10 variable | 'of'->'the'
10->9 as:modification | 'the'->'consequences'
12->11 as:modification | 'the'->'CPA'
13->0 as:punctuation | '.'->'and'
14->5 pred.arg.1 | "' m"->'I'
15->4 pred.arg.1 | 'could have predicted'->'I'
15->8 pred.arg.2 | 'could have predicted'->'of'
16->3 pred.arg.2 | 'has been doing'->'what'
16->12 pred.arg.1 | 'has been doing'->'the'
17->3 variable | 'Modification'->'what'
17->16 body | 'Modification'->'has been doing'
19->14 as:modification | 'even'->"' m"
20->19 as:modification | 'not'->'even'
18->20 variable | 'in'->'not'
18->6 body | 'in'->'Iraq'
```

### train:403 `there_be_merged_split`

+ there is a free cola with every pizza.

```text
+ there is a free cola with every pizza.
NODES
0: 'with' [[6, 6]] ModificationalFunctor
1: 'cola' [[5, 5]] NominalConstant
2: 'free' [[4, 4]] ModificationalConstant
3: 'a' [[3, 3]] DeterminerConstant
4: 'pizza' [[8, 8]] NominalConstant
5: 'every' [[7, 7]] DeterminerConstant
6: '+' [[0, 0]] PunctuationalConstant
7: '.' [[9, 9]] PunctuationalConstant
8: 'there is' [[1, 2]] FactualPredicator
EDGES
0->5 body | 'with'->'every'
0->8 variable | 'with'->'there is'
2->1 as:modification | 'free'->'cola'
3->2 as:modification | 'a'->'free'
5->4 as:modification | 'every'->'pizza'
6->0 as:punctuation | '+'->'with'
7->6 as:punctuation | '.'->'+'
8->3 pred.arg.1 | 'there is'->'a'

--- after ---
+ there is a free cola with every pizza.
NODES
0: 'with' [[6, 6]] ModificationalFunctor
1: 'cola' [[5, 5]] NominalConstant
2: 'free' [[4, 4]] ModificationalConstant
3: 'a' [[3, 3]] DeterminerConstant
4: 'pizza' [[8, 8]] NominalConstant
5: 'every' [[7, 7]] DeterminerConstant
6: '+' [[0, 0]] PunctuationalConstant
7: '.' [[9, 9]] PunctuationalConstant
8: 'is' [[2, 2]] FactualPredicator
9: 'there' [[1, 1]] NominalConstant
EDGES
0->5 body | 'with'->'every'
0->8 variable | 'with'->'is'
2->1 as:modification | 'free'->'cola'
3->2 as:modification | 'a'->'free'
5->4 as:modification | 'every'->'pizza'
6->0 as:punctuation | '+'->'with'
7->6 as:punctuation | '.'->'+'
8->3 pred.arg.2 | 'is'->'a'
8->9 pred.arg.1 | 'is'->'there'
```

### train:417 `there_be_merged_split`

There are many online sites offering the booking facility with affordable rates.

```text
There are many online sites offering the booking facility with affordable rates.
NODES
0: 'offering' [[5, 5]] FactualPredicator
1: 'with' [[9, 9]] ModificationalFunctor
2: 'sites' [[4, 4]] NominalConstant
3: 'many' [[2, 2]] ModificationalConstant
4: 'online' [[3, 3]] ModificationalConstant
5: 'facility' [[8, 8]] NominalConstant
6: 'booking' [[7, 7]] NominalConstant
7: 'the' [[6, 6]] DeterminerConstant
8: 'rates' [[11, 11]] NominalConstant
9: 'affordable' [[10, 10]] ModificationalConstant
10: '.' [[12, 12]] PunctuationalConstant
11: 'There are' [[0, 1]] FactualPredicator
12: 'Modification' 'Modification' ModificationalFunctor
EDGES
0->3 pred.arg.1 | 'offering'->'many'
0->1 pred.arg.2 | 'offering'->'with'
1->7 variable | 'with'->'the'
1->9 body | 'with'->'affordable'
3->4 as:modification | 'many'->'online'
4->2 as:modification | 'online'->'sites'
6->5 as:modification | 'booking'->'facility'
7->6 as:modification | 'the'->'booking'
9->8 as:modification | 'affordable'->'rates'
10->11 as:punctuation | '.'->'There are'
11->12 pred.arg.1 | 'There are'->'Modification'
12->3 variable | 'Modification'->'many'
12->0 body | 'Modification'->'offering'

--- after ---
There are many online sites offering the booking facility with affordable rates.
NODES
0: 'offering' [[5, 5]] FactualPredicator
1: 'with' [[9, 9]] ModificationalFunctor
2: 'sites' [[4, 4]] NominalConstant
3: 'many' [[2, 2]] ModificationalConstant
4: 'online' [[3, 3]] ModificationalConstant
5: 'facility' [[8, 8]] NominalConstant
6: 'booking' [[7, 7]] NominalConstant
7: 'the' [[6, 6]] DeterminerConstant
8: 'rates' [[11, 11]] NominalConstant
9: 'affordable' [[10, 10]] ModificationalConstant
10: '.' [[12, 12]] PunctuationalConstant
11: 'are' [[1, 1]] FactualPredicator
12: 'Modification' 'Modification' ModificationalFunctor
13: 'There' [[0, 0]] NominalConstant
EDGES
0->3 pred.arg.1 | 'offering'->'many'
0->1 pred.arg.2 | 'offering'->'with'
1->7 variable | 'with'->'the'
1->9 body | 'with'->'affordable'
3->4 as:modification | 'many'->'online'
4->2 as:modification | 'online'->'sites'
6->5 as:modification | 'booking'->'facility'
7->6 as:modification | 'the'->'booking'
9->8 as:modification | 'affordable'->'rates'
10->11 as:punctuation | '.'->'are'
11->12 pred.arg.2 | 'are'->'Modification'
12->3 variable | 'Modification'->'many'
12->0 body | 'Modification'->'offering'
11->13 pred.arg.1 | 'are'->'There'
```

### train:587 `merged_intervened_be_prep_split`

It is all about this very subject...bars and food.

```text
It is all about this very subject...bars and food.
NODES
0: 'It' [[0, 0]] NominalConstant
1: 'and' [[9, 9]] ConjunctionalFunctor
2: 'bars' [[8, 8]] NominalConstant
3: 'food' [[10, 10]] NominalConstant
4: '...' [[7, 7]] PunctuationalConstant
5: '.' [[11, 11]] PunctuationalConstant
6: 'is all about' [[1, 3]] FactualPredicator
7: 'this very subject' [[4, 6]] NominalConstant
8: 'Apposition' 'Apposition' AppositionalPredicator
EDGES
1->2 func.arg | 'and'->'bars'
1->3 func.arg | 'and'->'food'
4->7 as:punctuation | '...'->'this very subject'
5->6 as:punctuation | '.'->'is all about'
6->0 pred.arg.1 | 'is all about'->'It'
6->8 pred.arg.2 | 'is all about'->'Apposition'
8->4 pred.arg.1 | 'Apposition'->'...'
8->1 pred.arg.2 | 'Apposition'->'and'

--- after ---
It is all about this very subject...bars and food.
NODES
0: 'It' [[0, 0]] NominalConstant
1: 'and' [[9, 9]] ConjunctionalFunctor
2: 'bars' [[8, 8]] NominalConstant
3: 'food' [[10, 10]] NominalConstant
4: '...' [[7, 7]] PunctuationalConstant
5: '.' [[11, 11]] PunctuationalConstant
6: 'is all' [[1, 2]] FactualPredicator
7: 'this very subject' [[4, 6]] NominalConstant
8: 'Apposition' 'Apposition' AppositionalPredicator
9: 'about' [[3, 3]] ModificationalFunctor
EDGES
1->2 func.arg | 'and'->'bars'
1->3 func.arg | 'and'->'food'
4->7 as:punctuation | '...'->'this very subject'
5->9 as:punctuation | '.'->'about'
6->0 pred.arg.1 | 'is all'->'It'
8->4 pred.arg.1 | 'Apposition'->'...'
8->1 pred.arg.2 | 'Apposition'->'and'
9->6 variable | 'about'->'is all'
9->8 body | 'about'->'Apposition'
```

### train:623 `there_be_merged_split`

We pointed out to the PX that there was excess credit.

```text
We pointed out to the PX that there was excess credit.
NODES
0: 'to' [[3, 3]] ModificationalFunctor
1: 'We' [[0, 0]] NominalConstant
2: 'that' [[6, 6]] GeneralFunctor
3: 'credit' [[10, 10]] NominalConstant
4: 'excess' [[9, 9]] ModificationalConstant
5: 'PX' [[5, 5]] NominalConstant
6: 'the' [[4, 4]] DeterminerConstant
7: '.' [[11, 11]] PunctuationalConstant
8: 'pointed out' [[1, 2]] FactualPredicator
9: 'there was' [[7, 8]] FactualPredicator
EDGES
0->6 body | 'to'->'the'
0->8 variable | 'to'->'pointed out'
2->9 func.arg | 'that'->'there was'
4->3 as:modification | 'excess'->'credit'
6->5 as:modification | 'the'->'PX'
7->0 as:punctuation | '.'->'to'
8->1 pred.arg.1 | 'pointed out'->'We'
8->2 pred.arg.2 | 'pointed out'->'that'
9->4 pred.arg.1 | 'there was'->'excess'

--- after ---
We pointed out to the PX that there was excess credit.
NODES
0: 'to' [[3, 3]] ModificationalFunctor
1: 'We' [[0, 0]] NominalConstant
2: 'that' [[6, 6]] GeneralFunctor
3: 'credit' [[10, 10]] NominalConstant
4: 'excess' [[9, 9]] ModificationalConstant
5: 'PX' [[5, 5]] NominalConstant
6: 'the' [[4, 4]] DeterminerConstant
7: '.' [[11, 11]] PunctuationalConstant
8: 'pointed out' [[1, 2]] FactualPredicator
9: 'was' [[8, 8]] FactualPredicator
10: 'there' [[7, 7]] NominalConstant
EDGES
0->6 body | 'to'->'the'
0->8 variable | 'to'->'pointed out'
2->9 func.arg | 'that'->'was'
4->3 as:modification | 'excess'->'credit'
6->5 as:modification | 'the'->'PX'
7->0 as:punctuation | '.'->'to'
8->1 pred.arg.1 | 'pointed out'->'We'
8->2 pred.arg.2 | 'pointed out'->'that'
9->4 pred.arg.2 | 'was'->'excess'
9->10 pred.arg.1 | 'was'->'there'
```

### train:626 `merged_simple_copula_adverb_split`

Christie , who is retiring from international competition at the end of the season , was not due to compete in the German capital but Berlin promoter Rudi Thiel said : " We are still hopeful of getting him to come . "

```text
Christie , who is retiring from international competition at the end of the season , was not due to compete in the German capital but Berlin promoter Rudi Thiel said : " We are still hopeful of getting him to come . "
NODES
0: 'but' [[24, 24]] LogicalPredicator
1: 'Christie' [[0, 0]] NominalConstant
2: 'who' [[2, 2]] GeneralFunctor
3: 'from' [[5, 5]] ModificationalFunctor
4: 'competition' [[7, 7]] NominalConstant
5: 'international' [[6, 6]] ModificationalConstant
6: 'at' [[8, 8]] ModificationalFunctor
7: 'end' [[10, 10]] NominalConstant
8: 'the' [[9, 9]] DeterminerConstant
9: 'of' [[11, 11]] ModificationalFunctor
10: 'season' [[13, 13]] NominalConstant
11: 'the' [[12, 12]] DeterminerConstant
12: 'in' [[20, 20]] ModificationalFunctor
13: 'capital' [[23, 23]] NominalConstant
14: 'the' [[21, 21]] DeterminerConstant
15: 'German' [[22, 22]] ModificationalConstant
16: 'said' [[29, 29]] FactualPredicator
17: 'Thiel' [[28, 28]] NominalConstant
18: 'Rudi' [[27, 27]] NominalConstant
19: 'promoter' [[26, 26]] NominalConstant
20: 'Berlin' [[25, 25]] NominalConstant
21: 'We' [[32, 32]] NominalConstant
22: 'of' [[36, 36]] ModificationalFunctor
23: 'getting' [[37, 37]] FactualPredicator
24: 'him' [[38, 38]] NominalConstant
25: '.' [[41, 41]] PunctuationalConstant
26: 'was not due' [[15, 17]] FactualPredicator
27: ': " "' [[30, 31], [42, 42]] PunctuationalConstant
28: ', ,' [[1, 1], [14, 14]] PunctuationalConstant
29: 'are still hopeful' [[33, 35]] FactualPredicator
30: 'to compete' [[18, 19]] FactualPredicator
31: 'to come' [[39, 40]] FactualPredicator
32: 'is retiring' [[3, 4]] FactualPredicator
33: 'Modification' 'Modification' ModificationalFunctor
34: 'Expression' 'Expression' ExpressionFunctor
35: 'Ref' 'Ref' ReferentialPredicator
36: 'Modification' 'Modification' ModificationalFunctor
EDGES
0->16 pred.arg.2 | 'but'->'said'
0->26 pred.arg.1 | 'but'->'was not due'
3->5 body | 'from'->'international'
3->32 variable | 'from'->'is retiring'
5->4 as:modification | 'international'->'competition'
6->3 variable | 'at'->'from'
6->9 body | 'at'->'of'
8->7 as:modification | 'the'->'end'
9->8 variable | 'of'->'the'
9->11 body | 'of'->'the'
11->10 as:modification | 'the'->'season'
12->14 body | 'in'->'the'
12->30 variable | 'in'->'to compete'
14->15 as:modification | 'the'->'German'
15->13 as:modification | 'German'->'capital'
16->22 pred.arg.2 | 'said'->'of'
16->27 punctuation | 'said'->': " "'
16->33 pred.arg.1 | 'said'->'Modification'
20->19 as:modification | 'Berlin'->'promoter'
22->23 body | 'of'->'getting'
22->29 variable | 'of'->'are still hopeful'
23->24 pred.arg.2 | 'getting'->'him'
23->21 pred.arg.1 | 'getting'->'We'
23->31 pred.arg.4 | 'getting'->'to come'
25->0 as:punctuation | '.'->'but'
26->12 pred.arg.4 | 'was not due'->'in'
26->35 pred.arg.1 | 'was not due'->'Ref'
28->1 as:punctuation | ', ,'->'Christie'
29->21 pred.arg.1 | 'are still hopeful'->'We'
30->1 pred.arg.1 | 'to compete'->'Christie'
31->24 pred.arg.1 | 'to come'->'him'
32->2 pred.arg.1 | 'is retiring'->'who'
33->20 body | 'Modification'->'Berlin'
33->34 variable | 'Modification'->'Expression'
34->18 func.arg | 'Expression'->'Rudi'
34->17 func.arg | 'Expression'->'Thiel'
35->28 pred.arg.1 | 'Ref'->', ,'
35->36 pred.arg.2 | 'Ref'->'Modification'
36->2 variable | 'Modification'->'who'
36->6 body | 'Modification'->'at'

--- after ---
Christie , who is retiring from international competition at the end of the season , was not due to compete in the German capital but Berlin promoter Rudi Thiel said : " We are still hopeful of getting him to come . "
NODES
0: 'but' [[24, 24]] LogicalPredicator
1: 'Christie' [[0, 0]] NominalConstant
2: 'who' [[2, 2]] GeneralFunctor
3: 'from' [[5, 5]] ModificationalFunctor
4: 'competition' [[7, 7]] NominalConstant
5: 'international' [[6, 6]] ModificationalConstant
6: 'at' [[8, 8]] ModificationalFunctor
7: 'end' [[10, 10]] NominalConstant
8: 'the' [[9, 9]] DeterminerConstant
9: 'of' [[11, 11]] ModificationalFunctor
10: 'season' [[13, 13]] NominalConstant
11: 'the' [[12, 12]] DeterminerConstant
12: 'in' [[20, 20]] ModificationalFunctor
13: 'capital' [[23, 23]] NominalConstant
14: 'the' [[21, 21]] DeterminerConstant
15: 'German' [[22, 22]] ModificationalConstant
16: 'said' [[29, 29]] FactualPredicator
17: 'Thiel' [[28, 28]] NominalConstant
18: 'Rudi' [[27, 27]] NominalConstant
19: 'promoter' [[26, 26]] NominalConstant
20: 'Berlin' [[25, 25]] NominalConstant
21: 'We' [[32, 32]] NominalConstant
22: 'of' [[36, 36]] ModificationalFunctor
23: 'getting' [[37, 37]] FactualPredicator
24: 'him' [[38, 38]] NominalConstant
25: '.' [[41, 41]] PunctuationalConstant
26: 'was not due' [[15, 17]] FactualPredicator
27: ': " "' [[30, 31], [42, 42]] PunctuationalConstant
28: ', ,' [[1, 1], [14, 14]] PunctuationalConstant
29: 'are hopeful' [[33, 33], [35, 35]] FactualPredicator
30: 'to compete' [[18, 19]] FactualPredicator
31: 'to come' [[39, 40]] FactualPredicator
32: 'is retiring' [[3, 4]] FactualPredicator
33: 'Modification' 'Modification' ModificationalFunctor
34: 'Expression' 'Expression' ExpressionFunctor
35: 'Ref' 'Ref' ReferentialPredicator
36: 'Modification' 'Modification' ModificationalFunctor
37: 'still' [[34, 34]] ModificationalConstant
EDGES
0->16 pred.arg.2 | 'but'->'said'
0->26 pred.arg.1 | 'but'->'was not due'
3->5 body | 'from'->'international'
3->32 variable | 'from'->'is retiring'
5->4 as:modification | 'international'->'competition'
6->3 variable | 'at'->'from'
6->9 body | 'at'->'of'
8->7 as:modification | 'the'->'end'
9->8 variable | 'of'->'the'
9->11 body | 'of'->'the'
11->10 as:modification | 'the'->'season'
12->14 body | 'in'->'the'
12->30 variable | 'in'->'to compete'
14->15 as:modification | 'the'->'German'
15->13 as:modification | 'German'->'capital'
16->22 pred.arg.2 | 'said'->'of'
16->27 punctuation | 'said'->': " "'
16->33 pred.arg.1 | 'said'->'Modification'
20->19 as:modification | 'Berlin'->'promoter'
22->23 body | 'of'->'getting'
22->37 variable | 'of'->'still'
23->24 pred.arg.2 | 'getting'->'him'
23->21 pred.arg.1 | 'getting'->'We'
23->31 pred.arg.4 | 'getting'->'to come'
25->0 as:punctuation | '.'->'but'
26->12 pred.arg.4 | 'was not due'->'in'
26->35 pred.arg.1 | 'was not due'->'Ref'
28->1 as:punctuation | ', ,'->'Christie'
29->21 pred.arg.1 | 'are hopeful'->'We'
30->1 pred.arg.1 | 'to compete'->'Christie'
31->24 pred.arg.1 | 'to come'->'him'
32->2 pred.arg.1 | 'is retiring'->'who'
33->20 body | 'Modification'->'Berlin'
33->34 variable | 'Modification'->'Expression'
34->18 func.arg | 'Expression'->'Rudi'
34->17 func.arg | 'Expression'->'Thiel'
35->28 pred.arg.1 | 'Ref'->', ,'
35->36 pred.arg.2 | 'Ref'->'Modification'
36->2 variable | 'Modification'->'who'
36->6 body | 'Modification'->'at'
37->29 as:modification | 'still'->'are hopeful'
```

### train:647 `split_simple_copula_merged`

Taken together, our results indicate that Ran GTP is important for the targeting of export complexes to the cytoplasmic side of the NPC and that RanBP1 and probably RanBP2 are involved in the dissociation of nuclear export complexes from the NPC in a terminal step of transport .

```text
Taken together, our results indicate that Ran GTP is important for the targeting of export complexes to the cytoplasmic side of the NPC and that RanBP1 and probably RanBP2 are involved in the dissociation of nuclear export complexes from the NPC in a terminal step of transport .
NODES
0: 'Taken' [[0, 0]] FactualPredicator
1: 'together' [[1, 1]] ModificationalConstant
2: 'indicate' [[5, 5]] FactualPredicator
3: 'results' [[4, 4]] NominalConstant
4: 'our' [[3, 3]] ModificationalConstant
5: 'and' [[24, 24]] ConjunctionalFunctor
6: 'that' [[6, 6]] GeneralFunctor
7: 'is' [[9, 9]] FactualPredicator
8: 'GTP' [[8, 8]] NominalConstant
9: 'Ran' [[7, 7]] NominalConstant
10: 'important' [[10, 10]] ModificationalConstant
11: 'for' [[11, 11]] ModificationalFunctor
12: 'targeting' [[13, 13]] NominalConstant
13: 'the' [[12, 12]] DeterminerConstant
14: 'of' [[14, 14]] ModificationalFunctor
15: 'complexes' [[16, 16]] NominalConstant
16: 'export' [[15, 15]] NominalConstant
17: 'to' [[17, 17]] ModificationalFunctor
18: 'side' [[20, 20]] NominalConstant
19: 'the' [[22, 22]] DeterminerConstant
20: 'cytoplasmic' [[19, 19]] ModificationalConstant
21: 'of' [[21, 21]] ModificationalFunctor
22: 'NPC' [[43, 43]] NominalConstant
23: 'the' [[35, 35]] DeterminerConstant
24: 'that' [[25, 25]] GeneralFunctor
25: 'and' [[28, 28]] ConjunctionalFunctor
26: 'RanBP 1' [[26, 27]] NominalConstant
27: 'RanBP 2' [[30, 31]] NominalConstant
28: 'probably' [[29, 29]] ModificationalConstant
29: 'in' [[34, 34]] ModificationalFunctor
30: 'dissociation' [[36, 36]] NominalConstant
31: 'the' [[18, 18]] DeterminerConstant
32: 'of' [[37, 37]] ModificationalFunctor
33: 'complexes' [[40, 40]] NominalConstant
34: 'nuclear' [[38, 38]] ModificationalConstant
35: 'export' [[39, 39]] NominalConstant
36: 'from' [[41, 41]] ModificationalFunctor
37: 'NPC' [[23, 23]] NominalConstant
38: 'the' [[42, 42]] DeterminerConstant
39: 'in' [[44, 44]] ModificationalFunctor
40: 'step' [[47, 47]] NominalConstant
41: 'a' [[45, 45]] DeterminerConstant
42: 'terminal' [[46, 46]] ModificationalConstant
43: 'of' [[48, 48]] ModificationalFunctor
44: 'transport' [[49, 49]] NominalConstant
45: ',' [[2, 2]] PunctuationalConstant
46: '.' [[50, 50]] PunctuationalConstant
47: 'are involved' [[32, 33]] FactualPredicator
48: 'Parataxis' 'Parataxis' ConjunctionalFunctor
EDGES
0->4 pred.arg.1 | 'Taken'->'our'
1->0 as:modification | 'together'->'Taken'
2->5 pred.arg.2 | 'indicate'->'and'
2->4 pred.arg.1 | 'indicate'->'our'
4->3 as:modification | 'our'->'results'
5->6 func.arg | 'and'->'that'
5->24 func.arg | 'and'->'that'
6->11 func.arg | 'that'->'for'
7->10 pred.arg.2 | 'is'->'important'
7->9 pred.arg.1 | 'is'->'Ran'
9->8 as:modification | 'Ran'->'GTP'
11->7 variable | 'for'->'is'
11->14 body | 'for'->'of'
13->12 as:modification | 'the'->'targeting'
14->13 variable | 'of'->'the'
14->17 body | 'of'->'to'
16->15 as:modification | 'export'->'complexes'
17->16 variable | 'to'->'export'
17->19 body | 'to'->'the'
19->21 as:modification | 'the'->'of'
20->18 as:modification | 'cytoplasmic'->'side'
21->20 variable | 'of'->'cytoplasmic'
21->23 body | 'of'->'the'
23->22 as:modification | 'the'->'NPC'
24->29 func.arg | 'that'->'in'
25->26 func.arg | 'and'->'RanBP 1'
25->28 func.arg | 'and'->'probably'
28->27 as:modification | 'probably'->'RanBP 2'
29->31 body | 'in'->'the'
29->47 variable | 'in'->'are involved'
31->32 as:modification | 'the'->'of'
32->30 variable | 'of'->'dissociation'
32->34 body | 'of'->'nuclear'
34->35 as:modification | 'nuclear'->'export'
35->36 as:modification | 'export'->'from'
36->33 variable | 'from'->'complexes'
36->39 body | 'from'->'in'
38->37 as:modification | 'the'->'NPC'
39->38 variable | 'in'->'the'
39->43 body | 'in'->'of'
41->42 as:modification | 'a'->'terminal'
42->40 as:modification | 'terminal'->'step'
43->44 body | 'of'->'transport'
43->41 variable | 'of'->'a'
45->1 as:punctuation | ','->'together'
46->48 as:punctuation | '.'->'Parataxis'
47->25 pred.arg.1 | 'are involved'->'and'
48->45 func.arg | 'Parataxis'->','
48->2 func.arg | 'Parataxis'->'indicate'

--- after ---
Taken together, our results indicate that Ran GTP is important for the targeting of export complexes to the cytoplasmic side of the NPC and that RanBP1 and probably RanBP2 are involved in the dissociation of nuclear export complexes from the NPC in a terminal step of transport .
NODES
0: 'Taken' [[0, 0]] FactualPredicator
1: 'together' [[1, 1]] ModificationalConstant
2: 'indicate' [[5, 5]] FactualPredicator
3: 'results' [[4, 4]] NominalConstant
4: 'our' [[3, 3]] ModificationalConstant
5: 'and' [[24, 24]] ConjunctionalFunctor
6: 'that' [[6, 6]] GeneralFunctor
7: 'is important' [[9, 10]] FactualPredicator
8: 'GTP' [[8, 8]] NominalConstant
9: 'Ran' [[7, 7]] NominalConstant
10: 'for' [[11, 11]] ModificationalFunctor
11: 'targeting' [[13, 13]] NominalConstant
12: 'the' [[12, 12]] DeterminerConstant
13: 'of' [[14, 14]] ModificationalFunctor
14: 'complexes' [[16, 16]] NominalConstant
15: 'export' [[15, 15]] NominalConstant
16: 'to' [[17, 17]] ModificationalFunctor
17: 'side' [[20, 20]] NominalConstant
18: 'the' [[22, 22]] DeterminerConstant
19: 'cytoplasmic' [[19, 19]] ModificationalConstant
20: 'of' [[21, 21]] ModificationalFunctor
21: 'NPC' [[43, 43]] NominalConstant
22: 'the' [[35, 35]] DeterminerConstant
23: 'that' [[25, 25]] GeneralFunctor
24: 'and' [[28, 28]] ConjunctionalFunctor
25: 'RanBP 1' [[26, 27]] NominalConstant
26: 'RanBP 2' [[30, 31]] NominalConstant
27: 'probably' [[29, 29]] ModificationalConstant
28: 'in' [[34, 34]] ModificationalFunctor
29: 'dissociation' [[36, 36]] NominalConstant
30: 'the' [[18, 18]] DeterminerConstant
31: 'of' [[37, 37]] ModificationalFunctor
32: 'complexes' [[40, 40]] NominalConstant
33: 'nuclear' [[38, 38]] ModificationalConstant
34: 'export' [[39, 39]] NominalConstant
35: 'from' [[41, 41]] ModificationalFunctor
36: 'NPC' [[23, 23]] NominalConstant
37: 'the' [[42, 42]] DeterminerConstant
38: 'in' [[44, 44]] ModificationalFunctor
39: 'step' [[47, 47]] NominalConstant
40: 'a' [[45, 45]] DeterminerConstant
41: 'terminal' [[46, 46]] ModificationalConstant
42: 'of' [[48, 48]] ModificationalFunctor
43: 'transport' [[49, 49]] NominalConstant
44: ',' [[2, 2]] PunctuationalConstant
45: '.' [[50, 50]] PunctuationalConstant
46: 'are involved' [[32, 33]] FactualPredicator
47: 'Parataxis' 'Parataxis' ConjunctionalFunctor
EDGES
0->4 pred.arg.1 | 'Taken'->'our'
1->0 as:modification | 'together'->'Taken'
2->5 pred.arg.2 | 'indicate'->'and'
2->4 pred.arg.1 | 'indicate'->'our'
4->3 as:modification | 'our'->'results'
5->6 func.arg | 'and'->'that'
5->23 func.arg | 'and'->'that'
6->10 func.arg | 'that'->'for'
7->9 pred.arg.1 | 'is important'->'Ran'
9->8 as:modification | 'Ran'->'GTP'
10->7 variable | 'for'->'is important'
10->13 body | 'for'->'of'
12->11 as:modification | 'the'->'targeting'
13->12 variable | 'of'->'the'
13->16 body | 'of'->'to'
15->14 as:modification | 'export'->'complexes'
16->15 variable | 'to'->'export'
16->18 body | 'to'->'the'
18->20 as:modification | 'the'->'of'
19->17 as:modification | 'cytoplasmic'->'side'
20->19 variable | 'of'->'cytoplasmic'
20->22 body | 'of'->'the'
22->21 as:modification | 'the'->'NPC'
23->28 func.arg | 'that'->'in'
24->25 func.arg | 'and'->'RanBP 1'
24->27 func.arg | 'and'->'probably'
27->26 as:modification | 'probably'->'RanBP 2'
28->30 body | 'in'->'the'
28->46 variable | 'in'->'are involved'
30->31 as:modification | 'the'->'of'
31->29 variable | 'of'->'dissociation'
31->33 body | 'of'->'nuclear'
33->34 as:modification | 'nuclear'->'export'
34->35 as:modification | 'export'->'from'
35->32 variable | 'from'->'complexes'
35->38 body | 'from'->'in'
37->36 as:modification | 'the'->'NPC'
38->37 variable | 'in'->'the'
38->42 body | 'in'->'of'
40->41 as:modification | 'a'->'terminal'
41->39 as:modification | 'terminal'->'step'
42->43 body | 'of'->'transport'
42->40 variable | 'of'->'a'
44->1 as:punctuation | ','->'together'
45->47 as:punctuation | '.'->'Parataxis'
46->24 pred.arg.1 | 'are involved'->'and'
47->44 func.arg | 'Parataxis'->','
47->2 func.arg | 'Parataxis'->'indicate'
```

### train:690 `there_be_merged_split`

There are no other requirements other than we ask that members greet each new member and each new alliance and also feel free to post in any of the forums.

```text
There are no other requirements other than we ask that members greet each new member and each new alliance and also feel free to post in any of the forums.
NODES
0: 'ask' [[8, 8]] FactualPredicator
1: 'greet' [[11, 11]] FactualPredicator
2: 'in' [[25, 25]] ModificationalFunctor
3: 'we' [[7, 7]] NominalConstant
4: 'members' [[10, 10]] NominalConstant
5: 'that' [[9, 9]] GeneralFunctor
6: 'of' [[27, 27]] ModificationalFunctor
7: 'any' [[26, 26]] NominalConstant
8: 'requirements' [[4, 4]] NominalConstant
9: 'other' [[3, 3]] ModificationalConstant
10: 'no' [[2, 2]] DeterminerConstant
11: 'forums' [[29, 29]] NominalConstant
12: 'the' [[28, 28]] DeterminerConstant
13: 'member' [[14, 14]] NominalConstant
14: 'new' [[13, 13]] ModificationalConstant
15: 'alliance' [[18, 18]] NominalConstant
16: 'new' [[17, 17]] ModificationalConstant
17: 'each' [[16, 16]] DeterminerConstant
18: '.' [[30, 30]] PunctuationalConstant
19: 'There are' [[0, 1]] FactualPredicator
20: 'other than' [[5, 6]] LogicalPredicator
21: 'and also' [[19, 20]] ConjunctionalFunctor
22: 'feel free' [[21, 22]] FactualPredicator
23: 'each and' [[12, 12], [15, 15]] ConjunctionalFunctor
24: 'to post' [[23, 24]] FactualPredicator
25: 'Modification' 'Modification' ModificationalFunctor
EDGES
18->19 as:punctuation | '.'->'There are'
19->25 pred.arg.1 | 'There are'->'Modification'
25->10 variable | 'Modification'->'no'
25->20 body | 'Modification'->'other than'
20->10 pred.arg.1 | 'other than'->'no'
20->0 pred.arg.2 | 'other than'->'ask'
10->9 as:modification | 'no'->'other'
9->8 as:modification | 'other'->'requirements'
0->3 pred.arg.1 | 'ask'->'we'
0->5 pred.arg.2 | 'ask'->'that'
5->21 func.arg | 'that'->'and also'
21->1 func.arg | 'and also'->'greet'
21->22 func.arg | 'and also'->'feel free'
1->4 pred.arg.1 | 'greet'->'members'
1->23 pred.arg.2 | 'greet'->'each and'
23->14 func.arg | 'each and'->'new'
23->17 func.arg | 'each and'->'each'
14->13 as:modification | 'new'->'member'
17->16 as:modification | 'each'->'new'
16->15 as:modification | 'new'->'alliance'
22->4 pred.arg.1 | 'feel free'->'members'
22->2 pred.arg.4 | 'feel free'->'in'
2->24 variable | 'in'->'to post'
2->6 body | 'in'->'of'
24->4 pred.arg.1 | 'to post'->'members'
6->7 variable | 'of'->'any'
6->12 body | 'of'->'the'
12->11 as:modification | 'the'->'forums'

--- after ---
There are no other requirements other than we ask that members greet each new member and each new alliance and also feel free to post in any of the forums.
NODES
0: 'ask' [[8, 8]] FactualPredicator
1: 'greet' [[11, 11]] FactualPredicator
2: 'in' [[25, 25]] ModificationalFunctor
3: 'we' [[7, 7]] NominalConstant
4: 'members' [[10, 10]] NominalConstant
5: 'that' [[9, 9]] GeneralFunctor
6: 'of' [[27, 27]] ModificationalFunctor
7: 'any' [[26, 26]] NominalConstant
8: 'requirements' [[4, 4]] NominalConstant
9: 'other' [[3, 3]] ModificationalConstant
10: 'no' [[2, 2]] DeterminerConstant
11: 'forums' [[29, 29]] NominalConstant
12: 'the' [[28, 28]] DeterminerConstant
13: 'member' [[14, 14]] NominalConstant
14: 'new' [[13, 13]] ModificationalConstant
15: 'alliance' [[18, 18]] NominalConstant
16: 'new' [[17, 17]] ModificationalConstant
17: 'each' [[16, 16]] DeterminerConstant
18: '.' [[30, 30]] PunctuationalConstant
19: 'are' [[1, 1]] FactualPredicator
20: 'other than' [[5, 6]] LogicalPredicator
21: 'and also' [[19, 20]] ConjunctionalFunctor
22: 'feel free' [[21, 22]] FactualPredicator
23: 'each and' [[12, 12], [15, 15]] ConjunctionalFunctor
24: 'to post' [[23, 24]] FactualPredicator
25: 'Modification' 'Modification' ModificationalFunctor
26: 'There' [[0, 0]] NominalConstant
EDGES
18->19 as:punctuation | '.'->'are'
19->25 pred.arg.2 | 'are'->'Modification'
25->10 variable | 'Modification'->'no'
25->20 body | 'Modification'->'other than'
20->10 pred.arg.1 | 'other than'->'no'
20->0 pred.arg.2 | 'other than'->'ask'
10->9 as:modification | 'no'->'other'
9->8 as:modification | 'other'->'requirements'
0->3 pred.arg.1 | 'ask'->'we'
0->5 pred.arg.2 | 'ask'->'that'
5->21 func.arg | 'that'->'and also'
21->1 func.arg | 'and also'->'greet'
21->22 func.arg | 'and also'->'feel free'
1->4 pred.arg.1 | 'greet'->'members'
1->23 pred.arg.2 | 'greet'->'each and'
23->14 func.arg | 'each and'->'new'
23->17 func.arg | 'each and'->'each'
14->13 as:modification | 'new'->'member'
17->16 as:modification | 'each'->'new'
16->15 as:modification | 'new'->'alliance'
22->4 pred.arg.1 | 'feel free'->'members'
22->2 pred.arg.4 | 'feel free'->'in'
2->24 variable | 'in'->'to post'
2->6 body | 'in'->'of'
24->4 pred.arg.1 | 'to post'->'members'
6->7 variable | 'of'->'any'
6->12 body | 'of'->'the'
12->11 as:modification | 'the'->'forums'
19->26 pred.arg.1 | 'are'->'There'
```

### train:717 `there_be_merged_split`

There are some things to consider though ...

```text
There are some things to consider though ...
NODES
0: 'though' [[6, 6]] LogicalPredicator
1: 'things' [[3, 3]] NominalConstant
2: 'some' [[2, 2]] DeterminerConstant
3: '...' [[7, 7]] PunctuationalConstant
4: 'There are' [[0, 1]] FactualPredicator
5: 'to consider' [[4, 5]] FactualPredicator
6: 'Modification' 'Modification' ModificationalFunctor
EDGES
0->4 pred.arg.2 | 'though'->'There are'
2->1 as:modification | 'some'->'things'
3->0 as:punctuation | '...'->'though'
4->6 pred.arg.1 | 'There are'->'Modification'
5->2 pred.arg.2 | 'to consider'->'some'
6->2 variable | 'Modification'->'some'
6->5 body | 'Modification'->'to consider'

--- after ---
There are some things to consider though ...
NODES
0: 'though' [[6, 6]] LogicalPredicator
1: 'things' [[3, 3]] NominalConstant
2: 'some' [[2, 2]] DeterminerConstant
3: '...' [[7, 7]] PunctuationalConstant
4: 'are' [[1, 1]] FactualPredicator
5: 'to consider' [[4, 5]] FactualPredicator
6: 'Modification' 'Modification' ModificationalFunctor
7: 'There' [[0, 0]] NominalConstant
EDGES
0->4 pred.arg.2 | 'though'->'are'
2->1 as:modification | 'some'->'things'
3->0 as:punctuation | '...'->'though'
4->6 pred.arg.2 | 'are'->'Modification'
5->2 pred.arg.2 | 'to consider'->'some'
6->2 variable | 'Modification'->'some'
6->5 body | 'Modification'->'to consider'
4->7 pred.arg.1 | 'are'->'There'
```

### train:719 `split_simple_copula_merged`

New Yorkers know that this was a huge battle over support for Israel and whether the support for Palestine was strong enough .

```text
New Yorkers know that this was a huge battle over support for Israel and whether the support for Palestine was strong enough .
NODES
0: 'know' [[2, 2]] FactualPredicator
1: 'Yorkers' [[1, 1]] NominalConstant
2: 'New' [[0, 0]] ModificationalConstant
3: 'that' [[3, 3]] GeneralFunctor
4: 'and' [[13, 13]] ConjunctionalFunctor
5: 'was' [[5, 5]] FactualPredicator
6: 'this' [[4, 4]] NominalConstant
7: 'battle' [[8, 8]] NominalConstant
8: 'a' [[6, 6]] DeterminerConstant
9: 'huge' [[7, 7]] ModificationalConstant
10: 'over' [[9, 9]] ModificationalFunctor
11: 'support' [[10, 10]] NominalConstant
12: 'for' [[11, 11]] ModificationalFunctor
13: 'Israel' [[12, 12]] NominalConstant
14: 'whether' [[14, 14]] GeneralFunctor
15: 'was' [[19, 19]] FactualPredicator
16: 'support' [[16, 16]] NominalConstant
17: 'the' [[15, 15]] DeterminerConstant
18: 'for' [[17, 17]] ModificationalFunctor
19: 'Palestine' [[18, 18]] NominalConstant
20: '.' [[22, 22]] PunctuationalConstant
21: 'strong enough' [[20, 21]] ModificationalConstant
EDGES
0->3 pred.arg.2 | 'know'->'that'
0->2 pred.arg.1 | 'know'->'New'
2->1 as:modification | 'New'->'Yorkers'
3->4 func.arg | 'that'->'and'
4->5 func.arg | 'and'->'was'
4->14 func.arg | 'and'->'whether'
5->6 pred.arg.1 | 'was'->'this'
5->10 pred.arg.2 | 'was'->'over'
8->9 as:modification | 'a'->'huge'
9->7 as:modification | 'huge'->'battle'
10->8 variable | 'over'->'a'
10->12 body | 'over'->'for'
12->13 body | 'for'->'Israel'
12->11 variable | 'for'->'support'
14->15 func.arg | 'whether'->'was'
15->18 pred.arg.1 | 'was'->'for'
15->21 pred.arg.2 | 'was'->'strong enough'
17->16 as:modification | 'the'->'support'
18->19 body | 'for'->'Palestine'
18->17 variable | 'for'->'the'
20->0 as:punctuation | '.'->'know'

--- after ---
New Yorkers know that this was a huge battle over support for Israel and whether the support for Palestine was strong enough .
NODES
0: 'know' [[2, 2]] FactualPredicator
1: 'Yorkers' [[1, 1]] NominalConstant
2: 'New' [[0, 0]] ModificationalConstant
3: 'that' [[3, 3]] GeneralFunctor
4: 'and' [[13, 13]] ConjunctionalFunctor
5: 'was' [[5, 5]] FactualPredicator
6: 'this' [[4, 4]] NominalConstant
7: 'battle' [[8, 8]] NominalConstant
8: 'a' [[6, 6]] DeterminerConstant
9: 'huge' [[7, 7]] ModificationalConstant
10: 'over' [[9, 9]] ModificationalFunctor
11: 'support' [[10, 10]] NominalConstant
12: 'for' [[11, 11]] ModificationalFunctor
13: 'Israel' [[12, 12]] NominalConstant
14: 'whether' [[14, 14]] GeneralFunctor
15: 'was strong enough' [[19, 21]] FactualPredicator
16: 'support' [[16, 16]] NominalConstant
17: 'the' [[15, 15]] DeterminerConstant
18: 'for' [[17, 17]] ModificationalFunctor
19: 'Palestine' [[18, 18]] NominalConstant
20: '.' [[22, 22]] PunctuationalConstant
EDGES
0->3 pred.arg.2 | 'know'->'that'
0->2 pred.arg.1 | 'know'->'New'
2->1 as:modification | 'New'->'Yorkers'
3->4 func.arg | 'that'->'and'
4->5 func.arg | 'and'->'was'
4->14 func.arg | 'and'->'whether'
5->6 pred.arg.1 | 'was'->'this'
5->10 pred.arg.2 | 'was'->'over'
8->9 as:modification | 'a'->'huge'
9->7 as:modification | 'huge'->'battle'
10->8 variable | 'over'->'a'
10->12 body | 'over'->'for'
12->13 body | 'for'->'Israel'
12->11 variable | 'for'->'support'
14->15 func.arg | 'whether'->'was strong enough'
15->18 pred.arg.1 | 'was strong enough'->'for'
17->16 as:modification | 'the'->'support'
18->19 body | 'for'->'Palestine'
18->17 variable | 'for'->'the'
20->0 as:punctuation | '.'->'know'
```

### train:732 `merged_intervened_be_prep_split`

I think this location is no longer in business.

```text
I think this location is no longer in business.
NODES
0: 'think' [[1, 1]] FactualPredicator
1: 'I' [[0, 0]] NominalConstant
2: 'business' [[8, 8]] NominalConstant
3: 'location' [[3, 3]] NominalConstant
4: 'this' [[2, 2]] DeterminerConstant
5: '.' [[9, 9]] PunctuationalConstant
6: 'is no longer in' [[4, 7]] FactualPredicator
EDGES
0->1 pred.arg.1 | 'think'->'I'
0->6 pred.arg.2 | 'think'->'is no longer in'
4->3 as:modification | 'this'->'location'
5->0 as:punctuation | '.'->'think'
6->2 pred.arg.2 | 'is no longer in'->'business'
6->4 pred.arg.1 | 'is no longer in'->'this'

--- after ---
I think this location is no longer in business.
NODES
0: 'think' [[1, 1]] FactualPredicator
1: 'I' [[0, 0]] NominalConstant
2: 'business' [[8, 8]] NominalConstant
3: 'location' [[3, 3]] NominalConstant
4: 'this' [[2, 2]] DeterminerConstant
5: '.' [[9, 9]] PunctuationalConstant
6: 'is longer' [[4, 4], [6, 6]] FactualPredicator
7: 'in' [[7, 7]] ModificationalFunctor
8: 'no' [[5, 5]] ModificationalConstant
EDGES
0->1 pred.arg.1 | 'think'->'I'
0->7 pred.arg.2 | 'think'->'in'
4->3 as:modification | 'this'->'location'
5->0 as:punctuation | '.'->'think'
6->4 pred.arg.1 | 'is longer'->'this'
8->6 as:modification | 'no'->'is longer'
7->8 variable | 'in'->'no'
7->2 body | 'in'->'business'
```

### train:766 `there_be_merged_split`

There are no jobs in this country.

```text
There are no jobs in this country.
NODES
0: 'in' [[4, 4]] ModificationalFunctor
1: 'jobs' [[3, 3]] NominalConstant
2: 'no' [[2, 2]] DeterminerConstant
3: 'country' [[6, 6]] NominalConstant
4: 'this' [[5, 5]] DeterminerConstant
5: '.' [[7, 7]] PunctuationalConstant
6: 'There are' [[0, 1]] FactualPredicator
EDGES
0->4 body | 'in'->'this'
0->6 variable | 'in'->'There are'
2->1 as:modification | 'no'->'jobs'
4->3 as:modification | 'this'->'country'
5->0 as:punctuation | '.'->'in'
6->2 pred.arg.1 | 'There are'->'no'

--- after ---
There are no jobs in this country.
NODES
0: 'in' [[4, 4]] ModificationalFunctor
1: 'jobs' [[3, 3]] NominalConstant
2: 'no' [[2, 2]] DeterminerConstant
3: 'country' [[6, 6]] NominalConstant
4: 'this' [[5, 5]] DeterminerConstant
5: '.' [[7, 7]] PunctuationalConstant
6: 'are' [[1, 1]] FactualPredicator
7: 'There' [[0, 0]] NominalConstant
EDGES
0->4 body | 'in'->'this'
0->6 variable | 'in'->'are'
2->1 as:modification | 'no'->'jobs'
4->3 as:modification | 'this'->'country'
5->0 as:punctuation | '.'->'in'
6->2 pred.arg.2 | 'are'->'no'
6->7 pred.arg.1 | 'are'->'There'
```

### train:830 `merged_simple_copula_adverb_split`

My waiter was so excellent I gave him a 75% tip, and it was worht every penny ..

```text
My waiter was so excellent I gave him a 75% tip, and it was worht every penny ..
NODES
0: 'gave' [[6, 6]] FactualPredicator
1: 'I' [[5, 5]] NominalConstant
2: 'him' [[7, 7]] NominalConstant
3: 'it' [[14, 14]] NominalConstant
4: 'waiter' [[1, 1]] NominalConstant
5: 'My' [[0, 0]] NominalConstant
6: 'tip' [[11, 11]] NominalConstant
7: 'a' [[8, 8]] DeterminerConstant
8: 'penny' [[18, 18]] NominalConstant
9: 'every' [[17, 17]] DeterminerConstant
10: '%' [[10, 10]] NominalConstant
11: '75' [[9, 9]] NominalConstant
12: '..' [[19, 19]] PunctuationalConstant
13: ', and' [[12, 13]] ConjunctionalFunctor
14: 'was so excellent' [[2, 4]] FactualPredicator
15: 'was worht' [[15, 16]] FactualPredicator
16: 'Modification' 'Modification' ModificationalFunctor
17: 'Modification' 'Modification' ModificationalFunctor
EDGES
0->1 pred.arg.1 | 'gave'->'I'
0->2 pred.arg.2 | 'gave'->'him'
0->7 pred.arg.3 | 'gave'->'a'
5->4 as:modification | 'My'->'waiter'
7->17 as:modification | 'a'->'Modification'
9->8 as:modification | 'every'->'penny'
11->10 as:modification | '75'->'%'
12->13 as:punctuation | '..'->', and'
13->15 func.arg | ', and'->'was worht'
13->16 func.arg | ', and'->'Modification'
14->5 pred.arg.1 | 'was so excellent'->'My'
15->3 pred.arg.1 | 'was worht'->'it'
15->9 pred.arg.2 | 'was worht'->'every'
16->14 variable | 'Modification'->'was so excellent'
16->0 body | 'Modification'->'gave'
17->6 variable | 'Modification'->'tip'
17->11 body | 'Modification'->'75'

--- after ---
My waiter was so excellent I gave him a 75% tip, and it was worht every penny ..
NODES
0: 'gave' [[6, 6]] FactualPredicator
1: 'I' [[5, 5]] NominalConstant
2: 'him' [[7, 7]] NominalConstant
3: 'it' [[14, 14]] NominalConstant
4: 'waiter' [[1, 1]] NominalConstant
5: 'My' [[0, 0]] NominalConstant
6: 'tip' [[11, 11]] NominalConstant
7: 'a' [[8, 8]] DeterminerConstant
8: 'penny' [[18, 18]] NominalConstant
9: 'every' [[17, 17]] DeterminerConstant
10: '%' [[10, 10]] NominalConstant
11: '75' [[9, 9]] NominalConstant
12: '..' [[19, 19]] PunctuationalConstant
13: ', and' [[12, 13]] ConjunctionalFunctor
14: 'was excellent' [[2, 2], [4, 4]] FactualPredicator
15: 'was worht' [[15, 16]] FactualPredicator
16: 'Modification' 'Modification' ModificationalFunctor
17: 'Modification' 'Modification' ModificationalFunctor
18: 'so' [[3, 3]] ModificationalConstant
EDGES
0->1 pred.arg.1 | 'gave'->'I'
0->2 pred.arg.2 | 'gave'->'him'
0->7 pred.arg.3 | 'gave'->'a'
5->4 as:modification | 'My'->'waiter'
7->17 as:modification | 'a'->'Modification'
9->8 as:modification | 'every'->'penny'
11->10 as:modification | '75'->'%'
12->13 as:punctuation | '..'->', and'
13->15 func.arg | ', and'->'was worht'
13->16 func.arg | ', and'->'Modification'
14->5 pred.arg.1 | 'was excellent'->'My'
15->3 pred.arg.1 | 'was worht'->'it'
15->9 pred.arg.2 | 'was worht'->'every'
16->18 variable | 'Modification'->'so'
16->0 body | 'Modification'->'gave'
17->6 variable | 'Modification'->'tip'
17->11 body | 'Modification'->'75'
18->14 as:modification | 'so'->'was excellent'
```

### train:863 `merged_simple_copula_adverb_split`

The paperwork was a very easy and smooth.

```text
The paperwork was a very easy and smooth.
NODES
0: 'and' [[6, 6]] ConjunctionalFunctor
1: 'smooth' [[7, 7]] FactualPredicator
2: 'paperwork' [[1, 1]] NominalConstant
3: 'The' [[0, 0]] DeterminerConstant
4: '.' [[8, 8]] PunctuationalConstant
5: 'was a very easy' [[2, 5]] FactualPredicator
EDGES
0->1 func.arg | 'and'->'smooth'
0->5 func.arg | 'and'->'was a very easy'
1->3 pred.arg.1 | 'smooth'->'The'
3->2 as:modification | 'The'->'paperwork'
4->0 as:punctuation | '.'->'and'
5->3 pred.arg.1 | 'was a very easy'->'The'

--- after ---
The paperwork was a very easy and smooth.
NODES
0: 'and' [[6, 6]] ConjunctionalFunctor
1: 'smooth' [[7, 7]] FactualPredicator
2: 'paperwork' [[1, 1]] NominalConstant
3: 'The' [[0, 0]] DeterminerConstant
4: '.' [[8, 8]] PunctuationalConstant
5: 'was a easy' [[2, 3], [5, 5]] FactualPredicator
6: 'very' [[4, 4]] ModificationalConstant
EDGES
0->1 func.arg | 'and'->'smooth'
0->6 func.arg | 'and'->'very'
1->3 pred.arg.1 | 'smooth'->'The'
3->2 as:modification | 'The'->'paperwork'
4->0 as:punctuation | '.'->'and'
5->3 pred.arg.1 | 'was a easy'->'The'
6->5 as:modification | 'very'->'was a easy'
```

### train:869 `split_simple_copula_merged`

Suvit said the motive for their intrusions was not clear but he had ordered reinforcements to beef up security along the porous frontier .

```text
Suvit said the motive for their intrusions was not clear but he had ordered reinforcements to beef up security along the porous frontier .
NODES
0: 'but' [[10, 10]] LogicalPredicator
1: 'said' [[1, 1]] FactualPredicator
2: 'Suvit' [[0, 0]] NominalConstant
3: 'was' [[7, 7]] FactualPredicator
4: 'not' [[8, 8]] ModificationalConstant
5: 'motive' [[3, 3]] NominalConstant
6: 'the' [[2, 2]] DeterminerConstant
7: 'for' [[4, 4]] ModificationalFunctor
8: 'intrusions' [[6, 6]] NominalConstant
9: 'their' [[5, 5]] ModificationalConstant
10: 'clear' [[9, 9]] ModificationalConstant
11: 'he' [[11, 11]] NominalConstant
12: 'reinforcements' [[14, 14]] NominalConstant
13: 'security' [[18, 18]] NominalConstant
14: 'along' [[19, 19]] ModificationalFunctor
15: 'frontier' [[22, 22]] NominalConstant
16: 'the' [[20, 20]] DeterminerConstant
17: 'porous' [[21, 21]] ModificationalConstant
18: '.' [[23, 23]] PunctuationalConstant
19: 'had ordered' [[12, 13]] FactualPredicator
20: 'to beef up' [[15, 17]] FactualPredicator
21: 'Modification' 'Modification' ModificationalFunctor
EDGES
0->1 pred.arg.1 | 'but'->'said'
0->19 pred.arg.2 | 'but'->'had ordered'
1->2 pred.arg.1 | 'said'->'Suvit'
1->4 pred.arg.2 | 'said'->'not'
3->10 pred.arg.2 | 'was'->'clear'
3->7 pred.arg.1 | 'was'->'for'
4->3 as:modification | 'not'->'was'
6->5 as:modification | 'the'->'motive'
7->6 variable | 'for'->'the'
7->9 body | 'for'->'their'
9->8 as:modification | 'their'->'intrusions'
14->13 variable | 'along'->'security'
14->16 body | 'along'->'the'
16->17 as:modification | 'the'->'porous'
17->15 as:modification | 'porous'->'frontier'
18->0 as:punctuation | '.'->'but'
19->11 pred.arg.1 | 'had ordered'->'he'
19->21 pred.arg.4 | 'had ordered'->'Modification'
20->12 pred.arg.1 | 'to beef up'->'reinforcements'
20->14 pred.arg.2 | 'to beef up'->'along'
21->12 variable | 'Modification'->'reinforcements'
21->20 body | 'Modification'->'to beef up'

--- after ---
Suvit said the motive for their intrusions was not clear but he had ordered reinforcements to beef up security along the porous frontier .
NODES
0: 'but' [[10, 10]] LogicalPredicator
1: 'said' [[1, 1]] FactualPredicator
2: 'Suvit' [[0, 0]] NominalConstant
3: 'was clear' [[7, 7], [9, 9]] FactualPredicator
4: 'not' [[8, 8]] ModificationalConstant
5: 'motive' [[3, 3]] NominalConstant
6: 'the' [[2, 2]] DeterminerConstant
7: 'for' [[4, 4]] ModificationalFunctor
8: 'intrusions' [[6, 6]] NominalConstant
9: 'their' [[5, 5]] ModificationalConstant
10: 'he' [[11, 11]] NominalConstant
11: 'reinforcements' [[14, 14]] NominalConstant
12: 'security' [[18, 18]] NominalConstant
13: 'along' [[19, 19]] ModificationalFunctor
14: 'frontier' [[22, 22]] NominalConstant
15: 'the' [[20, 20]] DeterminerConstant
16: 'porous' [[21, 21]] ModificationalConstant
17: '.' [[23, 23]] PunctuationalConstant
18: 'had ordered' [[12, 13]] FactualPredicator
19: 'to beef up' [[15, 17]] FactualPredicator
20: 'Modification' 'Modification' ModificationalFunctor
EDGES
0->1 pred.arg.1 | 'but'->'said'
0->18 pred.arg.2 | 'but'->'had ordered'
1->2 pred.arg.1 | 'said'->'Suvit'
1->4 pred.arg.2 | 'said'->'not'
3->7 pred.arg.1 | 'was clear'->'for'
4->3 as:modification | 'not'->'was clear'
6->5 as:modification | 'the'->'motive'
7->6 variable | 'for'->'the'
7->9 body | 'for'->'their'
9->8 as:modification | 'their'->'intrusions'
13->12 variable | 'along'->'security'
13->15 body | 'along'->'the'
15->16 as:modification | 'the'->'porous'
16->14 as:modification | 'porous'->'frontier'
17->0 as:punctuation | '.'->'but'
18->10 pred.arg.1 | 'had ordered'->'he'
18->20 pred.arg.4 | 'had ordered'->'Modification'
19->11 pred.arg.1 | 'to beef up'->'reinforcements'
19->13 pred.arg.2 | 'to beef up'->'along'
20->11 variable | 'Modification'->'reinforcements'
20->19 body | 'Modification'->'to beef up'
```

### train:943 `merged_simple_copula_adverb_split`

These people were so helpful this week and did everything to sort out my windscreen and insurance.

```text
These people were so helpful this week and did everything to sort out my windscreen and insurance.
NODES
0: 'and' [[7, 7]] ConjunctionalFunctor
1: 'did' [[8, 8]] FactualPredicator
2: 'everything' [[9, 9]] NominalConstant
3: 'windscreen' [[14, 14]] NominalConstant
4: 'insurance' [[16, 16]] NominalConstant
5: 'my' [[13, 13]] ModificationalConstant
6: 'and' [[15, 15]] ConjunctionalFunctor
7: 'people' [[1, 1]] NominalConstant
8: 'These' [[0, 0]] DeterminerConstant
9: 'week' [[6, 6]] NominalConstant
10: 'this' [[5, 5]] DeterminerConstant
11: '.' [[17, 17]] PunctuationalConstant
12: 'were so helpful' [[2, 4]] FactualPredicator
13: 'to sort out' [[10, 12]] FactualPredicator
14: 'Modification' 'Modification' ModificationalFunctor
EDGES
0->14 func.arg | 'and'->'Modification'
0->1 func.arg | 'and'->'did'
1->2 pred.arg.2 | 'did'->'everything'
1->8 pred.arg.1 | 'did'->'These'
1->13 pred.arg.4 | 'did'->'to sort out'
8->7 as:modification | 'These'->'people'
10->9 as:modification | 'this'->'week'
11->0 as:punctuation | '.'->'and'
12->8 pred.arg.1 | 'were so helpful'->'These'
13->5 pred.arg.2 | 'to sort out'->'my'
13->8 pred.arg.1 | 'to sort out'->'These'
14->12 variable | 'Modification'->'were so helpful'
14->10 body | 'Modification'->'this'
6->3 func.arg | 'and'->'windscreen'
6->4 func.arg | 'and'->'insurance'
5->6 as:modification | 'my'->'and'

--- after ---
These people were so helpful this week and did everything to sort out my windscreen and insurance.
NODES
0: 'and' [[7, 7]] ConjunctionalFunctor
1: 'did' [[8, 8]] FactualPredicator
2: 'everything' [[9, 9]] NominalConstant
3: 'windscreen' [[14, 14]] NominalConstant
4: 'insurance' [[16, 16]] NominalConstant
5: 'my' [[13, 13]] ModificationalConstant
6: 'and' [[15, 15]] ConjunctionalFunctor
7: 'people' [[1, 1]] NominalConstant
8: 'These' [[0, 0]] DeterminerConstant
9: 'week' [[6, 6]] NominalConstant
10: 'this' [[5, 5]] DeterminerConstant
11: '.' [[17, 17]] PunctuationalConstant
12: 'were helpful' [[2, 2], [4, 4]] FactualPredicator
13: 'to sort out' [[10, 12]] FactualPredicator
14: 'Modification' 'Modification' ModificationalFunctor
15: 'so' [[3, 3]] ModificationalConstant
EDGES
0->14 func.arg | 'and'->'Modification'
0->1 func.arg | 'and'->'did'
1->2 pred.arg.2 | 'did'->'everything'
1->8 pred.arg.1 | 'did'->'These'
1->13 pred.arg.4 | 'did'->'to sort out'
8->7 as:modification | 'These'->'people'
10->9 as:modification | 'this'->'week'
11->0 as:punctuation | '.'->'and'
12->8 pred.arg.1 | 'were helpful'->'These'
13->5 pred.arg.2 | 'to sort out'->'my'
13->8 pred.arg.1 | 'to sort out'->'These'
14->15 variable | 'Modification'->'so'
14->10 body | 'Modification'->'this'
6->3 func.arg | 'and'->'windscreen'
6->4 func.arg | 'and'->'insurance'
5->6 as:modification | 'my'->'and'
15->12 as:modification | 'so'->'were helpful'
```

### train:992 `there_be_merged_split`

If there are any objections you need to get back to me ASAP.

```text
If there are any objections you need to get back to me ASAP.
NODES
0: 'need' [[6, 6]] FactualPredicator
1: 'If' [[0, 0]] LogicalPredicator
2: 'ASAP' [[12, 12]] ModificationalConstant
3: 'to' [[10, 10]] ModificationalFunctor
4: 'you' [[5, 5]] NominalConstant
5: 'me' [[11, 11]] NominalConstant
6: 'objections' [[4, 4]] NominalConstant
7: 'any' [[3, 3]] DeterminerConstant
8: '.' [[13, 13]] PunctuationalConstant
9: 'there are' [[1, 2]] FactualPredicator
10: 'to get back' [[7, 9]] FactualPredicator
EDGES
0->4 pred.arg.1 | 'need'->'you'
0->2 pred.arg.4 | 'need'->'ASAP'
1->0 pred.arg.1 | 'If'->'need'
1->9 pred.arg.2 | 'If'->'there are'
2->3 as:modification | 'ASAP'->'to'
3->5 body | 'to'->'me'
3->10 variable | 'to'->'to get back'
7->6 as:modification | 'any'->'objections'
8->1 as:punctuation | '.'->'If'
9->7 pred.arg.1 | 'there are'->'any'
10->4 pred.arg.1 | 'to get back'->'you'

--- after ---
If there are any objections you need to get back to me ASAP.
NODES
0: 'need' [[6, 6]] FactualPredicator
1: 'If' [[0, 0]] LogicalPredicator
2: 'ASAP' [[12, 12]] ModificationalConstant
3: 'to' [[10, 10]] ModificationalFunctor
4: 'you' [[5, 5]] NominalConstant
5: 'me' [[11, 11]] NominalConstant
6: 'objections' [[4, 4]] NominalConstant
7: 'any' [[3, 3]] DeterminerConstant
8: '.' [[13, 13]] PunctuationalConstant
9: 'are' [[2, 2]] FactualPredicator
10: 'to get back' [[7, 9]] FactualPredicator
11: 'there' [[1, 1]] NominalConstant
EDGES
0->4 pred.arg.1 | 'need'->'you'
0->2 pred.arg.4 | 'need'->'ASAP'
1->0 pred.arg.1 | 'If'->'need'
1->9 pred.arg.2 | 'If'->'are'
2->3 as:modification | 'ASAP'->'to'
3->5 body | 'to'->'me'
3->10 variable | 'to'->'to get back'
7->6 as:modification | 'any'->'objections'
8->1 as:punctuation | '.'->'If'
9->7 pred.arg.2 | 'are'->'any'
10->4 pred.arg.1 | 'to get back'->'you'
9->11 pred.arg.1 | 'are'->'there'
```

### train:1037 `there_be_merged_split`

There are a few life theories like that which working through.

```text
There are a few life theories like that which working through.
NODES
0: 'which' [[8, 8]] GeneralFunctor
1: 'like' [[6, 6]] ModificationalFunctor
2: 'working' [[9, 9]] FactualPredicator
3: 'through' [[10, 10]] ModificationalFunctor
4: 'that' [[7, 7]] NominalConstant
5: 'theories' [[5, 5]] NominalConstant
6: 'life' [[4, 4]] NominalConstant
7: '.' [[11, 11]] PunctuationalConstant
8: 'There are' [[0, 1]] FactualPredicator
9: 'a few' [[2, 3]] ModificationalConstant
10: 'Ref' 'Ref' ReferentialPredicator
11: 'Modification' 'Modification' ModificationalFunctor
EDGES
1->4 body | 'like'->'that'
1->9 variable | 'like'->'a few'
2->0 pred.arg.1 | 'working'->'which'
3->2 variable | 'through'->'working'
6->5 as:modification | 'life'->'theories'
7->8 as:punctuation | '.'->'There are'
8->10 pred.arg.1 | 'There are'->'Ref'
9->6 as:modification | 'a few'->'life'
10->1 pred.arg.1 | 'Ref'->'like'
10->11 pred.arg.2 | 'Ref'->'Modification'
11->0 variable | 'Modification'->'which'
11->3 body | 'Modification'->'through'

--- after ---
There are a few life theories like that which working through.
NODES
0: 'which' [[8, 8]] GeneralFunctor
1: 'like' [[6, 6]] ModificationalFunctor
2: 'working' [[9, 9]] FactualPredicator
3: 'through' [[10, 10]] ModificationalFunctor
4: 'that' [[7, 7]] NominalConstant
5: 'theories' [[5, 5]] NominalConstant
6: 'life' [[4, 4]] NominalConstant
7: '.' [[11, 11]] PunctuationalConstant
8: 'are' [[1, 1]] FactualPredicator
9: 'a few' [[2, 3]] ModificationalConstant
10: 'Ref' 'Ref' ReferentialPredicator
11: 'Modification' 'Modification' ModificationalFunctor
12: 'There' [[0, 0]] NominalConstant
EDGES
1->4 body | 'like'->'that'
1->9 variable | 'like'->'a few'
2->0 pred.arg.1 | 'working'->'which'
3->2 variable | 'through'->'working'
6->5 as:modification | 'life'->'theories'
7->8 as:punctuation | '.'->'are'
8->10 pred.arg.2 | 'are'->'Ref'
9->6 as:modification | 'a few'->'life'
10->1 pred.arg.1 | 'Ref'->'like'
10->11 pred.arg.2 | 'Ref'->'Modification'
11->0 variable | 'Modification'->'which'
11->3 body | 'Modification'->'through'
8->12 pred.arg.1 | 'are'->'There'
```

### train:1056 `merged_simple_copula_adverb_split`

I was very pleased to find my accommodations and the hotel staff to be a very calming and comforting part of my trip.

```text
I was very pleased to find my accommodations and the hotel staff to be a very calming and comforting part of my trip.
NODES
0: 'I' [[0, 0]] NominalConstant
1: 'of' [[20, 20]] ModificationalFunctor
2: 'and' [[17, 17]] ConjunctionalFunctor
3: 'comforting' [[18, 18]] NominalConstant
4: 'and' [[8, 8]] ConjunctionalFunctor
5: 'part' [[19, 19]] NominalConstant
6: 'a' [[14, 14]] DeterminerConstant
7: 'trip' [[22, 22]] NominalConstant
8: 'my' [[21, 21]] NominalConstant
9: 'accommodations' [[7, 7]] NominalConstant
10: 'my' [[6, 6]] NominalConstant
11: 'staff' [[11, 11]] NominalConstant
12: 'hotel' [[10, 10]] NominalConstant
13: 'the' [[9, 9]] DeterminerConstant
14: '.' [[23, 23]] PunctuationalConstant
15: 'was very pleased' [[1, 3]] FactualPredicator
16: 'to find' [[4, 5]] FactualPredicator
17: 'to be' [[12, 13]] FactualPredicator
18: 'very calming' [[15, 16]] NominalConstant
19: 'Modification' 'Modification' ModificationalFunctor
EDGES
1->5 variable | 'of'->'part'
1->19 body | 'of'->'Modification'
2->3 func.arg | 'and'->'comforting'
2->18 func.arg | 'and'->'very calming'
4->10 func.arg | 'and'->'my'
4->13 func.arg | 'and'->'the'
6->1 as:modification | 'a'->'of'
8->7 as:modification | 'my'->'trip'
10->9 as:modification | 'my'->'accommodations'
12->11 as:modification | 'hotel'->'staff'
13->12 as:modification | 'the'->'hotel'
14->15 as:punctuation | '.'->'was very pleased'
15->0 pred.arg.1 | 'was very pleased'->'I'
15->16 pred.arg.4 | 'was very pleased'->'to find'
16->0 pred.arg.1 | 'to find'->'I'
16->4 pred.arg.2 | 'to find'->'and'
16->17 pred.arg.4 | 'to find'->'to be'
17->4 pred.arg.1 | 'to be'->'and'
17->6 pred.arg.2 | 'to be'->'a'
19->8 variable | 'Modification'->'my'
19->2 body | 'Modification'->'and'

--- after ---
I was very pleased to find my accommodations and the hotel staff to be a very calming and comforting part of my trip.
NODES
0: 'I' [[0, 0]] NominalConstant
1: 'of' [[20, 20]] ModificationalFunctor
2: 'and' [[17, 17]] ConjunctionalFunctor
3: 'comforting' [[18, 18]] NominalConstant
4: 'and' [[8, 8]] ConjunctionalFunctor
5: 'part' [[19, 19]] NominalConstant
6: 'a' [[14, 14]] DeterminerConstant
7: 'trip' [[22, 22]] NominalConstant
8: 'my' [[21, 21]] NominalConstant
9: 'accommodations' [[7, 7]] NominalConstant
10: 'my' [[6, 6]] NominalConstant
11: 'staff' [[11, 11]] NominalConstant
12: 'hotel' [[10, 10]] NominalConstant
13: 'the' [[9, 9]] DeterminerConstant
14: '.' [[23, 23]] PunctuationalConstant
15: 'was pleased' [[1, 1], [3, 3]] FactualPredicator
16: 'to find' [[4, 5]] FactualPredicator
17: 'to be' [[12, 13]] FactualPredicator
18: 'very calming' [[15, 16]] NominalConstant
19: 'Modification' 'Modification' ModificationalFunctor
20: 'very' [[2, 2]] ModificationalConstant
EDGES
1->5 variable | 'of'->'part'
1->19 body | 'of'->'Modification'
2->3 func.arg | 'and'->'comforting'
2->18 func.arg | 'and'->'very calming'
4->10 func.arg | 'and'->'my'
4->13 func.arg | 'and'->'the'
6->1 as:modification | 'a'->'of'
8->7 as:modification | 'my'->'trip'
10->9 as:modification | 'my'->'accommodations'
12->11 as:modification | 'hotel'->'staff'
13->12 as:modification | 'the'->'hotel'
14->20 as:punctuation | '.'->'very'
15->0 pred.arg.1 | 'was pleased'->'I'
15->16 pred.arg.4 | 'was pleased'->'to find'
16->0 pred.arg.1 | 'to find'->'I'
16->4 pred.arg.2 | 'to find'->'and'
16->17 pred.arg.4 | 'to find'->'to be'
17->4 pred.arg.1 | 'to be'->'and'
17->6 pred.arg.2 | 'to be'->'a'
19->8 variable | 'Modification'->'my'
19->2 body | 'Modification'->'and'
20->15 as:modification | 'very'->'was pleased'
```

### train:1074 `merged_simple_copula_adverb_split`

I am very interested in energy - finance and think that your idea is a very good.

```text
I am very interested in energy - finance and think that your idea is a very good.
NODES
0: 'and' [[8, 8]] ConjunctionalFunctor
1: 'think' [[9, 9]] FactualPredicator
2: 'is' [[13, 13]] FactualPredicator
3: 'in' [[4, 4]] ModificationalFunctor
4: 'I' [[0, 0]] NominalConstant
5: 'that' [[10, 10]] GeneralFunctor
6: 'idea' [[12, 12]] NominalConstant
7: 'your' [[11, 11]] NominalConstant
8: 'finance' [[7, 7]] NominalConstant
9: 'energy' [[5, 5]] NominalConstant
10: '.' [[17, 17]] PunctuationalConstant
11: '-' [[6, 6]] PunctuationalConstant
12: 'am very interested' [[1, 3]] FactualPredicator
13: 'a very good' [[14, 16]] NominalConstant
EDGES
0->1 func.arg | 'and'->'think'
0->3 func.arg | 'and'->'in'
1->4 pred.arg.1 | 'think'->'I'
1->5 pred.arg.2 | 'think'->'that'
2->7 pred.arg.1 | 'is'->'your'
2->13 pred.arg.2 | 'is'->'a very good'
3->9 body | 'in'->'energy'
3->12 variable | 'in'->'am very interested'
5->2 func.arg | 'that'->'is'
7->6 as:modification | 'your'->'idea'
9->8 as:modification | 'energy'->'finance'
9->11 punctuation | 'energy'->'-'
10->0 as:punctuation | '.'->'and'
12->4 pred.arg.1 | 'am very interested'->'I'

--- after ---
I am very interested in energy - finance and think that your idea is a very good.
NODES
0: 'and' [[8, 8]] ConjunctionalFunctor
1: 'think' [[9, 9]] FactualPredicator
2: 'is' [[13, 13]] FactualPredicator
3: 'in' [[4, 4]] ModificationalFunctor
4: 'I' [[0, 0]] NominalConstant
5: 'that' [[10, 10]] GeneralFunctor
6: 'idea' [[12, 12]] NominalConstant
7: 'your' [[11, 11]] NominalConstant
8: 'finance' [[7, 7]] NominalConstant
9: 'energy' [[5, 5]] NominalConstant
10: '.' [[17, 17]] PunctuationalConstant
11: '-' [[6, 6]] PunctuationalConstant
12: 'am interested' [[1, 1], [3, 3]] FactualPredicator
13: 'a very good' [[14, 16]] NominalConstant
14: 'very' [[2, 2]] ModificationalConstant
EDGES
0->1 func.arg | 'and'->'think'
0->3 func.arg | 'and'->'in'
1->4 pred.arg.1 | 'think'->'I'
1->5 pred.arg.2 | 'think'->'that'
2->7 pred.arg.1 | 'is'->'your'
2->13 pred.arg.2 | 'is'->'a very good'
3->9 body | 'in'->'energy'
3->14 variable | 'in'->'very'
5->2 func.arg | 'that'->'is'
7->6 as:modification | 'your'->'idea'
9->8 as:modification | 'energy'->'finance'
9->11 punctuation | 'energy'->'-'
10->0 as:punctuation | '.'->'and'
12->4 pred.arg.1 | 'am interested'->'I'
14->12 as:modification | 'very'->'am interested'
```

### train:1182 `merged_simple_copula_adverb_split`

The other stylists around me kept pressuring me saying he's so wonderful, you're going to love your hair.

```text
The other stylists around me kept pressuring me saying he's so wonderful, you're going to love your hair.
NODES
0: 'kept' [[5, 5]] FactualPredicator
1: 'pressuring' [[6, 6]] FactualPredicator
2: 'saying' [[8, 8]] FactualPredicator
3: ',' [[14, 14]] ConjunctionalFunctor
4: 'me' [[7, 7]] NominalConstant
5: 'you' [[15, 15]] NominalConstant
6: 'he' [[9, 9]] NominalConstant
7: 'around' [[3, 3]] ModificationalFunctor
8: 'me' [[4, 4]] NominalConstant
9: 'hair' [[22, 22]] NominalConstant
10: 'your' [[21, 21]] NominalConstant
11: 'stylists' [[2, 2]] NominalConstant
12: 'other' [[1, 1]] ModificationalConstant
13: 'The' [[0, 0]] DeterminerConstant
14: '.' [[23, 23]] PunctuationalConstant
15: "' re going to love" [[16, 20]] FactualPredicator
16: "' s so wonderful" [[10, 13]] FactualPredicator
17: 'Modification' 'Modification' ModificationalFunctor
EDGES
0->7 pred.arg.1 | 'kept'->'around'
0->17 pred.arg.2 | 'kept'->'Modification'
1->4 pred.arg.2 | 'pressuring'->'me'
1->7 pred.arg.1 | 'pressuring'->'around'
2->3 pred.arg.2 | 'saying'->','
2->6 pred.arg.1 | 'saying'->'he'
3->15 func.arg | ','->"' re going to love"
3->16 func.arg | ','->"' s so wonderful"
7->8 body | 'around'->'me'
7->13 variable | 'around'->'The'
10->9 as:modification | 'your'->'hair'
12->11 as:modification | 'other'->'stylists'
13->12 as:modification | 'The'->'other'
14->0 as:punctuation | '.'->'kept'
15->5 pred.arg.1 | "' re going to love"->'you'
15->10 pred.arg.2 | "' re going to love"->'your'
16->6 pred.arg.1 | "' s so wonderful"->'he'
17->1 variable | 'Modification'->'pressuring'
17->2 body | 'Modification'->'saying'

--- after ---
The other stylists around me kept pressuring me saying he's so wonderful, you're going to love your hair.
NODES
0: 'kept' [[5, 5]] FactualPredicator
1: 'pressuring' [[6, 6]] FactualPredicator
2: 'saying' [[8, 8]] FactualPredicator
3: ',' [[14, 14]] ConjunctionalFunctor
4: 'me' [[7, 7]] NominalConstant
5: 'you' [[15, 15]] NominalConstant
6: 'he' [[9, 9]] NominalConstant
7: 'around' [[3, 3]] ModificationalFunctor
8: 'me' [[4, 4]] NominalConstant
9: 'hair' [[22, 22]] NominalConstant
10: 'your' [[21, 21]] NominalConstant
11: 'stylists' [[2, 2]] NominalConstant
12: 'other' [[1, 1]] ModificationalConstant
13: 'The' [[0, 0]] DeterminerConstant
14: '.' [[23, 23]] PunctuationalConstant
15: "' re going to love" [[16, 20]] FactualPredicator
16: "' s wonderful" [[10, 11], [13, 13]] FactualPredicator
17: 'Modification' 'Modification' ModificationalFunctor
18: 'so' [[12, 12]] ModificationalConstant
EDGES
0->7 pred.arg.1 | 'kept'->'around'
0->17 pred.arg.2 | 'kept'->'Modification'
1->4 pred.arg.2 | 'pressuring'->'me'
1->7 pred.arg.1 | 'pressuring'->'around'
2->3 pred.arg.2 | 'saying'->','
2->6 pred.arg.1 | 'saying'->'he'
3->15 func.arg | ','->"' re going to love"
3->18 func.arg | ','->'so'
7->8 body | 'around'->'me'
7->13 variable | 'around'->'The'
10->9 as:modification | 'your'->'hair'
12->11 as:modification | 'other'->'stylists'
13->12 as:modification | 'The'->'other'
14->0 as:punctuation | '.'->'kept'
15->5 pred.arg.1 | "' re going to love"->'you'
15->10 pred.arg.2 | "' re going to love"->'your'
16->6 pred.arg.1 | "' s wonderful"->'he'
17->1 variable | 'Modification'->'pressuring'
17->2 body | 'Modification'->'saying'
18->16 as:modification | 'so'->"' s wonderful"
```

### train:1238 `there_be_merged_split`

Please tell me there is a deeper reason.

```text
Please tell me there is a deeper reason.
NODES
0: 'tell' [[1, 1]] FactualPredicator
1: 'me' [[2, 2]] NominalConstant
2: 'Please' [[0, 0]] NominalConstant
3: 'reason' [[7, 7]] NominalConstant
4: 'deeper' [[6, 6]] ModificationalConstant
5: 'a' [[5, 5]] DeterminerConstant
6: '.' [[8, 8]] PunctuationalConstant
7: 'there is' [[3, 4]] FactualPredicator
8: 'Missing' 'Missing' NominalConstant
EDGES
0->1 pred.arg.2 | 'tell'->'me'
0->7 pred.arg.3 | 'tell'->'there is'
2->0 as:discourse | 'Please'->'tell'
4->3 as:modification | 'deeper'->'reason'
5->4 as:modification | 'a'->'deeper'
6->2 as:punctuation | '.'->'Please'
7->5 pred.arg.1 | 'there is'->'a'
0->8 pred.arg.1 | 'tell'->'Missing'

--- after ---
Please tell me there is a deeper reason.
NODES
0: 'tell' [[1, 1]] FactualPredicator
1: 'me' [[2, 2]] NominalConstant
2: 'Please' [[0, 0]] NominalConstant
3: 'reason' [[7, 7]] NominalConstant
4: 'deeper' [[6, 6]] ModificationalConstant
5: 'a' [[5, 5]] DeterminerConstant
6: '.' [[8, 8]] PunctuationalConstant
7: 'is' [[4, 4]] FactualPredicator
8: 'Missing' 'Missing' NominalConstant
9: 'there' [[3, 3]] NominalConstant
EDGES
0->1 pred.arg.2 | 'tell'->'me'
0->7 pred.arg.3 | 'tell'->'is'
2->0 as:discourse | 'Please'->'tell'
4->3 as:modification | 'deeper'->'reason'
5->4 as:modification | 'a'->'deeper'
6->2 as:punctuation | '.'->'Please'
7->5 pred.arg.2 | 'is'->'a'
0->8 pred.arg.1 | 'tell'->'Missing'
7->9 pred.arg.1 | 'is'->'there'
```

### train:1249 `merged_intervened_be_prep_split`

They are RIGHT OFF the kitchen so you hear everything.

```text
They are RIGHT OFF the kitchen so you hear everything.
NODES
0: 'so' [[6, 6]] LogicalPredicator
1: 'hear' [[8, 8]] FactualPredicator
2: 'They' [[0, 0]] NominalConstant
3: 'you' [[7, 7]] NominalConstant
4: 'everything' [[9, 9]] NominalConstant
5: 'kitchen' [[5, 5]] NominalConstant
6: 'the' [[4, 4]] DeterminerConstant
7: '.' [[10, 10]] PunctuationalConstant
8: 'are RIGHT OFF' [[1, 3]] FactualPredicator
EDGES
0->1 pred.arg.2 | 'so'->'hear'
0->8 pred.arg.1 | 'so'->'are RIGHT OFF'
1->3 pred.arg.1 | 'hear'->'you'
1->4 pred.arg.2 | 'hear'->'everything'
6->5 as:modification | 'the'->'kitchen'
7->0 as:punctuation | '.'->'so'
8->2 pred.arg.1 | 'are RIGHT OFF'->'They'
8->6 pred.arg.2 | 'are RIGHT OFF'->'the'

--- after ---
They are RIGHT OFF the kitchen so you hear everything.
NODES
0: 'so' [[6, 6]] LogicalPredicator
1: 'hear' [[8, 8]] FactualPredicator
2: 'They' [[0, 0]] NominalConstant
3: 'you' [[7, 7]] NominalConstant
4: 'everything' [[9, 9]] NominalConstant
5: 'kitchen' [[5, 5]] NominalConstant
6: 'the' [[4, 4]] DeterminerConstant
7: '.' [[10, 10]] PunctuationalConstant
8: 'are RIGHT' [[1, 2]] FactualPredicator
9: 'OFF' [[3, 3]] ModificationalFunctor
EDGES
0->1 pred.arg.2 | 'so'->'hear'
0->9 pred.arg.1 | 'so'->'OFF'
1->3 pred.arg.1 | 'hear'->'you'
1->4 pred.arg.2 | 'hear'->'everything'
6->5 as:modification | 'the'->'kitchen'
7->0 as:punctuation | '.'->'so'
8->2 pred.arg.1 | 'are RIGHT'->'They'
9->8 variable | 'OFF'->'are RIGHT'
9->6 body | 'OFF'->'the'
```

### train:1291 `there_be_merged_split`

" There is always the possibility they could jam us up, make us look like we are not numero uno, " Pike said.

```text
" There is always the possibility they could jam us up, make us look like we are not numero uno, " Pike said.
NODES
0: 'said' [[24, 24]] FactualPredicator
1: 'could' [[7, 7]] FactualPredicator
2: ',' [[11, 11]] ConjunctionalFunctor
3: 'make' [[12, 12]] FactualPredicator
4: 'Pike' [[23, 23]] NominalConstant
5: 'they' [[6, 6]] NominalConstant
6: 'us' [[9, 9]] NominalConstant
7: 'us' [[13, 13]] NominalConstant
8: 'we' [[16, 16]] NominalConstant
9: 'possibility' [[5, 5]] NominalConstant
10: 'the' [[4, 4]] DeterminerConstant
11: ',' [[21, 21]] PunctuationalConstant
12: '.' [[25, 25]] PunctuationalConstant
13: '" "' [[0, 0], [22, 22]] PunctuationalConstant
14: 'There is always' [[1, 3]] FactualPredicator
15: 'jam up' [[8, 8], [10, 10]] FactualPredicator
16: 'look like' [[14, 15]] FactualPredicator
17: 'are not' [[17, 18]] FactualPredicator
18: 'numero uno' [[19, 20]] NominalConstant
19: 'Modification' 'Modification' ModificationalFunctor
EDGES
0->4 pred.arg.1 | 'said'->'Pike'
0->13 pred.arg.2 | 'said'->'" "'
1->2 pred.arg.2 | 'could'->','
1->5 pred.arg.1 | 'could'->'they'
2->3 func.arg | ','->'make'
2->15 func.arg | ','->'jam up'
3->5 pred.arg.1 | 'make'->'they'
3->16 pred.arg.2 | 'make'->'look like'
10->9 as:modification | 'the'->'possibility'
11->14 as:punctuation | ','->'There is always'
12->0 as:punctuation | '.'->'said'
13->11 as:punctuation | '" "'->','
14->19 pred.arg.1 | 'There is always'->'Modification'
15->6 pred.arg.2 | 'jam up'->'us'
15->5 pred.arg.1 | 'jam up'->'they'
16->7 pred.arg.1 | 'look like'->'us'
16->17 pred.arg.2 | 'look like'->'are not'
17->8 pred.arg.1 | 'are not'->'we'
17->18 pred.arg.2 | 'are not'->'numero uno'
19->10 variable | 'Modification'->'the'
19->1 body | 'Modification'->'could'

--- after ---
" There is always the possibility they could jam us up, make us look like we are not numero uno, " Pike said.
NODES
0: 'said' [[24, 24]] FactualPredicator
1: 'could' [[7, 7]] FactualPredicator
2: ',' [[11, 11]] ConjunctionalFunctor
3: 'make' [[12, 12]] FactualPredicator
4: 'Pike' [[23, 23]] NominalConstant
5: 'they' [[6, 6]] NominalConstant
6: 'us' [[9, 9]] NominalConstant
7: 'us' [[13, 13]] NominalConstant
8: 'we' [[16, 16]] NominalConstant
9: 'possibility' [[5, 5]] NominalConstant
10: 'the' [[4, 4]] DeterminerConstant
11: ',' [[21, 21]] PunctuationalConstant
12: '.' [[25, 25]] PunctuationalConstant
13: '" "' [[0, 0], [22, 22]] PunctuationalConstant
14: 'is always' [[2, 3]] FactualPredicator
15: 'jam up' [[8, 8], [10, 10]] FactualPredicator
16: 'look like' [[14, 15]] FactualPredicator
17: 'are not' [[17, 18]] FactualPredicator
18: 'numero uno' [[19, 20]] NominalConstant
19: 'Modification' 'Modification' ModificationalFunctor
20: 'There' [[1, 1]] NominalConstant
EDGES
0->4 pred.arg.1 | 'said'->'Pike'
0->13 pred.arg.2 | 'said'->'" "'
1->2 pred.arg.2 | 'could'->','
1->5 pred.arg.1 | 'could'->'they'
2->3 func.arg | ','->'make'
2->15 func.arg | ','->'jam up'
3->5 pred.arg.1 | 'make'->'they'
3->16 pred.arg.2 | 'make'->'look like'
10->9 as:modification | 'the'->'possibility'
11->14 as:punctuation | ','->'is always'
12->0 as:punctuation | '.'->'said'
13->11 as:punctuation | '" "'->','
14->19 pred.arg.2 | 'is always'->'Modification'
15->6 pred.arg.2 | 'jam up'->'us'
15->5 pred.arg.1 | 'jam up'->'they'
16->7 pred.arg.1 | 'look like'->'us'
16->17 pred.arg.2 | 'look like'->'are not'
17->8 pred.arg.1 | 'are not'->'we'
17->18 pred.arg.2 | 'are not'->'numero uno'
19->10 variable | 'Modification'->'the'
19->1 body | 'Modification'->'could'
14->20 pred.arg.1 | 'is always'->'There'
```

### train:1297 `merged_simple_copula_adverb_split`

So be very careful and some feeds could be toxic to rabbits and safe for chickens.

```text
So be very careful and some feeds could be toxic to rabbits and safe for chickens.
NODES
0: 'So' [[0, 0]] LogicalPredicator
1: 'and' [[4, 4]] ConjunctionalFunctor
2: 'safe' [[13, 13]] FactualPredicator
3: 'for' [[14, 14]] ModificationalFunctor
4: 'to' [[10, 10]] ModificationalFunctor
5: 'chickens' [[15, 15]] NominalConstant
6: 'rabbits' [[11, 11]] NominalConstant
7: 'feeds' [[6, 6]] NominalConstant
8: 'some' [[5, 5]] DeterminerConstant
9: '.' [[16, 16]] PunctuationalConstant
10: 'could and' [[7, 7], [12, 12]] FactualPredicator
11: 'be very careful' [[1, 3]] FactualPredicator
12: 'be toxic' [[8, 9]] FactualPredicator
13: 'Missing' 'Missing' NominalConstant
EDGES
0->1 pred.arg.2 | 'So'->'and'
1->10 func.arg | 'and'->'could and'
1->11 func.arg | 'and'->'be very careful'
2->8 pred.arg.1 | 'safe'->'some'
3->5 body | 'for'->'chickens'
3->2 variable | 'for'->'safe'
4->6 body | 'to'->'rabbits'
4->12 variable | 'to'->'be toxic'
8->7 as:modification | 'some'->'feeds'
9->0 as:punctuation | '.'->'So'
10->3 pred.arg.2 | 'could and'->'for'
10->4 pred.arg.1 | 'could and'->'to'
12->8 pred.arg.1 | 'be toxic'->'some'
11->13 pred.arg.1 | 'be very careful'->'Missing'

--- after ---
So be very careful and some feeds could be toxic to rabbits and safe for chickens.
NODES
0: 'So' [[0, 0]] LogicalPredicator
1: 'and' [[4, 4]] ConjunctionalFunctor
2: 'safe' [[13, 13]] FactualPredicator
3: 'for' [[14, 14]] ModificationalFunctor
4: 'to' [[10, 10]] ModificationalFunctor
5: 'chickens' [[15, 15]] NominalConstant
6: 'rabbits' [[11, 11]] NominalConstant
7: 'feeds' [[6, 6]] NominalConstant
8: 'some' [[5, 5]] DeterminerConstant
9: '.' [[16, 16]] PunctuationalConstant
10: 'could and' [[7, 7], [12, 12]] FactualPredicator
11: 'be careful' [[1, 1], [3, 3]] FactualPredicator
12: 'be toxic' [[8, 9]] FactualPredicator
13: 'Missing' 'Missing' NominalConstant
14: 'very' [[2, 2]] ModificationalConstant
EDGES
0->1 pred.arg.2 | 'So'->'and'
1->10 func.arg | 'and'->'could and'
1->14 func.arg | 'and'->'very'
2->8 pred.arg.1 | 'safe'->'some'
3->5 body | 'for'->'chickens'
3->2 variable | 'for'->'safe'
4->6 body | 'to'->'rabbits'
4->12 variable | 'to'->'be toxic'
8->7 as:modification | 'some'->'feeds'
9->0 as:punctuation | '.'->'So'
10->3 pred.arg.2 | 'could and'->'for'
10->4 pred.arg.1 | 'could and'->'to'
12->8 pred.arg.1 | 'be toxic'->'some'
11->13 pred.arg.1 | 'be careful'->'Missing'
14->11 as:modification | 'very'->'be careful'
```

### train:1299 `merged_simple_copula_adverb_split`

I'm very happy with the piano lessons Mrs. Lynda Mcmanus taught me.

```text
I'm very happy with the piano lessons Mrs. Lynda Mcmanus taught me.
NODES
0: 'with' [[5, 5]] ModificationalFunctor
1: 'taught' [[13, 13]] FactualPredicator
2: 'I' [[0, 0]] NominalConstant
3: 'me' [[14, 14]] NominalConstant
4: 'lessons' [[8, 8]] NominalConstant
5: 'piano' [[7, 7]] NominalConstant
6: 'the' [[6, 6]] DeterminerConstant
7: 'Mcmanus' [[12, 12]] NominalConstant
8: 'Mrs .' [[9, 10]] NominalConstant
9: 'Lynda' [[11, 11]] NominalConstant
10: '.' [[15, 15]] PunctuationalConstant
11: "' m very happy" [[1, 4]] FactualPredicator
12: 'Modification' 'Modification' ModificationalFunctor
13: 'Expression' 'Expression' ExpressionFunctor
EDGES
0->11 variable | 'with'->"' m very happy"
0->12 body | 'with'->'Modification'
1->3 pred.arg.2 | 'taught'->'me'
1->6 pred.arg.3 | 'taught'->'the'
1->8 pred.arg.1 | 'taught'->'Mrs .'
5->4 as:modification | 'piano'->'lessons'
6->5 as:modification | 'the'->'piano'
8->13 as:modification | 'Mrs .'->'Expression'
10->0 as:punctuation | '.'->'with'
11->2 pred.arg.1 | "' m very happy"->'I'
12->6 variable | 'Modification'->'the'
12->1 body | 'Modification'->'taught'
13->9 func.arg | 'Expression'->'Lynda'
13->7 func.arg | 'Expression'->'Mcmanus'

--- after ---
I'm very happy with the piano lessons Mrs. Lynda Mcmanus taught me.
NODES
0: 'with' [[5, 5]] ModificationalFunctor
1: 'taught' [[13, 13]] FactualPredicator
2: 'I' [[0, 0]] NominalConstant
3: 'me' [[14, 14]] NominalConstant
4: 'lessons' [[8, 8]] NominalConstant
5: 'piano' [[7, 7]] NominalConstant
6: 'the' [[6, 6]] DeterminerConstant
7: 'Mcmanus' [[12, 12]] NominalConstant
8: 'Mrs .' [[9, 10]] NominalConstant
9: 'Lynda' [[11, 11]] NominalConstant
10: '.' [[15, 15]] PunctuationalConstant
11: "' m happy" [[1, 2], [4, 4]] FactualPredicator
12: 'Modification' 'Modification' ModificationalFunctor
13: 'Expression' 'Expression' ExpressionFunctor
14: 'very' [[3, 3]] ModificationalConstant
EDGES
0->14 variable | 'with'->'very'
0->12 body | 'with'->'Modification'
1->3 pred.arg.2 | 'taught'->'me'
1->6 pred.arg.3 | 'taught'->'the'
1->8 pred.arg.1 | 'taught'->'Mrs .'
5->4 as:modification | 'piano'->'lessons'
6->5 as:modification | 'the'->'piano'
8->13 as:modification | 'Mrs .'->'Expression'
10->0 as:punctuation | '.'->'with'
11->2 pred.arg.1 | "' m happy"->'I'
12->6 variable | 'Modification'->'the'
12->1 body | 'Modification'->'taught'
13->9 func.arg | 'Expression'->'Lynda'
13->7 func.arg | 'Expression'->'Mcmanus'
14->11 as:modification | 'very'->"' m happy"
```
