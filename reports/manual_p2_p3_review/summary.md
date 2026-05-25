# P2/P3 Manual Review

- Reviewed queue: 545
- Fixed: 16
- Left unchanged: 529
- P2 fixed/total: 13/264
- P3 fixed/total: 3/281

## Fixed

- `train:512`: make the coordinated name list the subject of `appeared`, not the comma — On July 26 , 2006 , Tanika Ray , Rihanna , as well as other celebrities , appeared on the show .
- `train:1680`: merge phrasal predicate `backed up` — His psychiatrist backed him up .
- `train:1710`: merge phrasal predicate `picked up` — Somebody picked them up .
- `train:3726`: make quoted `paste` the object of `select` rather than the quote marks — From the' edit' menu select' paste'.
- `train:3777`: treat `old` as result/state complement of `getting` — we 're getting old .
- `train:4499`: split adverb `truely` out of the copular predicate — Holly is truely the best hairstylist!
- `train:5271`: convert question-id key-value record to Attribute — Question ID: 420072
- `train:8996`: attach the fronted `In the second stage` modifier to the completed-event — In the second stage , Sobha Sapphire , Sobha Jade and Sobha City Mall will be completed .
- `train:9784`: make `firing off flares` a modifier of the flying event with the helicopter subject — A few helicopters flew overhead , firing off flares , but there was no shooting from the air .
- `train:9802`: classify standalone congratulations as an interjection — CONGRATULATIONS!!!!!!!
- `train:10533`: make the starting-place phrase the copular complement and mark `to look` subject implicit — It is a starting place to look
- `train:10551`: represent `Hello, Bill!` as interjection plus vocative — Hello, Bill!
- `train:11051`: attach `to Betsy` to the goodbye object and make `to say` the going complement — we ' re going to say good - bye to betsy .
- `train:11127`: treat `negative` as result/state complement of `came back` — the x - rays all came back negative .
- `dev:381`: convert attendees key-value record to Attribute — Attendees: Pam Butler
- `test:100`: treat `better` as result/state complement of `got` — But after they finished with the hormones , I gradually got better .
