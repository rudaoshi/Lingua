
# ending_punct_definition = """
# The EndingPunct of a chunk is the last punctuation word that means the end of the entire chunk, like ".", ",", "?", "!", etc. 
#    - If the chunk has such a punctuation word, let EndingPunct be the punct.
#    - Otherwise, let EndingPunct be None.
#    - Examples:
#      - For chunk "I've already seen the movie, so I'm going to see a new one.":
#         - EndingPunct is "."
#      - For chunk "I've already seen the movie, so I'm going to see a new one":
#         - EndingPunct is None
# """


# simple_complete_sentence_definition = """
# A simple complete sentence is defined as a chunk that :

# 1) describes one subject and it's predicate (may involve objects);
# 2) the subject and it's predicate are not modified as a whole by adverbial clauses or relative clauses.
#     - For example, "Although I have failed many times, finally I made it." is not a simple complete phrase, since there is an adverbial modification structure that the phrase "I have failed many times"  modifies the entire clause of "finally I made it";
#     - For example, "I won the game, which made me very happy." is not a simple complete phrase, since there is an relative clause "which made me very happy" that modifies the entire phrase of "I won the game";
# 3) may have dependent clauses that modify the suject or the objects of the subject's behavior.
#     - For example, "She went to the store where I lost my wallet" is a simple sentence;
# 3) may have partial verbial phrases that modify the subject's behavior.
#     - For example, "Walking down the street, I saw an old friend." is a simple sentence:
#         - "Walking down the street" is a partial verbial phrase that modifies the whole clause "I saw an old friend"
# 4) may have a compound subject, compound predicate or compound object. For example:
#     - Sentence with compound subject: "The man in the hat and the woman in the dress are walking to the park"
#     - Sentence with compound object: "I'm teaching both the graduate and the undergraduate students"
#     - Sentence with compound predicate: "I will go to the park and buy some ice cream"
# """

# complex_sentence_definition = """
# A complex sentence is defined as a chunk that contain conjunction/modification/complement relationships between the simple complete sentences or complex sentences.

# 1) conjunction of two or more simple complete sentences or complex sentences which are connected by conjunctions like "and", "but", "or", etc. or just a comma;
# 2) one main simple complete sentence/complex sentence and several subordinate simple complete sentences/complex sentences that modify the whole main simple complete sentence/complex sentence.
# """

gramma_definition = """

In the following semantic gramma definitions, we use "->" means sequentially occurance, "-+" means involvement without considering the order. "*" means zero or more occurance, "+" means one or more occurance, "?" means optional. "#" means the start of the comment, which is not part of the gramma. 

Types: The type of a chunk.

* DET (determiner) # means a determiner, such as "the", "a", "an", "this", "that", "these", "those", etc.
* AUX_VERB (auxiliary verb) # means an auxiliary verb, such as "do", "does", "did", "have", "has", "had", "will", "would", "should", "can", "could", "may", "might", "must", "ought", "shall", "should", "will", "would", etc.
* BE_VERB (be verb) # means a be verb, such as "is", "was", "were", "be", "being", "been", etc.
* SNP (simple noun phrase)  # means a noun phrase with adjective modifications but no prepositional modification
* SADJ (simple adjective modification) # means a phrase consists of adjective words
* SADV (simple adverbial modification) # means a phrase consists of adverbial words
* AMPW (adverbial modified prepositional words)  # means prepositional words, may be modified by adverbs, such as "deep in"
* SVP (simple verb phrase) # verb phrase with adverbial modifications or phrasal verbs
* SLVP (simple linking verb phrase) # verb phrase with linking verb, such as be-verb , "look", "sound", "became", "remain", etc. followed by a adjective phrase.
* CCP (coordinating conjunction phrase) # means a phrase that connects two or more components, including conjunctions like "and", "but", "or", "not only...but also...", "neither...nor...", "either...or...", etc. or just a comma. Notice that the CCP may be ommited. If exists, it should not contain any other kind of words like noun, verb, adj, adv, and prep.
* SCP (subordinating conjunction phrase) # means a phrase that connects a subordinate clause to a main clause, including conjunctions like "because", "although", "while", etc.
* NUMBER (number) # means a number, such as "one", "two", "three", "1", "2", etc.
* INTJ (interjection) # means a interjection, such as "um", "uh", "like", "you know", etc.
* SYM (symbol) # means a symbol, such as "$", "%", "§", "©", "+", "-", "×", "÷", "=", "<", ">", etc.
* OP (operator) # means a phrase that connects two or more sub-component into one-whole entity/expression, including "+", "-", "*", "/", "×", "÷", "=", "<", ">", "." etc. Notice that the CCP may be ommited. 

* NS_COMPLEMENTIZER (noun sentence complementizer) # means a complementizer that introduces a noun sentence, such as "that", "which", "who", "whom", "whether", etc.
* APPOS_CONJ (appositive conjunction) # means a conjunction that connects an appositive to a noun phrase, such as "that", "which", "who", "whom", etc. or a punctuation like ",", "-", etc.

* SVPC (simple verb phrase conjunction) = CCP[SVP+]

* COMP_NS (complementizer noun sentence) = (NS_COMPLEMENTIZER)? -> S # a noun sentence that is introduced by a complementizer ("that",  "whether", "if")

* WH_PHRASE (wh phrase) # a phrase that is introduced by wh words, such as "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "what time", "how much", "whose book", etc.

* WH_NS (wh noun sentence) = (AMPW)? -> WH_PHRASE -> (S | VP) # a wh sentence is a sentence that is introduced by a  or has the same form of relative sentence, generally used as a subject or object of a predicate

* NS (noun sentence) = COMP_NS | WH_NS  # a noun sentence 

* VP (verb phrase)  = (SPRED -+ (IOBJ)? -+ (DOBJ)? -+ (OBJ_COMPLEMENT)?  -> (MOD)*) | (SLVP  -> (PP|PPC)*  -+ (SADJ)*)
* VPC (verb phrase conjunction) = CCP[VP+] 

* ADV (adverbial modification) = SADV | VP | NP   # simple adverbial modification, or adverbial modification by a verb phrase generally in -ing form or -ed form.
* ADVC (adverbial modification conjunction) = CCP[ADV+]

* ADJC (adjective modification conjunction) = CCP[SADJ+]

* PP (prepositional phrase) = AMPW -> (NP | VP)
* PPC (prepositional phrase conjunction) = CCP[PP+]
* PPG (prepositional phrase group) = PP | PPC

* APPOS (appositive) = (APPOS_CONJ)? -> (S | NP) # means a sentence that is an appositive of previous noun phrase.

* LIST (list) =  (S | NP) +  # means a chains of comparable but independent items. Becareful that the list is different from the conjunction, as the union of the elements is not semantically important.

* POSS_MARKER (possessive marker) = "'s"
* POSS_SNP (possessive noun phrase) = OWNER -> POSS_MARKER -> (POSSESSION?)

* ACM_MARKER (attributive compound modifier marker) = "-" # only used in ACM structure 
* ACM_OBJ (attributive compound modifier's object) = (SNP) -> (ACM_MARKER)?
* ACM_OBJC (attributive compound modifier's object conjunction) = CCP[ACM_OBJ+]
* ACM (attributive compound modifier) = (ACM_OBJ| ACM_OBJC) -> (SVP | SADJ)
* ACMC (attributive compound modifier conjunction) = CCP[ACM+]

* ATTR (attribute) = (ATTR_NAME)? -> ATTR_VALUE  # means the attribute of a noun phrase, etc.
* NP (noun phrase) =  (ACM|ACMC)? -> HEAD -> (APOSS)? -+ RS * -+ (PP|PPC) *  -+ ATTR *  | POSS_SNP 
* NPC (noun phrase conjunction) = CCP[NP+]

* SS (simple sentence) = (SBJ)? -+ PRED -> (MOD)? # means a simple sentence, including a subject, a simple verb phrase, and an object. Notice that the sentence may be inverted word order.

* THAT_WORD (that word) = "that" 
* THAT_RS (that relative sentence) = (THAT_WORD)? -> S # a relative sentence that is introduced by "that"
* WH_RS (wh relative sentence) = (AMPW)? -> WH_PHRASE -> (S | VP) # a relative sentence that is introduced by "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "what time", "how much", etc.

* RS (relative sentence) = WH_RS | THAT_RS # wh sentence used as a relative clause/sentence 

* RSC (relative sentence conjunction) = CCP[RS+]

* SUBS (subordinate sentence) = (SCP -> S) 
* SUBSC (subordinate sentence conjunction) = CCP[SUBS+]

* SC (sentence conjunction)  = CCP[S+]  # means a set of sentences connected by a conjunction phrase.

* RSMS (relative sentence modified sentence) =  S -+ (RS | RSC)
* SUBSMS (subordinate sentence modified sentence) =  S -+ (SUBS | SUBSC)

* CS (complex sentence) =  SC | RSMS | SUBSMS

* S (sentence) = SS | CS  # means a sentence, including simple sentence and complex sentence. Maybe quoted or split into parts.

* APOSS (appositive sentence) = (APPOS_CONJ)? -> (S | NP) # means a sentence that is an appositive of previous noun phrase.

* Q_MARK (question mark) = "?"
* WH_QUESTION (wh question) = (AMPW)? -> (WH_PHRASE)? -> (S | VP) -> (Q_MARK)? # means a question that is introduced by a wh phrase.
* YN_QUESTION (yes/no question) = (AUX_VERB|BE_VERB) -> S -> (Q_MARK)? # means a yes/no question that is introduced by a be verb or auxiliary verb.

* EXPRESSION (expression) = OP[(S | NP | NUMBER | EXPRESSION)+] # means a expression that combines multiple sub-components into one-whole entity/expression with optional operator.

* OTHER (other) # means other types that do not fit into the above categories.

Roles: The role of a chunk in its parent chunk.

* SBJ (subject) = NP | NPC | NS | S | SC | RSS | RSC # means the subject of a predicate 
* PRED (predicate) = VP | VPC   # means the predicate of a sentence
* SPRED (simple predicate) = SVP | SLVP | BE_VERB | AUX_VERB # means the simple predicate of a sentence only contains a verb phrase and no other components like object, complement, prepositional phrase, etc.
* IOBJ (indirect object)  = NP | NPC | NS | S | SC | RSS | RSC   # means the indirect object of a predicate
* DOBJ (direct object) = NP | NPC | NS | S | SC | RSS | RSC   # means the direct object of a predicate
* OBJ_COMPLEMENT (object complement) = NP | NPC | VP | VPC | SADJ | ADJC # means the object complement 
* MOD (modifier) = ADV | ADVC ｜PP | PPC # means the modifier of a predicate
* VOC (vocative) = NP | NPC  # means the dialogue participant
* DISCOURSE (discourse) # means a interjections or filler word, such as "um", "uh", "like", "you know", etc.
* INDEX (index) # means the index of a list, such as "first", "second", "third", "1", "2", "(a)", "(b)", etc.
* ATTR_NAME (attribute key) = SNP | SYM  # means the name of attribute, such as "email" or "phone"
* ATTR_VALUE (attribute value) = SNP | SYM | NUMBER | EXPRESSION | OTHER # means the value of attribute, such as "1234567890", "1234567890@163.com", "1234567890-1234567890", etc.
* OWNER (owner) = NP | NPC  # means the owner of a possessive noun phrase
* POSSESSION (possession) = NP | NPC  # means the possession of a possessive noun phrase
* HEAD (head) = SNP  # means the head of a noun phrase

"""
 
punctuation_definition = """


The BracketPunct of a chunk is a pair of punctuation words used to enclose the other content of the chunk, like "\"", "\"", "(", ")", "[", "]", "{", "}", etc.
   - If the chunk has no such punctuation words, the BracketPunct is None.
   - If only one of the pair of punctuation words is present, the BracketPunct is that punctuation word.
   - Examples:
     - For chunk "(The man in the hat)":
        - BracketPunct is "()"
     - For chunk "\"I'm going to the park\"":
        - BracketPunct is "\"\""
     - For chunk "\"The man in the hat":
        - BracketPunct is "\""
     - For chunk "I'm going to the park":
        - BracketPunct is None

The TerminalPunct of a chunk is the last punctuation word that means the end of the entire chunk, like ".", ",", "?", "!", etc. 
   - If the chunk has no such punctuation word, TerminalPunct is None.
   - The TerminalPunct should not be the BracketPunct, even if it is at the last and not paired. 
   - Examples:
     - For chunk "I've already seen the movie, so I'm going to see a new one.":
        - TerminalPunct is "."
     - For chunk "I've already seen the movie,":
        - TerminalPunct is ","
     - For chunk "I've already seen the movie, so I'm going to see a new one":
        - TerminalPunct is None

The StartPunct of a chunk is the first punctuation word that appears at the start of the entire chunk, usually caused by incompleteness of the chunk.
   - If the chunk has no such punctuation word, StartPunct is None.
   - The StartPunct should not be the BracketPunct, even if it is at the first and not paired.
   - Examples:
     - For chunk "I've already seen the movie, so I'm going to see a new one.":
        - StartPunct is None
     - For chunk "; I've already seen the movie,":
        - StartPunct is ";"
"""

"""
* PUNCT (punctuation) # means a punctuation, such as ",", ".", "!", "?", etc.
"""