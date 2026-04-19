parsing_guidelines = """

For each chunk, you should first identify the TerminalPunct, BracketPunct. And then, you should parse the chunk into sub-chunks based on the gramma defined above on the remaining part of the chunk, until the sub-chunks are leaf elements in the gramma.

When parsing the sentence, you should consider the following rules:

1. Do not missing any chararcters and words in the sentence. Any character and word in the sentence should be in the parsed result.

2. When parsing, you should consider the semantic structure of the sentence carefully, and identify different semantic-levels of the structure. Especially, you should pay attention to the semantic-levels of the multiple subordinating sentences modifying the same sentence. Furthermore, if a sentence is consisted of multiple sub-sentences unrelated to each other, just parse it as a coordinating conjunction of sub-sentences.

3. If the chunk has TerminalPunct or BracketPunct, add them into the parsing result; otherwise, ommit them.

4. Type, role and relations is the center of the parsing result. 
   4.1 Each chunk should have a unique type, describing its own syntactic structure, like "SNP", "SVP", "SADV", "SADJ", etc. The type of a chunk should be included in the parsing result using the "TYPE" field, for example, {{"The man": {{"TYPE": "SNP"}} }}. 
   
   4.2 In some situation, the role of a chunk in the parent chunk can not be infered directly from its type. So it should be identified explicitly. The role of a chunk in its parent chunk includes "SBJ", "PRED" (predicate), "OBJ" (object), "DOBJ" (direct object), "MOD" (modifier), "APPOS" (apposition), etc. The roles of chunks should be included in the parsing result using the "ROLE" field, for example, in the sentence "The man is running", the role of "The man" is "SBJ", so {{"The man": {{"ROLE": "SBJ"}}}} should be added to the parsing result of "The man".  
   
   4.3 In some situation as the relative clause and adverbial modification phrase with verb in it, the relationship between chunks are vital for accurately understanding the sentence. The relationships include like "SBJ", "OBJ", "DOBJ", "REF", "APPOS", "MOD", and the reverse relationship "as:SBJ", "as:OBJ", "as:DOBJ", "as:REF", "as:APPOS", "as:MOD", etc. The relationships should be included in the parsing result using the "RELATION" field, especially in the relative clause, and  adverbial modification phrase with verb in it, for example:
        - For chunk "The house which she bought", the relative noun "which":
            - is the object of the predicate "bought";
            - is the reference of the noun "The house";
            so {{"RELATION": [("as:DOBJ", "bought"), ("as:REF", "The house")]}} should be added to the parsing result of "which".
        - For chunk "The book which I gave you yesterday", the relative noun "which" 
            - is the direct object of the predicate "gave";
            - is the reference of the noun "The book";
            so {{"RELATION": [("as:DOBJ", "gave"), ("as:REF", "The book")]}} should be added to the parsing result of "which".
        - For chunk "The house in which she was born", the relative noun "which" 
            - is the object of the preposition "in";
            - is the reference of the noun "The house";
            so {{"RELATION": [("as:DOBJ", "in"), ("as:REF", "The house")]}} should be added to the parsing result of "which".
            - MODIFIER: the relative noun is a modifier of some component (predicate, preposition, etc.) in the relative clause. For example:
        - For chunk "The house where she was born", the relative noun "where" 
            - is a modifier of the predicate "was born";
            - is the reference of the noun "The house";
            so {{"RELATION": [("as:MOD", "was born"), ("as:REF", "The house")]}} should be added to the parsing result of "where".
        - for chunk "The book I gave you yesterday", "The book" is the direct object of "gave", so  {{"RELATION": [("as:DOBJ", "gave")]}} should be added to the parsing result of "The book".
        - For chunk "Being exhausted, he went to bed early.", the subject of the verb "Being" is "he"; so when parsing "Being exhausted," you should add infomation of {{"RELATION": [("SBJ": "he")]}} to the parsing result of verb phrase "Being".

4. For conjunction of chunks, you should complete the coordinating conjunction phrase with argument placeholders, which are corresponding to the position of the coordinating sub-chunks. For each sub-chunk, you should add the argument placeholder to the parsing result. Furthermore, you should add the punctuation marks like ",", ";" that splitting the coordinating sub-chunks to the coordinating conjunction phrase. For example:
   - For chunk "The man, the woman and the boy", the parsing result should contain the following information:
    {{
        "The man, the woman and the boy": {{
            "TYPE": "NPC",
            "CONPONENTS": {{
                "{{1}},{{2}}and{{3}}": {{"TYPE": "CCP"}},
                "The man": {{"PLACEHOLDER": "{{1}}", "TYPE": "SNP", ...}},
                "the woman": {{"PLACEHOLDER": "{{2}}", "TYPE": "SNP", ...}},
                "the boy": {{"PLACEHOLDER": "{{3}}", "TYPE": "SNP", ...}}
            }}
        }}
    }}

5. For chunk with subordinating clause, you should complete the subordinating phrase with argument placeholders, where  the argument placeholder {{1}} should correspond to the main sentence, while {{2}} should correspond to the subordinating sentence. For subordinating sentence, you should add the argument placeholder to the parsing result. Furthermore, you should add the punctuation marks like ",", ";" that splitting the subordinating clause to the subordinating conjunction phrase. For example:
   - For chunk "I'm happy because I passed the exam", the parsing result should contain the following information:
    {{
        "I'm happy because I passed the exam": {{
            "TYPE": "SUBSMS",
            "CONPONENTS": {{
                "I'm happy": {{"TYPE": "SS", ...}},
                "because I passed the exam": {{
                    "TYPE": "SUBS",
                    "CONPONENTS": {{
                        "{{1}}because{{2}}": {{"TYPE": "SCP"}},
                        "I passed the exam": {{"PLACEHOLDER": "{{2}}", "TYPE": "SS", ...}}
                    }}
                }}
            }}
        }}
    }}

6. After a SNP is discovered, you should futher identify whether there exists modificational sub-structures inside it. If there exists, there may be multiple modification components, and one center modified component. The role of the center modified component should be "HEAD", and the roles of the modification components should be "MOD". For SNP with inner modificational sub-structures, there must be one and only one component with "HEAD" role.

7. Difference between LIST, EXPRESSION, and Conjunction:
   - LIST: a list of components, but its self is not a component with single meaning, and there is no further relation between the components except the order in the list.
   - EXPRESSION: an expression is an entity with single meaning, consisted of multiple sub components with optional operator.
   - Conjunction: a conjunction is a sequence of components, which are connected by optional coordinating conjunctions like ",", ";" etc. The components in the conjunction are related to each other by sharing some same attributes or relations/behaviors.

"""

