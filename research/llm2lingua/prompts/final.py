final_prompt_template = """

You are an expert in linguistics.

Given a gramma defined as follows: 

```
{gramma_definition}
```

And a punctuation definition as follows:

```
{punctuation_definition}
```

You will be given a sentence, and your task is to parse the sentence with above gramma and punctuation definition.

The parsing guidelines are as follows:

```
{parsing_guidelines}
```

Output the parsing result as a json object following the examples below. Be careful to escape the quotation marks in the sentence in the output json object. And make sure the json attribute names are enclosed in double quotes.

{examples}

You will be given a sentence. Parse the sentence with above instructions.

"""

