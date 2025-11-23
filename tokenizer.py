"""
================================================================================
tokenizer.py - Converting Text into Numbers
================================================================================

WHAT IS A TOKENIZER?
--------------------
A tokenizer is like a translator that converts words (text) into numbers.
Computers and neural networks can only work with numbers, not words. So before
we can process text, we need to convert each word into a unique number (ID).

Think of it like a dictionary where:
- Each word has a page number (ID)
- To look up a word, you use its page number
- To find what a page number means, you look it up in the dictionary

WHY DO WE NEED THIS?
--------------------
Imagine you want to teach a computer to understand "Who created Python?".
The computer doesn't understand words, but it can understand:
- 0, 1, 7, 3 (these numbers represent "Who", "created", "Python", "?")

By converting text to numbers:
1. We can feed it into mathematical operations (neural networks)
2. We can do calculations and patterns on the data
3. We can train the model to learn relationships between words

THIS IS A SIMPLIFIED VERSION
-----------------------------
Real-world systems (like ChatGPT) use much larger vocabularies:
- They have vocabularies with 50,000+ tokens
- They use "subword" tokenization (breaking words into smaller pieces)
- They handle unknown words with special tokens

We use a tiny vocabulary (just 9 words) so you can:
- Easily see every step of the process
- Understand exactly what's happening
- Debug problems by printing everything

THE TWO KEY CONCEPTS HERE:
--------------------------
1. VOCAB - An ordered list of all words we know (our "dictionary")
2. Mappings - Converting between words and numbers (two-way translation)

EXAMPLE WALKTHROUGH:
--------------------
If our VOCAB is: ['Who', 'created', 'JavaScript', '?', 'Brendan']

Then:
- "Who" gets ID 0 (first position)
- "created" gets ID 1 (second position)
- "JavaScript" gets ID 2 (third position)
- "?" gets ID 3 (fourth position)
- "Brendan" gets ID 4 (fifth position)

When we tokenize "Who created JavaScript ?"
We get: [0, 1, 2, 3]

IMPORTANT LIMITATION:
---------------------
If you type a word that's NOT in VOCAB, this tokenizer will skip it!
Example: "Hello world" would return [] (empty) because neither word is in VOCAB.

In professional systems, there's usually an "UNK" (unknown) token to handle
words not in the vocabulary. We skip that here to keep things simple.

ADDING NEW WORDS:
-----------------
If you want to add new words:
1. Add them to the VOCAB list below
2. Restart Python (so the model rebuilds with the new vocabulary size)
3. Re-train the model (it needs to learn about the new words)
"""

# ============================================================================
# THE VOCABULARY - Our "Dictionary" of Known Words
# ============================================================================
# This is a Python list - an ordered collection of words/tokens.
# - Each position in the list has an index (starting from 0)
# - Position 0: "Who", Position 1: "created", Position 2: "JavaScript", etc.
# - The order matters! It never changes once set.

VOCAB = ["Who", "created", "JavaScript", "?", "Brendan", "Guido", "Linus", "Python", "Linux"]

# VOCAB has 9 items (words/tokens). We call this the "vocabulary size".
# Index:  0      1          2             3    4          5        6        7         8


# ============================================================================
# MAPPING #1: tok2id - From Words to Numbers
# ============================================================================
# This creates a Python "dictionary" (like a lookup table or phone book).
# 
# Think of it like: {"Word": ID_number}
# Example result: {"Who": 0, "created": 1, "JavaScript": 2, ...}
#
# How it works:
# - `enumerate(VOCAB)` goes through VOCAB and gives us pairs: (index, word)
#   Example: (0, "Who"), (1, "created"), (2, "JavaScript"), ...
# - `{tok: i for i, tok in enumerate(VOCAB)}` is a "dictionary comprehension"
#   It builds a dictionary where each word (tok) maps to its index (i)
#
# Why we need this:
# - When we see "Who" in text, we can quickly look up: tok2id["Who"] -> 0
# - This converts words into numbers the computer can work with

tok2id = {tok: i for i, tok in enumerate(VOCAB)}

# ============================================================================
# MAPPING #2: id2tok - From Numbers Back to Words
# ============================================================================
# This is the reverse mapping: {ID_number: "Word"}
# Example result: {0: "Who", 1: "created", 2: "JavaScript", ...}
#
# Why we need this:
# - After the model predicts a number (like 4), we can look up: id2tok[4] -> "Brendan"
# - This lets us convert the model's numeric predictions back into readable words
# - Essential for displaying results to humans!

id2tok = {i: tok for i, tok in enumerate(VOCAB)}


# ============================================================================
# THE TOKENIZER FUNCTION - Converting Text to Numbers
# ============================================================================
def tokenizer(text):
    """
    Convert a text string (like "Who created Python ?") into a list of numbers.
    
    PARAMETERS (Input):
    -------------------
    text : str (string = text)
        A sentence or phrase you want to convert. Example: "Who created Python ?"
    
    RETURNS (Output):
    -----------------
    list of integers
        A list of numbers representing each word. Example: [0, 1, 7, 3]
    
    HOW IT WORKS (Step-by-Step):
    -----------------------------
    1. text.split() - Splits the text into individual words using spaces
       Example: "Who created Python ?" becomes ["Who", "created", "Python", "?"]
    
    2. for n in text.split() - Loop through each word one at a time
    
    3. if n in tok2id - Check if this word exists in our vocabulary
       - If YES: proceed to step 4
       - If NO: skip this word (it's not in our dictionary)
    
    4. tok2id[n] - Look up the ID number for this word
       Example: tok2id["Who"] gives us 0
    
    5. [tok2id[n] for n in text.split() if n in tok2id] - "List comprehension"
       This is Python shorthand for:
       ```
       result = []
       for n in text.split():
           if n in tok2id:
               result.append(tok2id[n])
       return result
       ```
    
    FULL EXAMPLE:
    -------------
    Input:  "Who created Python ?"
    
    Step 1 - Split into words:
        ["Who", "created", "Python", "?"]
    
    Step 2 - Convert each word to its ID:
        "Who" -> tok2id["Who"] -> 0
        "created" -> tok2id["created"] -> 1
        "Python" -> tok2id["Python"] -> 7
        "?" -> tok2id["?"] -> 3
    
    Output: [0, 1, 7, 3]
    
    WHAT IF A WORD ISN'T IN VOCAB?
    -------------------------------
    The `if n in tok2id` check means unknown words are simply skipped.
    
    Example: tokenizer("Hello world")
    - "Hello" is not in VOCAB -> skipped
    - "world" is not in VOCAB -> skipped
    - Result: [] (empty list)
    
    This is a simplification! Real tokenizers handle unknown words better.
    """
    # This one line does all the work explained above!
    return [tok2id[n] for n in text.split() if n in tok2id]