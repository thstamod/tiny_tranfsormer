"""
================================================================================
embeddings.py - Converting Numbers into Meaningful Vectors
================================================================================

WHAT ARE EMBEDDINGS?
--------------------
Remember: the tokenizer converts words into numbers (IDs).
But a simple number like "5" doesn't tell the computer anything about the MEANING
of a word. 

Embeddings convert these simple IDs into "rich" number vectors that can capture
meaning and relationships.

ANALOGY - The GPS Coordinates of Words:
----------------------------------------
Imagine describing locations:
- BAD: Just saying "Location 1", "Location 2", "Location 3"
  (These numbers tell you nothing about where things are)

- GOOD: Using GPS coordinates like [40.7128, -74.0060] for New York
  (These numbers tell you the actual position, how far places are from each other)

Word embeddings work the same way!
- BAD: Word "created" is just ID 1
- GOOD: Word "created" is vector [0.2, -0.5, 0.8, 0.1, -0.3, 0.6, 0.4, -0.2]

Similar words get similar vectors:
- "Python" and "JavaScript" (both programming languages) would have similar vectors
- "Python" and "?" (very different) would have very different vectors

THE BIG IDEA:
-------------
We convert:
  Word -> ID (just a number) -> Embedding (a list of numbers with meaning)
  
  "Who" -> 0 -> [0.5, -0.2, 0.8, -0.1, 0.3, -0.6, 0.4, 0.9]
  
The model LEARNS these numbers during training to make similar words have similar vectors.

WHAT IS A VECTOR?
-----------------
A vector is just a list of numbers. Like: [1.5, -0.3, 0.8, 2.1]

In our case:
- Each word becomes a vector with D_MODEL numbers (we chose 8)
- So "Who" becomes 8 numbers, "created" becomes 8 numbers, etc.

WHY 8 NUMBERS? (D_MODEL = 8)
-----------------------------
This is the "dimensionality" of our embeddings:
- Real systems (like GPT-3) use 12,288 dimensions!
- We use 8 to keep it simple and easy to print/debug
- More dimensions = can capture more subtle relationships
- Fewer dimensions = faster but less expressive
"""

# ============================================================================
# IMPORTS - Bringing in Tools We Need
# ============================================================================
# `import torch` - PyTorch is a library for numerical computing and neural networks
#                  Think of it like a super-powered calculator for AI
import torch

# `from tiny_transformer.tokenizer import VOCAB` - Get the vocabulary list
#   We need this to know how many words we have (vocabulary size)
from tiny_transformer.tokenizer import VOCAB


# ============================================================================
# CONFIGURATION - Setting Up the Sizes
# ============================================================================

# D_MODEL: The number of dimensions (numbers) we use to represent each word
# Think of this as: "How many numbers do we use to describe each word?"
# - Bigger = more expressive but slower
# - Smaller = faster but less expressive
# We chose 8 for simplicity. Real models use hundreds or thousands!
D_MODEL = 8

# VOCAB_SIZE: How many words are in our vocabulary?
# len(VOCAB) counts the items in the VOCAB list
# If VOCAB = ["Who", "created", "JavaScript", ...] with 9 words, then VOCAB_SIZE = 9
VOCAB_SIZE = len(VOCAB)


# ============================================================================
# THE EMBEDDING MATRIX - Our Lookup Table of Word Vectors
# ============================================================================

# W is the "embedding matrix" - a big table of numbers
# 
# SHAPE: [VOCAB_SIZE, D_MODEL] which is [9, 8] in our case
# This means:
# - 9 rows (one for each word in vocabulary)
# - 8 columns (one for each dimension of the embedding)
#
# Visual representation:
#                    8 dimensions →
#              [0.1, -0.3, 0.5, 0.2, -0.1, 0.8, -0.4, 0.6]  ← Row 0 (embedding for "Who")
#              [0.4, 0.2, -0.7, 0.1, 0.5, -0.2, 0.9, -0.3]  ← Row 1 (embedding for "created")
# 9 words      [...]                                         ← Row 2 (embedding for "JavaScript")
#  ↓           [...]                                         ← Row 3 (embedding for "?")
#              [...]                                         ← Row 4 (embedding for "Brendan")
#              ...and so on...
#
# torch.randn() creates RANDOM numbers (from a "normal distribution")
# - Initially, these numbers are random (meaningless)
# - During TRAINING, the model learns better values
# - After training, similar words will have similar vectors
#
# requires_grad=True is CRITICAL for learning!
# - "grad" is short for "gradient" (calculus term)
# - This tells PyTorch: "Track changes to W so we can update it during training"
# - Without this, the embeddings would never improve!

W = torch.randn(VOCAB_SIZE, D_MODEL, requires_grad=True)


# ============================================================================
# THE EMBED FUNCTION - Looking Up Word Vectors
# ============================================================================

def embed(ids):
        """
        Look up the embedding vectors for a list of token IDs.
        
        WHAT THIS DOES (Simple Version):
        --------------------------------
        You give it a list of word IDs (numbers), and it gives you back
        the corresponding word vectors (lists of numbers).
        
        ANALOGY:
        --------
        Think of W as a phone book where:
        - Each row number is a person's ID
        - Each row contains that person's phone number
        
        If you want phone numbers for people [0, 1, 7]:
        - Look up row 0 -> get that phone number
        - Look up row 1 -> get that phone number  
        - Look up row 7 -> get that phone number
        
        Here we're looking up word vectors instead of phone numbers!
        
        PARAMETERS (Input):
        -------------------
        ids : list of integers OR a PyTorch tensor
            The token IDs you want embeddings for
            Example: [0, 1, 7, 3] means get embeddings for words 0, 1, 7, and 3
        
        RETURNS (Output):
        -----------------
        PyTorch tensor with shape [seq_len, D_MODEL]
        - seq_len = how many IDs you passed in (length of the list)
        - D_MODEL = 8 (the size of each embedding vector)
        
        STEP-BY-STEP EXAMPLE:
        ---------------------
        Input: ids = [0, 1, 7]  (asking for embeddings of words 0, 1, and 7)
        
        Step 1: Look up W[0] -> [0.1, -0.3, 0.5, 0.2, -0.1, 0.8, -0.4, 0.6]
        Step 2: Look up W[1] -> [0.4, 0.2, -0.7, 0.1, 0.5, -0.2, 0.9, -0.3]
        Step 3: Look up W[7] -> [0.2, 0.6, -0.4, 0.8, -0.5, 0.3, 0.1, -0.8]
        
        Output: A tensor with shape [3, 8] containing these three vectors stacked:
        [[0.1, -0.3, 0.5, 0.2, -0.1, 0.8, -0.4, 0.6],   ← embedding for ID 0
         [0.4, 0.2, -0.7, 0.1, 0.5, -0.2, 0.9, -0.3],   ← embedding for ID 1
         [0.2, 0.6, -0.4, 0.8, -0.5, 0.3, 0.1, -0.8]]   ← embedding for ID 7
        
        WHAT IS A TENSOR?
        -----------------
        A tensor is PyTorch's name for a multi-dimensional array of numbers.
        - 1D tensor = a list of numbers [1, 2, 3]
        - 2D tensor = a table of numbers (rows and columns)
        - 3D tensor = multiple tables stacked together
        
        Our output is a 2D tensor (a table).
        
        TECHNICAL NOTE:
        ---------------
        The notation W[ids] is called "indexing" or "fancy indexing"
        - If ids = [0, 1, 7], then W[ids] means "get rows 0, 1, and 7 from W"
        - PyTorch automatically handles this and returns them stacked together
        - This is MUCH faster than writing a loop!
        
        WHY THIS MATTERS:
        -----------------
        This converts abstract word IDs into rich numerical representations
        that the neural network can process and learn from!
        """
        # This one line does all the magic!
        # W[ids] means: "Give me the rows of W at positions specified by ids"
        return W[ids]