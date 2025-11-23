"""
================================================================================
positional.py - Teaching the Model About Word Order
================================================================================

THE PROBLEM WE'RE SOLVING:
--------------------------
Imagine you have these two sentences:
1. "Who created Python ?"
2. "Python created Who ?"

After converting to embeddings, the model sees:
1. [embedding_Who, embedding_created, embedding_Python, embedding_?]
2. [embedding_Python, embedding_created, embedding_Who, embedding_?]

BUT THERE'S A PROBLEM!
When we do mathematical operations (like attention), the model doesn't know
which word came FIRST, SECOND, THIRD, etc. It treats them like a bag of words!

Without position information, both sentences would look the same to the model,
even though they have completely different meanings!

REAL-WORLD ANALOGY:
-------------------
Imagine reading a book where all pages are blank except for the words:
"not", "do", "I", "understand"

You see these four words but don't know the order:
- "I do not understand" (makes sense!)
- "do I not understand" (a question)
- "understand I do not" (Yoda speaking!)

The ORDER matters! Same with language models.

THE SOLUTION: POSITIONAL ENCODING
----------------------------------
We add a special "position marker" to each word's embedding:
- Word at position 0 gets a "position 0" marker added
- Word at position 1 gets a "position 1" marker added
- Word at position 2 gets a "position 2" marker added
- And so on...

Think of it like page numbers in a book - they tell you the order!

MATHEMATICAL VIEW:
------------------
If we have embeddings:
  Position 0: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  ← "Who"
  Position 1: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  ← "created"

We add positional vectors:
  Position 0: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] + [0.01, 0.02, ...]
  Position 1: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [0.05, 0.03, ...]

Now each word's vector contains BOTH:
1. What the word means (from embedding)
2. Where it appears in the sentence (from positional encoding)

TWO COMMON APPROACHES:
----------------------
1. LEARNED (what we use here):
   - Start with random position vectors
   - Let the model learn good position markers during training
   - Simple and works well for small models

2. SINUSOIDAL (used in original Transformer paper):
   - Use mathematical sine/cosine functions to create position patterns
   - Doesn't need learning, works for any length
   - More common in large production models

We use learned positions because:
- It's easier to understand (just a lookup table!)
- Works great for our small teaching example
- Shows the pattern clearly
"""

# ============================================================================
# IMPORTS
# ============================================================================
import torch  # PyTorch library for tensor operations


# ============================================================================
# CONFIGURATION
# ============================================================================

# D_MODEL: Same as embeddings.py - each position vector has 8 numbers
# This MUST match the embedding size! Otherwise we can't add them together.
# (You can only add lists of the same length: [1,2,3] + [4,5,6] = [5,7,9])
D_MODEL = 8

# MAX_LEN: Maximum number of words/tokens we can handle in one sentence
# - We create position markers for positions 0 through 31 (32 total)
# - If your sentence has more than 32 words, you need to increase this
# - Most of our examples have 4-5 words, so 32 is plenty
# - Real models like GPT use much higher limits (2048, 4096, or even 100k+)
MAX_LEN = 32


# ============================================================================
# THE POSITIONAL ENCODING TABLE
# ============================================================================

# POS is a table that stores position markers for each position in a sequence
#
# SHAPE: [MAX_LEN, D_MODEL] which is [32, 8]
# This means:
# - 32 rows (one for each position: 0, 1, 2, ..., 31)
# - 8 columns (one for each dimension, matching D_MODEL)
#
# Visual representation:
#                    8 dimensions →
#     Position 0:  [0.2, -0.1, 0.4, 0.3, -0.5, 0.6, -0.2, 0.8]
#     Position 1:  [0.5, 0.3, -0.4, 0.7, 0.2, -0.6, 0.9, -0.1]
#     Position 2:  [0.1, -0.8, 0.3, -0.2, 0.6, 0.4, -0.7, 0.5]
#     Position 3:  [...]
#     ...
#     Position 31: [...]
#
# torch.randn() creates RANDOM numbers initially
# - These start random but the model LEARNS good values during training
# - After training, each position gets a distinctive "fingerprint"
#
# requires_grad=True means "let the model learn these during training"
# - Without this, the position markers would stay random forever!
# - With this, the model adjusts them to work better with the task

POS = torch.randn(MAX_LEN, D_MODEL, requires_grad=True)


# ============================================================================
# THE FUNCTION TO ADD POSITIONAL INFORMATION
# ============================================================================

def add_positional_encoding(x):
    """
    Add position information to word embeddings.
    
    WHAT THIS DOES:
    ---------------
    Takes word embeddings (which contain meaning but not position) and adds
    positional encodings (markers that indicate where each word appears).
    
    INPUT:
    ------
    x : PyTorch tensor with shape [seq_len, D_MODEL]
        - seq_len = number of words in your sentence
        - D_MODEL = 8 (size of each word vector)
        
        Example: If you have "Who created Python ?" (4 words), 
                 x has shape [4, 8]
    
    OUTPUT:
    -------
    PyTorch tensor with shape [seq_len, D_MODEL] (same shape as input)
        - Each word now has position information added to it
    
    STEP-BY-STEP EXAMPLE:
    ---------------------
    Imagine x is embeddings for "Who created Python ?" with shape [4, 8]:
    
    x = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],   ← "Who" embedding
         [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],   ← "created" embedding
         [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],   ← "Python" embedding
         [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]   ← "?" embedding
    
    Step 1: Get seq_len (how many words?)
            seq_len = x.shape[0] = 4
    
    Step 2: Get the first 4 position vectors from POS
            POS[:4] means "get rows 0, 1, 2, 3 from POS"
            
            This gives us:
            [[0.2, -0.1, 0.4, ...],   ← position 0 marker
             [0.5, 0.3, -0.4, ...],   ← position 1 marker
             [0.1, -0.8, 0.3, ...],   ← position 2 marker
             [-0.3, 0.4, 0.2, ...]]   ← position 3 marker
    
    Step 3: Add them together element-by-element
            x + POS[:seq_len]
            
            Result:
            [["Who" embedding + position 0 marker],
             ["created" embedding + position 1 marker],
             ["Python" embedding + position 2 marker],
             ["?" embedding + position 3 marker]]
    
    Now each word knows both WHAT it means and WHERE it is!
    
    WHAT IS x.shape[0]?
    -------------------
    In PyTorch (and NumPy):
    - .shape gives you the dimensions of a tensor
    - For a 2D tensor (table), shape is [rows, columns]
    - .shape[0] gives you the number of rows (first dimension)
    - .shape[1] gives you the number of columns (second dimension)
    
    Example:
    If x has shape [4, 8]:
    - x.shape[0] = 4 (number of rows/words)
    - x.shape[1] = 8 (number of columns/dimensions)
    
    WHAT IS POS[:seq_len]?
    -----------------------
    This is called "slicing" - taking a portion of a list/tensor
    
    POS[:4] means "give me rows 0, 1, 2, 3 from POS"
    POS[:seq_len] means "give me the first seq_len rows"
    
    Think of it like:
    - POS is a book with 32 pages
    - POS[:4] means "give me the first 4 pages"
    
    WHY ADD INSTEAD OF CONCATENATE?
    --------------------------------
    We could have stuck the position markers on the end:
    [word_vector | position_vector] (length 16)
    
    But we ADD them instead:
    word_vector + position_vector (length 8)
    
    Advantages of adding:
    - Keeps the vector size constant (still 8 numbers)
    - More parameter-efficient
    - Empirically works better in practice
    - The model learns to disentangle meaning and position
    """
    # Step 1: Figure out how many words we have
    seq_len = x.shape[0]
    
    # Step 2: Add position markers to word embeddings and return result
    return x + POS[:seq_len]