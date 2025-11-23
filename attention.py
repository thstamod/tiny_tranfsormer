"""
================================================================================
attention.py - The Heart of the Transformer: Making Words Talk to Each Other
================================================================================

WHAT IS ATTENTION? (Simple Explanation)
----------------------------------------
Attention is the mechanism that lets words in a sentence "communicate" with
each other and understand context.

REAL-WORLD ANALOGY #1: Reading Comprehension
---------------------------------------------
When you read "The animal didn't cross the street because IT was too tired",
your brain knows "IT" refers to "animal", not "street".

How? You pay ATTENTION to different words when understanding "IT":
- You look back at "animal" (high attention)
- You barely consider "street" (low attention)
- You ignore "didn't" completely (zero attention)

This is exactly what the attention mechanism does!

REAL-WORLD ANALOGY #2: A Team Meeting
--------------------------------------
Imagine 4 people in a meeting: Alice, Bob, Carol, Dan

When Alice speaks, she might:
- Listen carefully to Bob (60% attention)
- Listen a bit to Carol (30% attention)
- Barely listen to Dan (10% attention)
- Also listen to herself (consider her own previous points)

Each person decides how much to "pay attention" to each other person!

In our model, each WORD is like a person deciding how much to listen to
every other word (including itself).

THE ATTENTION MECHANISM EXPLAINED:
-----------------------------------
For a sentence "Who created Python ?"

Each word computes:
1. "How much should I pay attention to every word (including myself)?"
2. Creates attention weights (numbers that sum to 1.0, like percentages)
3. Takes a weighted average of all words based on these weights

Example attention weights for "created":
- "Who": 0.15 (15%)
- "created": 0.30 (30% - it listens to itself!)
- "Python": 0.50 (50% - pays most attention here!)
- "?": 0.05 (5%)

THREE KEY CONCEPTS: QUERY, KEY, VALUE
--------------------------------------
Attention uses three "projections" of each word:

1. QUERY (Q): "What am I looking for?"
   - Like a search query you type into Google
   - "I'm the word 'created', what should I look for?"

2. KEY (K): "What do I have to offer?"
   - Like a book's index entries
   - "I'm the word 'Python', here's what I represent"

3. VALUE (V): "What information do I actually contain?"
   - Like the actual content of a book page
   - "Here's the meaningful information I carry"

ANALOGY: Library Search
------------------------
Imagine you're in a library:
- Your QUERY: "I want information about programming languages"
- Each book's KEY: "I contain: Java", "I contain: Python", "I contain: cooking"
- Each book's VALUE: The actual content inside

You compare your QUERY to each book's KEY to decide which books to read.
Then you read the VALUEs from the most relevant books!

ATTENTION MECHANISM STEP-BY-STEP:
----------------------------------
For each word in "Who created Python ?":

Step 1: Create Q, K, V vectors for all words
Step 2: Compare queries to keys → get attention scores
Step 3: Convert scores to probabilities (softmax) → attention weights
Step 4: Take weighted average of values → output

MATHEMATICAL INTUITION (Don't Worry If This Seems Complex):
------------------------------------------------------------
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V

Breaking it down:
- Q @ K^T: Compare every query to every key (how similar are they?)
- / √d: Scale down the numbers (prevents them from getting too big)
- softmax: Convert to probabilities that sum to 1.0
- @ V: Use those probabilities to mix the values together

WHY IS THIS POWERFUL?
----------------------
1. CONTEXT AWARENESS: Words understand their neighbors
2. FLEXIBLE: Different words can pay attention to different things
3. PARALLEL: All words can compute attention simultaneously
4. LEARNED: The model learns WHAT to pay attention to during training

This is why transformers are so powerful compared to older models!
"""

# ============================================================================
# IMPORTS
# ============================================================================
import torch  # PyTorch for tensor operations
import math   # We need math.sqrt for scaling

# ============================================================================
# CONFIGURATION
# ============================================================================

# D_MODEL: The size of each word vector (must match embeddings.py and positional.py)
# All our vectors have 8 numbers
D_MODEL = 8


# ============================================================================
# PROJECTION MATRICES: Converting Words to Queries, Keys, and Values
# ============================================================================

# These are "weight matrices" - tables of numbers that transform word vectors.
# Each is shape [D_MODEL, D_MODEL] which is [8, 8]
#
# Think of them like "recipe books" that say:
# "Take a word vector and transform it into a query/key/value vector"
#
# VISUAL:
#   word_vector (8 numbers) @ Wq (8×8 matrix) = query_vector (8 numbers)
#
# Why three separate matrices?
# - Wq creates "queries" (what am I looking for?)
# - Wk creates "keys" (what do I represent?)
# - Wv creates "values" (what information do I contain?)
#
# These start RANDOM but during training, the model learns good transformations!

# Query projection matrix: transforms words into "what I'm looking for" vectors
Wq = torch.randn(D_MODEL, D_MODEL, requires_grad=True)

# Key projection matrix: transforms words into "what I represent" vectors
Wk = torch.randn(D_MODEL, D_MODEL, requires_grad=True)

# Value projection matrix: transforms words into "information I carry" vectors
Wv = torch.randn(D_MODEL, D_MODEL, requires_grad=True)


# ============================================================================
# THE SELF-ATTENTION FUNCTION
# ============================================================================

def self_attention(x):
    """
    Compute self-attention for a sequence of words.
    
    WHAT THIS DOES (High-Level):
    ----------------------------
    Takes a sequence of word vectors and returns new word vectors where each
    word has "looked at" and incorporated information from other words.
    
    INPUT:
    ------
    x : PyTorch tensor with shape [seq_len, D_MODEL]
        - seq_len = number of words (e.g., 4 for "Who created Python ?")
        - D_MODEL = 8 (size of each word vector)
        
        This is the output from add_positional_encoding(), so each word
        already has both meaning (embedding) and position information.
    
    OUTPUT:
    -------
    out : PyTorch tensor with shape [seq_len, D_MODEL]
        - Same shape as input
        - But now each word has been updated with context from other words
    
    THE 5 STEPS OF ATTENTION:
    --------------------------
    
    STEP 1: Create Q, K, V (Queries, Keys, Values)
    -----------------------------------------------
    Transform the input vectors into three different representations.
    
    Mathematical operation: matrix multiplication (@)
    - Q = x @ Wq   (queries: what each word is looking for)
    - K = x @ Wk   (keys: what each word represents)
    - V = x @ Wv   (values: information each word contains)
    
    Think of @ as "applying a transformation recipe"
    
    Example with 4 words:
    x shape: [4, 8]
    Wq shape: [8, 8]
    Q = x @ Wq → shape [4, 8]
    (Each of the 4 words gets transformed into an 8-number query vector)
    
    STEP 2: Compute Attention Scores
    ---------------------------------
    Compare every query to every key to see how well they match.
    
    scores = Q @ K.T
    
    What is K.T? 
    - T means "transpose" (flip rows and columns)
    - If K is [4, 8], then K.T is [8, 4]
    
    What does Q @ K.T give us?
    - Q: [4, 8]  (4 queries, each with 8 numbers)
    - K.T: [8, 4]  (keys transposed)
    - Q @ K.T: [4, 4]  (a table of similarity scores!)
    
    Visual of the scores matrix [4, 4]:
    
              "Who"  "created"  "Python"  "?"   ← keys
    "Who"      5.2      3.1        2.8     1.5   ← query "Who"
    "created"  4.1      6.3        8.9     2.1   ← query "created"
    "Python"   3.5      7.2        9.4     1.8   ← query "Python"
    "?"        2.1      3.4        2.9     4.2   ← query "?"
    
    Each number represents "how much does query[i] match with key[j]?"
    Higher number = better match = should pay more attention!
    
    STEP 3: Scale the Scores
    -------------------------
    Divide by √D_MODEL (square root of 8 ≈ 2.83)
    
    scores = scores / math.sqrt(D_MODEL)
    
    Why? When vectors are large, dot products can become huge numbers.
    This makes the softmax in the next step too "sharp" (all weight on one item).
    Scaling keeps the numbers in a reasonable range.
    
    STEP 4: Convert Scores to Attention Weights (Softmax)
    ------------------------------------------------------
    Use softmax to convert raw scores into probabilities that sum to 1.0
    
    attn_weights = torch.softmax(scores, dim=-1)
    
    What is softmax?
    - It's a function that converts any numbers into probabilities (0 to 1)
    - All probabilities sum to exactly 1.0 (100%)
    - Larger input values get larger probabilities
    
    Example: scores = [2.0, 1.0, 3.0, 0.5]
    After softmax: [0.26, 0.10, 0.52, 0.06]  (these sum to 1.0!)
    
    dim=-1 means "apply softmax along the last dimension (across each row)"
    
    Our attention weights now look like:
    
              "Who"  "created"  "Python"  "?"   ← attending TO these words
    "Who"      0.35     0.25       0.30    0.10   ← weights for "Who" (sum=1.0)
    "created"  0.10     0.15       0.65    0.10   ← weights for "created" (sum=1.0)
    "Python"   0.05     0.30       0.60    0.05   ← weights for "Python" (sum=1.0)
    "?"        0.20     0.40       0.25    0.15   ← weights for "?" (sum=1.0)
    
    Each row is a probability distribution showing "how much attention
    should this word pay to each word?"
    
    STEP 5: Weighted Average of Values
    -----------------------------------
    Use the attention weights to create a weighted average of the value vectors.
    
    out = attn_weights @ V
    
    - attn_weights: [4, 4]  (attention weights table)
    - V: [4, 8]  (value vectors for each word)
    - out: [4, 8]  (new contextual vectors!)
    
    For each word, we're computing:
    new_vector = (weight1 × value1) + (weight2 × value2) + (weight3 × value3) + ...
    
    Example for "created":
    new_"created" = 0.10×value("Who") + 0.15×value("created") + 
                    0.65×value("Python") + 0.10×value("?")
    
    Since "created" paid most attention (0.65) to "Python", its new vector
    is heavily influenced by Python's information!
    
    FINAL RESULT:
    -------------
    We get new vectors for each word that contain:
    - The word's original meaning
    - Context from other words it paid attention to
    
    This is how "IT" in "the animal... IT was tired" can "know" it refers to "animal"!
    
    WHAT DOES @ MEAN IN PYTHON?
    ----------------------------
    The @ operator is matrix multiplication (introduced in Python 3.5)
    - It's used for multiplying matrices/tensors
    - For 2D matrices A[m,n] @ B[n,p] = C[m,p]
    - Each element C[i,j] = sum of A[i,:] × B[:,j]
    """

    # ========================================================================
    # STEP 1: Compute Queries, Keys, and Values
    # ========================================================================
    # Transform input word vectors into Q, K, V representations
    # Each has shape [seq_len, D_MODEL] - same as input
    Q = x @ Wq  # "What am I looking for?" vectors
    K = x @ Wk  # "What do I represent?" vectors
    V = x @ Wv  # "What information do I have?" vectors

    # ========================================================================
    # STEP 2: Compute Attention Scores
    # ========================================================================
    # Compare every query to every key (how similar are they?)
    # Result shape: [seq_len, seq_len] - a square matrix of similarity scores
    scores = Q @ K.T  # K.T is the transpose of K

    # ========================================================================
    # STEP 3: Scale the Scores
    # ========================================================================
    # Divide by √D_MODEL to prevent scores from being too large
    # This helps softmax work better (prevents vanishing gradients)
    scores = scores / math.sqrt(D_MODEL)

    # ========================================================================
    # STEP 4: Convert Scores to Attention Weights
    # ========================================================================
    # Apply softmax to make each row sum to 1.0 (convert to probabilities)
    # dim=-1 means apply softmax across columns (each row independently)
    attn_weights = torch.softmax(scores, dim=-1)  # shape [seq_len, seq_len]

    # ========================================================================
    # STEP 5: Compute Weighted Average of Values
    # ========================================================================
    # Multiply attention weights by values to get context-aware word vectors
    # Result shape: [seq_len, D_MODEL] - same as input
    out = attn_weights @ V
    
    # Return the new, context-aware word vectors!
    return out