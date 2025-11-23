"""
================================================================================
output_head.py - Making Predictions: Converting Vectors to Word Probabilities
================================================================================

WHAT IS THE OUTPUT HEAD?
-------------------------
After all the processing (embeddings, positions, attention, feedforward),
we have rich vector representations for each word. But we need to PREDICT
the next word! The output head does this final transformation.

THE PROBLEM WE'RE SOLVING:
--------------------------
Input: "Who created Python"
We have: Vector for "Python" (8 numbers like [0.5, -0.2, 0.8, ...])
We want: Probabilities for what comes next:
    - "?" → 70%
    - "Guido" → 20%
    - "JavaScript" → 5%
    - "Who" → 5%

How do we go from 8 numbers to probabilities for 9 vocabulary words?
Answer: A LINEAR PROJECTION (multiply by a matrix, add bias)!

ANALOGY: Multiple Choice Question
----------------------------------
Imagine you're taking a test. You've read a passage and now must answer:
"Who created Python?"

Your brain (the transformer) has processed all the information.
Now you need to choose from options A, B, C, D (our vocabulary).

The output head is like your brain assigning confidence scores to each option:
- Option A ("Brendan"): confidence = 2.1
- Option B ("Guido"): confidence = 8.5  ← highest!
- Option C ("Linus"): confidence = 1.3
- Option D ("Who"): confidence = 0.2

The highest score wins! (Or we can convert to probabilities with softmax)

THE OUTPUT HEAD ARCHITECTURE:
-----------------------------
It's a simple linear layer (also called "fully connected" or "dense" layer):

Input: [8 numbers]  (the last word's vector after transformer)
   ↓
Multiply by Wout (weight matrix)
   ↓
Add bout (bias vector)
   ↓
Output: [9 numbers]  (one score per vocabulary word)

WHY USE THE LAST TOKEN?
------------------------
In language modeling, we predict the NEXT word based on PREVIOUS words.

For "Who created Python", we want to predict what comes after "Python".
So we use "Python"'s final vector (which has seen all previous context
through attention) to make the prediction!

This is called "causal" or "autoregressive" language modeling:
- Each word can only look at words BEFORE it
- We predict the next word in sequence
- This is how GPT models work!

LOGITS VS PROBABILITIES:
------------------------
LOGITS: Raw scores (can be any number: positive, negative, large, small)
    Example: [2.1, 8.5, 1.3, 0.2, -1.5, 3.2, 0.8, 4.1, 1.9]

PROBABILITIES: Values between 0 and 1 that sum to 1.0
    Example: [0.05, 0.52, 0.03, 0.01, 0.00, 0.12, 0.02, 0.18, 0.07]
    (Sum = 1.0 or 100%)

We output LOGITS because:
1. More numerically stable for training
2. The loss function (cross-entropy) expects logits
3. We can convert to probabilities later with softmax if needed

WHAT IS SOFTMAX?
----------------
Softmax converts any numbers into probabilities:
    probability_i = exp(logit_i) / sum(exp(all_logits))

Example:
Logits: [1.0, 2.0, 0.5]
Step 1 - exp: [2.72, 7.39, 1.65]
Step 2 - sum: 2.72 + 7.39 + 1.65 = 11.76
Step 3 - divide: [2.72/11.76, 7.39/11.76, 1.65/11.76]
Result: [0.23, 0.63, 0.14]  (these sum to 1.0!)

Higher logits → higher probabilities
"""

# ============================================================================
# IMPORTS
# ============================================================================
import torch  # PyTorch for tensor operations
from tiny_transformer.tokenizer import VOCAB  # Need to know vocabulary size


# ============================================================================
# CONFIGURATION
# ============================================================================

# D_MODEL: Input dimension (size of word vectors from transformer)
# This is the size of the last token's vector we'll use for prediction
D_MODEL = 8

# VOCAB_SIZE: Output dimension (number of words in our vocabulary)
# We need one score for each possible word we could predict
# len(VOCAB) counts how many words are in the vocabulary (9 in our case)
VOCAB_SIZE = len(VOCAB)


# ============================================================================
# OUTPUT PROJECTION PARAMETERS
# ============================================================================

# Wout: Output weight matrix [D_MODEL, VOCAB_SIZE] = [8, 9]
# This transforms an 8-number vector into a 9-number vector
# Think of it as: "How do we map from representation space to vocabulary space?"
#
# Visual:
#   [8 numbers] @ [8×9 matrix] = [9 numbers]
#   last token     Wout           one per vocab word
#
# Each column of Wout corresponds to one vocabulary word
# The more a token vector aligns with a column, the higher that word's score!
Wout = torch.randn(D_MODEL, VOCAB_SIZE, requires_grad=True)

# bout: Output bias vector [VOCAB_SIZE] = [9]
# One bias value for each vocabulary word
# This lets each word have a "baseline" score before considering the input
# (Some words might be more common overall and get higher biases)
bout = torch.randn(VOCAB_SIZE, requires_grad=True)

# Both parameters start RANDOM and are LEARNED during training!
# requires_grad=True enables learning through backpropagation


# ============================================================================
# THE FORWARD PASS FUNCTION
# ============================================================================

def logits_for_ids(ids, embed, add_positional_encoding, transformer_block):
        """
        Run a complete forward pass through the model and return prediction scores.
        
        This function orchestrates the entire model pipeline:
        1. Convert word IDs to embeddings
        2. Add positional information
        3. Process through transformer block
        4. Project to vocabulary scores
        
        WHAT THIS DOES (High-Level):
        ----------------------------
        You give it a sequence like "Who created Python" (as token IDs),
        and it returns scores for what word should come next.
        
        INPUT:
        ------
        ids : list of integers
            Token IDs representing the input sequence
            Example: [0, 1, 7] for "Who created Python"
        
        embed : function
            The embedding function from embeddings.py
            Converts IDs to vectors
        
        add_positional_encoding : function
            The positional encoding function from positional.py
            Adds position information to vectors
        
        transformer_block : function
            The transformer block function from transformer_block.py
            Processes vectors through attention and feedforward
        
        OUTPUT:
        -------
        logits : PyTorch tensor with shape [VOCAB_SIZE]
            Raw scores (logits) for each word in the vocabulary
            Example: [1.2, 3.5, 0.8, 5.2, 2.1, 1.5, 0.9, 4.1, 2.3]
            
            Higher score = model thinks this word is more likely to come next
        
        THE FOUR STEPS:
        ---------------
        
        STEP 1: Embeddings
        ------------------
        x = embed(ids)
        
        Converts token IDs to embedding vectors:
        - ids: [0, 1, 7]  ("Who", "created", "Python")
        - x: [[vec0], [vec1], [vec7]]  shape [3, 8]
        
        Each word becomes an 8-number vector that captures its meaning.
        
        STEP 2: Positional Encoding
        ----------------------------
        x = add_positional_encoding(x)
        
        Adds position information so model knows word order:
        - x[0]: "Who" at position 0
        - x[1]: "created" at position 1
        - x[2]: "Python" at position 2
        
        Shape stays [3, 8] but each vector now includes position info.
        
        STEP 3: Transformer Block
        --------------------------
        h = transformer_block(x)
        
        Processes the sequence through attention and feedforward:
        - Words communicate via attention
        - Words process information via feedforward
        - Residual connections preserve information
        
        Output h has shape [3, 8] - same shape but enriched representations!
        
        STEP 4: Output Projection
        --------------------------
        h_last = h[-1]
        
        Get the LAST token's vector (position 2, "Python")
        - h_last shape: [8]
        
        Why the last token?
        In autoregressive (left-to-right) language modeling:
        - "Who" can see nothing before it
        - "created" can see "Who"
        - "Python" can see "Who created Python"
        
        The last token has seen ALL previous context, so it's best for prediction!
        
        logits = h_last @ Wout + bout
        
        Project the 8-number vector to 9 scores (one per vocabulary word):
        - h_last: [8]
        - Wout: [8, 9]
        - h_last @ Wout: [9]  (matrix multiplication)
        - bout: [9]
        - h_last @ Wout + bout: [9]  (add bias)
        
        Result: [score_for_Who, score_for_created, score_for_JavaScript, ...]
        
        FULL EXAMPLE WITH INTERPRETATION:
        ---------------------------------
        Input: "Who created Python" → ids = [0, 1, 7]
        
        After all processing, logits might be:
        [
          1.2,  ← score for "Who" (index 0)
          0.8,  ← score for "created" (index 1)
          0.5,  ← score for "JavaScript" (index 2)
          8.5,  ← score for "?" (index 3) ← HIGHEST!
          2.1,  ← score for "Brendan" (index 4)
          3.2,  ← score for "Guido" (index 5)
          1.5,  ← score for "Linus" (index 6)
          1.9,  ← score for "Python" (index 7)
          0.9   ← score for "Linux" (index 8)
        ]
        
        The highest score is 8.5 for "?" (index 3), so the model predicts "?"
        should come next! This makes sense: "Who created Python ?" is complete!
        
        TRAINING VS INFERENCE:
        ----------------------
        During TRAINING:
        - We pass logits to cross_entropy loss function
        - Loss compares logits to the correct answer
        - Model learns to give higher scores to correct words
        
        During INFERENCE (making predictions):
        - We convert logits to probabilities with softmax
        - We pick the highest probability word (or sample from distribution)
        - That's our prediction!
        
        WHY NOT APPLY SOFTMAX HERE?
        ---------------------------
        For training, cross_entropy expects raw logits (more numerically stable).
        For predictions, the caller can apply softmax if needed.
        This flexibility makes the function more useful!
        
        Returns:
            logits: tensor of shape [VOCAB_SIZE] with raw prediction scores
        """

        # ====================================================================
        # STEP 1: Look up embeddings for token IDs
        # ====================================================================
        x = embed(ids)                           # [seq_len, D_MODEL]

        # ====================================================================
        # STEP 2: Add positional information (word order)
        # ====================================================================
        x = add_positional_encoding(x)           # [seq_len, D_MODEL]

        # ====================================================================
        # STEP 3: Process through transformer (attention + feedforward)
        # ====================================================================
        h = transformer_block(x)                 # [seq_len, D_MODEL]

        # ====================================================================
        # STEP 4a: Extract last token's representation
        # ====================================================================
        # h[-1] means "get the last row of h" (last token)
        # Python negative indexing: -1 is last, -2 is second-to-last, etc.
        h_last = h[-1]                           # [D_MODEL]

        # ====================================================================
        # STEP 4b: Project to vocabulary scores
        # ====================================================================
        # Matrix multiply last token vector by output weights, add bias
        logits = h_last @ Wout + bout            # [VOCAB_SIZE]
        
        # Return raw scores for each vocabulary word!
        return logits
