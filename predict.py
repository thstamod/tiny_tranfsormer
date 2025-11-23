"""
================================================================================
predict.py - Quick Predictions: Interactive Testing Script
================================================================================

WHAT IS THIS FILE FOR?
-----------------------
This is a STANDALONE prediction script for quick experiments and testing.
It's separate from the main training pipeline (train_tiny.py).

IMPORTANT DISTINCTION:
----------------------
- train_tiny.py: Uses output_head.py's Wout/bout (the TRAINED weights)
- predict.py: Has its OWN Wout/bout (independent, for quick testing)

Think of this as a "scratch pad" for trying things out without affecting
the main trained model!

WHY SEPARATE?
-------------
In a learning/experimental environment, you might want to:
1. Quickly test model components without full training
2. Experiment with the forward pass independently
3. Debug individual parts of the pipeline

This file lets you do that without interfering with the trained weights!

USAGE:
------
You can import and use predict_next() function:
    from tiny_transformer.predict import predict_next
    probs = predict_next("Who created Python ?")
    print(probs)  # See probabilities for each word

NOTE: Since this uses untrained weights, predictions will be random/meaningless!
For actual trained predictions, use train_tiny.py's test() function.
"""

# ============================================================================
# IMPORTS
# ============================================================================
from tiny_transformer.tokenizer import tokenizer, VOCAB, id2tok
from tiny_transformer.embeddings import embed
from tiny_transformer.positional import add_positional_encoding
from tiny_transformer.transformer_block import transformer_block
import torch  # PyTorch library


# ============================================================================
# CONFIGURATION
# ============================================================================
# These must match the values in other modules for compatibility

D_MODEL = 8  # Size of word vectors
VOCAB_SIZE = len(VOCAB)  # Number of words in vocabulary (9)


# ============================================================================
# INDEPENDENT OUTPUT PROJECTION WEIGHTS
# ============================================================================
# IMPORTANT: These are SEPARATE from output_head.py's weights!
# These are random and NOT trained. They're just for quick local experiments.
#
# If you want to use trained weights, use the test() function in train_tiny.py!

# Output weight matrix: transforms 8-number vectors to 9-number vectors
# Shape: [D_MODEL, VOCAB_SIZE] = [8, 9]
Wout = torch.randn(D_MODEL, VOCAB_SIZE, requires_grad=True)

# Output bias vector: one bias for each vocabulary word
# Shape: [VOCAB_SIZE] = [9]
bout = torch.randn(VOCAB_SIZE, requires_grad=True)

# These start RANDOM and stay random (unless you train them separately)!


def predict_next(text: str):
    """
    Run a complete forward pass and return next-word probabilities.
    
    This is the COMPLETE MODEL PIPELINE in one function!
    Takes text input, processes it through all components, returns probabilities.
    
    ⚠️  WARNING: This uses UNTRAINED weights (random Wout/bout)!
    Predictions will be meaningless until weights are trained.
    For trained predictions, use train_tiny.py's test() function!
    
    PARAMETERS:
    -----------
    text : str
        Input text to process
        Example: "Who created Python ?"
    
    RETURNS:
    --------
    probs : PyTorch tensor with shape [VOCAB_SIZE]
        Probabilities for each word in the vocabulary
        Values are between 0 and 1, and sum to exactly 1.0
        Example: [0.10, 0.15, 0.05, 0.23, 0.12, 0.08, 0.11, 0.09, 0.07]
        
        Higher probability = model thinks that word is more likely next
    
    THE 7 STEPS (Complete Model Pipeline):
    ---------------------------------------
    
    STEP 1: Tokenization
    --------------------
    Convert text string to a list of integer IDs.
    
    Example:
    Input: "Who created Python ?"
    Output: [0, 1, 7, 3]
    
    STEP 2: Embeddings
    ------------------
    Look up embedding vectors for each token ID.
    
    [0, 1, 7, 3] → [[vec_0], [vec_1], [vec_7], [vec_3]]
    Shape: [4, 8] (4 words, each represented by 8 numbers)
    
    STEP 3: Positional Encoding
    ----------------------------
    Add position information so model knows word order.
    
    Each word vector gets its position marker added to it.
    Shape stays [4, 8] but now includes position info.
    
    STEP 4: Transformer Block
    --------------------------
    Process through attention and feedforward layers.
    - Words communicate via attention
    - Words process via feedforward
    - Residual connections preserve information
    
    Output shape: [4, 8] (enriched representations!)
    
    STEP 5: Extract Last Token
    ---------------------------
    Take the last word's vector (it has seen all previous context).
    
    h_last = h[-1] means "get the last row"
    Shape: [8] (just 8 numbers now, not a table)
    
    STEP 6: Project to Vocabulary
    ------------------------------
    Transform the 8-number vector to 9 scores (one per vocab word).
    
    h_last @ Wout + bout
    [8] @ [8,9] + [9] = [9]
    
    These are LOGITS (raw scores), can be any value (positive/negative/large/small).
    
    STEP 7: Softmax (Convert to Probabilities)
    -------------------------------------------
    Convert logits to probabilities that sum to 1.0
    
    Example transformation:
    Logits: [2.1, 0.5, -0.3, 3.2, 1.1, 0.8, -0.5, 1.5, 0.9]
    Probs:  [0.15, 0.03, 0.01, 0.42, 0.05, 0.04, 0.01, 0.08, 0.04]
    
    Now we can interpret: "The model thinks word 3 is most likely (42%)"
    
    WHAT IS SOFTMAX MATHEMATICALLY?
    --------------------------------
    For each element i:
        probability_i = exp(logit_i) / sum(exp(all_logits))
    
    Properties:
    - All probabilities are between 0 and 1
    - All probabilities sum to exactly 1.0
    - Larger logits get larger probabilities
    - The relative differences are preserved
    
    LOGITS VS PROBABILITIES:
    ------------------------
    LOGITS (raw scores):
    - Can be any number: positive, negative, large, small
    - Don't sum to anything particular
    - Example: [2.1, -0.3, 5.2]
    
    PROBABILITIES (after softmax):
    - Between 0 and 1
    - Sum to exactly 1.0 (100%)
    - Example: [0.12, 0.01, 0.87]
    
    For training: use logits (more numerically stable)
    For interpretation: use probabilities (easier to understand)
    
    EXAMPLE USAGE:
    --------------
    >>> probs = predict_next("Who created Python ?")
    >>> print(probs)
    tensor([0.10, 0.15, 0.05, 0.23, 0.12, 0.08, 0.11, 0.09, 0.07])
    
    >>> # Find the most likely word
    >>> most_likely_idx = probs.argmax()
    >>> print(f"Most likely next word: {VOCAB[most_likely_idx]}")
    Most likely next word: ?
    
    >>> # Show all words with their probabilities
    >>> for idx, prob in enumerate(probs):
    ...     print(f"{VOCAB[idx]}: {prob:.2%}")
    Who: 10.00%
    created: 15.00%
    JavaScript: 5.00%
    ?: 23.00%
    ...
    """

    # ========================================================================
    # STEP 1: Tokenize Text to IDs
    # ========================================================================
    ids = tokenizer(text)
    # Example: "Who created Python ?" → [0, 1, 7, 3]

    # ========================================================================
    # STEP 2: Lookup Embeddings
    # ========================================================================
    x = embed(ids)
    # Shape: [seq_len, D_MODEL]
    # Example: [4, 8] for a 4-word input
    
    # ========================================================================
    # STEP 3: Add Positional Encodings
    # ========================================================================
    x = add_positional_encoding(x)
    # Shape stays [seq_len, D_MODEL] but vectors now include position info

    # ========================================================================
    # STEP 4: Process Through Transformer Block
    # ========================================================================
    h = transformer_block(x)
    # Shape: [seq_len, D_MODEL]
    # Each word now has contextual information from other words

    # ========================================================================
    # STEP 5: Extract Last Token's Vector
    # ========================================================================
    # In autoregressive (left-to-right) language modeling,
    # the last token has seen all previous context
    h_last = h[-1]
    # Shape: [D_MODEL] = [8]

    # ========================================================================
    # STEP 6: Project to Vocabulary Logits
    # ========================================================================
    # Matrix multiply by output weights and add bias
    logits = h_last @ Wout + bout
    # Shape: [VOCAB_SIZE] = [9]
    # These are raw scores, not yet probabilities

    # ========================================================================
    # STEP 7: Convert Logits to Probabilities (Softmax)
    # ========================================================================
    # dim=-1 means apply softmax along the last dimension
    # Converts any numbers into probabilities between 0 and 1 that sum to 1.0
    probs = torch.softmax(logits, dim=-1)
    # Shape: [VOCAB_SIZE] = [9]
    # Now each value is a probability!

    # Return the probability distribution over the vocabulary
    return probs