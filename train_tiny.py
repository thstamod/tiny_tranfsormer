"""
================================================================================
train_tiny.py - Teaching the Model: The Training Loop
================================================================================

WHAT IS TRAINING?
-----------------
Training is the process where the model LEARNS from examples. Initially, all
the weights (W, Wq, Wk, etc.) are random numbers. Training adjusts these
numbers so the model makes better predictions!

ANALOGY: Learning to Play Basketball
-------------------------------------
Imagine teaching someone basketball:
1. They try to shoot (make a prediction)
2. You tell them if they made the basket or missed (compute error/loss)
3. They adjust their technique (update weights)
4. Repeat thousands of times → they get better!

This is exactly what happens in neural network training!

SUPERVISED LEARNING:
--------------------
We're using "supervised learning" where we provide:
- INPUT: "Who created JavaScript ?"
- CORRECT OUTPUT: "Brendan"

The model tries to predict the output, we tell it if it's right or wrong,
and it adjusts its parameters to do better next time.

THE TRAINING DATASET:
---------------------
We have 3 simple examples teaching the model about programming language creators:
1. "Who created JavaScript ?" → "Brendan" (Brendan Eich)
2. "Who created Python ?" → "Guido" (Guido van Rossum)
3. "Who created Linux ?" → "Linus" (Linus Torvalds)

This is TINY! Real models train on billions of examples.
But our tiny dataset is perfect for understanding the process!

THE TRAINING LOOP (5 Key Steps):
---------------------------------
1. FORWARD PASS: Run the model to get predictions
2. COMPUTE LOSS: Measure how wrong the predictions are
3. BACKWARD PASS: Calculate gradients (how to improve)
4. UPDATE WEIGHTS: Adjust parameters to reduce error
5. REPEAT: Do this many times until the model is accurate

WHAT IS A GRADIENT?
-------------------
A gradient tells you "which direction to adjust a weight to reduce error".

ANALOGY: Hiking Down a Mountain
If you're blindfolded on a mountain and want to go down:
- Feel which direction is steeper (compute gradient)
- Take a step in that direction (update weight)
- Repeat until you reach the bottom (minimize loss)

WHAT IS AN OPTIMIZER?
---------------------
An optimizer is the algorithm that updates weights based on gradients.
We use "Adam" - a popular, sophisticated optimizer.

Think of it as a smart hiking guide that:
- Remembers previous steps
- Adjusts step size automatically
- Takes momentum into account

THE LEARNING RATE (lr=0.01):
-----------------------------
This controls how BIG each update step is:
- Too large: might overshoot and miss the minimum (unstable)
- Too small: learns very slowly
- 0.01 is a reasonable middle ground for our tiny model

WHAT IS AN EPOCH?
-----------------
One epoch = one complete pass through ALL training examples.
We train for 500 epochs, meaning we show the model all 3 examples 500 times!

Each time, the model gets slightly better at predicting the right answers.
"""

# ============================================================================
# IMPORTS - Bringing in Tools and Components
# ============================================================================

import torch  # PyTorch - our deep learning library
import torch.nn.functional as F  # Functional interface (has cross_entropy loss)

# ============================================================================
# REPRODUCIBILITY (Currently Commented Out)
# ============================================================================
# torch.manual_seed(0) would make the random initialization identical every run
# This is useful for debugging (same results every time)
# Currently commented out, so each run starts with different random weights
#torch.manual_seed(0)

from tiny_transformer.tokenizer import tokenizer, tok2id, VOCAB
from tiny_transformer.embeddings import embed
from tiny_transformer.positional import add_positional_encoding
from tiny_transformer.transformer_block import transformer_block
from tiny_transformer.output_head import logits_for_ids  # we’ll define this in a second


# Small example dataset: a few prompts -> target next token
dataset = [
    ("Who created JavaScript ?", "Brendan"),
    ("Who created Python ?", "Guido"),
    ("Who created Linux ?", "Linus"),
]

# We import the global weight tensors from the modules so the optimizer can
# update them in-place. In a real project you'd likely wrap the model in an
# `nn.Module` and register parameters cleanly, but for this tiny example we
# keep the tensors at module scope.
from tiny_transformer import embeddings, positional, attention, feedforward, output_head

params = [
    embeddings.W,
    positional.POS,
    attention.Wq, attention.Wk, attention.Wv,
    feedforward.W1, feedforward.b1, feedforward.W2, feedforward.b2,
    output_head.Wout, output_head.bout,
]

optimizer = torch.optim.Adam(params, lr=0.01)


def train(num_epochs=500):
    """
    The main training loop that teaches the model from our dataset.
    
    WHAT THIS FUNCTION DOES:
    ------------------------
    Repeatedly shows the model examples, measures mistakes, and adjusts
    the weights to reduce those mistakes. After 500 iterations (epochs),
    the model should have learned the pattern!
    
    PARAMETERS:
    -----------
    num_epochs : int, default=500
        How many times to go through ALL training examples.
        One epoch = seeing all 3 examples once.
        500 epochs = seeing each example 500 times!
        
        Why so many? Neural networks learn gradually through repetition.
    
    THE 5-STEP TRAINING PROCESS (Repeated for Each Example):
    ---------------------------------------------------------
    
    STEP 1: FORWARD PASS - Make a Prediction
    -----------------------------------------
    Run the input through the entire model to get predictions (logits).
    
    Like a student answering a question before seeing the answer.
    
    STEP 2: COMPUTE LOSS - Measure How Wrong We Are
    ------------------------------------------------
    Compare the model's prediction to the correct answer.
    Calculate a number (loss) that represents "how bad" the prediction is.
    
    - Low loss: prediction is close to correct (good!)
    - High loss: prediction is far from correct (bad!)
    
    We use "cross-entropy loss" - a standard for classification tasks.
    
    WHAT IS CROSS-ENTROPY?
    Cross-entropy measures the distance between two probability distributions:
    - Model's predicted probabilities: [0.1, 0.2, 0.6, 0.1] 
    - Correct answer (one-hot): [0, 0, 1, 0]  (third word is correct)
    
    The more the model's probabilities match the correct answer, the lower the loss!
    
    STEP 3: BACKWARD PASS - Calculate How to Improve
    -------------------------------------------------
    Use calculus (automatic differentiation) to compute gradients.
    Gradients tell us: "If we adjust each weight by a small amount, how much
    will the loss change?"
    
    This is the MAGIC of deep learning! PyTorch does this automatically.
    
    STEP 4: UPDATE WEIGHTS - Actually Improve
    ------------------------------------------
    Use the gradients to adjust all the weights slightly.
    The optimizer (Adam) handles the math of how much to adjust each weight.
    
    Weights that contributed to errors get adjusted more.
    Weights that helped get adjusted to help even more!
    
    STEP 5: REPEAT
    --------------
    Do this for every example in the dataset, for many epochs.
    Over time, the loss decreases and accuracy increases!
    
    IMPORTANT TECHNICAL DETAILS:
    ----------------------------
    
    .unsqueeze(0):
    --------------
    Adds a "batch dimension" to the tensor.
    - Before: [9] (just 9 numbers)
    - After: [1, 9] (1 batch with 9 numbers)
    
    Why? PyTorch functions like cross_entropy expect batches of examples.
    Even though we have 1 example, we need to format it as a "batch of 1".
    
    torch.long (dtype):
    -------------------
    The target must be an integer type (torch.long = 64-bit integer).
    cross_entropy expects target to be class indices (0, 1, 2, ...), not floats.
    
    optimizer.zero_grad():
    ----------------------
    CRITICAL! Clears old gradients.
    PyTorch ACCUMULATES gradients by default (adds new gradients to old ones).
    If we don't zero them, we'd be using gradients from previous examples!
    
    Think of it like erasing a whiteboard before writing new calculations.
    
    loss.backward():
    ----------------
    The MAGIC happens here! This one line:
    1. Computes gradients for ALL parameters (11 tensors!)
    2. Uses the chain rule from calculus automatically
    3. Stores gradients in each tensor's .grad attribute
    
    This is "backpropagation" - the key algorithm for training neural networks!
    
    optimizer.step():
    -----------------
    Uses the computed gradients to UPDATE all the weights.
    Adam algorithm decides exactly how much to adjust each weight based on:
    - The gradient (direction of improvement)
    - Learning rate (step size)
    - Momentum (memory of previous updates)
    
    After this, the model is slightly better than before!
    
    loss.item():
    ------------
    Converts a PyTorch tensor (with one number) to a regular Python number.
    We use this to print the loss value.
    
    THE OUTER LOOP (Epochs):
    -------------------------
    We loop 500 times through the entire dataset.
    Each epoch, the model sees all 3 examples and updates weights 3 times.
    Total updates: 500 epochs × 3 examples = 1,500 weight updates!
    
    Gradually, the model learns:
    "Who created JavaScript ?" → "Brendan"
    "Who created Python ?" → "Guido"
    "Who created Linux ?" → "Linus"
    """

    # ========================================================================
    # OUTER LOOP: Iterate Through Epochs
    # ========================================================================
    # An epoch is one complete pass through the entire dataset.
    # range(num_epochs) generates numbers: 0, 1, 2, ..., 499
    
    for epoch in range(num_epochs):
        # Track total loss for this epoch (across all 3 examples)
        total_loss = 0.0

        # ====================================================================
        # INNER LOOP: Iterate Through Each Training Example
        # ====================================================================
        # dataset has 3 items: [(prompt1, target1), (prompt2, target2), ...]
        # On each iteration, we get one (prompt, target_token) pair
        
        for prompt, target_token in dataset:
            # ================================================================
            # STEP 1: FORWARD PASS - Get Model's Prediction
            # ================================================================
            # Convert the prompt text into token IDs
            # Example: "Who created JavaScript ?" → [0, 1, 2, 3]
            ids = tokenizer(prompt)
            
            # Run the complete model forward pass:
            # ids → embeddings → +positions → transformer → output projection
            # Returns logits (raw scores) for each vocabulary word
            logits = logits_for_ids(
                ids,                      # Input token IDs
                embed,                    # Embedding function
                add_positional_encoding,  # Positional encoding function
                transformer_block,        # Transformer processing
            )  # Output shape: [VOCAB_SIZE] which is [9]
            
            # At this point, logits contains 9 numbers (one per vocab word)
            # Example: [1.2, 0.8, 0.5, 2.1, 5.3, 1.9, 0.9, 1.5, 1.1]
            # Higher numbers = model thinks that word is more likely
            
            # ================================================================
            # STEP 2: PREPARE THE TARGET (Correct Answer)
            # ================================================================
            # Look up the ID of the target word
            # Example: if target_token = "Brendan", then target_id = 4
            target_id = tok2id[target_token]
            
            # Create a tensor containing the target ID
            # torch.tensor([...]) creates a PyTorch tensor
            # dtype=torch.long means 64-bit integer (required for cross_entropy)
            # Shape: [1] (batch size of 1)
            target = torch.tensor([target_id], dtype=torch.long)
            
            # ================================================================
            # STEP 3: COMPUTE LOSS - How Wrong Is The Prediction?
            # ================================================================
            # Add a batch dimension to logits
            # unsqueeze(0) adds a dimension at position 0
            # Before: [9]  After: [1, 9]
            # This makes it a "batch of 1 example"
            logits_batch = logits.unsqueeze(0).float()  # shape [1, VOCAB_SIZE]
            
            # Compute cross-entropy loss
            # F.cross_entropy compares predictions to the correct answer
            # It internally applies log-softmax to logits (for numerical stability)
            # Then compares to target using negative log-likelihood
            #
            # Lower loss = better prediction
            # If model perfectly predicts "Brendan", loss ≈ 0
            # If model completely wrong, loss is large (can be > 10)
            loss = F.cross_entropy(logits_batch, target)
            
            # ================================================================
            # STEP 4: BACKWARD PASS + OPTIMIZATION
            # ================================================================
            
            # SUBSTEP 4a: Clear Old Gradients
            # PyTorch accumulates gradients, so we must zero them first!
            # Think of this as erasing the whiteboard before new calculations
            optimizer.zero_grad()
            
            # SUBSTEP 4b: Compute Gradients (Backpropagation!)
            # This is where the magic happens!
            # loss.backward() automatically computes gradients for ALL parameters
            # using the chain rule from calculus
            #
            # After this line, every tensor in `params` has its .grad filled
            # with the gradient (how much that parameter contributed to the loss)
            loss.backward()
            
            # SUBSTEP 4c: Update Weights Using Gradients
            # The optimizer (Adam) adjusts all 11 parameter tensors
            # It uses the gradients to move weights in the direction that
            # reduces loss
            #
            # New_weight = Old_weight - learning_rate × gradient
            # (Actually more complex for Adam, but that's the basic idea)
            optimizer.step()
            
            # ================================================================
            # ACCUMULATE LOSS FOR REPORTING
            # ================================================================
            # .item() extracts the loss value as a regular Python number
            # We add it to total_loss to track progress this epoch
            total_loss += loss.item()
        
        # ====================================================================
        # PROGRESS REPORTING (Every 50 Epochs)
        # ====================================================================
        # % is the modulo operator (remainder after division)
        # epoch % 50 == 0 is True when epoch is 0, 50, 100, 150, ...
        # This prints progress every 50 epochs so we can see learning happen!
        
        if epoch % 50 == 0:
            # Print the epoch number and total loss
            # .4f means format as floating point with 4 decimal places
            # You should see loss decrease over time as the model learns!
            print(f"Epoch {epoch}, loss = {total_loss:.4f}")


def test(prompt):
    """
    Test the trained model on a prompt and show top predictions.
    
    After training, we can use this function to see what the model predicts!
    It runs the model forward (no training/updates) and shows the top 5
    most likely next words according to the model.
    
    PARAMETERS:
    -----------
    prompt : str
        The input text to test on.
        Example: "Who created JavaScript ?"
    
    WHAT THIS DOES:
    ---------------
    1. Converts prompt to token IDs
    2. Runs model forward pass
    3. Converts logits to probabilities
    4. Finds top 5 most likely words
    5. Prints them with their probabilities
    
    EXPECTED OUTPUT AFTER TRAINING:
    -------------------------------
    For "Who created JavaScript ?", we should see:
    - "Brendan" with high probability (hopefully > 90%)
    - Other words with low probabilities
    
    This shows the model has learned the pattern!
    """
    # ========================================================================
    # STEP 1: Convert Prompt to Token IDs
    # ========================================================================
    ids = tokenizer(prompt)
    # Example: "Who created JavaScript ?" → [0, 1, 2, 3]
    
    # ========================================================================
    # STEP 2: Run Model Forward Pass
    # ========================================================================
    # Same as during training, but without computing gradients or updating weights
    logits = logits_for_ids(
        ids,                      # Token IDs
        embed,                    # Embedding function
        add_positional_encoding,  # Positional encoding function
        transformer_block,        # Transformer block
    )
    # Output: [VOCAB_SIZE] = [9] raw scores
    
    # ========================================================================
    # STEP 3: Convert Logits to Probabilities
    # ========================================================================
    # softmax converts any numbers into probabilities (0 to 1) that sum to 1.0
    # dim=-1 means apply softmax along the last dimension
    #
    # Example transformation:
    # Logits:       [1.0, 8.5, 0.5, 2.0, 3.0, 1.5, 0.8, 2.5, 1.2]
    # Probabilities:[0.02, 0.62, 0.01, 0.05, 0.08, 0.02, 0.01, 0.08, 0.02]
    #                     ↑ highest logit gets highest probability
    probs = torch.softmax(logits, dim=-1)
    
    # ========================================================================
    # STEP 4: Find Top 5 Most Likely Words
    # ========================================================================
    # torch.topk finds the k largest values and their indices
    # k=5 means get the top 5
    #
    # Returns:
    # - topv: the top 5 probability values
    # - topi: the indices (positions) of those top values
    #
    # Example:
    # topv: [0.62, 0.08, 0.08, 0.05, 0.02]
    # topi: [1, 4, 7, 3, 0]  (indices in VOCAB)
    topv, topi = torch.topk(probs, k=5)
    
    # ========================================================================
    # STEP 5: Print Results
    # ========================================================================
    # zip pairs up elements from topv and topi
    # For each pair (probability_value, vocab_index):
    
    for v, idx in zip(topv, topi):
        # idx.item() converts tensor index to Python integer
        # VOCAB[...] looks up the word at that index
        # v.detach() detaches tensor from computation graph
        # float(...) converts to Python float for printing
        #
        # Example output line:
        # "created 0.6231"  (the word "created" has 62.31% probability)
        print(VOCAB[idx.item()], float(v.detach()))


# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
# This code only runs when you execute this file directly (not when importing it)
#
# if __name__ == "__main__" is a Python idiom that means:
# "Only run this code if this file is the main program being run"
#
# This lets us import functions from this file without automatically running training

if __name__ == "__main__":
    # ========================================================================
    # STEP 1: Train the Model
    # ========================================================================
    # Call the train() function to optimize all parameters
    # This will run for 500 epochs and print progress every 50 epochs
    # You should see the loss decrease over time!
    #
    # Initial loss: ~2.5-3.0 (random guessing)
    # Final loss: <0.1 (learned the pattern!)
    #
    # Training takes a few seconds on a modern computer
    print("Starting training...")
    train()
    
    # ========================================================================
    # STEP 2: Test the Trained Model
    # ========================================================================
    # After training, test the model on one of our training examples
    # We expect it to correctly predict "Brendan" for "Who created JavaScript ?"
    print("\nAfter training:\n")
    test("Who created JavaScript ?")
    
    # EXPECTED OUTPUT (after successful training):
    # Brendan 0.9234  ← High probability! Model learned correctly
    # Guido 0.0312
    # Linus 0.0245
    # ... (other words with very low probabilities)
    #
    # If you see "Brendan" with the highest probability (>80%), training worked!
    #
    # Try testing other prompts:
    # test("Who created Python ?")   # Should predict "Guido"
    # test("Who created Linux ?")    # Should predict "Linus"
