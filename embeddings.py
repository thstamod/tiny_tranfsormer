import torch
D_MODEL = 8 
VOCAB_SIZE = 4

W = torch.randn(VOCAB_SIZE, D_MODEL, requires_grad=True) 

def embed(ids): # creates an array of shape (VOCAB_SIZE, D_MODEL) with random values
    return W[ids]

if __name__ == "__main__":
    ids = [0, 1, 2]
    print(embed(ids))
