VOCAB = ["Who", "created", "JavaScript", "?", "Brendan", "Guido", "Linus"]
tok2id = {
    "Who": 0,
    "created": 1,
    "JavaScript": 2,
    "?": 3,
    "Brendan": 4,
    "Guido": 5,
    "Linus": 6,
}
id2tok = {
    0: "Who",
    1: "created",
    2: "JavaScript",
    3: "?",
    4: "Brendan",
    5: "Guido",
    6: "Linus",
}

def tokenizer(text):
    return [tok2id[n] for n in text.split() if n in tok2id]

#print(tokenizer("Who created JavaScript ?"))