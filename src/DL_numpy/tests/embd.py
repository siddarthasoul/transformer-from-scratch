import numpy as np
import json

# 1. Load trained weights and vocabulary
checkpoint_path = "/home/soul/Desktop/re/DL_numpy/checkpoint/latest.npz"
vocab_path = "/home/soul/Desktop/re/DL_numpy/data/vocab/vocab.json"

checkpoint = np.load(checkpoint_path)
param_0 = checkpoint["param_0"]  # Shape: (1003, 32)

with open(vocab_path, "r", encoding="utf-8") as f:
    raw_vocab = json.load(f)

# 2. Normalize vocabulary to guarantee {word (str) -> word_id (int)} mapping
first_key = list(raw_vocab.keys())[0]
if str(first_key).isdigit():
    # Vocabulary was saved as ID -> Word, so we reverse it
    vocab = {str(v): int(k) for k, v in raw_vocab.items()}
else:
    # Vocabulary was saved as Word -> ID
    vocab = {str(k): int(v) for k, v in raw_vocab.items()}

# 3. Safely look up the word
word = "is"

# Check exact match or find partial/subword matches (e.g. case insensitive or spaces)
matches = [k for k in vocab.keys() if word.lower() in k.lower()]

if matches:
    matched_token = matches[0]
    word_id = vocab[matched_token]
    word_embedding = param_0[word_id]

    print(f"Matched Token : '{matched_token}'")
    print(f"Token ID      : {word_id}")
    print(f"Embedding shape: {word_embedding.shape}")
    print("Vector values :\n", word_embedding)
else:
    print(f"Error: Neither '{word}' nor any matching subword token was found in vocab.json.")
    print("Sample vocabulary words:", list(vocab.keys())[:10])