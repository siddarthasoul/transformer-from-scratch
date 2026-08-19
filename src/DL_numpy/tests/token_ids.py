import numpy as np
import json


# 1. Load your numpy file
token_ids = np.load("/home/soul/Desktop/re/src/data/processed/token_ids.npy")

print("=== ARRAY METADATA ===")
print("Data Type :", token_ids.dtype)
print("Shape     :", token_ids.shape)
print("Total IDs :", token_ids.size)
print("\n=== FIRST 20 RAW TOKEN IDs ===")
print(token_ids.flatten()[:20])

# 2. Decode IDs into Words (if you have a vocab dictionary)
# Replace 'vocab.json' with your vocabulary/mapping file if available
try:
    with open("/home/soul/Desktop/re/src/data/vocab/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    # Create ID -> Word mapping
    id_to_word = {idx: word for word, idx in vocab.items()}

    print("\n=== DECODED SAMPLE TEXT ===")
    sample_ids = token_ids.flatten()[:30]
    decoded_words = [id_to_word.get(int(idx), f"<UNK_{idx}>") for idx in sample_ids]
    
    print(" ".join(decoded_words))

except FileNotFoundError:
    print("\n[!] 'vocab.json' not found. Displaying raw IDs only.")