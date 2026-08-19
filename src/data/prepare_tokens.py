import os
import numpy as np
from tokenizer.encode import Encoder

def process_tokens():
    output_dir = "src/data/processed"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize tokenizer
    encoder = Encoder()

    # 2. Read raw training corpus
    with open("src/data/raw/train.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 3. Encode text into token IDs
    token_ids = encoder.encode(text)

    # 4. Save as int32 NumPy array
    token_ids = np.array(token_ids, dtype=np.int32)
    output_path = os.path.join(output_dir, "token_ids.npy")
    
    np.save(output_path, token_ids)

    print("=" * 40)
    print("Tokenization Complete!")
    print(f"Saved to     : {output_path}")
    print(f"Total Tokens : {len(token_ids):,}")
    print("=" * 40)

if __name__ == "__main__":
    process_tokens()