import os
import numpy as np


DATA_DIR = "src/utils/test/data"

INPUT_FILE = os.path.join(DATA_DIR, "raw/data.txt")
ENCODE_VOCAB_FILE = os.path.join(DATA_DIR, "vocab/encode_vocab.txt")
DECODE_VOCAB_FILE = os.path.join(DATA_DIR, "vocab/decode_vocab.txt")
TOKEN_IDS_FILE = os.path.join(DATA_DIR, "processed/token_ids.npy")


def prepare_data():

    # -----------------------------------------
    # 1. Read text
    # -----------------------------------------

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        text = file.read()

    text = text.lower()

    print("=" * 60)
    print("RAW TEXT")
    print("=" * 60)
    print(text)


    # -----------------------------------------
    # 2. Basic whitespace tokenizer
    # -----------------------------------------

    words = text.split()

    print("\n" + "=" * 60)
    print("TOKENS")
    print("=" * 60)
    print(words)


    # -----------------------------------------
    # 3. Build vocabulary
    # -----------------------------------------

    vocab = {}

    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)

    print("\n" + "=" * 60)
    print("VOCABULARY")
    print("=" * 60)

    for word, token_id in vocab.items():
        print(f"{word:20} -> {token_id}")


    # -----------------------------------------
    # 4. Encode words -> token IDs
    # -----------------------------------------

    token_ids = []

    for word in words:
        token_ids.append(vocab[word])

    token_ids = np.array(token_ids, dtype=np.int32)

    print("\n" + "=" * 60)
    print("TOKEN IDS")
    print("=" * 60)
    print(token_ids)


    # -----------------------------------------
    # 5. Save encode vocabulary
    # -----------------------------------------

    with open(ENCODE_VOCAB_FILE, "w", encoding="utf-8") as file:
        for word, token_id in vocab.items():
            file.write(f"{word}\t{token_id}\n")


    # -----------------------------------------
    # 6. Save decode vocabulary
    # -----------------------------------------

    with open(DECODE_VOCAB_FILE, "w", encoding="utf-8") as file:
        for word, token_id in vocab.items():
            file.write(f"{token_id}\t{word}\n")


    # -----------------------------------------
    # 7. Save token IDs
    # -----------------------------------------

    np.save(TOKEN_IDS_FILE, token_ids)


    # -----------------------------------------
    # 8. Summary
    # -----------------------------------------

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    print(f"Vocabulary size : {len(vocab)}")
    print(f"Total tokens    : {len(token_ids)}")
    print(f"Token IDs       : {TOKEN_IDS_FILE}")
    print(f"Encode vocab    : {ENCODE_VOCAB_FILE}")
    print(f"Decode vocab    : {DECODE_VOCAB_FILE}")


if __name__ == "__main__":
    prepare_data()