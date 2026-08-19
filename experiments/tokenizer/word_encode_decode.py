VOCAB_DIR = "src/utils/test/data/vocab"


def load_encode_vocab():
    vocab = {}

    with open(f"{VOCAB_DIR}/encode_vocab.txt", "r", encoding="utf-8") as f:
        for line in f:
            word, token_id = line.strip().split("\t")
            vocab[word] = int(token_id)

    return vocab


def load_decode_vocab():
    vocab = {}

    with open(f"{VOCAB_DIR}/decode_vocab.txt", "r", encoding="utf-8") as f:
        for line in f:
            token_id, word = line.strip().split("\t")
            vocab[int(token_id)] = word

    return vocab


def encode(text, vocab):
    tokens = text.lower().split()

    token_ids = []

    for token in tokens:
        if token in vocab:
            token_ids.append(vocab[token])
        else:
            print(f"Unknown word: {token}")

    return token_ids


def decode(token_ids, vocab):
    words = []

    for token_id in token_ids:
        if token_id in vocab:
            words.append(vocab[token_id])
        else:
            print(f"Unknown token ID: {token_id}")

    return words


def main():
    text = "Schedulers adjust learning rates."

    encode_vocab = load_encode_vocab()
    decode_vocab = load_decode_vocab()

    # Text → Token IDs
    token_ids = encode(text, encode_vocab)

    print("=" * 50)
    print("ENCODING")
    print("=" * 50)

    print("Text:")
    print(text)

    print("\nToken IDs:")
    print(token_ids)

    # Token IDs → Text
    decoded_words = decode(token_ids, decode_vocab)

    print("\n" + "=" * 50)
    print("DECODING")
    print("=" * 50)

    print("Token IDs:")
    print(token_ids)

    print("\nDecoded Text:")
    print(" ".join(decoded_words))


if __name__ == "__main__":
    main()