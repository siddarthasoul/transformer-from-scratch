import json
import os


def train_bpe():
    input_file = "src/data/raw/train.txt"
    vocab_dir = "src/data/vocab"

    os.makedirs(vocab_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Pre-tokenize text into words prefixed with word-boundary token '▁'
    words = []
    for word in text.split():
        word = "▁" + word
        words.append(list(word))

    vocab = set()
    for word in words:
        for ch in word:
            vocab.add(ch)

    special_tokens = ["<unk>", "<s>", "</s>"]
    merges = []
    target_vocab_size = 1000

    print(f"Initial Vocabulary Size: {len(vocab)}")

    while len(vocab) < target_vocab_size:
        # 1. Count adjacent pairs across all words
        pair_count = {}
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_count[pair] = pair_count.get(pair, 0) + 1

        if not pair_count:
            break

        # 2. Find most frequent pair
        best_pair = max(pair_count, key=pair_count.get)
        best_count = pair_count[best_pair]

        new_token = best_pair[0] + best_pair[1]

        # Skip if token already exists instead of breaking the entire loop
        if new_token in vocab:
            # Mask out this pair to try the next best
            del pair_count[best_pair]
            if not pair_count:
                break
            continue

        vocab.add(new_token)

        merges.append(
            {
                "left": best_pair[0],
                "right": best_pair[1],
                "token": new_token,
                "count": best_count,
            }
        )

        print(
            f"MERGE ({len(vocab)}/{target_vocab_size}): "
            f"'{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}' ({best_count})"
        )

        # 3. Replace all occurrences of best_pair in the dataset
        updated_words = []
        for word in words:
            merged_word = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == best_pair[0]
                    and word[i + 1] == best_pair[1]
                ):
                    merged_word.append(new_token)
                    i += 2
                else:
                    merged_word.append(word[i])
                    i += 1
            updated_words.append(merged_word)

        words = updated_words

    # 4. Build final vocabulary mapping token -> integer ID
    final_vocab = {}
    current_id = 0

    for token in special_tokens:
        final_vocab[token] = current_id
        current_id += 1

    for token in sorted(vocab):
        final_vocab[token] = current_id
        current_id += 1

    # Save vocabulary mappings
    with open(
        os.path.join(vocab_dir, "vocab.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(final_vocab, f, indent=4, ensure_ascii=False)

    with open(
        os.path.join(vocab_dir, "merges.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(merges, f, indent=4, ensure_ascii=False)

    print("\nTraining Complete!")
    print(f"Final Vocabulary Size: {len(final_vocab)}")
    print(f"Merge Rules Generated: {len(merges)}")


if __name__ == "__main__":
    train_bpe()