import json


class Encoder:

    def __init__(
        self,
        vocab_file="src/data/vocab/vocab.json",
        merges_file="src/data/vocab/merges.json",
    ):
        self.vocab_file = vocab_file
        self.merges_file = merges_file

        self.vocab = {}
        self.merges = []

        self.load_vocab()
        self.load_merges()

    def load_vocab(self):
        with open(self.vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

    def load_merges(self):
        with open(self.merges_file, "r", encoding="utf-8") as f:
            self.merges = json.load(f)

    def tokenize(self, text):
        words = []
        for word in text.split():
            # Add word boundary marker
            word = "▁" + word
            words.append(list(word))
        return words

    def apply_merges(self, words):
        for merge in self.merges:
            left = merge["left"]
            right = merge["right"]
            token = merge["token"]

            new_words = []
            for word in words:
                merged = []
                i = 0
                while i < len(word):
                    if (
                        i < len(word) - 1
                        and word[i] == left
                        and word[i + 1] == right
                    ):
                        merged.append(token)
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                new_words.append(merged)
            words = new_words
        return words

    def encode(self, text):
        words = self.tokenize(text)
        words = self.apply_merges(words)

        ids = []
        unk_id = self.vocab.get("<unk>", 0)

        for word in words:
            for token in word:
                ids.append(self.vocab.get(token, unk_id))

        return ids


