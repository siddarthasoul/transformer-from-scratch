import json

class Decoder:

    def __init__(self, vocab_file="src/data/vocab/vocab.json"):
        self.vocab_file = vocab_file
        self.vocab = {}
        self.reverse_vocab = {}

        self.load_vocab()

    def load_vocab(self):
        with open(self.vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        # Map ID -> String Token
        self.reverse_vocab = {
            int(idx) if isinstance(idx, (int, str)) and str(idx).isdigit() else idx: token
            for token, idx in self.vocab.items()
        }
        # Invert to handle int lookup safely
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def decode(self, ids):
        tokens = []
        for idx in ids:
            token = self.reverse_vocab.get(idx, "<unk>")
            if token in ("<unk>", "<s>", "</s>"):
                continue
            tokens.append(token)

        # Concatenate tokens and replace word boundary markers with spaces
        full_text = "".join(tokens)
        
        # Replace boundary markers with spaces, cleaning up leading whitespace
        reconstructed = full_text.replace("▁", " ").strip()
        
        return reconstructed