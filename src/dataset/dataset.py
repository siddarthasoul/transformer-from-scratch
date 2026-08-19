import numpy as np


class Dataset:

    def __init__(self, token_file, seq_len):
        self.tokens = np.load(token_file)
        self.seq_len = seq_len
        print(f"Tokens loaded : {len(self.tokens)}")
        print(f"Sequence length: {self.seq_len}")

    def __len__(self):
        # Subtract 1 extra so index + seq_len + 1 never overflows array bounds
        length = len(self.tokens) - self.seq_len

        if length <= 0:
            raise ValueError(
                f"Dataset too small for sequence length.\n"
                f"Tokens = {len(self.tokens)}, Required >= {self.seq_len + 1}"
            )

        return length

    def __getitem__(self, index):
        input_ids = self.tokens[index : index + self.seq_len]
        target_ids = self.tokens[index + 1 : index + self.seq_len + 1]

        return input_ids, target_ids