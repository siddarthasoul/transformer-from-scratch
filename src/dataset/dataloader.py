import numpy as np
import torch


class DataLoader:

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]

            # Drop partial batch if requested
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            batch_input = []
            batch_target = []

            for idx in batch_indices:
                x, y = self.dataset[idx]
                batch_input.append(x)
                batch_target.append(y)

            yield (
                torch.tensor(batch_input, dtype=torch.long),
                torch.tensor(batch_target, dtype=torch.long)
            )