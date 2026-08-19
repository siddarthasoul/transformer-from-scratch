import numpy as np


class Metrics:

    def average_loss(self, losses):
        """
        losses:
            list or numpy array
            Example:
            [2.1, 1.8, 0.9, 1.2]

        Returns:
            float
        """
        return np.mean(losses)


    def accuracy(self, logits, target):
        """
        logits.shape
            (seq_len, vocab_size)

        target.shape
            (seq_len,)
        """

        # Predicted token ids
        prediction = np.argmax(logits, axis=-1)

        # Count correct predictions
        correct = np.sum(prediction == target)

        # Total predictions
        total = len(target)

        return correct / total


    def perplexity(self, average_loss):
        """
        Perplexity = exp(loss)
        """
        return np.exp(average_loss)


    def summary(self, losses, logits, target):
        """
        Returns all metrics together.
        """

        avg_loss = self.average_loss(losses)

        acc = self.accuracy(
            logits,
            target
        )

        ppl = self.perplexity(
            avg_loss
        )

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "perplexity": ppl
        }