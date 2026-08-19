import numpy as np


class Loss:

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)


    def forward(self, logits, target):

        B, T, V = logits.shape

        logits = logits.reshape(B * T, V)
        target = target.reshape(B * T)

        self.target = target
        self.probs = []

        losses = []

        for i in range(len(target)):

            probs = self.softmax(logits[i])

            self.probs.append(probs)

            losses.append(
                -np.log(probs[target[i]] + 1e-12)
            )

        self.probs = np.array(self.probs)

        return np.mean(losses)


    def backward(self):

        grad = self.probs.copy()

        for i in range(len(self.target)):
            grad[i, self.target[i]] -= 1

        grad /= len(self.target)

        return grad