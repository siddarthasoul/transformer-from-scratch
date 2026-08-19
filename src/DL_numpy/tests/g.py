import numpy as np

# 1. Assume this is the final contextual vector for "I am ai" after W* transformation
final_context_vector = np.array([0.85, -0.40, 1.20, 0.15, -0.90, 0.50, 0.10, 0.75]) # Shape: (8,)

# 2. LM Head Weight Matrix: maps 8 features to 4 vocabulary words
# Shape: (8 features, 4 vocabulary tokens)
W_lm_head = np.random.randn(8, 4) 

vocab = ["sidd", "student", "smart", "banana"]

# 3. Calculate raw prediction scores (logits)
logits = np.dot(final_context_vector, W_lm_head)

# 4. Softmax function to convert logits into percentages
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

probabilities = softmax(logits)

# 5. Print prediction results
for word, prob in zip(vocab, probabilities):
    print(f"Word: '{word}' -> Probability: {prob * 100:.2f}%")

predicted_index = np.argmax(probabilities)
print(f"\nNext predicted token: '{vocab[predicted_index]}'")