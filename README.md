# Transformer From Scratch

A Transformer language model implemented from scratch using NumPy,
then reproduced using PyTorch.

The project is designed to understand the internal architecture,
mathematics, training process, and text generation of a Transformer
language model.

---

## 1. What This Project Demonstrates

- Tokenization and vocabulary
- BPE tokenizer
- Token IDs and dataset preparation
- Token embeddings
- Multi-Head Attention
- Causal attention
- Rotary Position Embeddings (RoPE)
- Layer Normalization
- MLP / Feed-Forward Network
- Residual connections
- Linear layers
- Softmax
- Language Model Head
- Cross-Entropy Loss
- Backpropagation
- Adam optimizer
- Training loop
- Model checkpointing
- Text generation

---

## 2. Model Flow

The input text passes through the following pipeline:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Token Embedding
 ↓
Transformer Blocks
 │
 ├── RoPE
 │
 ├── Multi-Head Attention
 │
 ├── Add & Norm
 │
 ├── MLP / Feed-Forward Network
 │
 └── Add & Norm
 ↓
Language Model Head
 ↓
Logits
 ↓
Next-Token Prediction
```

### Flow Explanation

1. **Text**

   Raw input text is provided to the tokenizer.

2. **Tokenizer**

   The tokenizer converts the input text into tokens according to
   the learned vocabulary.

3. **Token IDs**

   Each token is mapped to an integer ID from the vocabulary.

4. **Token Embedding**

   Token IDs are converted into dense vector representations.

5. **Transformer Blocks**

   The embeddings pass through the Transformer blocks.

   Each block contains:

   - **RoPE** — provides positional information to the attention
     representations.
   - **Multi-Head Attention** — allows tokens to attend to other
     relevant tokens.
   - **Add & Norm** — applies the residual connection and
     normalization.
   - **MLP / Feed-Forward Network** — transforms the hidden
     representations.
   - **Add & Norm** — applies another residual connection and
     normalization.

6. **Language Model Head**

   The final hidden representations are projected into the
   vocabulary space.

7. **Logits**

   The model produces a score for every token in the vocabulary.

8. **Next-Token Prediction**

   The logits are converted into probabilities and used to predict
   the next token.

---

## 3. Implementations

This project contains two implementations of the Transformer
language model.

### NumPy Implementation

The NumPy implementation is built from scratch to understand the
mathematical and computational foundations of a Transformer.

It contains implementations for:

- Parameter storage
- Linear transformations
- Token embeddings
- Layer Normalization
- Softmax
- Multi-Head Attention
- RoPE
- MLP
- Transformer blocks
- Language Model Head
- Loss calculation
- Backpropagation
- Adam optimization
- Training
- Checkpointing
- Text generation

The NumPy implementation focuses on understanding the underlying
mathematics and computations rather than relying on an automatic
differentiation framework.

### PyTorch Implementation

The PyTorch implementation reproduces the Transformer architecture
using PyTorch tensors and automatic differentiation.

It uses PyTorch for:

- Tensor operations
- Automatic differentiation
- Model computation
- GPU acceleration
- Optimization
- Model training
- Checkpointing
- Text generation

The two implementations use the same overall Transformer
architecture while demonstrating two different approaches to
building and training the model.

---

## 4. Repository Structure

```text
transformer-from-scratch/
│
├── experiments/
│   ├── data/
│   │   ├── prepare_data.py
│   │   ├── processed/
│   │   │   └── token_ids.npy
│   │   ├── raw/
│   │   │   └── data.txt
│   │   └── vocab/
│   │       ├── decode_vocab.txt
│   │       └── encode_vocab.txt
│   │
│   ├── numpy_vs_pytorch.py
│   ├── tensor_basics.py
│   │
│   └── tokenizer/
│       ├── basic_word_tokenizer.py
│       └── word_encode_decode.py
│
├── src/
│   │
│   ├── data/
│   │   ├── prepare_tokens.py
│   │   ├── processed/
│   │   │   └── token_ids.npy
│   │   ├── raw/
│   │   │   └── train.txt
│   │   └── vocab/
│   │       ├── merges.json
│   │       └── vocab.json
│   │
│   ├── dataset/
│   │   ├── dataloader.py
│   │   └── dataset.py
│   │
│   ├── DL_numpy/
│   │   ├── checkpoint/
│   │   ├── generate.py
│   │   ├── model/
│   │   ├── tests/
│   │   ├── training/
│   │   └── train.py
│   │
│   ├── DL_pytorch/
│   │   ├── checkpoint/
│   │   ├── generate.py
│   │   ├── model/
│   │   ├── tests/
│   │   └── train.py
│   │
│   ├── tokenizer/
│   │   ├── bpe_train.py
│   │   ├── decode.py
│   │   └── encode.py
│   │
│   └── utils/
│       ├── checkpoint/
│       │   ├── checkpoint_np.py
│       │   └── checkpoint_torch.py
│       └── config.py
│
├── README.md
└── requirements.txt
```

### `src/data/`

Contains training data preparation, raw training data, processed
token IDs, and tokenizer vocabulary files.

### `src/dataset/`

Contains the dataset and DataLoader implementations used for
training.

### `src/tokenizer/`

Contains the BPE tokenizer implementation, including:

- BPE training
- Encoding
- Decoding

### `src/DL_numpy/`

Contains the Transformer implementation built using NumPy.

This implementation focuses on the mathematical operations and
training process behind the model.

### `src/DL_pytorch/`

Contains the Transformer implementation built using PyTorch.

This implementation uses PyTorch tensors, automatic differentiation,
and GPU acceleration.

### `src/utils/`

Contains shared utilities such as configuration and checkpoint
handling.

### `experiments/`

Contains smaller experiments used to understand the fundamentals
behind the main implementation.

These experiments include:

- Basic tokenization
- Word encoding and decoding
- Tensor operations
- NumPy vs PyTorch comparisons

---

## 5. Installation

Clone the repository:

```bash
git clone <repository-url>
cd transformer-from-scratch
```

Create a virtual environment:

```bash
python3 -m venv re
```

Activate the environment:

```bash
source re/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## 6. Running

Make sure the virtual environment is activated before running the
commands.

### NumPy

Train the NumPy implementation:

```bash
python -m src.DL_numpy.train
```

Generate text using the NumPy implementation:

```bash
python -m src.DL_numpy.generate
```

### PyTorch

Train the PyTorch implementation:

```bash
python -m src.DL_pytorch.train
```

Generate text using the PyTorch implementation:

```bash
python -m src.DL_pytorch.generate
```

---

## 7. Hardware / Environment

The project was developed and tested with:

- Python 3.12
- NumPy 2.5.1
- PyTorch 2.13.0
- CUDA runtime 13.0
- NVIDIA GeForce RTX 2050

---

## 8. Checkpoints

Training checkpoints are stored separately for each implementation.

### NumPy

```text
src/DL_numpy/checkpoint/
├── epoch_1.npz
├── epoch_2.npz
├── ...
└── latest.npz
```

The `.npz` files contain NumPy model parameters and checkpoint
data.

### PyTorch

```text
src/DL_pytorch/checkpoint/
├── initial_model.pt
└── model.pt
```

The `.pt` files contain PyTorch model/checkpoint data.

---

## 9. Example Generation

The trained model can generate text from a given prompt.

Example:

```text
Prompt:
Kernels

Generated:
Kernels extract features.
```

The NumPy implementation can also display the top predicted
next tokens and their probabilities during generation.

---

## 10. Purpose

This project was built to understand how Transformer language
models work from the ground up, rather than only using high-level
deep-learning libraries.

The project contains two implementations of the Transformer
architecture:

- **NumPy** — implements the core Transformer operations manually
  to understand the underlying mathematics and computations.

- **PyTorch** — reproduces the architecture using tensors,
  automatic differentiation, and GPU acceleration.

The goal is to understand what happens inside a Transformer during:

- Tokenization
- Embedding
- Attention
- Forward propagation
- Training
- Next-token prediction
- Text generation

---

## 11. Future Work

Possible future improvements include:

- Improve tokenizer
- Train on a larger dataset
- Increase Transformer size
- Improve training metrics
- Add attention visualization
- Add model profiling
- Experiment with fine-tuning
- Add quantization
- Optimize inference

---

## License

This project is intended as a learning and experimentation project.
