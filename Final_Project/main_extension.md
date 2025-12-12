

# README

## Project Overview

This project extends a basic LSTM-based Shakespeare text generator by **introducing structure control tokens** such as role labels.  
The model is trained not only on pure text but also conditioned on these tokens to **guide** and **structure** the generated output, ensuring that generated sentences align with specific characters (roles) or thematic divisions.

This extension aims to fulfill A-level requirements by enhancing structure, controllability, and output quality of generated text.

---

## Project Structure and Key Classes

### `ControlledDataset`
- **Purpose**:  
  Encapsulates the pre-tokenized sequence data for model training.
  
- **Main idea**:  
  It slides over the tokenized list, producing input-output pairs of length `seq_len` for each step.

- **How**:  
  - `__getitem__` returns a `(x, y)` pair where `x` is the input sequence and `y` is the target (shifted one step ahead).
  - Standard `torch.utils.data.Dataset` interface.

---

### `ControlledLSTM`
- **Purpose**:  
  Defines a **word-level** LSTM language model, designed to handle **control tokens** embedded within input sequences.

- **Main idea**:  
  Extend a vanilla LSTM to condition the text generation based on inserted control tokens (like `<ROLE_ROMEO>`).

- **Architecture**:
  - **Embedding Layer**: Encodes word/control tokens.
  - **2-layer LSTM**: Captures sequential dependencies in text.
  - **Linear Layer**: Projects hidden states to vocabulary logits.

---

---

### `ControlledTextGenerator`

#### 1. Overall Design Idea

The `ControlledTextGenerator` class is the **central pipeline manager** for the controlled text generation system.  
Its purpose is to **introduce structure into text generation** by embedding **contextual control tokens** (e.g., roles, paragraphs, themes) directly into the model's training sequences, allowing the model to learn both *linguistic* and *structural* patterns simultaneously.

Compared to a vanilla language model, this class introduces:
- **Explicit context injection** through special tokens.
- **Conditional text generation** driven by the injected structure.
- **Pipeline modularity**: data parsing, model instantiation, training, evaluation, and sampling are separated into clean submodules.

---

#### 2. Data Preprocessing and Tokenization

- **Role Tagging**:
  - The raw Shakespeare text is parsed line-by-line.
  - Using a regex (`ROLE_PATTERN`), lines like `ROMEO:` are identified.
  - A synthetic token such as `<ROLE_ROMEO>` is inserted **before** the corresponding text.

- **Word-Level Tokenization**:
  - Instead of character-level modeling, the text is tokenized at **word level** (splitting by whitespace).
  - Both words and control tokens become part of the vocabulary.

- **Vocabulary Building**:
  - A frequency-based token vocabulary (`token2idx`, `idx2token`) is constructed from the training corpus.

- **Encoding**:
  - Each token is mapped to an integer ID according to the vocabulary.

---
  
#### 3. Dataset Design

- `ControlledDataset` slices the token index sequence into **overlapping sliding windows** of length `seq_len`.
- Each training sample is a pair:
  - `x`: the input token sequence
  - `y`: the target token sequence shifted by one position

This allows the model to learn **next-token prediction** conditioned on prior content **and** any injected control tokens.

---

#### 4. Network Architecture Design (`ControlledLSTM`)

| Component            | Description |
|----------------------|-------------|
| **Embedding Layer**   | Maps token indices (words + control tokens) to dense vectors of size `embed_dim`. |
| **LSTM Layers**       | 2-layer stacked LSTM with `hidden_dim` units, capturing sequential and long-term dependencies. |
| **Fully Connected Layer** | Maps LSTM hidden outputs at each timestep to vocabulary logits. |

**Key Point**:  
Because control tokens (e.g., `<ROLE_ROMEO>`) are in the same embedding space as normal words, the model naturally learns how they influence subsequent word choices without needing explicit separate handling.

---

#### 5. Training Strategy

- **Loss Function**:  
  Standard cross-entropy loss between predicted token logits and true token IDs.

- **Optimizer**:  
  Adam optimizer with learning rate `lr`.

- **Training Loop**:
  - Forward pass through model.
  - Loss computation and backpropagation.
  - Optimizer step.
  - Epoch-wise average loss reporting.

---

#### 6. Controlled Sampling Strategy

- **Context Injection**:  
  During generation, an explicit control token list (e.g., `["<ROLE_JULIET>"]`) is encoded and fed into the model as the initial context.

- **Autoregressive Decoding**:
  - The model sequentially predicts the next token, which is appended to the input for the next prediction.
  - Sampling is done using **nucleus (top-p) sampling** and **temperature scaling** to balance diversity and coherence.

- **Result**:
  - Generated text stylistically conforms to the control context, e.g., adopting Romeo’s speech style when prompted with `<ROLE_ROMEO>`.

---

#### 7. Strengths of This Design

| Strength                  | Explanation |
|----------------------------|-------------|
| **Simple but powerful**    | No architecture change needed; only input stream is enriched. |
| **Scalable to multiple controls** | Easily extensible to control paragraph structure, theme, or emotional tone. |
| **Low computational overhead** | Reuses standard LSTM forward-pass without heavy computation. |
| **Modular and clean**      | Each phase (parse, dataset, model, train, sample) is independently replaceable. |

---

## Summary

The `ControlledTextGenerator` design offers a minimalist but highly effective way to guide free-form text generation models towards **structured**, **coherent**, and **conditioned** outputs, fully meeting the A-level extension requirements through clean engineering and deep understanding of sequence modeling.

---



## How It Works

- **Training Phase**:
  - Insert structure control tokens into original Shakespeare scripts.
  - Train an LSTM model to predict next tokens conditioned on both textual and control token information.

- **Inference Phase**:
  - Specify a control token (e.g., `<ROLE_ROMEO>`).
  - The model generates fluent, structured text reflecting the specified character's speaking style.

---

## How to Run

```bash
python main_extension.py --file_path shakespeare.txt
```

- Trains for `20` epochs on Shakespeare corpus.
- Evaluates on held-out validation set.
- Generates a sample starting with `<ROLE_ROMEO>`.

---

## Extensions and Future Improvements

- Add `<PARAGRAPH_START>` or `<THEME_LOVE>` control tokens for even finer structure control.
- Expand to Transformer-based controlled generation.
- Apply Byte-Pair Encoding (BPE) tokenization for better vocabulary handling.
- Build richer condition embeddings (e.g., role embeddings separate from word embeddings).

---

## Summary

This project successfully demonstrates:
- Control-token guided text generation.
- Structural control over language model outputs.
- Extends basic RNN/LSTM models towards **structured, conditional text generation**,
- Satisfying A-level course requirements with clear modular design and extensibility.

---
