# README

## Project Overview

This project implements a text generation system using a **Decoder-Only Transformer model** to conduct a comparative study of different **tokenization methods**: Character, Simple Word, and Byte-Pair Encoding (BPE). The system processes a text corpus (e.g., Shakespeare) and trains the same Transformer architecture independently for each tokenization strategy.

The primary goal is to quantitatively and qualitatively evaluate the impact of the tokenization choice on the language model's performance, text generation quality, and diversity. This aligns with advanced requirements of the DD2424 project, specifically investigating modern tokenization techniques and the Transformer architecture.

---

## Project Structure and Key Classes

This script (`main_bep&char.py`) contains several key components:

### `Config`
- **Purpose**: Centralizes all hyperparameters and configuration settings for the experiments, such as sequence length, model dimensions, learning rate, batch size, epochs, tokenization specific settings (like BPE vocab size), evaluation parameters, and file paths.

### Tokenization Functions (`char_tokenize`, `word_tokenize`, `bpe_tokenize`)
- **Purpose**: These functions are responsible for converting the raw text corpus into sequences of numerical IDs based on their respective tokenization algorithms.
- **`char_tokenize`**: Splits text into individual characters and special tokens.
- **`word_tokenize`**: Splits text into words and special tokens (whitespace tokenization). Can optionally load pre-trained GloVe embeddings if a path is provided in the config and `gensim` is installed.
- **`bpe_tokenize`**: Trains and applies a Byte-Pair Encoding tokenizer to the text, creating a vocabulary of sub-word units. Requires the `tokenizers` library.

### `TextGenerationDataset`
- **Purpose**: A `torch.utils.data.Dataset` implementation that takes the token ID sequence and prepares it for training by creating input-output pairs (sliding windows) of a specified sequence length. Optionally supports data augmentation if `nlpaug` is installed and configured.

### `DecoderOnlyTransformerLanguageModel`
- **Purpose**: Defines the core neural network architecture used across all experiments. It is a standard decoder-only Transformer model, similar to architectures used in models like GPT.
- **Architecture**: Consists of an embedding layer, positional encoding, multiple Transformer Decoder layers with multi-head attention, and a final linear layer to predict the next token probabilities over the vocabulary. This same class is instantiated for each tokenization experiment, only varying in the input vocabulary size and potentially embedding initialization (for Word + GloVe).

### `ControlledTextGenerator` (Main Controller)
- **Purpose**: Acts as the orchestrator for the entire experimental pipeline.
- **Responsibilities**:
    - Loads and preprocesses the raw text data.
    - Initializes and stores the different tokenizer instances/information.
    - Manages the training and validation data splits.
    - Iterates through each configured tokenization method.
    - For each method:
        - Converts the corpus to token IDs.
        - Builds the `TextGenerationDataset` and `DataLoader`s.
        - Instantiates the `DecoderOnlyTransformerLanguageModel` with the appropriate vocabulary size and embedding initialization.
        - Runs the training loop.
        - Performs comprehensive evaluation (Perplexity, BLEU, Distinct-N, Average Length).
        - Generates sample texts.
        - Analyzes the effect of sampling temperature.
    - Logs quantitative results to a file (`evaluation_log.txt`).
    - Saves generated sample texts to a file (`sample_outputs.txt`).
    - Generates comparative plots (training curves, temperature analysis plots).

---

## How It Works

The script performs a series of independent experiments, one for each specified tokenization method, using a consistent Transformer model architecture.

1.  **Data Loading & Preprocessing**: Reads text from input files and applies basic preprocessing, including adding synthetic role/emotion tokens based on patterns (if applicable to the dataset format).
2.  **Tokenizer Preparation**: Initializes Character, Word, and BPE tokenizers based on the preprocessed corpus and config settings. Word tokenization can optionally use pre-trained GloVe embeddings.
3.  **Iterative Experimentation**: The script loops through the prepared tokenizers:
    * **Tokenization**: The entire corpus is converted into a sequence of token IDs using the current tokenizer.
    * **Data Preparation**: The token IDs are split into training and validation sets, and `TextGenerationDataset`/`DataLoader`s are created. Data augmentation might be applied during this step.
    * **Model Instantiation**: A `DecoderOnlyTransformerLanguageModel` is created, configured with the vocabulary size from the current tokenizer. For the Word tokenizer, pre-trained embeddings are loaded if available.
    * **Training**: The model is trained on the tokenized training data for a fixed number of epochs, minimizing the next-token prediction loss. Validation perplexity is tracked.
    * **Evaluation**: After training, the model's performance is evaluated quantitatively (Perplexity, BLEU, Distinct-N, Average Length) and qualitatively (generating text samples). The impact of different sampling temperatures on generated text characteristics is also analyzed.
    * **Results Logging**: The evaluation results and generated samples are saved to output files, and plots are generated to visualize training progress and temperature analysis.

This structured approach allows for a direct comparison of how the choice of tokenization impacts the learning process and the quality of generated text when the underlying model architecture remains constant.

---

## How to Run

To run the comparative experiments, you need to have Python and the required libraries installed. It's highly recommended to use a virtual environment.

1.  **Prerequisites**:
    * Python 3.10
    * PyTorch (`pip install torch torchvision torchaudio`)
    * NumPy (`pip install numpy`)
    * Scikit-learn (`pip install scikit-learn`)
    * `tokenizers` (`pip install tokenizers`)
    * NLTK (`pip install nltk`, you might also need to download NLTK data if prompted)
    * Matplotlib (`pip install matplotlib`)
    * Optional (for data augmentation): `nlpaug` (`pip install nlpaug[recommended]`)
    * Optional (for GloVe embeddings): `gensim` (`pip install gensim`)

2.  **Download Data**: Obtain your training text file(s) (e.g., `shakespeare.txt`).
3.  **Download GloVe (Optional)**: If you want to use pre-trained Word embeddings, download GloVe vectors (e.g., glove.6B.zip from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)) and update the `glove_path` in the `Config` class.
4.  **Run the Script**: Open your terminal or command prompt, navigate to the directory containing `main_bep&char.py`, and run the script, providing the path(s) to your text file(s).

```bash
python main_bep&char.py --file_paths path/to/your/text_file.txt [path/to/another_file.txt ...]