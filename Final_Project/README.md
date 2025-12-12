# DD2424 Deep Learning - Text Generation Project

**Course**: DD2424 Deep Learning in Data Science  
**Group**: 43  
**KTH Royal Institute of Technology**

---

## 📖 Project Overview

This project implements **neural network-based text generation** using the Shakespeare corpus. We explore and compare different model architectures and tokenization strategies to generate high-quality, diverse text.

### Key Features

- 🔤 **Multiple Tokenization Methods**: Character-level, Word-level, and Byte-Pair Encoding (BPE)
- 🧠 **Model Architectures**: RNN, LSTM, and Decoder-only Transformer
- 🎭 **Controlled Generation**: Role-based text generation with control tokens
- 📊 **Comprehensive Evaluation**: Perplexity, BLEU, Distinct-N metrics, and temperature analysis

---

## 🗂️ Project Structure

```
DD2424_43_Group_final_project/
├── main_basic_fast.py      # Basic RNN/LSTM character-level model
├── main_bep&char.py        # Transformer with tokenization comparison
├── main_extension.py       # Controlled generation with role tokens
├── shakespeare.txt         # Training corpus
├── main_bep&char.md        # Documentation for BPE/Char comparison
├── main_extension.md       # Documentation for controlled generation
└── README.md               # This file
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch numpy scikit-learn tokenizers nltk matplotlib
# Optional (for data augmentation)
pip install nlpaug
```

### Running the Models

#### 1. Basic RNN/LSTM Model
```bash
python main_basic_fast.py
```
Trains vanilla RNN, 1-layer LSTM, and 2-layer LSTM models on character-level tokenization.

#### 2. Transformer with Tokenization Comparison
```bash
python main_bep\&char.py --file_paths shakespeare.txt
```
Compares BPE, character, and word tokenization using the same Decoder-only Transformer architecture.

#### 3. Controlled Text Generation
```bash
python main_extension.py --file_path shakespeare.txt
```
Generates text conditioned on character roles (e.g., ROMEO, JULIET).

---

## 📋 Model Configurations

### Transformer Model (`main_bep&char.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `seq_len` | 128 | Context window length |
| `embed_dim` | 256 | Embedding dimension |
| `hidden_dim` | 512 | Feed-forward hidden dimension |
| `num_heads` | 8 | Number of attention heads |
| `num_layers` | 4 | Number of Transformer layers |
| `batch_size` | 32 | Training batch size |
| `epochs` | 10 | Training epochs |

### Training Optimizations

- **Learning Rate Scheduler**: Cosine Annealing
- **Label Smoothing**: 0.1 (prevents overconfidence)
- **Early Stopping**: Patience of 5 epochs
- **Repetition Penalty**: 1.2 (reduces repetitive text)
- **Gradient Clipping**: 1.0 (training stability)
- **Mixed Precision**: FP16 training for efficiency

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Perplexity** | Model uncertainty (lower is better) |
| **BLEU Score** | N-gram overlap with reference text |
| **Distinct-2/3** | Diversity of generated text (higher is better) |
| **Average Length** | Mean length of generated samples |

---

## 📈 Generated Outputs

After training, the following files are generated:

| File | Description |
|------|-------------|
| `comparison_curve_loss.png` | Training loss comparison |
| `comparison_curve_perplexity.png` | Validation perplexity comparison |
| `*_temperature_analysis.png` | Temperature scaling effect on diversity |
| `sample_outputs.txt` | Generated text samples |
| `evaluation_log.txt` | Quantitative evaluation results |

---

## 🖥️ Hardware Requirements

| Component | Recommended | Minimum |
|-----------|-------------|---------|
| **GPU** | NVIDIA 4GB+ VRAM | CPU only (slower) |
| **RAM** | 8GB+ | 4GB |
| **Storage** | 1GB free | 500MB |

**Note**: The code automatically detects and uses:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon Macs)
- CPU (fallback)

---

## 🧪 Experiments

### Tokenization Comparison

We compare three tokenization strategies on the same Transformer architecture:

1. **Character-level**: Simple vocabulary, struggles with long-range dependencies
2. **Word-level**: Captures semantics, but OOV issues
3. **BPE**: Best of both worlds, handles rare words gracefully

### Temperature Scaling

Temperature affects the diversity-quality tradeoff:
- Low temperature (0.5): More deterministic, potentially repetitive
- High temperature (1.5): More creative, potentially incoherent

---

## 📚 References

- Vaswani et al. "Attention Is All You Need" (2017)
- Sennrich et al. "Neural Machine Translation of Rare Words with Subword Units" (2016)
- Holtzman et al. "The Curious Case of Neural Text Degeneration" (2020)

---

## 👥 Contributors

- Project Group 43, DD2424 Deep Learning in Data Science
- KTH Royal Institute of Technology

---

## 📄 License

This project is for educational purposes as part of the DD2424 course at KTH.
