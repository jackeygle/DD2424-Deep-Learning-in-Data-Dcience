# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: Default_Project_3_Text_Generation
@File Name: main_bep&char.py
@Software: Python
@Time: May/2025
@Author: Project Group 43
@Contact:
@Version: 0.11.6
@Description:  This script implements a text generation system using a Decoder-Only Transformer model.
               It supports comparative experiments with different tokenization methods:
                  1. Character Tokenization: Treats each character as a token. Simple vocabulary,
                     often results in low perplexity but can struggle with coherence and style.
                  2. Word Tokenization: Treats each word as a token. Larger vocabulary,
                     can capture semantic meaning but struggles with out-of-vocabulary words.
                     Can optionally use pre-trained GloVe embeddings.
                  3. Byte-Pair Encoding (BPE): A sub-word tokenization method that balances vocabulary size
                     and the ability to handle unknown words. Common in modern large language models.
"""

import os
import re
import argparse
import random
import tempfile
import matplotlib.pyplot as plt
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from tokenizers import ByteLevelBPETokenizer # type: ignore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.char as nac
    NLP_AUG_AVAILABLE = True
except ImportError:
    print("nlpaug not found. Data augmentation will be disabled. Install with: pip install nlpaug[recommended]")
    NLP_AUG_AVAILABLE = False
    naw = None
    nac = None


# --- Configuration ---
class Config:
    def __init__(self):
        self.seq_len: int = 128  # 增加上下文长度以捕获更长依赖
        self.embed_dim: int = 256
        self.hidden_dim: int = 512
        self.num_heads: int = 8  # 增加注意力头数提升表达能力
        self.num_layers: int = 4
        self.lr: float = 3e-4
        self.batch_size: int = 32
        self.epochs: int = 10
        self.augment_prob: float = 0.2 if NLP_AUG_AVAILABLE else 0.0
        self.word_aug_prob: float = 0.1
        self.char_aug_prob: float = 0.05
        self.bpe_vocab_size: int = 8000
        self.weight_decay: float = 1e-5
        self.val_split_size: float = 0.1
        self.random_seed: int = 42
        self.grad_norm_clip: float = 1.0
        self.num_workers: int = 4
        self.dropout: float = 0.1
        self.glove_path: str | None = None
        self.num_evaluation_samples: int = 10
        self.evaluation_sample_max_len: int = 100
        self.num_temp_samples: int = 5
        self.plot_output_prefix: str = "comparison_curve"
        self.sample_output_file: str = "sample_outputs.txt"
        self.log_file: str = "evaluation_log.txt"
        self.temperatures: List[float] = [0.5, 0.8, 1.0, 1.2, 1.5]
        # === 新增优化参数 ===
        self.label_smoothing: float = 0.1  # 标签平滑防止过拟合
        self.use_scheduler: bool = True  # 是否使用学习率调度器
        self.scheduler_type: str = "cosine"  # "cosine" 或 "onecycle"
        self.warmup_epochs: int = 1  # warmup 轮数
        self.early_stopping_patience: int = 5  # 早停耐心值
        self.repetition_penalty: float = 1.2  # 重复惩罚系数
        self.checkpoint_dir: str = "checkpoints"  # 检查点保存目录

# --- Evaluation Metrics ---
def compute_distinct_n(sequences: List[List[Any]], n: int) -> float:
    total = 0
    unique = set()
    for seq in sequences:
        if len(seq) < n:
            continue
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:i+n])
            unique.add(ngram)
            total += 1
    return len(unique) / total if total > 0 else 0.0

def compute_average_length(sequences: List[List[Any]]) -> float:
    lengths = [len(seq) for seq in sequences if len(seq) > 0]
    return float(np.mean(lengths)) if lengths else 0.0

# --- Tokenization ---
def char_tokenize(lines: List[str], special_tokens: List[str]) -> Dict[str, Any]:
    text = "".join(lines)
    chars = sorted(list(set(text)))
    vocab = special_tokens + chars
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for i, token in enumerate(vocab)}
    return {
        'type': 'char',
        'vocab_size': len(vocab),
        'token_to_id': token_to_id,
        'id_to_token': id_to_token,
    }

def word_tokenize(lines: List[str], special_tokens: List[str], word_vecs=None) -> Dict[str, Any]:
    text = " ".join(lines)
    words = text.split()
    unique_words = sorted(list(set(words)))
    vocab = special_tokens + unique_words
    token_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_token = {i: word for i, word in enumerate(vocab)}
    embedding_matrix = None
    if word_vecs and hasattr(word_vecs, 'vector_size'):
        embedding_matrix = np.zeros((len(vocab), word_vecs.vector_size))
        num_found = 0
        unk_id = token_to_id.get("<unk>")
        for word, idx in token_to_id.items():
            if word in word_vecs:
                embedding_matrix[idx] = word_vecs[word]
                num_found += 1
            elif word in special_tokens:
                 pass
            elif unk_id is not None and idx == unk_id:
                 pass
        print(f"Initialized {num_found}/{len(vocab)} word embeddings using GloVe (excluding special/unknown).")

    return {
        'type': 'word',
        'vocab_size': len(vocab),
        'token_to_id': token_to_id,
        'id_to_token': id_to_token,
        'embedding_matrix': torch.from_numpy(embedding_matrix).float() if embedding_matrix is not None else None
    }

def bpe_tokenize(lines: List[str], cfg: Config, special_tokens: List[str]) -> Dict[str, Any]:
    text = "\n".join(lines)
    tokenizer = ByteLevelBPETokenizer()
    with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf-8') as tmp:
        tmp.write(text)
        tmp.flush()
        tokenizer.train(files=[tmp.name], vocab_size=cfg.bpe_vocab_size, special_tokens=special_tokens)
    os.unlink(tmp.name)
    vocab = tokenizer.get_vocab()
    return {
        'type': 'bpe',
        'tokenizer': tokenizer,
        'vocab_size': tokenizer.get_vocab_size(),
        'token_to_id': vocab,
        'id_to_token': {i: token for token, i in vocab.items()}
    }

# --- Dataset with Augmentation ---
class TextGenerationDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len: int, tokenizer_info: Dict[str, Any], augment_prob: float = 0.0):
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.tokenizer_info = tokenizer_info
        self.augment_prob = augment_prob
        self.augmenter = self._build_augmenter()
        self.pad_id = self._get_pad_id()

    def _get_pad_id(self):
         if self.tokenizer_info['type'] == 'bpe':
             return self.tokenizer_info['tokenizer'].token_to_id("<pad>")
         elif self.tokenizer_info['type'] in ['char', 'word']:
             return self.tokenizer_info['token_to_id'].get("<pad>")
         return None

    def _build_augmenter(self):
        if not NLP_AUG_AVAILABLE or self.augment_prob <= 0:
            return None

        tokenizer_type = self.tokenizer_info.get('type', '')
        if tokenizer_type in ['bpe', 'word']:
             return naw.RandomWordAug(
                 action=random.choice(["substitute", "delete", "insert"]),
                 aug_p=self.augment_prob
             )
        elif tokenizer_type == 'char':
             return nac.RandomCharAug(
                 action=random.choice(["swap", "delete"]),
                 aug_char_p=self.augment_prob
             )
        return None

    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_len)

    def __getitem__(self, idx):
        original_x_ids = self.token_ids[idx: idx + self.seq_len]
        y_ids = self.token_ids[idx + 1: idx + self.seq_len + 1]

        x = torch.tensor(original_x_ids, dtype=torch.long)
        y = torch.tensor(y_ids, dtype=torch.long)

        if self.augmenter and random.random() < self.augment_prob:
             try:
                 text_seq = self._ids_to_text(x.tolist())
                 augmented_text = self.augmenter.augment(text_seq)
                 if isinstance(augmented_text, list):
                     augmented_text = augmented_text[0]

                 if augmented_text:
                    augmented_x_ids = self._text_to_ids(augmented_text)

                    if len(augmented_x_ids) != self.seq_len:
                         if self.pad_id is not None:
                             if len(augmented_x_ids) < self.seq_len:
                                 padding = [self.pad_id] * (self.seq_len - len(augmented_x_ids))
                                 augmented_x_ids.extend(padding)
                             else:
                                 augmented_x_ids = augmented_x_ids[:self.seq_len]
                         else:
                             augmented_x_ids = original_x_ids

                    if len(augmented_x_ids) == self.seq_len:
                         x = torch.tensor(augmented_x_ids, dtype=torch.long)
                    else:
                         pass
             except Exception as e:
                 pass
        return x, y

    def _ids_to_text(self, ids: List[int]) -> str:
        # Corrected: Use self.tokenizer_info directly
        tokenizer_info = self.tokenizer_info
        if not tokenizer_info:
            return "Error: Tokenizer info not available for decoding."

        tokenizer_type = tokenizer_info.get('type')
        id_to_token = tokenizer_info.get('id_to_token')

        if tokenizer_type == 'bpe':
             hf_tokenizer = tokenizer_info.get('tokenizer')
             if hf_tokenizer and hasattr(hf_tokenizer, 'decode'):
                 return hf_tokenizer.decode(ids, skip_special_tokens=False)
        elif tokenizer_type in ['char', 'word'] and id_to_token:
             default_token_str = '<unk>'
             tokens = [id_to_token.get(id_val, default_token_str) for id_val in ids]
             if tokenizer_type == 'char':
                 return "".join(tokens)
             else:
                 return " ".join(tokens)
        return "Error: Could not decode tokens due to missing info."

    def _text_to_ids(self, text: str) -> List[int]:
        # Corrected: Use self.tokenizer_info directly
        tokenizer_info = self.tokenizer_info
        if not tokenizer_info:
             print("Warning: Tokenizer info not available for encoding text to ids.")
             return []

        tokenizer_type = tokenizer_info.get('type')
        token_to_id = tokenizer_info.get('token_to_id')
        unk_id = token_to_id.get("<unk>", 0) if token_to_id else 0

        if tokenizer_type == 'bpe':
            hf_tokenizer = tokenizer_info.get('tokenizer')
            if hasattr(hf_tokenizer, 'encode'):
                return hf_tokenizer.encode(text).ids
        elif tokenizer_type == 'char' and token_to_id:
            return [token_to_id.get(c, unk_id) for c in text]
        elif tokenizer_type == 'word' and token_to_id:
            words = text.split()
            return [token_to_id.get(word, unk_id) for word in words]
        return []


# --- Transformer Model (Decoder-Only) ---
class DecoderOnlyTransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, max_len: int, dropout: float = 0.1, padding_idx: int | None = None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=hidden_dim, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        mask = mask.masked_fill(mask, float('-inf'))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)

        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        output = self.transformer(x, x, tgt_mask=mask, memory_mask=mask)

        return self.fc(output)


# --- Main Controller ---
class ControlledTextGenerator:
    ROLE_PATTERN = re.compile(r"^([A-Z][A-Za-z ]+):")
    EMOTION_TAGS = ["<EMO_HAPPY>", "<EMO_SAD>", "<EMO_NEUTRAL>"]
    SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"] + EMOTION_TAGS

    def __init__(self, config: Config, device: torch.device):
        self.cfg = config
        self.device = device
        self.loss_history: Dict[str, List[float]] = {}
        self.perplexity_history: Dict[str, List[float]] = {}
        self.bleu_history: Dict[str, List[float]] = {}
        self.distinct_history: Dict[int, Dict[str, List[float]]] = {2: {}, 3: {}}
        self.avg_len_history: Dict[str, List[float]] = {}

        self.tokenizers_info: Dict[str, Dict[str, Any]] = {}
        self.full_corpus_lines: List[str] = []
        self.scaler = GradScaler()

        random.seed(self.cfg.random_seed)
        np.random.seed(self.cfg.random_seed)
        torch.manual_seed(self.cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.random_seed)

    def preprocess_lines(self, lines: List[str]) -> List[str]:
        processed = []
        for line in lines:
            line = line.strip()
            if not line: continue
            m = self.ROLE_PATTERN.match(line)
            if m:
                role = m.group(1).replace(' ', '_').upper()
                emotion_tag = random.choice(self.EMOTION_TAGS)
                processed.append(f"<ROLE_{role}> {emotion_tag} " + line[len(m.group(0)):].strip())
            else:
                emotion_tag = random.choice([token for token in self.EMOTION_TAGS if token != "<EMO_HAPPY>"])
                processed.append(f"{emotion_tag} " + line.strip())
        return processed

    def prepare_tokenizers(self, lines: List[str]):
        print("Preparing tokenizers (BPE, Character, Simple Word)...")
        processed_lines = self.preprocess_lines(lines)
        all_text = "\n".join(processed_lines)

        # BPE Tokenizer
        try:
            tmp_corpus_path = "tmp_corpus_bpe.txt"
            with open(tmp_corpus_path, "w", encoding="utf-8") as f:
                f.write(all_text)
            bpe_tokenizer = ByteLevelBPETokenizer()
            bpe_tokenizer.train(files=[tmp_corpus_path], vocab_size=self.cfg.bpe_vocab_size,
                                min_frequency=2, special_tokens=self.SPECIAL_TOKENS)
            os.remove(tmp_corpus_path)
            self.tokenizers_info["bpe"] = {
                'type': 'bpe',
                'tokenizer': bpe_tokenizer,
                'vocab_size': bpe_tokenizer.get_vocab_size(),
                'token_to_id': bpe_tokenizer.get_vocab(),
                'id_to_token': {i: token for token, i in bpe_tokenizer.get_vocab().items()}
            }
            print(f"BPE Vocab Size: {self.tokenizers_info['bpe']['vocab_size']}")
        except Exception as e:
            print(f"Error preparing BPE tokenizer: {e}. Skipping BPE.")
            self.tokenizers_info["bpe"] = None

        # Character Tokenizer
        try:
            self.tokenizers_info["char"] = char_tokenize(processed_lines, self.SPECIAL_TOKENS)
            print(f"Character Vocab Size: {self.tokenizers_info['char']['vocab_size']}")
        except Exception as e:
            print(f"Error preparing Character tokenizer: {e}. Skipping Character.")
            self.tokenizers_info["char"] = None

        # Simple Word Tokenizer
        try:
            glove_word_vecs = None
            if self.cfg.glove_path and os.path.exists(self.cfg.glove_path):
                 try:
                     from gensim.models import KeyedVectors
                     glove_word_vecs = KeyedVectors.load_word2vec_format(self.cfg.glove_path, no_header=True)
                     print(f"Loaded GloVe vectors with dimension: {glove_word_vecs.vector_size}")
                     if glove_word_vecs.vector_size != self.cfg.embed_dim:
                         print(f"Warning: GloVe vector size ({glove_word_vecs.vector_size}) mismatches model embed_dim ({self.cfg.embed_dim}). Embeddings will not be used.")
                         glove_word_vecs = None
                 except ImportError:
                      print("Warning: gensim not installed. Cannot load GloVe. Install with: pip install gensim")
                      glove_word_vecs = None
                 except Exception as e:
                     print(f"Error loading GloVe file {self.cfg.glove_path}: {e}. Skipping GloVe.")
                     glove_word_vecs = None

            self.tokenizers_info["word"] = word_tokenize(processed_lines, self.SPECIAL_TOKENS, glove_word_vecs)
            print(f"Word Vocab Size: {self.tokenizers_info['word']['vocab_size']}")

        except Exception as e:
             print(f"Error preparing Simple Word tokenizer: {e}. Skipping Word.")
             self.tokenizers_info["word"] = None

    def text_to_token_ids(self, lines: List[str], tokenizer_info: Dict[str, Any]) -> List[int]:
        processed_lines = self.preprocess_lines(lines)
        all_text = "\n".join(processed_lines)

        tokenizer_type = tokenizer_info.get('type')
        token_to_id = tokenizer_info.get('token_to_id')
        unk_id = token_to_id.get("<unk>", 0) if token_to_id else 0

        if tokenizer_type == 'bpe':
            hf_tokenizer = tokenizer_info.get('tokenizer')
            if hf_tokenizer and hasattr(hf_tokenizer, 'encode'):
                return hf_tokenizer.encode(all_text).ids
        elif tokenizer_type == 'char' and token_to_id:
            return [token_to_id.get(c, unk_id) for c in all_text]
        elif tokenizer_type == 'word' and token_to_id:
            words = all_text.split()
            return [token_to_id.get(word, unk_id) for word in words]
        raise ValueError(f"Invalid or unsupported tokenizer type in text_to_token_ids: {tokenizer_type}")

    def build_loader(self, token_ids: List[int], tokenizer_info: Dict[str, Any], shuffle: bool = True) -> DataLoader | None:
        if not token_ids:
            print("Warning: Cannot build loader from empty token_ids list.")
            return None

        dataset = TextGenerationDataset(
            token_ids=token_ids,
            seq_len=self.cfg.seq_len,
            tokenizer_info=tokenizer_info,
            augment_prob=self.cfg.augment_prob
        )

        if len(dataset) == 0:
            print(f"Warning: Empty dataset for sequence length {self.cfg.seq_len} after processing.")
            return None

        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            persistent_workers=True if self.device.type == "cuda" and self.cfg.num_workers > 0 else False
        )

    def calculate_perplexity(self, model: nn.Module, data_loader: DataLoader, tokenizer_info: Dict[str, Any]) -> float:
        model.eval()
        total_loss = 0.0
        total_valid_tokens = 0
        pad_id = self._get_pad_id_from_info(tokenizer_info)

        criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id if pad_id is not None else -1,
            reduction='sum'
        )

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

                if pad_id is not None:
                    valid_tokens_mask = (y != pad_id)
                else:
                    valid_tokens_mask = torch.ones_like(y, dtype=torch.bool)
                total_valid_tokens += valid_tokens_mask.sum().item()
                total_loss += loss.item()

        if total_valid_tokens == 0:
            return float('inf')
        avg_loss = total_loss / total_valid_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity

    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, model_type: str, tokenizer_info: Dict[str, Any]):
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        pad_id = self._get_pad_id_from_info(tokenizer_info)
        # 添加标签平滑
        criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id if pad_id is not None else -1,
            label_smoothing=self.cfg.label_smoothing
        )
        
        # 添加学习率调度器
        scheduler = None
        if self.cfg.use_scheduler:
            total_steps = len(train_loader) * self.cfg.epochs
            if self.cfg.scheduler_type == "cosine":
                scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=self.cfg.lr * 0.01)
            elif self.cfg.scheduler_type == "onecycle":
                scheduler = OneCycleLR(optimizer, max_lr=self.cfg.lr, 
                                       steps_per_epoch=len(train_loader), epochs=self.cfg.epochs)
            print(f"Using {self.cfg.scheduler_type} learning rate scheduler")

        if model_type not in self.loss_history: self.loss_history[model_type] = []
        if model_type not in self.perplexity_history: self.perplexity_history[model_type] = []

        # 早停机制
        best_val_perplexity = float('inf')
        patience_counter = 0
        best_model_state = None

        print(f"Starting training for {model_type} model...")
        for epoch in range(self.cfg.epochs):
            total_loss = 0
            model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                with autocast(device_type=self.device.type, dtype=torch.float16):
                    out = model(x)
                    loss = criterion(out.view(-1, out.size(-1)), y.view(-1))

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.cfg.grad_norm_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
                total_loss += loss.item()
                
                # 每步更新学习率（对于 OneCycleLR）
                if scheduler and self.cfg.scheduler_type == "onecycle":
                    scheduler.step()
            
            # 每轮结束更新学习率（对于 CosineAnnealingLR）
            if scheduler and self.cfg.scheduler_type == "cosine":
                scheduler.step()

            avg_train_loss = total_loss / len(train_loader)
            val_perplexity = self.calculate_perplexity(model, val_loader, tokenizer_info)

            self.loss_history[model_type].append(avg_train_loss)
            self.perplexity_history[model_type].append(val_perplexity)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{self.cfg.epochs} - {model_type} Train Loss: {avg_train_loss:.4f} - Val Perplexity: {val_perplexity:.4f} - LR: {current_lr:.2e}")
            
            # 早停检查
            if val_perplexity < best_val_perplexity:
                best_val_perplexity = val_perplexity
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}. Best Val Perplexity: {best_val_perplexity:.4f}")
                    if best_model_state:
                        model.load_state_dict(best_model_state)
                    break
        
        # 恢复最佳模型
        if best_model_state and val_perplexity > best_val_perplexity:
            print(f"Restoring best model with Val Perplexity: {best_val_perplexity:.4f}")
            model.load_state_dict(best_model_state)

    def top_p_sampling(self, logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0, 
                       bad_words_ids: List[int] | None = None, 
                       repetition_penalty: float = 1.0,
                       past_tokens: List[int] | None = None) -> int:
        """Samples the next token using Top-p filtering, temperature scaling, and repetition penalty."""
        if logits.dim() > 1:
            logits = logits.squeeze(0)
        
        # 应用重复惩罚 - 对已生成的 token 降低概率
        if past_tokens is not None and repetition_penalty != 1.0 and len(past_tokens) > 0:
            # 取最近 50 个 token 进行惩罚
            recent_tokens = set(past_tokens[-50:])
            for token_id in recent_tokens:
                if 0 <= token_id < logits.size(-1):
                    # 如果 logit > 0，除以惩罚；如果 < 0，乘以惩罚
                    if logits[token_id] > 0:
                        logits[token_id] = logits[token_id] / repetition_penalty
                    else:
                        logits[token_id] = logits[token_id] * repetition_penalty

        if temperature != 1.0:
             logits = torch.div(logits, temperature)

        probs = torch.softmax(logits, dim=-1)

        if bad_words_ids is not None:
            if not isinstance(bad_words_ids, torch.Tensor):
                 bad_words_ids = torch.tensor(bad_words_ids, device=probs.device, dtype=torch.long)
            valid_bad_words = bad_words_ids[(bad_words_ids >= 0) & (bad_words_ids < probs.size(-1))]
            if valid_bad_words.numel() > 0:
                probs.scatter_(0, valid_bad_words, 0.0)
                sum_after_filter = probs.sum(dim=-1, keepdim=True)
                if sum_after_filter > 1e-9:
                     probs = probs / sum_after_filter
                else:
                    tokenizer_info = self.tokenizers_info.get(getattr(self, 'last_sampled_model_type', 'bpe'))
                    unk_id = self._get_unk_id_from_info(tokenizer_info)
                    print(f"Warning: All probabilities zero after bad word filtering. Returning UNK token ID: {unk_id if unk_id is not None else 0}")
                    return unk_id if unk_id is not None else 0


        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        cutoff_index = (cumulative_probs > top_p).nonzero(as_tuple=True)[0]

        if cutoff_index.shape[0] == 0:
             num_tokens_to_keep = sorted_probs.size(0)
        else:
             num_tokens_to_keep = max(1, cutoff_index[0].item() + 1)

        filtered_sorted_probs = sorted_probs[:num_tokens_to_keep]
        filtered_sorted_indices = sorted_indices[:num_tokens_to_keep]

        sum_remaining_probs = filtered_sorted_probs.sum()

        if sum_remaining_probs < 1e-9:
             tokenizer_info = self.tokenizers_info.get(getattr(self, 'last_sampled_model_type', 'bpe'))
             unk_id = self._get_unk_id_from_info(tokenizer_info)
             print(f"Warning: Sum of remaining probabilities zero after Top-p filtering. Returning UNK token ID: {unk_id if unk_id is not None else 0}")
             return unk_id if unk_id is not None else 0

        probs = filtered_sorted_probs / (sum_remaining_probs + 1e-9)

        if not torch.isfinite(probs).all():
             tokenizer_info = self.tokenizers_info.get(getattr(self, 'last_sampled_model_type', 'bpe'))
             unk_id = self._get_unk_id_from_info(tokenizer_info)
             print(f"Warning: Probabilities contain non-finite values after re-normalization: {probs}. Returning UNK token ID.")
             return unk_id if unk_id is not None else 0

        next_token_index_in_filtered_list = torch.multinomial(probs, 1)
        next_token_id = filtered_sorted_indices[next_token_index_in_filtered_list].item()

        return next_token_id

    def sample(self, model: nn.Module, prefix: str, max_len: int = 50, top_p: float = 0.9, temperature: float = 1.0, model_type: str = "bpe") -> str:
        model.eval()
        tokenizer_info = self.tokenizers_info.get(model_type)
        self.last_sampled_model_type = model_type

        if tokenizer_info is None:
            print(f"Error: Tokenizer info not found for model type: {model_type}. Cannot sample.")
            return "Error: Could not find tokenizer."

        try:
            token_to_id_map = tokenizer_info.get('token_to_id')
            id_to_token_map = tokenizer_info.get('id_to_token')
            unk_id = self._get_unk_id_from_info(tokenizer_info)
            pad_id = self._get_pad_id_from_info(tokenizer_info)
            eos_id = token_to_id_map.get("<eos>") if token_to_id_map else None

            if tokenizer_info.get('type') == 'bpe' and hasattr(tokenizer_info.get('tokenizer'), 'encode'):
                prefix_ids = tokenizer_info['tokenizer'].encode(prefix).ids
            elif tokenizer_info.get('type') == 'char' and token_to_id_map:
                prefix_ids = [token_to_id_map.get(c, unk_id) for c in prefix]
            elif tokenizer_info.get('type') == 'word' and token_to_id_map:
                prefix_words = prefix.split()
                prefix_ids = [token_to_id_map.get(word, unk_id) for word in prefix_words]
            else:
                 raise ValueError("Unsupported tokenizer type or missing maps during prefix tokenization.")

            if not prefix_ids:
                print(f"Warning: Prefix '{prefix}' resulted in empty token IDs for {model_type}.")
                sos_id = token_to_id_map.get("<sos>") if token_to_id_map else unk_id
                prefix_ids = [sos_id] if sos_id is not None else [0]

            if len(prefix_ids) > self.cfg.seq_len:
                print(f"Warning: Prefix length ({len(prefix_ids)}) exceeds seq_len ({self.cfg.seq_len}). Truncating prefix.")
                prefix_ids = prefix_ids[-self.cfg.seq_len:]

            output_ids = list(prefix_ids)
            original_prefix_len = len(output_ids)

            bad_words_ids_list = [unk_id] if unk_id is not None else []
            if pad_id is not None and (unk_id is None or pad_id != unk_id):
                bad_words_ids_list.append(pad_id)
            sos_id = token_to_id_map.get("<sos>") if token_to_id_map else None
            if sos_id is not None and (unk_id is None or sos_id != unk_id) and (pad_id is None or sos_id != pad_id):
                 bad_words_ids_list.append(sos_id)


            with torch.no_grad():
                for _ in range(max_len):
                    current_sequence_ids = output_ids[-min(len(output_ids), self.cfg.seq_len):]

                    if len(current_sequence_ids) < self.cfg.seq_len:
                        if pad_id is not None:
                            padding = [pad_id] * (self.cfg.seq_len - len(current_sequence_ids))
                            current_sequence_ids = padding + current_sequence_ids
                        else:
                            pass

                    if not current_sequence_ids:
                         break

                    input_tensor = torch.tensor([current_sequence_ids], device=self.device, dtype=torch.long)

                    logits = model(input_tensor)[:, -1, :]

                    next_id = self.top_p_sampling(
                        logits.squeeze(0),
                        top_p=top_p,
                        temperature=temperature,
                        bad_words_ids=bad_words_ids_list if bad_words_ids_list else None,
                        repetition_penalty=self.cfg.repetition_penalty,
                        past_tokens=output_ids[original_prefix_len:]  # 只传递已生成的 token
                    )

                    output_ids.append(next_id)

                    if eos_id is not None and next_id == eos_id:
                        break

            generated_ids = output_ids[original_prefix_len:]
            decoded_text = self._ids_to_text(generated_ids)

            return decoded_text

        except Exception as e:
            print(f"Error during sampling for {model_type} with prefix '{prefix[:50]}...': {e}")
            import traceback
            traceback.print_exc()
            return f"Error during sampling: {e}"

    def _ids_to_text(self, ids: List[int]) -> str:
        tokenizer_info = self.tokenizers_info.get(getattr(self, 'last_sampled_model_type', 'bpe'))
        if not tokenizer_info:
            return "Error: Tokenizer info not available for decoding."

        tokenizer_type = tokenizer_info.get('type')
        id_to_token_map = tokenizer_info.get('id_to_token')

        if tokenizer_type == 'bpe':
             hf_tokenizer = tokenizer_info.get('tokenizer')
             if hf_tokenizer and hasattr(hf_tokenizer, 'decode'):
                 return hf_tokenizer.decode(ids, skip_special_tokens=False)
        elif tokenizer_type in ['char', 'word'] and id_to_token_map:
             default_token_str = '<unk>'
             tokens = [id_to_token_map.get(id_val, default_token_str) for id_val in ids]
             if tokenizer_type == 'char':
                 return "".join(tokens)
             else:
                 return " ".join(tokens)
        return "Error: Could not decode tokens due to missing info."

    def _text_to_token_list(self, text: str, model_type: str) -> List[Any]:
        """Converts text to a list of tokens (str or id depending on need) for metrics."""
        tokenizer_info = self.tokenizers_info.get(model_type)
        if not tokenizer_info:
             print(f"Warning: Tokenizer info not found for {model_type}. Cannot convert text to token list.")
             return []

        tokenizer_type = tokenizer_info.get('type')

        if tokenizer_type == 'bpe':
             hf_tokenizer = tokenizer_info.get('tokenizer')
             if hf_tokenizer and hasattr(hf_tokenizer, 'encode'):
                  ids = hf_tokenizer.encode(text).ids
                  return [hf_tokenizer.decode([id], skip_special_tokens=False) for id in ids]
        elif tokenizer_type == 'char':
             return list(text)
        elif tokenizer_type == 'word':
             return text.split()
        return []

    def _get_pad_id_from_info(self, tokenizer_info: Dict[str, Any] | None) -> int | None:
        """Helper to get pad token ID from tokenizer info dict/object."""
        if tokenizer_info:
            if tokenizer_info.get('type') == 'bpe' and hasattr(tokenizer_info.get('tokenizer'), 'token_to_id'):
                return tokenizer_info['tokenizer'].token_to_id("<pad>")
            elif tokenizer_info.get('type') in ['char', 'word'] and 'token_to_id' in tokenizer_info:
                return tokenizer_info['token_to_id'].get("<pad>")
        return None

    def _get_unk_id_from_info(self, tokenizer_info: Dict[str, Any] | None) -> int | None:
         """Helper to get unk token ID from tokenizer info dict/object."""
         if tokenizer_info:
              if tokenizer_info.get('type') == 'bpe' and hasattr(tokenizer_info.get('tokenizer'), 'token_to_id'):
                   return tokenizer_info['tokenizer'].token_to_id("<unk>")
              elif tokenizer_info.get('type') in ['char', 'word'] and 'token_to_id' in tokenizer_info:
                   return tokenizer_info['token_to_id'].get("<unk>")
         return None

    def perform_comprehensive_evaluation(self, model: nn.Module, model_type: str, val_loader: DataLoader):
        """Performs comprehensive evaluation including perplexity, BLEU, Distinct-N, and Avg Length."""
        print(f"\n--- Performing comprehensive evaluation for {model_type.upper()} model ---")
        model.eval()
        results: Dict[str, Any] = {}
        tokenizer_info = self.tokenizers_info.get(model_type)

        if tokenizer_info is None:
            print(f"Skipping evaluation for {model_type}: Tokenizer info not found.")
            return results

        if model_type in self.perplexity_history and self.perplexity_history[model_type]:
            results['val_perplexity'] = self.perplexity_history[model_type][-1]
            print(f" Last Val Perplexity: {results['val_perplexity']:.4f}")
        else:
            print(f" Val Perplexity not available for {model_type}.")
            results['val_perplexity'] = float('nan')


        num_samples = self.cfg.num_evaluation_samples
        evaluation_sample_max_len = self.cfg.evaluation_sample_max_len
        print(f" Generating {num_samples} samples (max len {evaluation_sample_max_len})...")
        generated_samples = [self.sample(model, "<sos>", max_len=evaluation_sample_max_len,
                                          model_type=model_type, temperature=1.0, top_p=0.9)
                               for _ in range(num_samples)]

        tokenized_generated_samples = [self._text_to_token_list(s, model_type) for s in generated_samples]

        if tokenized_generated_samples and any(len(seq) > 0 for seq in tokenized_generated_samples):
             print(" Calculating Distinct-N scores...")
             dist2 = compute_distinct_n(tokenized_generated_samples, n=2)
             dist3 = compute_distinct_n(tokenized_generated_samples, n=3)
             results['distinct_2'] = dist2
             results['distinct_3'] = dist3
             if model_type not in self.distinct_history[2]: self.distinct_history[2][model_type] = []
             if model_type not in self.distinct_history[3]: self.distinct_history[3][model_type] = []
             self.distinct_history[2][model_type].append(dist2)
             self.distinct_history[3][model_type].append(dist3)
             print(f" Distinct-2: {dist2:.4f}, Distinct-3: {dist3:.4f}")
        else:
             print(" Skipping Distinct-N calculation: No valid tokenized generated samples.")
             results['distinct_2'] = 0.0
             results['distinct_3'] = 0.0
        if tokenized_generated_samples and any(len(seq) > 0 for seq in tokenized_generated_samples):
             print(" Calculating Average Length...")
             avg_len = compute_average_length(tokenized_generated_samples)
             results['average_length'] = avg_len
             if model_type not in self.avg_len_history: self.avg_len_history[model_type] = []
             self.avg_len_history[model_type].append(avg_len)
             print(f" Average Length: {avg_len:.2f}")
        else:
             print(" Skipping Average Length calculation: No valid tokenized generated samples.")
             results['average_length'] = 0.0

        if self.full_corpus_lines and tokenized_generated_samples and any(len(seq) > 0 for seq in tokenized_generated_samples):
             try:
                 print(" Calculating BLEU scores against full corpus...")
                 tokenized_corpus_lines = []
                 for line in self.full_corpus_lines:
                      processed_line = self.preprocess_lines([line])[0]
                      tokenized_line = self._text_to_token_list(processed_line, model_type)
                      if tokenized_line:
                           tokenized_corpus_lines.append(tokenized_line)

                 if tokenized_corpus_lines:
                      bleu_scores = []
                      smooth = SmoothingFunction().method1
                      for sample_tokens in tokenized_generated_samples:
                           if len(sample_tokens) > 0:
                                bleu = sentence_bleu(tokenized_corpus_lines, sample_tokens, smoothing_function=smooth)
                                bleu_scores.append(bleu)
                           else:
                                bleu_scores.append(0.0)

                      avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
                      results['bleu'] = avg_bleu
                      if model_type not in self.bleu_history: self.bleu_history[model_type] = []
                      self.bleu_history[model_type].append(avg_bleu)
                      print(f" Avg BLEU: {avg_bleu:.4f}")
                 else:
                      print(f"Warning: No valid reference sentences found in corpus for BLEU calculation for {model_type}.")
                      results['bleu'] = 0.0

             except Exception as e:
                 print(f"Error calculating BLEU score for {model_type}: {e}")
                 results['bleu'] = 0.0
                 import traceback
                 traceback.print_exc()
        else:
             print(" Skipping BLEU calculation: No corpus lines or no valid tokenized generated samples.")
             results['bleu'] = 0.0

        print(f"--- Evaluation complete for {model_type.upper()} ---")
        return results

    def analyze_temperature_scaling(self, model: nn.Module, model_type: str, prefix: str = "<sos>"):
        """Analyzes the effect of temperature scaling on generated text diversity (Distinct-N, Avg Length)."""
        print(f"\n--- Analyzing Temperature Scaling for {model_type.upper()} Model ---")
        temperatures = self.cfg.temperatures
        num_samples_per_temp = self.cfg.num_temp_samples
        evaluation_sample_max_len = self.cfg.evaluation_sample_max_len

        tokenizer_info = self.tokenizers_info.get(model_type)
        if not tokenizer_info:
            print(f"Skipping temperature analysis for {model_type}: Tokenizer info not available.")
            return

        temp_results: Dict[float, Dict[str, float]] = {}

        for temp in temperatures:
            print(f" Sampling with Temperature: {temp}")
            generated_samples = [
                self.sample(model, prefix, max_len=evaluation_sample_max_len, temperature=temp, top_p=0.9,
                            model_type=model_type)
                for _ in range(num_samples_per_temp)]

            tokenized_samples = [self._text_to_token_list(s, model_type) for s in generated_samples]

            current_temp_metrics: Dict[str, float] = {}
            if tokenized_samples and any(len(seq) > 0 for seq in tokenized_samples):
                dist2 = compute_distinct_n(tokenized_samples, n=2)
                dist3 = compute_distinct_n(tokenized_samples, n=3)
                avg_len = compute_average_length(tokenized_samples)
                current_temp_metrics['distinct_2'] = dist2
                current_temp_metrics['distinct_3'] = dist3
                current_temp_metrics['average_length'] = avg_len
                print(f"  Distinct-2: {dist2:.4f}, Distinct-3: {dist3:.4f}, Avg Length: {avg_len:.2f}")
            else:
                 print("  No valid samples generated for metrics calculation.")
                 current_temp_metrics['distinct_2'] = 0.0
                 current_temp_metrics['distinct_3'] = 0.0
                 current_temp_metrics['average_length'] = 0.0

            temp_results[temp] = current_temp_metrics

        self.plot_temperature_analysis(temp_results, model_type)

        print("--- Temperature Scaling Analysis Complete ---")

    def plot_comparison_curves(self):
        """Plots training and validation metrics comparison across different model types."""
        metrics_to_plot = {
             'loss': 'Training Loss per Epoch Comparison',
             'perplexity': 'Validation Perplexity per Epoch Comparison',
        }
        ylabels = {
             'loss': 'Loss',
             'perplexity': 'Perplexity',
        }

        plt.switch_backend('Agg')

        for metric, title in metrics_to_plot.items():
            history = getattr(self, f"{metric}_history", None)

            if history and any(values for values in history.values()):
                plt.figure(figsize=(10, 6))
                for model_type, values in history.items():
                    if values:
                        plt.plot(values, label=f"{model_type.upper()}")

                plt.title(title)
                plt.xlabel("Epoch")
                plt.ylabel(ylabels.get(metric, metric))
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                filename = f"{self.cfg.plot_output_prefix}_{metric}.png"
                plt.savefig(filename)
                print(f"[Saved Plot] {filename}")
                plt.close()
            else:
                print(f"No history to plot for metric: {metric}")

    def plot_temperature_analysis(self, temp_results: Dict[float, Dict[str, float]], model_type: str):
        """Plots Distinct-N and Average Length vs. Temperature."""
        if not temp_results:
             print(f"No temperature results to plot for {model_type}.")
             return

        temps = sorted(temp_results.keys())
        distinct2_values = [temp_results[t].get('distinct_2', 0.0) for t in temps]
        distinct3_values = [temp_results[t].get('distinct_3', 0.0) for t in temps]
        avg_len_values = [temp_results[t].get('average_length', 0.0) for t in temps]

        plt.switch_backend('Agg')

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].plot(temps, distinct2_values, marker='o', label='Distinct-2')
        axes[0].plot(temps, distinct3_values, marker='x', label='Distinct-3')
        axes[0].set_title(f'{model_type.upper()} Distinct-N vs. Temperature')
        axes[0].set_xlabel('Temperature')
        axes[0].set_ylabel('Distinct-N Score')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(temps, avg_len_values, marker='o', color='green', label='Average Length')
        axes[1].set_title(f'{model_type.upper()} Average Length vs. Temperature')
        axes[1].set_xlabel('Temperature')
        axes[1].set_ylabel('Average Length')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        filename = f"{self.cfg.plot_output_prefix}_{model_type}_temperature_analysis.png"
        plt.savefig(filename)
        print(f"[Saved Plot] {filename}")
        plt.close(fig)

    def save_sample_outputs(self, samples: Dict[str, List[str]]):
        """Saves generated samples to a text file."""
        print(f"Saving generated samples to {self.cfg.sample_output_file}...")
        with open(self.cfg.sample_output_file, "w", encoding="utf-8") as f:
            for model_type, sample_list in samples.items():
                f.write(f"--- Samples for {model_type.upper()} Model ---\n\n")
                for i, sample_text in enumerate(sample_list):
                    f.write(f"Sample {i+1}:\n")
                    f.write(sample_text)
                    f.write("\n\n")
                f.write("-" * 30 + "\n\n")
        print("Sample outputs saved.")

    def log_evaluation_results(self, final_results: Dict[str, Dict[str, Any]]):
         """Logs final evaluation results (BLEU, Distinct-N, Avg Length) to a file."""
         print(f"Logging final evaluation results to {self.cfg.log_file}...")
         with open(self.cfg.log_file, "a", encoding="utf-8") as f:
              f.write(f"--- Final Evaluation Results ({os.path.basename(__file__)}) ---\n\n")
              for model_type, results in final_results.items():
                   f.write(f"--- {model_type.upper()} Results ---\n")
                   f.write(f"  Validation Perplexity: {results.get('val_perplexity', float('nan')):.4f}\n")
                   f.write(f"  Average BLEU Score: {results.get('bleu', 0.0):.4f}\n")
                   f.write(f"  Distinct-2 Score: {results.get('distinct_2', 0.0):.4f}\n")
                   f.write(f"  Distinct-3 Score: {results.get('distinct_3', 0.0):.4f}\n")
                   f.write(f"  Average Generated Length: {results.get('average_length', 0.0):.2f}\n")
                   f.write("-" * 20 + "\n")
              f.write("=" * 40 + "\n\n")
         print("Evaluation results logged.")

    def run(self, paths: List[str]):
        """Runs the comparative experiments for different tokenization methods and models."""
        print("Loading and preprocessing data...")
        lines = []
        for p in paths:
            try:
                with open(p, encoding="utf-8") as f:
                    lines.extend(f.read().splitlines())
            except FileNotFoundError:
                print(f"Error: File not found at {p}. Skipping.")
            except Exception as e:
                print(f"Error reading file {p}: {e}. Skipping.")

        if not lines:
            print("Error: No data loaded from the provided file paths. Exiting.")
            return

        self.full_corpus_lines = lines
        self.prepare_tokenizers(self.full_corpus_lines)
        tokenization_methods_to_run = [m for m, info in self.tokenizers_info.items() if info is not None]

        if not tokenization_methods_to_run:
            print("No tokenizers were successfully initialized. Exiting.")
            return

        print("\n--- Running experiments for initialized tokenizers ---")
        final_evaluation_results: Dict[str, Dict[str, Any]] = {}
        all_generated_samples: Dict[str, List[str]] = {}

        for method in tokenization_methods_to_run:
            print(f"\n--- Running experiment for {method.upper()} ---")

            tokenizer_info = self.tokenizers_info[method]
            if tokenizer_info is None:
                 print(f"Skipping {method}: Tokenizer info is None.")
                 continue

            print(f"Converting text to token IDs for {method}...")
            try:
                all_corpus_token_ids = self.text_to_token_ids(self.full_corpus_lines, tokenizer_info)
            except ValueError as e:
                print(f"Error converting text to token IDs for {method}: {e}. Skipping experiment.")
                continue

            if not all_corpus_token_ids:
                print(f"Skipping {method}: Failed to generate token IDs from corpus.")
                continue

            vocab_size = tokenizer_info.get('vocab_size')
            if vocab_size is None:
                print(f"Skipping {method}: Vocab size not found in tokenizer info.")
                continue

            min_data_needed_per_split = self.cfg.seq_len + 1
            if len(all_corpus_token_ids) < min_data_needed_per_split * 2:
                 print(f"Skipping {method}: Total token IDs ({len(all_corpus_token_ids)}) is too short to guarantee train/val splits of at least {min_data_needed_per_split} tokens each. Need at least {min_data_needed_per_split * 2}. Adjust seq_len or provide more data.")
                 continue

            try:
                train_token_ids_raw, val_token_ids_raw = train_test_split(
                    all_corpus_token_ids,
                    test_size=self.cfg.val_split_size,
                    random_state=self.cfg.random_seed
                )
            except ValueError as e:
                print(f"Error splitting data for {method}: {e}. Skipping.")
                continue

            if len(train_token_ids_raw) < min_data_needed_per_split:
                 print(f"Skipping {method}: Train split too short ({len(train_token_ids_raw)}) for sequence length ({self.cfg.seq_len}).")
                 continue
            if len(val_token_ids_raw) < min_data_needed_per_split:
                 print(f"Skipping {method}: Validation split too short ({len(val_token_ids_raw)}) for sequence length ({self.cfg.seq_len}).")
                 continue


            print("Building DataLoaders...")
            train_loader = self.build_loader(train_token_ids_raw, tokenizer_info, shuffle=True)
            val_loader = self.build_loader(val_token_ids_raw, tokenizer_info, shuffle=False)

            if train_loader is None or val_loader is None:
                print(f"Skipping training and evaluation for {method} due to DataLoader creation failure.")
                continue

            print("Instantiating model...")
            pad_id = self._get_pad_id_from_info(tokenizer_info)
            model = DecoderOnlyTransformerLanguageModel(
                vocab_size=vocab_size,
                embed_dim=self.cfg.embed_dim,
                num_heads=self.cfg.num_heads,
                hidden_dim=self.cfg.hidden_dim,
                num_layers=self.cfg.num_layers,
                max_len=self.cfg.seq_len,
                dropout=self.cfg.dropout,
                padding_idx=pad_id
            ).to(self.device)

            if method == "word" and tokenizer_info.get('embedding_matrix') is not None:
                 try:
                     print(f"Initializing word embedding layer with pre-trained vectors for {method}.")
                     model.embed.weight = nn.Parameter(tokenizer_info['embedding_matrix'])
                 except Exception as e:
                     print(f"Error initializing word embeddings for {method}: {e}. Proceeding with default initialization.")


            self.train(model, train_loader, val_loader, model_type=method, tokenizer_info=tokenizer_info)

            evaluation_results = self.perform_comprehensive_evaluation(
                model,
                method,
                val_loader
            )
            final_evaluation_results[method] = evaluation_results

            print(f"\nGenerating sample texts for {method.upper()}...")
            samples_for_method = [self.sample(model, "<sos>", max_len=self.cfg.evaluation_sample_max_len, model_type=method, temperature=1.0, top_p=0.9) for _ in range(self.cfg.num_evaluation_samples)]
            all_generated_samples[method] = samples_for_method

            self.analyze_temperature_scaling(model, method, prefix="<sos>")

        print("\n--- Comparative Analysis Summary ---")

        if final_evaluation_results:
             self.log_evaluation_results(final_evaluation_results)
        else:
             print("No successful evaluations to log.")

        if all_generated_samples:
            self.save_sample_outputs(all_generated_samples)
        else:
             print("No samples generated to save.")

        self.plot_comparison_curves()

        print("\nExperiment complete. Check generated plots and log file for results.")


# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comparative text generation experiments.")
    parser.add_argument("--file_paths", nargs="+", default=["shakespeare.txt"],
                        help="Paths to training text files.")
    args = parser.parse_args()
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    system = ControlledTextGenerator(config, device)
    system.run(args.file_paths)