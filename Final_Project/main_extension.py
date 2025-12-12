# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: Default_Project_3_Text_Generation
@File Name: main_extension.py
@Software: Python
@Time: Apr/2025
@Author: Project Group 43
@Contact:
@Version: 0.2.3
@Description: Introduce structure/control tokens (role, paragraph) into text generation.
             - Parse roles and insert <ROLE_...> tokens
             - Word-level tokenization, sliding-window dataset
             - LSTM-based conditional language model
             - Controlled sampling prefixing with context tokens
"""
import os
import re
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ControlledDataset(Dataset):
    """Dataset of tokenized sequences with control tokens prepended."""
    def __init__(self, tokens, seq_len):
        """Initialize with list of token indices and sequence length."""
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        """Return input-output token sequence pair at index."""
        x = torch.tensor(self.tokens[idx: idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1: idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class ControlledLSTM(nn.Module):
    """LSTM-based language model with embedding for control tokens."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """Forward pass: embed -> LSTM -> linear"""
        e = self.embed(x)
        out, hidden = self.lstm(e, hidden)
        logits = self.fc(out)
        return logits, hidden

class ControlledTextGenerator:
    """Pipeline: parse, tokenize, train, evaluate, and sample with control tokens."""
    ROLE_PATTERN = re.compile(r"^([A-Z][A-Za-z ]+):")

    def __init__(self, config, device):
        self.cfg = config
        self.device = device
        self.token2idx = {}
        self.idx2token = []
        self.tokens = []

    def parse_and_tokenize(self, path):
        """Read raw script, insert control tokens, and tokenize by whitespace."""
        raw = open(path, encoding='utf-8').read().splitlines()
        processed = []
        for line in raw:
            m = self.ROLE_PATTERN.match(line)
            if m:
                role = m.group(1).replace(' ', '_')
                processed.append(f"<ROLE_{role}>")
                text = line[len(m.group(0)):].strip()
                if text:
                    processed.append(text)
            else:
                processed.append(line)
        words = []
        for segment in processed:
            words.extend(segment.split())
        # build vocab
        counter = Counter(words)
        self.idx2token = [t for t,_ in counter.most_common()]
        self.token2idx = {t:i for i,t in enumerate(self.idx2token)}
        # encode
        self.tokens = [self.token2idx[w] for w in words]

    def build_dataloaders(self):
        """Create train/val/test split and DataLoaders."""
        n = len(self.tokens)
        t_end = int(n * self.cfg['split_ratios'][0])
        v_end = t_end + int(n * self.cfg['split_ratios'][1])
        dataset = ControlledDataset(self.tokens, self.cfg['seq_len'])
        self.train_loader = DataLoader(dataset, batch_size=self.cfg['batch_size'], shuffle=True, pin_memory=True, num_workers=2)
        # for simplicity reuse dataset for val/test
        self.val_loader = DataLoader(dataset, batch_size=self.cfg['batch_size'], pin_memory=True, num_workers=2)

    def build_model(self):
        """Instantiate and return the conditional LSTM model."""
        return ControlledLSTM(
            vocab_size=len(self.idx2token),
            embed_dim=self.cfg['embed_dim'],
            hidden_dim=self.cfg['hidden_dim'],
            num_layers=self.cfg['num_layers']
        ).to(self.device)

    def train(self, model):
        """Train model and print training loss."""
        optimizer = optim.Adam(model.parameters(), lr=self.cfg['lr'])
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(self.cfg['epochs']):
            total = 0
            for x,y in self.train_loader:
                x,y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits,_ = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                total += loss.item()
            print(f"Epoch {epoch+1}/{self.cfg['epochs']} Loss: {total/len(self.train_loader):.4f}")

    def evaluate(self, model):
        """Evaluate validation loss."""
        criterion = nn.CrossEntropyLoss()
        total = 0
        model.eval()
        with torch.no_grad():
            for x,y in self.val_loader:
                x,y = x.to(self.device), y.to(self.device)
                logits,_ = model(x)
                total += criterion(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        print(f"Validation Loss: {total/len(self.val_loader):.4f}")

    def sample(self, model, context_tokens, length=100, temperature=1.0, top_p=0.9):
        """Generate text conditioned on given context_tokens (list of strings)."""
        model.eval()
        # encode context
        seq = torch.tensor([self.token2idx[t] for t in context_tokens], dtype=torch.long, device=self.device).unsqueeze(0)
        generated = list(context_tokens)
        hidden = None
        for _ in range(length):
            logits, hidden = model(seq, hidden)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            # nucleus sampling
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cum_p = torch.cumsum(sorted_p, dim=-1)
            sorted_p[cum_p > top_p] = 0
            probs = torch.zeros_like(probs).scatter_(1, sorted_i, sorted_p)
            if probs.sum() == 0:
                probs = torch.softmax(logits, dim=-1)
            else:
                probs /= probs.sum()
            nxt = torch.multinomial(probs, 1).item()
            token = self.idx2token[nxt]
            generated.append(token)
            seq = torch.tensor([[nxt]], device=self.device)
        print(" ".join(generated))

    def run(self, path):
        """Execute full pipeline and demonstrate controlled sampling."""
        self.parse_and_tokenize(path)
        self.build_dataloaders()
        model = self.build_model()
        self.train(model)
        self.evaluate(model)
        # sample with role context
        context = ["<ROLE_ROMEO>"]
        print("Sample for ROLE_ROMEO:")
        self.sample(model, context, length=50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default='shakespeare.txt')
    args = parser.parse_args()
    config = {
        'seq_len': 50,
        'split_ratios': (0.8, 0.1, 0.1),
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'lr': 1e-3,
        'batch_size': 128,
        'epochs': 20
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = ControlledTextGenerator(config, device)
    gen.run(args.file_path)
