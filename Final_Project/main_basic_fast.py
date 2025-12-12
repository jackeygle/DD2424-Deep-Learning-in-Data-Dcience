# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
@Project Name: Default_Project_3_Text_Generation
@File Name: main_basic_fast.py
@Software: Python
@Time: Apr/2025
@Author: Project Group 43
@Contact:
@Version: 0.4.6
@Description: Basic E-level Shakespeare text generation.
             - Optimized to better utilize GPU
             - Larger batch size, pinned memory, faster data transfer
"""

import os
import argparse
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, data_tensor, seq_len):
        """Initialize the dataset with input tensor and sequence length."""
        self.data = data_tensor
        self.seq_len = seq_len

    def __len__(self):
        """Return the number of sequences that can be generated."""
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        """Return a tuple of input and target sequences at a specific index."""
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y

class CharModel(nn.Module):
    def __init__(self, model_type, vocab_size, embed_dim, hidden_dim, num_layers=1):
        """Initialize the character-level RNN or LSTM model."""
        super().__init__()
        self.model_type = model_type
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if model_type == 'rnn':
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """Forward pass through the model."""
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

class ShakespeareTextGenerator:
    DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

    def __init__(self, config, device):
        """Initialize the text generator with configuration and device."""
        self.cfg = config
        self.device = device
        self.char2idx = {}
        self.idx2char = {}
        self.encoded = None

    def download_data(self, path):
        """Download the Shakespeare dataset if not already present."""
        if not os.path.exists(path):
            urllib.request.urlretrieve(self.DATA_URL, path)

    def load_data(self, path):
        """Load and encode the dataset, split into train/val/test loaders."""
        text = open(path, encoding='utf-8').read()
        chars = sorted(set(text))
        self.char2idx = {c:i for i,c in enumerate(chars)}
        self.idx2char = {i:c for c,i in self.char2idx.items()}
        self.cfg['vocab_size'] = len(chars)
        tensor = torch.tensor([self.char2idx[c] for c in text], dtype=torch.long)
        self.encoded = tensor.to(self.device)
        n = len(tensor)
        t_end = int(n * self.cfg['split_ratios'][0])
        v_end = t_end + int(n * self.cfg['split_ratios'][1])
        self.train_loader = DataLoader(SequenceDataset(tensor[:t_end], self.cfg['seq_len']), batch_size=self.cfg['batch_size'], shuffle=True, pin_memory=True, num_workers=2)
        self.val_loader   = DataLoader(SequenceDataset(tensor[t_end:v_end], self.cfg['seq_len']), batch_size=self.cfg['batch_size'], pin_memory=True, num_workers=2)
        self.test_loader  = DataLoader(SequenceDataset(tensor[v_end:], self.cfg['seq_len']), batch_size=self.cfg['batch_size'], pin_memory=True, num_workers=2)

    def build_model(self, model_type):
        """Build and return an RNN or LSTM model based on the specified type."""
        vocab_size = self.cfg['vocab_size']
        embed_dim = self.cfg['embed_dim']
        hidden_dim = self.cfg['hidden_dim']
        num_layers = 1 if model_type == 'lstm1' else 2 if model_type == 'lstm2' else 1
        model_name = 'lstm' if 'lstm' in model_type else 'rnn'
        model = CharModel(model_name, vocab_size, embed_dim, hidden_dim, num_layers)
        return model.to(self.device)

    def train(self, model, model_name):
        """Train the model using the training dataset."""
        optimizer = optim.Adam(model.parameters(), lr=self.cfg['lr'])
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.cfg['epochs']):
            total = 0
            model.train()
            for x,y in self.train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward(); optimizer.step(); total += loss.item()
            print(f"{model_name.upper()} Epoch {epoch+1}/{self.cfg['epochs']} Loss: {total/len(self.train_loader):.4f}")

    def evaluate(self, model, model_name):
        """Evaluate the model on the validation dataset and print loss."""
        criterion = nn.CrossEntropyLoss()
        total = 0
        model.eval()
        with torch.no_grad():
            for x,y in self.val_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                logits, _ = model(x)
                total += criterion(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        print(f"{model_name.upper()} Validation Loss: {total/len(self.val_loader):.4f}")

    def sample(self, model, model_name):
        """Generate a sample text using the trained model."""
        model.eval()
        idx = torch.randint(0, len(self.encoded)-self.cfg['seq_len']-1, (1,)).item()
        seq = self.encoded[idx:idx+self.cfg['seq_len']].unsqueeze(0)
        generated = ''.join(self.idx2char[i.item()] for i in seq.flatten())
        hidden = None
        for _ in range(self.cfg['max_len']):
            logits, hidden = model(seq, hidden)
            logits = logits[:, -1, :] / self.cfg['temperature']
            probs = torch.softmax(logits, dim=-1)
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cum_p = torch.cumsum(sorted_p, dim=-1)
            sorted_p[cum_p > self.cfg['top_p']] = 0
            probs = torch.zeros_like(probs).scatter_(1, sorted_i, sorted_p)
            if probs.sum().item() == 0:
                probs = torch.softmax(logits, dim=-1)
            else:
                probs /= probs.sum()
            nxt = torch.multinomial(probs, 1)
            nxt_token = nxt.squeeze(1) if nxt.dim() == 2 else nxt
            generated += self.idx2char[nxt_token.item()]
            seq = torch.cat([seq[:, 1:], nxt_token.unsqueeze(1)], dim=1)
        print(f"Sample from {model_name.upper()}:\n{generated}\n")

    def run(self, path):
        """Main execution method to prepare data, train, evaluate, and generate samples for each model type."""
        self.download_data(path); self.load_data(path)
        for m in ['rnn','lstm1','lstm2']:
            model = self.build_model(m)
            self.train(model, m)
            self.evaluate(model, m)
            self.sample(model, m)

if __name__=='__main__':
    file_path = 'shakespeare.txt'
    config = {
        'seq_len':100,
        'split_ratios':(0.8,0.1,0.1),
        'embed_dim':128,
        'hidden_dim':256,
        'lr':1e-3,
        'batch_size':256,
        'epochs':5,
        'max_len':200,
        'temperature':1.0,
        'top_p':0.9
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = ShakespeareTextGenerator(config, device)
    gen.run(file_path)
