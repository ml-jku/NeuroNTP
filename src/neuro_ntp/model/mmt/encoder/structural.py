"""
+----------------------------------------------------------------------------------------------------------------------+
structural.py
Structural encoders of the multimodal tokenizer: token type, positional encodings and mask tokens

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+
"""
import torch
import torch.nn as nn

class TokenTypeEncoder(nn.Module):
    """
    Token type encoder module: maps type indices to learned embeddings.
    """
    def __init__(self, num_token_types, embedding_dim):
        super(TokenTypeEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_token_types, embedding_dim=embedding_dim)

    def forward(self, token_types):
        """
        Args:
            token_types: Tensor of shape (batch_size, seq_len) containing token type indices.

        Returns:
            Tensor of shape (batch_size, seq_len, embedding_dim): Encoded token type embeddings.
        """
        return self.embedding(token_types)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout_rate=0.15, seq_len=30):
        super(PositionalEncoder, self).__init__()
        self.seq_len = seq_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_rate)

    def encode(self, x):
        # adds the encoding of the fixed position seq_len to all tokens (a constant offset, no pos.info)
        # the tokenizer uses a continuous positional encoder
        out = x + self.pe[:, self.seq_len, :]
        out = self.dropout(out)
        return out


class ContinuousPositionalEncoder():
    def __init__(self, d_model, max_wavelength=10000):
        super(ContinuousPositionalEncoder, self).__init__()
        self.d_model = d_model
        self.omega = 1.0 / (max_wavelength ** (torch.arange(0, d_model, 2).float() / d_model))

    def encode(self, positions):
        positions = positions.unsqueeze(-1)  # (seq_len, 1)
        angles = positions * self.omega.unsqueeze(0)  # (seq_len, d_model // 2)
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return encoding


class MaskToken(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, 1, 1, dim))
        nn.init.normal_(self.embedding, std=0.02)
    def forward(self):
        return self.embedding