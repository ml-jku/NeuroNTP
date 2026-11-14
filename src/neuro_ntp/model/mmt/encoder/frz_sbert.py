"""
+----------------------------------------------------------------------------------------------------------------------+
frz_sbert.py
Frozen sentence encoder for clinical marker and notes embeddings

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+
"""
from typing import List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class FrozenSBERT(nn.Module):
    """
    Frozen sentence encoder for offline/precompute use.
    Produces SBERT-sized vectors (no projection). Not registered in MMT.
    """
    def __init__(self, model_name: str, max_length: int = 512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.max_length = max_length
        # expose output dim (hidden size)
        self.out_dim = self.model.config.hidden_size

    @torch.no_grad()
    def get_emb(self, texts: List[str]) -> torch.Tensor:
        """
        texts: list[str]
        returns: (N, out_dim) on current device
        """
        texts = [t if isinstance(t, str) else "" for t in texts]
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        enc = {k: v.to(next(self.parameters()).device) for k, v in enc.items()}
        out = self.model(**enc)                         # last_hidden_state: (N, L, H)
        cls = out.last_hidden_state[:, 0, :]            # (N, H) — CLS token as sentence embedding
        return cls
