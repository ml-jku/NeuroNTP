"""
+----------------------------------------------------------------------------------------------------------------------+
gender.py
Gender encoder of the multimodal tokenizer

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+
"""
from typing import ClassVar, List, Dict, Optional
import torch
import torch.nn as nn
from neuro_ntp.model.mmt.core.mmt_structs import MMTSample, ModalityEncoder, ModalityOutput
from neuro_ntp.model.mmt.core.token_type import TokenType


class GenderEncoder(ModalityEncoder):
    """
    Categorical gender → embedding. Defaults to 3 buckets (M/F/UNK).
    If there is only M/F, set num_classes=2 and unknown_index=1 (or adjust mapping).
    """
    token_type: ClassVar[TokenType] = TokenType.Gender

    def __init__(self, embedding_dim: int, num_classes: int = 3, unknown_index: int = 2):
        super().__init__()
        assert num_classes >= 2
        assert 0 <= unknown_index < num_classes
        self.embed = nn.Embedding(num_classes, embedding_dim)
        self.unknown_index = unknown_index
        # normalize strings → indices
        self._map: Dict[str, int] = {"m": 0, "male": 0, "1": 0, "f": 1, "female": 1, "2": 1}

    def encode(self, mmt, msample: MMTSample):
        if not bool(mmt.config.get("add_gender", False)):
            return None

        B, T, D = msample.batch_size, msample.seq_len, self.embed.embedding_dim
        device = next(self.parameters()).device

        idx_list = [
            self._map.get(str(meta.get("gender", "")).lower(), self.unknown_index)
            for meta in msample.metadata
        ]
        idx = torch.tensor(idx_list, dtype=torch.long, device=device)     # (B,)
        z = self.embed(idx)                                               # (B,D)
        z = z.unsqueeze(1).unsqueeze(2).expand(-1, T, 1, -1)              # (B,T,1,D)

        # optional masking
        z_alt = z
        mp = float(mmt.config.get("gender_mask_prob", 0.0))
        if mp > 0.0:
            mask_token = mmt.encoders.get("mask_tokens", "gender")()  # (1,1,1,D) tensor
            z_alt = mmt.apply_token_masking(emb_to_mask=z_alt, mask_prob=mp, mask_token=mask_token)

        return ModalityOutput(block=z, block_alt=z_alt, token_types=[self.token_type])
