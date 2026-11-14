"""
+----------------------------------------------------------------------------------------------------------------------+
markers.py
Clinical marker and notes encoders of the multimodal tokenizer

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+
"""
from typing import Optional, ClassVar, Dict
import torch
import torch.nn as nn
from neuro_ntp.model.mmt.core.mmt_structs import MMTSample, ModalityEncoder, ModalityOutput
from neuro_ntp.model.mmt.core.token_type import TokenType
import numpy as np

class _PrecomputedTextProjector(ModalityEncoder):
    """
    Generic projector for precomputed text embeddings (e.g., SBERT vectors produced by the DataModule).
    Expects msample.<prefix>_embs as a float tensor of shape (B, T, in_dim) or (B*T, in_dim).
    """
    token_type: ClassVar[TokenType] = TokenType.INVALID
    cfg_prefix: ClassVar[str] = "text"

    def __init__(self, in_dim: int, embedding_dim: int, mask_prob_key: Optional[str] = None):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.proj = nn.Linear(in_dim, embedding_dim)
        # e.g., "markers_mask_prob" or "notes_mask_prob"
        self.mask_prob_key = mask_prob_key or f"{self.cfg_prefix}_mask_prob"

    def encode(self, mmt, msample: MMTSample) -> Optional[ModalityOutput]:
        add = bool(mmt.config.get(f"add_{self.cfg_prefix}", False))
        dec = bool(mmt.config.get(f"decode_{self.cfg_prefix}", False))
        if not (add or dec):
            return None

        B, T = msample.batch_size, msample.seq_len
        device = next(self.parameters()).device

        # ---- targets ----
        targets = {}
        if self.cfg_prefix == "markers":
            arr = msample.chunks[:, :, 3]       # chunks column 3: marker text of the timestep
            mask_np = np.vectorize(lambda s: len(str(s).strip()) > 0)(arr)  # bool (B,T)
            targets["marker_present"] = torch.from_numpy(mask_np.astype(np.bool_))  # keep on CPU

        # pull precomputed embeddings from the sample
        X = getattr(msample, f"{self.cfg_prefix}_embs", None)
        if (not add) or (X is None):
            return ModalityOutput(block=None, block_alt=None, token_types=[], targets=(targets or None))

        if X.dim() == 2:
            # (B*T, in_dim)
            assert X.size(0) == B * T and X.size(1) == self.in_dim, \
                f"Expected {(B*T, self.in_dim)}, got {tuple(X.shape)} for {self.cfg_prefix}_embs"
        elif X.dim() == 3:
            # (B, T, in_dim) → flatten
            assert X.size(0) == B and X.size(1) == T and X.size(2) == self.in_dim, \
                f"Expected {(B, T, self.in_dim)}, got {tuple(X.shape)} for {self.cfg_prefix}_embs"
            X = X.reshape(B * T, self.in_dim)
        else:
            raise ValueError(f"Unexpected {self.cfg_prefix}_embs shape: {tuple(X.shape)}")

        X = X.to(device=device, dtype=self.proj.weight.dtype)

        # project to model space and reshape to (B,T,1,D)
        Z = self.proj(X).view(B, T, 1, self.embedding_dim)

        # optional masking on the altered stream
        Z_alt = Z
        mp = float(mmt.config.get(self.mask_prob_key, 0.0))
        if mp > 0.0:
            mask_tok = mmt.encoders.get("mask_tokens", self.cfg_prefix)()  # (1,1,1,D)
            Z_alt = mmt.apply_token_masking(Z_alt, mp, mask_tok)

        return ModalityOutput(block=Z, block_alt=Z_alt, token_types=[self.token_type], targets=(targets or None))


class MarkerEncoder(_PrecomputedTextProjector):
    token_type: ClassVar[TokenType] = TokenType.Marker
    cfg_prefix: ClassVar[str] = "markers"


class NotesEncoder(_PrecomputedTextProjector):
    token_type: ClassVar[TokenType] = TokenType.Notes
    cfg_prefix: ClassVar[str] = "notes"
