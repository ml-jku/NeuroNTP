"""
+----------------------------------------------------------------------------------------------------------------------+
segtype.py
Segment-type encoder of the multimodal tokenizer

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+
"""
from typing import Optional, Dict, List, ClassVar
import torch
import torch.nn as nn
from neuro_ntp.model.mmt.core.mmt_structs import MMTSample, ModalityEncoder, ModalityOutput

from neuro_ntp.model.mmt.core.token_type import TokenType


class SegmentTypeEncoder(ModalityEncoder):
    """
    Segment-type modality encoder class.
    - class-level SEG_TYPES sets both feature vectors (3-dim) and label→id order.
    - projects 3 → D via nn.Linear.
    - returns targets (ids) if predict_segtype=True.
    - returns (B,T,1,D) block if add_segtype=True (optional masking).
    """
    token_type: ClassVar[TokenType] = TokenType.Segtype

    # insertion order defines label ids: 0:'non-ictal', 1:'ictal', 2:'unspecified'
    SEG_TYPES: Dict[str, List[float]] = {
        'non-ictal' : [1.0, 0.0, 0.0],
        'ictal'     : [0.0, 1.0, 0.0],
        'unspecified': [0.0, 0.0, 1.0],
    }

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

        # label order & id mapping from class dict
        self.labels: List[str] = list(self.SEG_TYPES.keys())
        self.id_from_label: Dict[str, int] = {lbl: i for i, lbl in enumerate(self.labels)}

        # build K×3 feature LUT and register as buffer so it moves with .to(device)
        feat = torch.tensor([self.SEG_TYPES[lbl] for lbl in self.labels], dtype=torch.float32)  # (K,3)
        self.register_buffer("feat_lut", feat)

        # 3 → D projection
        self.proj = nn.Linear(feat.shape[1], embedding_dim)

    def encode(self, mmt, msample: MMTSample) -> Optional[ModalityOutput]:
        add_block   = bool(mmt.config.get("add_segtype", False))
        want_target = bool(mmt.config.get("predict_segtype", False))
        if not (add_block or want_target):
            return None

        B, T, D = msample.batch_size, msample.seq_len, self.embedding_dim
        device = next(self.parameters()).device

        # --- map labels → ids (B,T) ---
        # make mapping robust to case/whitespace and default to 'unspecified'
        labels_2d = msample.chunks[:, :, 2]
        flat = labels_2d.reshape(-1).tolist()

        def norm(s):
            return str(s).strip().lower()

        id_from_label_norm = {k: v for k, v in self.id_from_label.items()}
        id_from_label_norm = {norm(k): v for k, v in id_from_label_norm.items()}
        ids = torch.tensor([id_from_label_norm.get(norm(l), id_from_label_norm['unspecified'])
                            for l in flat], dtype=torch.long, device=device).view(B, T)

        targets = {"segtype": ids.detach().cpu()} if want_target else None

        if not add_block:
            return ModalityOutput(block=None, block_alt=None,
                                  token_types=[], targets=targets)

        # --- features via LUT (K,3) → project (3→D) → (B,T,1,D) ---
        x = self.feat_lut[ids.view(-1)]       # (B*T, 3)
        z = self.proj(x).view(B, T, 1, D)     # (B,T,1,D)

        # optional masking
        z_alt = z
        mp = float(mmt.config.get("segtype_mask_prob", 0.0))
        if mp > 0.0:
            mask_token = mmt.encoders.get("mask_tokens", "segtype")()  # (1,1,1,D) tensor
            z_alt = mmt.apply_token_masking(emb_to_mask=z_alt, mask_prob=mp, mask_token=mask_token)

        return ModalityOutput(block=z, block_alt=z_alt,
                              token_types=[self.token_type],
                              targets=targets)
