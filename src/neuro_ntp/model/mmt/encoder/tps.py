"""
+----------------------------------------------------------------------------------------------------------------------+
tps.py
Time-past-seizure (TPS) encoder of the multimodal tokenizer

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
import numpy as np


class TpsEncoder(ModalityEncoder):
    """
    Time-Past-Seizure (TPS) modality.

    Config keys used:
      - add_tps (bool):          add a (B,T,1,D) token to the sequence
      - predict_tps (bool):      expose TPS as a target (B,T) tensor (post-processed)
      - tps_cap (float|None):    cap value; if set, apply soft_cap (if available) or hard clamp
      - time_cap_softness (float): softness for soft_cap (if mmt.soft_cap exists)
      - tps_logspace (bool):     if True use log1p; else scale by cap (or batch max as fallback)
      - tps_mask_prob (float):   masking probability for TPS token (optional)
    """
    token_type: ClassVar[TokenType] = TokenType.TPS

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.proj = nn.Linear(1, embedding_dim)  # (N,1) -> (N,D)

    # helpers
    @staticmethod
    def _extract_tps(msample: MMTSample, device) -> torch.Tensor:
        """Raw TPS from chunks column 5 (NumPy object array) as a (B*T, 1) float tensor on device."""
        col = msample.chunks[:, :, 5].astype(np.float32)  # (B, T)
        return torch.as_tensor(col.reshape(-1, 1), dtype=torch.float, device=device)

    @staticmethod
    def _process_tps(mmt, tps: torch.Tensor) -> torch.Tensor:
        """
        Apply cap / soft-cap, clamp, and log/scale. Returns (B*T,1) on same device.
        """
        device = tps.device
        cap = mmt.config.get("tps_cap", None)
        if cap is not None:
            softness = float(mmt.config.get("time_cap_softness", 1.0))
            if hasattr(mmt, "soft_cap"):
                # Expecting signature: soft_cap(x, cap, softness)
                tps = mmt.soft_cap(tps, cap=cap, softness=softness)
            else:
                # Fallback: hard clamp to cap
                tps = torch.minimum(tps, torch.tensor(float(cap), device=device, dtype=tps.dtype))

        # avoid log(0) and negatives
        tps = torch.clamp(tps, min=1e-6)

        if bool(mmt.config.get("tps_logspace", False)):
            tps = torch.log1p(tps)  # natural log
        else:
            if cap is not None and float(cap) > 0:
                tps = tps / float(cap)
            else:
                # Fallback scaling: normalize by batch max to keep ~[0,1]
                denom = tps.max().clamp_min(1.0)
                tps = tps / denom
        return tps

    # MMT-facing
    def encode(self, mmt, msample: MMTSample) -> Optional[ModalityOutput]:
        add_block   = bool(mmt.config.get("add_tps", False))
        want_target = bool(mmt.config.get("predict_tps", False))
        if not (add_block or want_target):
            return None

        B, T = msample.batch_size, msample.seq_len
        D = self.proj.out_features
        device = next(self.parameters()).device

        # 1) build (B*T,1) TPS tensor and process it
        tps = self._extract_tps(msample, device=device)           # (B*T,1)
        tps = self._process_tps(mmt, tps)                         # (B*T,1)

        # 2) targets (post-processed) as (B,T) on CPU
        targets = {"tps": tps.view(B, T).detach().cpu()} if want_target else None

        # 3) optional block
        if not add_block:
            return ModalityOutput(block=None, block_alt=None, token_types=[], targets=targets)

        z_flat = self.proj(tps)                                   # (B*T, D)
        z = z_flat.view(B, T, 1, D)                               # (B,T,1,D)

        # 4) optional masking
        z_alt = z
        mp = float(mmt.config.get("tps_mask_prob", 0.0))
        if mp > 0.0:
            mask_token = mmt.encoders.get("mask_tokens", "tps")()
            z_alt = mmt.apply_token_masking(emb_to_mask=z_alt, mask_prob=mp, mask_token=mask_token)

        return ModalityOutput(block=z, block_alt=z_alt, token_types=[self.token_type], targets=targets)
