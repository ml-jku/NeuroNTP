"""
+----------------------------------------------------------------------------------------------------------------------+
tts.py
Time-to-seizure (TTS) target encoder of the multimodal tokenizer

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+
"""
from typing import Optional, Dict, ClassVar
import torch
import torch.nn as nn
from neuro_ntp.model.mmt.core.mmt_structs import MMTSample, ModalityEncoder, ModalityOutput
from neuro_ntp.model.mmt.core.token_type import TokenType
import numpy as np


class TtsEncoder(ModalityEncoder):
    """
    Time-To-Seizure (TTS) modality (targets only).

    Config keys:
      - predict_tts (bool):        if False → returns None
      - tts_cap (float|None):      if set, cap values (soft-cap if mmt.soft_cap available, else hard clamp)
      - time_cap_softness (float): softness for soft_cap
      - tts_logspace (bool):       if True: log1p(clamp(tts + seizure_len_cap, min=1))
                                   else:   scale by tts_cap (or batch max as fallback)
      - seizure_len_cap (float):   added before log to avoid log(0) and to include seizure len context
    """
    token_type: ClassVar[TokenType] = TokenType.TTS   # target-only, no tokens emitted

    def encode(self, mmt, msample: MMTSample) -> Optional[ModalityOutput]:
        if not bool(mmt.config.get("predict_tts", False)):
            return None
        B, T = msample.batch_size, msample.seq_len
        # TTS encoder has no parameters - infer device from MMT
        device = next(mmt.parameters()).device

        # 1) raw TTS from chunks column 6: (B, T) -> (B*T, 1)
        col = msample.chunks[:, :, 6].astype(np.float32)
        tts = torch.as_tensor(col.reshape(-1, 1), dtype=torch.float, device=device)

        # 2) apply cap / soft-cap
        cap = mmt.config.get("tts_cap", None)
        if cap is not None:
            cap = float(cap)
            softness = float(mmt.config.get("time_cap_softness", 1.0))
            if hasattr(mmt, "soft_cap"):
                tts = mmt.soft_cap(tts, cap=cap, softness=softness)
            else:
                tts = torch.minimum(tts, torch.tensor(cap, device=device, dtype=tts.dtype))

        # 3) logspace or scaling
        if bool(mmt.config.get("tts_logspace", False)):
            seiz_cap = float(mmt.config.get("seizure_len_cap", 0.0))
            # avoid log(0); keep strictly positive inside log1p
            tts = torch.log1p(torch.clamp(tts + seiz_cap, min=1.0))
        else:
            # scale to ~[0,1]
            if cap is not None and cap > 0:
                denom = cap
            else:
                # fallback: batch max (avoid div-by-zero)
                denom = float(tts.max().clamp_min(1.0))
            tts = tts / denom

        # 4) targets only (no token block)
        targets: Dict[str, torch.Tensor] = {"tts": tts.view(B, T).detach().cpu()}

        return ModalityOutput(block=None, block_alt=None, token_types=[], targets=targets)
