"""
+----------------------------------------------------------------------------------------------------------------------+
age.py
Age encoder of the multimodal tokenizer

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+
"""
from typing import ClassVar
import torch
import torch.nn as nn
from datetime import datetime, timezone
from neuro_ntp.model.mmt.core.mmt_structs import MMTSample, ModalityEncoder, ModalityOutput
from neuro_ntp.model.mmt.core.token_type import TokenType

class AgeEncoder(ModalityEncoder):
    """
    Continuous age in years, projected with Linear(1 -> D). Uses 'age' from the metadata if present, otherwise
    computes it from the timestamp and the birth date or year of birth. Clamped to [0, 120]; no normalisation applied.
    """
    token_type: ClassVar[TokenType] = TokenType.Age
    SEC_PER_YEAR = 31557600.0  # 365.2425 days

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.proj = nn.Linear(1, embedding_dim)

    def _age_years(self, meta: dict) -> float:
        # Use provided age directly if present
        if "age" in meta:
            a = float(meta["age"])
        else:
            ts = meta["timestamp"]
            if ts > 1e12:
                ts = ts / 1000.0
            dob_epoch = meta.get("dob_epoch", None)
            if dob_epoch is None:
                year_now = datetime.fromtimestamp(ts, tz=timezone.utc).year
                a = float(year_now - meta["year_of_birth"])
            else:
                a = max(0.0, (ts - float(dob_epoch)) / self.SEC_PER_YEAR)
        return float(min(max(a, 0.0), 120.0))

    def encode(self, mmt, msample: MMTSample):
        if not bool(mmt.config.get("add_age", False)):
            return None

        B, T, D = msample.batch_size, msample.seq_len, self.proj.out_features
        device = next(self.parameters()).device

        ages = [self._age_years(meta) for meta in msample.metadata]
        age = torch.tensor(ages, dtype=torch.float, device=device).unsqueeze(1)  # (B,1)

        z = self.proj(age)                                   # (B,D)
        z = z.unsqueeze(1).unsqueeze(2).expand(-1, T, 1, -1) # (B,T,1,D)

        # optional masking
        z_alt = z
        mp = float(mmt.config.get("age_mask_prob", 0.0))
        if mp > 0.0:
            mask_token = mmt.encoders.get("mask_tokens", "age")()  # (1,1,1,D) tensor
            z_alt = mmt.apply_token_masking(emb_to_mask=z_alt, mask_prob=mp, mask_token=mask_token)

        return ModalityOutput(block=z, block_alt=z_alt, token_types=[self.token_type])
