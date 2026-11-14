"""
+----------------------------------------------------------------------------------------------------------------------+
hod.py
Hour-of-day encoder of the multimodal tokenizer

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+
"""
from typing import ClassVar, List, Optional
import torch
import torch.nn as nn
from datetime import datetime, timezone

from neuro_ntp.model.mmt.core.token_type import TokenType

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None
from neuro_ntp.model.mmt.core.mmt_structs import MMTSample, ModalityEncoder, ModalityOutput

class HodEncoder(ModalityEncoder):
    """
    Hour-of-day (0..23) → embedding. Converts timestamps to local hour.
    Optionally set config['hod_timezone'] to an IANA tz (e.g., "Europe/Vienna").
    """
    token_type: ClassVar[TokenType] = TokenType.HoD

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embed = nn.Embedding(24, embedding_dim)

    def _hour_from_ts(self, ts: float, tzname: Optional[str]) -> int:
        # handle ms vs s
        if ts > 1e12:
            ts = ts / 1000.0
        dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        if tzname and ZoneInfo is not None:
            dt_local = dt_utc.astimezone(ZoneInfo(tzname))
        else:
            dt_local = dt_utc  # UTC if no tz specified / ZoneInfo unavailable
        return int(dt_local.hour)

    def encode(self, mmt, msample: MMTSample):
        if not bool(mmt.config.get("add_hod", False)):
            return None

        B, T, D = msample.batch_size, msample.seq_len, self.embed.embedding_dim
        device = next(self.parameters()).device
        tzname = mmt.config.get("hod_timezone", None)

        hours = [self._hour_from_ts(meta["timestamp"], tzname) for meta in msample.metadata]
        hrs = torch.tensor(hours, dtype=torch.long, device=device)        # (B,)
        z = self.embed(hrs)                                               # (B,D)
        z = z.unsqueeze(1).unsqueeze(2).expand(-1, T, 1, -1)              # (B,T,1,D)

        # optional masking
        z_alt = z
        mp = float(mmt.config.get("hod_mask_prob", 0.0))
        if mp > 0.0:
            mask_token = mmt.encoders.get("mask_tokens", "hod")()  # (1,1,1,D) tensor
            z_alt = mmt.apply_token_masking(emb_to_mask=z_alt, mask_prob=mp, mask_token=mask_token)

        return ModalityOutput(block=z, block_alt=z_alt, token_types=[self.token_type])
