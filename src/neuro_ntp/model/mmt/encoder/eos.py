"""
+----------------------------------------------------------------------------------------------------------------------+
eos.py
End-of-segment (EoS) token encoder of the multimodal tokenizer

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+
"""
from typing import Optional, ClassVar
import torch
import torch.nn as nn
from neuro_ntp.model.mmt.core.mmt_structs import MMTSample, ModalityEncoder, ModalityOutput
from neuro_ntp.model.mmt.core.token_type import TokenType


class EosEncoder(ModalityEncoder):
    """
    End-of-Segment token channel.
    Emits a (B, T, 1, D) block that’s either:
      - a fixed zero vector (default), or
      - a learnable token shared across all (B,T).
    """
    token_type: ClassVar[TokenType] = TokenType.EOS

    def __init__(self, embedding_dim: int, learnable: bool = False, init: str = "zeros"):
        super().__init__()
        self.embedding_dim = embedding_dim
        if learnable:
            # learnable (1,1,1,D) parameter that will be broadcast to (B,T,1,D)
            self.token = nn.Parameter(torch.empty(1, 1, 1, embedding_dim))
            if init == "xavier":
                nn.init.xavier_uniform_(self.token)
            elif init == "normal":
                nn.init.normal_(self.token, mean=0.0, std=0.02)
            else:  # "zeros" (default)
                nn.init.zeros_(self.token)
        else:
            # fixed zero buffer that follows device/mode; not trainable
            self.register_buffer("token", torch.zeros(1, 1, 1, embedding_dim))

    def encode(self, mmt, msample: MMTSample) -> Optional[ModalityOutput]:
        if not bool(mmt.config.get("add_end_of_segment", False)):
            return None

        B, T, D = msample.batch_size, msample.seq_len, self.embedding_dim
        z = self.token.expand(B, T, 1, D)   # broadcasted view; use .contiguous() if a materialized tensor is needed

        # No masking/noise for EOS; altered == original
        return ModalityOutput(block=z, block_alt=z, token_types=[self.token_type])
