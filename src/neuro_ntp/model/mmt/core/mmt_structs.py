"""
+----------------------------------------------------------------------------------------------------------------------+
mmt_structs.py
Data structures of the multimodal tokenizer

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
        MMTSample                   wrapper of a self-supervised batch with its shapes and channel layout
        MMTEncodedEmbeddings        accumulator of the token blocks produced by the modality encoders
        ModalityOutput              output of a single modality encoder (tokens, masked tokens, token types, targets)
        ModalityEncoder             base interface of the modality encoders
        MultiModalTokenSequence     token sequence passed between tokenizer, next-token predictor and heads
+----------------------------------------------------------------------------------------------------------------------+
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, ClassVar
import torch
import torch.nn as nn
from neuro_ntp.model.mmt.core.token_type import TokenType

# -------------------------------------------
# Sample wrapper (just shapes + handy attrs)
# -------------------------------------------
class MMTSample:
    """
    Wraps a single DataLoader batch for MMT.
    Expects: sample = (signal, chunks, markers_embs, notes_embs, metadata)
      - signal: (B, C, T, L) or (B, C, L) → we'll add T=1
      - chunks: arbitrary, e.g. (B, T, ..)
      - metadata: list[dict] of length B, all with same channel layout
    """
    def __init__(self, sample):
        self.signal, self.chunks, self.markers_embs, self.notes_embs, self.metadata = sample

        # consistent channel layout across batch (first item is canonical)
        first = self.metadata[0]
        self.signal_channels: List[str] = first["signal_channels"]
        self.modalities: List[str] = first.get("modalities", [])

        # infer signal shape → (B, C, T, L)
        if self.signal.dim() == 3:  # (B, C, L) → add T=1
            self.signal = self.signal.unsqueeze(2)
        assert self.signal.dim() == 4, f"signal must be (B,C,T,L), got {tuple(self.signal.shape)}"

        self.batch_size, self.num_ch, self.seq_len, self.ts = self.signal.shape

        # prefer model's device later; keep where the tensor lives for now
        self.device = self.signal.device

        # sanity check: metadata length == B
        assert len(self.metadata) == self.batch_size, "metadata length must equal batch size"


# ------------------------------
# Accumulator for modality blocks
# ------------------------------
@dataclass
class MMTEncodedEmbeddings:
    blocks: List[torch.Tensor] = field(default_factory=list)          # each: (B, T, C_or_1, D)
    blocks_altered: List[torch.Tensor] = field(default_factory=list)
    token_types: List[TokenType] = field(default_factory=list)
    encoded_targets: Dict[str, torch.Tensor] = field(default_factory=dict)

    def append_block(
        self,
        block: torch.Tensor,
        block_alt: Optional[torch.Tensor],
        token_types: List[str],
    ):
        # Basic shape checks
        assert block.dim() == 4, f"block must be (B,T,C,D), got {tuple(block.shape)}"
        if block_alt is not None:
            assert block_alt.shape == block.shape, "block_alt must match block shape"
        B, T, C, D = block.shape
        assert len(token_types) == C, f"token_types length {len(token_types)} != C {C}"

        self.blocks.append(block)
        self.blocks_altered.append(block if block_alt is None else block_alt)
        self.token_types.extend(token_types)

    @property
    def total_channels(self) -> int:
        return len(self.token_types)

    def as_tensor(self, altered: bool = False) -> torch.Tensor:
        """Concatenate all blocks along channel dim → (B, T, C_total, D)."""
        blocks = self.blocks_altered if altered else self.blocks
        assert len(blocks) > 0, "no blocks to concatenate"
        # Ensure (B,T,*,D) are consistent
        B, T, _, D = blocks[0].shape
        for b in blocks[1:]:
            assert b.shape[0] == B and b.shape[1] == T and b.shape[3] == D, "inconsistent block shapes"
        return torch.cat(blocks, dim=2)


# ------------------------------
# Output of a modality encoder
# ------------------------------
@dataclass
class ModalityOutput:
    block: Optional[torch.Tensor] = None          # (B, T, C_or_1, D)
    block_alt: Optional[torch.Tensor] = None      # same shape
    token_types: List[TokenType] = field(default_factory=list)
    targets: Optional[Dict[str, torch.Tensor]] = None  # e.g. {"segtype": (B, T)}


# ------------------------------
# Minimal interface: just encode()
# ------------------------------
class ModalityEncoder(nn.Module):
    token_type: ClassVar[TokenType] = TokenType.INVALID

    def encode(self, mmt, msample: MMTSample) -> Optional[ModalityOutput]:
        """
        return None to skip this modality for this batch.
        """
        raise NotImplementedError


@dataclass
class MultiModalTokenSequence:
    tokens: torch.Tensor
    target_tokens: torch.Tensor
    token_types: torch.Tensor
    tps: Optional[torch.Tensor] = None
    tts: Optional[torch.Tensor] = None
    seg_types: Optional[torch.Tensor] = None
    marker_pos: Optional[torch.Tensor] = None
    markers: Optional[torch.Tensor] = None
    marker_present: Optional[torch.Tensor] = None

