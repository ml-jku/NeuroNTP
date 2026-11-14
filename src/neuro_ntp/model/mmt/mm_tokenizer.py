"""
+----------------------------------------------------------------------------------------------------------------------+
mm_tokenizer.py
Multimodal tokenizer (MMT) of NeuroNTP

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Encodes a self-supervised sample into a sequence of tokens in a shared embedding space. The modality encoders
    (signal channels, channel subtypes, EEG electrode locations, clinical markers and notes, segment type, time past
    and time to seizure, demographic and time-of-day metadata, mask and end-of-segment tokens) are held in a lazily
    populated registry, so that encoders for new channels or sensors are created on demand. Each token is enriched
    with a token-type and positional encoding. Depending on fusion_level, the tokens of a timestep are either kept as
    a channel-level subsequence ("channel") or fused into one token by a small transformer ("timestep").
    The tokenizer returns the input tokens (with masking applied, if enabled), the target tokens and the per-token
    targets of the auxiliary tasks.
+----------------------------------------------------------------------------------------------------------------------+
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuro_ntp.model.mmt.core.mm_registry import MultiModuleRegistry
from neuro_ntp.model.mmt.core.mmt_structs import MMTEncodedEmbeddings, MMTSample, ModalityEncoder, MultiModalTokenSequence
from neuro_ntp.model.mmt.core.token_type import SUBTYPE_VOCABS, TokenType
from neuro_ntp.model.mmt.encoder.age import AgeEncoder
from neuro_ntp.model.mmt.encoder.ch_loc import ChannelLocationEncoder
from neuro_ntp.model.mmt.encoder.eos import EosEncoder
from neuro_ntp.model.mmt.encoder.gender import GenderEncoder
from neuro_ntp.model.mmt.encoder.hod import HodEncoder
from neuro_ntp.model.mmt.encoder.markers import MarkerEncoder, NotesEncoder
from neuro_ntp.model.mmt.encoder.segtype import SegmentTypeEncoder
from neuro_ntp.model.mmt.encoder.signals import make_nearest_init_fn, SignalCollectiveEncoder, ChannelLoRAHead
from neuro_ntp.model.mmt.encoder.structural import TokenTypeEncoder, ContinuousPositionalEncoder, PositionalEncoder, MaskToken
from neuro_ntp.model.mmt.encoder.tps import TpsEncoder
from neuro_ntp.model.mmt.encoder.tts import TtsEncoder
import neuro_ntp.misc.nntp_utils as nnutils


class MultiModalTokenizer(nn.Module):

    def __init__(self, config: dict, unflattened: bool = False):
        super(MultiModalTokenizer, self).__init__()
        # tokenizer configuration
        self.config = config

        # default mode upon initialization
        self.frozen = False         # MMT is not frozen
        self.token_masking = False    # normal pretraining / not masking
        self.fusion_level = self.config["fusion_level"]
        self.embedding_dim = self.config['embedding_dim']
        self.marker_emb_dim = self.config['marker_emb_dim']
        self.unflattened = unflattened
        self.payload_noise = self.config['payload_noise']

        ######################################
        # Register modality encoder modules  #
        ######################################

        # declare encoder module registry - for handling all encoder modules (lazily)
        self.encoders = MultiModuleRegistry(default_class_key="signals")

        # register separate signal channel encoders
        self.encoders.register(
            "signal_channels",
            make_module=lambda: ChannelLoRAHead(bottleneck=self.config['sigenc_bottleneck'],
                                                out_dim=self.config['embedding_dim'],
                                                rank=self.config.get('sigenc_lora_rank', 4)),
            nearest_init=make_nearest_init_fn(self.encoders),
            normalize_id=lambda s: s.strip().lower(),
        )

        # register collective orchestrator for all signals
        self.encoders.register(
            "signals",
            make_module=lambda: SignalCollectiveEncoder(
                member_family="signal_channels",
                embedding_dim=self.embedding_dim,
                config=self.config,
                mmt=self
            ),
        )

        # EEG channel location encoder
        self.encoders.register(
            "eeg_location",
            make_module=lambda: ChannelLocationEncoder(embedding_dim=self.embedding_dim))

        # register encoders for other main signal token types with optional subchannels
        for base, vocab in SUBTYPE_VOCABS.items():
            key = base.name.lower()
            v = len(vocab)
            pad = (vocab.index("unk") if "unk" in vocab else None)
            self.encoders.register(
                f"ch_subtype_{key}",
                lambda v=v, D=self.embedding_dim, pad=pad: nn.Embedding(v, D, padding_idx=pad)
            )

        # marker&notes text encoder (NB. processing by SBERT is done in the dataset class, this is mapping to emb_dim)
        self.encoders.register("markers", make_module=lambda: MarkerEncoder(self.marker_emb_dim, self.embedding_dim))
        self.encoders.register("notes", make_module=lambda: NotesEncoder(self.marker_emb_dim, self.embedding_dim))

        # segment type encoder
        self.encoders.register("segtype", make_module=lambda: SegmentTypeEncoder(embedding_dim=self.embedding_dim))

        # time-past-seizure / time-to-seizure encoder
        self.encoders.register("tps", lambda: TpsEncoder(embedding_dim=self.embedding_dim))
        self.encoders.register("tts", lambda: TtsEncoder())

        # other metadata encoders
        self.encoders.register("gender", lambda: GenderEncoder(self.embedding_dim))
        self.encoders.register("age", lambda: AgeEncoder(self.embedding_dim))
        self.encoders.register("hod", lambda: HodEncoder(self.embedding_dim))

        ######################################
        # Create structural encoder modules  #
        ######################################

        # token type encoder
        self.ttype_encoder = TokenTypeEncoder(TokenType.max_id() + 1, self.embedding_dim)

        # continuous positional encoder
        self.max_seq_len = config['max_seq_len']  # maximum sequence length to pre-calculate encoded positions
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        self.continuous_pos_encoder = ContinuousPositionalEncoder(d_model=self.embedding_dim)
        self.pos_enc = self.continuous_pos_encoder.encode(positions)  # (self.max_seq_len, self.embedding_dim)

        # seq. position encoder
        self.pos_encoder = PositionalEncoder(d_model=self.embedding_dim)

        # learnable mask tokens collection
        self.encoders.register("mask_tokens", lambda: MaskToken(self.embedding_dim))

        # End-of-Sequence token encoder
        self.encoders.register("eos", lambda: EosEncoder(self.embedding_dim, learnable=True, init="xavier"))

        # build token fusion transformer - if timestep fusion is used
        if self.fusion_level == "timestep":
            self._build_fusion()

    def registries(self) -> dict:
        # single unified registry
        return {"encoders": self.encoders}

    def set_frozen(self, freeze):
        nnutils.set_module_frozen(self, freeze)

    def soft_cap(self, value, cap, softness=1000):
        return cap - softness * F.softplus((cap - value) / softness)

    def apply_token_masking(self, emb_to_mask, mask_prob, mask_token):
        if not (self.training or self.token_masking) or mask_prob <= 0.0:
            return emb_to_mask

        if mask_prob >= 1.0:
            return mask_token.expand_as(emb_to_mask)  # no RNG, deterministic

        B, T = emb_to_mask.shape[:2]
        mask = (torch.rand(B, T, 1, 1, device=emb_to_mask.device) < mask_prob)
        return torch.where(mask, mask_token.expand_as(emb_to_mask), emb_to_mask)

    def get_sample(self, sample, i):
        return tuple(x[i] for x in sample)

    def _concat_blocks(self, emb: MMTEncodedEmbeddings):
        """
        Concatenate all modality blocks along channel dim.
        Returns:
          z, z_alt: (B,T,C_total,D)
          type_ids: (B,T,C_total) int
        """
        # concat tensors
        z = torch.cat(emb.blocks, dim=2)  # (B,T,C,D)
        z_alt = torch.cat(emb.blocks_altered, dim=2)  # (B,T,C,D)

        B, T, C, D = z.shape
        device = z.device

        # build per-channel TokenType ids, then broadcast over (B,T)
        # emb.token_types is a list length C_total of enums/ints
        per_ch_ids = torch.as_tensor([int(t) for t in emb.token_types], dtype=torch.long, device=device)  # (C,)
        type_ids = per_ch_ids.view(1, 1, C).expand(B, T, C).contiguous()  # (B,T,C)

        return z, z_alt, type_ids

    def _pick_target(self, target: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Targets stored as (B,T) or (B,T,1) or flattened already.
        Normalize to (B,T) if present.
        """
        if target is None:
            return None
        if target.dim() == 3 and target.size(-1) == 1:
            return target.squeeze(-1)
        return target

    def _repeat_over_C(self, target: Optional[torch.Tensor], C: int) -> Optional[torch.Tensor]:
        """
        Repeat per-timestep target over channels for late fusion flattening.
        Input (B,T) -> output (B, T*C)
        """
        if target is None: return None
        B, T = target.shape[:2]
        return target.unsqueeze(2).expand(-1, -1, C).reshape(B, T * C).contiguous()

    def _build_fusion(self):
        if hasattr(self, "fuse_encoder"):  # already built
            return
        embedding_dim = self.config.get("embedding_dim", 1024)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=self.config.get("fuse_nhead", 8),
            dim_feedforward=self.config.get("fuse_ffn_dim", 4 * embedding_dim),
            dropout=self.config.get("fuse_dropout", 0.1),
            batch_first=True,
        )
        self.fuse_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=self.config.get("fuse_layers", 2)
        )
        self.register_parameter("fuse_token", nn.Parameter(torch.zeros(1, 1, embedding_dim)))
        nn.init.normal_(self.fuse_token, std=0.02)

    # early fusion: per-timestep transformer
    def _fuse_per_timestep(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B,T,C,D) -> (B,T,D)
        Uses a small TransformerEncoder over the C tokens at each timestep.
        """
        B, T, C, D = z.shape
        x = z.view(B * T, C, D)  # (B*T, C, D)
        fuse_tok = self.fuse_token.expand(B * T, -1, -1)      # (B*T, 1, D)
        x_in = torch.cat([fuse_tok, x], dim=1)                # (B*T, 1+C, D)
        y = self.fuse_encoder(x_in)                           # (B*T, 1+C, D)
        y0 = y[:, 0, :]                                       # (B*T, D)  take [FUSE]
        return y0.view(B, T, D)                               # (B, T, D)

    def forward(self, sample, pos_offset: int = 0):
        msample = MMTSample(sample)
        device = msample.device
        B, T, D = msample.batch_size, msample.seq_len, self.embedding_dim

        # initialize a sequence holder object
        emb = MMTEncodedEmbeddings()
        # encode all modalities of the sample
        for modality in msample.modalities:                     # e.g. ["signals","tps","hod","gender","age","markers","notes","eos"]
            if modality.strip().lower() not in self.config['cfg_modalities']:
                continue
            enc = self.encoders.get(modality.strip().lower())   # retrieve encoder (lazy)
            out = enc.encode(self, msample)                     # encode the sample (block of embeddings: original & masked)
            if out is None:
                continue
            if out.block is not None:
                emb.append_block(out.block, out.block_alt, out.token_types)     # collect the encoded token blocks
            if out.targets:
                emb.encoded_targets.update(out.targets)                         # collect the encoded targets

        # nothing produced - empty sample
        if not emb.blocks:
            empty = torch.empty(B, 0, D, device=device)
            return MultiModalTokenSequence(tokens=empty, target_tokens=empty,
                                           token_types=empty.new_empty(B, 0, dtype=torch.long))

        # concat all blocks along channel dim
        z, z_alt, type_ids = self._concat_blocks(emb)  # (B,T,C,D), (B,T,C,D), (B,T,C)

        # add token-type embeddings to every token (before PE/fusion)
        tt_vec = self.ttype_encoder(type_ids)  # (B,T,C,D)  (nn.Embedding handles arbitrary shape)
        z = z + tt_vec
        z_alt = z_alt + tt_vec

        if self.fusion_level == "timestep":
            # fuse per timestep to one token
            z_fused = self._fuse_per_timestep(z)  # (B, T, D)
            z_fused_alt = self._fuse_per_timestep(z_alt)  # (B, T, D)

            # give the fused token its own type id (Composite) and add its embedding
            fused_tt = torch.full((B, T), int(TokenType.Composite), dtype=torch.long, device=z.device)
            mp = emb.encoded_targets.get("marker_present")
            if mp is not None:
                fused_tt.masked_fill_(torch.as_tensor(mp, device=z.device), int(TokenType.Marker))

            # add time positional encoding once
            z_fused = z_fused + self.pos_enc[pos_offset:pos_offset+T].to(z.device)
            z_fused_alt = z_fused_alt + self.pos_enc[pos_offset:pos_offset+T].to(z.device)

            # optional sequence PE over T
            if getattr(self, "pos_encoder", None) is not None and self.config.get("add_seq_pos_after_fuse", True):
                z_fused = self.pos_encoder.encode(z_fused)  # expect (B, T, D) -> (B, T, D)
                z_fused_alt = self.pos_encoder.encode(z_fused_alt)

            return MultiModalTokenSequence(
                tokens=z_fused_alt,  # (B, T, D)
                target_tokens=z_fused,  # (B, T, D)
                token_types=fused_tt,  # (B, T)
                tps=self._pick_target(emb.encoded_targets.get("tps")),
                tts=self._pick_target(emb.encoded_targets.get("tts")),
                seg_types=self._pick_target(emb.encoded_targets.get("segtype")),
                marker_present=mp
            )

        else:  #  fusion_level == "channel"
            # add time positional enc to every token in its timestep
            z = z + self.pos_enc[:T].view(1, T, 1, z.size(-1)).to(z.device)
            z_alt = z_alt + self.pos_enc[:T].view(1, T, 1, z.size(-1)).to(z.device)

            # flatten
            B, T, C, D = z.shape
            target_tokens = z.reshape(B, T * C, D)
            tokens = z_alt.reshape(B, T * C, D)
            tt_flat = type_ids.reshape(B, T * C)

            # broadcast per-timestep targets across channels
            tps = self._repeat_over_C(emb.encoded_targets.get("tps"), C)
            tts = self._repeat_over_C(emb.encoded_targets.get("tts"), C)
            seg = self._repeat_over_C(emb.encoded_targets.get("segtype"), C)

            # sequence PE over flattened stream
            if getattr(self, "pos_encoder", None) is not None and self.config.get("add_seq_pos_after_flatten", True):
                target_tokens = self.pos_encoder.encode(target_tokens)
                tokens = self.pos_encoder.encode(tokens)

            return MultiModalTokenSequence(
                tokens=tokens,
                target_tokens=target_tokens,
                token_types=tt_flat,
                tps=tps, tts=tts, seg_types=seg,
            )

