"""
+----------------------------------------------------------------------------------------------------------------------+
signals.py
Biosignal encoders of the multimodal tokenizer

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Signal channels are encoded by an STFT front end and a convolutional stem shared by all channels, followed by
    a shared projection with a low-rank (LoRA) residual per channel. Per-channel heads are created on demand and
    can be initialised from the nearest known channel (EEG electrode position, smell sensor index). Channel tokens
    are enriched with EEG electrode location and channel subtype embeddings; optional channel dropout is applied
    before encoding.
+----------------------------------------------------------------------------------------------------------------------+
"""
from typing import Optional, List, ClassVar, Dict, Union
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.checkpoint as cp
import math

from neuro_ntp.model.mmt.core.mmt_structs import MMTSample, ModalityEncoder, ModalityOutput
from neuro_ntp.model.mmt.core.token_type import NAME_BY_LOWER, TokenType, has_subtypes, subtype_id, decode_token_types

class STFT(nn.Module):
    """
    Batched STFT module.
    args:
      x: (N, S, L) or (N, L) -> returns |STFT| as (N, S, F, T) or (N, 1, F, T)
    """
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int = None,
        freq_range: tuple = None,    # (f0, f1) or None
        morlet: bool = False,        # placeholder flag
        normalized: bool = True,
        center: bool = True,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length or n_fft)
        self.freq_range = freq_range
        self.morlet = bool(morlet)
        self.normalized = bool(normalized)
        self.center = bool(center)
        self.onesided = bool(onesided)

        # window as buffer so it follows device/dtype
        # (swap with a Morlet window implementation if added)
        win = torch.hann_window(self.win_length)
        self.register_buffer("window", win)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, S, L) or (N, L)
        returns magnitude STFT: (N, S, F, T)
        """
        if x.dim() == 2:
            # (N, L) -> (N, 1, L)
            x = x.unsqueeze(1)
        assert x.dim() == 3, f"expected (N,S,L) or (N,L), got {tuple(x.shape)}"

        N, S, L = x.shape
        window = self.window.to(device=x.device, dtype=x.dtype)

        # flatten channels and do a single batched STFT: (N*S, L) -> (N*S, F, T)
        x_flat = x.reshape(N * S, L)
        spec = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )

        # optional frequency cropping
        if self.freq_range is not None:
            f0, f1 = self.freq_range
            spec = spec[:, f0:f1, :]

        # magnitude and reshape back: (N, S, F, T)
        spec = spec.abs().reshape(N, S, spec.size(-2), spec.size(-1))
        return spec


class ResBlock(nn.Module):
    """
    2D residual block (Conv-BN-ReLU -> Conv-BN) + optional downsample path,
    optional pooling and dropout. Supports tuple stride for time-only downsample.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False, p_drop=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # downsample path to match (C,H,W) when needed
        self.downsample = None
        if downsample or (in_channels != out_channels) or (isinstance(stride, tuple) and stride != (1,1)) or (isinstance(stride, int) and stride != 1):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        # optional pooling after residual add
        self.pooling = pooling
        self.maxpool = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0)  # time-only pool by default

        self.dropout = nn.Dropout(p_drop) if p_drop and p_drop > 0 else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out


class SharedSignalStem(nn.Module):
    """
    Shared 2D CNN feature extractor applied per-channel:
      spec: (N, S, F, T) -> (N, S, C_low, F', T')
    Uses a stack of ResBlocks; downsampling is time-only by default.
    """
    def __init__(self,
                 out_ch=64,                 # final channel width (C_low)
                 num_layers=3,
                 init_ch=32,
                 width_mult=2,              # growth per layer until out_ch
                 downsample_every=2,        # stride at every k-th block
                 pool_after_downsample=True,
                 p_drop=0.0):
        super().__init__()
        layers = []
        in_c = 1
        c = init_ch

        for i in range(num_layers):
            # set next width; clamp to out_ch, enforce last block -> out_ch
            out_c = min(c, out_ch) if i < num_layers - 1 else out_ch

            # apply time-only stride every k-th block (except maybe the first)
            do_down = (downsample_every > 0 and (i % downsample_every == downsample_every - 1))
            stride = (1, 2) if do_down else 1

            layers.append(
                ResBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    stride=stride,
                    downsample=do_down or (in_c != out_c),  # make residual path match when stride or channels change
                    pooling=(pool_after_downsample and do_down),
                    p_drop=p_drop,
                )
            )
            in_c = out_c
            c = min(out_c * width_mult, out_ch)  # grow until out_ch

        self.stem = nn.Sequential(*layers)

    def forward(self, spec):  # spec: (N, S, F, T)
        assert spec.dim() == 4, f"expected (N,S,F,T), got {tuple(spec.shape)}"
        N, S, F, T = spec.shape
        x = spec.reshape(N * S, 1, F, T)      # per-channel, shared weights

        # --- activation checkpointing over the ResBlocks ---
        y = x
        # activation checkpointing during training only (no recomputation in evaluation)
        use_ckpt = self.training
        if use_ckpt:
            for blk in self.stem:
                # each blk: nn.Module taking (N*S, C, F, T) -> (N*S, C', F', T')
                y = cp.checkpoint(blk, y, use_reentrant=False)
        else:
            # normal forward (no recompute on backward)
            for blk in self.stem:
                y = blk(y)

        C_low, Fp, Tp = y.shape[1:]
        return y.view(N, S, C_low, Fp, Tp)    # (N, S, C_low, F', T')


class ChannelLoRAHead(nn.Module):
    """
    Per-channel LoRA head that adds a low-rank residual to the shared 1x1 projection.
    Forward expects *pre-adapted* features of shape (N*S, BOTTLE, H, W)
    and the shared base weight W of shape (D, BOTTLE).
    """
    def __init__(self, bottleneck: int, out_dim: int, rank: int = 4):
        super().__init__()
        self.bottleneck = bottleneck
        self.out_dim = out_dim
        self.rank = rank

        # LoRA parameters: A (out_dim x rank), B (rank x bottleneck)
        self.A = nn.Parameter(torch.zeros(out_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, bottleneck))
        # init: B like a conv, A zeros so residual starts as no-op
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        nn.init.zeros_(self.A)

    def forward(self, y_pre: torch.Tensor, shared_W: torch.Tensor) -> torch.Tensor:
        """
        y_pre:   (N*S, bottleneck, H, W)   # pre-adapted by the shared pre-conv+relu
        shared_W:(out_dim, bottleneck)     # shared 1x1 projection (no bias)
        returns: (N*S, out_dim, H, W)
        """
        # build full weight: W + A @ B   -> (out_dim, bottleneck)
        full = shared_W + self.A @ self.B
        return F.conv2d(y_pre, full.unsqueeze(-1).unsqueeze(-1), bias=None)


class SignalCollectiveEncoder(ModalityEncoder):
    def __init__(self, member_family: str, embedding_dim: int, config: dict, mmt=None):
        super().__init__()
        self.member_family = member_family
        self.embedding_dim = embedding_dim
        self.config = config

        self.stft = STFT(
            n_fft=self.config['sigenc_fft'],
            hop_length=self.config['sigenc_hoplen'],
            win_length=self.config['sigenc_fft'],
            freq_range=self.config['sigenc_freq_range'],
            morlet=self.config['sigenc_morlet'],
        )

        self.shared_stem = SharedSignalStem(
            out_ch=config['sigenc_stem_out'],
            num_layers=config.get('sigenc_stem_layers', 3),
            init_ch=config.get('sigenc_stem_init_ch', 32),
            width_mult=config.get('sigenc_stem_width_mult', 2),
            downsample_every=config.get('sigenc_stem_down_every', 2),
            pool_after_downsample=True,
            p_drop=config.get('sigenc_stem_dropout', 0.2),
        )

        # shared pre-adapter (once for all channels)
        self.shared_pre = nn.Sequential(
            nn.Conv2d(config['sigenc_stem_out'], config['sigenc_bottleneck'], kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # shared base 1x1 projection weight W (D x bottleneck)
        D = config['embedding_dim']
        BOTTLE = config['sigenc_bottleneck']
        self.register_parameter("shared_W", nn.Parameter(torch.empty(D, BOTTLE)))
        nn.init.kaiming_uniform_(self.shared_W, a=math.sqrt(5))

    def set_adaptive_bn(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train()

    def encode(self, mmt, msample: MMTSample) -> Optional[ModalityOutput]:
        if not mmt.config.get("add_signals", True):
            return None

        B, T, D = msample.batch_size, msample.seq_len, self.embedding_dim
        ch_names = msample.signal_channels
        C = len(ch_names)
        device = msample.device

        # channel dropout (pre-encoding)
        p_drop = float(mmt.config.get("signals_channel_drop_prob", 0.0))  # e.g., 0.15
        train_only = bool(mmt.config.get("signals_channel_drop_train_only", True))
        min_keep = int(mmt.config.get("signals_channel_drop_min_keep", 1))
        never_drop_ = {s.strip().lower() for s in mmt.config.get("signals_channel_never_drop", [])}

        if p_drop > 0.0 and (self.training or not train_only) and C > 0:
            drop = (torch.rand(C, device=device) < p_drop)

            # protect "never drop" channels by name
            if never_drop_:
                for i, nm in enumerate(ch_names):
                    if nm.lower() in never_drop_:
                        drop[i] = False

            # ensure at least min_keep survive
            kept = (~drop).sum().item()
            if kept < max(1, min_keep):
                need = max(1, min_keep) - kept
                # randomly un-drop some of the dropped ones
                cand = torch.nonzero(drop, as_tuple=False).squeeze(1)
                if cand.numel() > 0:
                    sel = cand[torch.randperm(cand.numel(), device=device)[:need]]
                    drop[sel] = False

            keep_mask = ~drop
            keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(1).tolist()
        else:
            keep_idx = list(range(C))

        # subset channels we will actually encode
        active_ch = [ch_names[i] for i in keep_idx]

        # process STFT for kept channels
        SD = msample.signal.shape[3]
        keep = torch.as_tensor(keep_idx, device=device)

        sig_keep = msample.signal[:, keep, :, :]  # (B, C_keep, T, SD)
        x_btcl = sig_keep.permute(0, 2, 1, 3).reshape(B * T, len(keep_idx), SD)  # (B*T, C_keep, SD)

        spectr = self.stft(x_btcl)  # (B*T, C_keep, F, Tspec)

        # pass through the shared signal feature extractor
        features = self.shared_stem(spectr)  # (B*T, C_keep, C_low, F', T')

        # per-channel encoders (only for kept channels)
        z_list = []
        for i, ch_name in enumerate(active_ch):
            # select one channel’s shared-stem map: (B*T, C_low, F', T')
            y_c = features[:, i, :, :, :]  # drop the channel dim
            # shared pre-adapter: (B*T, bottleneck, F', T')
            y_pre = self.shared_pre(y_c)

            # tiny per-channel LoRA head
            head = mmt.encoders.get(self.member_family, ch_name)  # ChannelLoRAHead
            head.to(y_pre.device, dtype=y_pre.dtype)  # ensure device/dtype

            # project with shared W + LoRA residual -> (B*T, D, F', T'), then GAP
            y_proj = head(y_pre, self.shared_W)  # conv1x1 with (W + A@B)
            z_c = y_proj.mean(dim=(2, 3)).view(B, T, 1, D)  # (B, T, 1, D)
            z_list.append(z_c)

        if len(z_list) == 0:  # extreme edge case: nothing kept
            # Return an empty block of shape (B,T,0,D)
            empty = torch.empty(B, T, 0, D, device=device)
            return ModalityOutput(block=empty, block_alt=empty, token_types=[])

        z = torch.cat(z_list, dim=2)  # (B,T,C_kept,D)

        # add per-channel location/subtype embeddings
        z = self._augment_channel_embeddings_with_names(mmt, active_ch, z)

        # no masking of signal tokens: the altered path is identical
        z_alt = z

        # per-channel token types (for kept channels only)
        types_per_ch = decode_token_types(active_ch)

        return ModalityOutput(block=z, block_alt=z_alt, token_types=types_per_ch)


    # helpers
    def _augment_channel_embeddings_with_names(self, mmt, ch_names: List[str], z: torch.Tensor) -> torch.Tensor:
        """
        Add per-channel embeddings (EEG location + generic subtypes) and broadcast over (B,T).
        z: (B, T, C, D), len(ch_names) == C
        """
        device, (B, T, C, D) = z.device, z.shape
        assert len(ch_names) == C, "Channel list must match z's C dimension"

        ext = torch.zeros(C, D, dtype=z.dtype, device=device)

        # EEG electrode locations
        if mmt.config.get("add_eeg_location", False):
            eeg_idx = [i for i, nm in enumerate(ch_names) if nm.lower().startswith("eeg ")]
            if eeg_idx:
                eeg_names = [ch_names[i] for i in eeg_idx]
                loc = mmt.encoders.get("eeg_location")(eeg_names)  # (len, D)
                idx = torch.as_tensor(eeg_idx, device=ext.device, dtype=torch.long)
                loc = loc.to(device=ext.device, dtype=ext.dtype)
                ext[idx] = loc

        # Generic subtypes (covers EOG/EMG/ECG, ACC/Movement/Magnet/Gyro axes, PPG/NIRS bands, Smell c00..c63, etc.)
        if mmt.config.get("add_channel_subtypes", True):  # one master flag
            pending: dict[TokenType, dict[str, list]] = {}
            for i, nm in enumerate(ch_names):
                parts = nm.split(" ", 1)
                base = NAME_BY_LOWER.get(parts[0].lower())
                if base is None or not has_subtypes(base):
                    continue
                sub = parts[1].strip() if len(parts) > 1 else None
                sid = subtype_id(base, sub)  # maps unknowns to 'unk'
                d = pending.setdefault(base, {"idxs": [], "sid": []})
                d["idxs"].append(i)
                d["sid"].append(sid)

            for base, d in pending.items():
                emb_mod = mmt.encoders.get("ch_subtype_" + base.name.lower())  # nn.Embedding(len(vocab), D)
                ids = torch.tensor(d["sid"], dtype=torch.long, device=device)
                ext.index_copy_(0, torch.as_tensor(d["idxs"], device=device), emb_mod(ids))

        return z + ext.view(1, 1, C, D)


def make_nearest_init_fn(registry, montage_name: str = "standard_1005"):
    montage = mne.channels.make_standard_montage(montage_name)
    coords = {ch.lower(): np.asarray(pos, dtype=np.float32)
              for ch, pos in montage.get_positions()["ch_pos"].items()}
    if "i1" in coords:
        coords.setdefault("o9", coords["i1"])
    if "i2" in coords:
        coords.setdefault("o10", coords["i2"])

    def nearest_init_fn(channel_id: str):
        # EEG nearest corresponding channel - for warm-init
        if channel_id.startswith("eeg "):
            cid = channel_id.split()[1].lower()
            if cid not in coords:
                return None
            target = coords[cid]
            best_id, best_dist = None, float("inf")
            for known in registry.known_ids():
                if known.startswith("eeg "):
                    k = known.split()[1].lower()
                    if k in coords:
                        d = np.linalg.norm(coords[k] - target)
                        if d < best_dist:
                            best_id, best_dist = known, d
            if best_id is not None:
                return registry.modules_by_id[best_id].state_dict()

        # SMELL nearest channel - for warm-init
        elif channel_id.startswith("smell "):
            cid = channel_id.split()[1].lower()[1:]
            if not ("00" <= cid <= "63"):
                return None
            best_id, best_dist = None, int("inf")
            for known in registry.known_ids():
                if known.startswith("smell "):
                    k = known.split()[1].lower()[1:]
                    dist = abs(int(k) - int(cid))
                    if dist < best_dist:
                        best_id, best_dist = known, dist
            if best_id is not None:
                return registry.modules_by_id[best_id].state_dict()

        return None

    return nearest_init_fn



