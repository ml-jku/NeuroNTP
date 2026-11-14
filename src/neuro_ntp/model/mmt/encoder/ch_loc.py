"""
+----------------------------------------------------------------------------------------------------------------------+
ch_loc.py
EEG electrode location encoder of the multimodal tokenizer

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+
"""
from typing import List, Optional, Union
import torch
import torch.nn as nn
import mne


class ChannelEmbedding(nn.Module):
    def __init__(self, input_dim: int = 3, embedding_dim: int = 1024):
        super().__init__()
        self.proj = nn.Linear(input_dim, embedding_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  # (..., 3) -> (..., D)


class ChannelLocationEncoder(nn.Module):
    """
    EEG channel-location encoder:
      - Pulls 3D coords from an MNE montage (default: standard_1005).
      - Projects coords (x,y,z) -> embedding dim.
      - Uses a zero vector for unknown channels.
    Channel names are expected in canonical form; a type prefix such as 'EEG ' is stripped.
    """

    MAP_1005_ALIASES = {"O9": "I1", "O10": "I2"} # 10–20 label → 10–05 name

    def __init__(
        self,
        embedding_dim: int = 1024,
        channel_list: Optional[List[str]] = None,          # default: all channels in the montage
        montage_name: str = "standard_1005",
        ch_map: Optional[dict] = MAP_1005_ALIASES,         # e.g. {"O9": "I1", "O10": "I2"}
        default_coord: Optional[torch.Tensor] = None,      # (3,), default zeros
        coord_abl: Optional[bool] = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.montage_name = montage_name
        self.ch_map = ch_map or {}
        self.coord_abl = coord_abl

        # Load montage once
        montage = mne.channels.make_standard_montage(montage_name)
        ch_pos = montage.get_positions()["ch_pos"]  # only channel positions (fiducials are separate)

        # Build uppercase name -> coord dict for fast lookup
        self._coords_dict = {
            name.upper(): torch.tensor(coord, dtype=torch.float32)
            for name, coord in ch_pos.items()
        }

        # Default channel list = all channels from montage (sorted for stability)
        if channel_list is None:
            channel_list = sorted(ch_pos.keys(), key=str.upper)
        self.channel_list = list(channel_list)

        # Default coord (for unknowns)
        if default_coord is None:
            default_coord = torch.zeros(3, dtype=torch.float32)
        self.register_buffer("default_coord", default_coord, persistent=False)

        # Precompute a coord LUT (C,3) for the canonical list
        coords = [self._lookup_coord(n) for n in self.channel_list]
        self.register_buffer("coord_lut", torch.stack(coords, dim=0))  # (C,3)

        # 3 -> D projection
        self.embed = ChannelEmbedding(3, embedding_dim)

    # ---------- helpers ----------
    def _normalize_name(self, name: str) -> str:
        nm = self.ch_map.get(name, name)
        if " " in nm:
            nm = nm.split()[-1]
        return nm.upper()

    def _lookup_single(self, name: str) -> torch.Tensor:
        key = self._normalize_name(name)
        t = self._coords_dict.get(key, self.default_coord)
        return t.to(self.default_coord.device)

    def _lookup_coord(self, name: str) -> torch.Tensor:
        """
        Coordinates of a channel; for bipolar names (e.g. 'FP1-F7'), the midpoint of the two electrodes.
        """
        # bipolar?
        if "-" in name:
            left, right = name.split("-", 1)
            c1 = self._lookup_single(left)
            c2 = self._lookup_single(right)
            # if either is default, just return the other; if both default, default
            if torch.allclose(c1, self.default_coord) and torch.allclose(c2, self.default_coord):
                return self.default_coord
            if torch.allclose(c1, self.default_coord):
                return c2
            if torch.allclose(c2, self.default_coord):
                return c1
            return (c1 + c2) / 2.0
        # single electrode
        return self._lookup_single(name)

    @torch.no_grad()
    def coords_for(self, names: List[str]) -> torch.Tensor:
        """Return raw coords for a given list of names: (C_sel, 3) on the module's device."""
        device = next(self.parameters()).device
        ts = [self._lookup_coord(n).to(device=device, dtype=torch.float32) for n in names]
        out = torch.stack(ts, dim=0).to(device)
        return out

    # ---------- forward ----------
    def forward(
        self,
        channels: Optional[Union[List[str], List[List[str]]]] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Returns:
          - channels is None:
              (C,D) for canonical list; if batch_size given, (B,C,D) via broadcast.
          - channels is List[str]:
              (C_sel, D) for those names (order preserved).
          - channels is List[List[str]]:
              (B, C_sel, D) stacked per sample (inner lists must be equal length).
        """
        device = next(self.parameters()).device

        if channels is None:
            z = self.embed(self.coord_lut.to(device))  # (C,D)
            if batch_size is not None:
                z = z.unsqueeze(0).expand(batch_size, -1, -1)  # (B,C,D)
            return z

        if len(channels) == 0:
            return torch.empty(0, self.embedding_dim, device=device)

        # List[List[str]] (batched)
        if isinstance(channels[0], (list, tuple)):
            coords_per = [self.coords_for(names) for names in channels]  # each (C_sel,3)
            coords = torch.stack(coords_per, dim=0)                      # (B,C_sel,3)
            return self.embed(coords)                                    # (B,C_sel,D)

        # List[str]
        coords = self.coords_for(channels)                                # (C_sel,3)
        return self.embed(coords)                                         # (C_sel,D)