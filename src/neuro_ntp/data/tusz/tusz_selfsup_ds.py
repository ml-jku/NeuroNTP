"""
+----------------------------------------------------------------------------------------------------------------------+
tusz_selfsup_ds.py
Self-supervised dataset view of the TUH EEG Seizure Corpus (TUSZ)

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Self-supervised dataset view on the TUSZ core dataset (TUSZCoreDataset), used for pretraining.

    - Samples are built session by session from the stitched session-level HDF5 files (tusz2hdf.py).
    - Samples are drawn by priority sampling.
    - Markers are used as text (e.g. seizure types, including "bckg") and embedded downstream with a frozen
      sentence-BERT model.
    - Sample normalisation: per-channel mean subtraction and scaling by the 95th percentile of |x| (torch).

Expected core dataset tables:
    segments_df    patient_id, session_id, split, start_ts, end_ts, duration, seg_type, epoch_start_ts, epoch_end_ts,
                   time_to_prev_ictal, time_to_next_ictal, followed_by, filename
    session_df     patient_id, session_id, split, measurement_start_time, filename, sfreq, n_channels, signal_length,
                   duration_sec, n_segments, channels, n_seizures, units, n_events_total, n_seizure_events,
                   n_nonseizure_events, n_markers, n_seiz_intervals, ictal_duration, non_ictal_duration,
                   total_duration_sec, ictal_ratio
    patients_df    patient_id, n_sessions, total_duration_sec, n_markers, n_seiz_intervals, ictal_duration,
                   non_ictal_duration, ictal_ratio
+----------------------------------------------------------------------------------------------------------------------+
"""
from __future__ import annotations

import os
import json
import yaml
import random
import hashlib
from itertools import islice, combinations

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from neuro_ntp.misc.io import load_partial_hdf5
from neuro_ntp.data.epi_dataset import EpilepsiaDatasetView, EpiSelfSupervisedSample
from neuro_ntp.data.tusz.tusz_dataset import TUSZCoreDataset

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Iterable
from tqdm import tqdm


class TUSZSelfsupervisedDatasetView(EpilepsiaDatasetView):
    """
    KUK-style self-supervised dataset view for TUSZ.
    """

    DEFAULT_CFG = {
        "sampling_strategy": "pretraining",     # pretraining | downstream
        "split": "tusz_predefined",     # tusz_predefined | temporal | patient | patient_prop | random
        "split_ratio": [0.6, 0.15, 0.25],
        "dev_patients_keep": 30,
        "split_seed": 42,

        "sampling_freq": 256.0,

        "seq_len": 30,
        "chunk_size": 10.0,
        "overlap": 0.0,
        "sliding_window": False,                # if True: advance by stride_s, else by full sample span

        'max_ts': 1000000000,  # maximum timestamp value

        # keep fixed 17ch order
        "signal_channels": [
            'eeg fp1', 'eeg fp2',
            'eeg f7',  'eeg f3', 'eeg fz', 'eeg f4', 'eeg f8',
            'eeg c3',  'eeg cz', 'eeg c4',
            'eeg p3',  'eeg pz', 'eeg p4',
            'eeg o1',  'eeg o2',
            'eeg t1',  'eeg t2',
            'eeg t3',  'eeg t4', 'eeg t5', 'eeg t6',
            'eeg a1',  'eeg a2',
            'eeg ekg1'
          ],
        "min_num_channels": 17,

        # session selection knobs
        "omit_nonictal_sessions": False,
        "omit_nonictal_patients": False,         # keep only patients with >=1 seizure interval

        # if you want to respect TUSZ source split (train/dev/eval)
        "use_source_split": True,
        "map_split": {"train": "train", "dev": "val", "eval": "test"},

        "augment": True
    }

    TUSZ_MARKER_MAP = {
        "bckg": "background",
        "seiz": "seizure (unspecified)",
        "fnsz": "focal non-specific seizure",
        "gnsz": "generalized non-specific seizure",
        "spsz": "simple partial seizure",
        "cpsz": "complex partial seizure",
        "absz": "absence seizure",
        "tnsz": "tonic seizure",
        "tcsz": "tonic-clonic seizure",
        "mysz": "myoclonic seizure",
    }

    # --- predefined patient split - from the Temple University Hospital dataset (TUSZ) ---
    TUSZ_TEST_PATIENTS = {
        "aaaaaaaq", "aaaaaefp", "aaaaahlk", "aaaaaijh", "aaaaapli", "aaaaaqek", "aaaaaqmo", "aaaaaaqvx", "aaaaaardf",
        "aaaaarml",
        "aaaaarpv", "aaaaasip", "aaaaasoc", "aaaaatao", "aaaaatdt", "aaaaaamc", "aaaaafcf", "aaaaahsi", "aaaaajdj",
        "aaaaapsl",
        "aaaaqfm", "aaaaaqtq", "aaaaaqxr", "aaaaarei", "aaaaarnq", "aaaaarto", "aaaaasjz", "aaaaasvq", "aaaaatba",
        "aaaaaarq",
        "aaaaaghb", "aaaaaiih", "aaaaajru", "aaaaqbi", "aaaaqld", "aaaaaqtw", "aaaaaraf", "aaaaarjc", "aaaaarpu",
        "aaaaasfw",
        "aaaaasnu", "aaaaaswc", "aaaaatdq"
    }

    TUSZ_DEV_PATIENTS = {
        "aaaaaajy", "aaaaabep", "aaaaacqc", "aaaaadkj", "aaaaafjv", "aaaaagus", "aaaaahsn", "aaaaaiup", "aaaaakfk",
        "aaaaamck",
        "aaaaamod", "aaaaamqw", "aaaaandx", "aaaaaoiz", "aaaaaovm", "aaaaappr", "aaaaqrs", "aaaaaskf", "aaaaaayf",
        "aaaaabxe",
        "aaaaacyi", "aaaaaewf", "aaaaagbf", "aaaaahie", "aaaaaict", "aaaaaizb", "aaaaakis", "aaaaamnd", "aaaaampk",
        "aaaaamta",
        "aaaaaoec", "aaaaaool", "aaaaaoxa", "aaaaaptf", "aaaaarsp", "aaaatgc", "aaaaaazz", "aaaaaclc", "aaaaadkb",
        "aaaaaexe",
        "aaaaagdr", "aaaaahln", "aaaaaiij", "aaaaakcy", "aaaalhp", "aaaaamnk", "aaaaamqq", "aaaaamva", "aaaaaoek",
        "aaaaaopm",
        "aaaaaoys", "aaaaqbt", "aaaaasgd"
    }

    SUPPORTED_TOKEN_TYPES = ["EEG", "Marker", "Segtype", "TPS", "TTS", "Notes"]

    def __init__(
        self,
        name: str = "TUSZUnsupervisedView",
        cfg_filepath: str | None = None,
        core_ds: "TUSZCoreDataset" = None,
        dv_filepath: str | None = None,
        cfg: dict | None = None,
        use_cache: bool = False,
        mode: str = "unsupervised",
        only_last_n: int | None = None,
        test_only: bool = False,
        n_jobs: int = 8,
        precompute_stats: bool = True,
    ):
        super().__init__(name, core_ds, dv_filepath, cfg or self.DEFAULT_CFG, use_cache)

        self.mode = mode
        self.only_last_n = only_last_n
        self.test_only = test_only
        self.n_jobs = n_jobs
        self.precompute_stats = precompute_stats

        # KUK-style priority sampling flags
        self.priority_sampling = False
        self._bad_indices = set()

        if self.dv_filepath:
            self.__load()
            print(f"use cache: {use_cache}")
        else:
            if cfg_filepath:
                print("...reading dataset view configuration from file...")
                self.load_cfg(cfg_filepath)

            print(f"...creating dataset view {self.name} based on {self.core_ds.name}")

            self.samples_df = pd.DataFrame(
                columns=[
                    "sample_id",
                    "patient_id",
                    "session_id",
                    "timestamp",
                    "filename",
                    "start_index",
                    "end_index",
                    "split",
                    "source_split",
                ]
            )
            self.chunks_df = pd.DataFrame(
                columns=[
                    "sample_id",
                    "timestep",
                    "seg_type",
                    "markers",
                    "label",
                    "time_past_seizure",
                    "time_to_seizure",
                    "notes",
                ]
            )

            self.update_tps_tts_tusz(unknown_ts=float(self.cfg["max_ts"]))
            self.retrieve_samples()
            self.split_samples()

            if self.test_only:
                self.samples_df = self.samples_df[self.samples_df['split'] == 'test'].copy()
                self.chunks_df = self.chunks_df[
                    self.chunks_df['sample_id'].isin(self.samples_df['sample_id'].unique())].copy()

            self.samples_df.reset_index(drop=True, inplace=True)
            self.chunks_df.reset_index(drop=True, inplace=True)

            if self.precompute_stats:
                self.samples_df = self.precompute_norm_stats(
                    self.samples_df,
                    cfg=self.cfg,
                    base_dir="",  # set if needed
                    n_jobs=self.n_jobs,  # or 0/1 for serial
                    verbose=True,
                )

        self._init_split_views()

    # ----------------------------------------------------------------------------------
    # Config I/O
    # ----------------------------------------------------------------------------------
    def dump_cfg(self, cfg_filepath):
        with open(cfg_filepath, "w") as f:
            yaml.dump(self.cfg, f)

    def load_cfg(self, cfg_filepath):
        with open(cfg_filepath, "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

    # ----------------------------------------------------------------------------------
    # Save / load
    # ----------------------------------------------------------------------------------
    def save(self, dv_filepath: str):
        print(f"...saving dataset view {self.name} to {dv_filepath}")
        self.__set_explicit_types()
        with pd.HDFStore(dv_filepath, "w") as hfile:
            hfile.put("params", pd.DataFrame())
            hfile.get_storer("params").attrs["name"] = self.name
            hfile.get_storer("params").attrs["core_ds_filepath"] = self.core_ds.ds_filepath
            hfile.put("config", pd.Series(json.dumps(self.cfg)))
            hfile.put("samples", self.samples_df)
            hfile.put("chunks", self.chunks_df)
            hfile.put("marker_vocab", getattr(self, "marker_vocab", pd.DataFrame([""], columns=[0])))

    def __load(self):
        print(f"...loading dataset view {self.name} from {self.dv_filepath}")
        with pd.HDFStore(self.dv_filepath, "r") as hfile:
            self.name = hfile.get_storer("params").attrs["name"]
            self.core_ds_filepath = hfile.get_storer("params").attrs["core_ds_filepath"]
            self.cfg = json.loads(hfile["config"].item())
            self.samples_df = hfile["samples"]
            self.chunks_df = hfile["chunks"]
            if "marker_vocab" in hfile:
                self.marker_vocab = hfile["marker_vocab"]
            else:
                self.marker_vocab = pd.DataFrame([""], columns=[0])

        self.core_ds = TUSZCoreDataset(ds_filepath=self.core_ds_filepath)

    # ----------------------------------------------------------------------------------
    # Split views
    # ----------------------------------------------------------------------------------
    def _init_split_views(self):
        self.split_views = {
            split: self.samples_df.loc[self.samples_df["split"] == split].reset_index(drop=True)
            for split in ["train", "val", "test"]
        }

    # ----------------------------------------------------------------------------------
    # KUK-style normalization (torch)
    # ----------------------------------------------------------------------------------
    def normalize_torch(self, X: torch.Tensor) -> torch.Tensor:
        # X: (C, T) or (C, T, ...)
        X = X - X.mean(dim=-1, keepdim=True)
        q95 = torch.quantile(torch.abs(X), q=0.95, dim=-1, keepdim=True)
        return X / (q95 + 1e-8)

    @staticmethod
    def canonicalize_and_expand_marker_txt(marker_txt: str) -> str:
        if not marker_txt:
            return ""

        # tokenize, normalize
        toks = [t.strip().lower() for t in marker_txt.split(",") if t.strip()]
        if not toks:
            return ""

        # stable unique (preserve order)
        toks = list(dict.fromkeys(toks))

        # if 'seiz' co-occurs with any other label, drop it
        if "seiz" in toks and len(toks) > 1:
            toks = [t for t in toks if t != "seiz"]

        # if 'bckg' co-occurs with anything else, drop it
        if "bckg" in toks and any(t != "bckg" for t in toks):
            toks = [t for t in toks if t != "bckg"]

        # expand to full names, keep code in brackets
        expanded = []
        for t in toks:
            if t in TUSZSelfsupervisedDatasetView.TUSZ_MARKER_MAP:
                expanded.append(f"{TUSZSelfsupervisedDatasetView.TUSZ_MARKER_MAP[t]} ({t})")
            else:
                # fallback: keep raw token if unknown
                expanded.append(t)

        return ", ".join(expanded)

    # ----------------------------------------------------------------------------------
    # Sampling logic
    # ----------------------------------------------------------------------------------
    @staticmethod
    def prepare_session_samples(session_row, segments_df, markers_df, cfg):
        """
        Process a single session row, producing:
          - samples_list: one row per contiguous sample window
          - chunks_list:  one row per timestep in each sample

        Uses numeric start/end in seconds (start_ts/end_ts).
        Computes TPS/TTS robustly from seizure onset/offset arrays per session.
        """
        import hashlib
        import numpy as np
        import pandas as pd

        chunk_size = float(cfg["chunk_size"])
        overlap = float(cfg["overlap"])
        sfreq = float(cfg["sampling_freq"])
        seq_len = int(cfg["seq_len"])
        unknown_ts = float(cfg.get("unknown_ts", 1e9))
        augment = cfg.get("augment", False)
        aug_shifted = False

        pid = session_row.patient_id
        sid = session_row.session_id
        filename = str(session_row.filename)
        source_split = getattr(session_row, "split", None)
        t0 = getattr(session_row, "measurement_start_time", pd.NaT)
        total_dur = float(session_row.duration_sec)

        segs = segments_df[(segments_df["patient_id"] == pid) & (segments_df["session_id"] == sid)]
        if segs.empty:
            return [], []

        # --- build session-level seizure onset/offset arrays in seconds ---
        ictal_segs = segs[segs["seg_type"] == "ictal"]
        if not ictal_segs.empty:
            ictal_starts = np.sort(ictal_segs["start_ts"].to_numpy(dtype=np.float64))
            ictal_ends = np.sort(ictal_segs["end_ts"].to_numpy(dtype=np.float64))
        else:
            ictal_starts = np.array([], dtype=np.float64)
            ictal_ends = np.array([], dtype=np.float64)

        # ensure markers_df can be empty (still yield empty marker strings)
        mks = None
        if markers_df is not None and not markers_df.empty:
            mks = markers_df[(markers_df["patient_id"] == pid) & (markers_df["session_id"] == sid)]
            if not mks.empty:
                col = "marker_text" if "marker_text" in mks.columns else (
                    "description" if "description" in mks.columns else "marker"
                )
                mks = mks.copy()
                mks["_desc"] = mks[col].astype(str).str.lower()

        stride_s = chunk_size - overlap
        sample_span_s = stride_s * seq_len + overlap
        slide = bool(cfg.get("sliding_window", False))
        advance_s = stride_s if slide else sample_span_s

        from_ts, to_ts = 0.0, total_dur

        sample_cnt = 0
        ti = from_ts

        samples_list = []
        chunks_list = []

        # select onset col for markers if available
        onset_col = None
        if mks is not None and not mks.empty:
            if "onset" in mks.columns:
                onset_col = "onset"
            elif "onset_sec" in mks.columns:
                onset_col = "onset_sec"

        while ti + sample_span_s < to_ts:
            # sid basename already includes pid
            sample_id = f"{sid.split('/')[-1]}__{sample_cnt:08d}"

            # per-sample marker prefilter (optional; helps when marker density is high)
            if mks is not None and not mks.empty and onset_col is not None:
                sample_mks = mks[(mks[onset_col] >= ti) & (mks[onset_col] < (ti + sample_span_s))]
            else:
                sample_mks = mks

            valid = True
            sample_chunks = []
            any_ictal = False

            for i in range(seq_len):
                start = ti + i * stride_s
                end = start + chunk_size

                # validity / coverage check
                stsegs = segs[(segs["end_ts"] > start) & (segs["start_ts"] < end)]
                if stsegs.empty:
                    valid = False
                    break

                # ---- robust TPS/TTS from seizure event arrays ----
                if ictal_starts.size == 0:
                    in_ictal = False
                    tps = unknown_ts
                    tts = unknown_ts
                else:
                    # robust overlap test: find first seizure that ends after chunk start
                    k = np.searchsorted(ictal_ends, start, side="right")
                    in_ictal = (k < ictal_starts.size) and (ictal_starts[k] < end)

                    if in_ictal:
                        # semantics: during ictal, both are 0
                        tps = 0.0
                        tts = 0.0
                    else:
                        # next seizure start after chunk start
                        jn = np.searchsorted(ictal_starts, start, side="left")
                        tts = (ictal_starts[jn] - start) if (jn < ictal_starts.size) else unknown_ts
                        if tts < 0:
                            tts = 0.0  # numerical safety

                        # previous seizure end before chunk start
                        jp = np.searchsorted(ictal_ends, start, side="right") - 1
                        tps = (start - ictal_ends[jp]) if (jp >= 0) else unknown_ts
                        if tps < 0:
                            tps = 0.0  # numerical safety

                seg_type = "ictal" if in_ictal else "non-ictal"
                label = 1 if in_ictal else 0
                if label == 1:
                    any_ictal = True

                # markers for this chunk window
                marker_txt = ""
                if sample_mks is not None and not sample_mks.empty and onset_col is not None:
                    mk = sample_mks[(sample_mks[onset_col] >= start) & (sample_mks[onset_col] < end)]
                    if not mk.empty:
                        marker_txt = ", ".join(sorted(set(mk["_desc"].tolist())))
                marker_txt = TUSZSelfsupervisedDatasetView.canonicalize_and_expand_marker_txt(marker_txt)

                sample_chunks.append(
                    {
                        "sample_id": sample_id,
                        "timestep": i,
                        "seg_type": seg_type,
                        "markers": marker_txt,
                        "label": label,
                        "time_past_seizure": float(tps),
                        "time_to_seizure": float(tts),
                        "notes": "",
                    }
                )

            if not valid:
                ti += advance_s
                sample_cnt += 1
                continue

            # numeric timestamp (optional metadata)
            ts = 0.0
            if not pd.isna(t0):
                try:
                    ts = pd.to_datetime(t0).timestamp()
                except Exception:
                    ts = 0.0

            samples_list.append(
                {
                    "sample_id": sample_id,
                    "patient_id": pid,
                    "session_id": sid,
                    "timestamp": float(ts),
                    "filename": filename,
                    "start_index": int(ti * sfreq),
                    "end_index": int((ti + sample_span_s) * sfreq),
                    "source_split": source_split,
                }
            )
            chunks_list.extend(sample_chunks)

            if augment and aug_shifted:
                ti = ti - ((chunk_size - overlap) // 2) + advance_s
                aug_shifted = False
            elif augment and not aug_shifted and any_ictal:
                aug_shifted = True
                ti += ((chunk_size - overlap) // 2)
            else:
                aug_shifted = False
                ti += advance_s

            sample_cnt += 1

        return samples_list, chunks_list

    def __add_samples_parallel(self, session_df: pd.DataFrame, n_threads: int = 10):
        sess = session_df.reset_index(drop=True)

        with tqdm_joblib(total=len(sess), desc="Processing sessions") as _pbar:
            results = Parallel(n_jobs=n_threads, backend="loky")(
                delayed(TUSZSelfsupervisedDatasetView.prepare_session_samples)(
                    row, self.core_ds.segments_df, getattr(self.core_ds, "markers_df", None), self.cfg
                )
                for row in sess.itertuples(index=False)
            )

        all_samples, all_chunks = [], []
        for samples, chunks in results:
            all_samples.extend(samples)
            all_chunks.extend(chunks)

        if all_samples:
            self.samples_df = pd.concat([self.samples_df, pd.DataFrame(all_samples)], ignore_index=True)
        if all_chunks:
            self.chunks_df = pd.concat([self.chunks_df, pd.DataFrame(all_chunks)], ignore_index=True)

    def retrieve_samples(self):
        """
        Populate samples_df and chunks_df using the core_ds dataframes.
        Markers are used as TEXT (lowercased) from core_ds.markers_df['description'] (fallback: 'marker').
        """
        # marker vocab (for later frozen SBERT embedding)
        mdf = getattr(self.core_ds, "markers_df", None)
        if mdf is not None and not mdf.empty:
            col = "marker_text" if "marker_text" in mdf.columns else (
                "description" if "description" in mdf.columns else "marker")
            vocab = mdf[col].astype(str).str.lower().unique()
            self.marker_vocab = pd.DataFrame(vocab)
        else:
            self.marker_vocab = pd.DataFrame([""], columns=[0])

        filtered = self.core_ds.session_df.copy()

        if self.cfg.get("omit_nonictal_patients", True):
            patseiz = self.core_ds.patients_df[self.core_ds.patients_df["n_seiz_intervals"] > 0]
            filtered = filtered[filtered["patient_id"].isin(patseiz["patient_id"])]

        if self.cfg.get("omit_nonictal_sessions", False):
            filtered = filtered[filtered["n_seiz_intervals"] > 0]

        print(f"...retrieve samples: {len(filtered)} sessions considered")
        self.__add_samples_parallel(filtered, n_threads=10)

        self.samples_df.reset_index(drop=True, inplace=True)
        self.chunks_df.reset_index(drop=True, inplace=True)

    def update_channel_config(self, cfg):
        self.cfg['signal_channels'] = cfg['signal_channels']
        self.supported_channels = {ch.lower() for ch in cfg['signal_channels']}

    import numpy as np
    import pandas as pd

    def update_tps_tts_tusz(self, *, unknown_ts: float = 1e9) -> None:
        """
        Recompute time_to_prev_ictal and time_to_next_ictal in core_ds.segments_df
        using epoch_start_ts/epoch_end_ts in *seconds*.

        Semantics (seconds):
          - time_to_next_ictal: (next ictal onset) - (row epoch_start_ts)
              * 0 for ictal rows
              * unknown_ts if there is no next ictal in the session
          - time_to_prev_ictal: (row epoch_start_ts) - (previous ictal end)
              * 0 for ictal rows
              * unknown_ts if there is no previous ictal in the session

        Important:
          - No clipping to a finite max_ts here.
            Any capping/logspace should happen at encoding time (MMT), not in core labels.
        """
        segdf = self.core_ds.segments_df

        # ensure numeric seconds
        for col in ("epoch_start_ts", "epoch_end_ts", "duration"):
            if col in segdf.columns:
                segdf[col] = pd.to_numeric(segdf[col], errors="coerce")

        # ensure epoch_end_ts exists
        if "epoch_end_ts" not in segdf.columns or segdf["epoch_end_ts"].isna().any():
            if "duration" not in segdf.columns:
                raise ValueError("segments_df missing 'duration' needed to build epoch_end_ts")
            segdf["epoch_end_ts"] = segdf["epoch_start_ts"] + segdf["duration"]

        # allocate with unknown sentinel
        segdf["time_to_prev_ictal"] = float(unknown_ts)
        segdf["time_to_next_ictal"] = float(unknown_ts)

        gb = segdf.groupby(["patient_id", "session_id"], sort=False)

        for (pid, sid), idx in gb.groups.items():
            s = segdf.loc[idx].sort_values("epoch_start_ts", kind="stable")

            # ictal intervals within this session
            ictal = s[s["seg_type"] == "ictal"]
            ictal_starts = np.sort(ictal["epoch_start_ts"].to_numpy(np.float64, copy=True))
            ictal_ends = np.sort(ictal["epoch_end_ts"].to_numpy(np.float64, copy=True))

            row_t = s["epoch_start_ts"].to_numpy(dtype=np.float64, copy=True)

            # ---- TTS: next ictal start relative to row epoch_start ----
            if len(ictal_starts) > 0:
                j = np.searchsorted(ictal_starts, row_t, side="left")
                tts = np.full(len(row_t), float(unknown_ts), dtype=np.float64)
                mask = j < len(ictal_starts)
                tts[mask] = ictal_starts[j[mask]] - row_t[mask]
                # numerical safety (should already be >= 0 for mask)
                tts = np.maximum(tts, 0.0)
            else:
                tts = np.full(len(row_t), float(unknown_ts), dtype=np.float64)

            # ---- TPS: time since last ictal end relative to row epoch_start ----
            if len(ictal_ends) > 0:
                k = np.searchsorted(ictal_ends, row_t, side="right") - 1
                tps = np.full(len(row_t), float(unknown_ts), dtype=np.float64)
                mask = k >= 0
                tps[mask] = row_t[mask] - ictal_ends[k[mask]]
                tps = np.maximum(tps, 0.0)
            else:
                tps = np.full(len(row_t), float(unknown_ts), dtype=np.float64)

            # enforce semantics for ictal rows
            ictal_mask = (s["seg_type"].to_numpy() == "ictal")
            tts[ictal_mask] = 0.0
            tps[ictal_mask] = 0.0

            # write back
            segdf.loc[s.index, "time_to_next_ictal"] = tts
            segdf.loc[s.index, "time_to_prev_ictal"] = tps

        self.core_ds.segments_df = segdf

    @staticmethod
    def canonicalize_marker_txt(marker_txt: str) -> str:
        if not marker_txt:
            return ""

        toks = [t.strip().lower() for t in marker_txt.split(",") if t.strip()]
        if not toks:
            return ""

        # stable unique
        toks = list(dict.fromkeys(toks))

        # if 'seiz' co-occurs with any other label, drop it
        if "seiz" in toks and len(toks) > 1:
            toks = [t for t in toks if t != "seiz"]

        if "bckg" in toks and any(t not in {"bckg"} for t in toks):
            toks = [t for t in toks if t != "bckg"]

        return ", ".join(toks)

    # ----------------------------------------------------------------------------------
    # Splitting
    # ----------------------------------------------------------------------------------
    def split_samples(self):
        split_mode = str(self.cfg.get("split", "temporal")).lower()
        if split_mode in {"tusz_predefined", "predefined", "tusz"}:
            test_pats = set(getattr(self, "TUSZ_TEST_PATIENTS", set())) or set(self.TUSZ_TEST_PATIENTS)
            dev_pats_all = set(getattr(self, "TUSZ_DEV_PATIENTS", set())) or set(self.TUSZ_DEV_PATIENTS)

            # sanity: ensure disjoint
            dev_pats_all = dev_pats_all - test_pats

            # optional dev downsampling
            keep = self.cfg.get("dev_patients_keep", None)
            seed = int(self.cfg.get("split_seed", 42))
            dev_pats = sorted(dev_pats_all)

            if keep is None or keep == 0:
                pass  # keep all dev patients
            else:
                rng = random.Random(seed)
                if isinstance(keep, float):
                    k = int(round(len(dev_pats) * max(0.0, min(1.0, keep))))
                    k = max(1, k) if len(dev_pats) > 0 else 0
                else:
                    k = int(keep)
                    k = max(0, k)
                if k < len(dev_pats):
                    dev_pats = rng.sample(dev_pats, k)

            dev_pats = set(dev_pats)

            # assign splits
            pid = self.samples_df["patient_id"].astype(str)
            self.samples_df["split"] = "train"
            self.samples_df.loc[pid.isin(test_pats), "split"] = "test"
            self.samples_df.loc[(~pid.isin(test_pats)) & (pid.isin(dev_pats)), "split"] = "val"
            self.samples_df["split"] = self.samples_df["split"].astype(str)

            # quick log
            pats = self.samples_df[["patient_id", "split"]].drop_duplicates()
            print("[TUSZ split] unique patients:", pats["patient_id"].nunique())
            print("[TUSZ split] patients per split:\n", pats["split"].value_counts())
            print("[TUSZ split] samples per split:\n", self.samples_df["split"].value_counts())
            return

        # source split mapping (train/dev/eval) if desired
        if self.cfg.get("use_source_split", True) and "source_split" in self.samples_df.columns:
            mp = self.cfg.get("map_split", {"train": "train", "dev": "val", "eval": "test"})
            self.samples_df["split"] = self.samples_df["source_split"].map(mp).fillna("train").astype(str)
            return

        ratio = self.cfg["split_ratio"]

        if self.cfg["split"] == "patient":
            patients = self.samples_df["patient_id"].unique()
            n = len(patients)
            i1, i2 = int(n * ratio[0]), int(n * (ratio[0] + ratio[1]))
            lots = patients[:i1], patients[i1:i2], patients[i2:]
            self.samples_df["split"] = np.where(
                self.samples_df["patient_id"].isin(lots[0]),
                "train",
                np.where(self.samples_df["patient_id"].isin(lots[1]), "val", "test"),
            )

        elif split_mode == "patient_prop":
            vc = self.samples_df["patient_id"].value_counts()
            patients = vc.index.to_numpy()
            patients = sorted(patients, key=lambda p: vc[p], reverse=True)
            np.random.shuffle(patients)

            total = len(self.samples_df)
            target = np.array(ratio) * total
            assigned = {}
            cur = np.array([0, 0, 0])

            for p in patients:
                cnt = int(vc[p])
                split_idx = int(np.argmax(target - cur))
                assigned[p] = ["train", "val", "test"][split_idx]
                cur[split_idx] += cnt

            self.samples_df["split"] = self.samples_df["patient_id"].map(assigned).astype(str)

        elif split_mode == "temporal":
            for p in self.samples_df["patient_id"].unique():
                ps = self.samples_df[self.samples_df["patient_id"] == p].sort_values(
                    ["timestamp", "start_index"], kind="stable"
                )
                n = len(ps)
                i1 = round(n * ratio[0])

                remaining = ps.iloc[i1:]
                shuffled = remaining.sample(frac=1.0, random_state=42)
                n_val = round(len(shuffled) * (ratio[1] / (ratio[1] + ratio[2])))

                val_idx = shuffled.index[:n_val]
                test_idx = shuffled.index[n_val:]

                self.samples_df.loc[ps.index[:i1], "split"] = "train"
                self.samples_df.loc[val_idx, "split"] = "val"
                self.samples_df.loc[test_idx, "split"] = "test"

        else:
            np.random.seed(76)
            idx = np.random.permutation(self.samples_df.index)
            n = len(idx)
            i1, i2 = int(n * ratio[0]), int(n * (ratio[0] + ratio[1]))
            self.samples_df.loc[idx[:i1], "split"] = "train"
            self.samples_df.loc[idx[i1:i2], "split"] = "val"
            self.samples_df.loc[idx[i2:], "split"] = "test"

        self.samples_df["split"] = self.samples_df["split"].astype(str)

    # ----------------------------------------------------------------------------------
    # Length / indexing
    # ----------------------------------------------------------------------------------
    def get_len_type(self, ds_type):
        L = len(self.split_views[ds_type])
        if self.mode == "supervised":
            return L * int(self.cfg["seq_len"])
        elif self.mode == "supervised-last-n":
            return L * int(self.only_last_n or 1)
        return L

    def __len__(self):
        return self.get_len_type("train") + self.get_len_type("val") + self.get_len_type("test") \
            if self.mode in ("supervised", "supervised-last-n") else len(self.samples_df)

    def __getitem__(self, index):
        return self.get_item_of_type(None, index)

    # ----------------------------------------------------------------------------------
    # Channel name normalization (keep previous mapping)
    # ----------------------------------------------------------------------------------
    def _norm_ch_names(self, ch_names: Iterable[str]) -> List[str]:
        out = []
        for ch in ch_names:
            s = " ".join(str(ch).split())
            su = s.upper()

            if su in {"EEG EKG", "EEG EKG1"}:
                out.append("ECG 1")
            elif su.startswith("ECG") or su.startswith("EKG"):
                out.append("ECG 1")
            elif su == "EEG ROC":
                out.append("EOG r")
            elif su == "EEG LOC":
                out.append("EOG l")
            elif su in {"SAO2 SPO2", "SAO2", "SPO2", "O2SAT", "OSAT"}:
                out.append("SpO2")
            else:
                out.append(s)
        return out

    @property
    def signal_channels(self):
        return self.cfg["signal_channels"]

    @property
    def norm_signal_channels(self):
        return self._norm_ch_names(self.cfg["signal_channels"])

    # ----------------------------------------------------------------------------------
    # Robust retry helper
    # ----------------------------------------------------------------------------------
    def _pick_alt_index(self, ds_type, avoid_idx, max_tries=10):
        for _ in range(max_tries):
            j = np.random.randint(0, self.get_len_type(ds_type))
            if j != avoid_idx and (ds_type, j) not in self._bad_indices:
                return j
        return avoid_idx

    # ----------------------------------------------------------------------------------
    # Sampling getters
    # ----------------------------------------------------------------------------------
    def __get_selfsup_sample__(self, ds_type, index, max_tries=5):
        tries = 0
        while tries < max_tries:
            try:
                df = self.split_views[ds_type] if ds_type else self.samples_df
                row = df.iloc[index]

                sample_id = str(row["sample_id"])
                filename = str(row["filename"])
                start_index = int(row["start_index"])
                end_index = int(row["end_index"])

                hdf = load_partial_hdf5(
                    os.path.join("", filename),
                    start_idx=start_index,
                    stop_idx=end_index,
                    only_signal=True,
                    channel_order=self.cfg["signal_channels"],
                )

                #signal = torch.as_tensor(hdf["signal"], dtype=torch.float32)  # (C, T)
                #signal = self.normalize_torch(signal)  # KUK-style normalization

                signal = torch.as_tensor(hdf["signal"], dtype=torch.float32)  # (C, T)

                mean = row.get("norm_mean", None)
                q95 = row.get("norm_q95", None)

                mean_t = None if mean is None else torch.as_tensor(mean, dtype=torch.float32)
                q95_t = None if q95 is None else torch.as_tensor(q95, dtype=torch.float32)

                signal = self.normalize_torch_with_stats(signal, mean=mean_t, q95=q95_t)

                # chunkify via unfold into (C, seq_len, chunk_samples)
                stride = int((self.cfg["chunk_size"] - self.cfg["overlap"]) * self.cfg["sampling_freq"])
                win = int(self.cfg["chunk_size"] * self.cfg["sampling_freq"])
                signal = signal.unfold(dimension=1, size=win, step=stride)

                chdf = (
                    self.chunks_df[self.chunks_df["sample_id"] == sample_id]
                    .sort_values("timestep")
                    .reset_index(drop=True)
                )

                # force numeric dtypes for the numeric columns
                for c in ("time_past_seizure", "time_to_seizure", "label", "timestep"):
                    chdf[c] = pd.to_numeric(chdf[c], errors="coerce").fillna(0.0)

                chunks = chdf[EpilepsiaDatasetView.CHUNK_COLS].to_numpy()

                marker_embs = None
                if hasattr(self, "marker_emb_dict"):
                    fallback = self.marker_emb_dict.get("", torch.zeros(self.marker_emb_dim))
                    mlist = [str(m).strip().lower() for m in chunks[:, 3]]
                    marker_embs = torch.stack([self.marker_emb_dict.get(m, fallback) for m in mlist], dim=0)

                notes_embs = None  # currently unused

                metadata = {
                    "patient": row.get("patient_id", None),
                    "session": row.get("session_id", None),
                    "filename": filename,
                    "timestamp": row.get("timestamp", None),
                    "start_index": start_index,
                    "end_index": end_index,
                    "signal_channels": self._norm_ch_names(self.cfg["signal_channels"]),
                    "modalities": ["signals", "TTS", "segtype", "TPS", "Markers", "EoS"],
                }

                return EpiSelfSupervisedSample(
                    signal=signal, chunks=chunks, markers_embs=marker_embs, notes_embs=notes_embs, metadata=metadata
                )

            except (OSError, IOError, KeyError, RuntimeError) as e:
                self._bad_indices.add((ds_type, index))
                tries += 1
                print(f"[WARN] skipping sample idx={index} ({ds_type}): {e}")
                index = self._pick_alt_index(ds_type, index)

        raise RuntimeError(f"Failed to fetch sample after {max_tries} tries (ds_type={ds_type}, index={index})")

    def get_item_of_type(self, ds_type, index):
        if self.mode == "unsupervised":
            return self.__get_selfsup_sample__(ds_type, index)
        raise AttributeError("Only 'unsupervised' mode is supported in this simplified TUSZ view.")

    # ----------------------------------------------------------------------------------
    # Marker embedding utilities
    # ----------------------------------------------------------------------------------
    @staticmethod
    def batch_iterable(iterable, batch_size):
        iterable = iter(iterable)
        while True:
            chunk = list(islice(iterable, batch_size))
            if not chunk:
                break
            yield chunk

    def initialize_marker_embeddings(self, marker_embedder, marker_emb_dim, batch_size):
        """
        Creates embeddings for all marker strings occurring in chunks_df['markers']
        and also for marker_vocab entries (if present).
        """
        self.marker_emb_dim = int(marker_emb_dim)
        self.marker_embedder = marker_embedder
        self.marker_emb_dict = dict()
        self.vocab_emb_dict = dict()
        self.notes_emb_dict = dict()

        self.marker_embedder.eval()
        with torch.no_grad():
            # occurrences (including empty)
            markers = list(self.chunks_df["markers"].astype(str).unique())
            for mb in self.batch_iterable(markers, batch_size):
                embs = marker_embedder.get_emb(mb).detach().cpu()
                self.marker_emb_dict.update({m: e for m, e in zip(mb, embs)})

            # vocab
            vocab_series = self.marker_vocab.iloc[:, 0] if hasattr(self, "marker_vocab") and not self.marker_vocab.empty else pd.Series([""])
            vocab_list = list(vocab_series.astype(str).unique())
            for vb in self.batch_iterable(vocab_list, batch_size):
                embs = marker_embedder.get_emb(vb).detach().cpu()
                self.vocab_emb_dict.update({m: e for m, e in zip(vb, embs)})

        # ensure "" exists
        if "" not in self.marker_emb_dict:
            self.marker_emb_dict[""] = torch.zeros(self.marker_emb_dim)

    def build_extended_marker_dict(self, batch_size=32):
        if not hasattr(self, "marker_emb_dict"):
            raise RuntimeError("Call initialize_marker_embeddings() first.")

        full_marker_keys = list(self.marker_emb_dict.keys())
        atomic_markers = set()
        pair_markers = set()

        for full_marker in full_marker_keys:
            parts = [m.strip() for m in full_marker.split(",") if m.strip()]
            atomic_markers.update(parts)
            if len(parts) >= 3:
                for pair in combinations(parts, 2):
                    pair_markers.add(", ".join(pair))

        all_candidates = atomic_markers | pair_markers
        to_encode = [m for m in all_candidates if m not in self.marker_emb_dict]

        self.marker_embedder.eval()
        new_marker_embs = dict()
        with torch.no_grad():
            for batch in self.batch_iterable(to_encode, batch_size):
                embs = self.marker_embedder.get_emb(batch).detach().cpu()
                new_marker_embs.update({m: e for m, e in zip(batch, embs)})

        self.extended_marker_emb_dict = {**self.marker_emb_dict, **new_marker_embs}
        print(f"Extended marker dictionary created with {len(self.extended_marker_emb_dict)} entries.")

    # ----------------------------------------------------------------------------------
    # Priority sampling (KUK-style)
    # ----------------------------------------------------------------------------------
    def init_priority_sampling(self, event_frac=0.2, nonevent_frac=0.05, seed=42):
        self.priority_sampling = True
        self.prio_event_frac = event_frac
        self.prio_nonevent_frac = nonevent_frac
        self.random = random.Random(seed)
        self.all_samples_df = self.samples_df.copy()
        self.all_chunks_df = self.chunks_df.copy()

        train_ids = self.all_samples_df[self.all_samples_df['split'] == 'train']['sample_id']
        val_ids = self.all_samples_df[self.all_samples_df['split'] == 'val']['sample_id']
        test_ids = self.all_samples_df[self.all_samples_df['split'] == 'test']['sample_id']

        sample_ids = dict()
        sample_ids['ictal'], sample_ids['event'], sample_ids['nonevent'] = dict(), dict(), dict()
        ictal_sample_ids = self.all_chunks_df[self.all_chunks_df['label'] == 1]['sample_id'].unique()
        sample_ids['ictal']['train'] = list(set(ictal_sample_ids) & set(train_ids))
        sample_ids['ictal']['val'] = list(set(ictal_sample_ids) & set(val_ids))
        sample_ids['ictal']['test'] = list(set(ictal_sample_ids) & set(test_ids))

        event_sample_ids = self.all_chunks_df[self.all_chunks_df['markers'] != '']['sample_id'].unique()
        event_sample_ids = list(set(event_sample_ids) - set(ictal_sample_ids))
        sample_ids['event']['train'] = list(set(event_sample_ids) & set(train_ids))
        sample_ids['event']['val'] = list(set(event_sample_ids) & set(val_ids))
        sample_ids['event']['test'] = list(set(event_sample_ids) & set(test_ids))

        nonevent_sample_ids = list(
            set(self.all_chunks_df['sample_id'].unique()) - set(event_sample_ids) - set(ictal_sample_ids))
        sample_ids['nonevent']['train'] = list(set(nonevent_sample_ids) & set(train_ids))
        sample_ids['nonevent']['val'] = list(set(nonevent_sample_ids) & set(val_ids))
        sample_ids['nonevent']['test'] = list(set(nonevent_sample_ids) & set(test_ids))
        self.sample_ids = sample_ids
        self.priority_sample()

    def priority_sample(self):
        subset = list()
        for split in ['train', 'val', 'test']:
            rnd = self.random if split == 'train' else random.Random(42)
            subset += list(self.sample_ids['ictal'][split])
            e_sample_size = int(self.prio_event_frac * len(self.sample_ids['event'][split]))
            subset += rnd.sample(self.sample_ids['event'][split], e_sample_size)
            n_sample_size = int(self.prio_nonevent_frac * len(self.sample_ids['nonevent'][split]))
            subset += rnd.sample(self.sample_ids['nonevent'][split], n_sample_size)

        self.chunks_df = self.all_chunks_df[self.all_chunks_df['sample_id'].isin(subset)].copy()
        self.samples_df = self.all_samples_df[self.all_samples_df['sample_id'].isin(subset)].copy()

        # reproducible fingerprint
        sorted_subset = sorted(subset)
        subset_str = ','.join(map(str, sorted_subset))
        hash_digest = hashlib.sha1(subset_str.encode()).hexdigest()[:8]

        # update split views
        self._init_split_views()

        print(f"...randomized priority sampling: hash={hash_digest}, total={len(subset)} samples")

    # ----------------------------------------------------------------------------------
    # HDF-safe explicit dtypes (avoid pandas nullable IntegerArray)
    # ----------------------------------------------------------------------------------
    def __set_explicit_types(self):
        self.samples_df = self.samples_df.copy()
        self.chunks_df = self.chunks_df.copy()

        # samples_df
        for c in ["timestamp"]:
            if c in self.samples_df.columns:
                self.samples_df[c] = pd.to_numeric(self.samples_df[c], errors="coerce").fillna(0.0).astype(np.float64)

        for c in ["start_index", "end_index"]:
            if c in self.samples_df.columns:
                self.samples_df[c] = pd.to_numeric(self.samples_df[c], errors="coerce").fillna(0).astype(np.int64)

        for c in ["filename", "split", "source_split"]:
            if c in self.samples_df.columns:
                self.samples_df[c] = self.samples_df[c].fillna("").astype(str).astype(object)

        # chunks_df
        self.samples_df["sample_id"] = self.samples_df["sample_id"].fillna("").astype(str).astype(object)
        self.samples_df["patient_id"] = self.samples_df["patient_id"].fillna("").astype(str).astype(object)
        self.samples_df["session_id"] = self.samples_df["session_id"].fillna("").astype(str).astype(object)

        self.chunks_df["timestep"] = pd.to_numeric(self.chunks_df["timestep"], errors="coerce").fillna(0).astype(np.int64)
        self.chunks_df["seg_type"] = self.chunks_df["seg_type"].fillna("").astype(str).astype(object)
        self.chunks_df["markers"] = self.chunks_df["markers"].fillna("").astype(str).astype(object)
        self.chunks_df["notes"] = self.chunks_df["notes"].fillna("").astype(str).astype(object)
        self.chunks_df["label"] = pd.to_numeric(self.chunks_df["label"], errors="coerce").fillna(0).astype(np.int64)

        max_ts = float(self.cfg.get("max_ts", 1e15))
        for col in ("time_past_seizure", "time_to_seizure"):
            if col in self.chunks_df.columns:
                s = pd.to_numeric(self.chunks_df[col], errors="coerce")
                s = s.mask(~np.isfinite(s), max_ts).clip(lower=0.0, upper=max_ts)
                self.chunks_df[col] = s.astype(np.float64)

    @staticmethod
    def _channels_fingerprint(ch_names: List[str]) -> str:
        """Short stable fingerprint for channel order/config."""
        s = ",".join([str(c).strip().lower() for c in ch_names])
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

    @classmethod
    def _compute_norm_stats_for_row(
        cls,
        row: pd.Series,
        *,
        cfg: Dict[str, Any],
        base_dir: str = "",
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-channel mean and q95(|X-mean|) for one sample window.

        Returns:
            mean: (C,) float32
            q95:  (C,) float32
        """
        filename = str(row["filename"])
        start_index = int(row["start_index"])
        end_index = int(row["end_index"])

        hdf = load_partial_hdf5(
            os.path.join(base_dir, filename),
            start_idx=start_index,
            stop_idx=end_index,
            only_signal=True,
            channel_order=cfg["signal_channels"],
        )
        X = hdf["signal"].astype(np.float32)  # (C, T)

        mean = X.mean(axis=1).astype(np.float32)  # (C,)
        Xc = X - mean[:, None]
        q95 = np.quantile(np.abs(Xc), 0.95, axis=1).astype(np.float32)  # (C,)

        # avoid degenerate division
        q95 = np.maximum(q95, np.float32(eps))
        return mean, q95

    @classmethod
    def precompute_norm_stats(
        cls,
        samples_df: pd.DataFrame,
        *,
        cfg: Dict[str, Any],
        base_dir: str = "",
        n_jobs: int = 0,
        eps: float = 1e-8,
        max_items: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        One-time pass: attach 'norm_mean', 'norm_q95', 'norm_channels_hash' to samples_df.

        Args:
            samples_df: existing samples df
            cfg: must include cfg["signal_channels"]
            base_dir: optional base path prefix for files
            n_jobs: 0/1 => serial; >1 => joblib parallel
            max_items: optional limit for quick tests
        """
        out_df = samples_df.copy()

        ch_hash = cls._channels_fingerprint(cfg["signal_channels"])
        out_df["norm_channels_hash"] = ch_hash

        # optional: avoid recomputing if already present + matching hash
        if "norm_mean" in out_df.columns and "norm_q95" in out_df.columns:
            # if everything non-null and hash matches, skip
            if out_df["norm_channels_hash"].eq(ch_hash).all() and out_df["norm_mean"].notna().all() and out_df["norm_q95"].notna().all():
                if verbose:
                    print(f"[norm] stats already present for channels_hash={ch_hash}; skipping recompute")
                return out_df

        # allocate columns
        out_df["norm_mean"] = None
        out_df["norm_q95"] = None

        N = len(out_df) if max_items is None else min(len(out_df), max_items)

        if n_jobs is None or n_jobs <= 1:
            it = range(N)
            if verbose:
                it = tqdm(it, desc="[norm] precompute", total=N)

            for i in it:
                row = out_df.iloc[i]
                try:
                    mean, q95 = cls._compute_norm_stats_for_row(row, cfg=cfg, base_dir=base_dir, eps=eps)
                    out_df.at[out_df.index[i], "norm_mean"] = mean
                    out_df.at[out_df.index[i], "norm_q95"] = q95
                except Exception as e:
                    # keep None; caller may decide to drop these samples
                    if verbose:
                        print(f"[norm][WARN] failed sample_id={row.get('sample_id', 'n/a')}: {e}")
        else:
            rows = [out_df.iloc[i] for i in range(N)]

            def _safe(row):
                try:
                    return cls._compute_norm_stats_for_row(row, cfg=cfg, base_dir=base_dir, eps=eps)
                except Exception:
                    return None, None

            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_safe)(r) for r in tqdm(rows, disable=not verbose, desc="[norm] precompute")
            )

            for i, (mean, q95) in enumerate(results):
                if mean is None or q95 is None:
                    continue
                out_df.at[out_df.index[i], "norm_mean"] = mean
                out_df.at[out_df.index[i], "norm_q95"] = q95

        if verbose:
            ok = out_df["norm_mean"].notna().sum()
            print(f"[norm] computed stats for {ok}/{N} samples (channels_hash={ch_hash})")

        return out_df

    @classmethod
    def normalize_torch_with_stats(
        cls,
        X: torch.Tensor,
        *,
        mean: Optional[torch.Tensor] = None,
        q95: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Normalize signal using precomputed mean/q95 if provided,
        otherwise falls back to computing them on the fly.

        X: (C, T) or (C, T, ...)
        """
        if mean is not None and q95 is not None:
            # expect (C,)
            if mean.ndim == 1:
                mean = mean.view(-1, *([1] * (X.ndim - 1)))
            if q95.ndim == 1:
                q95 = q95.view(-1, *([1] * (X.ndim - 1)))
            Xc = X - mean.to(device=X.device, dtype=X.dtype)
            return Xc / (q95.to(device=X.device, dtype=X.dtype) + eps)

        # fallback: compute on the fly (original behavior)
        Xc = X - X.mean(dim=-1, keepdim=True)
        q = torch.quantile(torch.abs(Xc), q=0.95, dim=-1, keepdim=True)
        return Xc / (q + eps)
