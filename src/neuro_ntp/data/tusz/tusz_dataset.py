"""
+----------------------------------------------------------------------------------------------------------------------+
tusz_dataset.py
Core dataset of the TUH EEG Seizure Corpus (TUSZ)

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    The core dataset segments the recording sessions into ictal and non-ictal intervals of variable length. One or
    more dataset views can be created from a core dataset; a view provides the training, validation and test samples
    according to its sampling strategy. Both the segmentation policy and the sampling strategy are defined by
    configuration dictionaries.

    The core dataset is built from the HDF5 store created by tusz2hdf.py (the path to the store is the input). The
    patient, session, segment (ictal/non-ictal) and marker tables are extracted from the HDF5 files. The core dataset
    can serve both supervised and self-supervised dataset views.
+----------------------------------------------------------------------------------------------------------------------+
"""
from __future__ import annotations
from typing import List

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
import json
from neuro_ntp.misc.io import load_partial_hdf5
from neuro_ntp.data.epi_dataset import EpilepsiaCoreDataset, EpilepsiaDatasetView
import re
import math
from datetime import datetime, timedelta

class TUSZCoreDataset(EpilepsiaCoreDataset):
    """
    Core dataset of TUSZ v2.0.3, built from the stitched session-level HDF5 store (tusz2hdf.py).

    Implements the segmentation of the recording sessions into ictal and non-ictal intervals on top of
    EpilepsiaCoreDataset. The HDF5 store holds one file per session, converted from the EDF recordings into a format
    optimised for random access.

    Construction:
        TUSZCoreDataset(name=..., hdf_store_path=...)    build from the HDF5 store
        TUSZCoreDataset(ds_filepath=...)                 load a core dataset previously saved with save()

    Tables:
        session_df     one row per session, from sessions.csv of the conversion
        segments_df    ictal and non-ictal segments, derived from the merged seizure intervals (annotations.*)
        markers_df     event-level markers from markers.* (e.g. seizure types such as gnsz, fnsz), which can be
                       mapped to seizure-type tokens

    The saved core dataset contains metadata only and references the signal data in the HDF5 store.
    """

    DEFAULT_CFG = {
        "policy": {
            "min_session_sec": 300.0,
        },
        "labels": {
            "seizure_desc": "seizure",
            "seizure_marker_kind": "seizure_type",      # annotations.csv kind for typed seizure events
            "nonseiz_event_kind": "nonseizure_event",   # annotations.csv kind for e.g. bckg
        },
        "io": {
            "csv_sep": ",",
        },
    }

    def __init__(
        self,
        name: str = "TUSZCoreDS",
        ds_filepath: str | None = None,
        session_csv: str | None = None,  # ignored if hdf_store_path provided (kept for compatibility)
        hdf_store_path: str | None = None,
        cfg: dict = DEFAULT_CFG,
        *,
        session_t0: datetime = datetime(2000, 1, 1),
    ):
        super().__init__(
            name=name,
            ds_filepath=ds_filepath,
            session_csv=session_csv,
            hdf_store_path=hdf_store_path,
            cfg=cfg,
        )
        self.session_t0 = session_t0

        # cached annotations dataframe (from annotations.csv)
        self._ann_df: pd.DataFrame | None = None

        if self.ds_filepath:
            self.__load()
        else:
            self.init_from_input()
            self.__calc_segment_stats()

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------
    def save(self, ds_filepath: str):
        print(f"...saving core dataset {self.name} to {ds_filepath}")
        self.__set_explicit_types()
        with pd.HDFStore(ds_filepath, "w") as hfile:
            hfile.put("params", pd.DataFrame())
            hfile.get_storer("params").attrs["name"] = self.name
            hfile.get_storer("params").attrs["session_csv"] = self.session_csv
            hfile.get_storer("params").attrs["hdf_store_path"] = self.hdf_store_path
            hfile.get_storer("params").attrs["session_t0"] = self.session_t0.isoformat()

            hfile.put("config", pd.Series(json.dumps(self.cfg)))
            hfile.put("patients", self.patients_df)
            hfile.put("sessions", self.session_df)
            hfile.put("markers", self.markers_df)
            hfile.put("segments", self.segments_df)

    def __load(self):
        print(f"...loading core dataset {self.name} from {self.ds_filepath}")
        with pd.HDFStore(self.ds_filepath, "r") as hfile:
            st = hfile.get_storer("params")

            self.name = st.attrs["name"]
            self.session_csv = st.attrs["session_csv"]
            self.hdf_store_path = st.attrs["hdf_store_path"]

            # attrs is a PyTables AttributeSet (no .get)
            try:
                t0 = st.attrs.session_t0
            except Exception:
                t0 = None

            if t0:
                try:
                    self.session_t0 = datetime.fromisoformat(t0)
                except Exception:
                    self.session_t0 = datetime(2000, 1, 1)

            self.cfg = json.loads(hfile["config"].item())
            self.patients_df = hfile["patients"]
            self.session_df = hfile["sessions"]
            self.markers_df = hfile["markers"]
            self.segments_df = hfile["segments"]

    # ---------------------------------------------------------------------
    # Input initialization (CSV-driven; sessions.csv + annotations.csv mandatory under hdf_store_path)
    # ---------------------------------------------------------------------
    def init_from_input(self):
        if not self.hdf_store_path:
            raise ValueError("hdf_store_path must be provided (it must contain sessions.csv and annotations.csv).")

        sep = self.cfg.get("io", {}).get("csv_sep", ",")
        sessions_csv = os.path.join(self.hdf_store_path, "sessions.csv")
        ann_csv = os.path.join(self.hdf_store_path, "annotations.csv")

        if not os.path.exists(sessions_csv):
            raise FileNotFoundError(f"Missing required file: {sessions_csv}")
        if not os.path.exists(ann_csv):
            raise FileNotFoundError(f"Missing required file: {ann_csv}")

        print(f"...initializing new core dataset: {self.name}")
        print(f"...reading sessions csv: {sessions_csv}")
        df = pd.read_csv(sessions_csv, sep=sep)

        # Required base columns
        required_base = {
            "patient_id", "session_id", "split", "filename", "sfreq", "n_channels",
            "signal_length", "duration_sec", "n_segments", "channels", "units"
        }
        missing_base = required_base - set(df.columns)
        if missing_base:
            raise ValueError(f"sessions.csv missing columns: {sorted(missing_base)}")

        # Accept new counts (preferred) OR old legacy columns
        if "n_seizure_intervals" in df.columns:
            df["n_seizures"] = df["n_seizure_intervals"].astype("Int64")
        elif "n_seizures" in df.columns:
            df["n_seizures"] = df["n_seizures"].astype("Int64")
        else:
            # if neither exists, fall back to 0 (safe) or error
            df["n_seizures"] = 0

        # Optional: keep the other counts if present (recommended)
        for c in ["n_events_total", "n_seizure_events", "n_nonseizure_events"]:
            if c in df.columns:
                df[c] = df[c].astype("Int64")

        # Filename resolver for abs/relative paths
        def _resolve(p: str) -> str:
            p = str(p)
            return p if os.path.isabs(p) else os.path.join(self.hdf_store_path, p)

        df["filename"] = df["filename"].apply(_resolve)

        # Keep only existing HDFs
        exists = df["filename"].apply(lambda p: os.path.exists(str(p)))
        if (~exists).any():
            print(f"WARNING: {(~exists).sum()} session HDFs from CSV do not exist; dropping.")
        df = df[exists].copy()

        # Min duration check
        min_sec = float(self.cfg.get("policy", {}).get("min_session_sec", 0.0))
        if min_sec > 0:
            bad = df["duration_sec"].astype(float) < min_sec
            if bad.any():
                print(f"WARNING: dropping {bad.sum()} sessions shorter than {min_sec}s.")
                df = df[~bad].copy()

        # Synthetic anchor time
        df["measurement_start_time"] = pd.to_datetime(self.session_t0)

        # Normalize types (not overly defensive, as requested)
        df["patient_id"] = df["patient_id"].astype("string")
        df["session_id"] = df["session_id"].astype("string")
        df["split"] = df["split"].astype("string")
        df["filename"] = df["filename"].astype("string")
        df["units"] = df["units"].astype("string")
        df["channels"] = df["channels"].astype("string")

        df["sfreq"] = df["sfreq"].astype("float64")
        df["duration_sec"] = df["duration_sec"].astype("float64")
        for c in ["n_channels", "signal_length", "n_segments", "n_seizures"]:
            df[c] = df[c].astype("Int64")

        # KUK-like stable session_df
        base_cols = [
            "patient_id", "session_id", "split",
            "measurement_start_time",
            "filename", "sfreq", "n_channels", "signal_length", "duration_sec",
            "n_segments", "channels", "n_seizures", "units",
        ]
        extra_cols = [c for c in ["n_events_total", "n_seizure_events", "n_nonseizure_events"] if c in df.columns]
        self.session_df = df[base_cols + extra_cols].copy()

        # Load annotations.csv once (mandatory)
        print(f"...reading annotations csv: {ann_csv}")
        ann = pd.read_csv(ann_csv, sep=sep)
        # minimal normalization
        ann["patient_id"] = ann["patient_id"].astype("string")
        ann["session_id"] = ann["session_id"].astype("string")
        ann["split"] = ann["split"].astype("string")
        ann["filename"] = ann["filename"].astype("string")
        ann["description"] = ann["description"].astype("string")
        ann["kind"] = ann["kind"].astype("string")
        ann["onset_sec"] = ann["onset_sec"].astype("float64")
        ann["duration_sec"] = ann["duration_sec"].astype("float64")

        # Keep only sessions we kept (in case annotations.csv contains more)
        kept = self.session_df[["patient_id", "session_id"]].drop_duplicates()
        ann = ann.merge(kept, on=["patient_id", "session_id"], how="inner")

        self._ann_df = ann

        # Build core dfs
        print(f"...building segments dataframe from annotations.csv (kind == ictal_interval)")
        self.build_segments_df()

        print(f"...building markers dataframe from annotations.csv (kind == seizure_type)")
        self.build_markers_df()

        # Patients df (initial; refined stats later)
        self.patients_df = (
            self.session_df
            .groupby("patient_id", as_index=False)
            .agg(
                n_sessions=("session_id", "nunique"),
                total_duration_sec=("duration_sec", "sum"),
                n_seizures=("n_seizures", "sum"),
            )
        )

        # Sanity: validate if sessions have typed events
        typed = (
            self.markers_df[self.markers_df["marker"] == "seizure_type"]
            .groupby(["patient_id", "session_id"], as_index=False)
            .size()
            .rename(columns={"size": "n_typed_seizure_events"})
        )
        chk = self.session_df.merge(typed, on=["patient_id", "session_id"], how="left")
        chk["n_typed_seizure_events"] = chk["n_typed_seizure_events"].fillna(0).astype(int)

        bad = (chk["n_seizures"].fillna(0).astype(int) > 0) & (chk["n_typed_seizure_events"] == 0)
        if bad.any():
            ex = chk.loc[bad, ["patient_id", "session_id", "filename", "n_seizures", "n_typed_seizure_events"]].head(20)
            raise RuntimeError(f"Found sessions with seizure intervals but zero seizure_type events. Examples:\n{ex}")

        print(f"...finished initializing dataset: {self.name}")

    import pandas as pd
    import numpy as np
    from pandas.api.types import is_integer_dtype, is_bool_dtype, is_string_dtype

    def _hdf_fixed_safe(df: pd.DataFrame, *, int_na_value: int = 0) -> pd.DataFrame:
        """
        Make df safe for pd.HDFStore(..., format='fixed'):
          - Convert Pandas nullable integers (Int64 -> IntegerArray) to np.int64
          - Convert nullable booleans to np.int8 or bool
          - Convert pandas StringDtype to object
        """
        out = df.copy()

        for c in out.columns:
            s = out[c]

            # Pandas StringDtype -> object (PyTables fixed is happier)
            if str(s.dtype) == "string":
                out[c] = s.fillna("").astype(object)
                continue

            # Nullable integers (Int64, Int32, ...) -> np.int64
            # These are extension arrays that break fixed-format HDF.
            if pd.api.types.is_extension_array_dtype(s.dtype) and is_integer_dtype(s.dtype):
                out[c] = pd.to_numeric(s, errors="coerce").fillna(int_na_value).astype(np.int64)
                continue

            # Nullable boolean -> bool (or int8); fixed can struggle with BooleanArray
            if pd.api.types.is_extension_array_dtype(s.dtype) and is_bool_dtype(s.dtype):
                out[c] = s.fillna(False).astype(bool)
                continue

            # (Optional) categories -> object
            if str(s.dtype) == "category":
                out[c] = s.astype(object)

        return out

    # ---------------------------------------------------------------------
    # Core dataframe builders (CSV-driven)
    # ---------------------------------------------------------------------
    def build_segments_df(self) -> pd.DataFrame:
        """
        segments_df from merged ictal intervals in annotations.csv (kind == 'ictal_interval').
        """
        if self._ann_df is None:
            raise RuntimeError("annotations.csv not loaded (self._ann_df is None).")

        seiz_desc = str(self.cfg.get("labels", {}).get("seizure_desc", "seizure")).strip().lower()

        ann = self._ann_df
        seg_ann = ann[(ann["kind"].str.lower() == "ictal_interval") &
                      (ann["description"].str.lower() == seiz_desc)].copy()

        rows = []
        # iterate session_df so we can add complement non-ictal segments and epoch timestamps consistently
        for _, srow in tqdm(self.session_df.iterrows(), total=len(self.session_df)):
            pid = srow["patient_id"]
            sid = srow["session_id"]
            split = srow["split"]
            path = str(srow["filename"])
            t0 = srow["measurement_start_time"]
            total_dur = float(srow["duration_sec"])

            g = seg_ann[(seg_ann["patient_id"] == pid) & (seg_ann["session_id"] == sid)]
            ictals = []
            for r in g.itertuples(index=False):
                s = float(r.onset_sec)
                e = float(r.onset_sec + r.duration_sec)
                if e > s:
                    ictals.append([s, e])
            ictals.sort(key=lambda x: (x[0], x[1]))

            # non-ictal complement
            nonictals = self._complement_intervals(ictals, total_dur)

            segs = []
            for s, e in ictals:
                segs.append({"start_ts": float(s), "end_ts": float(e), "seg_type": "ictal"})
            for s, e in nonictals:
                segs.append({"start_ts": float(s), "end_ts": float(e), "seg_type": "non-ictal"})
            segs.sort(key=lambda d: (d["start_ts"], d["end_ts"]))

            ictal_onsets = np.array([s for s, _ in ictals], dtype=float) if ictals else np.array([], dtype=float)

            for i, seg in enumerate(segs):
                s, e = seg["start_ts"], seg["end_ts"]
                seg["duration"] = float(e - s)

                if isinstance(t0, (pd.Timestamp, datetime)):
                    seg["epoch_start_ts"] = pd.to_datetime(t0) + pd.to_timedelta(s, unit="s")
                    seg["epoch_end_ts"] = pd.to_datetime(t0) + pd.to_timedelta(e, unit="s")
                else:
                    seg["epoch_start_ts"] = pd.NaT
                    seg["epoch_end_ts"] = pd.NaT

                seg["followed_by"] = segs[i + 1]["seg_type"] if i + 1 < len(segs) else "N/A"

                if ictal_onsets.size == 0:
                    seg["time_to_prev_ictal"] = np.nan
                    seg["time_to_next_ictal"] = np.nan
                else:
                    idx_prev = np.searchsorted(ictal_onsets, s, side="left") - 1
                    seg["time_to_prev_ictal"] = float(s - ictal_onsets[idx_prev]) if idx_prev >= 0 else pd.NA
                    idx_next = np.searchsorted(ictal_onsets, e, side="left")
                    seg["time_to_next_ictal"] = float(
                        ictal_onsets[idx_next] - e) if idx_next < ictal_onsets.size else pd.NA

                seg.update({
                    "patient_id": pid,
                    "session_id": sid,
                    "split": split,
                    "filename": path,
                })
                rows.append(seg)

        seg_df = pd.DataFrame(rows, columns=[
            "patient_id", "session_id", "split",
            "start_ts", "end_ts", "duration", "seg_type",
            "epoch_start_ts", "epoch_end_ts",
            "time_to_prev_ictal", "time_to_next_ictal",
            "followed_by", "filename"
        ])
        if not seg_df.empty:
            seg_df.sort_values(["patient_id", "session_id", "start_ts"], inplace=True)
            seg_df.reset_index(drop=True, inplace=True)

            # --- KUK-style sentinel for missing prev/next ictal distances (keep numeric + HDF-safe) ---
            SENTINEL = 1e15  # KUK-style

            # after seg_df is built
            seg_df["time_to_prev_ictal"] = pd.to_numeric(seg_df["time_to_prev_ictal"], errors="coerce")
            seg_df["time_to_next_ictal"] = pd.to_numeric(seg_df["time_to_next_ictal"], errors="coerce")

            seg_df.loc[seg_df["time_to_prev_ictal"].isna(), "time_to_prev_ictal"] = SENTINEL
            seg_df.loc[seg_df["time_to_next_ictal"].isna(), "time_to_next_ictal"] = SENTINEL

            seg_df["time_to_prev_ictal"] = seg_df["time_to_prev_ictal"].astype("float64")
            seg_df["time_to_next_ictal"] = seg_df["time_to_next_ictal"].astype("float64")

        self.segments_df = seg_df
        return seg_df

    def build_markers_df(self) -> pd.DataFrame:
        """
        markers_df from event-level annotations in annotations.csv:
          - kind == 'seizure_type'      -> marker='seizure_type' (e.g., gnsz/fnsz/...)
          - kind == 'nonseizure_event'  -> marker='event_type'   (e.g., bckg)

        This way, bckg is available as a marker, but seizure counts remain based on segments_df only.
        """
        if self._ann_df is None:
            raise RuntimeError("annotations.csv not loaded (self._ann_df is None).")

        ann = self._ann_df

        ev = ann[ann["kind"].str.lower().isin(["seizure_type", "nonseizure_event"])].copy()
        if ev.empty:
            self.markers_df = pd.DataFrame(columns=[
                "patient_id", "session_id", "split",
                "onset", "epoch_onset", "duration",
                "description", "marker", "mk_state", "mk_notes",
                "filename"
            ])
            return self.markers_df

        rows = []
        for _, srow in tqdm(self.session_df.iterrows(), total=len(self.session_df)):
            pid = srow["patient_id"]
            sid = srow["session_id"]
            split = srow["split"]
            path = str(srow["filename"])
            t0 = srow["measurement_start_time"]

            g = ev[(ev["patient_id"] == pid) & (ev["session_id"] == sid)].copy()
            if g.empty:
                continue

            for r in g.itertuples(index=False):
                s = float(r.onset_sec)
                d = float(r.duration_sec)
                lab = str(r.description).strip().lower()
                kind = str(r.kind).strip().lower()

                if not lab:
                    raise RuntimeError(f"Empty event label in annotations.csv for {pid} {sid} @ {s}s")

                marker_key = "seizure_type" if kind == "seizure_type" else "event_type"

                rows.append({
                    "patient_id": pid,
                    "session_id": sid,
                    "split": split,
                    "onset": s,
                    "epoch_onset": (pd.to_datetime(t0) + pd.to_timedelta(s, unit="s"))
                    if isinstance(t0, (pd.Timestamp, datetime)) else pd.NaT,
                    "duration": d,
                    "description": lab,  # keep as-is
                    "marker": marker_key,  # keep for core sanity checks / grouping
                    "marker_text": lab,  # NEW: use this for token text (classes)
                    "mk_state": pd.NA,
                    "mk_notes": pd.NA,
                    "filename": path,
                })

        mk_df = pd.DataFrame(rows, columns=[
                "patient_id", "session_id", "split",
                "onset", "epoch_onset", "duration",
                "description", "marker", "marker_text", "mk_state", "mk_notes",
                "filename"
        ])
        if not mk_df.empty:
            mk_df.sort_values(["patient_id", "session_id", "onset"], inplace=True)
            mk_df.reset_index(drop=True, inplace=True)

        self.markers_df = mk_df
        return mk_df

    @staticmethod
    def _complement_intervals(ictals: List[List[float]], total_dur: float) -> List[List[float]]:
        if total_dur <= 0:
            return []
        if not ictals:
            return [[0.0, float(total_dur)]]
        out = []
        cur = 0.0
        for s, e in ictals:
            if s > cur:
                out.append([cur, s])
            cur = max(cur, e)
        if cur < total_dur:
            out.append([cur, total_dur])
        return out

    # ---------------------------------------------------------------------
    # Stats (unchanged in spirit; markers_df counts are typed seizure events)
    # ---------------------------------------------------------------------
    def __calc_segment_stats(self):
        if getattr(self, "session_df", None) is None or self.session_df.empty:
            return

        # markers per session (typed seizure events)
        if getattr(self, "markers_df", None) is not None and not self.markers_df.empty:
            mk_counts = (
                self.markers_df
                .groupby(["patient_id", "session_id"], as_index=False)
                .size()
                .rename(columns={"size": "n_markers"})
            )
        else:
            mk_counts = self.session_df[["patient_id", "session_id"]].copy()
            mk_counts["n_markers"] = 0

        # ictal segments per session (merged intervals)
        if getattr(self, "segments_df", None) is not None and not self.segments_df.empty:
            ictal_counts = (
                self.segments_df[self.segments_df["seg_type"] == "ictal"]
                .groupby(["patient_id", "session_id"], as_index=False)
                .size()
                .rename(columns={"size": "n_seiz_intervals"})
            )
            dur_stats = (
                self.segments_df
                .assign(
                    ictal_duration=lambda d: d["duration"].where(d["seg_type"] == "ictal", 0.0),
                    non_ictal_duration=lambda d: d["duration"].where(d["seg_type"] != "ictal", 0.0),
                )
                .groupby(["patient_id", "session_id"], as_index=False)[["ictal_duration", "non_ictal_duration"]]
                .sum()
            )
        else:
            ictal_counts = self.session_df[["patient_id", "session_id"]].copy()
            ictal_counts["n_seiz_intervals"] = 0
            dur_stats = self.session_df[["patient_id", "session_id"]].copy()
            dur_stats["ictal_duration"] = 0.0
            dur_stats["non_ictal_duration"] = self.session_df["duration_sec"].astype(float)

        key = ["patient_id", "session_id"]
        self.session_df = (
            self.session_df
            .merge(mk_counts, on=key, how="left")
            .merge(ictal_counts, on=key, how="left")
            .merge(dur_stats, on=key, how="left")
        )
        self.session_df["n_markers"] = self.session_df["n_markers"].fillna(0).astype(int)
        self.session_df["n_seiz_intervals"] = self.session_df["n_seiz_intervals"].fillna(0).astype(int)
        self.session_df["ictal_duration"] = self.session_df["ictal_duration"].fillna(0.0).astype(float)
        self.session_df["non_ictal_duration"] = self.session_df["non_ictal_duration"].fillna(0.0).astype(float)

        self.session_df["total_duration_sec"] = self.session_df["duration_sec"].astype(float)
        self.session_df = self.session_df.replace([np.inf, -np.inf], np.nan)
        self.session_df["ictal_ratio"] = (
            self.session_df["ictal_duration"] /
            self.session_df["total_duration_sec"].replace(0, np.nan)
        ).astype(float)

        self.patients_df = (
            self.session_df
            .groupby("patient_id", as_index=False)
            .agg(
                n_sessions=("session_id", "nunique"),
                total_duration_sec=("total_duration_sec", "sum"),
                n_markers=("n_markers", "sum"),
                n_seiz_intervals=("n_seiz_intervals", "sum"),
                ictal_duration=("ictal_duration", "sum"),
                non_ictal_duration=("non_ictal_duration", "sum"),
            )
        )

        self.patients_df = self.patients_df.replace([np.inf, -np.inf], np.nan)
        self.patients_df["ictal_ratio"] = (
            self.patients_df["ictal_duration"] /
            self.patients_df["total_duration_sec"].replace(0, np.nan)
        ).astype(float)

    # ---------------------------------------------------------------------
    # Types for persistence
    # ---------------------------------------------------------------------
    def __set_explicit_types(self):
        # ---------------- patients_df ----------------
        if getattr(self, "patients_df", None) is not None and not self.patients_df.empty:
            p = self.patients_df.copy()
            p["patient_id"] = p["patient_id"].astype(object)

            for c in ["n_sessions", "n_markers", "n_seiz_intervals"]:
                if c in p:
                    p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0).astype(np.int64)

            for c in ["total_duration_sec", "ictal_duration", "non_ictal_duration", "ictal_ratio"]:
                if c in p:
                    p[c] = pd.to_numeric(p[c], errors="coerce").astype(np.float64)

            self.patients_df = p

        # ---------------- session_df ----------------
        if getattr(self, "session_df", None) is not None and not self.session_df.empty:
            s = self.session_df.copy()
            for c in ["patient_id", "session_id", "split", "filename", "channels", "units"]:
                if c in s:
                    s[c] = s[c].astype(object)

            if "measurement_start_time" in s:
                s["measurement_start_time"] = pd.to_datetime(s["measurement_start_time"], errors="coerce")

            for c in ["sfreq", "duration_sec", "total_duration_sec", "ictal_duration", "non_ictal_duration",
                      "ictal_ratio"]:
                if c in s:
                    s[c] = pd.to_numeric(s[c], errors="coerce").astype(np.float64)

            for c in [
                "n_channels", "signal_length", "n_segments",
                "n_seizures", "n_markers", "n_seiz_intervals",
                # new schema
                "n_events_total", "n_seizure_events", "n_nonseizure_events",
                "n_seizure_intervals",
            ]:
                if c in s:
                    s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0).astype(np.int64)

            self.session_df = s

        # ---------------- markers_df ----------------
        if getattr(self, "markers_df", None) is not None and not self.markers_df.empty:
            m = self.markers_df.copy()
            for c in ["patient_id", "session_id", "split", "description", "marker", "mk_state", "mk_notes", "filename"]:
                if c in m:
                    m[c] = m[c].astype("string").fillna("").astype(object)

            for c in ["onset", "duration"]:
                if c in m:
                    m[c] = pd.to_numeric(m[c], errors="coerce").astype(np.float64)

            if "epoch_onset" in m:
                m["epoch_onset"] = pd.to_datetime(m["epoch_onset"], errors="coerce")

            self.markers_df = m

        # ---------------- segments_df ----------------
        if getattr(self, "segments_df", None) is not None and not self.segments_df.empty:
            g = self.segments_df.copy()
            for c in ["patient_id", "session_id", "split", "seg_type", "followed_by", "filename"]:
                if c in g:
                    g[c] = g[c].astype(object)

            for c in ["start_ts", "end_ts", "duration", "time_to_prev_ictal", "time_to_next_ictal"]:
                if c in g:
                    g[c] = pd.to_numeric(g[c], errors="coerce").astype(np.float64)

            for c in ["epoch_start_ts", "epoch_end_ts"]:
                if c in g:
                    g[c] = pd.to_datetime(g[c], errors="coerce")

            self.segments_df = g


class TUSZDatasetView(EpilepsiaDatasetView):
    """
    NB: THIS IS NOT USED IN THE SELF-SUPERVISED SETTING.
    Dataset view of TUSZ on top of EpilepsiaDatasetView, implementing one sampling strategy for the training,
    validation and test samples. A configuration dictionary controls the labelling logic, the dataset split and the
    selection and separation of the intervals used.

    The view uses the ictal, non-ictal and unspecified intervals defined in the core dataset and generates the
    datasets for the data loaders dynamically. Signal data are referenced and read partially when an item is
    retrieved. With sufficient CPU RAM, use_cache enables lazy caching of the preprocessed samples to reduce runtime.

    Persisting a view is not required, but it can be saved and reloaded for reproducibility, or to save processing
    time when a series of experiments uses the same view.
    """

    # this configuration dictionary is just an example - it will be used as default if no other is provided
    # when the view is instantiated; a configuration can also be read from a config (.json) file
    DEFAULT_CFG = {
        'sampling_strategy': 'binary_detection',  # binary_forecasting|binary_detection|threefold|seq8class
        'split': 'patient',                  # how to split the samples: temporal|patient|random
        'split_ratio': [0.6, 0.15, 0.25],     # the split ratio: training, validation, test datasets

        'sampling_freq': 256,                 # the sampling frequency of the signal data
        'sample_length': 15,                  # length of a single sample in seconds
        'sample_offset_tmin': -1,             # offset of the ictal sample relative to the offset (neg. offsets forward)
        'lead_seizures_only': False,          # indicates if only the first seizure is to be taken in case of a cluster
        'lead_seizure_separation': 900,       # distance of a lead seizure from any preceding seizure (separation)
        'postictal_separation': 900,          # min. time after ictal in sec. to consider a sample baseline (non-ictal)
        'samples_per_ictal': 3,               # nr. of samples to obtain from an ictal segment (if length permits it)
        'ictal_interleave': 0,                # how many samples to leave out when taking samples from interval

        'prepro_biot': False,                 # apply BIOT pre-processing (only with pre-trained BIOT model)

        'preictal_length': 600,               # length (in seconds) of pre-ictal interval to sample from
        'preictal_setback': 300,              # interval before ictal not to sample from (bin.forecasting and threefold)
        'preictal_interleave': 2,             # how many samples to leave out when taking samples from interval
                                              # ... nb: 'seq8class' strategy will sample immediately before ictal
        'channels': 'all',                    # which channels to use
        'min_num_channels': 32,               # min. number of channels for acceptable sample

        'interictal_samples_per_session': 15, # how many interictal samples to obtain from one recording session
        'preictal_samples_per_session': 5,    # how many pre-ictal sample to obtain per one recording session
        'omit_nonictal_sessions': False,      # should session (files) without any seizures be omitted in sampling
        'interictal_interleave': 1,           # how many samples to leave out when taking samples from interval

        # order of the channels in the signal tensor
        'signal_channels': ['EEG FP1', 'EEG FP2', 'EEG F7', 'EEG F3', 'EEG FZ', 'EEG F4', 'EEG F8', 'EEG T7', 'EEG C3',
                          'EEG CZ', 'EEG C4', 'EEG T8', 'EEG P7', 'EEG P3', 'EEG PZ', 'EEG P4',
                          'EEG P8', 'EEG O1', 'EEG O2', 'EEG M1', 'EEG M2', 'EEG F10', 'EEG F9', 'EEG FT10', 'EEG FT8',
                          'EEG FT7', 'EEG FT9', 'EEG T10', 'EEG T9',
                          'EEG EOGL', 'EEG EOGR', 'SAO2 SpO2', 'ECG EKG']

    }

    def __init__(self, name: str = 'TUSZDsView', core_ds: 'EpilepsiaCoreDataset' = None,
                 cfg_filepath: str = None, cfg: dict = DEFAULT_CFG, dv_filepath: str = None,  use_cache: bool = False):
        """
            Initializes the TUSZDatasetView using a core dataset, creating the samples based on:
            - a view configuration file (cfg_filepath),
            - a view configuration dictionary (cfg), or
            - previously saved (materialized) view object (dv_filepath).
            Optionally, using lazy-cache for storing/retrieving already prepared samples.

            Args:
                name (str, optional): name for the dataset view (to distinguish different views).
                cfg_filepath (str, optional): path to the .yaml configuration file (if config is read from file).
                cfg (dict, optional): configuration dictionary - if configuration is set directly.
                core_ds (EpilepsiaCoreDataset, optional): the core dataset to take high-level segments from
                dv_filepath (str, optional): the .h5 file to load a previously saved (materialized) view from
                use_cache (bool, optional): lazy-caching of samples (requires CPU/operative RAM and shared memory)
            """
        super().__init__(name, core_ds, dv_filepath, cfg, use_cache)

        if self.dv_filepath:
            # load a previously saved dataset view
            self.__load()
            print(f"use cache: {use_cache}")
        else:
            # use config file if provided
            if cfg_filepath:
                print("...reading dataset view configuration from file...")
                self.load_cfg(cfg_filepath)

            # create the dataset view dynamically
            print(f"...creating dataset view {self.name} based on {self.core_ds.name}")
            self.samples_df = pd.DataFrame(
                columns=['patient', 'timestamp', 'filename', 'start_index', 'end_index', 'label'])
            self.retrieve_samples()
            self.split_samples()
        self.print_label_split_stats()

    def save(self, dv_filepath: str):
        """ auxiliary method for saving the dataset view as an HDF5 file """
        print(f"...saving dataset view {self.name} to {dv_filepath}")
        self.__set_explicit_types()
        with pd.HDFStore(dv_filepath, 'w') as hfile:
            hfile.put('params', pd.DataFrame())
            hfile.get_storer('params').attrs['name'] = self.name
            hfile.get_storer('params').attrs['core_ds_filepath'] = self.core_ds.ds_filepath
            hfile.put('config', pd.Series(json.dumps(self.cfg)))
            hfile.put('samples', self.samples_df)

    def __load(self):
        """ auxiliary method for loading the dataset view from an HDF5 file """
        print(f"...loading dataset view {self.name} from {self.dv_filepath}")
        with pd.HDFStore(self.dv_filepath, 'r') as hfile:
            self.name = hfile.get_storer('params').attrs['name']
            self.core_ds_filepath = hfile.get_storer('params').attrs['core_ds_filepath']
            self.cfg = json.loads(hfile['config'].item())
            self.samples_df = hfile['samples']
        self.core_ds = TUSZCoreDataset(ds_filepath=self.core_ds_filepath)

    def __add_samples(self, seg_df, label, max_samples=100, interleave=0):
        """ adds one or more consecutive samples to the dataframe of the specified label """
        sfreq = self.cfg['sampling_freq']
        slen = self.cfg['sample_length']
        sdf  = seg_df[(seg_df['start_ts'] > 0)]
        print(f"...adding samples : {label}")
        for j in tqdm(range(len(sdf))):
            patient, filename, epoch_ts, start_ts, end_ts = sdf.iloc[j][
                ['patient_id', 'filename', 'epoch_start_ts', 'start_ts', 'end_ts']]
            step = slen * (interleave + 1)
            df = pd.DataFrame([[patient, epoch_ts, filename, int( (start_ts + (i * step)) * sfreq),
                                int( (start_ts + (i * step) + slen) * sfreq), label]
                               for i in range(min(round((end_ts - start_ts) / step), max_samples))],
                              columns=['patient', 'timestamp', 'filename', 'start_index', 'end_index', 'label'])
            self.samples_df = pd.concat([self.samples_df, df])

    def __add_ictals(self, seg_df):
        """ adds the ictal samples in case of detection and multi-class tasks """
        # add seizures "ICTAL" samples - class 1
        if self.cfg['lead_seizures_only']:
            ictals = seg_df[(seg_df['seg_type'] == 'ictal')
                            & (seg_df['time_to_prev_ictal'] > self.cfg['lead_seizure_separation'])].copy()
        else:
            ictals = seg_df[(seg_df['seg_type'] == 'ictal')].copy()

        ret_df = ictals.copy()
        ictals.loc[:, 'start_ts'] = ictals['start_ts'] - self.cfg['sample_offset_tmin']
        self.__add_samples(ictals, 1, max_samples=self.cfg['samples_per_ictal'],
                           interleave=self.cfg['ictal_interleave'])
        return ret_df

    def __add_sequential_preictals(self, ictals, n_classes):
        """ adds the preictals in the sequential-8class approach - immediately preceding the onset """
        # "PRE-ICTAL" - sequential pre-ictal sampling
        preictals = ictals.copy()
        for dist in range(n_classes - 2):
            preictals.loc[:, 'end_ts'] = preictals.loc[:, 'start_ts']
            preictals.loc[:, 'start_ts'] = preictals.loc[:, 'start_ts'] - self.cfg['sample_length']
            preictals = preictals[(preictals['start_ts'] >= 0) & (preictals['end_ts'] >= 0)]
            self.__add_samples(preictals, dist + 2, max_samples=1)

    def __add_simple_preictals(self, seg_df, label):
        """ adds preictal samples in case of conventional forecasting and threefold classification tasks """
        # "PRE-ICTAL" - get non-ictal segments suitable to obtain pre-ictal samples
        preictals = seg_df[(seg_df['seg_type'] == 'non-ictal')
                         & (seg_df['followed_by'] == 'ictal')
                         & (seg_df['duration'] > self.cfg['preictal_setback'] + self.cfg['preictal_length'])].copy()
        preictals.loc[:, 'end_ts'] = preictals['end_ts'] - self.cfg['preictal_setback']
        preictals.loc[:, 'start_ts'] = preictals['end_ts'] - self.cfg['preictal_length']
        self.__add_samples(preictals, label, max_samples=self.cfg['preictal_samples_per_session'],
                           interleave=self.cfg['preictal_interleave'])

    def __add_interictals(self, seg_df):
        """ adds the negative, interictal samples """
        # "INTERICTAL" - negative - class 0 - get non-ictal segments
        cfg = self.cfg
        postict = cfg['postictal_separation']
        preict = cfg['preictal_setback'] + cfg['preictal_length']
        L = cfg['sample_length']

        interictals = seg_df[(seg_df['seg_type'] == 'non-ictal')].copy()
        interictals = interictals[
            (interictals["time_to_prev_ictal"] > postict) &
            (interictals["time_to_next_ictal"] > preict)
            ].copy()

        self.__add_samples(
            interictals, 0,
            max_samples=cfg['interictal_samples_per_session'],
            interleave=cfg['interictal_interleave']
        )

    @property
    def number_of_classes(self):
        return (
            2 if self.cfg['sampling_strategy'].startswith('binary_')
            else 3 if self.cfg['sampling_strategy'].startswith('threefold_')
            else 8
        )

    def retrieve_samples(self):
        """ this method is populating the samples dataframe based on the configuration of the view """
        # TODO: check if the expected channels are overlapping the received signal in the expected manner

        # omit non-ictal sessions
        seg_df = self.core_ds.segments_df.merge(
                    self.core_ds.session_df[['patient_id','session_id','chunk_id','n_seizures']],
                    on=['patient_id','session_id','chunk_id'],
                    how='left'
                )
        if self.cfg['omit_nonictal_sessions']:
            seg_df = seg_df[seg_df['n_seizures'] != 0]

        if self.cfg['sampling_strategy'] == 'binary_forecasting':
            self.__add_simple_preictals(seg_df, 1)        # add pre-ictals as positive (1)
        else:
            ictals = self.__add_ictals(seg_df)                  # add ictals as positive (1)
            if self.cfg['sampling_strategy'] == 'threefold':
                self.__add_simple_preictals(seg_df, 2)  # add pre-ictals as third class (2)
            elif self.cfg['sampling_strategy'] == 'seq8class':
                self.__add_sequential_preictals(ictals, n_classes=8)    # add seq. pre-ictals (2-7)

        self.__add_interictals(seg_df)                          # add interictals as negative (0)

        self.samples_df = self.samples_df.reset_index(drop=True)

    def split_samples(self):
        """ splits the samples according to the config into a train, validation and test set """
        ratio = self.cfg['split_ratio']

        if self.cfg['split'] == 'patient':
            # patient split
            patients = self.samples_df['patient'].unique()
            n_patients = len(patients)
            i1, i2 = int(n_patients * ratio[0]), int(n_patients * (ratio[0] + ratio[1]))
            lots = patients[:i1], patients[i1:i2], patients[i2:]
            print(f"...patient split: {n_patients} : [{len(lots[0])}, {len(lots[1])}, {len(lots[2])}]")
            self.samples_df['split'] = np.where(self.samples_df['patient'].isin(lots[0]), 'train',
                                                np.where(self.samples_df['patient'].isin(lots[1]), 'val', 'test'))
        elif self.cfg['split'] == 'temporal':
            patients = self.samples_df['patient'].unique()
            for p in patients:
                psamp = self.samples_df[self.samples_df['patient'] == p]
                psamp = psamp.sort_values(['timestamp', 'start_index'])
                n = len(psamp)
                if n > 10:
                    i1, i2 = int(n * ratio[0]), int(n * (ratio[0] + ratio[1]))
                    self.samples_df.loc[psamp.index[:i1], 'split'] = 'train'
                    self.samples_df.loc[psamp.index[i1:i2], 'split'] = 'val'
                    self.samples_df.loc[psamp.index[i2:], 'split'] = 'test'
                else:
                    self.samples_df.loc[psamp.index, 'split'] = 'train'
        else:
            # random split
            labels = self.samples_df['label'].unique()
            for l in labels:
                psamp = self.samples_df[self.samples_df['label'] == l]
                np.random.seed(76)
                indices = np.random.permutation(psamp.index)
                n = len(indices)
                i1, i2 = int(n * ratio[0]), int(n * (ratio[0] + ratio[1]))
                self.samples_df.loc[indices[:i1], 'split'] = 'train'
                self.samples_df.loc[indices[i1:i2], 'split'] = 'val'
                self.samples_df.loc[indices[i2:], 'split'] = 'test'

    def __len__(self):
        return len(self.samples_df)

    def normalize(self, X):
        """ normalize X, subtract mean and divide by 95% quantile per channel """
        # X is (n_channels, n_samples)
        # subtract mean for each channel
        X = X - np.mean(X, axis=-1, keepdims=True)
        # normalize --> TODO might be non-optimal for very short signals
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        return X

    def preprocess_sample(self, signal):
        """ preprocess a single sample """
        # TODO: also invoke model-specific preprocessing here (if provided) - to benefit more from lazy caching

        signal = self.normalize(signal)         # normalization
        signal = torch.FloatTensor(signal)      # convert to tensor
        return signal

    def __set_explicit_types(self):
        """ auxiliary method for enabling storage in HDF5 """
        # explicit type - for saving in hdf format
        self.samples_df['patient'] = self.samples_df['patient'].astype(str)
        self.samples_df['filename'] = self.samples_df['filename'].astype(str)
        self.samples_df['start_index'] = self.samples_df['start_index'].astype(int)
        self.samples_df['end_index'] = self.samples_df['end_index'].astype(int)
        self.samples_df['label'] = self.samples_df['label'].astype(int)
        self.samples_df['split'] = self.samples_df['split'].astype(str)


    def get_len_type(self, ds_type):
        """ retrieve the length of the dataset of the given type: train,val,test """
        return len(self.samples_df[self.samples_df['split'] == ds_type])

    def __getitem__(self, index):
        """ return a single item """
        return self.get_item_of_type(None, index)

    def get_item_of_type(self, ds_type, index):
        """ return the indexed item in the given dataset type: train, val, test """
        df = self.samples_df[self.samples_df['split'] == ds_type] if ds_type else self.samples_df

        # get sample from core dataset based on specs
        filename, start_index, end_index, label = df.iloc[index][['filename', 'start_index', 'end_index', 'label']]

        # load data
        results = load_partial_hdf5(os.path.join(filename), start_idx=int(start_index),
                                    stop_idx=int(end_index), only_signal=True, channel_order=self.cfg['signal_channels'])
        signal = results["signal"]
        signal = self.preprocess_sample(signal)

        metadata = {'index': index, 'filename': filename, 'start_index': start_index, 'end_index': end_index, 'label': label }

        return signal, label, metadata

    def print_label_split_stats(self):
        # Basic safety: keep only needed cols
        cols = ['patient', 'timestamp', 'filename', 'start_index', 'end_index', 'label', 'split']
        df = self.samples_df[cols].copy()

        # 1) counts by split × label
        counts = pd.crosstab(df['split'], df['label'], margins=True, margins_name='Total')
        print("\n=== Counts by split × label ===")
        print(counts)

        # 2) row-wise percentages (within each split)
        row_pct = pd.crosstab(df['split'], df['label'], normalize='index') * 100
        print("\n=== % within each split (rows) ===")
        print(row_pct.round(2))

        # 3) column-wise percentages (within each label)
        col_pct = pd.crosstab(df['split'], df['label'], normalize='columns') * 100
        print("\n=== % within each label (columns) ===")
        print(col_pct.round(2))