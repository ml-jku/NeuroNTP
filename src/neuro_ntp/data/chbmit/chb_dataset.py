"""
+----------------------------------------------------------------------------------------------------------------------+
chb_dataset.py
Core dataset of the CHB-MIT Scalp EEG Database (Children's Hospital Boston / Massachusetts Institute of Technology)

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

    The core dataset is built from the HDF5 store created by chb2hdf.py (the path to the store is the input). The
    patient, session, segment (ictal/non-ictal) and marker tables are extracted from the HDF5 files. The core dataset
    can serve both supervised and self-supervised dataset views.
+----------------------------------------------------------------------------------------------------------------------+
"""
import re
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from neuro_ntp.data.epi_dataset import EpilepsiaCoreDataset


class CHBMITCoreDataset(EpilepsiaCoreDataset):
    """
    Core dataset of the CHB-MIT Scalp EEG Database, built from the outputs of the HDF5 conversion (chb2hdf.py):
        sessions.csv       one row per recording (EDF file / HDF5 file)
        annotations.csv    zero or more seizure annotations per recording
        HDF5 files         signal data, one file per recording, in the HDF5 store (hdf_store_path)

    Construction:
        CHBMITCoreDataset(name=..., hdf_store_path=...)    build from the HDF5 store
        CHBMITCoreDataset(ds_filepath=...)                 load a core dataset previously saved with save()

    The saved core dataset contains metadata only and references the signal data in the HDF5 store.
    """

    DEFAULT_CFG = {
        "markers": {
            "seizure_label": "seizure",
        },
        "policy": {
            "default_chunk_duration": 3600.0,  # 1 hour
        },
    }

    def __init__(
        self,
        name: str = "CHBMITCoreDS",
        ds_filepath: str = None,
        hdf_store_path: str = None,
        cfg: dict = None,
    ):
        cfg = cfg or self.DEFAULT_CFG
        super().__init__(
            name=name,
            ds_filepath=ds_filepath,
            hdf_store_path=hdf_store_path,
            cfg=cfg,
        )

        if self.ds_filepath:
            self._load()
        else:
            self.init_from_input()
            self._calc_segment_stats()

    def _load_sessions_csv(self) -> pd.DataFrame:
        csv_path = Path(self.hdf_store_path) / "sessions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"missing sessions.csv at {csv_path}")
        df = pd.read_csv(csv_path)
        return df

    def _load_annotations_csv(self) -> pd.DataFrame:
        csv_path = Path(self.hdf_store_path) / "annotations.csv"
        if not csv_path.exists():
            # no seizures anywhere: create empty DF
            return pd.DataFrame(
                columns=["patient_id", "record_name", "filename", "onset_sec", "duration_sec", "description", "source"]
            )
        df = pd.read_csv(csv_path)
        return df

    def init_from_input(self):
        print(f"...initializing new CHBMIT core dataset: {self.name}")

        sessions_df = self._load_sessions_csv()
        ann_df = self._load_annotations_csv()

        # 1) build session_df
        self.session_df = self._build_session_df(sessions_df)

        # 2) build markers_df from annotations.csv
        self.markers_df = self._build_markers_df_from_csv(ann_df)

        # 3) build segments_df from sessions + markers
        self.segments_df = self._build_segments_df_from_annotations(self.session_df, ann_df)

        # 4) patients_df aggregation
        self.patients_df = self._build_patients_df(self.session_df)

        print("...done building core_ds.")

    def _build_session_df(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        root = Path(self.hdf_store_path)

        # duration: if missing
        if "signal_length" in sessions_df.columns and "sfreq" in sessions_df.columns:
            sessions_df["chunk_duration_used_sec"] = sessions_df["signal_length"] / sessions_df["sfreq"]
        else:
            sessions_df["chunk_duration_used_sec"] = self.cfg["policy"]["default_chunk_duration"]

        # measurement_start_time: if present
        if "file_start_time" in sessions_df.columns:
            # summary had only time (HH:MM:SS), we synthesize date
            def _to_dt(x):
                try:
                    hh, mm, ss = str(x).split(":")
                    return datetime(2000, 1, 1, int(hh), int(mm), int(ss))
                except Exception:
                    return datetime(2000, 1, 1)
            sessions_df["measurement_start_time"] = sessions_df["file_start_time"].apply(_to_dt)
        else:
            sessions_df["measurement_start_time"] = datetime(2000, 1, 1)

        # add session_id, chunk_id
        # CHB is basically (patient, record_name)
        # record_name is e.g. chb01_03 → we can parse the trailing number
        def _parse_rec(rn: str) -> int:
            m = re.search(r"_(\d+)$", rn)
            return int(m.group(1)) if m else 0

        sessions_df["session_id"] = sessions_df["record_name"].apply(_parse_rec)
        sessions_df["chunk_id"] = 0

        # channels: wrote during conversion
        if "channels" not in sessions_df.columns:
            sessions_df["channels"] = ""

        # placeholder for counts
        sessions_df["n_markers"] = 0
        sessions_df["n_seizures"] = 0

        cols = [
            "patient_id",
            "session_id",
            "chunk_id",
            "measurement_start_time",
            "filename",
            "hdf_size_gb",
            "chunk_duration_used_sec",
            "channels",
            "n_markers",
            "n_seizures",
        ]

        # file size
        sessions_df["hdf_size_gb"] = sessions_df["filename"].apply(
            lambda p: os.path.getsize(p) / 1e9 if os.path.exists(p) else 0.0
        )

        # reorder / pick
        session_df = sessions_df[cols].copy().sort_values(
            ["patient_id", "session_id", "chunk_id"]
        ).reset_index(drop=True)

        return session_df

    def _build_markers_df_from_csv(self, ann_df: pd.DataFrame) -> pd.DataFrame:
        """
        annotations.csv → markers_df

        We deliberately convert every seizure row into TWO markers:
          - <seizure_delimiter_markers[0]> at onset
          - <seizure_delimiter_markers[1]> at onset + duration

        So CHB, which only has timing, becomes compatible with the KUK/TUSZ
        “EEG-Beginn / EEG-Ende” style.

        Expected ann_df columns:
          - patient_id
          - record_name
          - onset (seconds)
          - duration (seconds)
        description is ignored except for fallback.
        """
        if ann_df.empty:
            return pd.DataFrame(
                columns=[
                    "patient_id",
                    "session_id",
                    "chunk_id",
                    "onset",
                    "epoch_onset",
                    "duration",
                    "description",
                    "marker",
                    "mk_state",
                    "mk_notes",
                    "filename",
                ]
            )

        # get the two marker names from config
        # e.g. cfg["seizure_delimiter_markers"] = ["eeg-beginn", "eeg-ende"]
        delim = self.cfg.get("seizure_delimiter_markers", None)
        if not delim or len(delim) < 2:
            # sensible default, in case cfg didn’t define it
            delim = ["eeg-beginn", "eeg-ende"]

        start_marker, end_marker = delim[0], delim[1]

        # session info to join filename + start time
        sess = self.session_df[[
            "patient_id",
            "session_id",
            "chunk_id",
            "filename",
            "measurement_start_time",
        ]].copy()
        # derive record_name from filename (chb01_03.h5 → chb01_03)
        sess["record_name"] = sess["filename"].apply(lambda p: Path(p).stem)

        merged = ann_df.merge(
            sess,
            on=["patient_id", "record_name"],
            how="left",
            validate="m:1",
        )

        rows = []
        for _, r in merged.iterrows():
            onset = float(r["onset_sec"])
            dur = float(r["duration_sec"]) if not pd.isna(r["duration_sec"]) else 0.0
            mst = r["measurement_start_time"]
            fname = r["filename_x"]

            # 1) BEGIN marker
            epoch_onset = mst + timedelta(seconds=onset) if pd.notna(mst) else pd.NaT
            rows.append({
                "patient_id": r["patient_id"],
                "session_id": int(r["session_id"]),
                "chunk_id": int(r["chunk_id"]),
                "onset": onset,
                "epoch_onset": epoch_onset,
                "duration": 0.0,
                "description": start_marker,
                "marker": start_marker,
                "mk_state": pd.NA,
                "mk_notes": pd.NA,
                "filename": fname,
            })

            # 2) END marker
            end_time = onset + dur
            epoch_end = mst + timedelta(seconds=end_time) if pd.notna(mst) else pd.NaT
            rows.append({
                "patient_id": r["patient_id"],
                "session_id": int(r["session_id"]),
                "chunk_id": int(r["chunk_id"]),
                "onset": end_time,
                "epoch_onset": epoch_end,
                "duration": 0.0,
                "description": end_marker,
                "marker": end_marker,
                "mk_state": pd.NA,
                "mk_notes": pd.NA,
                "filename": fname,
            })

        mk_df = pd.DataFrame(rows)
        mk_df.sort_values(["patient_id", "session_id", "chunk_id", "onset"], inplace=True)
        mk_df.reset_index(drop=True, inplace=True)
        return mk_df

    def _build_segments_df_from_annotations(
        self,
        session_df: pd.DataFrame,
        ann_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build segments_df directly from annotations.csv (onset_sec + duration_sec),
        then fill remaining time with non-ictal segments.
        """
        seg_rows = []

        # If there are no annotations at all → every session becomes a single non-ictal segment
        if ann_df is None or ann_df.empty:
            for _, sess in session_df.iterrows():
                pid = sess["patient_id"]
                sid = int(sess["session_id"])
                cid = int(sess["chunk_id"])
                fname = sess["filename"]
                mst = sess["measurement_start_time"]
                total_dur = float(sess["chunk_duration_used_sec"])
                seg_rows.append({
                    "patient_id": pid, "session_id": sid, "chunk_id": cid,
                    "start_ts": 0.0, "end_ts": total_dur, "duration": total_dur,
                    "seg_type": "non-ictal",
                    "epoch_start_ts": mst,
                    "epoch_end_ts": mst + timedelta(seconds=total_dur),
                    "filename": fname,
                })
            seg_df = pd.DataFrame(seg_rows)
            if not seg_df.empty:
                seg_df.sort_values(["patient_id", "session_id", "chunk_id", "start_ts"], inplace=True)
                seg_df.reset_index(drop=True, inplace=True)
            return seg_df

        # Join annotations to sessions to get filename, start time, etc.
        sess = session_df[[
            "patient_id", "session_id", "chunk_id", "filename", "measurement_start_time", "chunk_duration_used_sec"
        ]].copy()
        # record_name is typically the stem of the filename (e.g., chb01_03)
        sess["record_name"] = sess["filename"].apply(lambda p: Path(p).stem)

        merged = ann_df.merge(
            sess,
            on=["patient_id", "record_name"],
            how="left",
            validate="m:1",
        )

        # Group annotations per (patient, session)
        by_ps = {
            (pid, int(sid)): g.sort_values("onset_sec").reset_index(drop=True)
            for (pid, sid), g in merged.groupby(["patient_id", "session_id"])
        }

        for _, sess in session_df.iterrows():
            pid = sess["patient_id"]
            sid = int(sess["session_id"])
            cid = int(sess["chunk_id"])
            fname = sess["filename"]
            mst = sess["measurement_start_time"]
            total_dur = float(sess["chunk_duration_used_sec"])

            g = by_ps.get((pid, sid), None)

            if g is None or g.empty:
                # no seizures in this recording
                seg_rows.append({
                    "patient_id": pid, "session_id": sid, "chunk_id": cid,
                    "start_ts": 0.0, "end_ts": total_dur, "duration": total_dur,
                    "seg_type": "non-ictal",
                    "epoch_start_ts": mst,
                    "epoch_end_ts": mst + timedelta(seconds=total_dur),
                    "filename": fname,
                })
                continue

            # Build ictal intervals from (onset_sec, duration_sec)
            ictals = []
            for _, r in g.iterrows():
                s = float(r["onset_sec"])
                d = float(r["duration_sec"]) if not pd.isna(r["duration_sec"]) else 0.0
                e = s + max(0.0, d)
                # Clip to file bounds
                s = max(0.0, min(total_dur, s))
                e = max(0.0, min(total_dur, e))
                if e > s:
                    ictals.append((s, e))

            # Merge touching/overlapping ictals
            ictals.sort()
            merged_ictals = []
            if ictals:
                cs, ce = ictals[0]
                for s, e in ictals[1:]:
                    if s <= ce:  # overlap or touch
                        ce = max(ce, e)
                    else:
                        merged_ictals.append((cs, ce))
                        cs, ce = s, e
                merged_ictals.append((cs, ce))

            # Complement → non-ictal segments
            nonictals = []
            cursor = 0.0
            for s, e in merged_ictals:
                if s > cursor:
                    nonictals.append((cursor, s))
                cursor = max(cursor, e)
            if cursor < total_dur:
                nonictals.append((cursor, total_dur))

            # Emit ictal rows
            for s, e in merged_ictals:
                seg_rows.append({
                    "patient_id": pid, "session_id": sid, "chunk_id": cid,
                    "start_ts": s, "end_ts": e, "duration": e - s,
                    "seg_type": "ictal",
                    "epoch_start_ts": mst + timedelta(seconds=s),
                    "epoch_end_ts": mst + timedelta(seconds=e),
                    "filename": fname,
                })
            # Emit non-ictal rows
            for s, e in nonictals:
                seg_rows.append({
                    "patient_id": pid, "session_id": sid, "chunk_id": cid,
                    "start_ts": s, "end_ts": e, "duration": e - s,
                    "seg_type": "non-ictal",
                    "epoch_start_ts": mst + timedelta(seconds=s),
                    "epoch_end_ts": mst + timedelta(seconds=e),
                    "filename": fname,
                })

        seg_df = pd.DataFrame(seg_rows)
        if not seg_df.empty:
            seg_df.sort_values(["patient_id", "session_id", "chunk_id", "start_ts"], inplace=True)
            seg_df.reset_index(drop=True, inplace=True)
        return seg_df

    def _build_patients_df(self, session_df: pd.DataFrame) -> pd.DataFrame:
        # aggregate duration per patient
        agg = (
            session_df
            .groupby("patient_id", as_index=False)
            .agg(
                n_sessions=("session_id", "nunique"),
                total_duration_sec=("chunk_duration_used_sec", "sum"),
            )
        )
        agg["n_seizures"] = 0  # filled in _calc_segment_stats
        return agg

    def _calc_segment_stats(self):
        # mark n_markers per session
        if self.markers_df is not None and not self.markers_df.empty:
            mk_counts = (
                self.markers_df
                .groupby(["patient_id", "session_id", "chunk_id"], as_index=False)
                .size()
                .rename(columns={"size": "n_markers"})
            )
        else:
            mk_counts = self.session_df[["patient_id", "session_id", "chunk_id"]].copy()
            mk_counts["n_markers"] = 0

        # seizures per session
        if self.segments_df is not None and not self.segments_df.empty:
            ict_counts = (
                self.segments_df[self.segments_df["seg_type"] == "ictal"]
                .groupby(["patient_id", "session_id", "chunk_id"], as_index=False)
                .size()
                .rename(columns={"size": "n_seizures"})
            )
        else:
            ict_counts = self.session_df[["patient_id", "session_id", "chunk_id"]].copy()
            ict_counts["n_seizures"] = 0

        key = ["patient_id", "session_id", "chunk_id"]
        self.session_df = (
            self.session_df
            .merge(mk_counts, on=key, how="left")
            .merge(ict_counts, on=key, how="left")
        )
        self.session_df["n_seizures"] = self.session_df["n_seizures_x"].fillna(0).astype(int)
        self.session_df["n_markers"] = self.session_df["n_seizures"] * 2
        self.session_df = self.session_df.drop(['n_markers_x', 'n_markers_y', 'n_seizures_x', 'n_seizures_y'], axis=1)

        # patients: sum seizures
        pat_seiz = (
            self.session_df
            .groupby("patient_id")["n_seizures"]
            .sum()
            .rename("n_seizures")
            .reset_index()
        )
        self.patients_df = (
            self.patients_df
            .drop(columns=["n_seizures"], errors="ignore")
            .merge(pat_seiz, on="patient_id", how="left")
        )
        self.patients_df["n_seizures"] = self.patients_df["n_seizures"].fillna(0).astype(int)

    def _set_explicit_marker_types(self):
        if getattr(self, "markers_df", None) is None or self.markers_df.empty:
            return
        m = self.markers_df.copy()

        # force to string/categorical-like columns
        for col in ["patient_id", "description", "marker", "mk_state", "mk_notes", "filename"]:
            if col in m.columns:
                m[col] = m[col].astype(str)

        # numeric columns
        for col in ["session_id", "chunk_id", "onset", "duration"]:
            if col in m.columns:
                m[col] = pd.to_numeric(m[col], errors="coerce")

        # datetimes
        if "epoch_onset" in m.columns:
            m["epoch_onset"] = pd.to_datetime(m["epoch_onset"], errors="coerce")

        self.markers_df = m

    def save(self, ds_filepath: str):
        print(f"...saving CHBMIT core dataset to {ds_filepath}")
        self._set_explicit_marker_types()
        with pd.HDFStore(ds_filepath, "w") as h:
            h.put("params", pd.DataFrame())
            h.get_storer("params").attrs["name"] = self.name
            h.get_storer("params").attrs["hdf_store_path"] = self.hdf_store_path
            h.put("config", pd.Series(json.dumps(self.cfg)))
            h.put("patients", self.patients_df)
            h.put("sessions", self.session_df)
            h.put("markers", self.markers_df)
            h.put("segments", self.segments_df)

    def _load(self):
        print(f"...loading CHBMIT core dataset from {self.ds_filepath}")
        with pd.HDFStore(self.ds_filepath, "r") as h:
            self.name = h.get_storer("params").attrs["name"]
            self.hdf_store_path = h.get_storer("params").attrs["hdf_store_path"]
            self.cfg = json.loads(h["config"].item())
            self.patients_df = h["patients"]
            self.session_df = h["sessions"]
            self.markers_df = h["markers"]
            self.segments_df = h["segments"]
