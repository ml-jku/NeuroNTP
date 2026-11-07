from __future__ import annotations

import os
import re
from pathlib import Path

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Iterable, Any
import h5py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

import mne

from neuro_ntp.data.epi_dataset import EpilepsiaCoreDataset


# ----------------------------
# small utilities
# ----------------------------

def _hms_to_sec(hms: str) -> int:
    # "22.34.48" -> seconds since 00:00:00
    parts = re.split(r"[.:]", hms.strip())
    if len(parts) != 3:
        raise ValueError(f"Bad HMS: {hms}")
    h, m, s = (int(parts[0]), int(parts[1]), int(parts[2]))
    return h * 3600 + m * 60 + s

def _diff_clock_sec(t1: int, t0: int) -> int:
    """Difference in seconds on a 24h clock, allowing wrap around midnight."""
    d = t1 - t0
    if d < 0:
        d += 24 * 3600
    return d

def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _normalize_torch_quantile95(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: (C, L) float
    x = x - x.mean(dim=-1, keepdim=True)
    q = torch.quantile(torch.abs(x), q=0.95, dim=-1, keepdim=True)
    return x / (q + eps)

def _downsample_512_to_256_decimate(x: torch.Tensor) -> torch.Tensor:
    # x: (C, L_512) -> (C, L_256)
    return x[..., ::2].contiguous()

def _read_edf_header(edf_path: str) -> Tuple[Optional[datetime], float, int, int]:
    """
    Returns: (meas_date, sfreq, n_times, n_channels) without loading signals.
    """

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    try:
        meas_date = raw.info.get("meas_date")
        sfreq = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        n_ch = int(raw.info["nchan"])
    finally:
        raw.close()
    return meas_date, sfreq, n_times, n_ch

def _read_edf_segment(
    raw: mne.io.BaseRaw,
    start_samp: int,
    stop_samp: int,
    picks: Optional[List[int]] = None,
) -> np.ndarray:
    """
    raw: opened MNE Raw (preload=False is fine)
    returns float64 ndarray (n_ch, n_samples) in Volts
    """
    data = raw.get_data(picks=picks, start=start_samp, stop=stop_samp, reject_by_annotation=None)
    return data

def _read_h5_header(h5_path: str) -> Tuple[float, int, int, List[str]]:
    """
    Returns: (sfreq, signal_length, n_channels, ch_names) from attrs without reading signal.
    """
    with h5py.File(h5_path, "r") as h:
        sfreq = float(h.attrs["sfreq"])
        signal_length = int(h.attrs["signal_length"])
        n_channels = int(h.attrs["n_channels"])
        ch_names = list(h.attrs.get("ch_names", []))
    return sfreq, signal_length, n_channels, ch_names



class SienaCoreDataset(EpilepsiaCoreDataset):
    """
    Core dataset of the Siena Scalp EEG Database, built from the outputs of the HDF5 conversion (siena2hdf.py).

    Construction:
        SienaCoreDataset(root_dir=...)       build from sessions.csv and annotations.csv in the HDF5 root
        SienaCoreDataset(ds_filepath=...)    load a core dataset previously saved with save()

    Tables:
        patients_df    one row per patient
        session_df     one row per recording (session_id = record name), with file path, sampling rate, signal
                       length, duration, number of seizures and channel names
        markers_df     seizure annotations (onset and duration in seconds per recording)

    The saved core dataset contains metadata only; the signal data remains in the HDF5 store.
    """

    def __init__(
        self,
        root_dir: str = None,              # keep for backwards compat / optional
        ds_filepath: str = None,
    ):
        super().__init__()

        self.ds_filepath = ds_filepath
        if self.ds_filepath:
            self.__load()
            return

        self.hdf_store_path = root_dir
        self.name = "SienaCoreDS"
        self._load_from_csvs(Path(root_dir) / "sessions.csv", Path(root_dir) / "annotations.csv")


    def save(self, ds_filepath: str):
        """ auxiliary method for persisting the core dataset as an HDF5 file """
        print(f"...saving core dataset {self.name} to {ds_filepath}")
        #self.__set_explicit_types()
        with pd.HDFStore(ds_filepath, 'w') as hfile:
            hfile.put('params', pd.DataFrame())
            hfile.get_storer('params').attrs['name'] = self.name
            hfile.get_storer('params').attrs['hdf_store_path'] = self.hdf_store_path
            hfile.put('patients', self.patients_df)
            hfile.put('sessions', self.session_df)
            hfile.put('markers', self.markers_df)

    def __load(self):
        """ auxiliary method for retrieving the core dataset object from an HDF5 file """
        print(f"...loading core dataset {self.name} from {self.ds_filepath}")
        with pd.HDFStore(self.ds_filepath, 'r') as hfile:
            self.name = hfile.get_storer('params').attrs['name']
            self.hdf_store_path = hfile.get_storer('params').attrs['hdf_store_path']
            self.patients_df = hfile['patients']
            self.session_df = hfile['sessions']
            self.markers_df = hfile['markers']



    def _load_from_csvs(self, sessions_csv: str, annotations_csv: Optional[str]):
        sess = pd.read_csv(sessions_csv)

        # minimal sanity
        need = {"patient_id", "record_name", "filename", "sfreq", "n_channels", "signal_length", "channels"}
        missing = need - set(sess.columns)
        if missing:
            raise ValueError(f"sessions.csv missing columns: {sorted(missing)}")

        # patients table
        self.patients_df = pd.DataFrame({"patient_id": sorted(sess["patient_id"].unique())})

        # sessions table: keep naming consistent with the rest of neuro_ntp:
        # - patient_id
        # - session_id (use record_name, stable)
        # - filename (path to .h5)
        # - sfreq, signal_length, n_channels, duration_s, channels
        sess = sess.copy()
        sess["duration_s"] = sess["signal_length"] / sess["sfreq"]

        self.session_df = pd.DataFrame({
            "patient_id": sess["patient_id"].astype(str),
            "session_id": sess["record_name"].astype(str),     # IMPORTANT: use record_name
            "filename": sess["filename"].astype(str),
            "sfreq": sess["sfreq"].astype(float),
            "n_channels": sess["n_channels"].astype(int),
            "signal_length": sess["signal_length"].astype(int),
            "duration_s": sess["duration_s"].astype(float),
            "n_seizures": sess.get("n_seizures", 0).fillna(0).astype(int),
            "channels": sess["channels"].astype(str),
        })

        # optional annotations table (not required for "core", but extremely useful for the view)
        self.markers_df = None
        if annotations_csv:
            ann = pd.read_csv(annotations_csv)
            need_a = {"patient_id", "record_name", "filename", "onset_sec", "duration_sec", "description"}
            missing_a = need_a - set(ann.columns)
            if missing_a:
                raise ValueError(f"annotations.csv missing columns: {sorted(missing_a)}")

            self.markers_df = ann.copy()
            # normalize types
            self.markers_df["patient_id"] = self.markers_df["patient_id"].astype(str)
            self.markers_df["session_id"] = self.markers_df["record_name"].astype(str)  # alias
            self.markers_df["onset_sec"] = self.markers_df["onset_sec"].astype(float)
            self.markers_df["duration_sec"] = self.markers_df["duration_sec"].astype(float)
            self.markers_df["description"] = self.markers_df["description"].astype(str)



@dataclass
class SienaSeizureEvent:
    patient_id: str
    session_id: str  # EDF filename
    seizure_start_s: float  # in-session seconds
    seizure_end_s: float    # in-session seconds


# --- drop-in replacements / additions ---

_SEIZ_FILE_RE = re.compile(r"^Seizures-list-(PN\d{2})\.txt$", re.IGNORECASE)

# capture a *time token* anywhere (allows "." or ":" separators)
_TIME_RE = re.compile(r"(\d{1,2}[.:]\d{2}[.:]\d{2})")

# helper so we can embed the same capture into multiple regexes
_TIME = r"(\d{1,2}[.:]\d{2}[.:]\d{2})"

_RE_FILE = re.compile(r"^\s*File\s+name\s*:\s*(PN\d{2}.*?\.edf)\b", re.IGNORECASE)

# IMPORTANT: these now capture ONLY the first HH.MM.SS/HH:MM:SS token after the label
_RE_REG_START = re.compile(r"^\s*Registration\s+start\s+time\s*:\s*" + _TIME, re.IGNORECASE)
_RE_REG_END   = re.compile(r"^\s*Registration\s+end\s+time\s*:\s*"   + _TIME, re.IGNORECASE)
_RE_SZ_START  = re.compile(r"^\s*Seizure\s+start\s+time\s*:\s*"       + _TIME, re.IGNORECASE)
_RE_SZ_END    = re.compile(r"^\s*Seizure\s+end\s+time\s*:\s*"         + _TIME, re.IGNORECASE)

def _find_time(line: str) -> str | None:
    m = _TIME_RE.search(line)
    return m.group(1) if m else None


def _resolve_edf_from_registration(patient_dir: str, reg_start_hms: str, reg_end_hms: str) -> Optional[str]:
    # If only one EDF: pick it.
    edfs = sorted([f for f in os.listdir(patient_dir) if f.lower().endswith(".edf")])
    if not edfs:
        return None
    if len(edfs) == 1:
        return edfs[0]

    rs = _hms_to_sec(reg_start_hms)
    re_ = _hms_to_sec(reg_end_hms)
    reg_dur = _diff_clock_sec(re_, rs)

    best = None
    best_err = float("inf")
    for fn in edfs:
        edf_path = os.path.join(patient_dir, fn)
        _, sfreq, n_times, _ = _read_edf_header(edf_path)
        dur_s = n_times / float(sfreq)
        err = abs(dur_s - reg_dur)
        if err < best_err:
            best_err = err
            best = fn
    return best

def parse_siena_seizure_lists(root_dir: str) -> List[SienaSeizureEvent]:
    """
    Reads all Seizures-list-PNxx.txt and returns seizure intervals in seconds w.r.t. the EDF start.
    Robust against trailing notes like "(CLINICAL ONSET)" or "opure ...".
    """
    root_dir = os.path.abspath(root_dir)
    events: List[SienaSeizureEvent] = []

    for pat in sorted(os.listdir(root_dir)):
        if not re.fullmatch(r"PN\d{2}", pat):
            continue
        patient_dir = os.path.join(root_dir, pat)
        if not os.path.isdir(patient_dir):
            continue

        seiz_files = [f for f in os.listdir(patient_dir) if _SEIZ_FILE_RE.match(f)]
        if not seiz_files:
            continue
        seiz_path = os.path.join(patient_dir, seiz_files[0])

        with open(seiz_path, "r", errors="ignore") as f:
            lines = [ln.rstrip("\n") for ln in f.readlines()]

        cur_file = None
        reg_start = None
        reg_end = None
        sz_start = None
        sz_end = None

        def flush_if_complete():
            nonlocal cur_file, reg_start, reg_end, sz_start, sz_end
            if cur_file and reg_start and sz_start and sz_end:
                rs = _hms_to_sec(reg_start)
                ss = _hms_to_sec(sz_start)
                se = _hms_to_sec(sz_end)
                start_s = float(_diff_clock_sec(ss, rs))
                end_s = float(_diff_clock_sec(se, rs))
                if end_s <= start_s:
                    end_s = start_s + 1.0

                events.append(SienaSeizureEvent(
                    patient_id=pat,
                    session_id=cur_file,
                    seizure_start_s=start_s,
                    seizure_end_s=end_s,
                ))

            # allow multiple seizures per same EDF: keep reg_start/reg_end & cur_file
            sz_start = None
            sz_end = None

        for ln in lines:
            s = ln.strip()

            m = _RE_FILE.match(s)
            if m:
                # new EDF context
                cur_file = m.group(1).strip()
                reg_start = None
                reg_end = None
                sz_start = None
                sz_end = None
                continue

            m = _RE_REG_START.match(s)
            if m:
                reg_start = m.group(1)
                continue

            m = _RE_REG_END.match(s)
            if m:
                reg_end = m.group(1)
                if cur_file and cur_file.endswith("-.edf"):
                    resolved = _resolve_edf_from_registration(
                        patient_dir,
                        reg_start or "00.00.00",
                        reg_end or "00.00.00"
                    )
                    if resolved is not None:
                        cur_file = resolved
                continue

            m = _RE_SZ_START.match(s)
            if m:
                sz_start = m.group(1)
                continue

            m = _RE_SZ_END.match(s)
            if m:
                sz_end = m.group(1)
                flush_if_complete()
                continue

            # Optional fallback: if a line starts with "Seizure ... time" but regex didn't match,
            # try to extract the first time token anywhere on the line.
            low = s.lower()
            if low.startswith("registration start time"):
                reg_start = _find_time(s) or reg_start
            elif low.startswith("registration end time"):
                reg_end = _find_time(s) or reg_end
            elif low.startswith("seizure start time"):
                sz_start = _find_time(s) or sz_start
            elif low.startswith("seizure end time"):
                sz_end = _find_time(s) or sz_end
                if sz_end is not None:
                    flush_if_complete()

    return events