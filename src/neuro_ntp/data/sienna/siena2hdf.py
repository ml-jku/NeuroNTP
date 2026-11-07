#!/usr/bin/env python3
"""
+----------------------------------------------------------------------------------------------------------------------+
siena2hdf.py
Convert the Siena Scalp EEG Database to per-recording HDF5 files

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    First preprocessing step for the Siena Scalp EEG Database. Creates the storage structure expected by the
    Epilepsia dataset logic in nntp: one HDF5 file per EDF recording, plus CSV index tables. The core datasets and
    dataset views used for pretraining are generated from these files in a subsequent step.

    The source EDF recordings are converted to compressed HDF5, which allows efficient partial reads of the signal
    without decoding the full EDF. The metadata required for the later core/view generation is stored with each file:
    normalised channel names, sampling rate, signal length, patient and record identifiers, and the seizure
    annotations.

    Processing per recording:
      1. Channel names are normalised to the common naming scheme ("eeg fp1", "spo2", "hr", "plet", ...).
      2. Only channels from a fixed list of supported channels are kept, in the order of that list: 31 EEG
         electrodes (10-20 system plus FC/CP/F9/F10/P9/P10 extensions) and the SpO2, heart rate and
         plethysmogram channels. Supported channels missing from a recording are omitted, so the channel set can
         differ between recordings.
      3. The signal is resampled to the target rate with polyphase filtering (scipy.signal.resample_poly) and
         stored as float32. No further filtering is applied.
            4. Seizure intervals are parsed from the per-patient Seizures-list-PNxx.txt files. Clock times are converted to
         seconds relative to the registration start, allowing for wrap-around at midnight.

Input:
    Siena directory layout as distributed on PhysioNet; the database root is passed as the positional argument:
        <siena_root>/PNxx/PNxx-N.edf
        <siena_root>/PNxx/Seizures-list-PNxx.txt

    Optionally, a precomputed seizure table can be given with --seizure_csv, with the columns
        patient_id, session_id, reg_start_hms, reg_end_hms, seizure_start_hms, seizure_end_hms

Output:
    <out_hdf_root>/PNxx/PNxx-N.h5
    <out_hdf_root>/sessions.csv       one row per recording (file path, sfreq, channels, signal length, seizure count)
    <out_hdf_root>/annotations.csv    one row per seizure (onset and duration in seconds, source "seizures-list")

    HDF5 file contents:
        signal                                       float32 (n_channels, n_samples), gzip-compressed with shuffle
        annotations.{onset,duration,description}     seizure intervals (description "seizure")
        raw.ch_names                                 original EDF names of the stored channels
        root attributes                              patient_id, record_name, sfreq, signal_length, n_channels,
                                                     ch_names (normalised, in output order)

Usage:
    python siena2hdf.py /data/siena/1.0.0 --out_hdf_root /data/nntp/siena_hdf --n_jobs 8

    Defaults: resampling to 256 Hz, gzip level 4.

Notes:
    - Storage: approx. 17 GB for the Siena Scalp EEG Database v1.0.0 with the default settings.
    - Recordings are matched to seizure-list entries by file name: by exact match first, then without a trailing
      "-<n>" suffix, then by a unique prefix match within the same patient.
    - Seizures whose end time does not follow the start time are given a duration of 1 s.
    - Annotation onsets and durations are in seconds relative to the start of each recording.
    - The Siena Scalp EEG Database is available from PhysioNet under the Creative Commons Attribution 4.0 license
      and is not included in this repository.
    - Seizure-list entries for files named "PNxx-.edf" are resolved to the matching EDF file via the registration
      start and end times.
+----------------------------------------------------------------------------------------------------------------------+
"""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import h5py
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import scipy.signal
import os
import re
from dataclasses import dataclass

_EEG_PREFIX_RE = re.compile(r"^(EEG|EDF)\s+", re.IGNORECASE)
_ECG_PREFIX_RE = re.compile(r"^(ECG|EKG)\s*", re.IGNORECASE)
_PAT_RE = re.compile(r"(PN\d{2})", re.IGNORECASE)

@dataclass
class SienaSeizureEvent:
    patient_id: str
    session_id: str         # EDF file name from the seizure list (resolved for "PNxx-.edf" entries)
    seizure_start_s: float  # seconds relative to the registration start
    seizure_end_s: float    # seconds relative to the registration start


_SEIZ_FILE_RE = re.compile(r"^Seizures-list-(PN\d{2})\.txt$", re.IGNORECASE)

# time token with "." or ":" separators, e.g. "21.51.02", "21:51:02", "16:13.23"
_TIME = r"(\d{1,2}[.:]\d{2}[.:]\d{2})"
_TIME_RE = re.compile(_TIME)

# "File name: PN06-3.edf"; the letter "O" is also accepted in place of a zero in the patient number ("PNO6-1.edf")
_RE_FILE = re.compile(r"^\s*File\s+name\s*:\s*(PN[0-9O]{2}.*?\.edf)\b", re.IGNORECASE)

# "Registration start/end time: ...", "Seizure start/end time: ..." and the short form "Start/End time: ..."
_RE_FIELD = re.compile(
    r"^\s*(?P<field>registration\s+start|registration\s+end|(?:seizure\s+)?start|(?:seizure\s+)?end)"
    r"\s+time\s*:(?P<value>.*)$",
    re.IGNORECASE,
)


def _find_time(line: str) -> Optional[str]:
    m = _TIME_RE.search(line)
    return m.group(1) if m else None


def _normalize_edf_name(name: str) -> str:
    """Normalise an EDF file name from a seizure list, replacing a letter "O" in the patient number by a zero."""
    name = name.strip()
    return re.sub(r"^PN([0-9O]{2})", lambda m: "PN" + m.group(1).upper().replace("O", "0"), name,
                  flags=re.IGNORECASE)


def _parse_time_value(value: str) -> Optional[str]:
    """
    Extract the time token from the value part of a seizure-list field.

    Whitespace inside the leading numeric part is removed first (e.g. "1 6.49.25" -> "16.49.25"). Trailing notes
    such as "(CLINICAL ONSET)" are ignored.
    """
    m = re.match(r"\s*([\d\s.:]+)", value)
    if m:
        m_tok = _TIME_RE.match(re.sub(r"\s+", "", m.group(1)))
        if m_tok:
            return m_tok.group(1)
    return _find_time(value)


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


def _resolve_edf_from_registration(patient_dir: str, reg_start_hms: str, reg_end_hms: str) -> Optional[str]:
    """
    Resolve an incomplete EDF file name ("PNxx-.edf") to the EDF file in patient_dir whose duration best matches
    the registration interval.
    """
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
    Read all Seizures-list-PNxx.txt files below root_dir and return the seizure intervals in seconds relative to the
    registration (EDF) start.

    The following irregularities of the original seizure lists are handled:
      - "Start time:" / "End time:" without the "Seizure" prefix
      - the letter "O" in place of a zero in file names ("PNO6-1.edf")
      - repeated "File name:" entries without registration times (the times given earlier for that file are used)
      - whitespace inside time values ("1 6.49.25") and mixed separators ("16:13.23")
      - trailing notes after time values ("(CLINICAL ONSET)")
      - incomplete file names ("PNxx-.edf"), resolved via the registration interval
    Entries that remain incomplete are reported and skipped.

    Args:
        root_dir: Siena database root containing the PNxx directories.

    Returns:
        List of SienaSeizureEvent, one per seizure.
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

        # registration context per file name: (reg_start, reg_end, resolved file name)
        reg_ctx: Dict[str, Tuple[Optional[str], Optional[str], str]] = {}
        cur_name: Optional[str] = None  # file name as given in the list
        cur_file: Optional[str] = None  # file name after resolution
        reg_start: Optional[str] = None
        reg_end: Optional[str] = None
        sz_start: Optional[str] = None
        sz_end: Optional[str] = None

        def save_context():
            if cur_name is not None:
                reg_ctx[cur_name] = (reg_start, reg_end, cur_file)

        def flush():
            nonlocal sz_start, sz_end
            if cur_file and reg_start and sz_start and sz_end:
                rs = _hms_to_sec(reg_start)
                start_s = float(_diff_clock_sec(_hms_to_sec(sz_start), rs))
                end_s = float(_diff_clock_sec(_hms_to_sec(sz_end), rs))
                if end_s <= start_s:
                    end_s = start_s + 1.0
                events.append(SienaSeizureEvent(
                    patient_id=pat,
                    session_id=cur_file,
                    seizure_start_s=start_s,
                    seizure_end_s=end_s,
                ))
            else:
                print(f"[siena] incomplete seizure entry skipped in {seiz_path}: "
                      f"file={cur_file} reg_start={reg_start} start={sz_start} end={sz_end}")
            sz_start = None
            sz_end = None

        for ln in lines:
            s = ln.strip()

            m = _RE_FILE.match(s)
            if m:
                save_context()
                cur_name = _normalize_edf_name(m.group(1))
                reg_start, reg_end, cur_file = reg_ctx.get(cur_name, (None, None, cur_name))
                sz_start = None
                sz_end = None
                continue

            m = _RE_FIELD.match(s)
            if not m:
                continue
            field = " ".join(m.group("field").lower().split())
            value = _parse_time_value(m.group("value"))
            if value is None:
                continue

            if field == "registration start":
                reg_start = value
            elif field == "registration end":
                reg_end = value
                if cur_name and cur_name.endswith("-.edf"):
                    resolved = _resolve_edf_from_registration(patient_dir, reg_start or "00.00.00", reg_end)
                    if resolved is not None:
                        cur_file = resolved
            elif field in ("start", "seizure start"):
                sz_start = value
            elif field in ("end", "seizure end"):
                sz_end = value
                flush()

    return events

def get_seizures_for_session(
    seizure_by_file: Dict[Tuple[str, str], List[Tuple[float, float]]],
    patient_id: str,
    session_key: str,
) -> List[Tuple[float, float]]:
    """
    Resolve seizures even if CSV session_id and EDF stem differ slightly:
      - try exact match
      - if EDF stem ends with "-<int>", also try base (strip suffix)
      - if still missing, try unique prefix match within patient
        (CSV key is prefix of EDF stem or vice versa)
    """
    # 1) exact
    z = seizure_by_file.get((patient_id, session_key))
    if z:
        return z

    # 2) strip trailing "-<digits>" from EDF stem (PN01-1 -> PN01)
    base = re.sub(r"-\d+$", "", session_key)
    if base != session_key:
        z = seizure_by_file.get((patient_id, base))
        if z:
            return z

    # 3) unique prefix match within patient
    cand = []
    for (pat, sess), intervals in seizure_by_file.items():
        if pat != patient_id:
            continue
        if sess.startswith(session_key) or session_key.startswith(sess):
            cand.append((sess, intervals))

    if len(cand) == 1:
        return cand[0][1]

    return []


def canon_patient(x: str) -> str:
    m = _PAT_RE.search(str(x))
    return m.group(1).upper() if m else str(x).strip().upper()


def canon_session(x: str) -> str:
    # treat "PN11-.edf" stem as "PN11" (matches your CSV PN11)
    s = Path(str(x).strip()).stem
    s = s.strip()
    if s.endswith("-"):
        s = s[:-1]
    return s


def _hms_to_sec(hms: str) -> int:
    # accepts "HH.MM.SS" or "HH:MM:SS"
    parts = re.split(r"[.:]", str(hms).strip())
    if len(parts) != 3:
        raise ValueError(f"Bad HMS: {hms!r}")
    h, m, s = map(int, parts)
    return h * 3600 + m * 60 + s


def _diff_clock_sec(t1: int, t0: int) -> int:
    """Difference in seconds on a 24h clock, allowing wrap around midnight."""
    d = t1 - t0
    if d < 0:
        d += 24 * 3600
    return d


def load_seizure_index_from_csv(csv_path) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
    """
    Reads all_seizures.csv with columns:
      patient_id, session_id, reg_start_hms, reg_end_hms, seizure_start_hms, seizure_end_hms

    Returns:
      seizure_by_file[(patient_id, session_key)] = [(start_sec, end_sec), ...]
      where session_key is Path(session_id).stem (so "PN10-4.5.6.edf" -> "PN10-4.5.6")
    """
    df = pd.read_csv(csv_path)

    need = {"patient_id","session_id","reg_start_hms","seizure_start_hms","seizure_end_hms"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    # normalize strings
    df = df.copy()
    for c in ["patient_id","session_id","reg_start_hms","seizure_start_hms","seizure_end_hms"]:
        df[c] = df[c].astype(str).str.strip()

    # compute in-session seconds relative to reg_start
    rs = df["reg_start_hms"].map(_hms_to_sec).to_numpy()
    ss = df["seizure_start_hms"].map(_hms_to_sec).to_numpy()
    se = df["seizure_end_hms"].map(_hms_to_sec).to_numpy()

    start_s = np.array([_diff_clock_sec(int(a), int(b)) for a, b in zip(ss, rs)], dtype=float)
    end_s   = np.array([_diff_clock_sec(int(a), int(b)) for a, b in zip(se, rs)], dtype=float)

    # fix pathological cases (shouldn't happen, but keep it robust)
    bad = end_s <= start_s
    end_s[bad] = start_s[bad] + 1.0

    df["start_s"] = start_s
    df["end_s"] = end_s

    seizure_by_file: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for row in df.itertuples(index=False):
        pat = canon_patient(row.patient_id)
        sess_key = canon_session(row.session_id)
        seizure_by_file.setdefault((pat, sess_key), []).append((float(row.start_s), float(row.end_s)))

    for k in seizure_by_file:
        seizure_by_file[k].sort()

    return seizure_by_file


def build_seizure_index_from_lists(root_dir: Path) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
    """
    Build the seizure index from the Seizures-list-PNxx.txt files of the Siena database.

    Returns the same structure as load_seizure_index_from_csv:
      seizure_by_file[(patient_id, session_key)] = [(start_sec, end_sec), ...]
    """
    seizure_by_file: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for ev in parse_siena_seizure_lists(str(root_dir)):
        key = (canon_patient(ev.patient_id), canon_session(ev.session_id))
        seizure_by_file.setdefault(key, []).append((float(ev.seizure_start_s), float(ev.seizure_end_s)))

    for k in seizure_by_file:
        seizure_by_file[k].sort()

    return seizure_by_file


def _norm_ch_names(ch_names: Iterable[str]) -> List[str]:
    out: List[str] = []
    for ch in ch_names:
        s = " ".join(str(ch).strip().split())  # collapse whitespace
        su = s.upper()

        # ---- ECG / EKG ----
        # KUK examples: "ECG EKG" (but also possible "ECG", "EKG", "ECG1", etc.)
        if su in {"EEG EKG", "EEG EKG1", "EKG", "EKG1", "ECG", "ECG1"} or su.startswith("ECG") or su.startswith(
            "EKG"):
            out.append("ecg 1")
            continue

        # ---- SpO2 ----
        # KUK: "SAO2 SpO2"; Siena: "SPO2"
        if su in {"SAO2 SPO2", "SAO2", "SPO2", "O2SAT", "OSAT"} or "SPO2" in su or su.startswith("SAO2"):
            out.append("spo2")
            continue

        # ---- HR ----
        if su in {"HR", "HEARTRATE", "HEART RATE"}:
            out.append("hr")
            continue

        # ---- Pleth / PLET ----
        if su in {"PLET", "PLETH", "PLETHYSMO", "PPG"}:
            out.append("plet")
            continue

        # ---- EOG ----
        # KUK: "EEG EOGL", "EEG EOGR"; sometimes LOC/ROC
        if su in {"EEG EOGL", "EOGL", "EOG L", "EEG LOC", "LOC"}:
            out.append("eog l")
            continue
        if su in {"EEG EOGR", "EOGR", "EOG R", "EEG ROC", "ROC"}:
            out.append("eog r")
            continue

        # ---- EEG electrodes ----
        # normalize to "eeg <label>" for anything that looks like an electrode label
        # Siena examples: "EEG Fp1", "EEG FP2", "EEG CZ", "EEG Fc1", ...
        s_no_prefix = _EEG_PREFIX_RE.sub("", s).strip()  # remove leading "EEG "
        lab = s_no_prefix.lower()

        # common electrode-like tokens (fp1, f3, cz, p10, ft7, t7, m1, etc.)
        # If it's already like "eeg x", keep it; otherwise, prefix with "eeg "
        if re.fullmatch(r"(fp|af|f|fc|ft|c|cp|p|po|o|t|tp|m)\d{1,2}", lab) or lab in {"fz", "cz", "pz"}:
            out.append(f"eeg {lab}")
            continue

        # keep KUK "M1/M2" (mastoids) aligned with Siena if present
        if lab in {"m1", "m2"}:
            out.append(f"eeg {lab}")
            continue

        # If channel already starts with "eeg " (after stripping weird whitespace), keep it canonicalized
        if lab.startswith("eeg "):
            out.append(lab)
            continue

        # ---- Other / digital / aux ----
        # Keep as a lowercase token; supported_channels filtering will remove if undesired.
        out.append(lab)

    return out


def pick_supported(raw: mne.io.BaseRaw, supported_norm: List[str]) -> Tuple[List[int], List[str], List[str]]:
    """
    Returns:
      picks: indices into raw.ch_names
      picked_raw_names: original raw channel names (for trace/debug)
      picked_norm_names: normalized names in the SAME order as supported_norm (strict ordering)
    """
    raw_names = list(raw.ch_names)
    raw_norm = _norm_ch_names(raw_names)

    idx_by_norm = {}
    for i, nm in enumerate(raw_norm):
        if nm not in idx_by_norm:
            idx_by_norm[nm] = i

    picks = []
    picked_raw = []
    picked_norm = []
    for want in supported_norm:
        i = idx_by_norm.get(want, None)
        if i is not None:
            picks.append(i)
            picked_raw.append(raw_names[i])
            picked_norm.append(want)

    return picks, picked_raw, picked_norm


def edf_to_hdf5_siena(
    edf_path: str,
    out_h5_path: str,
    seizure_by_file: Dict[Tuple[str, str], List[Tuple[float, float]]],  # (patient, session) -> [(s,e)]
    supported_channels: List[str],  # normalized names you want in output order
    target_sfreq: int = 256,
    compression_ratio: int = 4,
) -> Tuple[dict, List[dict]]:

    edf_path = Path(edf_path)
    out_h5_path = Path(out_h5_path)
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)

    patient_id = canon_patient(edf_path.stem)  # <-- key change: derive from filename
    session_key = canon_session(edf_path.stem)
    record_name = session_key

    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    sf_native = float(raw.info["sfreq"])

    # pick channels in a strict supported order
    supported_norm = [c.strip().lower() for c in supported_channels]
    picks, picked_raw_names, picked_norm_names = pick_supported(raw, supported_norm)

    if not picks:
        raw.close()
        raise RuntimeError(f"No supported channels found in {edf_path}")

    # load picked channels only
    x = raw.get_data(picks=picks, reject_by_annotation=None)  # float64, (C, Tnative)
    raw.close()

    # resample once (offline)
    if int(round(sf_native)) != int(target_sfreq):
        # rational polyphase
        up = int(target_sfreq)
        down = int(round(sf_native))
        x = scipy.signal.resample_poly(x, up=up, down=down, axis=1)

    x = x.astype(np.float32, copy=False)
    C, T = x.shape

    #seizures = seizure_by_file.get((patient_id, session_key), [])
    seizures = get_seizures_for_session(seizure_by_file, patient_id, session_key)
    if not seizures and patient_id in {"PN01", "PN11"}:
        print("NO SEIZURES MATCH:", patient_id, session_key,
              "available keys:", sorted([k[1] for k in seizure_by_file if k[0] == patient_id]))

    onsets = np.array([s for s, e in seizures], dtype=float)
    durations = np.array([e - s for s, e in seizures], dtype=float)
    desc = np.array(["seizure"] * len(seizures), dtype=object)

    # write HDF5 compatible with your loader
    with h5py.File(str(out_h5_path), "w") as hdf:
        hdf.create_dataset("signal", data=x, compression="gzip", compression_opts=compression_ratio, shuffle=True)

        dt = h5py.string_dtype("utf-8")
        hdf.create_dataset("annotations.onset", data=onsets, compression="gzip", compression_opts=compression_ratio)
        hdf.create_dataset("annotations.duration", data=durations, compression="gzip", compression_opts=compression_ratio)
        hdf.create_dataset("annotations.description", data=desc.astype(dt), dtype=dt,
                           compression="gzip", compression_opts=compression_ratio)

        # attrs (like CHB)
        hdf.attrs["patient_id"] = patient_id
        hdf.attrs["record_name"] = record_name
        hdf.attrs["sfreq"] = float(target_sfreq)
        hdf.attrs["signal_length"] = int(T)
        hdf.attrs["n_channels"] = int(C)
        # IMPORTANT: write normalized channel names in the OUTPUT order
        dt = h5py.string_dtype("utf-8")
        hdf.attrs["ch_names"] = np.array(picked_norm_names, dtype=object)

        # optional debug
        hdf.create_dataset("raw.ch_names", data=np.array(picked_raw_names, dtype=object), dtype=dt)

    session_row = {
        "patient_id": patient_id,
        "record_name": record_name,
        "filename": str(out_h5_path),
        "sfreq": float(target_sfreq),
        "n_channels": int(C),
        "signal_length": int(T),
        "n_seizures": int(len(seizures)),
        "channels": ",".join(picked_norm_names),
    }

    annot_rows = []
    for (s, e) in seizures:
        annot_rows.append({
            "patient_id": patient_id,
            "record_name": record_name,
            "filename": str(out_h5_path),
            "onset_sec": float(s),
            "duration_sec": float(e - s),
            "description": "seizure",
            "source": "seizures-list",
        })

    return session_row, annot_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("siena_root", type=str)               # folder containing PNxx
    ap.add_argument("--out_hdf_root", type=str, required=True)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--sampling_freq", type=int, default=256)
    ap.add_argument("--compression_ratio", type=int, default=4)
    ap.add_argument("--seizure_csv", type=str, default=None,
                    help="Optional precomputed seizure table (all_seizures.csv). "
                         "By default, the Seizures-list-PNxx.txt files are parsed.")
    args = ap.parse_args()

    siena_root = Path(args.siena_root)
    out_root = Path(args.out_hdf_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # build seizure index from the per-patient seizure lists, or from a precomputed table if given
    if args.seizure_csv is not None:
        seizure_by_file = load_seizure_index_from_csv(Path(args.seizure_csv))
    else:
        seizure_by_file = build_seizure_index_from_lists(siena_root)

    # optional sanity prints
    n_events = sum(len(v) for v in seizure_by_file.values())
    print("Total seizures from CSV:", n_events)
    cnt_pat = Counter([pat for (pat, _sess) in seizure_by_file.keys() for _ in seizure_by_file[(pat, _sess)]])
    # ^ above is awkward; better:
    cnt_pat = Counter()
    for (pat, _), intervals in seizure_by_file.items():
        cnt_pat[pat] += len(intervals)
    print("Per-patient seizure counts (top):", cnt_pat.most_common(10))

    supported_channels = [
    # 10–20 core
    "eeg fp1","eeg fp2","eeg f3","eeg f4","eeg f7","eeg f8","eeg fz",
    "eeg c3","eeg c4","eeg cz",
    "eeg p3","eeg p4","eeg pz",
    "eeg o1","eeg o2",
    "eeg t3","eeg t4","eeg t5","eeg t6",

    # extensions often present in Siena
    "eeg fc1","eeg fc2","eeg fc5","eeg fc6",
    "eeg cp1","eeg cp2","eeg cp5","eeg cp6",
    "eeg f9","eeg f10",
    "eeg p9","eeg p10",

    # non-EEG
    "spo2",
    "hr",
    "plet",
    ]

    edf_files = list(siena_root.rglob("*.edf"))

    def _worker(edf_file: Path):
        out_dir = out_root / edf_file.parent.name
        out_file = out_dir / (edf_file.stem + ".h5")
        return edf_to_hdf5_siena(
            str(edf_file),
            str(out_file),
            seizure_by_file=seizure_by_file,
            supported_channels=supported_channels,
            target_sfreq=args.sampling_freq,
            compression_ratio=args.compression_ratio,
        )

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_worker)(edf) for edf in tqdm(edf_files)
    )

    sessions, annotations = [], []
    for sess_row, annot_rows in results:
        sessions.append(sess_row)
        annotations.extend(annot_rows)

    pd.DataFrame(sessions).to_csv(out_root / "sessions.csv", index=False)
    pd.DataFrame(annotations).to_csv(out_root / "annotations.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
