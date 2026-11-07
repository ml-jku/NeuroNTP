#!/usr/bin/env python3
"""
+----------------------------------------------------------------------------------------------------------------------+
chb2hdf.py
Convert the CHB-MIT Scalp EEG Database to per-recording HDF5 files

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    First preprocessing step for CHB-MIT. Creates the storage structure expected by the Epilepsia dataset logic in
    nntp: one HDF5 file per EDF recording, plus CSV index tables. The core datasets and dataset views used for
    pretraining are generated from these files in a subsequent step.

    The source EDF recordings are converted to compressed HDF5, which allows efficient partial reads of the signal
    without decoding the full EDF. The metadata required for the later core/view generation is stored with each file:
    channel names, sampling rate, signal length, patient and record identifiers, and the seizure annotations.

    Seizure onsets and offsets are taken from the per-patient chbXX-summary.txt files, which are parsed once
    before conversion. Signals are stored as read by MNE (in V), without filtering; resampling is optional.

Input:
    CHB-MIT directory layout as distributed on PhysioNet; the database root is passed as the positional argument:
        <chb_root>/chbXX/chbXX_YY.edf
        <chb_root>/chbXX/chbXX-summary.txt

Output:
    <out_hdf_root>/chbXX/chbXX_YY.h5
    <out_hdf_root>/sessions.csv       one row per recording (file path, sfreq, channels, signal length, seizure count)
    <out_hdf_root>/annotations.csv    one row per seizure (onset and duration in seconds, source "summary")

    HDF5 file contents:
        signal                                       float64 (n_channels, n_samples), gzip-compressed
        annotations.{onset,duration,description}     seizure intervals from the summary file (description "seizure")
        summary.channels                             channel listing(s) from the patient summary file
        root and info/ attributes                    sfreq, signal_length, n_channels, ch_names
        root attributes                              patient_id, record_name

Usage:
    python chb2hdf.py /data/chbmit/1.0.0 --out_hdf_root /data/nntp/chb_hdf --n_jobs 8

    Defaults: native sampling rate (256 Hz), gzip level 4. Use --sampling_freq to resample.

Notes:
    - Storage: approx. 160 GB for CHB-MIT v1.0.0 with the default settings.
    - Channel names are taken from the EDF header. MNE appends "-0", "-1", ... to duplicate channel names; the "-0"
      suffix is removed, further duplicates keep their suffix.
    - Patient identifiers follow the directory names. chb01 and chb21 were recorded from the same subject
      1.5 years apart, which should be taken into account when defining patient-level splits.
    - Annotation onsets and durations are in seconds relative to the start of each recording.
    - CHB-MIT is available from PhysioNet under the Open Data Commons Attribution License v1.0 and is not included
      in this repository.
+----------------------------------------------------------------------------------------------------------------------+
"""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import re


def parse_single_chb_summary_txt(path: Path) -> Dict:
    """
    Parse a CHB-MIT per-patient summary file (chbXX-summary.txt).

    Extracts the channel listing and, for each EDF file listed in the summary, the number of seizures and the
    seizure intervals. Seizure start and end times are paired in order of appearance.

    Args:
        path: Path to a chbXX-summary.txt file.

    Returns:
        Dictionary with the keys:
            patient_id: Patient identifier taken from the file name (e.g. "chb01").
            channels:   Channel names from all "Channel N:" lines in the file, in order of appearance, without
                        placeholder entries ("-"). For patients with montage changes, the listings of all channel
                        blocks are concatenated.
            records:    Mapping from EDF file name (e.g. "chb01_03.edf") to a dictionary with
                        "n_seizures" (int) and "seizures" (list of (start_sec, end_sec) tuples, in seconds relative
                        to the start of the recording).
    """
    patient_id = path.stem.split("-")[0]  # chb01
    channels: List[str] = []
    records: Dict[str, Dict] = {}

    current_file = None
    current_n_seizures = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # channels block
            if line.startswith("Channel "):
                # e.g. "Channel 1: FP1-F7"
                parts = line.split(":", 1)
                if len(parts) == 2:
                    ch_name = parts[1].strip()
                    if ch_name != "-":
                        channels.append(ch_name)
                continue

            # file Name
            if line.lower().startswith("file name:"):
                # finalize previous file if needed
                if current_file is not None and current_file not in records:
                    records[current_file] = {
                        "n_seizures": current_n_seizures,
                        "seizures": []
                    }

                current_file = line.split(":", 1)[1].strip()
                current_n_seizures = 0
                continue

            # number of seizures in file
            if line.lower().startswith("number of seizures in file:"):
                current_n_seizures = int(line.split(":", 1)[1].strip())
                if current_file not in records:
                    records[current_file] = {
                        "n_seizures": current_n_seizures,
                        "seizures": []
                    }
                else:
                    records[current_file]["n_seizures"] = current_n_seizures
                continue

            if "Seizure" in line and "Start Time" in line:
                # extract number before "seconds"
                secs = float(line.split(":")[1].replace("seconds", "").strip())
                # store temp, we pair with End Time right after
                records[current_file].setdefault("_tmp_starts", []).append(secs)
                continue

            if "Seizure" in line and "End Time" in line:
                secs = float(line.split(":")[1].replace("seconds", "").strip())
                starts = records[current_file].get("_tmp_starts", [])
                if starts:
                    start = starts.pop(0)
                    records[current_file].setdefault("seizures", []).append((start, secs))
                continue

    # clean tmp
    for rec in records.values():
        if "_tmp_starts" in rec:
            rec.pop("_tmp_starts")

    return {
        "patient_id": patient_id,
        "channels": channels,
        "records": records,
    }


def load_all_summaries(chb_root: Path) -> Dict[str, Dict]:
    """
    Recursively find all chbXX-summary.txt files beneath chb_root.
    Return dict keyed by patient_id, e.g. {"chb01": {...}, ...}
    """
    summaries = {}
    for p in chb_root.rglob("chb*-summary.txt"):
        parsed = parse_single_chb_summary_txt(p)
        summaries[parsed["patient_id"]] = parsed
    return summaries


def edf_to_hdf5_with_summary(
    edf_file_path: str,
    hdf5_file_path: str,
    summary_dict: dict,
    new_sampling_freq: int = None,
    compression_ratio: int = 4,
    verbose: bool = False,
) -> Tuple[dict, List[dict]]:
    """
    Convert a single CHB-MIT EDF recording to HDF5 and attach its seizure annotations from the parsed summaries.

    The signal is read with MNE and optionally resampled; no filtering is applied. The patient identifier is
    taken from the name of the parent directory, and the seizure intervals are looked up in the summary of that
    patient by EDF file name. Recordings without an entry in the summary, or with zero seizures, are written with
    empty annotation datasets.

    Args:
        edf_file_path:     Path to the source EDF file.
        hdf5_file_path:    Path of the HDF5 file to write; parent directories are created if needed.
        summary_dict:      Parsed summaries keyed by patient identifier, as returned by load_all_summaries.
        new_sampling_freq: Target sampling rate in Hz; None keeps the native rate.
        compression_ratio: gzip compression level (0-9) for the signal and annotation datasets.
        verbose:           Print the output path after writing.

    Returns:
        Tuple (session_row, annot_rows):
            session_row: Recording-level metadata for sessions.csv (patient_id, record_name, filename, sfreq,
                         n_channels, signal_length, n_seizures, channels).
            annot_rows:  One dictionary per seizure for annotations.csv (onset_sec, duration_sec,
                         description "seizure", source "summary", plus identifiers).
    """
    edf_path = Path(edf_file_path)
    out_path = Path(hdf5_file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # read EDF
    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=verbose)

    # optional resample
    if new_sampling_freq is not None and float(new_sampling_freq) != raw.info["sfreq"]:
        raw.resample(new_sampling_freq)

    signal = raw.get_data()             # (C, T)
    ch_names = list(raw.ch_names)
    sfreq = float(raw.info["sfreq"])
    n_channels = signal.shape[0]
    signal_len = signal.shape[1]

    # infer IDs from path
    patient_id = edf_path.parent.name      # e.g. "chb01"
    recname = edf_path.name                # e.g. "chb01_03.edf"
    recstem = edf_path.stem                # e.g. "chb01_03"

    # summary lookup
    patient_summary = summary_dict.get(patient_id, {})
    rec_meta = patient_summary.get("records", {}).get(recname, None)
    base_channels = patient_summary.get("channels", [])

    if rec_meta and rec_meta.get("n_seizures", 0) > 0:
        seizures = rec_meta["seizures"]               # list[(start, end)]
        onsets = np.array([s for (s, e) in seizures], dtype=float)
        durations = np.array([e - s for (s, e) in seizures], dtype=float)
        descriptions = np.array(["seizure"] * len(onsets), dtype="object")
    else:
        seizures = []
        onsets = np.array([], dtype=float)
        durations = np.array([], dtype=float)
        descriptions = np.array([], dtype="object")

    # write HDF5
    with h5py.File(hdf5_file_path, "w") as hdf:
        # main signal
        hdf.create_dataset("signal", data=signal, compression="gzip", compression_opts=compression_ratio)

        # same flat layout as the other Epilepsia-format stores
        dt = h5py.string_dtype("utf-8")
        hdf.create_dataset("annotations.onset", data=onsets, compression="gzip", compression_opts=compression_ratio)
        hdf.create_dataset("annotations.duration", data=durations, compression="gzip", compression_opts=compression_ratio)
        hdf.create_dataset(
            "annotations.description",
            data=descriptions.astype(dt),
            dtype=dt,
            compression="gzip",
            compression_opts=compression_ratio,
        )

        channels = [re.sub(r"-0$", "", str(ch)) for ch in ch_names]

        # ROOT-LEVEL attrs – this is what load_partial_hdf5 expects
        hdf.attrs["patient_id"] = patient_id
        hdf.attrs["record_name"] = recstem
        hdf.attrs["sfreq"] = sfreq
        hdf.attrs["signal_length"] = signal_len
        hdf.attrs["n_channels"] = n_channels
        # important: put ch_names here too
        hdf.attrs["ch_names"] = channels

        # extra: original CHB "canonical" channels from summary
        if base_channels:
            hdf.create_dataset("summary.channels", data=np.array(base_channels, dtype=dt), dtype=dt)

        # we still keep an /info group for convenience
        info_grp = hdf.create_group("info")
        info_grp.attrs["sfreq"] = sfreq
        info_grp.attrs["signal_length"] = signal_len
        info_grp.attrs["n_channels"] = n_channels
        info_grp.attrs["ch_names"] = channels

    if verbose:
        print(f"[worker] wrote {hdf5_file_path}")

    session_row = {
        "patient_id": patient_id,
        "record_name": recstem,
        "filename": str(hdf5_file_path),
        "sfreq": sfreq,
        "n_channels": n_channels,
        "signal_length": signal_len,
        "n_seizures": len(seizures),
        "channels": ",".join(channels),
    }

    annot_rows: List[dict] = []
    for (s, e) in seizures:
        annot_rows.append({
            "patient_id": patient_id,
            "record_name": recstem,
            "filename": str(hdf5_file_path),
            "onset_sec": float(s),
            "duration_sec": float(e - s),
            "description": "seizure",
            "source": "summary",
        })

    return session_row, annot_rows


def main():
    parser = argparse.ArgumentParser(description="Convert CHB-MIT EDF to HDF5 with parallel workers.")
    parser.add_argument("chb_root", type=str, help="Root of the downloaded CHB-MIT (with chb01, chb02, ...)")
    parser.add_argument("--out_hdf_root", type=str, required=True, help="Where to put the .h5 files")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--sampling_freq", type=int, default=None)
    parser.add_argument("--compression_ratio", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    chb_root = Path(args.chb_root)
    out_root = Path(args.out_hdf_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) load all summaries once
    summaries_by_patient = load_all_summaries(chb_root)

    # 2) collect all EDFs
    # pattern: chb01/*.edf, chb02/*.edf, ...
    edf_files = [p for p in chb_root.rglob("*.edf")]

    # 3) parallel convert
    def _worker(edf_file):
        patient_dir = edf_file.parent.name  # chb01
        out_dir = out_root / patient_dir
        out_file = out_dir / (edf_file.stem + ".h5")
        return edf_to_hdf5_with_summary(
            str(edf_file),
            str(out_file),
            summary_dict=summaries_by_patient,
            new_sampling_freq=args.sampling_freq,
            compression_ratio=args.compression_ratio,
            verbose=args.verbose,
        )

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_worker)(edf_file) for edf_file in tqdm(edf_files)
    )

    # 4) flatten & save CSVs
    sessions = []
    annotations = []
    for sess_row, annot_rows in results:
        sessions.append(sess_row)
        annotations.extend(annot_rows)

    pd.DataFrame(sessions).to_csv(out_root / "sessions.csv", index=False)
    pd.DataFrame(annotations).to_csv(out_root / "annotations.csv", index=False)
    print("Done. Wrote sessions.csv and annotations.csv to", out_root)


if __name__ == "__main__":
    main()
