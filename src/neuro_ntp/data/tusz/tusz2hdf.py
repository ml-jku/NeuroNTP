#!/usr/bin/env python3
"""
+----------------------------------------------------------------------------------------------------------------------+
tusz2hdf.py
Convert the TUH EEG Seizure Corpus (TUSZ) to session-level HDF5 files

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    First preprocessing step for TUSZ. Creates the storage structure expected by the Epilepsia dataset logic in nntp:
    one HDF5 file per recording session, plus CSV index tables. The core datasets and dataset views used for
    pretraining are generated from these files in a subsequent step.

    The source EDF recordings are converted to compressed HDF5, which allows efficient partial reads of the signal
    without decoding the full EDF. The metadata required for the later core/view generation is stored with each file:
    channel names (normalised, cleaned and original), channel types, sampling rate, signal length, event annotations,
    provenance of the source files and the preprocessing parameters.

    Processing per session:
      1. The EDF segments of a session (*_t000.edf, *_t001.edf, ...) are loaded in order of their segment index.
      2. Channel names are normalised (reference suffixes such as -REF/-LE removed, bare electrode labels prefixed
         with "EEG ").
      3. Each segment is resampled to the target rate; the segments are then concatenated along time.
      4. Notch and band-pass filters are applied to the stitched signal.
      5. Annotations from the .csv_bi (term-based) and .csv (channel-based) files are shifted by the cumulative
         segment offsets onto the session time axis.

Input:
    TUSZ v2.0.x directory layout; the edf/ directory is passed as --tusz_edf_root:
        <tusz_edf_root>/{train,dev,eval}/<patient>/<session>/<montage>/<patient>_<session>_tNNN.{edf,csv,csv_bi}

Output:
    <out_hdf_root>/<split>/<patient>/<split>__<patient>__<session>__<montage>__<patient>_<session>.h5
    <out_hdf_root>/sessions.csv       one row per written session (file path, sfreq, channels, duration, event counts)
    <out_hdf_root>/annotations.csv    one row per annotation, with source and kind columns (see below)
    <out_hdf_root>/skipped.csv        sessions not written (shorter than --min_session_sec, or read errors)

    HDF5 file contents:
        signal                                       float32 (n_channels, n_samples), gzip-compressed
        ch_names, ch_names_clean, ch_names_raw       normalised / cleaned / original channel names
        ch_types                                     channel types as inferred by MNE from the EDF header
        events.{onset,duration,type}                 all term-based labels from .csv_bi, including "bckg"
        markers.{onset,duration,type}                identical to events.*
        annotations.{onset,duration,description}     merged seizure intervals (description "seizure")
        source_edfs, source_csv_bi                   source file paths of the stitched segments
        segment_durations_sec                        duration of each stitched segment
        root and info/ attributes                    sfreq, signal_length, n_channels, units, split, patient_id,
                                                     session_id, preprocessing and stitching parameters

    annotations.csv rows by source:
        csv_bi_event     all term-based events (kind: seizure_type or nonseizure_event)
        csv_typed        typed seizure events from the channel-based .csv, one row per annotated channel
        csv_bi_merged    merged seizure intervals (kind: ictal_interval)

Usage:
    python tusz2hdf.py --tusz_edf_root /data/tusz/v2.0.3/edf --out_hdf_root /data/nntp/tusz_hdf --n_jobs 16

    Defaults: resampling to 256 Hz, band-pass 0.5-100 Hz, notch 60 Hz, units V, gzip level 4, minimum session
    length 300 s. Filters can be disabled with --bandpass none / --notch none.

Notes:
    - Storage: approx. 173 GB for TUSZ v2.0.3 with the default settings.
    - Channel consistency across the segments of a session is only checked with --require_same_channels; otherwise
      all segments are assumed to share the channel layout of the first segment.
    - Annotation onsets and durations are in seconds on the stitched session time axis.
    - TUSZ is distributed by the Neural Engineering Data Consortium under the TUH data use agreement and is not
      included in this repository.
+----------------------------------------------------------------------------------------------------------------------+
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import h5py
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

DT_UTF8 = h5py.string_dtype("utf-8")
T_RE = re.compile(r"_t(\d+)$", re.IGNORECASE)

NON_SEIZURE_LABELS = {"bckg"}  # keep as event, but not seizure


def is_seizure_label(lab: str) -> bool:
    lab = (lab or "").strip().lower()
    return lab != "" and lab not in NON_SEIZURE_LABELS


def list_edfs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.edf"))


def split_from_path(edf_path: Path, tusz_root: Path) -> str:
    rel = edf_path.relative_to(tusz_root)
    return rel.parts[0]  # train/dev/eval


def clean_ch_name(s: str) -> str:
    s = " ".join(str(s).split())
    s = re.sub(r"-LE$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"-REF$", "", s, flags=re.IGNORECASE)

    # normalize common EEG electrode-only names to "EEG <NAME>"
    # if it already starts with EEG/ECG/EOG/EMG/SAO2 etc., we keep it.
    u = s.upper()
    if not re.match(r"^(EEG|ECG|EOG|EMG|SAO2|SPO2)\b", u):
        # heuristic: assume bare electrode labels are EEG
        # (FP1, F7, C3, CZ, T3, T5, O1, etc.)
        if re.match(r"^(FP\d|F\d+|C\d|CZ|P\d|O\d|T\d|T[3-6]|A\d|M\d)\b", u):
            s = "EEG " + u
        else:
            # leave unknowns as-is but uppercase for stability
            s = u
    else:
        s = u

    return s


def norm_ch_name(s: str) -> str:
    return clean_ch_name(s).lower()


def read_csv_bi_events_all(csv_bi_path: Path) -> List[Tuple[float, float, str]]:
    """
    Return ALL event-level annotations from csv_bi:
      [(start_sec, stop_sec, label), ...]
    Includes non-seizure events like 'bckg' (kept for provenance).
    """
    if not csv_bi_path.exists():
        return []
    df = pd.read_csv(csv_bi_path, comment="#")
    if df.empty:
        return []

    # standardize
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower()

    events: List[Tuple[float, float, str]] = []
    for r in df.itertuples(index=False):
        try:
            s = float(getattr(r, "start_time"))
            e = float(getattr(r, "stop_time"))
        except Exception:
            continue
        if not (np.isfinite(s) and np.isfinite(e) and e > s):
            continue
        lab = str(getattr(r, "label", "")).strip().lower()
        if lab == "":
            continue
        events.append((s, e, lab))

    events.sort(key=lambda x: x[0])
    return events

def read_csv_plain_events_all(csv_path: Path) -> List[Tuple[float, float, str]]:
    """
    Return ALL event-level annotations from the plain per-channel .csv:
      [(start_sec, stop_sec, label), ...]
    Typically contains typed seizure labels (e.g., cpsz) plus bckg.
    We ignore the 'channel' column (we just want the label vocabulary/timing).
    """
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, comment="#")
    if df.empty:
        return []

    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower()

    out: List[Tuple[float, float, str]] = []
    for r in df.itertuples(index=False):
        try:
            s = float(getattr(r, "start_time"))
            e = float(getattr(r, "stop_time"))
        except Exception:
            continue
        if not (np.isfinite(s) and np.isfinite(e) and e > s):
            continue
        lab = str(getattr(r, "label", "")).strip().lower()
        if lab == "":
            continue
        out.append((s, e, lab))

    out.sort(key=lambda x: x[0])
    return out



def infer_ch_types(raw: mne.io.BaseRaw) -> List[str]:
    types = []
    for i, _ in enumerate(raw.ch_names):
        t = mne.channel_type(raw.info, i)
        types.append(str(t))
    return types


def parse_rel_meta(edf_path: Path, tusz_root: Path) -> Dict[str, object]:
    """
    Expected layout:
      root/{train,dev,eval}/{patient}/{session}/{montage}/{record}.edf
    record like: aaaaagtt_s007_t001
    """
    rel = edf_path.relative_to(tusz_root)
    parts = rel.parts
    split = parts[0] if len(parts) > 0 else ""
    patient_dir = parts[1] if len(parts) > 1 else ""
    session_dir = parts[2] if len(parts) > 2 else ""
    montage_dir = parts[3] if len(parts) > 3 else ""

    stem = edf_path.stem
    m = T_RE.search(stem)
    if m:
        t_index = int(m.group(1))
        group_prefix = stem[: m.start()]
    else:
        t_index = -1
        group_prefix = stem

    patient_id = stem.split("_")[0] if "_" in stem else stem

    # session_id groups all *_tXXX segments into one logical recording
    session_id = "/".join([split, patient_dir, session_dir, montage_dir, group_prefix])

    return dict(
        split=split,
        patient_dir=patient_dir,
        session_dir=session_dir,
        montage_dir=montage_dir,
        patient_id=patient_id,
        record_stem=stem,
        t_index=t_index,
        group_prefix=group_prefix,
        session_id=session_id,
    )


def _safe_fname_from_session_id(session_id: str) -> str:
    # stable filename that preserves info but is FS-safe
    return session_id.replace("/", "__")


def merge_intervals(intervals: List[Tuple[float, float]], *, eps: float = 1e-6) -> List[Tuple[float, float]]:
    """
    Merge overlaps / adjacency in a list of (start,end) intervals.
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce + eps:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged


def stitch_one_session(
    session_id: str,
    df_sess: pd.DataFrame,
    tusz_root: Path,
    out_root: Path,
    *,
    target_sfreq: float,
    bandpass: Optional[Tuple[float, float]],
    notch: Optional[float],
    units: str,  # "V" or "uV"
    compression_ratio: int,
    min_session_sec: float,
    require_same_channels: bool = True,
    filter_after_stitch: bool = True,
) -> Tuple[Optional[Dict], List[Dict], Optional[Dict]]:
    """
    Returns (session_row or None, annot_rows, skipped_row or None)
    """

    # sort by t_index then record stem
    df_sess = df_sess.sort_values(["t_index", "record_stem", "edf_path"])
    edf_paths = [Path(p) for p in df_sess["edf_path"].tolist()]

    split = str(df_sess["split"].iloc[0])
    patient_id = str(df_sess["patient_id"].iloc[0])

    sig_parts: List[np.ndarray] = []
    source_edfs: List[str] = []
    source_csv_bi: List[str] = []
    seg_durs: List[float] = []
    typed_events: List[Tuple[float, float, str]] = []

    # channel reference for consistency checks
    ref_ch_names_norm: Optional[List[str]] = None
    ref_ch_names_clean: Optional[List[str]] = None
    ref_ch_names_raw: Optional[List[str]] = None
    ref_ch_types: Optional[List[str]] = None

    # ALL events from csv_bi (incl bckg)
    all_events: List[Tuple[float, float, str]] = []
    offset_sec = 0.0

    try:
        # --- load segments ---
        for edf in edf_paths:
            raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)

            # record raw names, then rename deterministically
            ch_names_raw = [str(ch) for ch in raw.ch_names]
            raw.rename_channels({ch: clean_ch_name(ch) for ch in raw.ch_names})
            ch_names_clean = [str(ch) for ch in raw.ch_names]
            ch_names_norm = [norm_ch_name(ch) for ch in ch_names_clean]
            ch_types = infer_ch_types(raw)

            # ALWAYS capture reference channels from the first segment
            if ref_ch_names_norm is None:
                ref_ch_names_norm = ch_names_norm
                ref_ch_names_clean = ch_names_clean
                ref_ch_names_raw = ch_names_raw
                ref_ch_types = ch_types
            else:
                # only enforce identity if requested
                if require_same_channels and (ch_names_norm != ref_ch_names_norm):
                    raise RuntimeError(
                        f"Channel mismatch within session. "
                        f"Expected={len(ref_ch_names_norm)} got={len(ch_names_norm)} in {edf}"
                    )

            # filtering: either per segment or after stitch
            if not filter_after_stitch:
                if notch is not None:
                    raw.notch_filter(freqs=[float(notch)], verbose=False)
                if bandpass is not None:
                    l, h = bandpass
                    raw.filter(l_freq=float(l), h_freq=float(h), verbose=False)

            # resample
            if float(raw.info["sfreq"]) != float(target_sfreq):
                raw.resample(float(target_sfreq))

            seg_sec = raw.n_times / raw.info["sfreq"]
            seg_durs.append(float(seg_sec))

            # ALL events from csv_bi (offset by cumulative duration)
            csv_bi_path = edf.with_suffix(".csv_bi")
            source_csv_bi.append(str(csv_bi_path) if csv_bi_path.exists() else "")

            events = read_csv_bi_events_all(csv_bi_path)  # (s,e,label) incl bckg
            for s, e, lab in events:
                all_events.append((offset_sec + s, offset_sec + e, lab))

            # typed events from plain .csv (per-channel), offset by cumulative duration
            csv_plain_path = edf.with_suffix(".csv")
            if csv_plain_path.exists():
                tev = read_csv_plain_events_all(csv_plain_path)
                for s, e, lab in tev:
                    typed_events.append((offset_sec + s, offset_sec + e, lab))
            # typed seizure events from plain .csv (exclude bckg; keep cpsz, etc.)
            typed_seiz_events = [(s, e, lab) for (s, e, lab) in typed_events if is_seizure_label(lab)]

            offset_sec += float(seg_sec)

            signal = raw.get_data().astype(np.float32)  # V
            if units.lower() == "uv":
                signal *= 1e6
            sig_parts.append(signal)

            source_edfs.append(str(edf))

        total_sec = float(offset_sec)
        if total_sec < float(min_session_sec):
            return None, [], dict(
                session_id=session_id,
                split=split,
                patient_id=patient_id,
                reason=f"session_too_short<{min_session_sec}s",
                total_duration_sec=total_sec,
                n_segments=len(edf_paths),
            )

        # concat signal
        signal_cat = np.concatenate(sig_parts, axis=1)  # (C, T)
        sfreq = float(target_sfreq)

        # filter after stitch (recommended)
        if filter_after_stitch:
            ch_names_for_info = (
                ref_ch_names_clean
                if ref_ch_names_clean is not None
                else [f"ch{i}" for i in range(signal_cat.shape[0])]
            )
            info = mne.create_info(
                ch_names=ch_names_for_info,
                sfreq=sfreq,
                ch_types=["eeg"] * len(ch_names_for_info),
            )
            raw_cat = mne.io.RawArray(signal_cat, info, verbose=False)
            if notch is not None:
                raw_cat.notch_filter(freqs=[float(notch)], verbose=False)
            if bandpass is not None:
                l, h = bandpass
                raw_cat.filter(l_freq=float(l), h_freq=float(h), verbose=False)
            signal_cat = raw_cat.get_data().astype(np.float32)

        # --------- build streams ---------
        all_events.sort(key=lambda x: x[0])

        # split into seizure vs non-seizure events
        seiz_events = [(s, e, lab) for (s, e, lab) in all_events if is_seizure_label(lab)]
        nonseiz_events = [(s, e, lab) for (s, e, lab) in all_events if not is_seizure_label(lab)]

        # (1) events.* -> ALL events (incl bckg)
        ev_onsets = np.array([s for s, _, _ in all_events], dtype=np.float32)
        ev_durs = np.array([e - s for s, e, _ in all_events], dtype=np.float32)
        ev_types = np.array([lab for _, _, lab in all_events], dtype=object)

        # (2) markers.* -> we keep everything from events.*
        mk_onsets = ev_onsets
        mk_durs = ev_durs
        mk_types = ev_types

        # (3) annotations.* -> merged seizure-only binary coverage
        merged = merge_intervals([(s, e) for (s, e, _) in seiz_events])
        ann_onsets = np.array([s for (s, _) in merged], dtype=np.float32)
        ann_durs = np.array([e - s for (s, e) in merged], dtype=np.float32)
        ann_desc = np.array(["seizure"] * len(merged), dtype=object)

        n_channels, signal_length = int(signal_cat.shape[0]), int(signal_cat.shape[1])

        # write output
        out_dir = out_root / split / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{_safe_fname_from_session_id(session_id)}.h5"

        ch_names_norm_out = ref_ch_names_norm if ref_ch_names_norm is not None else []
        ch_names_clean_out = ref_ch_names_clean if ref_ch_names_clean is not None else []
        ch_names_raw_out = ref_ch_names_raw if ref_ch_names_raw is not None else []
        ch_types_out = ref_ch_types if ref_ch_types is not None else []

        with h5py.File(out_file, "w") as hdf:
            hdf.create_dataset("signal", data=signal_cat, compression="gzip", compression_opts=int(compression_ratio))

            info = hdf.create_group("info")
            info.create_dataset("ch_names", data=ch_names_norm_out, dtype=DT_UTF8)

            # store the numeric fields in info attrs (match KUK store format)
            info.attrs["sfreq"] = float(sfreq)
            info.attrs["signal_length"] = int(signal_length)
            info.attrs["n_channels"] = int(n_channels)

            # --- for later use in load_partial_hdf5: it requires ch_names in ROOT ATTRS ---
            # use the naming convention expected by the Epilepsia dataset classes / pipeline
            hdf.attrs["ch_names"] = np.array(ch_names_norm_out, dtype=object)
            hdf.attrs["signal_length"] = int(signal_length)
            hdf.attrs["n_channels"] = int(n_channels)
            hdf.attrs["sfreq"] = float(sfreq)

            # annotations.* : seizure-only merged coverage
            hdf.create_dataset("annotations.onset", data=ann_onsets, compression="gzip", compression_opts=int(compression_ratio))
            hdf.create_dataset("annotations.duration", data=ann_durs, compression="gzip", compression_opts=int(compression_ratio))
            hdf.create_dataset(
                "annotations.description",
                data=ann_desc.astype(DT_UTF8),
                dtype=DT_UTF8,
                compression="gzip",
                compression_opts=int(compression_ratio),
            )

            # markers.* : seizure-only typed events
            hdf.create_dataset("markers.onset", data=mk_onsets, compression="gzip", compression_opts=int(compression_ratio))
            hdf.create_dataset("markers.duration", data=mk_durs, compression="gzip", compression_opts=int(compression_ratio))
            hdf.create_dataset(
                "markers.type",
                data=mk_types.astype(DT_UTF8),
                dtype=DT_UTF8,
                compression="gzip",
                compression_opts=int(compression_ratio),
            )

            # events.* : ALL events (incl bckg)
            hdf.create_dataset("events.onset", data=ev_onsets, compression="gzip", compression_opts=int(compression_ratio))
            hdf.create_dataset("events.duration", data=ev_durs, compression="gzip", compression_opts=int(compression_ratio))
            hdf.create_dataset(
                "events.type",
                data=ev_types.astype(DT_UTF8),
                dtype=DT_UTF8,
                compression="gzip",
                compression_opts=int(compression_ratio),
            )

            hdf.create_dataset("ch_names", data=np.array(ch_names_norm_out, dtype=object).astype(DT_UTF8), dtype=DT_UTF8)
            hdf.create_dataset("ch_names_clean", data=np.array(ch_names_clean_out, dtype=object).astype(DT_UTF8), dtype=DT_UTF8)
            hdf.create_dataset("ch_names_raw", data=np.array(ch_names_raw_out, dtype=object).astype(DT_UTF8), dtype=DT_UTF8)
            hdf.create_dataset("ch_types", data=np.array(ch_types_out, dtype=object).astype(DT_UTF8), dtype=DT_UTF8)

            # provenance for stitching
            hdf.create_dataset("source_edfs", data=np.array(source_edfs, dtype=object).astype(DT_UTF8), dtype=DT_UTF8)
            hdf.create_dataset("source_csv_bi", data=np.array(source_csv_bi, dtype=object).astype(DT_UTF8), dtype=DT_UTF8)
            hdf.create_dataset("segment_durations_sec", data=np.array(seg_durs, dtype=np.float32))

            # attrs
            hdf.attrs["patient_id"] = str(patient_id)
            hdf.attrs["session_id"] = str(session_id)
            hdf.attrs["split"] = str(split)
            hdf.attrs["sfreq"] = float(sfreq)
            hdf.attrs["signal_length"] = int(signal_length)
            hdf.attrs["n_channels"] = int(n_channels)
            hdf.attrs["units"] = str(units)

            hdf.attrs["preproc.notch_hz"] = float(notch) if notch is not None else -1.0
            hdf.attrs["preproc.bandpass_l_h"] = "none" if bandpass is None else f"{bandpass[0]}-{bandpass[1]}"
            hdf.attrs["preproc.resample_hz"] = float(target_sfreq)
            hdf.attrs["stitch.n_segments"] = int(len(edf_paths))
            hdf.attrs["stitch.total_duration_sec"] = float(total_sec)
            hdf.attrs["stitch.filter_after_stitch"] = int(1 if filter_after_stitch else 0)
            hdf.attrs["stitch.require_same_channels"] = int(1 if require_same_channels else 0)

            # handy counts
            hdf.attrs["n_events_total"] = int(len(all_events))
            hdf.attrs["n_events_seizure"] = int(len(seiz_events))
            hdf.attrs["n_events_nonseizure"] = int(len(nonseiz_events))
            hdf.attrs["n_intervals_merged_seizure"] = int(len(merged))

        # sessions.csv row (session-level)
        session_row = dict(
            patient_id=patient_id,
            session_id=session_id,
            split=split,
            filename=str(out_file),
            sfreq=sfreq,
            n_channels=n_channels,
            signal_length=signal_length,
            duration_sec=float(signal_length / sfreq),
            n_segments=int(len(edf_paths)),
            channels=",".join(ch_names_norm_out),

            # counts (clear separation)
            n_events_total=int(len(all_events)),
            n_seizure_events=int(len(seiz_events)),
            n_nonseizure_events=int(len(nonseiz_events)),
            n_seizure_intervals=int(len(merged)),
            units=units,
        )

        # annotations.csv rows: to keep both streams explicit
        annot_rows: List[Dict] = []

        # event-level rows: split seizure vs non-seizure
        for (s, e, lab) in all_events:
            kind = "seizure_type" if is_seizure_label(lab) else "nonseizure_event"
            annot_rows.append(dict(
                patient_id=patient_id,
                session_id=session_id,
                split=split,
                filename=str(out_file),
                onset_sec=float(s),
                duration_sec=float(e - s),
                description=str(lab),
                source="csv_bi_event",
                kind=kind,
            ))

        # typed seizure rows from plain .csv (these are the real classes like cpsz)
        for (s, e, lab) in typed_seiz_events:
            annot_rows.append(dict(
                patient_id=patient_id,
                session_id=session_id,
                split=split,
                filename=str(out_file),
                onset_sec=float(s),
                duration_sec=float(e - s),
                description=str(lab),  # e.g., cpsz ...
                source="csv_typed",  # distinguish provenance
                kind="seizure_type",  # important for the markers_df builder
            ))

        # merged binary seizure coverage (seizure-only)
        for (s, e) in merged:
            annot_rows.append(dict(
                patient_id=patient_id,
                session_id=session_id,
                split=split,
                filename=str(out_file),
                onset_sec=float(s),
                duration_sec=float(e - s),
                description="seizure",
                source="csv_bi_merged",
                kind="ictal_interval",
            ))

        return session_row, annot_rows, None

    except Exception as e:
        return None, [], dict(
            session_id=session_id,
            split=split,
            patient_id=patient_id,
            reason=f"exception:{repr(e)}",
            total_duration_sec=float(offset_sec),
            n_segments=len(edf_paths),
        )


def main():
    ap = argparse.ArgumentParser("Convert TUSZ EDF + CSV_BI -> stitched session-level HDF5 (all channels).")
    ap.add_argument("--tusz_edf_root", type=str, required=True)
    ap.add_argument("--out_hdf_root", type=str, required=True)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--sampling_freq", type=float, default=256.0)
    ap.add_argument("--compression_ratio", type=int, default=4)

    ap.add_argument("--bandpass", type=str, default="0.5,100", help="'0.5,100' or 'none'")
    ap.add_argument("--notch", type=str, default="60", help="'60' or 'none'")
    ap.add_argument("--units", type=str, default="V", choices=["V", "uV"])

    ap.add_argument("--min_session_sec", type=float, default=300.0)
    ap.add_argument("--require_same_channels", action="store_true", default=False)
    ap.add_argument("--filter_after_stitch", action="store_true", default=True)

    args = ap.parse_args()

    bandpass = None if args.bandpass.lower() == "none" else tuple(float(x) for x in args.bandpass.split(","))
    notch = None if args.notch.lower() == "none" else float(args.notch)

    tusz_root = Path(args.tusz_edf_root)
    out_root = Path(args.out_hdf_root)
    out_root.mkdir(parents=True, exist_ok=True)

    edfs = list_edfs(tusz_root)
    if not edfs:
        raise RuntimeError(f"No EDF files found under {tusz_root}")

    # Build dataframe and group to sessions
    meta_rows = []
    for p in edfs:
        m = parse_rel_meta(p, tusz_root)
        m["edf_path"] = str(p)
        meta_rows.append(m)
    df = pd.DataFrame(meta_rows)

    sessions = list(df.groupby("session_id").groups.keys())
    print(f"Found {len(sessions)} session groups.")

    def _worker(sess_id: str):
        df_sess = df[df["session_id"] == sess_id]
        return stitch_one_session(
            sess_id,
            df_sess,
            tusz_root=tusz_root,
            out_root=out_root,
            target_sfreq=float(args.sampling_freq),
            bandpass=bandpass,
            notch=notch,
            units=args.units,
            compression_ratio=int(args.compression_ratio),
            min_session_sec=float(args.min_session_sec),
            require_same_channels=bool(args.require_same_channels),
            filter_after_stitch=bool(args.filter_after_stitch),
        )

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_worker)(sess_id) for sess_id in tqdm(sessions, desc="TUSZ stitch->HDF5")
    )

    sess_rows, annot_rows, skipped_rows = [], [], []
    for sess_row, a_rows, sk_row in results:
        if sess_row is not None:
            sess_rows.append(sess_row)
        if a_rows:
            annot_rows.extend(a_rows)
        if sk_row is not None:
            skipped_rows.append(sk_row)

    pd.DataFrame(sess_rows).to_csv(out_root / "sessions.csv", index=False)
    pd.DataFrame(annot_rows).to_csv(out_root / "annotations.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(out_root / "skipped.csv", index=False)

    print("Done:", out_root)
    print("Wrote:", out_root / "sessions.csv")
    print("Wrote:", out_root / "annotations.csv")
    print("Wrote:", out_root / "skipped.csv")


if __name__ == "__main__":
    main()
