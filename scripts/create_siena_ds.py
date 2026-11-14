#!/usr/bin/env python3
"""
+----------------------------------------------------------------------------------------------------------------------+
create_siena_ds.py
Create the Siena core dataset and the dataset views for fine-tuning and evaluation

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Second preprocessing step for the Siena Scalp EEG Database, following the HDF5 conversion (siena2hdf.py).
    Builds the core dataset and the dataset views used by NeuroNTP. Core dataset and views contain metadata only
    (patients, sessions, annotations, sample and chunk indices); the signal data remains in the HDF5 store created
    in the first step. Further dataset views may be created by defining alternative view configurations (.yaml).

    Steps:
      1. Core dataset: built from sessions.csv and annotations.csv in the HDF5 root, using the SienaCoreDataset class.
      2. Default view: self-supervised view built from the core dataset with the default configuration of
         SienaSelfSupervisedDatasetView, or with an explicit YAML configuration (--view_cfg). The default view
         carries the patient-level split defined in SienaSelfSupervisedDatasetView (test: PN14, PN16, PN17;
         val: PN11, PN12, PN13; train: all other patients).
      3. Balanced view (fine-tuning): the default view restricted, per patient, to all samples containing at least
         one ictal chunk and an equal number of randomly drawn non-ictal samples (fewer if not enough are available).
         The patient-level split of the default view is kept, except that PN14 is removed from the test split, so
         that the test patients match the reference protocol. Test patients are not used for fine-tuning.
      4. Test view (reference protocol for downstream testing): the default view with its split replaced by the
         split of the reference protocol:
         - test:  all sessions of the held-out test patients (default: PN16, PN17)
         - val:   sessions of the remaining patients, selected greedily in order of decreasing seizure count until
                  the validation set holds at least --val_ratio of their seizures
         - train: all other sessions

Input:
    <hdf_root>/sessions.csv, <hdf_root>/annotations.csv and the HDF5 files written by siena2hdf.py

Output:
    <out_dir>/siena_core.h5                core dataset
    <out_dir>/siena_selfsup_dv.h5          default view
    <out_dir>/siena_selfsup_dv_bal.h5      balanced view
    <out_dir>/siena_selfsup_16_17_dv.h5    test view (name follows the test patients)

Usage:
    python create_siena_ds.py --hdf_root /data/nntp/siena_hdf --out_dir /data/nntp/siena

    Defaults: test patients PN16 PN17, validation ratio 0.2, seed 32. Existing core dataset and default view files
    are reused unless --overwrite is given; balanced and test views are always rebuilt from the default view.

    To use the released balanced view (the exact fine-tuning selection) instead of drawing a new one:
    python create_siena_ds.py --hdf_root /data/nntp/siena_hdf --out_dir /data/nntp/siena \
        --bal_template views/siena_selfsup_dv_bal.h5

Notes:
    - Core dataset and views store the HDF5 file paths as given in sessions.csv; they have to be created on the
      system where the HDF5 store resides.
    - A newly drawn balanced view depends on the random seed (--seed). The released template contains the
      selection used for fine-tuning, with the HDF5 root replaced by the placeholder "<ROOT>" and the core dataset
      reference by "<CORE>". When localizing it, both are set to the local paths, and the referenced HDF5 files
      are checked against the core dataset (existence, sampling rate, signal length, sample ranges), so the HDF5
      store must have been produced with siena2hdf.py at the default settings.
+----------------------------------------------------------------------------------------------------------------------+
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional, Sequence

import h5py
import pandas as pd

from neuro_ntp.data.sienna.siena_dataset import SienaCoreDataset
from neuro_ntp.data.sienna.siena_selfsup_ds import SienaSelfSupervisedDatasetView

DEFAULT_TEST_PATIENTS = ["PN16", "PN17"]

CORE_FILE = "siena_core.h5"
DEFAULT_VIEW_FILE = "siena_selfsup_dv.h5"
BAL_VIEW_FILE = "siena_selfsup_dv_bal.h5"

ROOT_PLACEHOLDER = "<ROOT>"
CORE_PLACEHOLDER = "<CORE>"


def test_view_file(test_patients: Sequence[str]) -> str:
    """File name of the test view, e.g. "siena_selfsup_16_17_dv.h5" for test patients PN16 and PN17."""
    suffix = "_".join(p.upper().replace("PN", "") for p in test_patients)
    return f"siena_selfsup_{suffix}_dv.h5"


def rebase_core_paths(core_ds: SienaCoreDataset, hdf_root: Path) -> None:
    """
    Point the HDF5 file paths of the core dataset to hdf_root (layout <hdf_root>/<patient>/<record>.h5). The paths
    in sessions.csv and annotations.csv are those of the conversion run, which differ if the HDF5 store was moved or
    renamed afterwards.
    """
    def _rebase(f) -> str:
        p = Path(str(f))
        return str(hdf_root / p.parent.name / p.name)

    for df in (core_ds.session_df, getattr(core_ds, "markers_df", None)):
        if df is not None and "filename" in df.columns:
            df["filename"] = df["filename"].map(_rebase)
    core_ds.hdf_store_path = str(hdf_root)


def build_core_dataset(hdf_root: Path, core_path: Path, overwrite: bool) -> SienaCoreDataset:
    """
    Build the core dataset from the conversion outputs in hdf_root and save it to core_path, with all HDF5 file
    paths pointing to hdf_root. An existing core dataset file is reused unless overwrite is set. The returned object
    is loaded from file.
    """
    if core_path.exists() and not overwrite:
        print(f"...using existing core dataset {core_path}")
    else:
        core_ds = SienaCoreDataset(root_dir=str(hdf_root))
        rebase_core_paths(core_ds, hdf_root)
        core_ds.save(str(core_path))
    return SienaCoreDataset(ds_filepath=str(core_path))


def build_default_view(core_ds: SienaCoreDataset, view_path: Path, view_cfg: Optional[Path], overwrite: bool) -> None:
    """
    Build the self-supervised default view from the core dataset and save it to view_path.
    An existing view file is reused unless overwrite is set.
    """
    if view_path.exists() and not overwrite:
        print(f"...using existing default view {view_path}")
        return
    kwargs = {} if view_cfg is None else {"cfg": str(view_cfg)}
    dv = SienaSelfSupervisedDatasetView(core_ds=core_ds, **kwargs)
    dv.save(str(view_path))


def make_balanced_view(default_view_path: Path, out_path: Path, seed: int) -> None:
    """
    Create the balanced view: per patient, keep all samples with at least one ictal chunk (label == 1) and an equal
    number of randomly drawn samples without ictal chunks (capped by the number available). The split column of the
    default view is kept, except that PN14 is removed from the test split.
    """
    rng = random.Random(seed)
    dv = SienaSelfSupervisedDatasetView(dv_filepath=str(default_view_path))
    samples_df = dv.samples_df
    chunks_df = dv.chunks_df

    ictal_sample_ids: List = []
    nonictal_sample_ids: List = []

    for _, pat_samples_df in samples_df.groupby("patient"):
        pat_sample_ids = pat_samples_df["sample_id"].unique()

        ictal_ids = chunks_df.loc[
            chunks_df["sample_id"].isin(pat_sample_ids) & (chunks_df["label"] == 1),
            "sample_id",
        ].unique().tolist()

        ictal_set = set(ictal_ids)
        nonictal_ids = [sid for sid in pat_sample_ids if sid not in ictal_set]

        k = min(len(nonictal_ids), len(ictal_ids))
        neg_subset = rng.sample(nonictal_ids, k) if k > 0 else []

        ictal_sample_ids.extend(ictal_ids)
        nonictal_sample_ids.extend(neg_subset)

    print(f"Ictal samples: {len(set(ictal_sample_ids))}, non-ictal samples: {len(set(nonictal_sample_ids))}")

    selected = set(ictal_sample_ids) | set(nonictal_sample_ids)
    dv.samples_df = samples_df[samples_df["sample_id"].isin(selected)].copy()
    dv.chunks_df = chunks_df[chunks_df["sample_id"].isin(selected)].copy()

    # remove PN14 from the test split, so that the test patients match the reference protocol (PN16, PN17)
    drop = (dv.samples_df["patient"] == "PN14") & (dv.samples_df["split"] == "test")
    dv.samples_df = dv.samples_df[~drop].copy()
    dv.chunks_df = dv.chunks_df[dv.chunks_df["sample_id"].isin(dv.samples_df["sample_id"])].copy()

    dv.save(str(out_path))


def select_val_sessions(trainval_sessions: pd.DataFrame, val_ratio: float) -> List[str]:
    """
    Greedy session-level validation selection: sessions are visited in order of decreasing seizure count and added
    to the validation set until it holds at least val_ratio of all seizures.
    """
    trainval_sessions = trainval_sessions.sort_values("n_seizures", ascending=False)
    target_val_seizures = trainval_sessions["n_seizures"].sum() * val_ratio

    val_sessions: List[str] = []
    val_seizures = 0
    for _, row in trainval_sessions.iterrows():
        if val_seizures < target_val_seizures:
            val_sessions.append(row["session_id"])
            val_seizures += row["n_seizures"]
    return val_sessions


def make_test_view(default_view_path: Path, out_path: Path, test_patients: Sequence[str], val_ratio: float) -> None:
    """
    Create the test view: assign train/val/test per session according to the reference protocol and propagate the
    split to the samples, replacing the patient-level split of the default view.
    """
    dv = SienaSelfSupervisedDatasetView(dv_filepath=str(default_view_path))
    session_df = dv.core_ds.session_df.copy()

    is_test = session_df["patient_id"].isin(test_patients)
    val_sessions = set(select_val_sessions(session_df[~is_test], val_ratio))

    session_df["split"] = "train"
    session_df.loc[session_df["session_id"].isin(val_sessions), "split"] = "val"
    session_df.loc[is_test, "split"] = "test"

    session_to_split = session_df.set_index("session_id")["split"].to_dict()
    dv.samples_df["split"] = dv.samples_df["session"].map(session_to_split)

    unmapped = dv.samples_df["split"].isna()
    if unmapped.any():
        missing = sorted(dv.samples_df.loc[unmapped, "session"].unique())
        raise RuntimeError(f"Samples of sessions without split assignment: {missing}")

    summary = pd.DataFrame({
        "sessions": session_df["split"].value_counts(),
        "patients": session_df.groupby("split")["patient_id"].nunique(),
        "seizures": session_df.groupby("split")["n_seizures"].sum(),
        "samples": dv.samples_df["split"].value_counts(),
    }).reindex(["train", "val", "test"])
    print("Test view split summary:")
    print(summary.to_string())

    dv.save(str(out_path))


def _copy_view_store(src: Path, dst: Path, old_root: str, new_root: str, core_ds_filepath: str) -> None:
    """
    Copy a dataset view store (pandas HDFStore) to dst, replacing old_root by new_root in all file paths and setting
    the reference to the core dataset. The view is copied at store level, so it is not loaded (loading a view also
    loads the core dataset it references, which may not exist on this system).
    """
    with pd.HDFStore(str(src), "r") as s_in, pd.HDFStore(str(dst), "w") as s_out:
        for key in s_in.keys():
            obj = s_in[key]
            if isinstance(obj, pd.DataFrame) and "filename" in obj.columns:
                obj = obj.copy()
                obj["filename"] = obj["filename"].astype(str).str.replace(old_root, new_root, regex=False)
            s_out.put(key, obj)

        # view parameters (name, core dataset reference); attributes written by put() are kept as they are
        a_in = s_in.get_storer("params").attrs
        a_out = s_out.get_storer("params").attrs
        existing = set(a_out._v_attrnamesuser)
        for name in a_in._v_attrnamesuser:
            if name not in existing:
                setattr(a_out, name, getattr(a_in, name))
        a_out.core_ds_filepath = str(core_ds_filepath)


def export_view_template(view_path: Path, template_path: Path, hdf_root: Path) -> None:
    """
    Create a releasable template of a dataset view: the HDF5 root in all file paths is replaced by ROOT_PLACEHOLDER
    and the core dataset reference by CORE_PLACEHOLDER, so that the template contains no local paths.
    """
    _copy_view_store(view_path, template_path, str(hdf_root).rstrip("/"), ROOT_PLACEHOLDER, CORE_PLACEHOLDER)


def verify_view_files(dv: SienaSelfSupervisedDatasetView) -> None:
    """
    Check that the HDF5 files referenced by a dataset view exist, belong to its core dataset, match the session
    metadata (sampling rate and signal length) and cover the sample index ranges of the view, i.e. that the local
    HDF5 store was produced by the same conversion as the one the view was created from.
    """
    def _norm(p) -> str:
        return str(Path(str(p)).resolve())

    session_df = dv.core_ds.session_df
    lengths = {_norm(f): int(n) for f, n in zip(session_df["filename"], session_df["signal_length"])}
    errors = []

    for row in session_df.itertuples(index=False):
        path = Path(row.filename)
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        with h5py.File(path, "r") as h:
            sfreq = float(h.attrs["sfreq"])
            signal_length = int(h.attrs["signal_length"])
        if sfreq != float(row.sfreq) or signal_length != int(row.signal_length):
            errors.append(f"mismatch: {path} (sfreq {sfreq} vs {row.sfreq}, "
                          f"signal_length {signal_length} vs {row.signal_length})")

    files = dv.samples_df["filename"].map(_norm)
    for fn in sorted(set(files)):
        if fn not in lengths:
            errors.append(f"not in the core dataset: {fn}")

    end_col = next((c for c in ("end_index", "end_index_256") if c in dv.samples_df.columns), None)
    if end_col is not None:
        for fn, end in dv.samples_df.groupby(files)[end_col].max().items():
            if fn in lengths and int(end) > lengths[fn]:
                errors.append(f"sample range exceeds the signal: {fn} ({end_col} {int(end)} > {lengths[fn]})")

    if errors:
        raise RuntimeError("HDF5 store does not match the dataset view:\n  " + "\n  ".join(errors))


def localize_view(template_path: Path, out_path: Path, hdf_root: Path, core_path: Path) -> None:
    """
    Create a dataset view from a released template: replace the root placeholder in all file paths with the local
    HDF5 root, reference the local core dataset (core_path), and verify the referenced HDF5 files.
    """
    _copy_view_store(template_path, out_path, ROOT_PLACEHOLDER, str(hdf_root).rstrip("/"), str(core_path))
    dv = SienaSelfSupervisedDatasetView(dv_filepath=str(out_path))
    verify_view_files(dv)
    print(f"...localized view template {template_path} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Create the Siena core dataset and dataset views.")
    ap.add_argument("--hdf_root", type=str, required=True, help="HDF5 root written by siena2hdf.py")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for core dataset and views")
    ap.add_argument("--view_cfg", type=str, default=None, help="Optional YAML configuration for the default view")
    ap.add_argument("--test_patients", type=str, nargs="+", default=DEFAULT_TEST_PATIENTS)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=32, help="Random seed for the balanced view")
    ap.add_argument("--bal_template", type=str, default=None,
                    help="Released balanced view template; if given, it is localized instead of drawing a new "
                         "balanced selection")
    ap.add_argument("--overwrite", action="store_true", help="Rebuild core dataset and default view if they exist")
    args = ap.parse_args()

    hdf_root = Path(args.hdf_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    view_cfg = Path(args.view_cfg) if args.view_cfg else None

    core_path = out_dir / CORE_FILE
    default_view_path = out_dir / DEFAULT_VIEW_FILE

    core_ds = build_core_dataset(hdf_root, core_path, args.overwrite)
    build_default_view(core_ds, default_view_path, view_cfg, args.overwrite)

    if args.bal_template:
        localize_view(Path(args.bal_template), out_dir / BAL_VIEW_FILE, hdf_root, core_path)
    else:
        make_balanced_view(default_view_path, out_dir / BAL_VIEW_FILE, args.seed)

    make_test_view(default_view_path, out_dir / test_view_file(args.test_patients), args.test_patients,
                   args.val_ratio)

    print("Done:", out_dir)


if __name__ == "__main__":
    main()