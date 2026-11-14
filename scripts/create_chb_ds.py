#!/usr/bin/env python3
"""
+----------------------------------------------------------------------------------------------------------------------+
create_chb_ds.py
Create the CHB-MIT core dataset and the dataset views for fine-tuning and evaluation

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Second preprocessing step for the CHB-MIT Scalp EEG Database, following the HDF5 conversion (chb2hdf.py).
    Builds the core dataset and the dataset views used by NeuroNTP. Core dataset and views contain metadata only
    (patients, sessions, annotations, sample and chunk indices); the signal data remains in the HDF5 store created
    in the first step. Further dataset views may be created with alternative view configurations (see below).

    Steps:
      1. Core dataset: built from the HDF5 store, using the CHBMITCoreDataset class.
      2. Default view (BIOT-compatible): built from the core dataset in sliding-window mode with the default
         configuration of CHBSelfsupervisedCompatDatasetView, which follows the BIOT/CBraMod reference protocol
         (sampling and augmentation logic of the reference preprocessing, 10 s chunks). The view carries the
         predefined patient-level split (train: chb01-chb20; val: chb21, chb22; test: chb23, chb24).
         Alternatively, the released default view (--template) is localized to the HDF5 store; it contains the exact
         sample selection (incl. the random sampling and augmentation of the reference protocol) used for the
         reported results.
      3. Proper view (optional, --proper): as the default view, but with correct ictal labelling (biot_compat=False).

    View configuration:
    CHBSelfsupervisedCompatDatasetView accepts a configuration dictionary (cfg), which updates the default
    configuration declared in the class. The following parameters control the deviations from the reference
    protocol:
        biot_compat                       True (default): windows are labelled as in the reference protocol, where
                                          windows lying entirely within the seizure onset/end boundaries are
                                          labelled 0. False: all windows overlapping a seizure, including those
                                          entirely within its boundaries, are labelled ictal.
        augment_ictal                     True (default): ictal samples are augmented as in the reference protocol.
                                          False: no augmentation.
        apply_biot_process1_file_filter   True (default): training samples are filtered as in the reference
                                          preprocessing. False: no filtering, resulting in a larger training set.

Input:
    <hdf_root>/sessions.csv, <hdf_root>/annotations.csv and the HDF5 files written by chb2hdf.py

Output:
    <out_dir>/chbmit_core_ds.h5                 core dataset
    <out_dir>/chbmit_selfsup_10s_biot.h5        default view (BIOT-compatible)
    <out_dir>/chbmit_selfsup_10s_proper.h5      proper view (only with --proper)

Usage:
    python create_chb_ds.py --hdf_root /data/nntp/chb_hdf --out_dir /data/nntp/chb [--proper]

    To use the released default view (the exact sample selection of the experiments) instead of building a new one:
    python create_chb_ds.py --hdf_root /data/nntp/chb_hdf --out_dir /data/nntp/chb \
        --template res/view_template/chbmit_selfsup_10s_biot_template.h5

    Existing core dataset and default view files are reused unless --overwrite is given; the proper view is always
    rebuilt.

Notes:
    - Core dataset and views store absolute HDF5 file paths below --hdf_root; they have to be created on the
      system where the HDF5 store resides, and rebuilt (--overwrite) if the store is moved.
    - The released template contains the HDF5 root as the placeholder "<ROOT>" and the core dataset reference as
      "<CORE>". When localizing it, both are set to the local paths, and the referenced HDF5 files are checked
      against the core dataset (existence, sampling rate, signal length, sample ranges), so the HDF5 store must have
      been produced with chb2hdf.py at the default settings.
+----------------------------------------------------------------------------------------------------------------------+
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import h5py
import pandas as pd

from neuro_ntp.data.chbmit.chb_dataset import CHBMITCoreDataset
from neuro_ntp.data.chbmit.chb_selfsup_compat_ds import CHBSelfsupervisedCompatDatasetView

CORE_NAME = "CHBMITCoreDS"
VIEW_NAME = "CHBMITCompatDsView"
VIEW_MODE = "sliding"

CORE_FILE = "chbmit_core_ds.h5"
DEFAULT_VIEW_FILE = "chbmit_selfsup_10s_biot.h5"
PROPER_VIEW_FILE = "chbmit_selfsup_10s_proper.h5"

PROPER_CFG = {"biot_compat": False}

ROOT_PLACEHOLDER = "<ROOT>"
CORE_PLACEHOLDER = "<CORE>"


def rebase_core_paths(core_ds: CHBMITCoreDataset, hdf_root: Path) -> None:
    """
    Set all HDF5 file paths of the core dataset to absolute paths below hdf_root (layout <hdf_root>/<patient>/
    <record>.h5). The CHB-MIT view joins the HDF5 root with these paths when loading samples, so relative paths from
    sessions.csv (conversion with a relative --out_hdf_root) would contain the root twice.
    """
    root = hdf_root.resolve()

    def _rebase(f) -> str:
        p = Path(str(f))
        return str(root / p.parent.name / p.name)

    for df in vars(core_ds).values():
        if isinstance(df, pd.DataFrame) and "filename" in df.columns:
            df["filename"] = df["filename"].map(_rebase)
    core_ds.hdf_store_path = str(root)


def build_core_dataset(hdf_root: Path, core_path: Path, overwrite: bool) -> CHBMITCoreDataset:
    """
    Build the core dataset from the HDF5 store and save it to core_path, with all HDF5 file paths as absolute paths
    below hdf_root. An existing core dataset file is reused unless overwrite is set. The returned object is loaded
    from file.
    """
    if core_path.exists() and not overwrite:
        print(f"...using existing core dataset {core_path}")
    else:
        core_ds = CHBMITCoreDataset(name=CORE_NAME, hdf_store_path=str(hdf_root.resolve()))
        rebase_core_paths(core_ds, hdf_root)
        core_ds.save(str(core_path))
    return CHBMITCoreDataset(ds_filepath=str(core_path))


def build_view(core_ds: CHBMITCoreDataset, view_path: Path, cfg: Optional[dict], overwrite: bool) -> None:
    """
    Build a dataset view from the core dataset and save it to view_path. cfg updates the default configuration of
    the view class (None: default configuration). An existing view file is reused unless overwrite is set.
    """
    if view_path.exists() and not overwrite:
        print(f"...using existing view {view_path}")
        return
    dv = CHBSelfsupervisedCompatDatasetView(name=VIEW_NAME, core_ds=core_ds, cfg=cfg, mode=VIEW_MODE)
    print_split_summary(dv, view_path.name)
    dv.save(str(view_path))


def _is_str_col(s: pd.Series) -> bool:
    return s.dtype == object or isinstance(s.dtype, pd.StringDtype)


def _replace_in(obj, old: str, new: str):
    """Replace old by new in all string values of a DataFrame or Series (other objects are returned unchanged)."""
    def _sub(s: pd.Series) -> pd.Series:
        return s.map(lambda v: v.replace(old, new) if isinstance(v, str) else v)

    if isinstance(obj, pd.DataFrame):
        obj = obj.copy()
        for col in obj.columns:
            if _is_str_col(obj[col]):
                obj[col] = _sub(obj[col])
        return obj
    if isinstance(obj, pd.Series) and _is_str_col(obj):
        return _sub(obj)
    return obj

def _copy_view_store(src: Path, dst: Path, old_root: str, new_root: str, core_ds_filepath: str) -> None:
    """
    Copy a dataset view store (pandas HDFStore) to dst, replacing old_root by new_root in all string values and
    setting the reference to the core dataset. The view is copied at store level, so it is not loaded (loading a view
    also loads the core dataset it references, which may not exist on this system).
    """
    with pd.HDFStore(str(src), "r") as s_in, pd.HDFStore(str(dst), "w") as s_out:
        for key in s_in.keys():
            s_out.put(key, _replace_in(s_in[key], old_root, new_root))

        # view parameters (name, core dataset reference); attributes written by put() are kept as they are
        a_in = s_in.get_storer("params").attrs
        a_out = s_out.get_storer("params").attrs
        existing = set(a_out._v_attrnamesuser)
        for name in a_in._v_attrnamesuser:
            if name not in existing:
                value = getattr(a_in, name)
                setattr(a_out, name, value.replace(old_root, new_root) if isinstance(value, str) else value)
        a_out.core_ds_filepath = str(core_ds_filepath)


def export_view_template(view_path: Path, template_path: Path, hdf_root: str) -> None:
    """
    Create a releasable template of a dataset view: the HDF5 root (as stored in the view) is replaced by
    ROOT_PLACEHOLDER and the core dataset reference by CORE_PLACEHOLDER, so that the template contains no local paths.
    """
    _copy_view_store(view_path, template_path, str(hdf_root).rstrip("/"), ROOT_PLACEHOLDER, CORE_PLACEHOLDER)


def verify_view_files(dv: CHBSelfsupervisedCompatDatasetView) -> None:
    """
    Check that the HDF5 files referenced by a dataset view exist and match the core dataset (sampling rate, signal
    length, sample index ranges), i.e. that the local HDF5 store was produced by the same conversion as the one the
    view was created from.
    """
    def _norm(p) -> str:
        return str(Path(str(p)).resolve())

    errors = []
    session_df = dv.core_ds.session_df
    lengths = {}
    if {"filename", "sfreq", "signal_length"} <= set(session_df.columns):
        for row in session_df.itertuples(index=False):
            path = Path(row.filename)
            lengths[_norm(path)] = int(row.signal_length)
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
        if not Path(fn).exists():
            errors.append(f"missing: {fn}")
        elif lengths and fn not in lengths:
            errors.append(f"not in the core dataset: {fn}")

    if "end_index" in dv.samples_df.columns and lengths:
        for fn, end in dv.samples_df.groupby(files)["end_index"].max().items():
            if fn in lengths and int(end) > lengths[fn]:
                errors.append(f"sample range exceeds the signal: {fn} (end_index {int(end)} > {lengths[fn]})")

    if errors:
        raise RuntimeError("HDF5 store does not match the dataset view:\n  " + "\n  ".join(sorted(set(errors))))


def localize_view(template_path: Path, out_path: Path, hdf_root: Path, core_path: Path) -> None:
    """
    Create a dataset view from a released template: replace the root placeholder with the absolute local HDF5 root,
    reference the local core dataset (core_path), and verify the referenced HDF5 files.
    """
    _copy_view_store(template_path, out_path, ROOT_PLACEHOLDER, str(hdf_root.resolve()), str(core_path.resolve()))
    dv = CHBSelfsupervisedCompatDatasetView(dv_filepath=str(out_path))
    verify_view_files(dv)
    print_split_summary(dv, out_path.name)
    print(f"...localized view template {template_path} -> {out_path}")


def print_split_summary(dv: CHBSelfsupervisedCompatDatasetView, label: str) -> None:
    """Print the number of samples per split."""
    counts = dv.samples_df["split"].value_counts()
    print(f"{label}: " + ", ".join(f"{k}={int(counts.get(k, 0))}" for k in ["train", "val", "test"]))


def main():
    ap = argparse.ArgumentParser(description="Create the CHB-MIT core dataset and dataset views.")
    ap.add_argument("--hdf_root", type=str, required=True, help="HDF5 root written by chb2hdf.py")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for core dataset and views")
    ap.add_argument("--template", type=str, default=None,
                    help="Released default view template; if given, it is localized instead of building a new "
                         "default view")
    ap.add_argument("--proper", action="store_true", help="Also create the view with correct ictal labelling")
    ap.add_argument("--overwrite", action="store_true", help="Rebuild core dataset and default view if they exist")
    args = ap.parse_args()

    hdf_root = Path(args.hdf_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    core_path = out_dir / CORE_FILE
    core_ds = build_core_dataset(hdf_root, core_path, args.overwrite)

    default_view_path = out_dir / DEFAULT_VIEW_FILE
    if args.template:
        localize_view(Path(args.template), default_view_path, hdf_root, core_path)
    else:
        build_view(core_ds, default_view_path, cfg=None, overwrite=args.overwrite)

    if args.proper:
        build_view(core_ds, out_dir / PROPER_VIEW_FILE, cfg=PROPER_CFG, overwrite=True)

    print("Done:", out_dir)


if __name__ == "__main__":
    main()