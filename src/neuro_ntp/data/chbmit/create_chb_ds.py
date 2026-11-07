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

    Existing core dataset and default view files are reused unless --overwrite is given; the proper view is always
    rebuilt.

Notes:
    - Core dataset and views store the HDF5 file paths of the conversion; they have to be created on the system
      where the HDF5 store resides.
      produced with chb2hdf.py at the default settings.
    - The reference protocol and its discrepancies are discussed in the NeuroNTP paper.
+----------------------------------------------------------------------------------------------------------------------+
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

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


def build_core_dataset(hdf_root: Path, core_path: Path, overwrite: bool) -> CHBMITCoreDataset:
    """
    Build the core dataset from the HDF5 store and save it to core_path.
    An existing core dataset file is reused unless overwrite is set. The returned object is loaded from file.
    """
    if core_path.exists() and not overwrite:
        print(f"...using existing core dataset {core_path}")
    else:
        core_ds = CHBMITCoreDataset(name=CORE_NAME, hdf_store_path=str(hdf_root))
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


def replace_root(dv: CHBSelfsupervisedCompatDatasetView, old_root: str, new_root: str) -> None:
    """
    Replace the HDF5 root in all file paths of a dataset view and its core dataset (in place).
    """
    def _repl_df(df: Optional[pd.DataFrame]) -> None:
        if df is not None and "filename" in df.columns:
            df["filename"] = df["filename"].astype(str).str.replace(old_root, new_root, regex=False)

    _repl_df(dv.samples_df)
    _repl_df(getattr(dv, "chunks_df", None))

    core = getattr(dv, "core_ds", None)
    if core is not None:
        _repl_df(getattr(core, "session_df", None))
        _repl_df(getattr(core, "markers_df", None))
        if isinstance(getattr(core, "hdf_store_path", None), str):
            core.hdf_store_path = core.hdf_store_path.replace(old_root, new_root)


def print_split_summary(dv: CHBSelfsupervisedCompatDatasetView, label: str) -> None:
    """Print the number of samples per split."""
    counts = dv.samples_df["split"].value_counts()
    print(f"{label}: " + ", ".join(f"{k}={int(counts.get(k, 0))}" for k in ["train", "val", "test"]))


def main():
    ap = argparse.ArgumentParser(description="Create the CHB-MIT core dataset and dataset views.")
    ap.add_argument("--hdf_root", type=str, required=True, help="HDF5 root written by chb2hdf.py")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for core dataset and views")
    ap.add_argument("--proper", action="store_true", help="Also create the view with correct ictal labelling")
    ap.add_argument("--overwrite", action="store_true", help="Rebuild core dataset and default view if they exist")
    args = ap.parse_args()

    hdf_root = Path(args.hdf_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    core_ds = build_core_dataset(hdf_root, out_dir / CORE_FILE, args.overwrite)

    default_view_path = out_dir / DEFAULT_VIEW_FILE
    build_view(core_ds, default_view_path, cfg=None, overwrite=args.overwrite)

    if args.proper:
        build_view(core_ds, out_dir / PROPER_VIEW_FILE, cfg=PROPER_CFG, overwrite=True)

    print("Done:", out_dir)


if __name__ == "__main__":
    main()