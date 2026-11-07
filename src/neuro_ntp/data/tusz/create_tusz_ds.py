#!/usr/bin/env python3
"""
+----------------------------------------------------------------------------------------------------------------------+
create_tusz_ds.py
Create the TUSZ core dataset and the dataset view for self-supervised pretraining

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Second preprocessing step for the TUH EEG Seizure Corpus (TUSZ), following the HDF5 conversion (tusz2hdf.py).
    Builds the core dataset and the dataset view used for pretraining NeuroNTP. TUSZ is used for self-supervised
    pretraining only, not for fine-tuning or evaluation. Core dataset and view contain metadata only (patients,
    sessions, annotations, sample and chunk indices); the signal data remains in the HDF5 store created in the
    first step. Further dataset views may be created by defining alternative view configurations (.yaml).

    Steps:
      1. Core dataset: built from the HDF5 store, using the TUSZCoreDataset class.
      2. Default view (pretraining): self-supervised view built from the core dataset in unsupervised mode with the
         default configuration of TUSZSelfsupervisedDatasetView (10 s chunks), or with an explicit YAML
         configuration (--view_cfg).

Input:
    <hdf_root>/sessions.csv, <hdf_root>/annotations.csv and the HDF5 files written by tusz2hdf.py

Output:
    <out_dir>/TUSZ_coreds.h5          core dataset
    <out_dir>/dv_selfsup_10s.h5       default view (pretraining)

Usage:
    python create_tusz_ds.py --hdf_root /data/nntp/tusz_hdf --out_dir /data/nntp/tusz

    Existing core dataset and default view files are reused unless --overwrite is given.

Notes:
    - Core dataset and view store the HDF5 file paths of the conversion; they have to be created on the system
      where the HDF5 store resides.
    - Samples for pretraining are drawn randomly from the view during training, so the view defines the sample
      pool rather than a fixed sequence of training samples.
+----------------------------------------------------------------------------------------------------------------------+
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from neuro_ntp.data.tusz.tusz_dataset import TUSZCoreDataset
from neuro_ntp.data.tusz.tusz_selfsup_ds import TUSZSelfsupervisedDatasetView

CORE_NAME = "TUSZCoreDS"
VIEW_NAME = "TUSZUnsupervisedView"
VIEW_MODE = "unsupervised"

CORE_FILE = "TUSZ_coreds.h5"
DEFAULT_VIEW_FILE = "dv_selfsup_10s.h5"


def build_core_dataset(hdf_root: Path, core_path: Path, overwrite: bool) -> TUSZCoreDataset:
    """
    Build the core dataset from the HDF5 store and save it to core_path.
    An existing core dataset file is reused unless overwrite is set. The returned object is loaded from file.
    """
    if core_path.exists() and not overwrite:
        print(f"...using existing core dataset {core_path}")
    else:
        core_ds = TUSZCoreDataset(name=CORE_NAME, hdf_store_path=str(hdf_root))
        core_ds.save(str(core_path))
    return TUSZCoreDataset(ds_filepath=str(core_path))


def build_default_view(core_ds: TUSZCoreDataset, view_path: Path, view_cfg: Optional[Path], overwrite: bool) -> None:
    """
    Build the self-supervised default view from the core dataset and save it to view_path. view_cfg optionally
    specifies a YAML configuration (None: default configuration of the view class). An existing view file is reused
    unless overwrite is set.
    """
    if view_path.exists() and not overwrite:
        print(f"...using existing default view {view_path}")
        return
    kwargs = {} if view_cfg is None else {"cfg_filepath": str(view_cfg)}
    dv = TUSZSelfsupervisedDatasetView(name=VIEW_NAME, core_ds=core_ds, use_cache=False, mode=VIEW_MODE, **kwargs)
    print_split_summary(dv, view_path.name)
    dv.save(str(view_path))


def print_split_summary(dv: TUSZSelfsupervisedDatasetView, label: str) -> None:
    """Print the number of samples per split, or the total number of samples if the view has no split column."""
    if "split" not in dv.samples_df.columns:
        print(f"{label}: samples={len(dv.samples_df)}")
        return
    counts = dv.samples_df["split"].value_counts()
    print(f"{label}: " + ", ".join(f"{k}={int(v)}" for k, v in counts.items()))


def main():
    ap = argparse.ArgumentParser(description="Create the TUSZ core dataset and pretraining dataset view.")
    ap.add_argument("--hdf_root", type=str, required=True, help="HDF5 root written by tusz2hdf.py")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for core dataset and view")
    ap.add_argument("--view_cfg", type=str, default=None, help="Optional YAML configuration for the default view")
    ap.add_argument("--overwrite", action="store_true", help="Rebuild core dataset and default view if they exist")
    args = ap.parse_args()

    hdf_root = Path(args.hdf_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    view_cfg = Path(args.view_cfg) if args.view_cfg else None

    core_ds = build_core_dataset(hdf_root, out_dir / CORE_FILE, args.overwrite)
    build_default_view(core_ds, out_dir / DEFAULT_VIEW_FILE, view_cfg, args.overwrite)

    print("Done:", out_dir)


if __name__ == "__main__":
    main()