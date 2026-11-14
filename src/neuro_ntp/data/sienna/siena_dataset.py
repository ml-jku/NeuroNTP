"""
+----------------------------------------------------------------------------------------------------------------------+
siena_dataset.py
Core dataset of the Siena Scalp EEG Database

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Core dataset of the Siena Scalp EEG Database (SienaCoreDataset), built from the index tables of the HDF5
    conversion (siena2hdf.py: sessions.csv, annotations.csv). The saved core dataset contains metadata only; the
    signal data remains in the HDF5 store.

    Also contains the per-sample normalisation used by the dataset view (per-channel mean subtraction and scaling by
    the 95th percentile of |x|). The seizure lists are parsed during the conversion (siena2hdf.py).

Core dataset tables:
    patients_df    patient_id
    session_df     patient_id, session_id (record name), filename, sfreq, n_channels, signal_length, duration_s,
                   n_seizures, channels
    markers_df     seizure annotations: patient_id, session_id, filename, onset_sec, duration_sec, description
+----------------------------------------------------------------------------------------------------------------------+
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from neuro_ntp.data.epi_dataset import EpilepsiaCoreDataset


def _normalize_torch_quantile95(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: (C, L) float
    x = x - x.mean(dim=-1, keepdim=True)
    q = torch.quantile(torch.abs(x), q=0.95, dim=-1, keepdim=True)
    return x / (q + eps)


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