"""
+----------------------------------------------------------------------------------------------------------------------+
epi_dataset.py
Base classes of the Epilepsia dataset framework

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Base classes for the unified management of the supported datasets:
        EpilepsiaCoreDataset     session-level description and segmentation of the recordings
        EpilepsiaDatasetView     view on a core dataset defining the training, validation and test samples
        EpilepsiaDataset         PyTorch Dataset; the train, val and test instances are provided by the view
        EpilepsiaDatasetCache    lazy in-memory (CPU RAM) cache of preprocessed samples

    Core dataset and dataset view are implemented for each dataset (e.g. TUSZ, CHB-MIT, Siena) according to its
    signal storage, metadata and directory structure. Normally only these derived classes are instantiated; the
    training and evaluation framework interacts with the interface defined here, so all datasets are handled in a
    uniform way. Processing common to all datasets is implemented once in the base classes and overridden where
    necessary.
+----------------------------------------------------------------------------------------------------------------------+
"""

import torch
import pandas as pd
import yaml
from multiprocessing import Manager
import collections
from typing import Optional, Any

class EpilepsiaDataset(torch.utils.data.Dataset):

    def __init__(self, ds_type: str, ds_view: 'EpilepsiaDatasetView'):
        # instantiation arguments
        self.ds_type = ds_type                # type of the dataset: train|val|test
        self.ds_view = ds_view                # the dataset view with the samples
        self.use_cache = ds_view.use_cache    # should the dataset use "lazy" cache
        self.init_cache()                     # initialize the cache

    def init_cache(self):
        if self.use_cache:
            #manager = Manager()
            #self.cache = EpilepsiaDatasetCache(manager)
            self.cache = EpilepsiaDatasetLocalCache()
        else:
            self.cache = None

    def __len__(self):
        return self.ds_view.get_len_type(self.ds_type)

    def __getitem__(self, index):
        if self.use_cache and self.cache.is_cached(index):
            sample = self.cache.get(index)
        else:
            sample = self.ds_view.get_item_of_type(self.ds_type, index)
            if self.use_cache:
                self.cache.cache(index, sample)

        return sample

    def bucket_key(self, idx: int):
        # one file → one fixed channel layout
        return self.ds_view.get_bucket_key_type(self.ds_type, idx)


class EpilepsiaDatasetView(torch.utils.data.Dataset):

    CHUNK_COLS = [
        "sample_id", "timestep", "seg_type", "markers", "label",
        "time_past_seizure", "time_to_seizure", "notes",
    ]

    def __init__(self, name: str = 'EpiDsView', core_ds: 'EpilepsiaCoreDataset' = None,
                 dv_filepath: str = None, cfg: dict = None,
                 use_cache: bool = False):
        # instatiation arguments
        self.name = name                      # name of the dataset view
        self.core_ds = core_ds                # core dataset to create the view from
        self.dv_filepath = dv_filepath        # file to restore materialized view from
        self.cfg = cfg                        # configuration dictionary for this view
        self.use_cache = use_cache            # indicates if samples should be cached in-memory (RAM)
        self.samples_df = pd.DataFrame(
            columns=['patient', 'timestamp', 'filename', 'start_index', 'end_index', 'label', 'split'])

    def dump_cfg(self, cfg_filepath):
        with open(cfg_filepath, 'w') as file:
            yaml.dump(self.cfg, file)
            print(self.cfg)

    def load_cfg(self, cfg_filepath):
        with open(cfg_filepath, 'r') as file:
            self.cfg = yaml.load(file, Loader=yaml.FullLoader)

    def normalize(self, x):
        pass

    def preprocess_sample(self, signal):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return None

    def get_len_type(self, ds_type):
        return len(self.samples_df[self.samples_df['split'] == ds_type])

    def get_item_of_type(self, ds_type, index):
        return None

    def get_datasets(self):
        return EpilepsiaDataset('train', self), EpilepsiaDataset('val', self), EpilepsiaDataset('test', self)

    def get_bucket_key_type(self, ds_type, idx):
        return None

class EpilepsiaCoreDataset():

    def __init__(self, name: str = 'EpiCoreDS', ds_filepath: str = None, session_csv: str = None,
                 hdf_store_path: str = None, cfg: dict = None):
        self.name = name
        self.ds_filepath = ds_filepath
        self.session_csv = session_csv
        self.hdf_store_path = hdf_store_path
        self.cfg = cfg

        self.patients_df = None
        self.session_df = None
        self.segments_df = None

    @property
    def n_patients(self):
        return 0 if self.patients_df is None else len(self.patients_df)

    @property
    def n_sessions(self):
        return 0 if self.session_df is None else len(self.session_df)

    def init_from_input(self):
        pass

    def get_view(self, name: str = 'EpiDSView', materialized: bool = False, sampling_rate: int = 256, length_seconds: int = 15, channel_order: list = None):
        return EpilepsiaDatasetView(name, self, sampling_rate, length_seconds, channel_order)


class EpiSelfSupervisedSample(object):
    """
    Single sample of the self-supervised setting: a continuous segment of num_timesteps chunks, each of
    sample_length duration. A sample holds the signal data (EEG, other biosignals, device-related signals), the
    temporally aligned markers and notes, and contextual data such as time past seizure, time to seizure, age and
    gender. Attributes are populated according to the dataset; not all datasets provide notes or every type of
    context information.

    Attributes:
        signal          FloatTensor (num_channels, num_timesteps, input_dim) with the input signal data
        chunks          numpy array (num_timesteps, num_ctx_features) with the per-timestep context data:
                            sample_id (int)           unique sample ID, generated when the view is created
                            timestep (int)            index of the timestep within the sample (e.g. 0..31)
                            seg_type (str)            segment type, if available: "ictal" | "non-ictal" | "unspecified"
                            markers (str)             markers aligned with the timestep, joined with ", "
                            label (int)               timestep label, if available (binary 0|1, or classes 0..K)
                            time_past_seizure (int)   time since the last seizure in seconds, capped at TPS_MAX
                            time_to_seizure (int)     time to the next seizure in seconds, capped at TTS_MAX
                            notes (str)               free-text clinical notes aligned with the timestep, joined with ", "
        markers_embs    tensor (num_timesteps, marker_emb_dim) with the sBERT embeddings of the markers
        notes_embs      tensor (num_timesteps, notes_emb_dim) with the sBERT embeddings of the notes
        metadata        dict with sample metadata (optional, extensible):
                            patient (str)             patient ID (e.g. for subject identification tasks)
                            gender (int)              1 | 2
                            year_of_birth (int)       year of birth of the patient
                            filename (str)            source file of the signals and context (logging, evaluation)
                            timestamp (int)           POSIX timestamp of the timestep, for aligning recordings of the
                                                      same patient on a common time axis
    """
    def __init__(self, signal=None, chunks=None, markers_embs=None, notes_embs=None, metadata=None):
        self.signal = signal
        self.chunks = chunks
        self.markers_embs = markers_embs
        self.notes_embs = notes_embs
        self.metadata = metadata

    def as_tuple(self):
        """Return the sample attributes as a tuple in a fixed order."""
        return (self.signal, self.chunks, self.markers_embs, self.notes_embs, self.metadata)


class EpilepsiaDatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return key in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        return self._dict[key]

    def cache(self, key, sample):
        if not self.use_cache or key in self._dict:
            return
        self._dict[key] = sample


class EpilepsiaDatasetLocalCache(object):
    """
    Per-process / per-worker cache.
    """
    def __init__(self, use_cache: bool = True, max_items: Optional[int] = None):
        self.use_cache = use_cache
        self.max_items = max_items
        self._dict = {}
        self._order = collections.deque()  # simple FIFO for eviction

    def is_cached(self, key: Any) -> bool:
        if not self.use_cache:
            return False
        return key in self._dict

    def reset(self):
        self._dict.clear()
        self._order.clear()

    def get(self, key: Any):
        return self._dict[key]

    def cache(self, key: Any, sample):
        if not self.use_cache:
            return
        if key in self._dict:
            return

        # bounded FIFO eviction (optional)
        if self.max_items is not None and len(self._order) >= self.max_items:
            old_key = self._order.popleft()
            self._dict.pop(old_key, None)

        self._dict[key] = sample
        self._order.append(key)
