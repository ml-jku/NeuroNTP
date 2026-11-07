from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Iterable
import numpy as np
import pandas as pd
import torch
import json
from neuro_ntp.misc.io import load_partial_hdf5
from neuro_ntp.data.sienna.siena_dataset import SienaCoreDataset, _normalize_torch_quantile95
from neuro_ntp.data.epi_dataset import EpilepsiaDatasetView, EpiSelfSupervisedSample


class SienaSelfSupervisedDatasetView(EpilepsiaDatasetView):
    """
    Dataset view of the Siena Scalp EEG Database in the self-supervised sample format.

    The view is described by two tables:
        samples_df    one row per sample (patient, session, window), with the assigned split
        chunks_df     one row per sample_id and timestep, with label, segment type, time past seizure and
                      time to seizure

    Samples are returned as EpiSelfSupervisedSample items via get_item_of_type(split, index). The default
    configuration targets fine-tuning and evaluation, with a patient-level split (test: PN14, PN16, PN17;
    val: PN11, PN12, PN13; train: all other patients).
    """

    DEFAULT_CFG = {
        "sampling_freq": 256,          # output sfreq
        "native_freq": 512,            # Siena typical; we handle per-file too
        "seq_len": 30,                 # context length (timesteps/chunks)
        "chunk_size_s": 10.0,          # per timestep duration
        "overlap_s": 0.0,              # overlap between chunks inside the sequence
        "sliding_window": True,        # advance by stride (like the continuous case)
        "split": "patient",            # patient | random
        "split_ratio": [0.6, 0.2, 0.2],
        "max_ts": int(1e9),
        # channel handling:
        "channels": "all",             # or list[str] names to keep (normalized)
    }

    def __init__(self, dv_filepath : str = None, core_ds: SienaCoreDataset = None, cfg: dict = None, use_cache = False):
        self.name = "SienaSelfSupView"
        self.dv_filepath = dv_filepath
        self.use_cache = use_cache
        if self.dv_filepath:
            # load a previously saved dataset view
            self.__load()
            self._init_split_views()
        else:
            self.core_ds = core_ds
            self.cfg = dict(self.DEFAULT_CFG)
            if cfg:
                self.cfg.update(cfg)

            self.samples_df = pd.DataFrame(columns=[
                "sample_id", "patient", "session", "filename", "start_index_256", "end_index_256", "split"
            ])
            self.chunks_df = pd.DataFrame(columns=[
                "sample_id", "timestep", "seg_type", "label", "time_past_seizure", "time_to_seizure", "markers", "notes"
            ])

            # derived / cached
            self.split_views: Dict[str, pd.DataFrame] = {}

            self._build()

        # ---- compatibility with existing KUK/TUH views ----
        if "chunk_size" not in self.cfg:
            # keep semantics as seconds (matches how you use chunk_size_s)
            self.cfg["chunk_size"] = float(self.cfg.get("chunk_size_s", 10.0))

        if "chunk_len" not in self.cfg:
            fs = int(self.cfg.get("sampling_freq", 256))
            self.cfg["chunk_len"] = int(round(float(self.cfg["chunk_size"]) * fs))

        # Sometimes code uses "overlap" not "overlap_s"
        if "overlap" not in self.cfg and "overlap_s" in self.cfg:
            self.cfg["overlap"] = float(self.cfg["overlap_s"])

        # Sometimes code uses "stride" (seconds or samples) – set both if helpful
        if "stride_s" not in self.cfg:
            self.cfg["stride_s"] = float(self.cfg["chunk_size"]) - float(self.cfg.get("overlap_s", 0.0))
        if "stride" not in self.cfg:
            self.cfg["stride"] = int(round(self.cfg["stride_s"] * int(self.cfg.get("sampling_freq", 256))))

    def save(self, dv_filepath: str):
        """ auxiliary method for saving the dataset view as an HDF5 file """
        print(f"...saving dataset view {self.name} to {dv_filepath}")
        self.__set_explicit_types()
        with pd.HDFStore(dv_filepath, 'w') as hfile:
            hfile.put('params', pd.DataFrame())
            hfile.get_storer('params').attrs['name'] = self.name
            hfile.get_storer('params').attrs['core_ds_filepath'] = self.core_ds.ds_filepath
            hfile.put('config', pd.Series(json.dumps(self.cfg)))
            hfile.put('samples', self.samples_df)
            hfile.put('chunks', self.chunks_df)

    def __load(self):
        """ auxiliary method for loading the dataset view from an HDF5 file """
        print(f"...loading dataset view {self.name} from {self.dv_filepath}")
        with pd.HDFStore(self.dv_filepath, 'r') as hfile:
            self.name = hfile.get_storer('params').attrs['name']
            self.core_ds_filepath = hfile.get_storer('params').attrs['core_ds_filepath']
            self.cfg = json.loads(hfile['config'].item())
            self.samples_df = hfile['samples']
            self.chunks_df = hfile['chunks']
        self.core_ds = SienaCoreDataset(ds_filepath=self.core_ds_filepath)

    def __set_explicit_types(self):
        """ auxiliary method for enabling storage in HDF5 """
        # explicit type - for saving in hdf format
        self.samples_df = self.samples_df.copy()
        self.chunks_df = self.chunks_df.copy()

        self.samples_df['sample_id'] = self.samples_df['sample_id'].astype(int)
        self.samples_df['patient'] = self.samples_df['patient'].astype(str)
        self.samples_df['session'] = self.samples_df['session'].astype(str)
        self.samples_df['filename'] = self.samples_df['filename'].astype(str)
        self.samples_df['start_index_256'] = self.samples_df['start_index_256'].astype(int)
        self.samples_df['end_index_256'] = self.samples_df['end_index_256'].astype(int)
        self.samples_df['split'] = self.samples_df['split'].astype(str)

        self.chunks_df['sample_id'] = self.chunks_df['sample_id'].astype(int)
        self.chunks_df['timestep'] = self.chunks_df['timestep'].astype(int)
        self.chunks_df['seg_type'] = self.chunks_df['seg_type'].astype(str)
        self.chunks_df['label'] = self.chunks_df['label'].astype(int)
        self.chunks_df['time_past_seizure'] = self.chunks_df['time_past_seizure'].astype(int)
        self.chunks_df['time_to_seizure'] = self.chunks_df['time_to_seizure'].astype(int)
        self.chunks_df['markers'] = self.chunks_df['markers'].astype(str)
        self.chunks_df['notes'] = self.chunks_df['notes'].astype(str)

    def update_channel_config(self, cfg):
        self.cfg["signal_channels"] = cfg["signal_channels"]
        self.supported_channels = [ch.lower() for ch in cfg["signal_channels"]]

    # ----- building -----

    def _build(self):
        sz_by = {}
        mdf = getattr(self.core_ds, "markers_df", None)
        if mdf is not None and len(mdf) > 0:
            # keep only seizure rows if there are other descriptions later
            mdf = mdf[mdf["description"].str.lower().eq("seizure")].copy()

            for r in mdf.itertuples(index=False):
                key = (str(r.patient_id), str(r.session_id))  # session_id is record_name alias you added
                start = float(r.onset_sec)
                end = start + float(r.duration_sec)
                sz_by.setdefault(key, []).append((start, end))

            for k in sz_by:
                sz_by[k].sort(key=lambda x: x[0])

        # build samples/chunks
        sample_rows = []
        chunk_rows = []

        sid = 1
        seq_len = int(self.cfg["seq_len"])
        fs = int(self.cfg["sampling_freq"])
        chunk_s = float(self.cfg["chunk_size_s"])
        overlap_s = float(self.cfg["overlap_s"])
        stride_s = max(chunk_s - overlap_s, 1e-9)
        sample_span_s = stride_s * seq_len + overlap_s
        advance_s = stride_s if bool(self.cfg.get("sliding_window", True)) else sample_span_s

        # For each EDF session: create windows spanning the whole recording
        for row in self.core_ds.session_df.itertuples(index=False):
            patient = row.patient_id
            session = row.session_id
            filename = row.filename   # full path to .h5

            # determine native sfreq per file; we output at fs (typically 256)
            sf_native = int(row.sfreq)
            duration_s = float(row.duration_s)

            # we will generate window start times in seconds (output timeline)
            t = 0.0
            # NOTE: we build indices in output sampling (fs)
            while (t + sample_span_s) <= duration_s:
                start_256 = int(round(t * fs))
                end_256 = int(round((t + sample_span_s) * fs))

                sample_id = sid
                sid += 1

                sample_rows.append({
                    "sample_id": sample_id,
                    "patient": patient,
                    "session": session,
                    "filename": filename,
                    "start_index_256": start_256,
                    "end_index_256": end_256,
                    "split": "unknown",
                })

                # per-timestep chunk annotations
                # We label a timestep as ictal if its chunk interval overlaps any seizure interval.
                evs = sz_by.get((patient, session), [])
                for i in range(seq_len):
                    cs = t + i * stride_s
                    ce = cs + chunk_s

                    seg_type = "non-ictal"
                    label = 0
                    markers = ""
                    notes = ""

                    # check overlap with seizures
                    for (sz_s, sz_e) in evs:
                        if not (ce <= sz_s or cs >= sz_e):
                            seg_type = "ictal"
                            label = 1
                            markers = "seizure"
                            break

                    tps = self.cfg["max_ts"]
                    tts = self.cfg["max_ts"]
                    if evs:
                        prev_ends = [sz_e for (sz_s, sz_e) in evs if sz_e <= cs]
                        if prev_ends:
                            tps = int(round(cs - max(prev_ends)))

                        next_starts = [sz_s for (sz_s, sz_e) in evs if sz_s >= cs]
                        if next_starts:
                            tts = int(round(min(next_starts) - cs))

                    chunk_rows.append({
                        "sample_id": sample_id,
                        "timestep": i,
                        "seg_type": seg_type,
                        "label": int(label),
                        "time_past_seizure": int(tps),
                        "time_to_seizure": int(tts),
                        "markers": markers,
                        "notes": notes,
                    })

                t += float(advance_s)

        self.samples_df = pd.DataFrame(sample_rows)
        self.chunks_df = pd.DataFrame(chunk_rows)

        self._split_samples()
        self._init_split_views()

    def _split_samples(self):
        ratio = self.cfg["split_ratio"]
        split_mode = self.cfg.get("split", "patient")

        if split_mode == "patient":
            pats = sorted(self.samples_df["patient"].unique().tolist())
            n = len(pats)
            i1 = int(round(n * ratio[0]))
            i2 = int(round(n * (ratio[0] + ratio[1])))
            train_p, val_p, test_p = pats[:i1], pats[i1:i2], pats[i2:]
            self.samples_df.loc[self.samples_df["patient"].isin(train_p), "split"] = "train"
            self.samples_df.loc[self.samples_df["patient"].isin(val_p), "split"] = "val"
            self.samples_df.loc[self.samples_df["patient"].isin(test_p), "split"] = "test"
        else:
            # random split by sample rows (not recommended for clinical eval, but useful for quick dev)
            rng = np.random.default_rng(42)
            idx = rng.permutation(self.samples_df.index.to_numpy())
            n = len(idx)
            i1 = int(round(n * ratio[0]))
            i2 = int(round(n * (ratio[0] + ratio[1])))
            self.samples_df.loc[idx[:i1], "split"] = "train"
            self.samples_df.loc[idx[i1:i2], "split"] = "val"
            self.samples_df.loc[idx[i2:], "split"] = "test"

    def _init_split_views(self):
        self.split_views = {
            s: self.samples_df[self.samples_df["split"] == s].reset_index(drop=True)
            for s in ["train", "val", "test"]
        }

    # ----- accessors -----

    def __len__(self):
        return len(self.samples_df)

    def get_len_type(self, split: str) -> int:
        return len(self.split_views[split])

    @property
    def signal_channels(self):
        return self.cfg['signal_channels']

    @property
    def norm_signal_channels(self):
        out = []
        for ch in self.signal_channels:
            s = " ".join(str(ch).strip().split())
            s_low = s.lower()

            # strip TUH suffixes for encoder matching
            s_low = s_low.replace("-ref", "")

            # common harmonization: "eeg ekg1" -> "ecg ekg1"
            if s_low.startswith("eeg ekg") or s_low.startswith("eeg ecg"):
                s_low = s_low.replace("eeg ", "ecg ", 1)

            out.append(s_low)
        return out

    def get_item_of_type(self, split: Optional[str], index: int):
        """
        Returns: (signals, chunks, marker_embs, notes_embs, metadata)
          - signals: FloatTensor (C, T, L) at 256 Hz
          - chunks:  numpy array of per-step metadata for the pipeline (like KUK)
          - marker_embs/notes_embs: None (you can add later)
          - metadata: dict
        """
        df = self.split_views[split] if split else self.samples_df
        row = df.iloc[index]

        patient = row["patient"]
        session = row["session"]
        filename = row["filename"]
        start_256 = int(row["start_index_256"])
        end_256 = int(row["end_index_256"])
        sample_id = int(row["sample_id"])

        # Determine per-sample chunks metadata rows
        chunks = self.chunks_df[self.chunks_df["sample_id"] == sample_id].copy()
        chunks = chunks.sort_values("timestep", kind="stable")
        chunks_np = chunks[[
            "sample_id", "timestep", "seg_type", "markers", "label",
            "time_past_seizure", "time_to_seizure", "notes"
        ]].to_numpy()

        channel_order = self.cfg.get("signal_channels", None) or None

        hdf = load_partial_hdf5(
            filename,
            start_idx=start_256,
            stop_idx=end_256,
            only_signal=True,
            channel_order=channel_order,  # can be wider than a file; OK
        )

        x = hdf["signal"]  # (C, L) numpy
        ch_names = hdf.get("ch_names", [])  # already normalized by loader to "eeg fp1" style

        x_t = torch.as_tensor(x, dtype=torch.float32)

        # Normalize per channel
        x_t = _normalize_torch_quantile95(x_t)  # (C, L)

        # Chunk into (C, T, L_chunk)
        fs = int(self.cfg["sampling_freq"])
        seq_len = int(self.cfg["seq_len"])
        chunk_s = float(self.cfg["chunk_size_s"])
        overlap_s = float(self.cfg["overlap_s"])
        chunk_L = int(round(chunk_s * fs))
        stride_L = int(round((chunk_s - overlap_s) * fs))
        # unfold: (C, L) -> (C, T, chunk_L)
        sig = x_t.unfold(dimension=1, size=chunk_L, step=stride_L)
        # Ensure we have at least seq_len steps
        if sig.shape[1] < seq_len:
            # pad last step(s) by repeating the last available chunk
            last = sig[:, -1:, :].repeat(1, seq_len - sig.shape[1], 1)
            sig = torch.cat([sig, last], dim=1)
        elif sig.shape[1] > seq_len:
            sig = sig[:, :seq_len, :]

        modalities = ["signals", "Notes", "TTS", "segtype", "TPS"]

        marker_embs = None
        if "markers" in chunks.columns and hasattr(self, "marker_emb_dict"):
            mks = chunks["markers"].fillna("").astype(str).tolist()
            if len(mks) > 0:
                marker_embs = torch.stack(
                    [self.marker_emb_dict.get(m, self.marker_emb_dict.get("", torch.zeros(768))) for m in mks], dim=0
                )
                modalities.append("Markers")

        notes_embs = None
        modalities.append("EoS")

        metadata = {
            "patient": patient,
            "filename": session,  # logical record id (record_name)
            "h5_path": filename,  # actual storage file
            "start_index": start_256,
            "end_index": end_256,
            "signal_channels": list(ch_names),  # already canonical from loader
            "modalities": modalities,
            "sfreq": int(self.cfg["sampling_freq"]),
        }

        return EpiSelfSupervisedSample(signal=sig, chunks=chunks_np,
                                      markers_embs=marker_embs, notes_embs=notes_embs,
                                      metadata=metadata)

    @staticmethod
    def batch_iterable(items, batch_size: int):
        """Yield consecutive batches from a list/array-like."""
        n = len(items)
        for i in range(0, n, batch_size):
            yield items[i:i + batch_size]

    def initialize_marker_embeddings(self, marker_embedder, marker_emb_dim, batch_size,
                                     reduced_vocab=False, filter_explicit=False):
        self.marker_emb_dim = marker_emb_dim
        self.marker_embedder = marker_embedder
        self.marker_emb_dict = dict()

        # calculate embeddings with frozen sentence BERT
        self.marker_embedder.eval()
        with torch.no_grad():
            # create marker occurrences
            markers = ['seizure', '']
            for marker_batch in self.batch_iterable(markers, batch_size):
                embeddings = marker_embedder.get_emb(marker_batch)
                embeddings = embeddings.detach().cpu()
                self.marker_emb_dict.update({m: e for m, e in zip(marker_batch, embeddings)})
