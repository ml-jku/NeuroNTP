"""
+----------------------------------------------------------------------------------------------------------------------+
chb_selfsup_compat_ds.py
BIOT-compatible self-supervised dataset view of the CHB-MIT Scalp EEG Database

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Dataset view of CHB-MIT on top of the EpilepsiaCoreDataset and EpilepsiaDatasetView base classes, in the
    self-supervised sample format. The default configuration follows the BIOT/CBraMod reference protocol: the
    sampling and augmentation logic of the reference preprocessing, 10 s chunks, and the predefined patient-level
    split (train: chb01-chb20; val: chb21, chb22; test: chb23, chb24).

    The deviations from the reference protocol are controlled by the configuration parameters biot_compat
    (labelling of windows within seizure boundaries), augment_ictal (augmentation of ictal samples) and
    apply_biot_process1_file_filter (filtering of training samples). The reference protocol and its discrepancies
    are discussed in the NeuroNTP paper.
+----------------------------------------------------------------------------------------------------------------------+
"""
import json
import os
import random
from datetime import timedelta
from contextlib import contextmanager
import joblib
import hashlib
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from neuro_ntp.data.chbmit.chb_dataset import CHBMITCoreDataset
from neuro_ntp.misc.io import load_partial_hdf5
from neuro_ntp.data.epi_dataset import EpilepsiaDatasetView, EpiSelfSupervisedSample
import re
from collections import Counter


CANON_PAIRS = [
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]

def _pairs_from_canon(canon_list):
    pairs = []
    for ch in canon_list:
        if "-" not in ch:
            raise ValueError(f"Expected bipolar 'A-B' in canon list, got: {ch}")
        a,b = [t.strip() for t in ch.split("-", 1)]
        pairs.append((a,b))
    return pairs

def _clean_and_tokenize(ch_string: str):
    toks = []
    for raw in str(ch_string).split(","):
        #s = _canonize(raw)
        s = raw
        if s == "-" or re.fullmatch(r"--\d+", s):
            continue
        toks.append(s)
    return toks

def _detect_scheme(avail):
    # if many 'A-B' where B not like common refs → bipolar
    bip = sum(1 for c in avail if "-" in c and not re.search(r"-(CS\d+|A1|A2|CZ|REF)$", c))
    # most-common suffix → likely reference
    suff = []
    for c in avail:
        m = re.search(r"-([A-Z0-9]+)$", c)
        if m: suff.append(m.group(1))
    ref, count = (None, 0)
    if suff:
        ref, count = Counter(suff).most_common(1)[0]
    if bip >= max(4, 0.2*len(avail)):
        return ("bipolar", None)
    if count >= max(6, 0.4*len(avail)):
        return ("ref", ref)
    return (None, None)

def plan_loading(ch_string: str, canon_pairs):
    avail = _clean_and_tokenize(ch_string)
    scheme, ref = _detect_scheme(avail)

    if scheme == "bipolar":
        # we may need to reorder to match canon_pairs
        have = {c:i for i,c in enumerate(avail)}  # current order in file header list
        desired = [f"{a}-{b}" for (a,b) in canon_pairs]
        if not all(d in have for d in desired):
            return None
        # channel_order to load in *desired* order → avoids later reindexing
        return dict(mode="bipolar", channels_to_load=desired, pairs=None, ref=None)

    if scheme == "ref" and ref:
        need = []
        pairs = []
        for a,b in canon_pairs:
            a_ref = f"{a}-{ref}"
            b_ref = f"{b}-{ref}"
            if a_ref not in avail or b_ref not in avail:
                return None
            need.extend([a_ref, b_ref])  # interleaved per pair to keep synthesis aligned
            pairs.append((a_ref, b_ref))
        return dict(mode="ref", channels_to_load=need, pairs=pairs, ref=ref)

    return None

# preprocessing filter - cases, sessions to include in the dataset from BIOT and CBraMod (comparability)
BIOT_PROCESS1_PARAMETERS = [
    ("01", "01", 2, 46, 0),
    ("02", "01", 2, 35, 0),
    ("03", "01", 2, 38, 0),
    ("05", "01", 2, 39, 0),
    ("06", "01", 2, 24, 0),
    ("07", "01", 2, 19, 0),
    ("08", "02", 3, 29, 0),
    ("10", "01", 2, 89, 0),
    ("11", "01", 2, 99, 0),
    ("14", "01", 2, 42, 0),
    ("20", "01", 2, 68, 0),
    ("21", "01", 2, 33, 0),
    ("22", "01", 2, 77, 0),
    ("23", "06", 7, 20, 0),
    ("24", "01", 3, 21, 0),
    ("04", "07", 1, 43, 1),
    ("09", "02", 1, 19, 1),
    ("15", "02", 1, 63, 1),
    ("16", "01", 2, 19, 0),
    ("18", "02", 1, 36, 1),
    ("19", "02", 1, 30, 1),
]


def build_biot_process1_allowed_files():
    allowed = set()

    for patient, ref, start, end, _ in BIOT_PROCESS1_PARAMETERS:
        allowed.add(f"chb{patient}_{ref}")

        for n in range(start, end + 1):
            allowed.add(f"chb{patient}_{n:02d}")

    return allowed


BIOT_PROCESS1_ALLOWED_FILES = build_biot_process1_allowed_files()


@contextmanager
def _tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument.
    """
    if tqdm_object is None:
        # no-op if tqdm not available
        yield None
        return

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


class CHBSelfsupervisedCompatDatasetView(EpilepsiaDatasetView):
    DEFAULT_CFG = {
        "sampling_strategy": "pretraining",
        "split": "predef",
        "split_ratio": [0.7, 0.2, 0.1],
        "split_def": {
            'train': [f'chb{i:02d}' for i in range(1, 21)],
            'val': ['chb21', 'chb22'],
            'test': ['chb23', 'chb24']
        },
        "sampling_freq": 256,
        "seq_len": 30,
        "chunk_size": 10.0,      # seconds
        "max_ts": 10**9,

        "omit_nonictal_sessions": False,
        "omit_nonictal_patients": False,

        "signal_channels": None,

        "seizure_delimiter_markers": ["eeg-beginn", "eeg-ende"],
        'markers_to_reconstruct': ['eeg-beginn', 'eeg-ende'],

        "context_padding": "replicate",
        "biot_compat": True,
        "augment_ictal": True,
        "context_label_mode": "overlap",
        "apply_biot_process1_file_filter": True,
    }

    def __init__(
        self,
        name: str = "CHBDatasetView",
        core_ds=None,
        cfg: dict = None,
        dv_filepath: str = None,
        use_cache: bool = False,
        mode: str = "unsupervised",
    ):
        requested_cfg = cfg or {}
        base_cfg = dict(self.DEFAULT_CFG)
        base_cfg.update(requested_cfg)

        super().__init__(name, core_ds, dv_filepath, base_cfg, use_cache)

        self.mode = mode
        self._bad_indices = set()

        if self.dv_filepath:
            self.__load()
        else:
            print(f"...creating dataset view {self.name} from {self.core_ds.name}")
            self._update_tps_tts_from_segments()
            self.samples_df = pd.DataFrame(
                columns=["sample_id", "patient_id", "timestamp", "filename", "start_index", "end_index"]
            )
            self.chunks_df = pd.DataFrame(
                columns=["sample_id", "timestep", "seg_type", "markers", "label", "time_past_seizure", "time_to_seizure", "notes"]
            )
            self._retrieve_samples_parallel()
            self._split_samples()

        self._init_split_views()

    # ------------------------------------------------------------------
    # loading / saving
    # ------------------------------------------------------------------
    def save(self, dv_filepath: str):
        print(f"...saving dataset view {self.name} to {dv_filepath}")
        self.__set_explicit_types()
        with pd.HDFStore(dv_filepath, "w") as h:
            h.put("params", pd.DataFrame())
            h.get_storer("params").attrs["name"] = self.name
            h.get_storer("params").attrs["core_ds_filepath"] = self.core_ds.ds_filepath
            h.put("config", pd.Series(json.dumps(self.cfg)))
            h.put("samples", self.samples_df)
            h.put("chunks", self.chunks_df)
            h.put('marker_vocab', self.marker_vocab)

    def __load(self):
        print(f"...loading dataset view {self.name} from {self.dv_filepath}")
        with pd.HDFStore(self.dv_filepath, "r") as h:
            self.name = h.get_storer("params").attrs["name"]
            self.core_ds_filepath = h.get_storer("params").attrs["core_ds_filepath"]
            self.cfg = json.loads(h["config"].item())
            self.samples_df = h["samples"]
            self.chunks_df = h["chunks"]
            self.marker_vocab = h['marker_vocab']
        self.core_ds = CHBMITCoreDataset(ds_filepath=self.core_ds_filepath)
        self._init_split_views()

    # ------------------------------------------------------------------
    # TPS / TTS calculations
    # ------------------------------------------------------------------
    def _update_tps_tts_from_segments(self):
        seg = self.core_ds.segments_df.copy()
        seg["time_to_next_ictal"] = self.cfg["max_ts"]
        seg["time_to_prev_ictal"] = self.cfg["max_ts"]

        for pid, g in seg.groupby("patient_id"):
            g = g.sort_values(["epoch_start_ts", "start_ts"]).reset_index()
            # collect ictal onsets
            ictal_starts = g[g["seg_type"] == "ictal"]["epoch_start_ts"]
            ictal_starts = ictal_starts.dropna().sort_values().to_list()

            last_ictal_end = None
            for idx, row in g.iterrows():
                est = row["epoch_start_ts"]
                if pd.isna(est):
                    continue

                # time to next ictal
                ttn = self.cfg["max_ts"]
                for s in ictal_starts:
                    if s >= est:
                        ttn = int((s - est).total_seconds())
                        break

                # time to prev ictal
                ttp = self.cfg["max_ts"]
                if last_ictal_end is not None:
                    ttp = int((est - last_ictal_end).total_seconds())

                seg.loc[row["index"], "time_to_next_ictal"] = ttn
                seg.loc[row["index"], "time_to_prev_ictal"] = ttp

                if row["seg_type"] == "ictal":
                    end_abs = row["epoch_start_ts"] + timedelta(seconds=float(row["duration"]))
                    last_ictal_end = end_abs

        self.core_ds.segments_df = seg

    # ------------------------------------------------------------------
    # build samples/chunks
    # ------------------------------------------------------------------

    def _retrieve_samples_parallel(self, n_jobs: int = 8, desc: str = "Building samples"):
        sess_df = self.core_ds.session_df.copy()

        # marker vocabulary - for later encoding with frozen sentence BERT
        self.marker_vocab = pd.DataFrame(self.core_ds.markers_df['marker'].unique())

        # filter by patients with seizures
        if self.cfg.get("omit_nonictal_patients", False):
            pats_with_sz = self.core_ds.patients_df[self.core_ds.patients_df["n_seizures"] > 0]["patient_id"]
            sess_df = sess_df[sess_df["patient_id"].isin(pats_with_sz)]

        if self.cfg.get("omit_nonictal_sessions", False):
            sess_df = sess_df[sess_df["n_seizures"] > 0]

        sess_df = sess_df.reset_index(drop=True)

        # materialize rows once (avoids re-iterating generators)
        rows = [row for _, row in sess_df.iterrows()]
        total = len(rows)

        if n_jobs == 1:
            # sequential with (nice) tqdm if available
            all_samples, all_chunks = [], []
            iterator = rows
            if tqdm is not None:
                iterator = tqdm(rows, total=total, desc=desc, unit="session")
            for row in iterator:
                s_list, c_list = self._build_samples_for_session(row)
                all_samples.extend(s_list)
                all_chunks.extend(c_list)
        else:
            # parallel + tqdm via context manager
            pbar = tqdm(total=total, desc=desc, unit="session") if tqdm is not None else None
            with _tqdm_joblib(pbar):
                results = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(self._build_samples_for_session)(row) for row in rows
                )
            all_samples, all_chunks = [], []
            for s_list, c_list in results:
                all_samples.extend(s_list)
                all_chunks.extend(c_list)

        self.samples_df = pd.DataFrame(all_samples)
        self.chunks_df = pd.DataFrame(all_chunks)

    def _build_samples_for_session(self, sess_row: pd.Series):
        """ This version replicates the BIOT/CBraMod/... preprocessing steps (process2.py)
            using the data structures and classes used in NeuroNTP.
        """
        cfg = self.cfg
        sfreq = cfg["sampling_freq"]
        chunk_size = cfg["chunk_size"]
        max_ts = cfg["max_ts"]

        pid = sess_row["patient_id"]
        sid = int(sess_row["session_id"])
        cid = int(sess_row["chunk_id"])
        fname = sess_row["filename"]

        # if BIOT / CBraMod (process1.py) file filter is active - include only allowed files
        if cfg.get("apply_biot_process1_file_filter", True):
            base = os.path.basename(str(fname)).replace(".h5", "").replace(".pkl", "").replace(".edf", "")
            if base not in BIOT_PROCESS1_ALLOWED_FILES:
                return [], []

        mst = sess_row["measurement_start_time"]
        dur = float(sess_row["chunk_duration_used_sec"])
        channels = sess_row["channels"]

        segs = self.core_ds.segments_df[
            (self.core_ds.segments_df["patient_id"] == pid)
            & (self.core_ds.segments_df["session_id"] == sid)
            & (self.core_ds.segments_df["chunk_id"] == cid)
        ].sort_values("start_ts")

        # markers for this (patient, session)
        mk_all = self.core_ds.markers_df
        if mk_all is not None and not mk_all.empty:
            mk_session = mk_all[(mk_all["patient_id"] == pid) & (mk_all["session_id"] == sid)].copy()
        else:
            mk_session = pd.DataFrame(columns=["epoch_onset", "marker", "notes"])

        if segs.empty:
            segs = pd.DataFrame([{
                "patient_id": pid,
                "session_id": sid,
                "chunk_id": cid,
                "start_ts": 0.0,
                "end_ts": dur,
                "duration": dur,
                "seg_type": "non-ictal",
                "epoch_start_ts": mst,
                "epoch_end_ts": mst + timedelta(seconds=dur),
                "time_to_prev_ictal": max_ts,
                "time_to_next_ictal": max_ts,
                "filename": fname,
                "channels": channels
            }])

        # BIOT/CBraMod compatible sampling (process2.py):
        # 1) regular non-overlapping 10-sec chunks
        # 2) additional positive 10-sec chunks around seizure intervals with 5-sec stride
        win_samples = int(round(chunk_size * sfreq))
        step_regular = win_samples
        step_aug = win_samples // 2
        n_samples_total = int(round(dur * sfreq))

        samples = []
        chunks = []

        base_sample_id = int(
            hashlib.md5(f"{pid}-{sid}-{fname}".encode()).hexdigest()[:12],
            16
        )
        sample_id = base_sample_id

        seizure_times = [
            (float(r["start_ts"]), float(r["end_ts"]))
            for _, r in segs[segs["seg_type"] == "ictal"].iterrows()
        ]

        def add_sample(start_sample: int, augmented: bool = False, seizure_idx: int = -1):
            nonlocal sample_id

            end_sample = start_sample + win_samples
            start = start_sample / sfreq
            end = end_sample / sfreq

            # positive only if seizure start or end boundary lies inside this 10-sec segment (process2.py)
            label = 0
            for sz_start_s, sz_end_s in seizure_times:
                sz_start = int(round(sz_start_s * sfreq)) - 1
                sz_end = int(round(sz_end_s * sfreq)) - 1

                if cfg.get("biot_compat", False):
                    # BIOT/CBraMod-compatible boundary-only labeling
                    if (
                        start_sample < sz_start < end_sample
                        or start_sample < sz_end < end_sample
                    ):
                        label = 1
                        break
                else:
                    # proper interval-overlap labeling
                    if start_sample < sz_end and end_sample > sz_start:
                        label = 1
                        break

            # augmentation loop: force-positive chunks around seizure interval, as in process2.py
            if augmented and cfg.get("biot_compat", False):
                label = 1

            seg_type = "ictal" if label == 1 else "non-ictal"

            # keep the previous TPS/TTS estimate logic, but evaluated for this target chunk
            ov = segs[(segs["end_ts"] > start) & (segs["start_ts"] < end)]

            if ov.empty:
                tps = float(max_ts)
                tts = float(max_ts)
            else:
                s0 = segs[(segs["start_ts"] <= start) & (segs["end_ts"] > start)].sort_values(
                    ["start_ts", "end_ts"]
                ).tail(1)

                s1 = segs[(segs["start_ts"] < end) & (segs["end_ts"] >= end)].sort_values(
                    ["start_ts", "end_ts"]
                ).head(1)

                if not s0.empty and s0.iloc[0]["seg_type"] == "ictal":
                    tps = 0.0
                else:
                    base = float(s0.iloc[0].get("time_to_prev_ictal", max_ts)) if not s0.empty else float(max_ts)
                    elapsed = start - (s0.iloc[0]["start_ts"] if not s0.empty else start)
                    tps = base + float(max(0.0, elapsed))

                if not s1.empty and s1.iloc[0]["seg_type"] == "ictal":
                    tts = 0.0
                else:
                    base = float(s1.iloc[0].get("time_to_next_ictal", max_ts)) if not s1.empty else float(max_ts)
                    remaining = (s1.iloc[0]["end_ts"] - end) if not s1.empty else 0.0
                    tts = base + float(max(0.0, remaining))

            def _cap(x):
                return float(min(max_ts, max(0.0, x))) if np.isfinite(x) else float(max_ts)

            tps = _cap(tps)
            tts = _cap(tts)

            # markers belonging to this 10-sec target chunk [start, end)
            if not mk_session.empty:
                epoch_start = mst + timedelta(seconds=float(start))
                epoch_end = mst + timedelta(seconds=float(end))
                mk = mk_session[
                    (mk_session["epoch_onset"] >= epoch_start) &
                    (mk_session["epoch_onset"] < epoch_end)
                    ]
                if not mk.empty:
                    mkr = ", ".join(sorted(set(mk["marker"].astype(str))))
                    if "notes" in mk.columns:
                        notes = ", ".join([n for n in mk["notes"].fillna("").astype(str) if n.strip() != ""])
                    else:
                        notes = ""
                else:
                    mkr, notes = "", ""
            else:
                mkr, notes = "", ""

            samples.append({
                "sample_id": sample_id,
                "patient_id": pid,
                "timestamp": mst + timedelta(seconds=float(start)),
                "filename": fname,
                "start_index": int(start_sample),
                "end_index": int(end_sample),
                "channels": channels,
                "label": int(label),
                "seg_type": seg_type,
                "augmented": bool(augmented),
                "seizure_aug_idx": int(seizure_idx),
                "session_id": sid,
                "chunk_id": cid,
                "measurement_start_time": mst,
            })

            chunks.append({
                "sample_id": sample_id,
                "timestep": 0,
                "seg_type": seg_type,
                "markers": mkr,
                "label": int(label),
                "time_past_seizure": tps,
                "time_to_seizure": tts,
                "notes": notes,
            })

            sample_id += 1

        # regular loop over the recorded session (process2.py from BIOT)
        for i in range(0, n_samples_total, step_regular):
            if i + win_samples <= n_samples_total:
                add_sample(start_sample=i, augmented=False)

        # positive augmentation loop (process2.py from BIOT)
        if cfg.get("augment_ictal", False):
            for idx, (sz_start_s, sz_end_s) in enumerate(seizure_times):
                sz_start = int(round(sz_start_s * sfreq)) - 1
                sz_end = int(round(sz_end_s * sfreq)) - 1

                for i in range(
                    max(0, sz_start - int(sfreq)),
                    min(sz_end + int(sfreq), n_samples_total),
                    step_aug,
                ):
                    #if i + win_samples <= n_samples_total:
                    add_sample(start_sample=i, augmented=True, seizure_idx=idx)

        return samples, chunks

    # ------------------------------------------------------------------
    # split
    # ------------------------------------------------------------------
    def _split_samples(self):
        df = self.samples_df.copy()
        ratio = self.cfg["split_ratio"]
        kind = self.cfg["split"]

        if kind == "predef":
            split_def = self.cfg.get("split_def", {})

            # build reverse lookup: patient_id -> split
            pat_to_split = {}
            for split_name, pats in split_def.items():
                for p in pats:
                    pat_to_split[p] = split_name

            # map patient_id to split
            df["split"] = df["patient_id"].map(pat_to_split)

            # optional safety: drop or warn on unmapped patients
            if df["split"].isna().any():
                missing = df[df["split"].isna()]["patient_id"].unique()
                raise ValueError(f"Patients not covered by split_def: {missing}")

        elif kind == "patient":
            pats = df["patient_id"].unique()
            n = len(pats)
            i1, i2 = int(n * ratio[0]), int(n * (ratio[0] + ratio[1]))
            train_p = set(pats[:i1])
            val_p = set(pats[i1:i2])
            df["split"] = np.where(
                df["patient_id"].isin(train_p), "train",
                np.where(df["patient_id"].isin(val_p), "val", "test")
            )

        elif kind == "temporal":
            df = df.sort_values(["patient_id", "timestamp", "start_index"])
            df["split"] = "train"
            for pid, g in df.groupby("patient_id"):
                idx = g.index
                m = len(idx)
                i1, i2 = int(m * ratio[0]), int(m * (ratio[0] + ratio[1]))
                df.loc[idx[:i1], "split"] = "train"
                df.loc[idx[i1:i2], "split"] = "val"
                df.loc[idx[i2:], "split"] = "test"

        else:
            perm = np.random.permutation(len(df))
            m = len(df)
            i1, i2 = int(m * ratio[0]), int(m * (ratio[0] + ratio[1]))
            df.loc[perm[:i1], "split"] = "train"
            df.loc[perm[i1:i2], "split"] = "val"
            df.loc[perm[i2:], "split"] = "test"

        self.samples_df = df.reset_index(drop=True)

    def _init_split_views(self):
        self.split_views = {
            split: self.samples_df[self.samples_df["split"] == split].reset_index(drop=True)
            for split in ["train", "val", "test"]
        }

        sid_by_split = {s: set(df["sample_id"].unique()) for s, df in self.split_views.items()}

        self.chunks_split_views = {}
        for split, sids in sid_by_split.items():
            cdf = self.chunks_df[self.chunks_df["sample_id"].isin(sids)]
            cdf = cdf.reset_index(drop=True)
            self.chunks_split_views[split] = cdf

    def init_random_subsampling(
        self,
        seed=42,
        subset_size=0.2,
        balance_ratio=0.2,
        proximity_radius=None,
    ):
        self.random = random.Random(seed)
        self.all_samples_df = self.samples_df.copy()
        self.all_chunks_df = self.chunks_df.copy()
        self.subset_size = subset_size
        self.bal_ictal_ratio = balance_ratio
        self.proximity_radius = proximity_radius

    def random_subsample(self):
        subset = []

        for split in ["train", "val", "test"]:
            sdf = self.all_samples_df[self.all_samples_df["split"] == split]

            # keep test untouched  --- test is always run in a separate session w/o rnd
            #if split == "test":
            #    subset += sdf["sample_id"].tolist()
            #    print(f"...{split}: kept full set total={len(sdf)}")
            #    continue

            rnd = self.random if split == "train" else random.Random(42)

            # --------------------------------------------------
            # Optional seizure-proximity candidate restriction
            # --------------------------------------------------
            if self.proximity_radius is not None:
                prox_ids = self.all_chunks_df.loc[
                    (self.all_chunks_df["time_to_seizure"] < self.proximity_radius)
                    | (self.all_chunks_df["time_past_seizure"] < self.proximity_radius),
                    "sample_id",
                ].unique()

                sdf = sdf[sdf["sample_id"].isin(prox_ids)]

                print(
                    f"...{split}: proximity filter radius={self.proximity_radius}, "
                    f"candidates={len(sdf)}"
                )

            sample_size = int(self.subset_size * len(sdf))

            # --------------------------------------------------
            # Balanced sampling, if requested
            # --------------------------------------------------
            if self.bal_ictal_ratio is not None:
                pos_ids = sdf[sdf["label"] == 1]["sample_id"].tolist()
                neg_ids = sdf[sdf["label"] == 0]["sample_id"].tolist()

                n_pos = int(round(sample_size * self.bal_ictal_ratio))
                n_neg = sample_size - n_pos

                n_pos = min(n_pos, len(pos_ids))
                n_neg = min(n_neg, len(neg_ids))

                subset += rnd.sample(pos_ids, n_pos)
                subset += rnd.sample(neg_ids, n_neg)

                print(
                    f"...{split}: ratio subsample "
                    f"pos={n_pos}, neg={n_neg}, total={n_pos + n_neg}, "
                    f"ictal_ratio={n_pos / max(1, n_pos + n_neg):.4f}"
                )

            # --------------------------------------------------
            # Plain random sampling
            # --------------------------------------------------
            else:
                ids = sdf["sample_id"].tolist()
                n = min(sample_size, len(ids))
                subset += rnd.sample(ids, n)

                print(f"...{split}: random subsample total={n}")

        self.chunks_df = self.all_chunks_df[
            self.all_chunks_df["sample_id"].isin(subset)
        ].copy()

        self.samples_df = self.all_samples_df[
            self.all_samples_df["sample_id"].isin(subset)
        ].copy()

        self._init_split_views()

        print(f"...randomized subsampling: total={len(subset)} samples")

    # ------------------------------------------------------------------
    # getters, helpers
    # ------------------------------------------------------------------
    def __len__(self):
        if self.mode in ["unsupervised", "supervised"]:
            return len(self.samples_df)
        else:
            raise ValueError(f"unknown mode: {self.mode}")

    def get_len_type(self, ds_type):
        return len(self.split_views[ds_type])

    def __getitem__(self, idx):
        return self.get_item_of_type("train", idx)

    def get_item_of_type(self, ds_type, index):
        if self.mode == "unsupervised":
            return self._get_unsupervised_sample(ds_type, index)
        elif self.mode == "supervised":
            return self._get_supervised_sample(ds_type, index)
        else:
            raise ValueError(f"unknown mode: {self.mode}")

    def _build_seq_from_sample(self, sample_row: pd.Series, start_timestep: int):
        # config
        sfreq = self.cfg["sampling_freq"]
        chunk_size = self.cfg["chunk_size"]
        seq_len = self.cfg["seq_len"]

        stride_s = chunk_size
        stride = int(round(stride_s * sfreq))
        win = int(round(chunk_size * sfreq))
        sid = int(sample_row["session_id"])
        mst = sample_row["measurement_start_time"]

        fname = sample_row["filename"]

        target_start = int(sample_row["start_index"])
        target_end = int(sample_row["end_index"])

        context_start = target_start - (seq_len - 1) * stride
        load_start = max(0, context_start)
        load_end = target_end
        missing_left = max(0, (load_start - context_start + stride - 1) // stride)

        sample_id = sample_row["sample_id"]
        patient_id = sample_row.get("patient_id", None)
        timestamp = sample_row.get("timestamp", None)
        channels = sample_row["channels"]

        if start_timestep < 0:
            raise ValueError("start_timestep < 0 (cannot pad; dropping)")

        # window start/end indices in the source file
        window_start_idx = load_start
        window_end_idx = load_end

        # channels for this sample
        canon_pairs = _pairs_from_canon(self.canon_ch_names)  # preserves the exact order
        #ch_names = [ch for ch in channels.upper().split(',') if ch.upper() in self.canon_ch_names]

        plan = plan_loading(channels, canon_pairs)
        if plan is None:
            raise ValueError("Recording channels don’t match canonical 16-ch requirements (drop)")

        fpath = (os.path.join(self.core_ds.hdf_store_path, fname)
                 if self.core_ds and getattr(self.core_ds, "hdf_store_path", None) and not os.path.isabs(fname)
                 else fname)

        data = load_partial_hdf5(
            fpath,
            start_idx=window_start_idx,
            stop_idx=window_end_idx,
            only_signal=True,
            channel_order=plan["channels_to_load"],
        )

        # --------------------------------------- #
        # SIGNAL ................................ #
        # --------------------------------------- #
        sig = torch.tensor(data["signal"], dtype=torch.float32)  # shape: (C_loaded, N)

        if plan["mode"] == "bipolar":
            # already in canonical order, shape should be (16, N)
            pass

        elif plan["mode"] == "ref":
            # synthesize canonical bipolars in the fixed CANON_PAIRS order
            C_loaded, N = sig.shape
            assert C_loaded == 2 * len(CANON_PAIRS)
            sig = sig[0::2, :] - sig[1::2, :]

        # final guard
        expected_c = len(canon_pairs)
        if sig.shape[0] != expected_c:
            raise RuntimeError(f"channel count mismatch: got {sig.shape[0]}, expected {expected_c}")

        # If we asked beyond EOF, shrink to what we actually got
        N = sig.shape[1]  # available samples loaded
        if N < win:
            raise RuntimeError(
                f"too short window: N={N} < win={win} (sid={sample_id}, start_timestep={start_timestep})")

        eff_T = 1 + (N - win) // stride
        if eff_T <= 0:
            raise RuntimeError(
                f"insufficient signal: eff_T={eff_T} "
                f"(sid={sample_id}, start_timestep={start_timestep})"
            )

        sig = self._normalize_torch(sig)
        sig = sig.unfold(dimension=1, size=win, step=stride)  # (C, T_available, win)

        # left-pad if this is an early recording chunk
        if missing_left > 0:
            pad_mode = self.cfg.get("context_padding", "replicate")

            if pad_mode == "zero":
                pad = torch.zeros(
                    sig.shape[0],
                    missing_left,
                    sig.shape[2],
                    dtype=sig.dtype,
                    device=sig.device,
                )
            elif pad_mode == "replicate":
                pad = sig[:, :1, :].repeat(1, missing_left, 1)
            else:
                raise ValueError(f"unknown context_padding: {pad_mode}")

            sig = torch.cat([pad, sig], dim=1)

        # keep exactly the last seq_len chunks ending at the target chunk
        if sig.shape[1] < seq_len:
            raise RuntimeError(
                f"still insufficient after padding: T={sig.shape[1]} < seq_len={seq_len} "
                f"(sid={sample_id})"
            )

        signal = sig[:, -seq_len:, :].contiguous()

        # --------------------------------------- #
        # CHUNKS ................................ #
        # --------------------------------------- #

        # Dynamically build chunk metadata for the seq_len context ending at target chunk.
        # This mirrors the signal sequence. Initial negative positions are left-replicated.
        pid = sample_row["patient_id"]

        segs = self.core_ds.segments_df[
            (self.core_ds.segments_df["patient_id"] == pid)
            & (self.core_ds.segments_df["filename"] == fname)
            ].sort_values("start_ts")

        mk_all = self.core_ds.markers_df
        if mk_all is not None and not mk_all.empty:
            mk_session = mk_all[
                (mk_all["patient_id"] == pid)
                & (mk_all["session_id"] == int(sample_row.get("session_id", -1)))
                ].copy()
        else:
            mk_session = pd.DataFrame(columns=["epoch_onset", "marker", "notes"])

        seizure_times = [
            (float(r["start_ts"]), float(r["end_ts"]))
            for _, r in segs[segs["seg_type"] == "ictal"].iterrows()
        ]

        def make_chunk_row(timestep: int, start_sample: int):
            end_sample = start_sample + win
            start = start_sample / sfreq
            end = end_sample / sfreq

            # historic context chunks
            label = 0
            for sz_start_s, sz_end_s in seizure_times:
                sz_start = int(round(sz_start_s * sfreq)) - 1
                sz_end = int(round(sz_end_s * sfreq)) - 1
                # interval-overlap labelling
                if start_sample < sz_end and end_sample > sz_start:
                    label = 1
                    break

            # For the final target timestep, preserve the stored sample label.
            # This is important for biot_compat and augmented positives.
            if timestep == seq_len - 1:
                label = int(sample_row["label"])

            seg_type = "ictal" if label == 1 else "non-ictal"

            ov = segs[(segs["end_ts"] > start) & (segs["start_ts"] < end)]

            if ov.empty:
                tps = float(self.cfg["max_ts"])
                tts = float(self.cfg["max_ts"])
            else:
                s0 = segs[
                    (segs["start_ts"] <= start)
                    & (segs["end_ts"] > start)
                    ].sort_values(["start_ts", "end_ts"]).tail(1)

                s1 = segs[
                    (segs["start_ts"] < end)
                    & (segs["end_ts"] >= end)
                    ].sort_values(["start_ts", "end_ts"]).head(1)

                if not s0.empty and s0.iloc[0]["seg_type"] == "ictal":
                    tps = 0.0
                else:
                    base = float(s0.iloc[0].get("time_to_prev_ictal", self.cfg["max_ts"])) if not s0.empty else float(
                        self.cfg["max_ts"])
                    elapsed = start - (s0.iloc[0]["start_ts"] if not s0.empty else start)
                    tps = base + float(max(0.0, elapsed))

                if not s1.empty and s1.iloc[0]["seg_type"] == "ictal":
                    tts = 0.0
                else:
                    base = float(s1.iloc[0].get("time_to_next_ictal", self.cfg["max_ts"])) if not s1.empty else float(
                        self.cfg["max_ts"])
                    remaining = (s1.iloc[0]["end_ts"] - end) if not s1.empty else 0.0
                    tts = base + float(max(0.0, remaining))

            def _cap(x):
                return float(min(self.cfg["max_ts"], max(0.0, x))) if np.isfinite(x) else float(self.cfg["max_ts"])

            tps = _cap(tps)
            tts = _cap(tts)

            if not mk_session.empty:
                epoch_start = mst + timedelta(seconds=float(start))
                epoch_end = mst + timedelta(seconds=float(end))
                mk = mk_session[
                    (mk_session["epoch_onset"] >= epoch_start)
                    & (mk_session["epoch_onset"] < epoch_end)
                    ]
                if not mk.empty:
                    mkr = ", ".join(sorted(set(mk["marker"].astype(str))))
                    if "notes" in mk.columns:
                        notes = ", ".join([n for n in mk["notes"].fillna("").astype(str) if n.strip() != ""])
                    else:
                        notes = ""
                else:
                    mkr, notes = "", ""
            else:
                mkr, notes = "", ""

            return {
                "sample_id": sample_id,
                "timestep": timestep,
                "seg_type": seg_type,
                "markers": mkr,
                "label": int(label),
                "time_past_seizure": tps,
                "time_to_seizure": tts,
                "notes": notes,
            }

        context_starts = [
            target_start - (seq_len - 1 - t) * stride
            for t in range(seq_len)
        ]

        # Left-replicate initial chunk metadata if context_start < 0.
        context_starts = [
            0 if st < 0 else st
            for st in context_starts
        ]

        chks = pd.DataFrame([
            make_chunk_row(t, st)
            for t, st in enumerate(context_starts)
        ])

        chks = chks[
            [
                "sample_id",
                "timestep",
                "seg_type",
                "markers",
                "label",
                "time_past_seizure",
                "time_to_seizure",
                "notes",
            ]
        ]

        # --------------------------------------- #
        # MARKERS/NOTES (optional) .............. #
        # --------------------------------------- #
        # optional marker/notes embeddings (same as before)
        marker_embs, notes_embs, modalities = None, None, ["signals", "TTS", "segtype", "TPS"]
        if "markers" in chks.columns and hasattr(self, "marker_emb_dict"):
            mks = chks["markers"].fillna("").astype(str).tolist()
            if len(mks) > 0:
                marker_embs = torch.stack(
                    [self.marker_emb_dict.get(m, self.marker_emb_dict.get("", torch.zeros(768))) for m in mks], dim=0
                )
                modalities.append("Markers")
        if "notes" in chks.columns and hasattr(self, "notes_emb_dict"):
            nts = chks["notes"].fillna("").astype(str).tolist()
            if len(nts) > 0:
                notes_embs = torch.stack(
                    [self.notes_emb_dict.get(n, self.notes_emb_dict.get("", torch.zeros(768))) for n in nts], dim=0
                )
                modalities.append("Notes")

        modalities.append("EoS")

        metadata = {
            "patient_id": patient_id,
            "filename": fname,
            "timestamp": timestamp,
            #"signal_channels": data.get("ch_names", None),
            "signal_channels": self.norm_signal_channels,
            "modalities": modalities,
        }

        return EpiSelfSupervisedSample(
            signal=signal,  # (C, T, D)
            chunks=chks.to_numpy(),
            markers_embs=marker_embs,
            notes_embs=notes_embs,
            metadata=metadata,
        )

    def _assert_nonempty(self, sig: torch.Tensor, msg: str):
        if sig.numel() == 0 or sig.shape[1] == 0:
            raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # unsupervised: whole sequence
    # ------------------------------------------------------------------
    def _get_unsupervised_sample(self, ds_type, index, max_tries=5):
        tries = 0

        while tries < max_tries:
            df = self.split_views[ds_type] if ds_type else self.samples_df
            row = df.iloc[index]

            try:
                return self._build_seq_from_sample(row, start_timestep=0)

            except (OSError, IOError, KeyError, RuntimeError, ValueError) as e:
                # collect debug info
                info = {
                    "ds_type": ds_type,
                    "index": index,
                    "sample_id": row.get("sample_id"),
                    "patient_id": row.get("patient_id"),
                    "filename": row.get("filename"),
                    "start_index": row.get("start_index"),
                    "end_index": row.get("end_index"),
                    "label": row.get("label"),
                    "augmented": row.get("augmented", None),
                    "seizure_aug_idx": row.get("seizure_aug_idx", None),
                    "error": repr(e),
                }

                # store structured info
                if not hasattr(self, "_bad_samples"):
                    self._bad_samples = []
                self._bad_samples.append(info)

                # print only a few (avoid flooding)
                if len(self._bad_samples) <= 20:
                    print("\n[WARN sample load failure]")
                    for k, v in info.items():
                        print(f"{k}: {v}")

                self._bad_indices.add((ds_type, index))

                tries += 1
                index = self._pick_alt_index(ds_type, index)

        raise RuntimeError("too many failed attempts to load sample")

    def _get_supervised_sample(self, ds_type, index):
        df = self.split_views[ds_type] if ds_type else self.samples_df
        row = df.iloc[index]

        fname = row["filename"]
        start_idx = int(row["start_index"])
        end_idx = int(row["end_index"])

        canon_pairs = _pairs_from_canon(self.canon_ch_names)
        plan = plan_loading(row["channels"], canon_pairs)
        if plan is None:
            raise ValueError("Recording channels don’t match canonical 16-ch requirements")

        fpath = (
            os.path.join(self.core_ds.hdf_store_path, fname)
            if self.core_ds
               and getattr(self.core_ds, "hdf_store_path", None)
               and not os.path.isabs(fname)
            else fname
        )

        data = load_partial_hdf5(
            fpath,
            start_idx=start_idx,
            stop_idx=end_idx,
            only_signal=True,
            channel_order=plan["channels_to_load"],
        )

        signal = torch.tensor(data["signal"], dtype=torch.float32)

        if plan["mode"] == "ref":
            signal = signal[0::2, :] - signal[1::2, :]

        signal = self._normalize_torch(signal)

        label = int(row["label"])

        metadata = {
            "patient_id": row.get("patient_id"),
            "filename": fname,
            "timestamp": row.get("timestamp"),
            "start_index": start_idx,
            "end_index": end_idx,
            "label": label,
            "sample_id": row["sample_id"],
            "signal_channels": self.norm_signal_channels,
        }

        return signal, label, metadata

    # ------------------------------------------------------------------
    # misc utils
    # ------------------------------------------------------------------
    def _normalize_torch(self, X: torch.Tensor) -> torch.Tensor:
        X = X - X.mean(dim=-1, keepdim=True)
        q = torch.quantile(torch.abs(X), 0.95, dim=-1, keepdim=True)
        X = X / (q + 1e-8)
        return X

    def _pick_alt_index(self, ds_type, avoid_idx, max_tries=10):
        df = self.split_views[ds_type] if ds_type else self.samples_df
        n = len(df)
        for _ in range(max_tries):
            j = np.random.randint(0, n)
            if j != avoid_idx and (ds_type, j) not in self._bad_indices:
                return j
        return avoid_idx

    def __set_explicit_types(self):
        self.samples_df = self.samples_df.copy()
        self.chunks_df = self.chunks_df.copy()

        self.samples_df["sample_id"] = self.samples_df["sample_id"].astype(int)
        self.samples_df["patient_id"] = self.samples_df["patient_id"].astype(str)
        self.samples_df["filename"] = self.samples_df["filename"].astype(str)
        self.samples_df["start_index"] = self.samples_df["start_index"].astype(int)
        self.samples_df["end_index"] = self.samples_df["end_index"].astype(int)
        self.samples_df["split"] = self.samples_df["split"].astype(str)

        self.chunks_df["sample_id"] = self.chunks_df["sample_id"].astype(int)
        self.chunks_df["timestep"] = self.chunks_df["timestep"].astype(int)
        self.chunks_df["seg_type"] = self.chunks_df["seg_type"].astype(str)
        self.chunks_df["label"] = self.chunks_df["label"].astype(int)

        for col in ("time_past_seizure", "time_to_seizure"):
            s = self.chunks_df[col].astype("float64")
            s = s.mask(~np.isfinite(s), float(self.cfg["max_ts"]))
            s = s.clip(lower=0.0, upper=float(self.cfg["max_ts"]))
            self.chunks_df[col] = s.astype("float32")

        for col in ("session_id", "chunk_id", "label", "seizure_aug_idx"):
            if col in self.samples_df.columns:
                self.samples_df[col] = self.samples_df[col].astype(int)

        if "seg_type" in self.samples_df.columns:
            self.samples_df["seg_type"] = self.samples_df["seg_type"].astype(str)

        if "augmented" in self.samples_df.columns:
            self.samples_df["augmented"] = self.samples_df["augmented"].astype(bool)


    def update_channel_config(self, cfg):
        self.cfg['signal_channels'] = cfg['signal_channels']
        self.supported_channels = {ch.lower() for ch in cfg['signal_channels']}

    @property
    def norm_signal_channels(self):
        return self._norm_ch_names(self.cfg['signal_channels'])

    def _norm_ch_names(self, ch_names):
        out = []
        for ch in ch_names:
            s = ' '.join(str(ch).split())  # collapse whitespace
            out.append(s)
        return out

    @property
    def canon_ch_names(self):
        return [ch.upper()[4:] for ch in self.norm_signal_channels]

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
        self.vocab_emb_dict = dict()
        self.marker_emb_dict = dict()
        self.notes_emb_dict = dict()

        # filter chunks labels
        if reduced_vocab:
            filter_keywords = dict.fromkeys(self.cfg['filter_keywords'], '')
            if filter_explicit:
                self.chunks_df.loc[~self.chunks_df['markers'].isin(filter_keywords.keys()), 'markers'] = ''
            else:
                mask = self.chunks_df['markers'].apply(lambda x: any(key in x for key in filter_keywords.keys()))
                self.chunks_df.loc[~mask, 'markers'] = ''

        # calculate embeddings with frozen sentence BERT
        self.marker_embedder.eval()
        with torch.no_grad():
            # create marker occurrences
            markers = list(self.chunks_df['markers'].unique())
            for marker_batch in self.batch_iterable(markers, batch_size):
                embeddings = marker_embedder.get_emb(marker_batch)
                embeddings = embeddings.detach().cpu()
                self.marker_emb_dict.update({m: e for m, e in zip(marker_batch, embeddings)})

            # create marker dictionary embeddings
            for marker_batch in self.batch_iterable(filter_keywords.keys() if reduced_vocab else self.marker_vocab[0], batch_size):
                embeddings = marker_embedder.get_emb(marker_batch)
                embeddings = embeddings.detach().cpu()
                self.vocab_emb_dict.update({m: e for m, e in zip(marker_batch, embeddings)})

            # create notes occurrences
            notes = list(self.chunks_df['notes'].unique())
            for notes_batch in self.batch_iterable(notes, batch_size):
                embeddings = marker_embedder.get_emb(notes_batch)
                embeddings = embeddings.detach().cpu()
                self.notes_emb_dict.update({n: e for n, e in zip(notes_batch, embeddings)})

        # markers relevant in reconstruction
        self.markers_to_reconstruct = self.cfg['markers_to_reconstruct']
        self.reconstr_marker_vocab = {word: i for i, word in enumerate(self.markers_to_reconstruct or [])}
        self.reconstr_vocab_size = len(self.reconstr_marker_vocab)
        self.reconstr_marker_decode_dict = {i: word for i, word in enumerate(self.markers_to_reconstruct or [])}

