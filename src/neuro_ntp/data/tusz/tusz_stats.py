#!/usr/bin/env python3
"""
Compute normalization statistics for TUSZ HDF repository.

Outputs a single HDF5 file with per-recording stats:
  - mean[C]
  - q95_abs_centered[C]  where q95 = quantile(|x - mean|, 0.95)
  - metadata: split, sfreq, n_channels, n_samples, channels

Designed to be multi-worker-friendly at training time: workers only load stats.

Example:
  python tusz_hdf_collect_norm_stats.py \
      --hdf_root /system/user/publicwork/eeg/data/processed/TUSZ_v2.0.3_hdf \
      --out_hdf  /system/user/publicwork/eeg/data/processed/TUSZ_v2.0.3_hdf/norm_stats.h5 \
      --q 0.95 \
      --max_points_per_ch 200000 \
      --chunk_samples 524288 \
      --n_jobs 8 \
      --write_csv

Notes:
- Uses streaming mean (exact) and q-quantile via random subsampling (approx, but very good with enough points).
- If recordings are short, you can set max_points_per_ch very high (or 0 for "take all" if memory allows).
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def infer_split_from_path(p: Path) -> str:
    parts = [x.lower() for x in p.parts]
    for s in ("train", "dev", "eval", "val", "test"):
        if s in parts:
            # keep TUSZ naming train/dev/eval
            return s
    return "unknown"


def list_h5_files(hdf_root: Path) -> List[Path]:
    # crawl train/dev/eval
    return sorted(hdf_root.rglob("*.h5")) + sorted(hdf_root.rglob("*.hdf5"))


def load_file_list(hdf_root: Path) -> pd.DataFrame:
    """
    Returns a DataFrame with at least columns:
      - filename (absolute path)
      - split (train/dev/eval if possible)
    If sessions.csv exists under root, use it.
    """
    sessions_csv = hdf_root / "sessions.csv"
    if sessions_csv.exists():
        df = pd.read_csv(sessions_csv)
        # Expect a filename column. If relative, make absolute.
        if "filename" not in df.columns:
            raise RuntimeError(f"{sessions_csv} exists but has no 'filename' column.")
        df = df.copy()
        df["filename"] = df["filename"].astype(str).apply(lambda x: str((hdf_root / x).resolve()) if not os.path.isabs(x) else x)
        if "split" not in df.columns:
            df["split"] = df["filename"].apply(lambda x: infer_split_from_path(Path(x)))
        return df[["filename", "split"]].drop_duplicates().reset_index(drop=True)

    # otherwise crawl
    files = list_h5_files(hdf_root)
    if not files:
        raise RuntimeError(f"No .h5/.hdf5 files found under {hdf_root}")
    df = pd.DataFrame({"filename": [str(p.resolve()) for p in files]})
    df["split"] = df["filename"].apply(lambda x: infer_split_from_path(Path(x)))
    return df


def _read_channels(h5: h5py.File) -> Optional[List[str]]:
    # Try common patterns
    for key in ("ch_names", "channels"):
        if key in h5:
            arr = h5[key][()]
            # handle bytes or str
            out = []
            for v in arr:
                if isinstance(v, (bytes, np.bytes_)):
                    out.append(v.decode("utf-8"))
                else:
                    out.append(str(v))
            return out
    # try attrs
    if "ch_names" in h5.attrs:
        v = h5.attrs["ch_names"]
        # might be array-like
        try:
            out = []
            for x in v:
                if isinstance(x, (bytes, np.bytes_)):
                    out.append(x.decode("utf-8"))
                else:
                    out.append(str(x))
            return out
        except TypeError:
            return [str(v)]
    return None


def _read_sfreq(h5: h5py.File) -> Optional[float]:
    for key in ("sfreq",):
        if key in h5.attrs:
            try:
                return float(h5.attrs[key])
            except Exception:
                pass
    # try info group
    if "info" in h5 and isinstance(h5["info"], h5py.Group):
        if "sfreq" in h5["info"].attrs:
            try:
                return float(h5["info"].attrs["sfreq"])
            except Exception:
                pass
    return None


def compute_stats_one_file(
    filename: str,
    q: float,
    max_points_per_ch: int,
    chunk_samples: int,
    seed: int,
) -> Dict:
    """
    Streaming mean (exact) + q-quantile of abs centered via random subsampling (approx).
    signal: (C,T)
    """
    rng = np.random.default_rng(seed)

    with h5py.File(filename, "r") as h5:
        if "signal" not in h5:
            raise RuntimeError("Missing dataset 'signal'")
        dset = h5["signal"]
        if dset.ndim != 2:
            raise RuntimeError(f"Expected signal ndim=2, got {dset.ndim}")

        C, T = int(dset.shape[0]), int(dset.shape[1])
        sfreq = _read_sfreq(h5)
        channels = _read_channels(h5)

        # --- pass 1: streaming mean ---
        sum_ = np.zeros((C,), dtype=np.float64)
        n_total = 0

        for s in range(0, T, chunk_samples):
            e = min(T, s + chunk_samples)
            x = dset[:, s:e].astype(np.float64)  # (C, chunk)
            sum_ += x.sum(axis=1)
            n_total += (e - s)

        mean = (sum_ / max(1, n_total)).astype(np.float32)

        # --- pass 2: subsample abs(x-mean) for quantile estimation ---
        # We want up to max_points_per_ch samples per channel.
        # If max_points_per_ch <= 0: fall back to exact by reading everything (may be memory heavy).
        if max_points_per_ch <= 0:
            # exact, but can be big
            x = dset[:, :].astype(np.float32)
            dev = np.abs(x - mean[:, None])
            qv = np.quantile(dev, q=q, axis=1).astype(np.float32)
        else:
            k = int(max_points_per_ch)
            # reservoir-style: sample indices uniformly from [0,T)
            # if T < k -> take all
            if T <= k:
                idx = np.arange(T, dtype=np.int64)
            else:
                idx = rng.choice(T, size=k, replace=False)
                idx.sort()

            # read in index-chunks to avoid too many tiny reads
            dev_samples = np.empty((C, idx.size), dtype=np.float32)

            # batch contiguous-ish ranges
            # (simple batching: read in blocks of consecutive indices)
            start = 0
            while start < idx.size:
                end = start + 1
                while end < idx.size and idx[end] == idx[end - 1] + 1:
                    end += 1

                # if run is too short, extend to a small window to reduce overhead
                # (optional heuristic)
                run_len = end - start
                if run_len < 1024 and end < idx.size:
                    # extend to include more indices even if not strictly consecutive
                    end = min(idx.size, start + 8192)

                cols = idx[start:end]
                # This read is not contiguous if cols are not consecutive,
                # but h5py supports fancy indexing. Keep batches moderate.
                x = dset[:, cols].astype(np.float32)  # (C, batch)
                dev_samples[:, start:end] = np.abs(x - mean[:, None])

                start = end

            qv = np.quantile(dev_samples, q=q, axis=1).astype(np.float32)

        # guard
        qv = np.maximum(qv, 1e-8).astype(np.float32)

        return {
            "filename": filename,
            "split": infer_split_from_path(Path(filename)),
            "sfreq": sfreq if sfreq is not None else np.nan,
            "n_channels": C,
            "n_samples": T,
            "channels": channels,
            "mean": mean,
            "q": float(q),
            "qv": qv,
        }


def write_stats_hdf5(out_hdf: Path, rows: List[Dict]) -> None:
    """
    Writes a single HDF5 file with:
      /index   (table-like)
      /mean    float32 [N,C]
      /qv      float32 [N,C]
      /channels (optional ragged via JSON)
    """
    out_hdf.parent.mkdir(parents=True, exist_ok=True)

    # Build index dataframe
    idx_df = pd.DataFrame([{
        "filename": r["filename"],
        "split": r["split"],
        "sfreq": float(r["sfreq"]) if r["sfreq"] is not None else np.nan,
        "n_channels": int(r["n_channels"]),
        "n_samples": int(r["n_samples"]),
        "q": float(r["q"]),
    } for r in rows])

    # Determine C (assume fixed; if not, we still store with padding)
    max_C = int(idx_df["n_channels"].max())
    N = len(rows)

    mean_mat = np.full((N, max_C), np.nan, dtype=np.float32)
    qv_mat   = np.full((N, max_C), np.nan, dtype=np.float32)

    # store channels as JSON strings (per row) for simplicity
    ch_json = []
    for i, r in enumerate(rows):
        c = int(r["n_channels"])
        mean_mat[i, :c] = r["mean"]
        qv_mat[i, :c]   = r["qv"]
        ch = r["channels"]
        if ch is None:
            ch_json.append("")
        else:
            ch_json.append("|".join([str(x) for x in ch]))

    # Write via h5py (fast + simple)
    with h5py.File(str(out_hdf), "w") as h5:
        # index table columns
        dt = h5py.string_dtype("utf-8")
        h5.create_dataset("index/filename", data=idx_df["filename"].astype(str).to_numpy(dtype=object), dtype=dt)
        h5.create_dataset("index/split",    data=idx_df["split"].astype(str).to_numpy(dtype=object), dtype=dt)
        h5.create_dataset("index/sfreq",    data=idx_df["sfreq"].to_numpy(dtype=np.float32))
        h5.create_dataset("index/n_channels", data=idx_df["n_channels"].to_numpy(dtype=np.int32))
        h5.create_dataset("index/n_samples",  data=idx_df["n_samples"].to_numpy(dtype=np.int64))
        h5.create_dataset("index/q",          data=idx_df["q"].to_numpy(dtype=np.float32))

        h5.create_dataset("stats/mean", data=mean_mat, compression="gzip", compression_opts=4)
        h5.create_dataset("stats/qv",   data=qv_mat,   compression="gzip", compression_opts=4)
        h5.create_dataset("stats/channels_joined", data=np.array(ch_json, dtype=object), dtype=dt)

        h5.attrs["schema"] = "tusz_norm_stats_v1"
        h5.attrs["note"] = "Per-file mean and q-quantile(|x-mean|) per channel. Rows align with index/* datasets."


def main():
    ap = argparse.ArgumentParser("Collect normalization stats for TUSZ HDF repository.")
    ap.add_argument("--hdf_root", type=str, required=True,
                    help="Root directory containing train/dev/eval and optionally sessions.csv")
    ap.add_argument("--out_hdf", type=str, required=True,
                    help="Output HDF5 file path (single file with all stats)")
    ap.add_argument("--q", type=float, default=0.95, help="Quantile for robust scale (default 0.95)")
    ap.add_argument("--max_points_per_ch", type=int, default=200_000,
                    help="Max subsampled points per channel for quantile estimation. "
                         "Set <=0 to compute exact (may use lots of RAM).")
    ap.add_argument("--chunk_samples", type=int, default=524_288,
                    help="Chunk length in samples for streaming mean (default ~2^19)")
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional limit number of files for quick test (0 = all)")
    ap.add_argument("--write_csv", action="store_true",
                    help="Also write a CSV summary next to out_hdf")
    args = ap.parse_args()

    hdf_root = Path(args.hdf_root)
    out_hdf = Path(args.out_hdf)

    files_df = load_file_list(hdf_root)
    files = files_df["filename"].astype(str).tolist()
    if args.limit and args.limit > 0:
        files = files[:int(args.limit)]

    # compute
    def _work(i_fn: Tuple[int, str]) -> Optional[Dict]:
        i, fn = i_fn
        try:
            return compute_stats_one_file(
                filename=fn,
                q=float(args.q),
                max_points_per_ch=int(args.max_points_per_ch),
                chunk_samples=int(args.chunk_samples),
                seed=int(args.seed) + i,
            )
        except Exception as e:
            # keep going, but report
            return {"filename": fn, "error": repr(e)}

    results = Parallel(n_jobs=int(args.n_jobs), backend="loky")(
        delayed(_work)(x) for x in tqdm(list(enumerate(files)), desc="Computing stats")
    )

    # separate failures
    ok_rows = [r for r in results if r is not None and "error" not in r]
    bad_rows = [r for r in results if r is not None and "error" in r]

    if not ok_rows:
        raise RuntimeError("No stats computed successfully. First error: " + (bad_rows[0]["error"] if bad_rows else "unknown"))

    write_stats_hdf5(out_hdf, ok_rows)
    print(f"Wrote stats HDF5: {out_hdf}")
    print(f"OK: {len(ok_rows)} files, FAILED: {len(bad_rows)} files")

    if bad_rows:
        bad_csv = out_hdf.with_suffix(".failed.csv")
        pd.DataFrame(bad_rows).to_csv(bad_csv, index=False)
        print(f"Wrote failures CSV: {bad_csv}")

    if args.write_csv:
        # summary CSV (no vectors)
        summary = pd.DataFrame([{
            "filename": r["filename"],
            "split": r["split"],
            "sfreq": float(r["sfreq"]) if r["sfreq"] is not None else np.nan,
            "n_channels": int(r["n_channels"]),
            "n_samples": int(r["n_samples"]),
            "q": float(r["q"]),
            "mean_min": float(np.nanmin(r["mean"])),
            "mean_max": float(np.nanmax(r["mean"])),
            "qv_min": float(np.nanmin(r["qv"])),
            "qv_max": float(np.nanmax(r["qv"])),
        } for r in ok_rows])
        out_csv = out_hdf.with_suffix(".summary.csv")
        summary.to_csv(out_csv, index=False)
        print(f"Wrote summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
