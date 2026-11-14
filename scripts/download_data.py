#!/usr/bin/env python3
"""
+----------------------------------------------------------------------------------------------------------------------+
download_data.py
Download and verify the raw EEG datasets used by NeuroNTP (CHB-MIT, Siena, TUSZ)

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Downloads the raw data and verifies every file against a SHA-256 checksum list, so that an incomplete or
    corrupted download is detected before preprocessing (an incomplete download otherwise surfaces only later, as
    missing recordings in the core dataset or the dataset views).

    CHB-MIT, Siena (PhysioNet, open access)
        The checksum list is PhysioNet's SHA256SUMS.txt of the dataset version. Files are downloaded in parallel
        with resume and retries; a file is only moved into place after its checksum has been verified. Files that are
        already present and correct are skipped, so the script can simply be rerun after an interruption.

    TUSZ (TUH EEG Seizure Corpus, registration required)
        TUSZ is not redistributable and has no published checksum list. It is downloaded with rsync using the
        password provided by NEDC upon registration (https://isip.piconepress.com/projects/nedc/html/tuh_eeg/), and
        verified against the checksum manifest shipped in this repository (res/manifests/tusz_v2.0.3.sha256), which
        lists the files of the copy of TUSZ v2.0.3 used in the paper.
        - The paper uses v2.0.3. Later versions differ substantially (v2.0.5 re-annotated the corpus and limited
          recordings to 1800 s), so do not substitute a newer version.
        - NEDC has patched v2.0.3 in place (header, corrupted-file and annotation fixes, last change 2025-06-18).
          Files whose checksum differs from the manifest are therefore reported as "changed" (warning), and only
          missing files count as an error; use --tusz_strict to treat changed files as errors as well.
        - The rsync source changes occasionally on NEDC's side. If the default (TUSZ_RSYNC_SRC below) fails, take the
          current address from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml or the NEDC access
          e-mail and pass it with --tusz_src.

Output layout:
    <out_root>/chbmit/1.0.0/chb01/chb01_01.edf, ...       (input of chb2hdf.py:  <out_root>/chbmit/1.0.0)
    <out_root>/siena/1.0.0/PN00/PN00-1.edf, ...           (input of siena2hdf.py: <out_root>/siena/1.0.0)
    <out_root>/tusz/v2.0.3/edf/{train,dev,eval}/...       (input of tusz2hdf.py:  <out_root>/tusz/v2.0.3/edf)
    This is the same layout as a wget download following the PhysioNet instructions (wget -r -np -nH --cut-dirs=2).

Usage:
    python scripts/download_data.py --datasets chbmit siena --out_root datasets/raw
    python scripts/download_data.py --datasets tusz --out_root datasets/raw [--tusz_src <user>@<host>:<path>/v2.0.3/]
    python scripts/download_data.py --datasets all --out_root datasets/raw --verify_only

    Maintainers (create the TUSZ manifest from a verified local copy of the corpus):
    python scripts/download_data.py --make_manifest /data/tusz/v2.0.x --manifest_out res/manifests/tusz_v2.0.x.sha256

Exit status:
    0 if all requested datasets are complete and verified, 1 otherwise (a summary of missing and corrupted files
    is printed, and written to <out_root>/<dataset>/<version>/VERIFY_FAILED.txt).
+----------------------------------------------------------------------------------------------------------------------+
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "res" / "manifests"

# TUSZ corpus version used in the paper; the manifest res/manifests/tusz_<version>.sha256 must exist for it
TUSZ_VERSION = "v2.0.3"
# rsync source following NEDC's download page (user "nedc-eeg", password by e-mail after registration).
# UPDATE HERE (or pass --tusz_src) if NEDC changes the address; list the available corpora with
#   rsync nedc-eeg@www.isip.piconepress.com:data/eeg/
TUSZ_RSYNC_SRC = "nedc-eeg@www.isip.piconepress.com:data/eeg/tuh_eeg_seizure/{version}/"


@dataclass(frozen=True)
class PhysioNetDataset:
    name: str
    slug: str
    version: str

    @property
    def base_url(self) -> str:
        return f"https://physionet.org/files/{self.slug}/{self.version}/"


PHYSIONET = {
    "chbmit": PhysioNetDataset("chbmit", "chbmit", "1.0.0"),
    "siena": PhysioNetDataset("siena", "siena-scalp-eeg", "1.0.0"),
}
ALL_DATASETS = ["chbmit", "siena", "tusz"]

CHUNK = 1 << 20
USER_AGENT = "NeuroNTP-download/1.0 (+https://github.com/ml-jku/NeuroNTP)"


# ---------------------------------------------------------------------------------------------------------------------
# checksums
# ---------------------------------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def parse_sums(text: str) -> Dict[str, str]:
    """Parse a sha256sum-style list ('<hash>  <relative path>' per line) into {relative path: hash}."""
    sums = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*").strip()
        if rel.startswith("./"):
            rel = rel[2:]
        sums[rel] = digest.lower()
    return sums


def verify(root: Path, sums: Dict[str, str], jobs: int, label: str) -> Tuple[List[str], List[str]]:
    """Verify all files listed in sums below root. Returns (missing, corrupted) relative paths."""
    missing, to_hash = [], []
    for rel in sums:
        (to_hash if (root / rel).is_file() else missing).append(rel)

    corrupted = []
    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(sha256_file, root / rel): rel for rel in to_hash}
        for fut in as_completed(futs):
            rel = futs[fut]
            if fut.result() != sums[rel]:
                corrupted.append(rel)
            done += 1
            _progress(f"[{label}] verifying", done, len(to_hash))
    _progress_end()
    return sorted(missing), sorted(corrupted)


def report(root: Path, label: str, n_total: int, missing: List[str], corrupted: List[str]) -> bool:
    fail_file = root / "VERIFY_FAILED.txt"
    if not missing and not corrupted:
        print(f"[{label}] OK: all {n_total} files present and verified ({root})")
        if fail_file.exists():
            fail_file.unlink()
        return True
    print(f"[{label}] FAILED: {len(missing)} missing, {len(corrupted)} corrupted (of {n_total} files)")
    for tag, items in (("missing", missing), ("corrupted", corrupted)):
        for rel in items[:20]:
            print(f"    {tag}: {rel}")
        if len(items) > 20:
            print(f"    ... and {len(items) - 20} more {tag}")
    root.mkdir(parents=True, exist_ok=True)
    fail_file.write_text("".join(f"missing\t{r}\n" for r in missing) + "".join(f"corrupted\t{r}\n" for r in corrupted))
    print(f"[{label}] full list: {fail_file}")
    return False


# ---------------------------------------------------------------------------------------------------------------------
# PhysioNet (HTTPS, parallel, resumable, verified)
# ---------------------------------------------------------------------------------------------------------------------
def _request(url: str, offset: int = 0, timeout: float = 60.0):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_text(url: str, retries: int) -> str:
    for attempt in range(1, retries + 1):
        try:
            with _request(url) as r:
                return r.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                raise RuntimeError(f"could not fetch {url}: {e}") from e
            time.sleep(min(2 ** attempt, 30))
    raise AssertionError("unreachable")


def download_file(url: str, dst: Path, digest: str, retries: int) -> Optional[str]:
    """
    Download url to dst via dst.part (resuming a partial download), verify the checksum and move it into place.
    Returns None on success, an error message otherwise.
    """
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            offset = part.stat().st_size if part.exists() else 0
            with _request(url, offset) as r:
                if offset and r.status != 206:      # server ignored the range request: start over
                    offset = 0
                with open(part, "ab" if offset else "wb") as f:
                    shutil.copyfileobj(r, f, CHUNK)
            if sha256_file(part) == digest:
                os.replace(part, dst)
                return None
            last_err = "checksum mismatch"
            part.unlink()                           # corrupted: do not resume from it
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code == 416:                       # range not satisfiable: stale .part file
                part.unlink(missing_ok=True)
            elif e.code in (403, 404):
                return last_err
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = str(e)
        time.sleep(min(2 ** attempt, 30))
    return last_err


def run_physionet(ds: PhysioNetDataset, out_root: Path, jobs: int, retries: int, verify_only: bool) -> bool:
    root = out_root / ds.name / ds.version
    root.mkdir(parents=True, exist_ok=True)
    sums_path = root / "SHA256SUMS.txt"
    if not sums_path.exists() or not verify_only:
        sums_path.write_text(fetch_text(ds.base_url + "SHA256SUMS.txt", retries))
    sums = parse_sums(sums_path.read_text())
    sums.pop("SHA256SUMS.txt", None)

    missing, corrupted = verify(root, sums, jobs, ds.name)
    if verify_only:
        return report(root, ds.name, len(sums), missing, corrupted)

    todo = missing + corrupted
    errors: Dict[str, str] = {}
    if todo:
        size_note = "" if len(todo) == len(sums) else f" (of {len(sums)}; the rest is already verified)"
        print(f"[{ds.name}] downloading {len(todo)} files{size_note} from {ds.base_url}")
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(download_file, ds.base_url + rel, root / rel, sums[rel], retries): rel for rel in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                err = fut.result()
                if err:
                    errors[futs[fut]] = err
                _progress(f"[{ds.name}] downloading", i, len(todo))
        _progress_end()
        for rel, err in sorted(errors.items())[:20]:
            print(f"    failed: {rel} ({err})")

    # files downloaded in this run were verified before being moved into place, the rest in the first pass;
    # only the failed downloads remain (missing, or still the corrupted file from before)
    missing = sorted(rel for rel in errors if not (root / rel).is_file())
    corrupted = sorted(rel for rel in errors if (root / rel).is_file())
    return report(root, ds.name, len(sums), missing, corrupted)


# ---------------------------------------------------------------------------------------------------------------------
# TUSZ (rsync, verified against the repository manifest)
# ---------------------------------------------------------------------------------------------------------------------
def tusz_manifest(version: str) -> Path:
    return MANIFEST_DIR / f"tusz_{version}.sha256"


def run_tusz(out_root: Path, version: str, src: Optional[str], jobs: int, verify_only: bool, strict: bool) -> bool:
    root = out_root / "tusz" / version
    manifest = tusz_manifest(version)
    if not manifest.exists():
        print(f"[tusz] no manifest for TUSZ {version} ({manifest}); available: "
              f"{sorted(p.stem.removeprefix('tusz_') for p in MANIFEST_DIR.glob('tusz_*.sha256')) or 'none'}")
        return False

    if not verify_only:
        if shutil.which("rsync") is None:
            print("[tusz] rsync not found; please install rsync")
            return False
        root.mkdir(parents=True, exist_ok=True)
        src = src or TUSZ_RSYNC_SRC.format(version=version)
        if not src.endswith("/"):
            src += "/"                              # copy the contents of the version directory into <out_root>/tusz/<version>
        cmd = ["rsync", "-auxvL", "--partial", src, str(root) + "/"]
        print(f"[tusz] {' '.join(cmd)}\n[tusz] (rsync will ask for the NEDC password)")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[tusz] rsync exited with status {rc}; verifying what is present. If the connection or the path "
                  f"failed, see the note on TUSZ_RSYNC_SRC / --tusz_src.")

    sums = parse_sums(manifest.read_text())
    missing, changed = verify(root, sums, jobs, "tusz")
    changed_file = root / "CHANGED_FILES.txt"
    if changed_file.exists():
        changed_file.unlink()                       # rewritten below if files still differ
    if changed and not strict:
        print(f"[tusz] WARNING: {len(changed)} files differ from the copy used in the paper (NEDC has updated v2.0.3 "
              f"in place); results may deviate slightly. Changed files:")
        for rel in changed[:20]:
            print(f"    changed: {rel}")
        if len(changed) > 20:
            print(f"    ... and {len(changed) - 20} more")
        root.mkdir(parents=True, exist_ok=True)
        changed_file.write_text("".join(f"{r}\n" for r in changed))
        changed = []
    return report(root, f"tusz {version}", len(sums), missing, changed)


def make_manifest(src: Path, out: Path, jobs: int, patterns: List[str]) -> None:
    """Write a sha256sum-style manifest of all files below src matching patterns (relative paths, sorted)."""
    files = sorted({p for pat in patterns for p in src.rglob(pat) if p.is_file()})
    digests = {}
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(sha256_file, p): p for p in files}
        for i, fut in enumerate(as_completed(futs), 1):
            digests[futs[fut]] = fut.result()
            _progress("hashing", i, len(files))
    _progress_end()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# manifest of {src.name}: {len(files)} files, patterns {patterns}\n")
        for p in files:
            f.write(f"{digests[p]}  {p.relative_to(src).as_posix()}\n")
    print(f"wrote {out} ({len(files)} files)")


# ---------------------------------------------------------------------------------------------------------------------
def _progress(label: str, i: int, n: int) -> None:
    if sys.stdout.isatty():
        print(f"\r{label}: {i}/{n}", end="", flush=True)


def _progress_end() -> None:
    if sys.stdout.isatty():
        print("\r\033[K", end="", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download and verify the raw datasets used by NeuroNTP.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=ALL_DATASETS + ["all"], help="datasets to download/verify")
    ap.add_argument("--out_root", type=str, default="datasets/raw", help="root directory for the raw data")
    ap.add_argument("--verify_only", action="store_true", help="only verify existing files, do not download")
    ap.add_argument("--jobs", type=int, default=8, help="parallel downloads / checksum workers")
    ap.add_argument("--retries", type=int, default=5, help="attempts per file")
    ap.add_argument("--tusz_version", type=str, default=TUSZ_VERSION, help="TUSZ corpus version")
    ap.add_argument("--tusz_src", type=str, default=None,
                    help=f"rsync source of TUSZ (default: {TUSZ_RSYNC_SRC.format(version='<tusz_version>')})")
    ap.add_argument("--tusz_strict", action="store_true",
                    help="treat TUSZ files that differ from the manifest as errors (default: warning)")
    ap.add_argument("--make_manifest", type=str, default=None, metavar="DIR",
                    help="(maintainers) write a checksum manifest of DIR instead of downloading")
    ap.add_argument("--manifest_out", type=str, default=None, help="output path for --make_manifest")
    ap.add_argument("--manifest_patterns", nargs="+", default=["*.edf", "*.csv", "*.csv_bi"],
                    help="file patterns included by --make_manifest")
    args = ap.parse_args()

    if args.make_manifest:
        if not args.manifest_out:
            ap.error("--make_manifest requires --manifest_out")
        make_manifest(Path(args.make_manifest), Path(args.manifest_out), args.jobs, args.manifest_patterns)
        return 0
    if not args.datasets:
        ap.error("--datasets is required")

    datasets = ALL_DATASETS if "all" in args.datasets else list(dict.fromkeys(args.datasets))
    out_root = Path(args.out_root)
    ok = {}
    for name in datasets:
        if name in PHYSIONET:
            ok[name] = run_physionet(PHYSIONET[name], out_root, args.jobs, args.retries, args.verify_only)
        else:
            ok[name] = run_tusz(out_root, args.tusz_version, args.tusz_src, args.jobs, args.verify_only,
                                args.tusz_strict)

    print("\nSummary: " + ", ".join(f"{k}: {'OK' if v else 'INCOMPLETE'}" for k, v in ok.items()))
    if not all(ok.values()):
        print("Rerun the same command to fetch the remaining files; do not start preprocessing before all datasets "
              "report OK.")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())