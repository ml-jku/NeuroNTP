#!/usr/bin/env python3
"""
+----------------------------------------------------------------------------------------------------------------------+
run_pipeline.py
Run one or more NeuroNTP sessions (nntp.py) defined in a pipeline configuration file

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Authors: Jozsef Kovacs (jozsef.kovacs@jku.at), Philipp Seidl (philipp.seidl@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Reads a pipeline configuration (.yaml) and runs its stages one after another, each as a separate nntp.py session.
    Every stage is translated into a plain nntp.py command line, which is printed before it runs; a failing stage
    stops the pipeline.

    Pipeline configuration:
        vars      named values, referenced as ${name} anywhere in the configuration; resolved in order, so later
                  entries may reference earlier ones. ${run_id} (timestamp + random suffix) is always defined.
        common    nntp.py arguments shared by all stages
        downloads optional mapping of local file paths to URLs; missing files are downloaded before the first stage
        stages    list of stages, each with a name, a mode and its own arguments (overriding common ones)

    Arguments are mapped to the command line as: key: value -> --key=value, key: true -> --key, key: false or
    key: null -> omitted (also removes an argument set in common).

    After a stage in pretrain, linprobe or finetune mode, its checkpoint is looked up and made available to later
    stages, and its directory as ${<stage>.ckpt_dir}:
        linprobe, finetune  ${<stage>.best_ckpt}, the best checkpoint by the monitored metric, searched as
                            <save_path>/checkpoints/<run_name>*/best*.ckpt (fine-tuning) or as
                            <save_path>/logs/<run_name>*/**/checkpoints/*.ckpt (linear probing; last*.ckpt ignored)
        pretrain            ${<stage>.last_ckpt}, the checkpoint at the end of the training schedule, searched as
                            <save_path>/logs/<run_name>*/**/checkpoints/last*.ckpt or, if not present, as the
                            latest scheduled dump <save_path>/checkpoints/<run_name>*/epoch_<N>.ckpt (--dump_every)
    A checkpoint is only made available if the match is unique (one file, or one run directory for the dumps).

Usage:
    python run_pipeline.py configs/pipelines/chb_finetune_biot.yaml --device 2
    python run_pipeline.py configs/pipelines/chb_finetune_biot.yaml --set pretrained_ckpt=models/pretrain/x.ckpt
    python run_pipeline.py configs/pipelines/tusz_pretrain.yaml --dry_run

    --device sets CUDA_VISIBLE_DEVICES for all stages (the stages then use device 0), --set overrides entries of
    vars, --dry_run only prints the commands. Relative paths are resolved from the repository root.
+----------------------------------------------------------------------------------------------------------------------+
"""
from __future__ import annotations

import argparse
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent
NNTP_MAIN = REPO_ROOT / "src" / "neuro_ntp" / "nntp.py"
CKPT_MODES = {"pretrain", "linprobe", "finetune"}

_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def substitute(value: Any, ctx: Dict[str, Any]) -> Any:
    """Replace ${name} references in strings (also inside lists and dicts) by the values in ctx."""
    if isinstance(value, str):
        def _repl(m: re.Match) -> str:
            key = m.group(1)
            if key not in ctx:
                raise KeyError(key)
            return str(ctx[key])
        return _VAR_RE.sub(_repl, value)
    if isinstance(value, list):
        return [substitute(v, ctx) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, ctx) for k, v in value.items()}
    return value


def to_cli(args: Dict[str, Any]) -> List[str]:
    """Translate an argument dictionary into nntp.py command line arguments."""
    out: List[str] = []
    for key, value in args.items():
        if value is None or value is False:
            continue
        if value is True:
            out.append(f"--{key}")
        else:
            out.append(f"--{key}={value}")
    return out


BEST_CKPT_PATTERNS = [
    "checkpoints/{run_name}*/best*.ckpt",           # fine-tuning (best model by the monitored metric)
    "logs/{run_name}*/**/checkpoints/*.ckpt",      # pretraining and linear probing (model selection checkpoint)
]


def find_best_ckpt(save_path: str, run_name: str) -> Optional[Path]:
    """
    Return the best checkpoint of a run: the first pattern of BEST_CKPT_PATTERNS that matches exactly one file,
    ignoring last*.ckpt (checkpoint of the final epoch, not of the model selection). Prints the candidates if the
    match is not unique, or the searched patterns if nothing matches.
    """
    root = REPO_ROOT / save_path
    for pattern in BEST_CKPT_PATTERNS:
        glob_pattern = pattern.format(run_name=run_name)
        found = sorted(p for p in root.glob(glob_pattern) if p.is_file() and not p.name.startswith("last"))
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            print(f"several checkpoints match {root / glob_pattern}, none selected:")
            for p in found:
                print(f"  {p}")
            return None
    print(f"no checkpoint found for run '{run_name}', searched:")
    for pattern in BEST_CKPT_PATTERNS:
        print(f"  {root / pattern.format(run_name=run_name)}")
    return None


LAST_CKPT_PATTERN = "logs/{run_name}*/**/checkpoints/last*.ckpt"    # Lightning checkpoint of the final epoch
DUMP_DIR_PATTERN = "checkpoints/{run_name}*"                         # scheduled dumps epoch_<N>.ckpt (--dump_every)
_DUMP_RE = re.compile(r"epoch_(\d+)\.ckpt$")


def find_last_ckpt(save_path: str, run_name: str) -> Optional[Path]:
    """
    Return the checkpoint at the end of the training schedule of a run: the Lightning last*.ckpt if exactly one
    matches, otherwise the scheduled dump with the highest epoch number, if exactly one run directory has dumps.
    Prints the candidates if the match is not unique, or the searched patterns if nothing matches.
    """
    root = REPO_ROOT / save_path
    last_pattern = LAST_CKPT_PATTERN.format(run_name=run_name)
    found = sorted(p for p in root.glob(last_pattern) if p.is_file())
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        print(f"several checkpoints match {root / last_pattern}, none selected:")
        for p in found:
            print(f"  {p}")
        return None

    dump_pattern = DUMP_DIR_PATTERN.format(run_name=run_name)
    dump_dirs = sorted(d for d in root.glob(dump_pattern)
                       if d.is_dir() and any(_DUMP_RE.search(p.name) for p in d.iterdir()))
    if len(dump_dirs) == 1:
        dumps = [p for p in dump_dirs[0].iterdir() if _DUMP_RE.search(p.name)]
        return max(dumps, key=lambda p: int(_DUMP_RE.search(p.name).group(1)))
    if len(dump_dirs) > 1:
        print(f"several run directories with scheduled dumps match {root / dump_pattern}, none selected:")
        for d in dump_dirs:
            print(f"  {d}")
        return None
    print(f"no checkpoint found for run '{run_name}', searched:\n  {root / last_pattern}\n  "
          f"{root / dump_pattern}/epoch_<N>.ckpt")
    return None


def resolve_vars(raw_vars: Dict[str, Any], overrides: Dict[str, str], run_id: str) -> Dict[str, Any]:
    """Resolve the vars section in order, after applying command line overrides."""
    ctx: Dict[str, Any] = {"run_id": run_id}
    merged = dict(raw_vars)
    merged.update(overrides)
    for key, value in merged.items():
        ctx[key] = substitute(value, ctx)
    return ctx


def download_missing(downloads: Dict[str, str], dry_run: bool) -> None:
    """Download the files of the downloads section that are missing locally (paths relative to the repository root)."""
    for rel_path, url in downloads.items():
        path = REPO_ROOT / rel_path
        if path.exists():
            continue
        print(f"download: {url}\n       -> {path}", flush=True)
        if dry_run:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        with urllib.request.urlopen(url) as resp:
            if resp.headers.get("Content-Type", "").startswith("text/html"):
                sys.exit(f"download of {rel_path} returned a web page instead of a file, check the URL: {url}")
            size = resp.headers.get("Content-Length")
            if size:
                print(f"       {int(size) / 2**20:.1f} MB", flush=True)
            with open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
        tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description="Run a NeuroNTP pipeline configuration.")
    ap.add_argument("config", type=str, help="pipeline configuration (.yaml)")
    ap.add_argument("--device", type=str, default=None, help="GPU ordinal, sets CUDA_VISIBLE_DEVICES")
    ap.add_argument("--set", type=str, action="append", default=[], metavar="KEY=VALUE",
                    help="override an entry of vars (repeatable)")
    ap.add_argument("--dry_run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    overrides = dict(s.split("=", 1) for s in args.set)
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(4)}"
    try:
        ctx = resolve_vars(cfg.get("vars") or {}, overrides, run_id)
    except KeyError as e:
        sys.exit(f"vars: undefined variable ${{{e.args[0]}}}")

    env = os.environ.copy()
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.device
        print(f"CUDA_VISIBLE_DEVICES={args.device}")
    print(f"run_id={run_id}")

    try:
        downloads = substitute(cfg.get("downloads") or {}, ctx)
    except KeyError as e:
        sys.exit(f"downloads: undefined variable ${{{e.args[0]}}}")
    download_missing(downloads, args.dry_run)

    common = cfg.get("common") or {}
    for stage in cfg["stages"]:
        name = stage["name"]
        stage_args = {"mode": stage["mode"], **common, **(stage.get("args") or {})}
        try:
            resolved = substitute(stage_args, ctx)
        except KeyError as e:
            sys.exit(f"stage '{name}': undefined variable ${{{e.args[0]}}} "
                     f"(stage checkpoint variables are only defined if the stage saved its checkpoints)")

        cmd = [sys.executable, str(NNTP_MAIN), *to_cli(resolved)]
        print(f"\n=== stage: {name} ({resolved['mode']}) ===")
        print(shlex.join(cmd), flush=True)

        ckpt_kind = "last" if resolved["mode"] == "pretrain" else "best"
        if args.dry_run:
            ctx[f"{name}.ckpt_dir"] = f"<{name}.ckpt_dir>"
            ctx[f"{name}.{ckpt_kind}_ckpt"] = f"<{name}.{ckpt_kind}_ckpt>"
            continue

        rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode
        if rc != 0:
            sys.exit(f"stage '{name}' failed (exit code {rc}), pipeline stopped")

        if resolved["mode"] in CKPT_MODES and resolved.get("save_path") and resolved.get("run_name"):
            find = find_last_ckpt if ckpt_kind == "last" else find_best_ckpt
            ckpt = find(str(resolved["save_path"]), str(resolved["run_name"]))
            if ckpt is not None:
                ctx[f"{name}.{ckpt_kind}_ckpt"] = str(ckpt)
                ctx[f"{name}.ckpt_dir"] = str(ckpt.parent)
                print(f"stage '{name}': {ckpt_kind} checkpoint {ckpt}")

    print("\npipeline finished")


if __name__ == "__main__":
    main()