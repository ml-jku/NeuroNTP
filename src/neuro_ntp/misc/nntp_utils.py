"""
+----------------------------------------------------------------------------------------------------------------------+
nntp_utils.py
Helper functions for NeuroNTP

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Utilities used across training and evaluation: dynamic class loading from configuration paths, seeding,
    checkpoint restoration for transfer between dataset configurations, EEG channel geometry (electrode positions,
    nearest-channel substitution), time-to-seizure encoding, module freezing, parameter inspection, and the set of
    modalities enabled by a model configuration.
+----------------------------------------------------------------------------------------------------------------------+
"""
import importlib
import numpy as np
import torch
import torch.nn as nn
import mne
import math


import logging
logger = logging.getLogger(__name__)

_MONTAGE_CACHE = {}

def load_class(path: str):
    module_path, _, class_name = path.replace(":", ".").rpartition(".")
    if not module_path:
        raise ValueError(f"Expected fully-qualified path like 'pkg.mod.Class', got: {path}")
    mod = importlib.import_module(module_path)
    try:
        return getattr(mod, class_name)
    except AttributeError as e:
        raise ImportError(f"{class_name} not found in {module_path}") from e


def instantiate(path: str, /, *args, **kwargs):
    cls = load_class(path)
    return cls(*args, **kwargs)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_trainable_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def restore_from_ckpt(ckpt_path, pretraining_datamodule,
                      lightning_model,
                      verbose=False):
    logger.info(f"Restoring pre-trained model from {ckpt_path}...")
    if pretraining_datamodule is not None:
        pretraining_datamodule.setup("fit")
        lightning_model.enc_warmup("fit", pretraining_datamodule)
    restore_report = init_from_ckpt(lightning_model, ckpt_path)
    lightning_model.set_runtime_flags()
    if verbose:
        for key_type in ['loaded', 'restored', 'skipped']:
            logger.info(f"...{key_type}:", restore_report[key_type])

    # Count trainable params (rank-0 only if under DDP)
    total = count_trainable_params(lightning_model)
    logger.info(
        f"Trainable parameters after restore & flags: {total} "
        f"({total:,} params)"
    )

    logger.info(f"Finished restoring pre-trained model from {ckpt_path}...")


def init_from_ckpt(lightning_model, ckpt_path, *, map_location="cpu",
                   load_non_registry_weights=True, strict_shapes=True,
                   skip_registry_prefix="model.mm_tokenizer.encoders._families."):
    """
    Weights-only initialization for transfer between dataset configurations (different channels, sensor inputs).
    This custom implementation is necessary to endure the model weights are restored in channels-config-sensitive way.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # import registry (encoder families)
    enc = None
    inner = getattr(lightning_model, "model", lightning_model)
    if hasattr(inner, "encoders"):
        enc = inner.encoders
    elif hasattr(inner, "mm_tokenizer") and hasattr(inner.mm_tokenizer, "encoders"):
        enc = inner.mm_tokenizer.encoders
    if enc is not None:
        nested = ckpt.get("mmt_registry_states", {})
        if nested:
            enc.import_states(nested, merge=True)   # create missing ids, overwrite overlaps

    # apply model meta
    meta = ckpt.get("meta", {}).get("neuroNTP", {})
    if meta and hasattr(inner, "import_meta"):
        inner.import_meta(meta)

    # load remaining model weights (skip registry subtree, match by shape)
    report = {"loaded": [], "skipped": [], "missing": []}
    if load_non_registry_weights:
        logger.debug(">>> loading non-registry weights")
        full_sd = ckpt.get("state_dict", {})
        filtered_sd = {k: v for k, v in full_sd.items()
                       if not (skip_registry_prefix and k.startswith(skip_registry_prefix))}
        curr = lightning_model.state_dict()
        if strict_shapes:
            compat = {k: v for k, v in filtered_sd.items()
                      if k in curr and curr[k].shape == v.shape}
        else:
            compat = {k: v for k, v in filtered_sd.items() if k in curr}
        curr.update(compat)
        lightning_model.load_state_dict(curr, strict=False)
        report["loaded"] = list(compat.keys())
        report["skipped"] = [k for k in filtered_sd.keys() if k not in compat]
        # keys present in current model but absent in ckpt (fine—random init)
        report["missing"] = [k for k in curr.keys() if k not in full_sd]

    return report


def _get_montage(montage_name="standard_1005"):
    if montage_name not in _MONTAGE_CACHE:
        _MONTAGE_CACHE[montage_name] = mne.channels.make_standard_montage(montage_name)
    return _MONTAGE_CACHE[montage_name]


def _coord_for_label(label: str, montage_name: str = "standard_1005"):
    """
    Return 3D coord for:
      - single channel: 'FP1'
      - bipolar: 'FP1-F7'
      - weird CHB: 'T8-P8-0' -> treat as 'T8-P8'
    If both ends missing -> None.
    """
    montage = _get_montage(montage_name)
    ch_pos = montage.get_positions()["ch_pos"]

    def _single(name: str):
        key = name.upper()
        return np.array(ch_pos[key], dtype=float) if key in ch_pos else None

    # strip leading 'EEG ' if present
    label = label.strip()
    if label.upper().startswith("EEG "):
        label = label[4:]

    # CHB sometimes has 'T8-P8-0' → use first two parts
    parts = label.split("-")
    if len(parts) == 1:
        return _single(parts[0])

    # bipolar: take first two tokens only
    a, b = parts[0], parts[1]
    ca = _single(a)
    cb = _single(b)

    if ca is None and cb is None:
        return None
    if ca is None:
        return cb
    if cb is None:
        return ca
    return (ca + cb) / 2.0  # midpoint


def distance(chan_x: str, chan_y: str, montage_name: str = "standard_1005",
             head_radius_m: float = None):
    """
    Geodesic-ish distance between two (possibly bipolar) channel labels.
    Returns None if either can't be located.
    """
    p = _coord_for_label(chan_x, montage_name)
    q = _coord_for_label(chan_y, montage_name)
    if p is None or q is None:
        return None

    # normalize
    pn = p / np.linalg.norm(p)
    qn = q / np.linalg.norm(q)
    cosang = np.clip(np.dot(pn, qn), -1.0, 1.0)
    central_angle = float(np.arccos(cosang))
    if head_radius_m is None:
        r = 0.5 * (np.linalg.norm(p) + np.linalg.norm(q))
    else:
        r = head_radius_m
    return r * central_angle


def _norm_name(ch: str) -> str:
    """Lowercase, strip 'eeg ' prefix & spaces; map legacy aliases to 10-05/10-20."""
    s = ch.strip().lower()
    if s.startswith("eeg "):
        s = s[4:]
    s = s.replace(" ", "")
    # legacy alias map (common CHB variants)
    alias = {
        "t3": "t7", "t4": "t8",
        "t5": "p7", "t6": "p8",
        # sometimes seen variants
        "fpz": "fz",
    }
    return alias.get(s, s)

def _only_eeg_single(available):
    """return set of normalized EEG singles (no '-') from available keys."""
    out = set()
    for k in available:
        s = _norm_name(k)
        if "-" not in s and s.startswith(("f","fp","fz","t","c","p","o","a","m","z")):
            out.add(s)
    return out

def find_nearest_substitute(target: str, available, montage_name: str = "standard_1005"):
    """
    target: e.g. 'eeg FP1' or 'eeg FP1-F7'
    available: iterable of keys like 'eeg CZ', 'eeg FP1', ...
    returns: channel name WITHOUT the 'eeg ' prefix, lowercased (single-ended)
    """
    if not available:
        return "eeg cz"  # ultimate fallback

    tgt = _norm_name(target)
    avail = _only_eeg_single(available)
    if not avail:
        # fall back to any normalized names we can get
        avail = {_norm_name(a) for a in available}

    # if the target itself is a single and exists, return it
    if "-" not in tgt and tgt in avail:
        return tgt

    # helper to get the nearest in 'avail' to a single-ended target name
    def _nearest_single(single_name: str) -> str:
        if single_name in avail:
            return single_name
        closest_dist = float("inf")
        closest = None
        for ch in avail:
            d = distance(single_name, ch, montage_name=montage_name)
            if d is not None and d < closest_dist:
                closest_dist = d
                closest = ch
        if closest is None:
            # deterministic fallback
            return "cz" if "cz" in avail else sorted(list(avail))[0]
        return closest

    # bipolar target handling
    if "-" in tgt:
        a, b = tgt.split("-", 1)
        a = a.strip()
        b = b.strip()
        na = _nearest_single(a)
        nb = _nearest_single(b)
        # choose the one that is *actually* nearer to its own endpoint
        da = distance(a, na, montage_name=montage_name)
        db = distance(b, nb, montage_name=montage_name)
        # If distance() returns None, treat as inf
        da = float("inf") if da is None else da
        db = float("inf") if db is None else db
        return na if da <= db else nb

    # Single-ended but not in avail → nearest
    return _nearest_single(tgt)


def nanmean(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if len(xs) else float("nan")


def _count_params(mod: nn.Module, recurse: bool):
    total = sum(p.numel() for p in mod.parameters(recurse=recurse))
    train = sum(p.numel() for p in mod.parameters(recurse=recurse) if p.requires_grad)
    return train, total


def print_param_tree(root: nn.Module, include_children_totals=True, max_name=60):
    """
    prints a tree with per-module parameter counts.
    - include_children_totals=True: shows params INCLUDING children (useful to spot big subtrees)
      False --> only shows parameters defined directly on that module (no children)
    """
    # precompute per-module totals including children
    inc_cache = {}
    if include_children_totals:
        for name, mod in root.named_modules():
            inc_cache[mod] = _count_params(mod, recurse=True)

    def line(prefix, name, train, total):
        pct = 0.0 if total == 0 else 100.0 * train / total
        n = (prefix + name)[:max_name]
        logger.debug(f"{n:<{max_name}}  train={train:>12,}  total={total:>12,}  ({pct:5.1f}% trainable)")

    logger.debug("\n=== Parameter tree ===")
    # show root totals
    t_train, t_total = _count_params(root, recurse=True)
    line("", "[ROOT]", t_train, t_total)

    # per-node (only direct params) and subtree totals
    for full_name, mod in root.named_modules():
        if full_name == "":  # skip root here (already printed)
            continue
        depth = full_name.count(".")
        prefix = "  " * depth + "└─ "
        if include_children_totals:
            train, total = inc_cache[mod]
        else:
            train, total = _count_params(mod, recurse=False)
        line(prefix, full_name.split(".")[-1], train, total)
    logger.debug("=== end ===\n")


def print_top_modules_by_params(root: nn.Module, topk=20):
    rows = []
    for name, mod in root.named_modules():
        train, total = _count_params(mod, recurse=True)
        rows.append((total, train, name or "[ROOT]"))
    rows.sort(reverse=True)
    logger.info(f"\n=== Top {topk} modules by total params ===")
    for total, train, name in rows[:topk]:
        pct = 0.0 if total == 0 else 100.0 * train / total
        logger.info(f"{name:<40} total={total:>12,}  train={train:>12,}  ({pct:5.1f}%)")
    logger.info("=== end ===\n")


def find_unregistered_tensors(mod: nn.Module):
    """Find plain tensors sitting on attributes (not in state_dict)."""
    registered = set(n for n, _ in mod.named_parameters()) | set(n for n, _ in mod.named_buffers())
    offenders = []
    for k, v in mod.__dict__.items():
        if isinstance(v, torch.Tensor) and k not in registered:
            offenders.append((k, tuple(v.shape), v.requires_grad))
    if offenders:
        logger.debug("\n[warn] Unregistered tensors on module:")
        for k, shape, req in offenders:
            logger.debug(f" - {k}: shape={shape}, requires_grad={req}")


def decode_logspace(z, seiz_cap=300.0):
    return torch.expm1(z) - seiz_cap


def decode_tts_seconds(pred_enc, cap=10800.0, softness=1000.0, seiz_cap=300.0):
    tts_capped = torch.expm1(pred_enc) - seiz_cap
    # clamp because numerically pred_enc might imply > cap etc.
    tts_capped = torch.minimum(tts_capped, torch.tensor(cap - 1e-4, device=tts_capped.device, dtype=tts_capped.dtype))
    tts_raw = soft_cap_inverse(tts_capped, cap=cap, softness=softness)
    return tts_raw


def encode_tts_seconds(tts_raw, cap=10800.0, softness=1000.0, seiz_cap=300.0):
    tts_capped = cap - softness * F.softplus((cap - tts_raw) / softness)
    tts_capped = torch.clamp(tts_capped + seiz_cap, min=1.0)
    return torch.log1p(tts_capped)


def soft_cap_inverse(y, cap, softness=1000.0, eps=1e-8):
    # y is the soft-capped value (<= cap)
    a = (cap - y) / softness                     # a >= 0
    u = torch.log(torch.clamp(torch.expm1(a), min=eps))
    x = cap - softness * u
    return x


def set_module_frozen(module: nn.Module, freeze: bool, *, verbose: bool = True):
    if module is None:
        return
    if not isinstance(module, nn.Module):
        raise TypeError(f"set_frozen expected nn.Module, got {type(module)}")

    freeze = bool(freeze)
    setattr(module, "frozen", freeze)

    for p in module.parameters():
        p.requires_grad = not freeze

    if verbose:
        logger.info(f"{'Freezing' if freeze else 'Unfreezing'} {module.__class__.__name__}...")


def w_onset_pos_only(label, tps, *, tau=20.0, max_w=4.0, max_ts=10**9):
    if label != 1:
        return 1.0
    if tps is None or tps >= max_ts:
        return 1.0
    # tps=0 -> max_w, then decays
    return 1.0 + (max_w - 1.0) * np.exp(-float(tps) / tau)


def w_preictal_neg_only(label, tts, *, pre_window=60.0, max_w=2.0, max_ts=10**9):
    if label != 0:
        return 1.0
    if tts is None or tts >= max_ts or tts <= 0:
        return 1.0
    # tts→0 increases weight
    x = 1.0 - float(tts) / pre_window
    x = max(0.0, min(1.0, x))
    return 1.0 + (max_w - 1.0) * x


class BlockNormCache:
    def __init__(self, block_len_samp: int, q: float = 0.95):
        self.block_len = int(block_len_samp)
        self.q = float(q)
        self._cache = {}  # block_id -> (mean(C,1), q95(C,1))

    def get_stats(self, signal: torch.Tensor, block_id: int):
        # signal: (C, Tfull) of current file
        if block_id in self._cache:
            return self._cache[block_id]

        s0 = block_id * self.block_len
        s1 = min(signal.shape[1], (block_id + 1) * self.block_len)
        blk = signal[:, s0:s1]  # (C, B)

        mean = blk.mean(dim=-1, keepdim=True)
        q95 = torch.quantile(blk.abs(), q=self.q, dim=-1, keepdim=True).clamp_min(1e-6)

        self._cache[block_id] = (mean, q95)
        return mean, q95


def configured_modalities_as_list(cfg):
    conf_modalities = {"signals"}
    if cfg.get("add_eeg_location", False):
        conf_modalities.add("eeg_location")
    if cfg.get("add_markers", False) or cfg.get("decode_markers", False):
        conf_modalities.add("markers")
    if cfg.get("add_notes", False) or cfg.get("decode_notes", False):
        conf_modalities.add("notes")
    if cfg.get("add_tps", False) or cfg.get("predict_tps", False):
        conf_modalities.add("tps")
    if cfg.get("predict_tts", False):
        conf_modalities.add("tts")
    if cfg.get("add_segtype", False) or cfg.get("predict_segtype", False):
        conf_modalities.add("segtype")
    if cfg.get("add_gender", False):
        conf_modalities.add("gender")
    if cfg.get("add_age", False):
        conf_modalities.add("age")
    if cfg.get("add_hod", False):
        conf_modalities.add("hod")
    if cfg.get("add_end_of_segment", False):
        conf_modalities.add("eos")
    return conf_modalities