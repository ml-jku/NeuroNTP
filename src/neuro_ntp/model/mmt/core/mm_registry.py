"""
+----------------------------------------------------------------------------------------------------------------------+
mm_registry.py
Registry of families of lazy modules.
Each module family has its own factory, id normalizer, and optional nearest-init.

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+
"""
import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Tuple, Any
import neuro_ntp.misc.nntp_utils as nnutils

import logging
logger = logging.getLogger(__name__)


class MultiModuleRegistry(nn.Module):
    """
    A registry of *families* of lazy modules - the purpose of this module is facilitating the handling of different
    NeuroNTP pretraining, finetuning and evaluation setups across different dataset configurations progressively.

    Each module family has its own factory, id normalizer, and optional nearest-init.
      - register(class_key, make_module, nearest_init=None, normalize_id=None)
      - get(class_key, module_id) -> nn.Module
    Export/import persists states per (class_key, module_id).
    The registry can be used to hold encoders for modalities, channels, context and metadata encoders, etc.
    *Nearest init* allows warm-starting a previously non-existant encoder from a pre-defined similar module
    (e.g. EEG F7 from F5, smell channel C03 from C04, etc. using a distance/ranked similarity mapping).
    *Normalize ID* is used to handle all module family and module/channel IDs in a standard way.
    """

    class _Family(nn.Module):
        def __init__(
            self,
            make_module: Callable[[], nn.Module],
            nearest_init: Optional[Callable[[str], Optional[Dict[str, torch.Tensor]]]] = None,
            normalize_id: Optional[Callable[[str], str]] = None,
        ):
            super().__init__()
            self.make_module = make_module
            self.nearest_init = nearest_init
            self.normalize_id = normalize_id
            self.modules_by_id = nn.ModuleDict()                   # module_id -> nn.Module
            self._stash: Dict[str, Dict[str, torch.Tensor]] = {}   # module_id -> state_dict
            self.register_buffer("_device_anchor", torch.empty(0))

        def _canon(self, module_id: str) -> str:
            return self.normalize_id(module_id) if self.normalize_id else module_id

        def get(self, module_id: str) -> nn.Module:
            key = self._canon(module_id)
            if key not in self.modules_by_id:
                # create on same device as this family holder !important: GPU / DDP env.
                device = self._device_anchor.device
                mod = self.make_module()
                if device is not None:
                    mod = mod.to(device)
                self.modules_by_id[key] = mod
                # load exact stash or nearest init
                if key in self._stash:
                    mod.load_state_dict(self._stash.pop(key), strict=False)
                elif self.nearest_init is not None:
                    sd = self.nearest_init(key)
                    if sd is not None:
                        mod.load_state_dict(sd, strict=False)
            return self.modules_by_id[key]

        def has(self, module_id: str) -> bool:
            return self._canon(module_id) in self.modules_by_id

        def known_ids(self):
            return list(self.modules_by_id.keys())

        @torch.no_grad()
        def export_states(self) -> Dict[str, Dict[str, torch.Tensor]]:
            out: Dict[str, Dict[str, torch.Tensor]] = {}
            for key, mod in self.modules_by_id.items():
                sd = mod.state_dict()
                out[key] = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}
            return out

        @torch.no_grad()
        def import_states(self, stash: Optional[Dict[str, Dict[str, torch.Tensor]]], merge: bool = True):
            if not stash:
                return

            # normalize: keys like "eeg fp1" etc.
            canon = lambda k: self._canon(k)
            stash_norm = {canon(k): v for k, v in stash.items()}

            # merge with existing family stash
            if merge and self._stash:
                for k, v in stash_norm.items():
                    if k not in self._stash:
                        self._stash[k] = v
            else:
                self._stash = dict(stash_norm)

            def _avg_state_dict(sd1: Dict[str, torch.Tensor], sd2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                # average common params; if a key is missing in one, copy from the present one
                out = {}
                keys = set(sd1.keys()) | set(sd2.keys())
                for k in keys:
                    if k in sd1 and k in sd2:
                        t1, t2 = sd1[k], sd2[k]
                        # keep dtype/device of t1
                        out[k] = 0.5 * t1 + 0.5 * t2.to(dtype=t1.dtype, device=t1.device)
                    elif k in sd1:
                        out[k] = sd1[k]
                    else:
                        out[k] = sd2[k]
                return out

            def _get_sd_for(name: str) -> Optional[Dict[str, torch.Tensor]]:
                """return state_dict for exact canonical name or None."""
                return self._stash.get(name, None)

            for m_id in list(self.modules_by_id.keys()):
                # exact single-ended match
                if m_id in self._stash:
                    self.modules_by_id[m_id].load_state_dict(self._stash[m_id], strict=False)
                    logger.debug(f"... loaded state dictionary for: {m_id}")
                    continue

                # if bipolar: "eeg a-b"
                if m_id.startswith("eeg") and "-" in m_id:
                    # parse endpoints
                    try:
                        _, rest = m_id.split(" ", 1)
                        a_raw, b_raw = [s.strip() for s in rest.split("-", 1)]
                    except Exception:
                        a_raw = b_raw = None

                    if a_raw and b_raw:
                        a_key = canon("eeg " + a_raw)
                        b_key = canon("eeg " + b_raw)

                        sd_a = _get_sd_for(a_key)
                        sd_b = _get_sd_for(b_key)

                        # first endpoint exists: use it
                        if sd_a is not None:
                            self.modules_by_id[m_id].load_state_dict(sd_a, strict=False)
                            logger.debug(f"... loaded ENDPOINT-A dictionary for: {m_id} from {a_key}")
                            continue
                        # second endpoint exists: use it
                        elif sd_b is not None:
                            self.modules_by_id[m_id].load_state_dict(sd_b, strict=False)
                            logger.debug(f"... loaded ENDPOINT-B dictionary for: {m_id} from {b_key}")
                            continue

                        # fallback: nearest substitute for endpoints, then blend if possible
                        subst_a = nnutils.find_nearest_substitute(a_key, self._stash.keys())
                        subst_b = nnutils.find_nearest_substitute(b_key, self._stash.keys())
                        sd_sa = _get_sd_for(subst_a)
                        sd_sb = _get_sd_for(subst_b)
                        if sd_sa is not None:
                            self.modules_by_id[m_id].load_state_dict(sd_sa, strict=False)
                            logger.debug(f"... loaded SUBSTITUTE-A dictionary for: {m_id} from {subst_a}")
                            continue
                        if sd_sb is not None:
                            self.modules_by_id[m_id].load_state_dict(sd_sb, strict=False)
                            logger.debug(f"... loaded SUBSTITUTE-B dictionary for: {m_id} from {subst_b}")
                            continue

                # final single-channel fallback
                if m_id.startswith("eeg"):
                    subst = nnutils.find_nearest_substitute(m_id, self._stash.keys())
                    if subst in self._stash:
                        self.modules_by_id[m_id].load_state_dict(self._stash[subst], strict=False)
                        logger.debug(f"... loaded SUBSTITUTE dictionary for: {m_id} from {subst}")
                        continue

                subst = "eeg cz"
                if subst in self._stash.keys():
                    logger.debug(f"... loaded DEFAULT EEG CZ dictionary for: {m_id} from {subst}")
                    self.modules_by_id[m_id].load_state_dict(self._stash[subst], strict=False)
                    continue

                logger.debug(f"[WARN] no suitable pretrained state found for: {m_id} (left as init)")

    # multi-family wrapper
    def __init__(self, default_class_key: Optional[str] = None):
        super().__init__()
        self._families = nn.ModuleDict()  # class_key -> _Family
        self._default = default_class_key

    def register(
        self,
        class_key: str,
        make_module: Callable[[], nn.Module],
        *,
        nearest_init: Optional[Callable[[str], Optional[Dict[str, torch.Tensor]]]] = None,
        normalize_id: Optional[Callable[[str], str]] = None,
    ) -> None:
        """This method is used to register/declare a new module family. Only needs to be called once per family."""
        if class_key in self._families:
            raise KeyError(f"Module family '{class_key}' already registered.")
        fam = MultiModuleRegistry._Family(make_module, nearest_init, normalize_id)
        self._families[class_key] = fam
        # set default if not set
        if self._default is None:
            self._default = class_key

    def families(self):
        return list(self._families.keys())

    def family(self, class_key: Optional[str] = None) -> "_Family":
        key = class_key or self._default
        if key is None or key not in self._families:
            raise KeyError(f"No module family registered under key '{class_key}'.")
        return self._families[key]

    # standard API methods
    def get(self, class_key: str, module_id: Optional[str] = None) -> nn.Module:
        """retrieve/create a new module by (class_key, module_id). If class_key is None, uses default family."""
        return self.family(class_key).get(class_key if module_id is None else module_id)

    def has(self, class_key: str, module_id: str) -> bool:
        return self.family(class_key).has(module_id)

    def known_ids(self, class_key: Optional[str] = None):
        return self.family(class_key).known_ids()

    # export/import (nested or flat)
    @torch.no_grad()
    def export_states(self) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
        """return {class_key: {module_id: state_dict}}"""
        out: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        for ck, fam in self._families.items():
            out[ck] = fam.export_states()
        return out

    @torch.no_grad()
    def import_states(self, stash: Any, merge: bool = True):
        """
        accepts either:
        - nested: {class_key: {module_id: state_dict}}
        - flat: { (class_key, module_id): state_dict }
        """
        if not stash:
            return
        # detect flat vs nested
        if isinstance(next(iter(stash.keys())), tuple):
            # flat
            nested: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
            for (ck, mid), sd in stash.items():
                nested.setdefault(ck, {})[mid] = sd
            stash = nested

        # now nested
        for ck, sub in stash.items():
            if ck not in self._families:
                # silently ignore unknown families (forward compatibility)
                continue
            logger.debug(f"[registry] importing family={ck} n_ids={len(sub)}")
            self._families[ck].import_states(sub, merge=merge)

    def warmup(self, family: str, ids):
        for _id in ids:
            _ = self.get(family, _id)
