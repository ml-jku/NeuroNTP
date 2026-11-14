"""
+----------------------------------------------------------------------------------------------------------------------+
token_type.py
Token types and channel subtype vocabularies of the multimodal tokenizer

Project: NeuroNTP
Institute for Machine Learning, Johannes Kepler University Linz
Author:  Jozsef Kovacs (jozsef.kovacs@jku.at)
License: MIT (see LICENSE)
+----------------------------------------------------------------------------------------------------------------------+

Description:
    Defines the token types (TokenType) of the signal, metadata and structural tokens, the subtype vocabularies of
    multi-channel sensors (e.g. accelerometer axes, PPG wavelengths), and the mapping of channel names to token types
    and subtypes. Token type IDs are stored in checkpoints; new types must only be added at the end of the enum.
+----------------------------------------------------------------------------------------------------------------------+
"""
from typing import Dict, List, Optional
from enum import IntEnum
import re

class TokenType(IntEnum):
    INVALID   = 0    # tokens padded for conformity, no relevant information

    EEG       = 1    # EEG signal; channel/location specified separately (e.g., 10–20 / 10–10)
    EOG       = 2    # electro-oculography (e.g., EOG L / EOG R)
    ECG       = 3    # electrocardiography (e.g., EKGL / EKGR / ECG1, etc.)
    EMG       = 4    # electromyography (e.g., L / R)

    SpO2      = 5    # blood oxygen saturation (pulse oximetry)
    EDA       = 6    # electrodermal activity
    HR        = 7    # heart rate
    SkinTEMP  = 8    # skin temperature
    ACC       = 9    # accelerometry (e.g., acc_x, acc_y, acc_z, acc_mag)
    Composite = 10   # composite features extracted from multiple channels
    Marker    = 11   # annotation / marker text
    Notes     = 12   # clinical free-text notes / remarks
    Segtype   = 13   # segment type: non-ictal / ictal / pre-ictal
    TPS       = 14   # time-past-seizure (time since last seizure; large value ≈ unknown)
    TTS       = 15   # time-to-seizure (until next seizure; usually not used in causal models)
    HoD       = 16   # hour-of-day (circadian rhythm)
    DoW       = 17   # day-of-week (septadian rhythm)
    EOS       = 18   # end-of-segment token
    Gender    = 19   # patient gender
    Age       = 20   # patient age at recording time

    Smell     = 21   # smell/odor sensor (e.g., bank channels c00..c63)
    Movement  = 22   # movement sensor (axes: x, y, z, mag)
    PPG       = 23   # photoplethysmography (e.g., green, red, IR, ambient, quality)
    RR        = 24   # respiration rate
    BodyTEMP  = 25   # body (core) temperature
    NIRS      = 26   # near-infrared spectroscopy (subtypes: red / IR)
    Magnet    = 27   # magnetometer (axes: x, y, z, mag)
    Gyro      = 28   # gyroscope (axes: x, y, z, mag)
    OSat      = 29   # oxygen saturation (derived/alt channel if distinct from SpO2)

    # technical / device state / meta channels
    FanSpeed  = 30   # device fan speed
    Humidity  = 31   # device/environment humidity
    DevTemp   = 32   # device temperature
    Voltage   = 33   # device supply voltage
    Battery   = 34   # device battery level/state
    Seq       = 35   # sequence/order indicator
    Reference = 36   # reference channel / baseline

    # NOTE: add new types only at the END to keep IDs stable across checkpoints.

    @classmethod
    def max_id(cls) -> int:
        return max(int(m) for m in cls)


# forward/backward maps for strings stored in emb.token_types
NAME_TO_ID: Dict[str, int] = {t.name: int(t) for t in TokenType}
_ALIASES: Dict[str, int] = {
    "EoS": TokenType.EOS, "eos": TokenType.EOS, "hod": TokenType.HoD, "dow": TokenType.DoW,
    "spo2": TokenType.SpO2, "skintemp": TokenType.SkinTEMP, "bodytemp": TokenType.BodyTEMP,
    "ekg": TokenType.ECG, "heartRate": TokenType.HR, "heartrate": TokenType.HR,
    "resp": TokenType.RR, "respiration": TokenType.RR, "resprate": TokenType.RR,
}
NAME_BY_LOWER: Dict[str, TokenType] = {t.name.lower(): t for t in TokenType}

def _canon(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())

# auto-build a lookup from enum names
_ENUM_CANON = { _canon(t.name): t for t in TokenType }

def token_type_from_base(base: str) -> TokenType:
    key = _canon(base)
    return _ALIASES.get(key, _ENUM_CANON.get(key, TokenType.Composite))

def token_type_id(name: str) -> int:
    if name in NAME_TO_ID: return NAME_TO_ID[name]
    lower = name.lower()
    if lower in _ALIASES: return int(_ALIASES[lower])
    if name.title() in NAME_TO_ID: return NAME_TO_ID[name.title()]
    raise KeyError(f"Unknown token type name: {name}")


SUBTYPE_VOCABS: Dict[TokenType, List[str]] = {TokenType.EOG: ["l", "r", "unk"], TokenType.EMG: ["l", "r", "unk"],
                                              TokenType.ECG: ["l", "r", "1", "unk"],
                                              TokenType.ACC: ["x", "y", "z", "mag", "unk"],
                                              TokenType.Movement: ["x", "y", "z", "mag", "unk"],
                                              TokenType.Magnet: ["x", "y", "z", "mag", "unk"],
                                              TokenType.Gyro: ["x", "y", "z", "mag", "unk"],
                                              TokenType.PPG: ["green", "red", "ir", "ambient", "quality", "unk"],
                                              TokenType.NIRS: ["red", "ir", "unk"],
                                              TokenType.Smell: [f"c{i:02d}" for i in range(64)] + ["unk"]}


def has_subtypes(base: TokenType) -> bool:
    return base in SUBTYPE_VOCABS


def subtype_id(base: TokenType, sub: Optional[str]) -> int:
    vocab = SUBTYPE_VOCABS.get(base)
    if not vocab:
        return -1  # indicates "no subtype for this base"
    s = (sub or "").strip().lower()
    if s in vocab:
        return vocab.index(s)
    # fallback to 'unk'
    return vocab.index("unk")


def parse_channel_base_and_sub(name: str):
    # expects "<base> <sub>" or "<base>"
    parts = name.strip().split(maxsplit=1)
    base = parts[0]
    sub  = parts[1].strip().lower() if len(parts) > 1 else None
    tok  = token_type_from_base(base)
    return tok, sub


def decode_token_types(ch_names: List[str]) -> List[TokenType]:
    """Map channel names to per-channel TokenType list."""
    return [parse_channel_base_and_sub(n)[0] for n in ch_names]
