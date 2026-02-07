from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class CdmEvent:
    raw: dict[str, Any]
    primary_state_eci: Optional[tuple[np.ndarray, np.ndarray]]
    secondary_state_eci: Optional[tuple[np.ndarray, np.ndarray]]
    primary_pos_cov_rtn: Optional[np.ndarray]
    secondary_pos_cov_rtn: Optional[np.ndarray]
    hard_body_radius_km: Optional[float]


def parse_cdm_kvn(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip()
        if not key:
            continue
        out[key] = val
    return out


def _get_float(d: dict[str, str], *keys: str) -> Optional[float]:
    for k in keys:
        if k in d and d[k] != "":
            try:
                return float(d[k])
            except Exception:
                return None
    return None


def _get_vec3(d: dict[str, str], keys: tuple[str, str, str]) -> Optional[np.ndarray]:
    x = _get_float(d, keys[0])
    y = _get_float(d, keys[1])
    z = _get_float(d, keys[2])
    if x is None or y is None or z is None:
        return None
    return np.array([x, y, z], dtype=float)


def _parse_state_eci(d: dict[str, str], prefix: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    r = _get_vec3(d, (f"{prefix}_X", f"{prefix}_Y", f"{prefix}_Z"))
    v = _get_vec3(d, (f"{prefix}_X_DOT", f"{prefix}_Y_DOT", f"{prefix}_Z_DOT"))
    if r is None or v is None:
        return None
    return r, v


def _parse_pos_cov_rtn(d: dict[str, str], prefix: str) -> Optional[np.ndarray]:
    crr = _get_float(d, f"{prefix}_CRR", f"{prefix}_COV_RR", f"{prefix}_POS_COV_RR")
    ctt = _get_float(d, f"{prefix}_CTT", f"{prefix}_COV_TT", f"{prefix}_POS_COV_TT")
    cnn = _get_float(d, f"{prefix}_CNN", f"{prefix}_COV_NN", f"{prefix}_POS_COV_NN")
    crt = _get_float(d, f"{prefix}_CRT", f"{prefix}_COV_RT", f"{prefix}_POS_COV_RT")
    crn = _get_float(d, f"{prefix}_CRN", f"{prefix}_COV_RN", f"{prefix}_POS_COV_RN")
    ctn = _get_float(d, f"{prefix}_CTN", f"{prefix}_COV_TN", f"{prefix}_POS_COV_TN")

    if crr is None or ctt is None or cnn is None:
        return None
    crt = 0.0 if crt is None else crt
    crn = 0.0 if crn is None else crn
    ctn = 0.0 if ctn is None else ctn
    cov = np.array(
        [
            [crr, crt, crn],
            [crt, ctt, ctn],
            [crn, ctn, cnn],
        ],
        dtype=float,
    )
    return (cov + cov.T) / 2.0


def parse_cdm_event(text: str) -> CdmEvent:
    kv = parse_cdm_kvn(text)

    primary = _parse_state_eci(kv, "OBJECT1")
    secondary = _parse_state_eci(kv, "OBJECT2")

    cov1 = _parse_pos_cov_rtn(kv, "OBJECT1")
    cov2 = _parse_pos_cov_rtn(kv, "OBJECT2")

    hbr = _get_float(
        kv,
        "HARD_BODY_RADIUS",
        "HARD_BODY_RADIUS_KM",
        "COMBINED_HARD_BODY_RADIUS",
        "COMBINED_HARD_BODY_RADIUS_KM",
    )
    if hbr is None:
        obj1_hbr = _get_float(kv, "OBJECT1_HBR", "OBJECT1_HBR_KM")
        obj2_hbr = _get_float(kv, "OBJECT2_HBR", "OBJECT2_HBR_KM")
        if obj1_hbr is not None and obj2_hbr is not None:
            hbr = float(obj1_hbr + obj2_hbr)

    raw: dict[str, Any] = dict(kv)
    return CdmEvent(
        raw=raw,
        primary_state_eci=primary,
        secondary_state_eci=secondary,
        primary_pos_cov_rtn=cov1,
        secondary_pos_cov_rtn=cov2,
        hard_body_radius_km=hbr,
    )
