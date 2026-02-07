from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def apply_cdm_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "MIN_RNG" in work.columns:
        min_rng = pd.to_numeric(work["MIN_RNG"], errors="coerce")
        work["MIN_RNG"] = min_rng
        work["log_min_rng"] = np.log10(min_rng.clip(lower=1.0))

    if "sat1_gp_INCLINATION" in work.columns and "sat2_gp_INCLINATION" in work.columns:
        work["delta_inclination"] = (pd.to_numeric(work["sat1_gp_INCLINATION"], errors="coerce") - pd.to_numeric(work["sat2_gp_INCLINATION"], errors="coerce")).abs()

    if "sat1_gp_MEAN_MOTION" in work.columns and "sat2_gp_MEAN_MOTION" in work.columns:
        work["delta_mean_motion"] = (pd.to_numeric(work["sat1_gp_MEAN_MOTION"], errors="coerce") - pd.to_numeric(work["sat2_gp_MEAN_MOTION"], errors="coerce")).abs()

    if "SAT_1_EXCL_VOL" in work.columns and "SAT_2_EXCL_VOL" in work.columns:
        work["excl_vol_sum"] = pd.to_numeric(work["SAT_1_EXCL_VOL"], errors="coerce").fillna(0.0) + pd.to_numeric(work["SAT_2_EXCL_VOL"], errors="coerce").fillna(0.0)

    if "hours_to_tca" in work.columns:
        work["hours_to_tca"] = pd.to_numeric(work["hours_to_tca"], errors="coerce")

    return work


def ensure_feature_columns(df: pd.DataFrame, *, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    work = df.copy()

    for col in numeric_cols:
        if col not in work.columns:
            work[col] = np.nan

    for col in categorical_cols:
        if col not in work.columns:
            work[col] = None

    return work[numeric_cols + categorical_cols]


def feature_frame_from_record(record: dict[str, Any], *, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([record])
    df = apply_cdm_derived_features(df)
    return ensure_feature_columns(df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)

