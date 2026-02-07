import argparse
import json
import os
import glob
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv


def _env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def spacetrack_login(session: requests.Session, base_url: str, username: str, password: str) -> None:
    url = f"{base_url.rstrip('/')}/ajaxauth/login"
    response = session.post(url, data={"identity": username, "password": password}, timeout=30)
    if response.status_code != 200:
        raise SystemExit(f"Space-Track login failed: HTTP {response.status_code}")
    if "Login failed" in response.text or "invalid" in response.text.lower():
        raise SystemExit("Space-Track login failed: invalid credentials or insufficient access")


def spacetrack_query(session: requests.Session, base_url: str, controller: str, class_name: str, extra_parts: list[str]) -> list[dict]:
    parts = [base_url.rstrip("/"), controller, "query", "class", class_name]
    parts.extend(extra_parts)
    parts.extend(["format", "json"])
    url = "/".join(parts)
    response = session.get(url, timeout=60)
    if response.status_code != 200:
        preview = (response.text or "")[:300].replace("\n", " ").strip()
        raise SystemExit(f"Space-Track query failed: HTTP {response.status_code}{(': ' + preview) if preview else ''}")
    try:
        data = response.json()
    except Exception as e:
        raise SystemExit(f"Space-Track returned non-JSON response: {e}")
    if not isinstance(data, list):
        raise SystemExit(f"Unexpected response type: {type(data)}")
    return data


def _chunked(seq: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0:
        return [seq]
    return [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]


def fetch_satcat_and_gp_for_ids(
    session: requests.Session,
    base_url: str,
    norad_ids: list[int],
    *,
    id_chunk_size: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not norad_ids:
        return pd.DataFrame(), pd.DataFrame()
    ids = sorted(set(int(i) for i in norad_ids if i is not None))
    satcat_rows_all: list[dict] = []
    gp_rows_all: list[dict] = []
    for batch in _chunked(ids, int(id_chunk_size)):
        ids_csv = ",".join(str(i) for i in batch)
        satcat_rows_all.extend(spacetrack_query(session, base_url, "basicspacedata", "satcat", ["NORAD_CAT_ID", ids_csv]))
        gp_rows_all.extend(spacetrack_query(session, base_url, "basicspacedata", "gp", ["NORAD_CAT_ID", ids_csv]))
    satcat = pd.DataFrame(satcat_rows_all)
    gp = pd.DataFrame(gp_rows_all)
    if not satcat.empty and "NORAD_CAT_ID" in satcat.columns:
        satcat["NORAD_CAT_ID"] = pd.to_numeric(satcat["NORAD_CAT_ID"], errors="coerce").astype("Int64")
    if not gp.empty and "NORAD_CAT_ID" in gp.columns:
        gp["NORAD_CAT_ID"] = pd.to_numeric(gp["NORAD_CAT_ID"], errors="coerce").astype("Int64")
    return satcat, gp


def normalize_cdm_public(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ("SAT_1_ID", "SAT_2_ID"):
        if col in df.columns:
            df[col] = df[col].apply(_maybe_int)
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ("MIN_RNG", "PC", "SAT_1_EXCL_VOL", "SAT_2_EXCL_VOL"):
        if col in df.columns:
            df[col] = df[col].apply(_maybe_float)

    if "CREATED" in df.columns:
        df["created_dt"] = df["CREATED"].apply(_parse_dt)
    if "TCA" in df.columns:
        df["tca_dt"] = df["TCA"].apply(_parse_dt)

    if "created_dt" in df.columns and "tca_dt" in df.columns:
        df["hours_to_tca"] = (df["tca_dt"] - df["created_dt"]).dt.total_seconds() / 3600.0

    if "EMERGENCY_REPORTABLE" in df.columns:
        df["emergency_reportable"] = df["EMERGENCY_REPORTABLE"].astype(str).str.upper().eq("Y").astype(int)

    if "PC" in df.columns:
        pc_num = pd.to_numeric(df["PC"], errors="coerce")
        df["pc_present"] = pc_num.notna().astype(int)
        df["pc_log10"] = pd.Series(np.nan, index=df.index, dtype="float64")
        df.loc[df["pc_present"] == 1, "pc_log10"] = np.log10(pc_num[df["pc_present"] == 1].clip(lower=1e-12))

        df["pc_bucket_high"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df.loc[df["pc_present"] == 1, "pc_bucket_high"] = (pc_num[df["pc_present"] == 1] >= 1e-4).astype(int)

        df["pc_risk_class"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df.loc[(df["pc_present"] == 1) & (pc_num >= 1e-7), "pc_risk_class"] = 1
        df.loc[(df["pc_present"] == 1) & (pc_num >= 1e-5), "pc_risk_class"] = 2
        df.loc[(df["pc_present"] == 1) & (pc_num >= 1e-4), "pc_risk_class"] = 3
        df.loc[(df["pc_present"] == 1) & (pc_num < 1e-7), "pc_risk_class"] = 0
        df["pc_risk_level"] = df["pc_risk_class"].map({0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"})

        if int(df["pc_present"].sum()) >= 100:
            qs = pc_num[df["pc_present"] == 1].quantile([0.25, 0.5, 0.75]).to_list()
            q25, q50, q75 = float(qs[0]), float(qs[1]), float(qs[2])
            df["pc_quantile_class"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
            df.loc[df["pc_present"] == 1, "pc_quantile_class"] = 0
            df.loc[(df["pc_present"] == 1) & (pc_num >= q25), "pc_quantile_class"] = 1
            df.loc[(df["pc_present"] == 1) & (pc_num >= q50), "pc_quantile_class"] = 2
            df.loc[(df["pc_present"] == 1) & (pc_num >= q75), "pc_quantile_class"] = 3
            df["pc_quantile_level"] = df["pc_quantile_class"].map({0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"})

    return df


def prefix_columns(df: pd.DataFrame, prefix: str, id_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if id_col in df.columns:
        df = df.rename(columns={id_col: f"{prefix}NORAD_CAT_ID"})
    rename = {}
    for col in df.columns:
        if col == f"{prefix}NORAD_CAT_ID":
            continue
        rename[col] = f"{prefix}{col}"
    return df.rename(columns=rename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdm-json", action="append", default=None)
    parser.add_argument("--cdm-dir", default=None)
    parser.add_argument("--out-csv", default="datasets/cdm_public_dataset.csv")
    parser.add_argument("--fetch-satcat-gp", action="store_true")
    parser.add_argument("--satcat-json", default=None)
    parser.add_argument("--gp-json", default=None)
    parser.add_argument("--id-chunk-size", type=int, default=200)
    parser.add_argument("--dedupe", action="store_true")
    args = parser.parse_args()

    sources: list[str] = []
    if args.cdm_json:
        sources.extend(args.cdm_json)
    if args.cdm_dir:
        sources.extend(sorted(glob.glob(os.path.join(args.cdm_dir, "*.json"))))
    sources = [s for s in sources if s]
    if not sources:
        raise SystemExit("Provide --cdm-json (one or more) or --cdm-dir")

    cdm_rows_all: list[dict] = []
    for path in sources:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise SystemExit(f"CDM input must be a JSON array: {path}")
        cdm_rows_all.extend(payload)

    if args.dedupe:
        seen: set[str] = set()
        deduped: list[dict] = []
        for row in cdm_rows_all:
            key = str(row.get("CDM_ID") or "").strip() or json.dumps(row, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        cdm_rows_all = deduped

    cdm = normalize_cdm_public(cdm_rows_all)
    if cdm.empty:
        raise SystemExit("No CDM rows found in input")

    satcat = pd.DataFrame()
    gp = pd.DataFrame()

    if args.satcat_json:
        with open(args.satcat_json, "r", encoding="utf-8") as f:
            satcat = pd.DataFrame(json.load(f))
    if args.gp_json:
        with open(args.gp_json, "r", encoding="utf-8") as f:
            gp = pd.DataFrame(json.load(f))

    if args.fetch_satcat_gp and (satcat.empty or gp.empty):
        load_dotenv()
        base_url = (os.getenv("SPACETRACK_BASE_URL") or "https://www.space-track.org").strip()
        username = _env("SPACETRACK_USERNAME")
        password = _env("SPACETRACK_PASSWORD")
        ids = [i for i in pd.concat([cdm["SAT_1_ID"], cdm["SAT_2_ID"]]).dropna().astype(int).tolist()]
        session = requests.Session()
        spacetrack_login(session, base_url, username, password)
        satcat, gp = fetch_satcat_and_gp_for_ids(session, base_url, ids, id_chunk_size=int(args.id_chunk_size))

    satcat_sat1 = pd.DataFrame()
    satcat_sat2 = pd.DataFrame()
    gp_sat1 = pd.DataFrame()
    gp_sat2 = pd.DataFrame()

    if not satcat.empty and "NORAD_CAT_ID" in satcat.columns:
        satcat_sat1 = prefix_columns(satcat, "sat1_", "NORAD_CAT_ID")
        satcat_sat2 = prefix_columns(satcat, "sat2_", "NORAD_CAT_ID")

    if not gp.empty and "NORAD_CAT_ID" in gp.columns:
        gp_sat1 = prefix_columns(gp, "sat1_gp_", "NORAD_CAT_ID")
        gp_sat2 = prefix_columns(gp, "sat2_gp_", "NORAD_CAT_ID")

    out = cdm.copy()

    if not satcat_sat1.empty:
        out = out.merge(satcat_sat1, left_on="SAT_1_ID", right_on="sat1_NORAD_CAT_ID", how="left")
    if not satcat_sat2.empty:
        out = out.merge(satcat_sat2, left_on="SAT_2_ID", right_on="sat2_NORAD_CAT_ID", how="left")

    if not gp_sat1.empty:
        out = out.merge(gp_sat1, left_on="SAT_1_ID", right_on="sat1_gp_NORAD_CAT_ID", how="left")
    if not gp_sat2.empty:
        out = out.merge(gp_sat2, left_on="SAT_2_ID", right_on="sat2_gp_NORAD_CAT_ID", how="left")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
