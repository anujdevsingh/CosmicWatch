from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from conjunction.cdm import parse_cdm_event
from conjunction.frames import combined_pos_cov_encounter_plane
from conjunction.pc import pc_circle_grid


@dataclass(frozen=True)
class DatasetRow:
    file: str
    pc: float
    miss_distance_km: float
    hbr_km: float
    sigma1_km: float
    sigma2_km: float
    rel_speed_km_s: float
    label: int


def _iter_files(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith((".cdm", ".kvn", ".txt")):
                out.append(os.path.join(dirpath, name))
    out.sort()
    return out


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _extract_reference_pc(raw: dict[str, Any]) -> Optional[float]:
    candidates = [
        "COLLISION_PROBABILITY",
        "COLLISION_PROBABILITY_POC",
        "PC",
        "P_C",
    ]
    for k in candidates:
        if k in raw:
            f = _safe_float(raw[k])
            if f is not None:
                return f
    return None


def build_dataset(
    cdm_dir: str,
    *,
    pc_threshold: float,
    n_theta: int,
    n_r: int,
    use_reference_pc_if_present: bool,
) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    for path in _iter_files(cdm_dir):
        try:
            text = open(path, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue

        ev = parse_cdm_event(text)
        if ev.primary_state_eci is None or ev.secondary_state_eci is None:
            continue
        if ev.primary_pos_cov_rtn is None or ev.secondary_pos_cov_rtn is None:
            continue
        if ev.hard_body_radius_km is None:
            continue

        (r1, v1) = ev.primary_state_eci
        (r2, v2) = ev.secondary_state_eci

        mu2, cov2 = combined_pos_cov_encounter_plane(
            primary_r_eci=r1,
            primary_v_eci=v1,
            secondary_r_eci=r2,
            secondary_v_eci=v2,
            cov1_rtn_pos=ev.primary_pos_cov_rtn,
            cov2_rtn_pos=ev.secondary_pos_cov_rtn,
        )

        hbr = float(ev.hard_body_radius_km)
        miss = float(np.linalg.norm(mu2))
        rel_speed = float(np.linalg.norm(v2 - v1))

        pc_ref = _extract_reference_pc(ev.raw) if use_reference_pc_if_present else None
        if pc_ref is not None:
            pc = float(max(0.0, min(pc_ref, 1.0)))
        else:
            pc = pc_circle_grid(mu2, cov2, hbr, n_theta=n_theta, n_r=n_r).pc

        evals = np.linalg.eigvalsh(cov2)
        evals = np.clip(evals, 0.0, None)
        sigma1 = float(np.sqrt(evals[-1]))
        sigma2 = float(np.sqrt(evals[0]))

        label = 1 if pc >= pc_threshold else 0
        rows.append(
            DatasetRow(
                file=os.path.relpath(path, cdm_dir),
                pc=pc,
                miss_distance_km=miss,
                hbr_km=hbr,
                sigma1_km=sigma1,
                sigma2_km=sigma2,
                rel_speed_km_s=rel_speed,
                label=label,
            )
        )
    return rows


def _to_xy(rows: list[DatasetRow]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x = np.array(
        [
            [
                r.miss_distance_km,
                r.hbr_km,
                r.sigma1_km,
                r.sigma2_km,
                r.rel_speed_km_s,
                r.pc,
            ]
            for r in rows
        ],
        dtype=float,
    )
    y = np.array([r.label for r in rows], dtype=int)
    files = [r.file for r in rows]
    return x, y, files


def train_and_evaluate(rows: list[DatasetRow], *, seed: int = 42) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    x, y, _ = _to_xy(rows)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y if len(set(y)) > 1 else None
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    metrics: dict[str, Any] = {
        "n": int(len(rows)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
    }

    if len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        metrics["avg_precision"] = float(average_precision_score(y_test, proba))
    else:
        metrics["roc_auc"] = None
        metrics["avg_precision"] = None

    return {"model": model, "metrics": metrics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdm-dir", required=True)
    ap.add_argument("--pc-threshold", type=float, default=1e-7)
    ap.add_argument("--n-theta", type=int, default=360)
    ap.add_argument("--n-r", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-reference-pc", action="store_true")
    ap.add_argument("--out-model", default="conjunction_model.pkl")
    ap.add_argument("--out-report", default="conjunction_model_report.json")
    args = ap.parse_args()

    rows = build_dataset(
        args.cdm_dir,
        pc_threshold=float(args.pc_threshold),
        n_theta=int(args.n_theta),
        n_r=int(args.n_r),
        use_reference_pc_if_present=bool(args.use_reference_pc),
    )
    if len(rows) < 50:
        raise SystemExit(f"Not enough usable CDM events for training: {len(rows)}")

    result = train_and_evaluate(rows, seed=int(args.seed))
    model = result["model"]
    metrics = result["metrics"]

    with open(args.out_model, "wb") as f:
        pickle.dump(model, f)

    report = {
        "metrics": metrics,
        "features": [
            "miss_distance_km",
            "hbr_km",
            "sigma1_km",
            "sigma2_km",
            "rel_speed_km_s",
            "pc",
        ],
        "pc_threshold": float(args.pc_threshold),
        "dataset_size": int(len(rows)),
        "sample_rows": [asdict(r) for r in rows[:10]],
    }

    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"saved_model": args.out_model, "saved_report": args.out_report, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
