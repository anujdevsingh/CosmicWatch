from __future__ import annotations

import argparse
import json
import pickle
from typing import Any

import numpy as np

from conjunction.train_conjunction_model import build_dataset, _to_xy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdm-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pc-threshold", type=float, default=1e-7)
    ap.add_argument("--n-theta", type=int, default=360)
    ap.add_argument("--n-r", type=int, default=300)
    ap.add_argument("--use-reference-pc", action="store_true")
    ap.add_argument("--out", default="conjunction_eval.json")
    args = ap.parse_args()

    rows = build_dataset(
        args.cdm_dir,
        pc_threshold=float(args.pc_threshold),
        n_theta=int(args.n_theta),
        n_r=int(args.n_r),
        use_reference_pc_if_present=bool(args.use_reference_pc),
    )
    if not rows:
        raise SystemExit("No usable CDM events found")

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    x, y, files = _to_xy(rows)
    proba = model.predict_proba(x)[:, 1]

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.calibration import calibration_curve

    metrics: dict[str, Any] = {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "brier": float(brier_score_loss(y, proba)),
    }

    if len(np.unique(y)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, proba))
        metrics["avg_precision"] = float(average_precision_score(y, proba))
    else:
        metrics["roc_auc"] = None
        metrics["avg_precision"] = None

    pred = (proba >= 0.5).astype(int)
    metrics["accuracy@0.5"] = float(accuracy_score(y, pred))
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    metrics["precision@0.5"] = float(p)
    metrics["recall@0.5"] = float(r)
    metrics["f1@0.5"] = float(f1)

    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="uniform")
    cal = [{"mean_pred": float(a), "frac_pos": float(b)} for a, b in zip(mean_pred, frac_pos)]

    out = {
        "metrics": metrics,
        "calibration": cal,
        "pc_threshold": float(args.pc_threshold),
        "features": [
            "miss_distance_km",
            "hbr_km",
            "sigma1_km",
            "sigma2_km",
            "rel_speed_km_s",
            "pc",
        ],
        "top_positive": [
            {"file": files[i], "score": float(proba[i])}
            for i in np.argsort(-proba)[:20]
        ],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"saved": args.out, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
