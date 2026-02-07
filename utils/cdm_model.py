from __future__ import annotations

import glob
import json
import os
import pickle
from typing import Any

from utils.cdm_features import feature_frame_from_record


def _latest_file(paths: list[str]) -> str | None:
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        return None
    return max(paths, key=lambda p: os.path.getmtime(p))


def load_latest_baseline_model(
    *,
    models_dir: str,
    target: str = "pc_quantile_class",
    metrics_path: str | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    metrics_file = metrics_path
    if not metrics_file:
        pattern = os.path.join(models_dir, f"{target}_*_metrics.json")
        metrics_file = _latest_file(glob.glob(pattern))
    if not metrics_file:
        raise FileNotFoundError(f"No metrics found for target={target} in {models_dir}")

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    if not isinstance(metrics, dict):
        raise ValueError("Invalid metrics JSON")

    inferred_model_path = model_path
    if not inferred_model_path:
        if metrics_file.endswith("_metrics.json"):
            inferred_model_path = metrics_file[: -len("_metrics.json")] + ".pkl"
        else:
            inferred_model_path = None
    if not inferred_model_path or not os.path.isfile(inferred_model_path):
        raise FileNotFoundError(f"Model file not found for metrics: {metrics_file}")

    with open(inferred_model_path, "rb") as f:
        model = pickle.load(f)

    numeric_cols = metrics.get("features_numeric") or []
    categorical_cols = metrics.get("features_categorical") or []
    if not isinstance(numeric_cols, list) or not isinstance(categorical_cols, list):
        raise ValueError("Invalid feature lists in metrics")
    numeric_cols = [str(c) for c in numeric_cols]
    categorical_cols = [str(c) for c in categorical_cols]

    return {
        "kind": "baseline",
        "target": metrics.get("target") or target,
        "metrics_path": metrics_file,
        "model_path": inferred_model_path,
        "metrics": metrics,
        "model": model,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def predict_baseline(
    loaded: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    model = loaded["model"]
    numeric_cols: list[str] = loaded["numeric_cols"]
    categorical_cols: list[str] = loaded["categorical_cols"]

    x = feature_frame_from_record(record, numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    proba = model.predict_proba(x)[0]
    classes = None
    try:
        classes = list(getattr(model.named_steps["classifier"], "classes_", []))
    except Exception:
        classes = list(range(len(proba)))

    class_probs: dict[str, float] = {}
    for c, p in zip(classes, proba):
        class_probs[str(int(c)) if isinstance(c, (int, float)) and float(c).is_integer() else str(c)] = float(p)

    best = max(class_probs.items(), key=lambda kv: kv[1])[0]

    return {
        "predicted_class": best,
        "probabilities": class_probs,
    }

