import argparse
import glob
import json
import os
from typing import Any


def _load_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _latest(patterns: list[str]) -> str | None:
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        return None
    return max(paths, key=lambda p: os.path.getmtime(p))


def _extract_report_macro_f1(report: Any) -> float | None:
    if not isinstance(report, dict):
        return None
    macro = report.get("macro avg")
    if not isinstance(macro, dict):
        return None
    f1 = macro.get("f1-score")
    try:
        return float(f1)
    except Exception:
        return None


def _summarize_baseline(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    target = payload.get("target")
    test_report = payload.get("test_classification_report") or payload.get("classification_report")
    val_report = payload.get("val_classification_report")
    out = {
        "kind": "baseline",
        "path": path,
        "target": target,
        "n_train": payload.get("n_rows_train"),
        "n_val": payload.get("n_rows_val"),
        "n_test": payload.get("n_rows_test"),
        "test_accuracy": (test_report or {}).get("accuracy") if isinstance(test_report, dict) else None,
        "test_macro_f1": _extract_report_macro_f1(test_report),
        "val_accuracy": (val_report or {}).get("accuracy") if isinstance(val_report, dict) else None,
        "val_macro_f1": _extract_report_macro_f1(val_report),
    }
    return out


def _summarize_cim(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    mode = payload.get("mode") or "classification"
    out = {
        "kind": "cim",
        "path": path,
        "mode": mode,
        "label_col": payload.get("label_col"),
        "target_col": payload.get("target_col"),
        "test_acc": payload.get("test_acc"),
        "balanced_acc": payload.get("balanced_acc"),
        "test_rmse": payload.get("test_rmse"),
        "test_mae": payload.get("test_mae"),
        "macro_f1": _extract_report_macro_f1(payload.get("classification_report")),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-metrics", default=None)
    ap.add_argument("--cim-metrics", default=None)
    ap.add_argument("--models-dir", default="models/cdm_public")
    args = ap.parse_args()

    baseline_path = args.baseline_metrics or _latest([os.path.join(args.models_dir, "*_metrics.json")])
    cim_path = args.cim_metrics or _latest(["cosmic_intelligence_cdm_public_*_metrics.json"])

    if not baseline_path and not cim_path:
        raise SystemExit("No metrics files found. Provide --baseline-metrics and/or --cim-metrics.")

    summaries: list[dict[str, Any]] = []
    if baseline_path:
        data = _load_json(baseline_path)
        if data:
            summaries.append(_summarize_baseline(baseline_path, data))
    if cim_path:
        data = _load_json(cim_path)
        if data:
            summaries.append(_summarize_cim(cim_path, data))

    print(json.dumps({"summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()

