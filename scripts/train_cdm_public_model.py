import argparse
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils.cdm_features import apply_cdm_derived_features


def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    work = apply_cdm_derived_features(df)

    categorical = []
    for col in (
        "SAT1_OBJECT_TYPE",
        "SAT2_OBJECT_TYPE",
        "SAT1_RCS",
        "SAT2_RCS",
        "sat1_OBJECT_TYPE",
        "sat2_OBJECT_TYPE",
        "sat1_RCS_SIZE",
        "sat2_RCS_SIZE",
    ):
        if col in work.columns:
            categorical.append(col)

    numeric = []
    for col in ("MIN_RNG", "log_min_rng", "hours_to_tca"):
        if col in work.columns:
            numeric.append(col)

    for col in ("delta_inclination", "delta_mean_motion", "excl_vol_sum"):
        if col in work.columns:
            numeric.append(col)

    return work, numeric, categorical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument(
        "--target",
        choices=["emergency_reportable", "pc_bucket_high", "pc_risk_class", "pc_quantile_class", "pc_present"],
        default="emergency_reportable",
    )
    parser.add_argument("--out-dir", default="models/cdm_public")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_csv and args.test_csv:
        df_train_raw = pd.read_csv(args.train_csv)
        df_test_raw = pd.read_csv(args.test_csv)
        df_val_raw = pd.read_csv(args.val_csv) if args.val_csv else None
    else:
        df_all = pd.read_csv(args.dataset_csv)
        if args.target not in df_all.columns:
            raise SystemExit(f"Target column not found: {args.target}")
        if len(df_all) < 100:
            raise SystemExit("Dataset too small for meaningful evaluation (need >= 100 rows)")
        df_train_raw, df_test_raw = train_test_split(
            df_all,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=df_all[args.target] if df_all[args.target].nunique() > 1 else None,
        )
        df_val_raw = None

    for df_part in [p for p in [df_train_raw, df_val_raw, df_test_raw] if p is not None]:
        if args.target not in df_part.columns:
            raise SystemExit(f"Target column not found: {args.target}")

    df_train, numeric_cols, categorical_cols = _prepare_features(df_train_raw)
    df_test, _, _ = _prepare_features(df_test_raw)
    df_val = None
    if df_val_raw is not None:
        df_val, _, _ = _prepare_features(df_val_raw)

    drop_cols = {args.target}
    if "PC" in df_train.columns:
        drop_cols.add("PC")

    def to_x_y(df_in: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        y_raw = pd.to_numeric(df_in[args.target], errors="coerce")
        mask = y_raw.notna()
        y_out = y_raw[mask].astype(int)
        x_out = df_in.drop(columns=[c for c in drop_cols if c in df_in.columns], errors="ignore")
        x_out = x_out[[c for c in numeric_cols + categorical_cols if c in x_out.columns]]
        x_out = x_out.loc[mask].copy()
        if x_out.empty:
            raise SystemExit("No features available to train on")
        return x_out, y_out

    X_train, y_train = to_x_y(df_train)
    X_test, y_test = to_x_y(df_test)
    X_val = None
    y_val = None
    if df_val is not None:
        X_val, y_val = to_x_y(df_val)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=4000, class_weight="balanced")
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred).tolist()
    val_report = None
    val_matrix = None
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        val_report = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
        val_matrix = confusion_matrix(y_val, y_val_pred).tolist()

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = os.path.join(args.out_dir, f"{args.target}_{stamp}.pkl")
    metrics_path = os.path.join(args.out_dir, f"{args.target}_{stamp}_metrics.json")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target": args.target,
                "n_rows_train": int(len(df_train)),
                "n_rows_val": int(len(df_val)) if df_val is not None else 0,
                "n_rows_test": int(len(df_test)),
                "features_numeric": numeric_cols,
                "features_categorical": categorical_cols,
                "test_classification_report": report,
                "test_confusion_matrix": matrix,
                "val_classification_report": val_report,
                "val_confusion_matrix": val_matrix,
            },
            f,
            indent=2,
        )

    print(f"Saved model -> {model_path}")
    print(f"Saved metrics -> {metrics_path}")
    print(json.dumps(report, indent=2)[:1200])


if __name__ == "__main__":
    main()
