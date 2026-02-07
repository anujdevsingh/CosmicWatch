import argparse
import os

import pandas as pd


def _require_col(df: pd.DataFrame, name: str) -> None:
    if name not in df.columns:
        raise SystemExit(f"Missing required column: {name}")


def _parse_dt_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    parsed = pd.to_datetime(s, errors="coerce", utc=False)
    return parsed


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if "CDM_ID" in df.columns:
        return df.drop_duplicates(subset=["CDM_ID"]).copy()
    return df.drop_duplicates().copy()


def _print_distribution(df: pd.DataFrame, label_col: str, name: str) -> None:
    if label_col not in df.columns:
        print(f"{name}: rows={len(df)}")
        return
    counts = df[label_col].value_counts(dropna=False).sort_index()
    parts = ", ".join(f"{k}:{int(v)}" for k, v in counts.items())
    print(f"{name}: rows={len(df)} {label_col}={{ {parts} }}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-dir", default="datasets/splits")
    ap.add_argument("--time-col", default="CREATED", choices=["CREATED", "TCA", "created_dt", "tca_dt"])
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--label-col", default="pc_risk_class")
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--drop-na-label", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if abs((args.train_frac + args.val_frac + args.test_frac) - 1.0) > 1e-6:
        raise SystemExit("train/val/test fractions must sum to 1.0")
    if args.train_frac <= 0 or args.val_frac <= 0 or args.test_frac <= 0:
        raise SystemExit("train/val/test fractions must be > 0")

    df = pd.read_csv(args.in_csv)
    _require_col(df, args.time_col)

    df = df.copy()
    df["_time"] = _parse_dt_series(df, args.time_col)
    df = df[df["_time"].notna()].copy()
    if args.drop_na_label and args.label_col in df.columns:
        df = df[df[args.label_col].notna()].copy()
    df = df.sort_values("_time").reset_index(drop=True)

    if args.dedupe:
        df = _dedupe(df).sort_values("_time").reset_index(drop=True)

    n = len(df)
    if n < 100:
        raise SystemExit(f"Not enough rows for splitting: {n} (need >= 100)")

    train_end = int(n * args.train_frac)
    val_end = train_end + int(n * args.val_frac)
    val_end = min(val_end, n - 1)

    train_df = df.iloc[:train_end].drop(columns=["_time"])
    val_df = df.iloc[train_end:val_end].drop(columns=["_time"])
    test_df = df.iloc[val_end:].drop(columns=["_time"])

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "cdm_train.csv")
    val_path = os.path.join(args.out_dir, "cdm_val.csv")
    test_path = os.path.join(args.out_dir, "cdm_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    _print_distribution(train_df, args.label_col, "train")
    _print_distribution(val_df, args.label_col, "val")
    _print_distribution(test_df, args.label_col, "test")
    print(f"wrote: {train_path}")
    print(f"wrote: {val_path}")
    print(f"wrote: {test_path}")


if __name__ == "__main__":
    main()
