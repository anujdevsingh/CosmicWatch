import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cosmic_intelligence_model import CosmicConfig, CosmicIntelligenceModel


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _map_rcs(value: object) -> float:
    v = str(value).strip().upper()
    return {"SMALL": 0.0, "MEDIUM": 1.0, "LARGE": 2.0}.get(v, 0.0)


def _map_obj_type(value: object) -> float:
    v = str(value).strip().upper()
    return {"DEBRIS": 0.0, "ROCKET BODY": 1.0, "PAYLOAD": 2.0, "UNKNOWN": 3.0}.get(v, 3.0)


def build_features(df: pd.DataFrame) -> np.ndarray:
    work = df.copy()

    work["sat1_rcs_num"] = work.get("SAT1_RCS", "").map(_map_rcs) if "SAT1_RCS" in work.columns else 0.0
    work["sat2_rcs_num"] = work.get("SAT2_RCS", "").map(_map_rcs) if "SAT2_RCS" in work.columns else 0.0
    work["sat1_obj_num"] = work.get("SAT1_OBJECT_TYPE", "").map(_map_obj_type) if "SAT1_OBJECT_TYPE" in work.columns else 3.0
    work["sat2_obj_num"] = work.get("SAT2_OBJECT_TYPE", "").map(_map_obj_type) if "SAT2_OBJECT_TYPE" in work.columns else 3.0

    gp_cols = [
        "sat1_gp_INCLINATION",
        "sat1_gp_ECCENTRICITY",
        "sat1_gp_MEAN_MOTION",
        "sat1_gp_BSTAR",
        "sat2_gp_INCLINATION",
        "sat2_gp_ECCENTRICITY",
        "sat2_gp_MEAN_MOTION",
        "sat2_gp_BSTAR",
        "sat1_gp_APOGEE",
        "sat1_gp_PERIGEE",
        "sat2_gp_APOGEE",
        "sat2_gp_PERIGEE",
    ]
    for col in gp_cols:
        if col in work.columns:
            work[col] = _to_float(work[col])
        else:
            work[col] = 0.0

    min_rng = _to_float(work["MIN_RNG"]) if "MIN_RNG" in work.columns else pd.Series(0.0, index=work.index)
    hours_to_tca = _to_float(work["hours_to_tca"]) if "hours_to_tca" in work.columns else pd.Series(0.0, index=work.index)
    excl1 = _to_float(work["SAT_1_EXCL_VOL"]) if "SAT_1_EXCL_VOL" in work.columns else pd.Series(0.0, index=work.index)
    excl2 = _to_float(work["SAT_2_EXCL_VOL"]) if "SAT_2_EXCL_VOL" in work.columns else pd.Series(0.0, index=work.index)
    incl1 = _to_float(work["sat1_gp_INCLINATION"])
    incl2 = _to_float(work["sat2_gp_INCLINATION"])
    mm1 = _to_float(work["sat1_gp_MEAN_MOTION"])
    mm2 = _to_float(work["sat2_gp_MEAN_MOTION"])
    bst1 = _to_float(work["sat1_gp_BSTAR"])
    bst2 = _to_float(work["sat2_gp_BSTAR"])
    ecc1 = _to_float(work["sat1_gp_ECCENTRICITY"])
    ecc2 = _to_float(work["sat2_gp_ECCENTRICITY"])
    apo1 = _to_float(work["sat1_gp_APOGEE"])
    apo2 = _to_float(work["sat2_gp_APOGEE"])
    per1 = _to_float(work["sat1_gp_PERIGEE"])
    per2 = _to_float(work["sat2_gp_PERIGEE"])

    feat = np.zeros((len(work), 36), dtype=np.float32)

    feat[:, 0] = work["sat1_gp_INCLINATION"].to_numpy()
    feat[:, 1] = work["sat1_gp_ECCENTRICITY"].to_numpy()
    feat[:, 2] = work["sat1_gp_MEAN_MOTION"].to_numpy()
    feat[:, 3] = work["sat1_gp_BSTAR"].to_numpy()
    feat[:, 4] = work["sat2_gp_INCLINATION"].to_numpy()
    feat[:, 5] = work["sat2_gp_ECCENTRICITY"].to_numpy()

    feat[:, 6] = work["sat2_gp_MEAN_MOTION"].to_numpy()
    feat[:, 7] = work["sat2_gp_BSTAR"].to_numpy()
    feat[:, 8] = work["sat1_gp_APOGEE"].to_numpy()
    feat[:, 9] = work["sat1_gp_PERIGEE"].to_numpy()
    feat[:, 10] = work["sat2_gp_APOGEE"].to_numpy()
    feat[:, 11] = work["sat2_gp_PERIGEE"].to_numpy()
    feat[:, 12] = work["sat1_rcs_num"].to_numpy(dtype=np.float32)
    feat[:, 13] = work["sat2_rcs_num"].to_numpy(dtype=np.float32)
    feat[:, 14] = work["sat1_obj_num"].to_numpy(dtype=np.float32)
    feat[:, 15] = work["sat2_obj_num"].to_numpy(dtype=np.float32)

    feat[:, 16] = min_rng.to_numpy(dtype=np.float32)
    feat[:, 17] = np.log10(np.clip(feat[:, 16], 1.0, None))
    feat[:, 18] = hours_to_tca.to_numpy(dtype=np.float32)
    feat[:, 19] = excl1.to_numpy(dtype=np.float32)
    feat[:, 20] = excl2.to_numpy(dtype=np.float32)
    feat[:, 21] = np.abs(incl1.to_numpy(dtype=np.float32) - incl2.to_numpy(dtype=np.float32))
    feat[:, 22] = np.abs(mm1.to_numpy(dtype=np.float32) - mm2.to_numpy(dtype=np.float32))
    feat[:, 23] = np.abs(bst1.to_numpy(dtype=np.float32) - bst2.to_numpy(dtype=np.float32))
    feat[:, 24] = np.abs(ecc1.to_numpy(dtype=np.float32) - ecc2.to_numpy(dtype=np.float32))
    feat[:, 25] = np.abs(apo1.to_numpy(dtype=np.float32) - apo2.to_numpy(dtype=np.float32))
    feat[:, 26] = np.abs(per1.to_numpy(dtype=np.float32) - per2.to_numpy(dtype=np.float32))
    feat[:, 27] = (feat[:, 19] + feat[:, 20]).astype(np.float32)
    feat[:, 28] = (feat[:, 16] / (1.0 + feat[:, 27])).astype(np.float32)
    feat[:, 29] = np.abs(feat[:, 12] - feat[:, 13]).astype(np.float32)
    feat[:, 30] = np.abs(feat[:, 14] - feat[:, 15]).astype(np.float32)
    feat[:, 31] = np.clip(feat[:, 18], 0.0, 365.0).astype(np.float32)
    feat[:, 32] = np.clip(feat[:, 16], 0.0, 50000.0).astype(np.float32)
    feat[:, 33] = np.clip(feat[:, 17], -3.0, 6.0).astype(np.float32)
    feat[:, 34] = np.clip(feat[:, 21], 0.0, 180.0).astype(np.float32)
    feat[:, 35] = 1.0

    return feat


def make_batch_data(config: CosmicConfig, features: torch.Tensor) -> dict:
    batch_size = features.shape[0]
    seq_len = config.sequence_length
    orbital_elements = features[:, :6].unsqueeze(1).repeat(1, seq_len, 1)
    physical_properties = features[:, 6:16].unsqueeze(1).repeat(1, seq_len, 1)
    observations = features[:, 16:24].unsqueeze(1).repeat(1, seq_len, 1)
    environment = features[:, 24:36].unsqueeze(1).repeat(1, seq_len, 1)
    return {
        "orbital_elements": orbital_elements,
        "physical_properties": physical_properties,
        "observations": observations,
        "environment": environment,
    }


def train_one_epoch(model: CosmicIntelligenceModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    criterion = model.risk_criterion
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        batch = make_batch_data(model.config, x)
        out = model(batch, task="risk_assessment")["risk_logits"]
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        pred = torch.argmax(out, dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return total_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model: CosmicIntelligenceModel, loader: DataLoader, device: torch.device) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    criterion = model.risk_criterion
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        batch = make_batch_data(model.config, x)
        out = model(batch, task="risk_assessment")["risk_logits"]
        loss = criterion(out, y)
        total_loss += float(loss.item())
        pred = torch.argmax(out, dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    return total_loss / max(len(loader), 1), correct / max(total, 1), y_true, y_pred


def train_one_epoch_regression(
    model: CosmicIntelligenceModel,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    head.train()
    total_loss = 0.0
    total_abs_err = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()
        batch = make_batch_data(model.config, x)
        seq = model(batch, task="risk_assessment")["sequence_representation"]
        pred = head(seq).squeeze(-1)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        total_abs_err += float(torch.abs(pred - y).sum().item())
        total += int(y.numel())
    mae = total_abs_err / max(total, 1)
    return total_loss / max(len(loader), 1), mae


@torch.no_grad()
def eval_epoch_regression(
    model: CosmicIntelligenceModel,
    head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    head.eval()
    total_loss = 0.0
    total_abs_err = 0.0
    total_sq_err = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()
        batch = make_batch_data(model.config, x)
        seq = model(batch, task="risk_assessment")["sequence_representation"]
        pred = head(seq).squeeze(-1)
        loss = criterion(pred, y)
        total_loss += float(loss.item())
        diff = pred - y
        total_abs_err += float(torch.abs(diff).sum().item())
        total_sq_err += float((diff * diff).sum().item())
        total += int(y.numel())
    mae = total_abs_err / max(total, 1)
    rmse = float(np.sqrt(total_sq_err / max(total, 1)))
    return total_loss / max(len(loader), 1), mae, rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--mode", choices=["classification", "regression"], default="classification")
    parser.add_argument("--label-col", default="pc_risk_class")
    parser.add_argument("--target-col", default="pc_log10")
    parser.add_argument("--out-pth", default="cosmic_intelligence_cdm_public.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.train_csv and args.test_csv:
        df_train = pd.read_csv(args.train_csv)
        df_test = pd.read_csv(args.test_csv)
        df_val = pd.read_csv(args.val_csv) if args.val_csv else None
    else:
        df_all = pd.read_csv(args.dataset_csv)
        df_train, df_test = train_test_split(
            df_all,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=df_all[args.label_col] if args.label_col in df_all.columns and df_all[args.label_col].nunique() > 1 else None,
        )
        df_val = None

    def clean_classification(df_in: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if args.label_col not in df_in.columns:
            raise SystemExit(f"Missing label column: {args.label_col}")
        y_raw = pd.to_numeric(df_in[args.label_col], errors="coerce")
        d = df_in[y_raw.notna()].copy()
        d[args.label_col] = pd.to_numeric(d[args.label_col], errors="coerce").astype(int)
        d = d[d[args.label_col].between(0, 3)]
        if d.empty:
            raise SystemExit("No usable rows after filtering labels")
        return build_features(d), d[args.label_col].to_numpy(dtype=np.int64)

    def clean_regression(df_in: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if args.target_col not in df_in.columns:
            raise SystemExit(f"Missing target column: {args.target_col}")
        y_raw = pd.to_numeric(df_in[args.target_col], errors="coerce")
        d = df_in[y_raw.notna()].copy()
        y = pd.to_numeric(d[args.target_col], errors="coerce").astype(float).to_numpy(dtype=np.float32)
        if len(d) == 0:
            raise SystemExit("No usable rows after filtering regression targets")
        return build_features(d), y

    if args.mode == "regression":
        x_train_np, y_train_np = clean_regression(df_train)
        x_test_np, y_test_np = clean_regression(df_test)
        x_val_np = None
        y_val_np = None
        if df_val is not None:
            x_val_np, y_val_np = clean_regression(df_val)
    else:
        x_train_np, y_train_np = clean_classification(df_train)
        x_test_np, y_test_np = clean_classification(df_test)
        x_val_np = None
        y_val_np = None
        if df_val is not None:
            x_val_np, y_val_np = clean_classification(df_val)

    if len(y_train_np) < 500:
        raise SystemExit("Train split too small for meaningful evaluation (need >= 500 rows)")

    if args.mode != "regression":
        unique, counts = np.unique(y_train_np, return_counts=True)
        if len(unique) < 2:
            raise SystemExit("Need at least 2 label classes to train")
        if int(counts.min()) < 10:
            raise SystemExit("Need at least 10 samples in each class in the TRAIN split")

    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)
    x_test = torch.from_numpy(x_test_np)
    y_test = torch.from_numpy(y_test_np)
    x_val = torch.from_numpy(x_val_np) if x_val_np is not None else None
    y_val = torch.from_numpy(y_val_np) if y_val_np is not None else None

    train_mean = x_train.float().mean(dim=0, keepdim=True)
    train_std = x_train.float().std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train.float() - train_mean) / train_std
    x_test = (x_test.float() - train_mean) / train_std
    if x_val is not None:
        x_val = (x_val.float() - train_mean) / train_std

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False) if x_val is not None and y_val is not None else None

    config = CosmicConfig()
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.max_epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CosmicIntelligenceModel(config).to(device)
    weights: list[float] | None = None
    head: nn.Module | None = None
    if args.mode == "regression":
        head = nn.Linear(int(model.config.hidden_dim), 1).to(device)
        criterion_reg = nn.SmoothL1Loss(beta=0.5)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-5)
    else:
        class_counts = np.bincount(y_train_np, minlength=4).astype(np.float32)
        weights_np = (class_counts.sum() / np.clip(class_counts, 1.0, None))
        weights_np = weights_np / weights_np.mean()
        weights = weights_np.tolist()
        model.risk_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights_np, dtype=torch.float32, device=device),
            label_smoothing=max(0.0, min(float(args.label_smoothing), 0.2)),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_bal_acc = -1.0
    best_state = None
    best_head_state = None
    epochs_no_improve = 0
    patience = max(1, int(args.early_stop_patience))
    min_delta = max(0.0, float(args.early_stop_min_delta))
    for epoch in range(args.epochs):
        if args.mode == "regression":
            assert head is not None
            tr_loss, tr_mae = train_one_epoch_regression(model, head, train_loader, optimizer, criterion_reg, device)
            if val_loader is not None:
                va_loss, va_mae, va_rmse = eval_epoch_regression(model, head, val_loader, criterion_reg, device)
                score = -va_rmse
                msg = f"val_loss={va_loss:.4f} val_mae={va_mae:.4f} val_rmse={va_rmse:.4f}"
            else:
                te_loss, te_mae, te_rmse = eval_epoch_regression(model, head, test_loader, criterion_reg, device)
                score = -te_rmse
                msg = f"test_loss={te_loss:.4f} test_mae={te_mae:.4f} test_rmse={te_rmse:.4f}"
            tr_acc = tr_mae
        else:
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
            if val_loader is not None:
                va_loss, va_acc, va_true, va_pred = eval_epoch(model, val_loader, device)
                va_bal = float(balanced_accuracy_score(va_true, va_pred))
                score = va_bal
                msg = f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_bal_acc={va_bal:.4f}"
            else:
                te_loss, te_acc, te_true, te_pred = eval_epoch(model, test_loader, device)
                te_bal = float(balanced_accuracy_score(te_true, te_pred))
                score = te_bal
                msg = f"test_loss={te_loss:.4f} test_acc={te_acc:.4f} test_bal_acc={te_bal:.4f}"

        if score > (best_bal_acc + min_delta):
            best_bal_acc = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if head is not None:
                best_head_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if args.mode == "regression":
            print(f"epoch={epoch+1} train_loss={tr_loss:.4f} train_mae={tr_acc:.4f} {msg}")
        else:
            print(f"epoch={epoch+1} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} {msg}")
        if epochs_no_improve >= patience:
            print(f"early_stop: patience={patience} best_score={best_bal_acc:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    if head is not None and best_head_state is not None:
        head.load_state_dict(best_head_state, strict=True)

    if args.mode == "regression":
        assert head is not None
        te_loss, te_mae, te_rmse = eval_epoch_regression(model, head, test_loader, criterion_reg, device)
        report = None
        matrix = None
        test_acc = float("nan")
        bal_acc = float("nan")
    else:
        _, test_acc, y_true, y_pred = eval_epoch(model, test_loader, device)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_true, y_pred).tolist()
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_pth = args.out_pth
    if out_pth.endswith(".pth"):
        out_pth = out_pth[:-4] + f"_{stamp}.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "dataset": os.path.basename(args.dataset_csv),
            "mode": args.mode,
            "label_col": args.label_col if args.mode != "regression" else None,
            "target_col": args.target_col if args.mode == "regression" else None,
            "test_acc": float(test_acc) if args.mode != "regression" else None,
            "balanced_acc": bal_acc if args.mode != "regression" else None,
            "class_weights": weights,
            "feature_norm": {"mean": train_mean.squeeze(0).tolist(), "std": train_std.squeeze(0).tolist()},
            "head_state_dict": head.state_dict() if head is not None else None,
            "metrics": {"classification_report": report, "confusion_matrix": matrix, "test_rmse": te_rmse if args.mode == "regression" else None, "test_mae": te_mae if args.mode == "regression" else None},
        },
        out_pth,
    )

    metrics_path = os.path.splitext(out_pth)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "out_pth": out_pth,
                "dataset_csv": args.dataset_csv,
                "mode": args.mode,
                "label_col": args.label_col if args.mode != "regression" else None,
                "target_col": args.target_col if args.mode == "regression" else None,
                "test_acc": float(test_acc) if args.mode != "regression" else None,
                "balanced_acc": bal_acc if args.mode != "regression" else None,
                "class_weights": weights,
                "feature_norm": {"mean": train_mean.squeeze(0).tolist(), "std": train_std.squeeze(0).tolist()},
                "classification_report": report,
                "confusion_matrix": matrix,
                "test_rmse": te_rmse if args.mode == "regression" else None,
                "test_mae": te_mae if args.mode == "regression" else None,
            },
            f,
            indent=2,
        )

    print(f"saved_weights={out_pth}")
    print(f"saved_metrics={metrics_path}")


if __name__ == "__main__":
    main()
