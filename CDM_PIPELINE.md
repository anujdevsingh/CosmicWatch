# CDM Public Pipeline (Space-Track) — Research Dataset

This pipeline is for **offline research/training/evaluation**. It does **not** modify the runtime catalog database (`space_debris.db`) used by the dashboard.

## What “cdm_public” is

- Space-Track provides a public conjunction dataset via `basicspacedata/class/cdm_public`.
- It is a **subset**. Your Space-Track UI may show more events depending on account permissions/time filters.

## 1) Download more public CDM data (recommended way)

Use the downloader script:
- [spacetrack_download_cdm_public.py](file:///e:/projects/SpaceDebrisDashboard/SpaceDebrisDashboard/CosmicWatch/scripts/spacetrack_download_cdm_public.py)

It supports:
- date ranges (`--created-start`, `--created-end`)
- chunked downloads (`--chunk-days`)
- retry/backoff and dedupe by `CDM_ID`

Example (one year of public CDM in weekly chunks):
```bash
python scripts/spacetrack_download_cdm_public.py --out-dir data/cdm_public --created-start 2025-01-01 --created-end 2026-01-01 --chunk-days 7 --write-latest
```

### Backfill until the earliest available public record
If you want to “get everything available in cdm_public”, use backfill mode. It walks backwards in time in chunks and stops after repeated empty chunks:
```bash
python scripts/spacetrack_download_cdm_public.py --out-dir data/cdm_public --backfill-until-empty --chunk-days 7 --backfill-max-years 15 --write-latest
```

### Show what you already downloaded (local summary)
```bash
python scripts/spacetrack_download_cdm_public.py --out-dir data/cdm_public --summary
```

### Preview planned ranges (dry-run)
```bash
python scripts/spacetrack_download_cdm_public.py --out-dir data/cdm_public --created-since-days 365 --chunk-days 7 --dry-run
```

## 2) Build a training dataset CSV

Convert downloaded JSON into a single CSV:
- [build_cdm_public_dataset.py](file:///e:/projects/SpaceDebrisDashboard/SpaceDebrisDashboard/CosmicWatch/scripts/build_cdm_public_dataset.py)

It supports:
- `--cdm-dir` (read all JSON files in a directory)
- optional dedupe (`--dedupe`)
- optional SATCAT/GP enrichment (requires Space-Track credentials)

Example:
```bash
python scripts/build_cdm_public_dataset.py --cdm-dir data/cdm_public --dedupe --out-csv datasets/cdm_public_dataset.csv
```

### Recommended: build an enriched dataset (better model accuracy)
This merges extra satellite metadata and orbital parameters from Space-Track (requires credentials in `.env`):
```bash
python scripts/build_cdm_public_dataset.py --cdm-dir data/cdm_public --dedupe --fetch-satcat-gp --out-csv datasets/cdm_public_dataset_enriched.csv
```

## 2.5) Split into train/val/test (recommended)

Use a **time-based** split (avoids leakage across repeated updates of the same event):
- [split_cdm_dataset.py](file:///e:/projects/SpaceDebrisDashboard/SpaceDebrisDashboard/CosmicWatch/scripts/split_cdm_dataset.py)

Example:
```bash
python scripts/split_cdm_dataset.py --in-csv datasets/cdm_public_dataset.csv --out-dir datasets/splits --time-col CREATED --dedupe
```

This writes:
- `datasets/splits/cdm_train.csv`\n- `datasets/splits/cdm_val.csv`\n- `datasets/splits/cdm_test.csv`

## 3) Train baselines + CIM on CDM labels

Baseline (logistic regression) — good benchmark:
- [train_cdm_public_model.py](file:///e:/projects/SpaceDebrisDashboard/SpaceDebrisDashboard/CosmicWatch/scripts/train_cdm_public_model.py)

```bash
python scripts/train_cdm_public_model.py --dataset-csv datasets/cdm_public_dataset.csv --target pc_risk_class
```

Baseline using explicit splits:
```bash
python scripts/train_cdm_public_model.py --dataset-csv datasets/cdm_public_dataset.csv --train-csv datasets/splits/cdm_train.csv --val-csv datasets/splits/cdm_val.csv --test-csv datasets/splits/cdm_test.csv --target pc_risk_class
```

### Recommended target for public CDM: pc_quantile_class
`pc_risk_class` can become degenerate on some `cdm_public` subsets (ex: mostly emergency-reportable events).\n\nFor a meaningful 4-class task, use `pc_quantile_class` (computed from PC quartiles, only where PC exists):\n\n```bash
python scripts/split_cdm_dataset.py --in-csv datasets/cdm_public_dataset.csv --out-dir datasets/splits_quant --time-col CREATED --label-col pc_quantile_class --dedupe --drop-na-label
python scripts/train_cdm_public_model.py --dataset-csv datasets/cdm_public_dataset.csv --train-csv datasets/splits_quant/cdm_train.csv --val-csv datasets/splits_quant/cdm_val.csv --test-csv datasets/splits_quant/cdm_test.csv --target pc_quantile_class
```\n
CIM training on the CDM-derived risk label:
- [train_cim_on_cdm_public.py](file:///e:/projects/SpaceDebrisDashboard/SpaceDebrisDashboard/CosmicWatch/scripts/train_cim_on_cdm_public.py)

```bash
python scripts/train_cim_on_cdm_public.py --dataset-csv datasets/cdm_public_dataset.csv --label-col pc_risk_class
```

CIM training using explicit splits (recommended):
```bash
python scripts/train_cim_on_cdm_public.py --dataset-csv datasets/cdm_public_dataset.csv --train-csv datasets/splits/cdm_train.csv --val-csv datasets/splits/cdm_val.csv --test-csv datasets/splits/cdm_test.csv --label-col pc_risk_class
```

Full CIM training command (PowerShell one-liner):
```bash
python scripts/train_cim_on_cdm_public.py --dataset-csv datasets/cdm_public_dataset.csv --train-csv datasets/splits_quant/cdm_train.csv --val-csv datasets/splits_quant/cdm_val.csv --test-csv datasets/splits_quant/cdm_test.csv --label-col pc_quantile_class --epochs 50 --batch-size 64 --lr 5e-5 --label-smoothing 0.05
```

Notes:
- The training script enforces minimum dataset size and minimum samples per class so metrics are meaningful.
- It uses class-weighted loss and reports balanced accuracy.

## How this connects to the dashboard

- The dashboard shows the **latest evaluation metrics file** (if present) via `/api/model-info`.
- Runtime risk displayed for debris objects comes from `space_debris.db` (catalog) plus optional cached `ai_*` fields.
