# CosmicWatch — Space Debris Dashboard + CDM Research Pipeline

CosmicWatch is an educational dashboard for exploring the public space-object population and a set of scripts for building a reproducible Space-Track CDM (`cdm_public`) research dataset.

- Live catalog source: **CelesTrak** (public TLE-derived products)
- Backend: **FastAPI** (Python)
- Frontend: **Next.js (App Router)** (TypeScript)
- Models:
  - **CIM**: a coarse “risk bucket” model for dashboard visualization (not operational Pc)
  - **CDM baseline model**: trained on Space-Track `cdm_public` labels (reproducible metrics)

Read the responsible-use notes in [MODEL_CARD.md](MODEL_CARD.md).

## What’s In The App

Right sidebar panels:
- **System Status**: DB status, data freshness, AI cache status
- **CDM Model**: baseline CDM model metrics + a form to score a CDM-like record
- **Cosmic Intelligence Model (CIM)**: shows checkpoint metadata (accuracy/F1/params)
- **Pc Calculator (CDM)**: paste CDM text and compute probability of collision (Pc) using the conjunction engine
- **Collision Alerts**: “closest pairs” alerts from the current catalog snapshot

## Quick Start (Windows / PowerShell)

### 1) Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend\run.py
```

Backend runs on `http://127.0.0.1:8000`.

### 2) Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend runs on `http://127.0.0.1:3000`.

## Core API Endpoints

### Catalog + Dashboard
- `GET /api/debris?limit=500`
- `GET /api/stats`
- `POST /api/refresh` (forces a CelesTrak refresh)
- `GET /api/data-status`
- `GET /api/collisions?limit=10&sample_size=300&threshold_km=2000`

### Models
- `GET /api/model-info` (CIM checkpoint metadata)
- `GET /api/cdm/model-info` (CDM baseline model info + test/val metrics)
- `POST /api/cdm/predict` (score a CDM-like record)

### Conjunction Engine
- `POST /api/conjunction/from-cdm` (paste CDM text → Pc + geometry)
- `POST /api/conjunction/pair` (compute Pc from two states/covariances)

## CDM Research Pipeline (Space-Track `cdm_public`)

All CDM scripts are under [scripts/](scripts/). The full walkthrough is in [CDM_PIPELINE.md](CDM_PIPELINE.md).

### 1) Configure Space-Track credentials (local only)

Copy `.env.example` → `.env` and set:
- `SPACETRACK_USERNAME`
- `SPACETRACK_PASSWORD`

Never commit `.env`.

### 2) Download CDM public data

```powershell
python scripts\spacetrack_download_cdm_public.py --out-dir data\cdm_public --created-since-days 365 --chunk-days 7 --write-latest
```

### 3) Build dataset CSV (optionally enriched)

Basic dataset:
```powershell
python scripts\build_cdm_public_dataset.py --cdm-dir data\cdm_public --dedupe --out-csv datasets\cdm_public_dataset.csv
```

Enriched dataset (SATCAT + GP) — recommended for better model quality:
```powershell
python scripts\build_cdm_public_dataset.py --cdm-dir data\cdm_public --dedupe --fetch-satcat-gp --id-chunk-size 50 --out-csv datasets\cdm_public_dataset_enriched.csv
```

### 4) Split time-based and train baseline

```powershell
python scripts\split_cdm_dataset.py --in-csv datasets\cdm_public_dataset_enriched.csv --out-dir datasets\splits_enriched_quant --time-col CREATED --label-col pc_quantile_class --dedupe --drop-na-label
python scripts\train_cdm_public_model.py --dataset-csv datasets\cdm_public_dataset_enriched.csv --train-csv datasets\splits_enriched_quant\cdm_train.csv --val-csv datasets\splits_enriched_quant\cdm_val.csv --test-csv datasets\splits_enriched_quant\cdm_test.csv --target pc_quantile_class
```

Compare latest baseline vs CIM training metrics:
```powershell
python scripts\compare_cdm_metrics.py
```

## Configuration

### Backend server
- `PORT` (default 8000)
- `RELOAD` (`true` by default; set `false` for stable logs)
- `LOG_LEVEL` (default `INFO`)
- `BACKEND_CORS_ORIGINS` (default allows localhost:3000)

### Background updates / logging noise
- `COSMICWATCH_BACKGROUND_UPDATES` (default true)
- `COSMICWATCH_REFRESH_INTERVAL_HOURS` (default 2 hours)
- `COSMICWATCH_REFRESH_PROGRESS` (default false) — logs DB refresh progress if true
- `COSMICWATCH_CELESTRAK_PROGRESS` (default false) — logs CelesTrak fetch progress if true

### CIM startup logging
- `COSMICWATCH_CIM_STARTUP_LOGS` (default false) — show checkpoint/accuracy load logs if true

## Notes On CIM “Accuracy / F1”

The CIM metrics shown in the UI come from the loaded checkpoint metadata (`cosmic_intelligence_best.pth`) and are intended as *project-internal* reporting. They are not operational collision-avoidance metrics. See [MODEL_CARD.md](MODEL_CARD.md).

## Model Architecture

This section documents the two main model tracks in this repo:
- **CIM (dashboard)**: coarse risk-bucket model used for visualization and ranking.
- **CDM baseline model (research)**: scikit-learn logistic regression trained on Space-Track `cdm_public` derived labels with reproducible metrics.

### 1) CIM (Cosmic Intelligence Model) — dashboard risk buckets

Implementation: [cosmic_intelligence_model.py](cosmic_intelligence_model.py)

High-level flow:

```text
Incoming object (from DB / CelesTrak transform)
  ├─ altitude, velocity, inclination, size, position (x,y,z), etc.
  ├─ CIM wrapper predicts:
  │    1) risk_level + probabilities (physics/heuristic fallback)
  │    2) uncertainty fields (optionally from neural net if checkpoint loaded)
  └─ Dashboard displays risk buckets + confidence
```

Neural network architecture (16.58M params reported by the checkpoint):

```text
CosmicIntelligenceModel
  Inputs (sequence_length = 10)
    orbital_elements      : [B, T, 6]   (orbital element vector)
    physical_properties   : [B, T, 10]  (size/mass-like proxies, etc.)
    observations          : [B, T, 8]   (sensor/obs placeholders)
    environment           : [B, T, 12]  (environment placeholders)

  Embeddings (Linear → hidden_dim/4 each; hidden_dim=256)
    orbital_embedding        6  → 64
    physical_embedding      10  → 64
    observational_embedding  8  → 64
    environmental_embedding 12  → 64

  Fusion
    concat(4x64) → 256
    feature_fusion: Linear(256→256) + LayerNorm + SiLU + Dropout

  Physics Engine (physics-informed residual)
    learnable constants: μ_earth, J2, earth_radius
    orbital_energy_net    : 256 → 128 → 256 (SiLU + LayerNorm)
    angular_momentum_net  : 256 → 128 → 256 (SiLU + LayerNorm)
    atmospheric_drag_net  : 256 → 128 → 256 (SiLU + LayerNorm)
    perturbation_processor: MultiHeadAttention(embed=256, heads=8)
    residual: x + 0.1 * perturbed_features

  Transformer stack (12 layers)
    each layer = CosmicAttentionModule
      temporal_attention: MultiHeadAttention(embed=256, heads=16)
      spatial_attention : MultiHeadAttention(embed=256, heads=8)
      ffn              : 256 → 1024 → 256 (SiLU + Dropout)
      residual + LayerNorm around each block

  Heads (multi-task, even if dashboard mostly uses risk bucket)
    risk_classifier        : 256 → 128 → 4 (LOW/MEDIUM/HIGH/CRITICAL)
    trajectory_predictor   : 256 → 128 → (6 * horizon)
    anomaly_detector       : 256 → 64  → 1 (sigmoid)
    collision_assessor     : (256*2) → 256 → 1 (sigmoid)
    uncertainty heads      : 256 → 64 → 1 (softplus) for epistemic/aleatoric
```

Important note about what “accuracy” means here:
- The checkpoint-reported accuracy/F1 are model metadata bundled with `cosmic_intelligence_best.pth`.
- They are not a validated Pc (probability of collision) metric.
- For operational-grade conjunction risk you would evaluate against CDM/covariance Pc references (see the conjunction engine + CDM pipeline).

### 2) CDM baseline model (Space-Track `cdm_public`) — reproducible research model

Implementation: [train_cdm_public_model.py](scripts/train_cdm_public_model.py)

Purpose:
- Predict CDM-derived labels like `pc_quantile_class` using features from the CDM record (and optional enrichment).
- Produce reproducible, stored metrics: confusion matrix + classification report in `models/cdm_public/*_metrics.json`.

Pipeline diagram:

```text
CDM rows (datasets/cdm_public_dataset*.csv)
  ├─ Feature engineering (utils/cdm_features.py)
  │    MIN_RNG, log_min_rng, hours_to_tca
  │    delta_inclination, delta_mean_motion
  │    excl_vol_sum
  │    categorical: SAT1/SAT2 object type + RCS categories (if present)
  ├─ Preprocessing (sklearn ColumnTransformer)
  │    numeric:
  │      SimpleImputer(median) → StandardScaler
  │    categorical:
  │      SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown=ignore)
  └─ Classifier
       LogisticRegression(max_iter=4000, class_weight=balanced)
```

Why this baseline is useful:
- It is fast, interpretable, and usually hard to beat without better features/labels.
- It makes it obvious whether a bigger model is actually adding value.

## Troubleshooting

### Hydration mismatch warning in browser
Some extensions inject DOM attributes before React hydrates. The main Dashboard is rendered client-only to avoid hydration mismatch warnings.

### Collision Alerts panel is empty
Try calling:
`/api/collisions?limit=10&sample_size=500&threshold_km=3000`
to increase sampling and distance threshold.

## Repository Map

Key folders/files:
- [backend/main.py](backend/main.py): FastAPI API (catalog, collisions, model-info, CDM endpoints)
- [frontend/src/app/Dashboard.tsx](frontend/src/app/Dashboard.tsx): main UI
- [cosmic_intelligence_model.py](cosmic_intelligence_model.py): CIM implementation + checkpoint loader
- [conjunction/](conjunction/): Pc engine + CDM parsing utilities
- [scripts/](scripts/): Space-Track download, dataset build, split, train, compare
- [CDM_PIPELINE.md](CDM_PIPELINE.md): reproducible CDM research pipeline
- [DASHBOARD_ARCHITECTURE.md](DASHBOARD_ARCHITECTURE.md): app architecture notes

## Tests

These repo tests use Python’s built-in `unittest`:

```powershell
python -m unittest discover -s conjunction\\tests -p "test_*.py" -v
```

## Release Checklist (Before You Commit / Publish)

### Do not commit
- `.env` (Space-Track credentials)
- `datasets/` (generated datasets)
- `models/` (trained models and metrics)
- `*.pth` (PyTorch checkpoints)
- `*.pkl` (scikit-learn pipelines)
- `*_metrics.json` (generated evaluation artifacts)
- Local DB files like `space_debris.db` (if present)

### Confirm `.gitignore` covers these artifacts
- Verify these folders/files are ignored:
  - `datasets/`
  - `models/`
  - `.env`
  - `*.pth`, `*.pkl`, `*_metrics.json`

### Sanity checks before commit
- Backend boots cleanly:
  - `python backend/run.py`
- Frontend builds:
  - `cd frontend && npm run build`
- CDM endpoints respond:
  - `GET /api/cdm/model-info`
  - `POST /api/cdm/predict`
- No secrets in git diff:
  - check that no Space-Track username/password appears anywhere
- Optional: run unit tests:
  - `python -m unittest discover -s conjunction\\tests -p "test_*.py" -v`

## License

MIT — see [LICENSE](LICENSE).
