# CosmicWatch Dashboard Architecture (React + FastAPI)

CosmicWatch uses a **production-style split**:

1. **Next.js frontend (React)** — renders the UI in the browser.
2. **FastAPI backend (Python)** — owns data refresh, database access, and CIM inference.

The backend loads the **Cosmic Intelligence Model (CIM)** from `cosmic_intelligence_model.py` and, if present, loads trained weights from `cosmic_intelligence_*.pth`.

---

## System Overview

**Entry points**
- Backend API: `python backend/run.py` → `backend/main.py`
- Frontend UI: `npm run dev` in `frontend/` → `frontend/src/app/*`

**Data flow**
```
Browser (Next.js UI)
  ⇅ (HTTP fetch)
FastAPI server (backend/main.py)
  ├─ reads SQLite
  ├─ refreshes catalog data (CelesTrak → SQLite) via API endpoints
  ├─ runs CIM predictions (on-demand, small batches) and/or uses cached AI columns
  ├─ background refresh (optional) keeps catalog data fresh
  ├─ background AI cache job can fill ai_risk_level/ai_confidence for all objects
  └─ returns JSON to frontend
```

**Key files**
- `backend/main.py` — FastAPI routes for debris/stats/collisions and conjunction endpoints.
- `backend/run.py` — starts uvicorn for the API.
- `frontend/src/app/page.tsx` + `frontend/src/app/Dashboard.tsx` — page shell and main dashboard.
- `frontend/src/components/*` — UI widgets (Earth3D, alerts, stats cards, Pc calculator).
- `frontend/src/components/SystemStatusPanel.tsx` — API/DB/data-freshness/AI-cache status and controls.
- `frontend/README.md` — frontend-specific run instructions and required env vars.
- `utils/database.py` — catalog refresh, schema/migrations, SQLAlchemy models.
- `utils/background_updater.py` — optional scheduled refresh when data is stale.

**Strengths**
- Cleaner separation of concerns (UI vs data/model).
- Better fit if you want a “real web app” UX, deployable separately.

**Tradeoffs**
- More moving parts (CORS, env vars, two dev servers).
- Backend must have all dependencies that CIM/conjunction code needs.

---

## Do they use the same model?

Yes — the system uses the CIM wrapper from `cosmic_intelligence_model.py`:
- FastAPI imports `get_cosmic_intelligence_model` and serves predictions/metrics via API endpoints.

However:
- If no checkpoint file (`cosmic_intelligence_best.pth`, `cosmic_intelligence_improved.pth`, etc.) is found, CIM runs in “physics-based fallback” mode (still produces risk levels, but not from trained neural weights).
- The frontend (React/Next.js) does **not** run the model in the browser; it only displays what the FastAPI backend returns.

---

## Important: Catalog Data vs CDM Training Data

CosmicWatch has **two separate data pipelines**:

1. **Catalog / debris visualization (runtime)**  
   - Source: CelesTrak catalog download.  
   - Storage: `space_debris.db` (SQLite).  
   - Used by: the running dashboard UI (Earth markers, collision alerts, stats).

2. **Conjunction CDM dataset (offline training/evaluation)**  
   - Source: Space-Track `cdm_public` (public conjunction records).  
   - Storage: JSON/CSV files under `data/` and `datasets/`.  
   - Used by: training scripts under `scripts/` to produce honest evaluation metrics.

The CDM scripts do **not** overwrite `space_debris.db`. They generate separate datasets for research.

---

## Current “Production-Style” Runtime Behavior

**Auto refresh (optional)**  
- The backend can run a background updater that refreshes catalog data when it becomes stale.  
- Config via env:\n  - `COSMICWATCH_BACKGROUND_UPDATES=true|false`\n  - `COSMICWATCH_REFRESH_INTERVAL_HOURS=2`

**AI cache (recommended)**  
- The backend can fill `ai_risk_level/ai_confidence/ai_last_predicted/ai_enhanced` in the DB via a background job.\n- This keeps `/api/debris` fast while still serving AI-labeled risk levels.
