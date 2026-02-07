"""
FastAPI Backend for Space Debris Dashboard
Serves debris data with REAL Cosmic Intelligence Model predictions
"""

import sys
import os
import math
import logging
import threading

# Add parent directory to path to import CIM model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from conjunction.cdm import parse_cdm_event
from conjunction.frames import combined_pos_cov_encounter_plane
from conjunction.pc import pc_circle_grid
from utils.cdm_model import load_latest_baseline_model, predict_baseline

# Import the REAL Cosmic Intelligence Model
try:
    from cosmic_intelligence_model import get_cosmic_intelligence_model
    CIM_AVAILABLE = True
    logging.getLogger(__name__).info("Cosmic Intelligence Model imported successfully")
except Exception as e:
    logging.getLogger(__name__).warning("Could not import CIM: %s", e)
    CIM_AVAILABLE = False

# Load environment variables (optional)
load_dotenv()

_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=_log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("cosmicwatch.api")

# Global CIM instance
cosmic_model = None
_ai_cache_state: Dict[str, Any] = {"running": False}
_cdm_model_state: Dict[str, Any] = {}

def load_cosmic_model():
    """Load the Cosmic Intelligence Model at startup"""
    global cosmic_model
    if CIM_AVAILABLE:
        try:
            cosmic_model = get_cosmic_intelligence_model()
            model_info = cosmic_model.get_model_info()
            logger.info(
                "CIM loaded name=%s version=%s accuracy=%.4f params=%s",
                model_info.get("model_name"),
                model_info.get("model_version"),
                float(model_info.get("accuracy", 0.0)),
                model_info.get("num_parameters"),
            )
            return True
        except Exception as e:
            logger.exception("Failed to load CIM: %s", e)
            return False
    return False

app = FastAPI(
    title="🛰️ Space Debris API with Cosmic Intelligence",
    description="API serving REAL AI predictions from Cosmic Intelligence Model",
    version="2.0.0"
)

# CORS for React frontend
_cors_origins_env = os.getenv("BACKEND_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CIM on startup
@app.on_event("startup")
async def startup_event():
    load_cosmic_model()
    try:
        from utils.database import init_db, migrate_database_for_ai_caching, DATABASE_URL
        init_db()
        migrate_database_for_ai_caching()
        logger.info("Debris database schema initialized database_url=%s", DATABASE_URL)
    except Exception as e:
        logger.exception("Failed to initialize debris database schema: %s", e)
    try:
        enabled = (os.getenv("COSMICWATCH_BACKGROUND_UPDATES", "true").strip().lower() in {"1", "true", "yes", "y", "on"})
        if enabled:
            from utils.background_updater import start_background_updates, get_background_manager
            manager = get_background_manager()
            interval_hours = os.getenv("COSMICWATCH_REFRESH_INTERVAL_HOURS")
            if interval_hours:
                manager.update_interval_hours = max(0.25, float(interval_hours))
            start_background_updates()
    except Exception as e:
        logger.exception("Failed to start background updates: %s", e)
    try:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "cdm_public")
        _cdm_model_state.update(load_latest_baseline_model(models_dir=models_dir, target=os.getenv("CDM_TARGET", "pc_quantile_class")))
        logger.info("CDM model loaded kind=%s target=%s model=%s", _cdm_model_state.get("kind"), _cdm_model_state.get("target"), _cdm_model_state.get("model_path"))
    except Exception as e:
        logger.warning("CDM model not loaded: %s", e)


@app.on_event("shutdown")
async def shutdown_event():
    try:
        from utils.background_updater import stop_background_updates
        stop_background_updates()
    except Exception:
        pass


def _ensure_space_debris_schema() -> None:
    from utils.database import init_db, migrate_database_for_ai_caching
    init_db()
    migrate_database_for_ai_caching()


def _run_ai_cache_job(max_rows: int, batch_size: int) -> None:
    global _ai_cache_state
    started_at = datetime.now().isoformat()
    _ai_cache_state = {"running": True, "started_at": started_at, "updated": 0, "errors": 0, "max_rows": max_rows}
    try:
        _ensure_space_debris_schema()
        if cosmic_model is None:
            _ai_cache_state["running"] = False
            _ai_cache_state["error"] = "CIM not loaded"
            _ai_cache_state["finished_at"] = datetime.now().isoformat()
            return

        from sqlalchemy import or_
        from utils.database import get_db_session, SpaceDebris

        with get_db_session() as db:
            rows = (
                db.query(
                    SpaceDebris.id,
                    SpaceDebris.altitude,
                    SpaceDebris.velocity,
                    SpaceDebris.inclination,
                    SpaceDebris.size,
                    SpaceDebris.latitude,
                    SpaceDebris.longitude,
                    SpaceDebris.x,
                    SpaceDebris.y,
                    SpaceDebris.z,
                )
                .filter(or_(SpaceDebris.ai_enhanced == 0, SpaceDebris.ai_risk_level.is_(None), SpaceDebris.ai_confidence.is_(None)))
                .limit(max_rows)
                .all()
            )

            updates: list[dict] = []
            now = datetime.now()
            for row in rows:
                debris_data = {
                    "altitude": row.altitude if row.altitude is not None else 400,
                    "velocity": row.velocity if row.velocity is not None else 7.5,
                    "inclination": row.inclination if row.inclination is not None else 51.6,
                    "size": row.size if row.size is not None else 1.0,
                    "latitude": row.latitude if row.latitude is not None else 0.0,
                    "longitude": row.longitude if row.longitude is not None else 0.0,
                    "x": row.x if row.x is not None else 0.0,
                    "y": row.y if row.y is not None else 0.0,
                    "z": row.z if row.z is not None else 0.0,
                }
                try:
                    pred = cosmic_model.predict_debris_risk(debris_data)
                    risk_level = str(pred.get("risk_level", "MEDIUM")).upper()
                    confidence = float(pred.get("confidence", 0.0))
                    if confidence != confidence:
                        confidence = 0.0
                    confidence = max(0.0, min(confidence, 1.0))
                    updates.append(
                        {
                            "id": row.id,
                            "ai_risk_level": risk_level,
                            "ai_confidence": confidence,
                            "ai_last_predicted": now,
                            "ai_enhanced": 1,
                        }
                    )
                except Exception:
                    _ai_cache_state["errors"] = int(_ai_cache_state.get("errors", 0)) + 1

                if len(updates) >= batch_size:
                    db.bulk_update_mappings(SpaceDebris, updates)
                    db.commit()
                    _ai_cache_state["updated"] = int(_ai_cache_state.get("updated", 0)) + len(updates)
                    updates = []

            if updates:
                db.bulk_update_mappings(SpaceDebris, updates)
                db.commit()
                _ai_cache_state["updated"] = int(_ai_cache_state.get("updated", 0)) + len(updates)

        _ai_cache_state["running"] = False
        _ai_cache_state["finished_at"] = datetime.now().isoformat()
    except Exception as e:
        _ai_cache_state["running"] = False
        _ai_cache_state["error"] = str(e)
        _ai_cache_state["finished_at"] = datetime.now().isoformat()


def _get_ai_cache_metrics() -> Dict[str, Any]:
    try:
        from utils.database import get_db_session, SpaceDebris
        from sqlalchemy import func
        from datetime import timedelta

        now = datetime.now()
        day_ago = now - timedelta(hours=24)

        with get_db_session() as db:
            total = int(db.query(func.count(SpaceDebris.id)).scalar() or 0)
            cached = int(
                db.query(func.count(SpaceDebris.id))
                .filter(SpaceDebris.ai_enhanced == 1)
                .filter(SpaceDebris.ai_risk_level.isnot(None))
                .filter(SpaceDebris.ai_confidence.isnot(None))
                .scalar()
                or 0
            )
            predicted_24h = int(
                db.query(func.count(SpaceDebris.id))
                .filter(SpaceDebris.ai_last_predicted.isnot(None))
                .filter(SpaceDebris.ai_last_predicted >= day_ago)
                .scalar()
                or 0
            )
            coverage = float(cached / total) if total else 0.0
            return {"total_objects": total, "cached_objects": cached, "cache_coverage": coverage, "predicted_last_24h": predicted_24h}
    except Exception:
        return {"total_objects": 0, "cached_objects": 0, "cache_coverage": 0.0, "predicted_last_24h": 0}


def _load_latest_evaluation_metrics() -> Dict[str, Any] | None:
    try:
        import glob
        import json

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates: list[str] = []
        candidates.extend(glob.glob(os.path.join(project_root, "cosmic_intelligence_cdm_public_*_metrics.json")))
        candidates.extend(glob.glob(os.path.join(project_root, "models", "cdm_public", "*_metrics.json")))
        candidates.extend(glob.glob(os.path.join(project_root, "models", "*", "*_metrics.json")))

        candidates = [p for p in candidates if os.path.isfile(p)]
        if not candidates:
            return None

        latest = max(candidates, key=lambda p: os.path.getmtime(p))
        with open(latest, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return None

        report = payload.get("classification_report")
        if not isinstance(report, dict):
            return None

        accuracy = report.get("accuracy")
        macro = report.get("macro avg") if isinstance(report.get("macro avg"), dict) else {}
        f1 = macro.get("f1-score")

        return {
            "accuracy": float(accuracy) if accuracy is not None else None,
            "f1_score": float(f1) if f1 is not None else None,
            "evaluation_source": os.path.basename(latest),
            "evaluation_label": payload.get("label_col") or payload.get("target") or None,
        }
    except Exception:
        return None

# Pydantic models
class DebrisObject(BaseModel):
    id: str
    altitude: float
    latitude: float
    longitude: float
    x: float
    y: float
    z: float
    size: float
    velocity: float
    inclination: float
    risk_score: float
    risk_level: str
    confidence: float
    object_name: Optional[str] = None
    object_type: Optional[str] = None
    last_updated: Optional[str] = None

class CollisionAlert(BaseModel):
    id: str
    object1_id: str
    object2_id: str
    distance_km: float
    probability: float
    severity: str
    time_to_approach: Optional[float] = None

class DashboardStats(BaseModel):
    total_objects: int
    ai_enhanced: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    cache_hit_rate: float
    model_accuracy: float
    model_parameters: str
    inference_speed: str

class ModelInfo(BaseModel):
    name: str
    version: str
    evaluated: bool
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    evaluation_source: Optional[str] = None
    evaluation_label: Optional[str] = None
    parameters: int
    inference_speed_ms: float


class CdmModelInfo(BaseModel):
    kind: str
    target: str
    metrics_path: str
    model_path: str
    test_accuracy: Optional[float] = None
    test_macro_f1: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_macro_f1: Optional[float] = None


class CdmPredictRequest(BaseModel):
    MIN_RNG: Optional[float] = None
    hours_to_tca: Optional[float] = None
    SAT1_OBJECT_TYPE: Optional[str] = None
    SAT2_OBJECT_TYPE: Optional[str] = None
    SAT1_RCS: Optional[str] = None
    SAT2_RCS: Optional[str] = None
    SAT_1_EXCL_VOL: Optional[float] = None
    SAT_2_EXCL_VOL: Optional[float] = None

    class Config:
        extra = "allow"


class CdmPredictResponse(BaseModel):
    predicted_class: str
    probabilities: Dict[str, float]
    target: str
    kind: str

class ConjunctionVector3(BaseModel):
    x: float
    y: float
    z: float

class ConjunctionState(BaseModel):
    r_eci_km: ConjunctionVector3
    v_eci_km_s: ConjunctionVector3

class ConjunctionCovRtn(BaseModel):
    crr: float
    ctt: float
    cnn: float
    crt: float = 0.0
    crn: float = 0.0
    ctn: float = 0.0

class ConjunctionPairRequest(BaseModel):
    object1: ConjunctionState
    object2: ConjunctionState
    object1_pos_cov_rtn_km2: ConjunctionCovRtn
    object2_pos_cov_rtn_km2: ConjunctionCovRtn
    hard_body_radius_km: float
    n_theta: int = 360
    n_r: int = 300

class ConjunctionFromCdmRequest(BaseModel):
    cdm_text: str
    n_theta: int = 360
    n_r: int = 300

class ConjunctionPcResponse(BaseModel):
    pc: float
    hard_body_radius_km: float
    miss_distance_km: float
    rel_speed_km_s: float
    mu_plane_km: List[float]
    cov_plane_km2: List[List[float]]
    sigma_major_km: float
    sigma_minor_km: float
    notes: str

def _normalize_risk_level_filter(risk_level: Optional[str]) -> Optional[str]:
    if risk_level is None:
        return None
    risk_level_upper = risk_level.upper()
    if risk_level_upper not in {"CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"}:
        raise HTTPException(status_code=400, detail="Invalid risk_level")
    return risk_level_upper

_ALLOWED_MODEL_RISK_LEVELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

def _sanitize_model_output(risk_level: Any, confidence: Any) -> tuple[str, float]:
    risk_level_str = str(risk_level).upper() if risk_level is not None else "MEDIUM"
    if risk_level_str not in _ALLOWED_MODEL_RISK_LEVELS:
        risk_level_str = "MEDIUM"
    try:
        confidence_f = float(confidence)
    except Exception:
        confidence_f = 0.0
    if confidence_f != confidence_f:
        confidence_f = 0.0
    confidence_f = max(0.0, min(confidence_f, 1.0))
    return risk_level_str, confidence_f


def _risk_from_score(score: Any) -> tuple[str, float, float]:
    try:
        score_f = float(score)
    except Exception:
        score_f = 0.5
    if score_f != score_f:
        score_f = 0.5
    score_f = max(0.0, min(score_f, 1.0))

    if score_f >= 0.85:
        level = "CRITICAL"
    elif score_f >= 0.65:
        level = "HIGH"
    elif score_f >= 0.35:
        level = "MEDIUM"
    else:
        level = "LOW"

    confidence = max(0.0, min(0.6 + abs(score_f - 0.5), 1.0))
    return level, confidence, score_f

@app.get("/")
async def root():
    return {
        "message": "🛰️ Space Debris API with Cosmic Intelligence",
        "version": "2.0.0",
        "cim_loaded": cosmic_model is not None,
        "endpoints": ["/api/debris", "/api/stats", "/api/collisions", "/api/model-info"]
    }

@app.get("/healthz")
async def healthz():
    db_ok = False
    try:
        from sqlalchemy import text
        from utils.database import get_db_session
        with get_db_session() as db:
            db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"ok": True, "db_ok": db_ok, "cim_loaded": cosmic_model is not None}

def _vec3(v: ConjunctionVector3) -> List[float]:
    return [float(v.x), float(v.y), float(v.z)]

def _cov_rtn_to_matrix(c: ConjunctionCovRtn) -> List[List[float]]:
    return [
        [float(c.crr), float(c.crt), float(c.crn)],
        [float(c.crt), float(c.ctt), float(c.ctn)],
        [float(c.crn), float(c.ctn), float(c.cnn)],
    ]

@app.post("/api/conjunction/pair", response_model=ConjunctionPcResponse)
async def conjunction_pair_pc(req: ConjunctionPairRequest):
    try:
        if req.hard_body_radius_km <= 0:
            raise HTTPException(status_code=400, detail="hard_body_radius_km must be > 0")
        if req.n_theta < 64 or req.n_r < 64:
            raise HTTPException(status_code=400, detail="n_theta and n_r too small")

        r1 = _vec3(req.object1.r_eci_km)
        v1 = _vec3(req.object1.v_eci_km_s)
        r2 = _vec3(req.object2.r_eci_km)
        v2 = _vec3(req.object2.v_eci_km_s)
        cov1 = _cov_rtn_to_matrix(req.object1_pos_cov_rtn_km2)
        cov2 = _cov_rtn_to_matrix(req.object2_pos_cov_rtn_km2)

        import numpy as np

        mu2, cov2p = combined_pos_cov_encounter_plane(
            primary_r_eci=np.array(r1, dtype=float),
            primary_v_eci=np.array(v1, dtype=float),
            secondary_r_eci=np.array(r2, dtype=float),
            secondary_v_eci=np.array(v2, dtype=float),
            cov1_rtn_pos=np.array(cov1, dtype=float),
            cov2_rtn_pos=np.array(cov2, dtype=float),
        )

        pc = pc_circle_grid(mu2, cov2p, float(req.hard_body_radius_km), n_theta=req.n_theta, n_r=req.n_r).pc
        miss = float(np.linalg.norm(mu2))
        rel_speed = float(np.linalg.norm(np.array(v2, dtype=float) - np.array(v1, dtype=float)))

        evals = np.linalg.eigvalsh(cov2p)
        evals = np.clip(evals, 0.0, None)
        sigma_major = float(np.sqrt(evals[-1]))
        sigma_minor = float(np.sqrt(evals[0]))

        return ConjunctionPcResponse(
            pc=float(pc),
            hard_body_radius_km=float(req.hard_body_radius_km),
            miss_distance_km=miss,
            rel_speed_km_s=rel_speed,
            mu_plane_km=[float(mu2[0]), float(mu2[1])],
            cov_plane_km2=[[float(cov2p[0, 0]), float(cov2p[0, 1])], [float(cov2p[1, 0]), float(cov2p[1, 1])]],
            sigma_major_km=sigma_major,
            sigma_minor_km=sigma_minor,
            notes="Pc computed in encounter plane from RTN position covariances",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conjunction/from-cdm", response_model=ConjunctionPcResponse)
async def conjunction_from_cdm(req: ConjunctionFromCdmRequest):
    try:
        ev = parse_cdm_event(req.cdm_text)
        if ev.primary_state_eci is None or ev.secondary_state_eci is None:
            raise HTTPException(status_code=400, detail="CDM missing OBJECT1/OBJECT2 state vectors")
        if ev.primary_pos_cov_rtn is None or ev.secondary_pos_cov_rtn is None:
            raise HTTPException(status_code=400, detail="CDM missing OBJECT1/OBJECT2 RTN position covariance")
        if ev.hard_body_radius_km is None or ev.hard_body_radius_km <= 0:
            raise HTTPException(status_code=400, detail="CDM missing HARD_BODY_RADIUS_KM (or equivalent)")
        if req.n_theta < 64 or req.n_r < 64:
            raise HTTPException(status_code=400, detail="n_theta and n_r too small")

        import numpy as np

        (r1, v1) = ev.primary_state_eci
        (r2, v2) = ev.secondary_state_eci

        mu2, cov2p = combined_pos_cov_encounter_plane(
            primary_r_eci=r1,
            primary_v_eci=v1,
            secondary_r_eci=r2,
            secondary_v_eci=v2,
            cov1_rtn_pos=ev.primary_pos_cov_rtn,
            cov2_rtn_pos=ev.secondary_pos_cov_rtn,
        )

        pc = pc_circle_grid(mu2, cov2p, float(ev.hard_body_radius_km), n_theta=req.n_theta, n_r=req.n_r).pc
        miss = float(np.linalg.norm(mu2))
        rel_speed = float(np.linalg.norm(v2 - v1))

        evals = np.linalg.eigvalsh(cov2p)
        evals = np.clip(evals, 0.0, None)
        sigma_major = float(np.sqrt(evals[-1]))
        sigma_minor = float(np.sqrt(evals[0]))

        return ConjunctionPcResponse(
            pc=float(pc),
            hard_body_radius_km=float(ev.hard_body_radius_km),
            miss_distance_km=miss,
            rel_speed_km_s=rel_speed,
            mu_plane_km=[float(mu2[0]), float(mu2[1])],
            cov_plane_km2=[[float(cov2p[0, 0]), float(cov2p[0, 1])], [float(cov2p[1, 0]), float(cov2p[1, 1])]],
            sigma_major_km=sigma_major,
            sigma_minor_km=sigma_minor,
            notes="Computed from CDM (KVN) states + RTN covariances",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debris", response_model=List[DebrisObject])
async def get_all_debris(
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0, le=500000),
    risk_level: Optional[str] = None,
    use_cim: bool = Query(default=False),
):
    """Get debris objects (fast path uses cached/DB risk; optional CIM inference for small limits)"""
    try:
        risk_level_upper = _normalize_risk_level_filter(risk_level)
        from sqlalchemy import or_
        from utils.database import get_db_session, SpaceDebris

        try:
            with get_db_session() as db:
                q = db.query(SpaceDebris).order_by(SpaceDebris.id)
                if risk_level_upper:
                    if risk_level_upper == "UNKNOWN":
                        q = q.filter(or_(SpaceDebris.ai_risk_level.is_(None), SpaceDebris.ai_risk_level == "UNKNOWN"))
                    else:
                        q = q.filter(SpaceDebris.ai_risk_level == risk_level_upper)
                rows = q.offset(offset).limit(limit).all()
        except Exception:
            _ensure_space_debris_schema()
            with get_db_session() as db:
                q = db.query(SpaceDebris).order_by(SpaceDebris.id)
                if risk_level_upper:
                    if risk_level_upper == "UNKNOWN":
                        q = q.filter(or_(SpaceDebris.ai_risk_level.is_(None), SpaceDebris.ai_risk_level == "UNKNOWN"))
                    else:
                        q = q.filter(SpaceDebris.ai_risk_level == risk_level_upper)
                rows = q.offset(offset).limit(limit).all()
        
        if use_cim and limit > 50:
            raise HTTPException(status_code=400, detail="use_cim=true supports limit<=50 for performance")

        debris_list = []
        for row in rows:
            debris_data = {
                'altitude': row.altitude or 400,
                'velocity': row.velocity or 7.5,
                'inclination': row.inclination or 51.6,
                'size': row.size or 1.0,
                'latitude': row.latitude or 0,
                'longitude': row.longitude or 0,
                'x': row.x or 0,
                'y': row.y or 0,
                'z': row.z or 0,
            }

            risk_score_db = row.risk_score if row.risk_score is not None else 0.5

            risk_level_pred = row.ai_risk_level if row.ai_risk_level else None
            confidence = row.ai_confidence if row.ai_confidence is not None else None
            risk_score = risk_score_db

            if risk_level_pred is None or confidence is None:
                risk_level_pred, confidence, risk_score = _risk_from_score(risk_score_db)

            if use_cim and cosmic_model is not None:
                try:
                    prediction = cosmic_model.predict_debris_risk(debris_data)
                    risk_level_pred_raw = prediction.get("risk_level", risk_level_pred)
                    confidence_raw = prediction.get("confidence", confidence)
                    risk_level_pred, confidence = _sanitize_model_output(risk_level_pred_raw, confidence_raw)
                    risk_score_map = {"CRITICAL": 0.9, "HIGH": 0.7, "MEDIUM": 0.4, "LOW": 0.15}
                    risk_score = risk_score_map.get(risk_level_pred, risk_score)
                except Exception as pred_error:
                    logger.warning("CIM prediction error for %s: %s", row["id"], pred_error)
            
            obj_type = "DEBRIS"

            debris_list.append(DebrisObject(
                id=row.id,
                altitude=debris_data['altitude'],
                latitude=debris_data['latitude'],
                longitude=debris_data['longitude'],
                x=debris_data['x'],
                y=debris_data['y'],
                z=debris_data['z'],
                size=debris_data['size'],
                velocity=debris_data['velocity'],
                inclination=debris_data['inclination'],
                risk_score=risk_score,
                risk_level=risk_level_pred,
                confidence=confidence,
                object_name=row.id,
                object_type=obj_type,
                last_updated=datetime.now().isoformat()
            ))
        
        return debris_list
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debris/{debris_id}", response_model=DebrisObject)
async def get_debris_by_id(debris_id: str):
    """Get single debris object by ID with CIM prediction"""
    try:
        from utils.database import get_db_session, SpaceDebris
        try:
            with get_db_session() as db:
                row = db.query(SpaceDebris).filter(SpaceDebris.id == debris_id).first()
        except Exception:
            _ensure_space_debris_schema()
            with get_db_session() as db:
                row = db.query(SpaceDebris).filter(SpaceDebris.id == debris_id).first()
        
        if row is None:
            raise HTTPException(status_code=404, detail="Debris not found")
        
        debris_data = {
            'altitude': row.altitude or 400,
            'velocity': row.velocity or 7.5,
            'inclination': row.inclination or 51.6,
            'size': row.size or 1.0,
            'latitude': row.latitude or 0,
            'longitude': row.longitude or 0,
            'x': row.x or 0,
            'y': row.y or 0,
            'z': row.z or 0,
        }

        risk_score_db = row.risk_score if row.risk_score is not None else 0.5
        risk_level_pred = row.ai_risk_level if row.ai_risk_level else None
        confidence = row.ai_confidence if row.ai_confidence is not None else None
        risk_score = risk_score_db
        if risk_level_pred is None or confidence is None:
            risk_level_pred, confidence, risk_score = _risk_from_score(risk_score_db)

        if cosmic_model is not None:
            try:
                prediction = cosmic_model.predict_debris_risk(debris_data)
                risk_level_pred_raw = prediction.get("risk_level", risk_level_pred)
                confidence_raw = prediction.get("confidence", confidence)
                risk_level_pred, confidence = _sanitize_model_output(risk_level_pred_raw, confidence_raw)
                risk_score_map = {"CRITICAL": 0.9, "HIGH": 0.7, "MEDIUM": 0.4, "LOW": 0.15}
                risk_score = risk_score_map.get(risk_level_pred, risk_score)
            except Exception:
                pass
        
        return DebrisObject(
            id=row.id,
            altitude=debris_data['altitude'],
            latitude=debris_data['latitude'],
            longitude=debris_data['longitude'],
            x=debris_data['x'],
            y=debris_data['y'],
            z=debris_data['z'],
            size=debris_data['size'],
            velocity=debris_data['velocity'],
            inclination=debris_data['inclination'],
            risk_score=risk_score,
            risk_level=risk_level_pred,
            confidence=confidence,
            object_name=row.id,
            object_type="DEBRIS",
            last_updated=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get dashboard statistics with real CIM data"""
    try:
        from utils.database import get_db_session, SpaceDebris
        from sqlalchemy import func, case

        try:
            with get_db_session() as db:
                total = int(db.query(func.count(SpaceDebris.id)).scalar() or 0)
                ai_enhanced_count = int(db.query(func.coalesce(func.sum(SpaceDebris.ai_enhanced), 0)).scalar() or 0)
                grouped = db.query(SpaceDebris.ai_risk_level, func.count(SpaceDebris.id)).group_by(SpaceDebris.ai_risk_level).all()
                risk_counts_raw = {k: int(v) for (k, v) in grouped}
        except Exception:
            _ensure_space_debris_schema()
            total = 0
            ai_enhanced_count = 0
            risk_counts_raw = {}

        ai_counts = {k: int(v) for k, v in risk_counts_raw.items() if k in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}}
        if sum(ai_counts.values()) == 0 and total > 0:
            with get_db_session() as db:
                critical_count = db.query(func.coalesce(func.sum(case((SpaceDebris.risk_score >= 0.85, 1), else_=0)), 0)).scalar()
                high_count = db.query(
                    func.coalesce(func.sum(case(((SpaceDebris.risk_score >= 0.65) & (SpaceDebris.risk_score < 0.85), 1), else_=0)), 0)
                ).scalar()
                medium_count = db.query(
                    func.coalesce(func.sum(case(((SpaceDebris.risk_score >= 0.35) & (SpaceDebris.risk_score < 0.65), 1), else_=0)), 0)
                ).scalar()
                low_count = db.query(func.coalesce(func.sum(case((SpaceDebris.risk_score < 0.35, 1), else_=0)), 0)).scalar()
            ai_counts = {"CRITICAL": int(critical_count or 0), "HIGH": int(high_count or 0), "MEDIUM": int(medium_count or 0), "LOW": int(low_count or 0)}
        
        metrics = _load_latest_evaluation_metrics() or {}
        model_accuracy_val = metrics.get("accuracy")
        if model_accuracy_val is None and cosmic_model is not None:
            try:
                info = cosmic_model.get_model_info()
                model_accuracy_val = info.get("accuracy")
            except Exception:
                pass
        model_accuracy = float(model_accuracy_val) if model_accuracy_val is not None else 0.0
        model_params = "16.58M"
        
        if cosmic_model is not None:
            try:
                info = cosmic_model.get_model_info()
                model_params = f"{info.get('num_parameters', 16583477)/1e6:.2f}M"
            except Exception:
                pass
        
        return DashboardStats(
            total_objects=total,
            ai_enhanced=ai_enhanced_count,
            critical_count=ai_counts.get('CRITICAL', 0),
            high_count=ai_counts.get('HIGH', 0),
            medium_count=ai_counts.get('MEDIUM', 0),
            low_count=ai_counts.get('LOW', 0),
            cache_hit_rate=_get_ai_cache_metrics().get("cache_coverage", 0.0),
            model_accuracy=model_accuracy,
            model_parameters=model_params,
            inference_speed="<0.2ms"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/collisions", response_model=List[CollisionAlert])
async def get_collision_alerts(
    limit: int = Query(default=10, ge=1, le=200),
    sample_size: int = Query(default=300, ge=50, le=2000),
    threshold_km: float = Query(default=2000.0, ge=50.0, le=20000.0),
):
    """Get collision alerts using fast risk heuristics (no per-request CIM inference)"""
    try:
        import numpy as np
        from utils.database import get_db_session, SpaceDebris
        try:
            with get_db_session() as db:
                objects = (
                    db.query(SpaceDebris.id, SpaceDebris.x, SpaceDebris.y, SpaceDebris.z, SpaceDebris.risk_score)
                    .order_by(SpaceDebris.id)
                    .limit(sample_size)
                    .all()
                )
        except Exception:
            _ensure_space_debris_schema()
            objects = []
        
        if not objects:
            return []

        ids = [o.id for o in objects]
        xyz = np.array([[float(o.x or 0.0), float(o.y or 0.0), float(o.z or 0.0)] for o in objects], dtype=np.float64)
        risks = np.array([float(o.risk_score) if o.risk_score is not None else 0.5 for o in objects], dtype=np.float64)

        n = xyz.shape[0]
        diffs = xyz[:, None, :] - xyz[None, :, :]
        dist = np.sqrt(np.sum(diffs * diffs, axis=2))
        np.fill_diagonal(dist, np.inf)

        tri = np.triu_indices(n, k=1)
        d_flat = dist[tri]
        if d_flat.size == 0:
            return []

        closish = np.where(d_flat < float(threshold_km))[0]
        if closish.size == 0:
            best = np.argpartition(d_flat, min(50, d_flat.size - 1))[: min(50, d_flat.size)]
        else:
            best = closish

        best = best[np.argsort(d_flat[best])][: max(int(limit) * 4, 10)]

        alerts: list[CollisionAlert] = []
        for idx in best:
            i = int(tri[0][idx])
            j = int(tri[1][idx])
            distance = float(d_flat[idx])
            combined_risk = float((risks[i] + risks[j]) / 2.0)

            proximity = max(0.0, min(1.0, (float(threshold_km) - distance) / float(threshold_km)))
            probability = round(min(0.1, 0.1 * (0.25 + 0.75 * combined_risk) * (0.3 + 0.7 * proximity)), 4)

            if probability >= 0.07:
                severity = "high"
            elif probability >= 0.04:
                severity = "medium"
            else:
                severity = "low"

            alerts.append(
                CollisionAlert(
                    id=f"CA-{len(alerts)+1:04d}",
                    object1_id=str(ids[i]),
                    object2_id=str(ids[j]),
                    distance_km=round(distance, 2),
                    probability=probability,
                    severity=severity,
                )
            )
        
        # Sort by probability
        alerts.sort(key=lambda x: x.probability, reverse=True)
        return alerts[:limit]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get Cosmic Intelligence Model information"""
    name = "Cosmic Intelligence Model (CIM)"
    version = "1.2"
    params = 16583477
    accuracy = None
    f1_score = None
    evaluated = False
    evaluation_source = None
    evaluation_label = None

    if cosmic_model is not None:
        try:
            info = cosmic_model.get_model_info()
            name = info.get("model_name", name)
            version = info.get("model_version", version)
            params = int(info.get("num_parameters", params))
            accuracy = info.get("accuracy")
            f1_score = info.get("f1_score")
            evaluated = bool(info.get("is_loaded"))
            ckpt = info.get("checkpoint_path")
            if ckpt:
                evaluation_source = os.path.basename(str(ckpt))
                evaluation_label = "checkpoint"
        except Exception:
            pass

    return ModelInfo(
        name=name,
        version=version,
        evaluated=bool(evaluated),
        accuracy=float(accuracy) if accuracy is not None else None,
        f1_score=float(f1_score) if f1_score is not None else None,
        evaluation_source=evaluation_source,
        evaluation_label=evaluation_label,
        parameters=params,
        inference_speed_ms=0.2,
    )


@app.get("/api/cdm/model-info", response_model=CdmModelInfo)
async def get_cdm_model_info():
    if not _cdm_model_state:
        raise HTTPException(status_code=503, detail="CDM model not loaded")
    metrics = _cdm_model_state.get("metrics") or {}
    test_report = metrics.get("test_classification_report") or metrics.get("classification_report") or {}
    val_report = metrics.get("val_classification_report") or {}
    macro = test_report.get("macro avg") if isinstance(test_report, dict) else None
    val_macro = val_report.get("macro avg") if isinstance(val_report, dict) else None

    def _f1(rep: Any) -> Optional[float]:
        try:
            return float(rep.get("f1-score")) if isinstance(rep, dict) and rep.get("f1-score") is not None else None
        except Exception:
            return None

    return CdmModelInfo(
        kind=str(_cdm_model_state.get("kind") or "baseline"),
        target=str(_cdm_model_state.get("target") or "pc_quantile_class"),
        metrics_path=str(_cdm_model_state.get("metrics_path") or ""),
        model_path=str(_cdm_model_state.get("model_path") or ""),
        test_accuracy=float(test_report.get("accuracy")) if isinstance(test_report, dict) and test_report.get("accuracy") is not None else None,
        test_macro_f1=_f1(macro),
        val_accuracy=float(val_report.get("accuracy")) if isinstance(val_report, dict) and val_report.get("accuracy") is not None else None,
        val_macro_f1=_f1(val_macro),
    )


@app.post("/api/cdm/predict", response_model=CdmPredictResponse)
async def predict_cdm(req: CdmPredictRequest):
    if not _cdm_model_state:
        raise HTTPException(status_code=503, detail="CDM model not loaded")
    record = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    pred = predict_baseline(_cdm_model_state, record)
    return CdmPredictResponse(
        predicted_class=pred["predicted_class"],
        probabilities=pred["probabilities"],
        target=str(_cdm_model_state.get("target") or "pc_quantile_class"),
        kind=str(_cdm_model_state.get("kind") or "baseline"),
    )

class RefreshResponse(BaseModel):
    success: bool
    message: str
    objects_updated: int
    timestamp: str

@app.post("/api/refresh")
async def refresh_data():
    """Refresh debris data from Celestrak API"""
    try:
        # Import the database utilities
        from utils.database import init_db, populate_real_data_force_refresh, get_cached_objects_count, is_data_fresh
        
        logger.info("Starting data refresh from Celestrak")
        
        # Ensure database table exists with correct schema
        logger.info("Ensuring database schema is up to date")
        init_db()
        
        # Force refresh data from Celestrak
        success = populate_real_data_force_refresh()
        
        if success:
            count = get_cached_objects_count()
            logger.info("Data refresh completed objects_updated=%s", count)
            return RefreshResponse(
                success=True,
                message=f"Successfully refreshed data from Celestrak!",
                objects_updated=count,
                timestamp=datetime.now().isoformat()
            )
        else:
            return RefreshResponse(
                success=False,
                message="Data refresh failed. Check network connection.",
                objects_updated=0,
                timestamp=datetime.now().isoformat()
            )
            
    except Exception as e:
        logger.exception("Data refresh error: %s", e)
        return RefreshResponse(
            success=False,
            message=f"Error: {str(e)}",
            objects_updated=0,
            timestamp=datetime.now().isoformat()
        )


@app.post("/api/cache-ai")
async def cache_ai_predictions(
    max_rows: int = Query(default=5000, ge=1, le=50000),
    batch_size: int = Query(default=200, ge=10, le=2000),
):
    global _ai_cache_state
    if _ai_cache_state.get("running"):
        return {"started": False, "status": _ai_cache_state}
    t = threading.Thread(target=_run_ai_cache_job, args=(max_rows, batch_size), daemon=True)
    t.start()
    return {"started": True, "status": _ai_cache_state}


@app.get("/api/cache-ai/status")
async def cache_ai_status():
    status = dict(_ai_cache_state)
    status["metrics"] = _get_ai_cache_metrics()
    return status

@app.get("/api/data-status")
async def get_data_status():
    """Get data freshness status"""
    try:
        from utils.database import get_cached_objects_count, is_data_fresh, get_metadata_value
        
        return {
            "objects_count": get_cached_objects_count(),
            "is_fresh": is_data_fresh(max_age_hours=2),
            "last_update": get_metadata_value('last_celestrak_download', 'Never'),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "objects_count": 0,
            "is_fresh": False,
            "last_update": "Unknown",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

