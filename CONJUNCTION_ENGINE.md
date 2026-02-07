# Conjunction + Pc Engine (Prototype)

## What This Adds
This repository now includes a minimal conjunction mathematics module intended for research and educational use:
- Encounter-plane geometry
- RTN↔ECI covariance projection
- A baseline Probability of Collision (Pc) computation for a circular hard-body region
- A CDM (KVN) parser to extract states, covariances, and hard-body radius

## Important Scope Note
This is not a certified operational tool. If you need operational conjunction assessment, you must use validated ephemerides + covariance and follow CCSDS/NASA CA practices.

## Pc Definition Used
We model the relative position in the encounter plane as a 2D Gaussian with mean μ and covariance C, then compute:
Pc = P(‖X‖ ≤ HBR),  X ~ N(μ, C)

## Files
- Pc integration: [conjunction/pc.py](file:///e:/projects/SpaceDebrisDashboard/SpaceDebrisDashboard/CosmicWatch/conjunction/pc.py)
- Frames + encounter plane: [conjunction/frames.py](file:///e:/projects/SpaceDebrisDashboard/SpaceDebrisDashboard/CosmicWatch/conjunction/frames.py)
- CDM parsing (KVN): [conjunction/cdm.py](file:///e:/projects/SpaceDebrisDashboard/SpaceDebrisDashboard/CosmicWatch/conjunction/cdm.py)
- Tests: [conjunction/tests](file:///e:/projects/SpaceDebrisDashboard/SpaceDebrisDashboard/CosmicWatch/conjunction/tests)

## How To Run Tests
```powershell
cd E:\projects\SpaceDebrisDashboard\SpaceDebrisDashboard\CosmicWatch
.\.venv\Scripts\Activate.ps1
python -m unittest conjunction.tests.test_pc -q
python -m unittest conjunction.tests.test_frames -q
python -m unittest conjunction.tests.test_cdm -q
```

## Training a Baseline ML Model From CDMs
You need a folder of CDM-like KVN files that contain:
- OBJECT1/OBJECT2 ECI states (X,Y,Z and X_DOT,Y_DOT,Z_DOT)
- OBJECT1/OBJECT2 position covariance in RTN (CRR, CTT, CNN and optionally CRT/CRN/CTN)
- HARD_BODY_RADIUS_KM (or equivalent)

Train:
```powershell
cd E:\projects\SpaceDebrisDashboard\SpaceDebrisDashboard\CosmicWatch
.\.venv\Scripts\Activate.ps1
python -m conjunction.train_conjunction_model --cdm-dir .\\cdm_data --pc-threshold 1e-7 --out-model conjunction_model.pkl --out-report conjunction_model_report.json
```

Evaluate:
```powershell
python -m conjunction.evaluate_conjunction_model --cdm-dir .\\cdm_data --model conjunction_model.pkl --pc-threshold 1e-7 --out conjunction_eval.json
```

## Where To Get CDM Data (Practical Options)
- If you are a satellite owner/operator, CDMs are typically received via operational channels (conjunction screening services). In many cases, CDMs are not broadly public.
- You can start development using:
  - Example CDM formats and sample blocks from CCSDS CDM documentation
  - NASA CARA materials describing the process and standards
- For true NASA-grade evaluation, you will need real conjunction events with covariance (CDMs) from an authorized source or partnership.
- For an educational public demo without CDMs, you can still visualize close approaches, but you should label results as screening-only (no covariance => no validated Pc).

See also: [CDM_DATA_SOURCES.md](CDM_DATA_SOURCES.md)
