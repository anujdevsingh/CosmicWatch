# Model Card — Cosmic Intelligence Model (CIM)

## Summary
CIM is a software component used in CosmicWatch to produce a coarse, ML-assisted risk category for space objects. It is intended for visualization, educational exploration, and prioritization, not for operational collision avoidance decisions.

## Intended Use
- Educational dashboarding and exploration of the space object population
- Relative prioritization for “which objects should I inspect first?”
- Demonstrations of how ML outputs can be integrated into a data pipeline

Not intended use:
- Flight operations, safety-of-flight, maneuver planning, or collision avoidance decisions
- Any workflow that requires validated Probability of Collision (Pc) with covariance and certified ephemerides

## Inputs
Typical inputs include (availability depends on data source and DB schema):
- altitude (km)
- velocity (km/s)
- inclination (deg)
- size (m)
- latitude/longitude (deg) and/or x/y/z (Earth-centered coordinates used for visualization)

## Outputs
CIM returns a dictionary-like prediction containing:
- risk_level: one of CRITICAL, HIGH, MEDIUM, LOW
- confidence: a value in [0, 1] representing model confidence
- probabilities: per-class probabilities when available
- metadata fields (model name/version, enhanced flag, optional uncertainty fields)

## Data Sources
CosmicWatch currently uses CelesTrak feeds as the primary public catalog source. CelesTrak objects are commonly distributed via TLE-based products suitable for visualization and coarse screening.

## Evaluation
Any accuracy/F1 numbers shown in the UI or README must be treated as project-internal results unless:
- The dataset, splits, and scripts are fully reproducible
- Metrics are computed on a documented evaluation set
- Results are independently verified

Current note:
- TLE/catalog data does not come with ground-truth “collision risk” labels. Any supervised label must be explicitly defined (e.g., Space-Track `cdm_public` emergency_reportable/PC buckets, or a physics-based reference Pc engine on synthetic encounters).

## Limitations
- TLE-based propagation and derived features are not collision-avoidance grade without covariance and high-accuracy ephemerides.
- Model outputs are sensitive to feature assumptions (e.g., simplified size estimation).
- “Confidence” is not a certified uncertainty bound.

## Responsible Use Notes
If you share this project publicly or with operational organizations, describe it as:
- an educational dashboard and risk-ranking prototype
- not a validated conjunction assessment system
