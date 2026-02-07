# Where To Get CDM Data

## Reality Check
Conjunction Data Messages (CDMs) with covariance are usually exchanged in operational contexts and are not always published openly.

## Practical Sources
- Satellite owner/operator workflows: the mission team typically receives conjunction messages from their screening provider (government or commercial), including state and covariance.
- NASA CARA program (process + standards guidance): https://www.nasa.gov/cara/
- NASA CA Best Practices Handbook (formats, covariance, CDM usage): https://nodis3.gsfc.nasa.gov/OCE_docs/OCE_51.pdf
- CCSDS CDM / collision probability standardization (includes example fields and method options): https://ccsds.org/wp-content/uploads/gravity_forms/9-6f599803174a64f5da08b9814720b5c4/2025/02/508x0p11.pdf

## For Development Without Real CDMs
- Use example CDM text blocks from CCSDS/NASA docs to validate parsing and math.
- Generate synthetic “CDM-like” events (state + covariance + HBR) for testing only, and label results as non-operational.

## What You Need For Training
For the training scripts in this repo, each CDM-like KVN file should include:
- OBJECT1/OBJECT2 state vectors: X, Y, Z and X_DOT, Y_DOT, Z_DOT (ECI)
- OBJECT1/OBJECT2 RTN position covariance: CRR, CTT, CNN (and optional CRT/CRN/CTN)
- HARD_BODY_RADIUS_KM (combined) or per-object HBRs
