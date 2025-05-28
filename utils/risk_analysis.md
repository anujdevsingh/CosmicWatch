# Space Debris Risk Analysis System

## Risk Score Calculation

The risk score is calculated using multiple factors:

1. **Altitude Risk (40% weight)**
   - Lower orbits (300-2000km) have higher risk due to:
     * More crowded orbital space
     * Atmospheric drag effects
     * Higher collision velocities
   - Formula: `altitude_risk = 1.0 - (altitude - 300) / (36000 - 300)`

2. **Inclination Risk (30% weight)**
   - Higher inclinations mean more orbital crossings
   - More potential collision points
   - Formula: `inclination_risk = inclination / 90.0`

3. **Density Factor (30% weight)**
   - Exponential decay with altitude
   - Reflects higher object density in LEO
   - Formula: `density_factor = exp(-altitude/500)`

Final Risk Score = `0.4 * altitude_risk + 0.3 * inclination_risk + 0.3 * density_factor`

## Collision Detection System

1. **Proximity Analysis**
   - Calculates minimum distance between objects
   - Uses current position (x, y, z coordinates)
   - Triggers alerts for objects within 50km

2. **Collision Probability**
   - Based on:
     * Combined object sizes
     * Relative velocities
     * Orbital parameters
     * Altitude effects

3. **Severity Classification**
   - High: probability > 0.6 and distance < 10km
   - Medium: probability > 0.3 or distance < 20km
   - Low: probability > 0.1 or distance < 40km

## Data Processing Pipeline

1. **Data Generation**
   - Simulates realistic orbital scenarios
   - Uses NASA orbital mechanics formulas
   - Generates 1000+ objects across LEO, MEO, GEO

2. **Orbit Types**
   - LEO (Low Earth Orbit): 300-2000km
   - MEO (Medium Earth Orbit): 2000-35500km
   - GEO (Geostationary Orbit): 35500-36000km

3. **Position Updates**
   - Real-time position calculations
   - Uses orbital period and mean motion
   - Updates every 5 minutes
