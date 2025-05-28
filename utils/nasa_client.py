import requests
from datetime import datetime
from typing import List, Dict, Any
import time
import numpy as np

class NASAClient:
    """Client for generating realistic space debris data based on NASA orbital parameters."""

    def get_debris_data(self, limit: int = 1500) -> List[Dict[Any, Any]]:
        """Generate realistic space debris data with proper orbital mechanics."""
        try:
            print("Generating NASA-based space debris data...")
            current_time = datetime.now()
            transformed_data = []

            # Generate more realistic debris objects
            for i in range(limit):
                try:
                    # Realistic orbital parameters for different orbit types
                    orbit_type = np.random.choice(['LEO', 'MEO', 'GEO'], p=[0.7, 0.2, 0.1])

                    if orbit_type == 'LEO':
                        altitude = float(np.random.uniform(300, 2000))
                        # Higher risk of collisions in popular LEO orbits
                        risk_base = 0.7
                    elif orbit_type == 'MEO':
                        altitude = float(np.random.uniform(2000, 35500))
                        risk_base = 0.4
                    else:  # GEO
                        altitude = float(np.random.uniform(35500, 36000))
                        risk_base = 0.3

                    # Most satellites are under 98 degrees
                    inclination = float(np.random.uniform(0, 98))

                    # Calculate orbital period using Kepler's laws
                    period = 2 * np.pi * np.sqrt((altitude + 6371)**3 / 398600.4418)
                    mean_motion = current_time.timestamp() % period / period
                    theta = mean_motion * 2 * np.pi
                    phi = inclination * np.pi / 180.0

                    # Calculate cartesian coordinates
                    r = altitude + 6371
                    x = float(r * np.cos(phi) * np.cos(theta))
                    y = float(r * np.cos(phi) * np.sin(theta))
                    z = float(r * np.sin(phi))

                    # Multi-factor risk calculation
                    # 1. Altitude risk - lower orbits are riskier
                    altitude_risk = 1.0 - (altitude - 300) / (36000 - 300)

                    # 2. Inclination risk - higher inclinations mean more orbital crossings
                    inclination_risk = inclination / 90.0

                    # 3. Density factor - more objects in lower orbits
                    density_factor = np.exp(-altitude/500)

                    # 4. Orbit type specific risk
                    orbit_risk = risk_base

                    # Combined risk score with weights
                    risk_score = float(min(1.0, max(0.1, (
                        0.3 * altitude_risk +    # 30% weight for altitude
                        0.2 * inclination_risk + # 20% weight for inclination
                        0.3 * density_factor +   # 30% weight for density
                        0.2 * orbit_risk        # 20% weight for orbit type
                    ))))

                    # Size varies by orbit (typically larger in higher orbits)
                    base_size = np.random.uniform(0.1, 5.0)
                    size_factor = 1 + altitude/36000
                    size = float(base_size * size_factor)

                    # Calculate realistic orbital velocity using vis-viva equation
                    velocity = float(np.sqrt(398600.4418 / (altitude + 6371)))  # km/s

                    transformed_data.append({
                        'id': f"NASA-{i:04d}",
                        'altitude': altitude,
                        'latitude': float(np.degrees(np.arcsin(z/r))),
                        'longitude': float(np.degrees(np.arctan2(y, x))),
                        'x': float(x),
                        'y': float(y),
                        'z': float(z),
                        'size': size,
                        'velocity': velocity,
                        'inclination': inclination,
                        'risk_score': risk_score,
                        'last_updated': current_time
                    })

                    if i % 100 == 0:
                        print(f"Generated {i} objects...")

                except (ValueError, TypeError) as e:
                    print(f"Error processing object {i}: {str(e)}")
                    continue

            print(f"Generated {len(transformed_data)} objects with realistic orbital parameters")
            return transformed_data

        except Exception as e:
            print(f"Error in NASA data generation: {str(e)}")
            raise