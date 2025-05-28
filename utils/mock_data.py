import numpy as np
from datetime import datetime, timedelta

def get_debris_data():
    """Generate mock space debris data for demonstration."""

    np.random.seed(int(datetime.now().timestamp()))

    n_objects = 100
    data = []

    for i in range(n_objects):
        # Generate random orbital parameters
        altitude = float(np.random.uniform(300, 1000))  # km
        inclination = float(np.random.uniform(0, 180))  # degrees

        # Calculate position
        theta = float(np.random.uniform(0, 2*np.pi))
        phi = float(np.random.uniform(0, np.pi))

        r = altitude + 6371  # Earth radius + altitude
        x = float(r * np.sin(phi) * np.cos(theta))
        y = float(r * np.sin(phi) * np.sin(theta))
        z = float(r * np.cos(phi))

        # Calculate lat/lon
        longitude = float(np.degrees(np.arctan2(y, x)))
        latitude = float(np.degrees(np.arcsin(z/r)))

        # Generate object properties
        data.append({
            'id': f'DEB-{i:04d}',
            'altitude': altitude,
            'latitude': latitude,
            'longitude': longitude,
            'x': x,
            'y': y,
            'z': z,
            'size': float(np.random.uniform(0.1, 5.0)),  # meters
            'velocity': float(np.random.uniform(7.0, 8.0)),  # km/s
            'inclination': inclination,
            'risk_score': float(np.random.beta(2, 5)),  # Weighted towards lower risk
            'last_updated': datetime.now() - timedelta(minutes=np.random.randint(0, 60))
        })

    return data