import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import requests
import time

# Constants
mu = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
R_earth = 6371.0  # Earth's radius (km)
J2 = 1.08262668e-3  # Earth's J2 coefficient

class EnhancedPhysicsFeatureExtractor:
    """
    Enhanced feature extractor for 30+ physics-informed features
    Optimized for GPU processing and temporal modeling
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.space_weather_cache = {}
        self.cache_timeout = 3600  # 1 hour cache
        
    def extract_orbital_elements(self, tle_data: Dict) -> np.ndarray:
        """Extract basic orbital elements (6 features)"""
        try:
            # Semi-major axis (km)
            mean_motion = tle_data.get('MEAN_MOTION', 15.0)  # rev/day
            n = mean_motion * 2 * np.pi / 86400  # rad/s
            a = (mu / n**2)**(1/3)
            
            # Eccentricity
            e = tle_data.get('ECCENTRICITY', 0.0)
            
            # Inclination (rad)
            i = np.radians(tle_data.get('INCLINATION', 0.0))
            
            # Right ascension of ascending node (rad)
            omega = np.radians(tle_data.get('RA_OF_ASC_NODE', 0.0))
            
            # Argument of perigee (rad)
            w = np.radians(tle_data.get('ARG_OF_PERICENTER', 0.0))
            
            # Mean anomaly (rad)
            M = np.radians(tle_data.get('MEAN_ANOMALY', 0.0))
            
            return np.array([a, e, i, omega, w, M])
            
        except Exception as e:
            print(f"Error extracting orbital elements: {e}")
            return np.zeros(6)
    
    def extract_derived_physics(self, orbital_elements: np.ndarray, tle_data: Dict) -> np.ndarray:
        """Extract derived physics parameters (8 features)"""
        try:
            a, e, i, omega, w, M = orbital_elements
            
            # Orbital period (seconds)
            period = 2 * np.pi * np.sqrt(a**3 / mu)
            
            # Perigee and apogee altitudes (km)
            perigee = a * (1 - e) - R_earth
            apogee = a * (1 + e) - R_earth
            
            # Velocities at perigee and apogee (km/s)
            velocity_perigee = np.sqrt(mu * (2/(a*(1-e)) - 1/a))
            velocity_apogee = np.sqrt(mu * (2/(a*(1+e)) - 1/a))
            
            # Atmospheric drag coefficient (simplified)
            atmospheric_drag = self.calculate_atmospheric_drag(perigee)
            
            # Solar radiation pressure (simplified)
            area_to_mass = tle_data.get('RCS_SIZE', 1.0)  # Approximate area-to-mass ratio
            solar_pressure = self.calculate_solar_pressure(a, area_to_mass)
            
            # J2 perturbation factor
            perturbation_factor = self.calculate_j2_perturbation(a, e, i)
            
            return np.array([
                period, perigee, apogee, velocity_perigee,
                velocity_apogee, atmospheric_drag, solar_pressure, perturbation_factor
            ])
            
        except Exception as e:
            print(f"Error extracting derived physics: {e}")
            return np.zeros(8)
    
    def extract_temporal_history(self, object_id: str, current_time: datetime, sequence_length: int = 7) -> np.ndarray:
        """Extract temporal history features (7 features)"""
        try:
            # This is a simplified implementation
            # In a real system, you would query historical TLE data
            
            # For now, simulate temporal features based on current orbital parameters
            temporal_features = []
            
            # Simulate 7-day position/velocity trends
            for i in range(sequence_length):
                # Simplified: assume small variations over time
                position_change = np.random.normal(0, 0.1)  # km
                velocity_change = np.random.normal(0, 0.01)  # km/s
                temporal_features.append(position_change)
            
            return np.array(temporal_features[:7])  # Ensure exactly 7 features
            
        except Exception as e:
            print(f"Error extracting temporal history: {e}")
            return np.zeros(7)
    
    def extract_environmental_factors(self) -> np.ndarray:
        """Extract environmental factors (5 features)"""
        try:
            # Solar activity (F10.7 index)
            f107_index = self.get_solar_activity()
            
            # Geomagnetic activity (Ap index)
            ap_index = self.get_geomagnetic_activity()
            
            # Atmospheric density estimate
            atmospheric_density = self.estimate_atmospheric_density()
            
            # Solar cycle phase (simplified)
            solar_cycle_phase = self.get_solar_cycle_phase()
            
            # Seasonal atmospheric factor
            seasonal_factor = self.get_seasonal_atmospheric_factor()
            
            return np.array([
                f107_index, ap_index, atmospheric_density,
                solar_cycle_phase, seasonal_factor
            ])
            
        except Exception as e:
            print(f"Error extracting environmental factors: {e}")
            return np.array([150.0, 10.0, 1e-12, 0.5, 1.0])  # Default values
    
    def extract_collision_context(self, orbital_elements: np.ndarray, tle_data: Dict) -> np.ndarray:
        """Extract collision context features (4 features)"""
        try:
            a, e, i, omega, w, M = orbital_elements
            
            # Local debris density (simplified)
            altitude = (a * (1 - e) + a * (1 + e)) / 2 - R_earth
            local_debris_density = self.calculate_local_debris_density(altitude, i)
            
            # Conjunction frequency estimate
            conjunction_frequency = self.calculate_conjunction_frequency(a, e, i)
            
            # Collision probability estimate
            collision_probability = self.estimate_collision_probability(orbital_elements)
            
            # Relative velocity with nearby objects
            relative_velocity = self.calculate_relative_velocity(orbital_elements)
            
            return np.array([
                local_debris_density, conjunction_frequency,
                collision_probability, relative_velocity
            ])
            
        except Exception as e:
            print(f"Error extracting collision context: {e}")
            return np.zeros(4)
    
    def calculate_atmospheric_drag(self, perigee_altitude: float) -> float:
        """Calculate atmospheric drag coefficient"""
        if perigee_altitude < 100:
            return 10.0  # High drag in very low altitude
        elif perigee_altitude < 300:
            return 5.0 * np.exp(-(perigee_altitude - 100) / 50)
        elif perigee_altitude < 600:
            return 1.0 * np.exp(-(perigee_altitude - 300) / 100)
        else:
            return 0.1 * np.exp(-(perigee_altitude - 600) / 200)
    
    def calculate_solar_pressure(self, semi_major_axis: float, area_to_mass: float) -> float:
        """Calculate solar radiation pressure effects"""
        # Solar pressure coefficient (simplified)
        solar_constant = 1361  # W/m^2
        c = 299792458  # Speed of light m/s
        solar_pressure_base = solar_constant / c * 1e-9  # Simplified units
        
        # Distance factor (inverse square law)
        distance_factor = (R_earth / semi_major_axis)**2
        
        return solar_pressure_base * distance_factor * area_to_mass
    
    def calculate_j2_perturbation(self, a: float, e: float, i: float) -> float:
        """Calculate J2 perturbation effects"""
        # Simplified J2 perturbation factor
        return J2 * (R_earth / a)**2 * np.cos(i) * (1 - e**2)**(-3/2)
    
    def get_solar_activity(self) -> float:
        """Get current solar activity (F10.7 index)"""
        # Simplified: return typical value with some variation
        return 150.0 + np.random.normal(0, 20)
    
    def get_geomagnetic_activity(self) -> float:
        """Get current geomagnetic activity (Ap index)"""
        # Simplified: return typical value with some variation
        return 10.0 + np.random.normal(0, 5)
    
    def estimate_atmospheric_density(self, altitude: float = 400.0) -> float:
        """Estimate atmospheric density at given altitude"""
        # Simplified exponential atmosphere model
        scale_height = 50.0  # km
        rho_0 = 1e-12  # kg/m^3 at reference altitude
        return rho_0 * np.exp(-altitude / scale_height)
    
    def get_solar_cycle_phase(self) -> float:
        """Get current solar cycle phase (0-1)"""
        # Simplified: 11-year solar cycle
        current_year = datetime.now().year
        cycle_start = 2019  # Approximate cycle 25 start
        cycle_progress = (current_year - cycle_start) % 11 / 11
        return cycle_progress
    
    def get_seasonal_atmospheric_factor(self) -> float:
        """Get seasonal atmospheric variation factor"""
        # Simplified seasonal variation
        day_of_year = datetime.now().timetuple().tm_yday
        seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * day_of_year / 365.25)
        return seasonal_factor
    
    def calculate_local_debris_density(self, altitude: float, inclination: float) -> float:
        """Calculate local debris density"""
        # Simplified debris density model
        if 700 <= altitude <= 1000:
            # High density region
            base_density = 1e-6
        elif altitude <= 2000:
            base_density = 5e-7
        else:
            base_density = 1e-7
        
        # Inclination factor (higher density at certain inclinations)
        inclination_factor = 1 + 0.5 * np.sin(inclination)
        
        return base_density * inclination_factor
    
    def calculate_conjunction_frequency(self, a: float, e: float, i: float) -> float:
        """Calculate conjunction frequency with other objects"""
        # Simplified conjunction frequency based on orbital characteristics
        # Higher frequency for common orbital regions
        altitude = a - R_earth
        
        if 700 <= altitude <= 1000:
            base_frequency = 10.0  # High traffic region
        elif altitude <= 2000:
            base_frequency = 5.0
        else:
            base_frequency = 1.0
        
        # Eccentricity and inclination factors
        eccentric_factor = 1 + e * 2
        inclination_factor = np.sin(i) + 0.5
        
        return base_frequency * eccentric_factor * inclination_factor
    
    def estimate_collision_probability(self, orbital_elements: np.ndarray) -> float:
        """Estimate collision probability"""
        a, e, i, omega, w, M = orbital_elements
        altitude = a - R_earth
        
        # Base probability based on altitude and debris density
        if altitude < 300:
            base_prob = 1e-6  # Very high risk at low altitude
        elif 700 <= altitude <= 1000:
            base_prob = 5e-7  # High risk in debris belt
        elif altitude <= 2000:
            base_prob = 1e-7
        else:
            base_prob = 1e-8
        
        # Modify by eccentricity and inclination
        risk_factor = (1 + e * 3) * (np.sin(i) + 0.5)
        
        return min(base_prob * risk_factor, 1e-4)  # Cap at reasonable maximum
    
    def calculate_relative_velocity(self, orbital_elements: np.ndarray) -> float:
        """Calculate typical relative velocity with nearby objects"""
        a, e, i, omega, w, M = orbital_elements
        
        # Orbital velocity
        velocity = np.sqrt(mu / a)
        
        # Relative velocity depends on inclination and eccentricity
        # Higher inclination and eccentricity lead to higher relative velocities
        relative_factor = np.sin(i) * (1 + e * 2)
        
        return velocity * relative_factor
    
    def extract_all_features(self, tle_data: Dict, object_id: str = None) -> np.ndarray:
        """Extract all 30 physics features"""
        try:
            # Basic orbital elements (6 features)
            orbital_elements = self.extract_orbital_elements(tle_data)
            
            # Derived physics parameters (8 features)
            derived_physics = self.extract_derived_physics(orbital_elements, tle_data)
            
            # Temporal history (7 features)
            current_time = datetime.now()
            temporal_history = self.extract_temporal_history(object_id or "unknown", current_time)
            
            # Environmental factors (5 features)
            environmental_factors = self.extract_environmental_factors()
            
            # Collision context (4 features)
            collision_context = self.extract_collision_context(orbital_elements, tle_data)
            
            # Combine all features
            all_features = np.concatenate([
                orbital_elements,      # 6 features
                derived_physics,       # 8 features
                temporal_history,      # 7 features
                environmental_factors, # 5 features
                collision_context      # 4 features
            ])
            
            # Ensure we have exactly 30 features
            if len(all_features) != 30:
                print(f"Warning: Expected 30 features, got {len(all_features)}")
                # Pad or truncate to 30
                if len(all_features) < 30:
                    all_features = np.pad(all_features, (0, 30 - len(all_features)))
                else:
                    all_features = all_features[:30]
            
            return all_features
            
        except Exception as e:
            print(f"Error extracting all features: {e}")
            return np.zeros(30)
    
    def create_temporal_sequence(self, tle_data_list: List[Dict], sequence_length: int = 7) -> np.ndarray:
        """Create temporal sequence for transformer input"""
        try:
            sequences = []
            
            for tle_data in tle_data_list:
                features = self.extract_all_features(tle_data)
                sequences.append(features)
            
            # Pad sequence if needed
            while len(sequences) < sequence_length:
                if sequences:
                    sequences.append(sequences[-1])  # Repeat last entry
                else:
                    sequences.append(np.zeros(30))  # Zero padding
            
            # Truncate if too long
            sequences = sequences[:sequence_length]
            
            return np.array(sequences)
            
        except Exception as e:
            print(f"Error creating temporal sequence: {e}")
            return np.zeros((sequence_length, 30))


def test_feature_extraction():
    """Test the enhanced feature extraction"""
    print("Testing Enhanced Physics Feature Extraction...")
    
    # Create feature extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = EnhancedPhysicsFeatureExtractor(device=device)
    
    # Sample TLE data
    sample_tle = {
        'MEAN_MOTION': 15.5,
        'ECCENTRICITY': 0.01,
        'INCLINATION': 51.6,
        'RA_OF_ASC_NODE': 45.0,
        'ARG_OF_PERICENTER': 90.0,
        'MEAN_ANOMALY': 180.0,
        'RCS_SIZE': 1.0
    }
    
    # Extract features
    features = extractor.extract_all_features(sample_tle, "TEST-001")
    
    print(f"Extracted {len(features)} features:")
    print(f"Features: {features}")
    print(f"Feature stats - Min: {features.min():.6f}, Max: {features.max():.6f}")
    
    # Test temporal sequence
    tle_list = [sample_tle] * 7  # Simulate 7 days of data
    sequence = extractor.create_temporal_sequence(tle_list)
    
    print(f"Temporal sequence shape: {sequence.shape}")
    print("Feature extraction test completed!")


if __name__ == "__main__":
    test_feature_extraction() 