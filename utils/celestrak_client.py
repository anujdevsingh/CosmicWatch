import requests
import json
import numpy as np
import math
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TLEData:
    """Two-Line Element data structure."""
    norad_id: str
    name: str
    line1: str
    line2: str

class CelesTrakClient:
    """Client for fetching real-time satellite and debris data from CelesTrak."""
    
    BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"
    
    def __init__(self):
        """Initialize CelesTrak client."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SpaceDebrisDashboard/1.0 (Educational)',
            'Accept': 'application/json'
        })
    
    def fetch_active_satellites(self, format_type: str = "json") -> List[Dict[str, Any]]:
        """
        Fetch all active satellites and debris from CelesTrak.
        
        Args:
            format_type: Data format ('json', 'tle', 'xml', 'csv')
            
        Returns:
            List of satellite/debris objects
        """
        try:
            print("🛰️ Fetching active satellites from CelesTrak...")
            
            url = f"{self.BASE_URL}?GROUP=active&FORMAT={format_type}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            if format_type.lower() == "json":
                data = response.json()
                print(f"✅ Retrieved {len(data)} active objects from CelesTrak")
                return data
            else:
                # Handle TLE format
                tle_data = self._parse_tle_data(response.text)
                print(f"✅ Retrieved {len(tle_data)} TLE objects from CelesTrak")
                return [self._tle_to_dict(tle) for tle in tle_data]
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching CelesTrak data: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing CelesTrak JSON: {str(e)}")
            raise
    
    def fetch_debris_only(self) -> List[Dict[str, Any]]:
        """Fetch space debris specifically from active satellites data."""
        try:
            print("🗑️ Fetching space debris from CelesTrak...")
            
            # Get all active satellites and filter for debris
            active_data = self.fetch_active_satellites()
            
            # Filter for debris objects based on object names
            debris_objects = []
            for obj in active_data:
                name = obj.get('OBJECT_NAME', '').upper()
                if any(keyword in name for keyword in ['DEBRIS', 'DEB', 'FRAGMENT', 'FRAG']):
                    debris_objects.append(obj)
            
            print(f"✅ Retrieved {len(debris_objects)} debris objects from active satellites")
            return debris_objects
            
        except Exception as e:
            print(f"❌ Error fetching debris data: {str(e)}")
            return []
    
    def fetch_starlink_satellites(self) -> List[Dict[str, Any]]:
        """Fetch Starlink constellation satellites."""
        try:
            print("🌌 Fetching Starlink satellites from CelesTrak...")
            
            url = f"{self.BASE_URL}?GROUP=starlink&FORMAT=json"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Retrieved {len(data)} Starlink satellites")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching Starlink data: {str(e)}")
            return []
    
    def fetch_recent_launches(self) -> List[Dict[str, Any]]:
        """Fetch recently launched objects (last 30 days)."""
        try:
            print("🚀 Fetching recent launches from CelesTrak...")
            
            url = f"{self.BASE_URL}?GROUP=last-30-days&FORMAT=json"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Retrieved {len(data)} recent launches")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching recent launches: {str(e)}")
            return []
    
    def _parse_tle_data(self, tle_text: str) -> List[TLEData]:
        """Parse TLE format data."""
        lines = tle_text.strip().split('\n')
        tle_objects = []
        
        for i in range(0, len(lines), 3):
            if i + 2 < len(lines):
                name = lines[i].strip()
                line1 = lines[i + 1].strip()
                line2 = lines[i + 2].strip()
                
                # Extract NORAD ID from line 1
                norad_id = line1[2:7].strip()
                
                tle_objects.append(TLEData(norad_id, name, line1, line2))
        
        return tle_objects
    
    def _tle_to_dict(self, tle: TLEData) -> Dict[str, Any]:
        """Convert TLE data to dictionary format."""
        return {
            'NORAD_CAT_ID': tle.norad_id,
            'OBJECT_NAME': tle.name,
            'TLE_LINE1': tle.line1,
            'TLE_LINE2': tle.line2
        }
    
    def transform_to_dashboard_format(self, satellite_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform CelesTrak data to match our dashboard's expected format.
        
        Args:
            satellite_data: Raw satellite data from CelesTrak
            
        Returns:
            List of transformed objects ready for the dashboard
        """
        transformed_objects = []
        
        for obj in satellite_data:
            try:
                # Extract basic info
                norad_id = str(obj.get('NORAD_CAT_ID', 'UNKNOWN'))
                object_name = obj.get('OBJECT_NAME', 'UNKNOWN').strip()
                
                # Get orbital elements (from JSON format)
                mean_motion = float(obj.get('MEAN_MOTION', 15.0))  # revolutions per day
                inclination = float(obj.get('INCLINATION', 0.0))  # degrees
                eccentricity = float(obj.get('ECCENTRICITY', 0.0))
                
                # Calculate orbital period (minutes)
                period = 1440.0 / mean_motion if mean_motion > 0 else 90.0
                
                # Calculate semi-major axis using Kepler's third law
                # T^2 = (4π^2/μ) * a^3, where μ = 398600.4418 km³/s² (Earth's gravitational parameter)
                period_seconds = period * 60
                mu = 398600.4418  # km³/s²
                semi_major_axis = ((mu * (period_seconds / (2 * math.pi))**2)**(1/3))
                
                # Calculate apogee and perigee
                apogee = semi_major_axis * (1 + eccentricity) - 6371  # km above Earth
                perigee = semi_major_axis * (1 - eccentricity) - 6371  # km above Earth
                
                # Ensure reasonable values
                apogee = max(150, min(apogee, 50000))  # 150km to 50,000km
                perigee = max(150, min(perigee, 50000))
                altitude = (apogee + perigee) / 2
                
                # Calculate current position using simplified orbital mechanics
                # This is an approximation for visualization purposes
                current_time = datetime.now().timestamp()
                
                # Mean anomaly progression
                mean_anomaly = (current_time % period_seconds) / period_seconds * 2 * math.pi
                
                # Convert to true anomaly (simplified for circular orbits)
                true_anomaly = mean_anomaly  # Simplified assumption
                
                # Calculate position in orbital plane
                r = semi_major_axis
                
                # Convert to Earth-centered coordinates
                # Simplified model using inclination and current time
                lon_offset = (current_time % 86400) / 86400 * 360  # Earth rotation
                
                # Calculate latitude and longitude
                lat = math.sin(inclination * math.pi / 180) * math.sin(true_anomaly) * 90
                lon = (true_anomaly * 180 / math.pi + lon_offset) % 360
                if lon > 180:
                    lon -= 360
                
                # Calculate 3D coordinates
                earth_radius = 6371  # km
                total_radius = earth_radius + altitude
                
                lat_rad = lat * math.pi / 180
                lon_rad = lon * math.pi / 180
                
                x = total_radius * math.cos(lat_rad) * math.cos(lon_rad)
                y = total_radius * math.cos(lat_rad) * math.sin(lon_rad)
                z = total_radius * math.sin(lat_rad)
                
                # Calculate orbital velocity
                velocity = math.sqrt(mu / total_radius)  # km/s
                
                # Determine object characteristics
                object_type = self._classify_object(object_name, norad_id)
                size = self._estimate_size(object_type, object_name)
                risk_score = self._calculate_risk_score(altitude, size, object_type)
                
                # Create transformed object
                transformed_obj = {
                    'id': f"CT-{norad_id}",
                    'norad_id': norad_id,
                    'object_name': object_name,
                    'object_type': object_type,
                    'altitude': float(altitude),
                    'latitude': float(lat),
                    'longitude': float(lon),
                    'x': float(x),
                    'y': float(y),
                    'z': float(z),
                    'size': float(size),
                    'velocity': float(velocity),
                    'inclination': float(inclination),
                    'eccentricity': float(eccentricity),
                    'period': float(period),
                    'apogee': float(apogee),
                    'perigee': float(perigee),
                    'mean_motion': float(mean_motion),
                    'risk_score': float(risk_score),
                    'last_updated': datetime.now().isoformat(),
                    'data_source': 'CelesTrak',
                    'confidence': 0.95  # CelesTrak is highly reliable
                }
                
                transformed_objects.append(transformed_obj)
                
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
                print(f"⚠️ Error processing object {obj.get('NORAD_CAT_ID', 'UNKNOWN')}: {str(e)}")
                continue
        
        print(f"✅ Successfully transformed {len(transformed_objects)} objects")
        return transformed_objects
    
    def _classify_object(self, name: str, norad_id: str) -> str:
        """Classify object type based on name and characteristics."""
        name_upper = name.upper()
        
        if any(keyword in name_upper for keyword in ['DEBRIS', 'DEB', 'FRAGMENT', 'FRAG']):
            return 'DEBRIS'
        elif any(keyword in name_upper for keyword in ['ROCKET', 'R/B', 'STAGE']):
            return 'ROCKET BODY'
        elif any(keyword in name_upper for keyword in ['STARLINK', 'ONEWEB', 'IRIDIUM']):
            return 'CONSTELLATION'
        elif any(keyword in name_upper for keyword in ['ISS', 'STATION', 'TIANGONG']):
            return 'SPACE STATION'
        else:
            return 'PAYLOAD'
    
    def _estimate_size(self, object_type: str, name: str) -> float:
        """Estimate object size based on type and name."""
        if object_type == 'DEBRIS':
            return np.random.uniform(0.1, 2.0)  # Small debris
        elif object_type == 'ROCKET BODY':
            return np.random.uniform(5.0, 20.0)  # Large rocket bodies
        elif object_type == 'SPACE STATION':
            return np.random.uniform(50.0, 100.0)  # Very large
        elif object_type == 'CONSTELLATION':
            return np.random.uniform(2.0, 5.0)  # Small satellites
        else:  # PAYLOAD
            return np.random.uniform(1.0, 10.0)  # Various sizes
    
    def _calculate_risk_score(self, altitude: float, size: float, object_type: str) -> float:
        """Calculate risk score based on altitude, size, and type."""
        # Base risk factors
        altitude_risk = 1.0
        if altitude < 500:  # Low Earth Orbit - higher collision risk
            altitude_risk = 0.8
        elif altitude < 1000:
            altitude_risk = 0.6
        elif altitude < 2000:
            altitude_risk = 0.4
        else:
            altitude_risk = 0.2
        
        # Size risk
        size_risk = min(size / 10.0, 1.0)
        
        # Type risk
        type_risk = {
            'DEBRIS': 0.9,
            'ROCKET BODY': 0.7,
            'PAYLOAD': 0.5,
            'CONSTELLATION': 0.3,
            'SPACE STATION': 0.8
        }.get(object_type, 0.5)
        
        # Combined risk score (0-1 scale)
        risk_score = (altitude_risk + size_risk + type_risk) / 3.0
        return min(max(risk_score, 0.0), 1.0)

def fetch_celestrak_data(include_debris: bool = True, include_starlink: bool = True) -> List[Dict[str, Any]]:
    """
    Main function to fetch comprehensive satellite and debris data from CelesTrak.
    
    Args:
        include_debris: Whether to include specific debris data
        include_starlink: Whether to include Starlink constellation
        
    Returns:
        List of all satellite and debris objects
    """
    client = CelesTrakClient()
    all_objects = []
    
    try:
        # Get all active satellites (includes debris)
        active_satellites = client.fetch_active_satellites()
        all_objects.extend(active_satellites)
        
        # Optionally add specific debris data
        if include_debris:
            try:
                debris_objects = client.fetch_debris_only()
                # Filter out duplicates based on NORAD ID
                existing_ids = {obj.get('NORAD_CAT_ID') for obj in all_objects}
                new_debris = [obj for obj in debris_objects 
                             if obj.get('NORAD_CAT_ID') not in existing_ids]
                all_objects.extend(new_debris)
            except Exception as e:
                print(f"⚠️ Could not fetch additional debris data: {e}")
        
        # Optionally add Starlink data (usually included in active)
        if include_starlink:
            try:
                starlink_objects = client.fetch_starlink_satellites()
                existing_ids = {obj.get('NORAD_CAT_ID') for obj in all_objects}
                new_starlink = [obj for obj in starlink_objects 
                               if obj.get('NORAD_CAT_ID') not in existing_ids]
                all_objects.extend(new_starlink)
            except Exception as e:
                print(f"⚠️ Could not fetch additional Starlink data: {e}")
        
        # Transform to dashboard format
        transformed_objects = client.transform_to_dashboard_format(all_objects)
        
        print(f"🌍 CELESTRAK DATA SUMMARY:")
        print(f"   📡 Total Objects: {len(transformed_objects)}")
        
        # Count by type
        type_counts = {}
        for obj in transformed_objects:
            obj_type = obj.get('object_type', 'UNKNOWN')
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        
        for obj_type, count in type_counts.items():
            print(f"   🔹 {obj_type}: {count}")
        
        return transformed_objects
        
    except Exception as e:
        print(f"❌ Error fetching CelesTrak data: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the CelesTrak client
    try:
        data = fetch_celestrak_data()
        print(f"\n✅ Successfully fetched {len(data)} objects from CelesTrak")
        
        if data:
            sample_obj = data[0]
            print(f"\n📊 Sample object:")
            for key, value in sample_obj.items():
                print(f"   {key}: {value}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}") 