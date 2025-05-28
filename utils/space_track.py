import os
import requests
from datetime import datetime
from typing import List, Dict, Any
import time
import numpy as np

class SpaceTrackClient:
    """Client for interacting with Space-Track.org API."""

    BASE_URL = "https://www.space-track.org/ajaxauth/login"

    def __init__(self):
        self.username = os.getenv("SPACETRACK_USERNAME")
        self.password = os.getenv("SPACETRACK_PASSWORD")
        if not self.username or not self.password:
            raise ValueError("Space-Track credentials not found in environment variables")
        self.session = requests.Session()

    def login(self) -> None:
        """Authenticate with Space-Track.org."""
        try:
            print(f"Attempting to login with username: {self.username}")

            # Clear any existing cookies
            self.session.cookies.clear()

            # Set up headers
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            login_data = {
                "identity": self.username,
                "password": self.password
            }

            # Perform login
            login_response = self.session.post(
                f"{self.BASE_URL}/ajaxauth/login",
                data=login_data,
                headers=headers,
                allow_redirects=True
            )
            login_response.raise_for_status()

            # Check if login was successful
            if "Login Failed" in login_response.text:
                raise ValueError("Login failed. Check your credentials.")

            print("Successfully logged into Space-Track.org")

        except requests.exceptions.RequestException as e:
            print(f"Error during Space-Track login: {str(e)}")
            raise

    def get_latest_debris_data(self, limit: int = 100) -> List[Dict[Any, Any]]:
        """Fetch latest debris data from Space-Track.org."""
        try:
            print("Attempting to fetch space debris data...")

            # Ensure we're logged in
            self.login()

            # Use a simpler approach - directly query the satellite catalog
            url = f"{self.BASE_URL}/basicspacedata/query/class/satcat/orderby/NORAD_CAT_ID/limit/{limit}/format/json"
            print(f"Requesting data from: {url}")

            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/json'
            }

            response = self.session.get(url, headers=headers)
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error response: {response.text}")
                raise Exception(f"Failed to fetch data: HTTP {response.status_code}")
                
            response.raise_for_status()

            data = response.json()
            print(f"Retrieved {len(data)} objects from Space-Track")

            # Transform data to match our database schema
            transformed_data = []
            for item in data:
                try:
                    # Extract basic parameters from satcat
                    norad_id = item.get('NORAD_CAT_ID', 'UNKNOWN')
                    object_name = item.get('OBJECT_NAME', 'UNKNOWN')
                    object_type = item.get('OBJECT_TYPE', 'UNKNOWN')
                    
                    # Get orbital parameters if available
                    inclination = float(item.get('INCLINATION', 0))
                    period = float(item.get('PERIOD', 90))  # minutes
                    apogee = float(item.get('APOGEE', 500))  # km
                    perigee = float(item.get('PERIGEE', 500))  # km
                    
                    # Calculate altitude as average of apogee and perigee
                    altitude = (apogee + perigee) / 2
                    
                    # Calculate approximate position
                    # This is a simplified model
                    theta = time.time() % (24*3600) / (24*3600) * 2 * np.pi
                    phi = inclination * np.pi / 180.0
                    
                    r = altitude + 6371  # Earth radius + altitude
                    x = r * np.cos(phi) * np.cos(theta)
                    y = r * np.cos(phi) * np.sin(theta)
                    z = r * np.sin(phi)
                    
                    # Calculate latitude and longitude
                    lat = np.arcsin(z / r) * 180 / np.pi
                    lon = np.arctan2(y, x) * 180 / np.pi
                    
                    # Calculate orbital velocity (circular orbit approximation)
                    velocity = np.sqrt(398600.4418 / r)
                    
                    # Determine object size (approximation based on object type)
                    if object_type == 'PAYLOAD':
                        size = np.random.uniform(1.0, 10.0)
                    elif object_type == 'ROCKET BODY':
                        size = np.random.uniform(3.0, 15.0)
                    else:  # DEBRIS or other
                        size = np.random.uniform(0.1, 1.0)
                    
                    # Calculate risk score based on altitude, inclination, and object type
                    # Lower orbits and higher inclinations have higher collision risks
                    altitude_factor = max(0, min(1, 1 - (altitude - 300) / 1000))
                    inclination_factor = inclination / 180.0
                    
                    # Object type risk factor
                    if object_type == 'PAYLOAD':
                        type_factor = 0.3  # Lower risk, usually controlled
                    elif object_type == 'ROCKET BODY':
                        type_factor = 0.7  # Higher risk, usually uncontrolled
                    else:  # DEBRIS or other
                        type_factor = 0.9  # Highest risk
                    
                    risk_score = 0.4 * altitude_factor + 0.3 * inclination_factor + 0.3 * type_factor
                    
                    transformed_data.append({
                        'id': f"ST-{norad_id}",
                        'altitude': float(altitude),
                        'latitude': float(lat),
                        'longitude': float(lon),
                        'x': float(x),
                        'y': float(y),
                        'z': float(z),
                        'size': float(size),
                        'velocity': float(velocity),
                        'inclination': float(inclination),
                        'risk_score': float(risk_score),
                        'last_updated': datetime.now()
                    })
                except (ValueError, KeyError, TypeError) as e:
                    print(f"Error processing object {item.get('NORAD_CAT_ID')}: {str(e)}")
                    continue

            return transformed_data

        except Exception as e:
            print(f"Error fetching space debris data: {str(e)}")
            raise