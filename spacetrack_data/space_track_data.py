import os
import requests
from datetime import datetime
from typing import List, Dict, Any
import time
import numpy as np
import sqlite3
import json
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SpaceTrackClient:
    """Client for interacting with Space-Track.org API."""

    BASE_URL = "https://www.space-track.org"

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
                        'last_updated': datetime.now().isoformat(),
                        'object_name': object_name,
                        'object_type': object_type,
                        'apogee': float(apogee),
                        'perigee': float(perigee),
                        'period': float(period)
                    })
                except (ValueError, KeyError, TypeError) as e:
                    print(f"Error processing object {item.get('NORAD_CAT_ID')}: {str(e)}")
                    continue

            return transformed_data

        except Exception as e:
            print(f"Error fetching space debris data: {str(e)}")
            raise

class SpaceDebrisDatabase:
    """Class for managing space debris data in a SQLite database."""
    
    def __init__(self, db_path="spacetrack_data/space_debris_test.db"):
        """Initialize the database connection."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print(f"Connected to database: {self.db_path}")
            return True
        except sqlite3.Error as e:
            print(f"Error connecting to database: {str(e)}")
            return False
            
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed")
            
    def create_tables(self):
        """Create the necessary tables if they don't exist."""
        try:
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS space_debris (
                id TEXT PRIMARY KEY,
                altitude REAL NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                size REAL NOT NULL,
                velocity REAL NOT NULL,
                inclination REAL NOT NULL,
                risk_score REAL NOT NULL,
                last_updated TEXT NOT NULL,
                object_name TEXT,
                object_type TEXT,
                apogee REAL,
                perigee REAL,
                period REAL
            )
            ''')
            
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_fetches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fetch_time TEXT NOT NULL,
                num_objects INTEGER NOT NULL,
                source TEXT NOT NULL
            )
            ''')
            
            self.conn.commit()
            print("Tables created successfully")
            return True
        except sqlite3.Error as e:
            print(f"Error creating tables: {str(e)}")
            return False
            
    def store_debris_data(self, debris_data):
        """Store space debris data in the database."""
        try:
            # Clear existing data
            self.cursor.execute("DELETE FROM space_debris")
            print(f"Cleared existing records")
            
            # Insert new data
            for item in debris_data:
                placeholders = ', '.join(['?'] * len(item))
                columns = ', '.join(item.keys())
                sql = f"INSERT INTO space_debris ({columns}) VALUES ({placeholders})"
                self.cursor.execute(sql, list(item.values()))
                
            # Record this fetch
            self.cursor.execute(
                "INSERT INTO data_fetches (fetch_time, num_objects, source) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), len(debris_data), "Space-Track.org")
            )
            
            self.conn.commit()
            print(f"Successfully stored {len(debris_data)} objects in the database")
            return True
        except sqlite3.Error as e:
            print(f"Error storing data: {str(e)}")
            self.conn.rollback()
            return False
            
    def get_debris_data(self, limit=None):
        """Retrieve space debris data from the database."""
        try:
            if limit:
                self.cursor.execute("SELECT * FROM space_debris LIMIT ?", (limit,))
            else:
                self.cursor.execute("SELECT * FROM space_debris")
                
            columns = [description[0] for description in self.cursor.description]
            result = []
            
            for row in self.cursor.fetchall():
                item = dict(zip(columns, row))
                result.append(item)
                
            print(f"Retrieved {len(result)} objects from database")
            return result
        except sqlite3.Error as e:
            print(f"Error retrieving data: {str(e)}")
            return []
            
    def export_to_csv(self, filename="spacetrack_data/space_debris_export.csv"):
        """Export the database contents to a CSV file."""
        try:
            data = self.get_debris_data()
            if data:
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                print(f"Data exported to {filename}")
                return True
            else:
                print("No data to export")
                return False
        except Exception as e:
            print(f"Error exporting data: {str(e)}")
            return False
            
    def export_to_json(self, filename="spacetrack_data/space_debris_export.json"):
        """Export the database contents to a JSON file."""
        try:
            data = self.get_debris_data()
            if data:
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Data exported to {filename}")
                return True
            else:
                print("No data to export")
                return False
        except Exception as e:
            print(f"Error exporting data: {str(e)}")
            return False

def fetch_and_store_data(limit=1000):
    """Fetch data from Space-Track.org and store it in the database."""
    try:
        # Create client and fetch data
        client = SpaceTrackClient()
        debris_data = client.get_latest_debris_data(limit=limit)
        
        if not debris_data:
            print("No data retrieved from Space-Track.org")
            return False
            
        # Store data in database
        db = SpaceDebrisDatabase()
        if db.connect():
            db.create_tables()
            success = db.store_debris_data(debris_data)
            db.close()
            return success
        else:
            return False
    except Exception as e:
        print(f"Error in fetch_and_store_data: {str(e)}")
        return False

def export_data_to_files():
    """Export the database contents to CSV and JSON files."""
    db = SpaceDebrisDatabase()
    if db.connect():
        db.export_to_csv()
        db.export_to_json()
        db.close()
        return True
    else:
        return False

if __name__ == "__main__":
    print("Space Debris Data Fetcher")
    print("------------------------")
    
    # Fetch and store data
    print("\nFetching data from Space-Track.org...")
    if fetch_and_store_data(limit=1000):
        print("\nData successfully fetched and stored.")
        
        # Export data to files
        print("\nExporting data to files...")
        if export_data_to_files():
            print("\nData successfully exported to CSV and JSON files.")
        else:
            print("\nFailed to export data to files.")
    else:
        print("\nFailed to fetch and store data.")