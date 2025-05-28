"""
Test script to fetch data from Space-Track.org and store it in a separate SQLite database.
This script is for testing purposes only and doesn't affect the main project.
"""

import os
import sys
from dotenv import load_dotenv
from space_track_data import SpaceTrackClient, SpaceDebrisDatabase

# Load environment variables
load_dotenv()

def main():
    """Main function to test Space-Track data fetching and storage."""
    print("Space Debris Data Test Fetcher")
    print("-----------------------------")
    
    # Check if environment variables are set
    username = os.getenv("SPACETRACK_USERNAME")
    password = os.getenv("SPACETRACK_PASSWORD")
    
    if not username or not password:
        print("Error: SPACETRACK_USERNAME and SPACETRACK_PASSWORD must be set in .env file")
        return False
    
    print(f"Using Space-Track credentials for: {username}")
    
    try:
        # Create client and fetch data
        print("\nFetching data from Space-Track.org (test mode)...")
        client = SpaceTrackClient()
        debris_data = client.get_latest_debris_data(limit=100)  # Smaller limit for testing
        
        if not debris_data:
            print("No data retrieved from Space-Track.org")
            return False
        
        # Store data in database
        print("\nStoring data in test database...")
        db = SpaceDebrisDatabase()
        if db.connect():
            db.create_tables()
            success = db.store_debris_data(debris_data)
            
            if success:
                print("\nData successfully stored in test database.")
                
                # Query and display some data
                print("\nRetrieving sample data from database...")
                # Get a few records
                sample_data = db.get_debris_data(limit=5)
                
                # Display the data
                print("\nSample data (5 records):")
                for i, item in enumerate(sample_data):
                    print(f"\nRecord {i+1}:")
                    print(f"  ID: {item['id']}")
                    print(f"  Object Name: {item['object_name']}")
                    print(f"  Object Type: {item['object_type']}")
                    print(f"  Altitude: {item['altitude']:.2f} km")
                    print(f"  Inclination: {item['inclination']:.2f}°")
                    print(f"  Risk Score: {item['risk_score']:.2f}")
                
                # Export data to files
                print("\nExporting data to CSV and JSON files...")
                db.export_to_csv()
                db.export_to_json()
                
                db.close()
                return True
            else:
                print("\nFailed to store data in database.")
                db.close()
                return False
        else:
            print("\nFailed to connect to database.")
            return False
    except Exception as e:
        print(f"\nError in test_fetch: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 