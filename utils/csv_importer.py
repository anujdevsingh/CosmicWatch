import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import os

def import_csv_data(csv_files: List[str], validate: bool = True) -> List[Dict[Any, Any]]:
    """
    Import space debris data from CSV files and transform it to match the database schema.
    
    Args:
        csv_files: List of paths to CSV files containing space debris data
        validate: Whether to validate and clean the data
        
    Returns:
        List of dictionaries containing transformed space debris data
    """
    all_data = []
    
    for file_path in csv_files:
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            continue
            
        try:
            print(f"Importing data from {file_path}...")
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check for required columns
            required_columns = ['id', 'altitude', 'latitude', 'longitude', 'inclination']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                # Try alternative column names (common variations)
                column_alternatives = {
                    'id': ['object_id', 'norad_id', 'catalog_id'],
                    'altitude': ['alt', 'height', 'mean_altitude'],
                    'latitude': ['lat'],
                    'longitude': ['lon', 'long'],
                    'inclination': ['inc', 'incl']
                }
                
                # Map alternative column names
                for missing_col in missing_columns[:]:  # Copy to avoid modifying during iteration
                    for alt_col in column_alternatives.get(missing_col, []):
                        if alt_col in df.columns:
                            df.rename(columns={alt_col: missing_col}, inplace=True)
                            missing_columns.remove(missing_col)
                            break
            
            # If still missing required columns, skip this file
            if missing_columns:
                print(f"Error: Missing required columns in {file_path}: {missing_columns}")
                continue
                
            # Transform the data to match our schema
            transformed_data = []
            
            for _, row in df.iterrows():
                try:
                    # Extract basic parameters
                    object_id = str(row.get('id', f"CSV-{len(transformed_data):04d}"))
                    
                    # Get basic parameters with defaults
                    altitude = float(row.get('altitude', 500))
                    latitude = float(row.get('latitude', 0))
                    longitude = float(row.get('longitude', 0))
                    inclination = float(row.get('inclination', 0))
                    
                    # Calculate x, y, z coordinates if not provided
                    if 'x' not in row or 'y' not in row or 'z' not in row:
                        # Convert lat/lon to radians
                        lat_rad = np.radians(latitude)
                        lon_rad = np.radians(longitude)
                        
                        # Calculate cartesian coordinates
                        r = altitude + 6371  # Earth radius + altitude
                        x = r * np.cos(lat_rad) * np.cos(lon_rad)
                        y = r * np.cos(lat_rad) * np.sin(lon_rad)
                        z = r * np.sin(lat_rad)
                    else:
                        x = float(row.get('x'))
                        y = float(row.get('y'))
                        z = float(row.get('z'))
                    
                    # Get or estimate velocity
                    if 'velocity' in row:
                        velocity = float(row.get('velocity'))
                    else:
                        # Calculate orbital velocity (circular orbit approximation)
                        r = altitude + 6371  # Earth radius + altitude
                        velocity = float(np.sqrt(398600.4418 / r))  # km/s
                    
                    # Get or estimate size
                    size = float(row.get('size', 1.0))
                    
                    # Get or calculate risk score
                    if 'risk_score' in row:
                        risk_score = float(row.get('risk_score'))
                    else:
                        # Calculate risk based on altitude and inclination
                        altitude_factor = max(0, min(1, 1 - (altitude - 300) / 1000))
                        inclination_factor = inclination / 180.0
                        risk_score = 0.7 * altitude_factor + 0.3 * inclination_factor
                    
                    # Get last updated time or use current time
                    if 'last_updated' in row:
                        last_updated = row.get('last_updated')
                    else:
                        last_updated = datetime.now()
                    
                    transformed_data.append({
                        'id': object_id,
                        'altitude': altitude,
                        'latitude': latitude,
                        'longitude': longitude,
                        'x': float(x),
                        'y': float(y),
                        'z': float(z),
                        'size': size,
                        'velocity': velocity,
                        'inclination': inclination,
                        'risk_score': risk_score,
                        'last_updated': last_updated
                    })
                    
                except (ValueError, KeyError, TypeError) as e:
                    print(f"Error processing row: {str(e)}")
                    continue
            
            print(f"Successfully transformed {len(transformed_data)} rows from {file_path}")
            all_data.extend(transformed_data)
            
        except Exception as e:
            print(f"Error importing CSV file {file_path}: {str(e)}")
            continue
    
    return all_data

def populate_from_csv(csv_files: List[str]) -> bool:
    """
    Import data from CSV files and populate the database.
    
    Args:
        csv_files: List of paths to CSV files containing space debris data
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Import required functions here to avoid circular imports
        from utils.database import get_db, SpaceDebris
        
        # Import and transform CSV data
        debris_data = import_csv_data(csv_files)
        
        if not debris_data:
            print("No valid data found in CSV files")
            return False
        
        # Store data in database
        db = next(get_db())
        
        # Optionally clear existing data
        existing_count = db.query(SpaceDebris).count()
        db.query(SpaceDebris).delete()
        print(f"Cleared {existing_count} existing records")
        
        # Add new data from CSV
        success_count = 0
        error_count = 0
        
        for item in debris_data:
            try:
                debris = SpaceDebris(**item)
                db.add(debris)
                success_count += 1
                
                # Commit in smaller batches to avoid transaction size issues
                if success_count % 100 == 0:
                    db.commit()
                    print(f"Committed {success_count} records so far")
            except Exception as item_error:
                error_count += 1
                print(f"Error adding debris item: {str(item_error)}")
                # Skip this item but continue with others
                db.rollback()
        
        # Final commit for remaining items
        try:
            db.commit()
            print(f"Successfully updated database with {success_count} objects from CSV")
            print(f"Skipped {error_count} objects due to errors")
            return True
        except Exception as commit_error:
            print(f"Final commit error: {str(commit_error)}")
            db.rollback()
            return False
            
    except Exception as e:
        print(f"Error in populate_from_csv: {str(e)}")
        return False 