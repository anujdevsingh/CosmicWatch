from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from .nasa_client import NASAClient
from .space_track import SpaceTrackClient
from .mock_data import get_debris_data
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Use SQLite as fallback if DATABASE_URL is not set
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///space_debris.db"
    print(f"DATABASE_URL not found in environment. Using default SQLite database: {DATABASE_URL}")

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

class SpaceDebris(Base):
    """Model for space debris objects."""
    __tablename__ = "space_debris"

    id = Column(String, primary_key=True)
    altitude = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    z = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    velocity = Column(Float, nullable=False)
    inclination = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    last_updated = Column(DateTime, default=func.now())

def init_db():
    """Initialize the database, creating all tables."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def populate_real_data():
    """Populate database with real space debris data from Space-Track.org."""
    print("Starting real data population from Space-Track.org...")
    try:
        # First try to get data from Space-Track.org
        try:
            client = SpaceTrackClient()
            print("Created Space-Track client, fetching data...")
            debris_data = client.get_latest_debris_data(limit=1500)
            print(f"Successfully fetched {len(debris_data)} objects from Space-Track.org")
            
            # If we got too few objects, fall back to NASA data
            if len(debris_data) < 100:
                print(f"Only received {len(debris_data)} objects from Space-Track.org, which is too few. Falling back to NASA synthetic data...")
                client = NASAClient()
                print("Created NASA client, fetching data...")
                debris_data = client.get_debris_data()
                print(f"Successfully fetched {len(debris_data)} objects from NASA")
        except Exception as space_track_error:
            print(f"Error fetching data from Space-Track.org: {str(space_track_error)}")
            print("Falling back to NASA synthetic data...")
            client = NASAClient()
            print("Created NASA client, fetching data...")
            debris_data = client.get_debris_data()
            print(f"Successfully fetched {len(debris_data)} objects from NASA")

        db = next(get_db())
        print("Connected to database")

        # Clear existing data
        existing_count = db.query(SpaceDebris).count()
        db.query(SpaceDebris).delete()
        print(f"Cleared {existing_count} existing records")

        # Add new data with better error handling
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
            print(f"Successfully updated database with {success_count} objects")
            print(f"Skipped {error_count} objects due to errors")
            
            # Only fall back to mock data if we didn't get any real data
            if success_count == 0:
                print("No Space-Track data was added, falling back to mock data...")
                populate_test_data()
                return False
            return True
            
        except Exception as commit_error:
            print(f"Final commit error: {str(commit_error)}")
            db.rollback()
            if success_count == 0:
                print("No Space-Track data was added, falling back to mock data...")
                populate_test_data()
                return False
            return True

    except Exception as e:
        print(f"Error updating database with Space-Track data: {str(e)}")
        if 'db' in locals():
            db.rollback()
        print("Falling back to mock data...")
        populate_test_data()
        return False

def populate_test_data():
    """Populate database with test data if empty."""
    print("Starting test data population...")

    db = next(get_db())
    if db.query(SpaceDebris).count() == 0:
        mock_data = get_debris_data()
        for item in mock_data:
            debris = SpaceDebris(**item)
            db.add(debris)
        db.commit()
        print(f"Populated database with {len(mock_data)} mock objects")