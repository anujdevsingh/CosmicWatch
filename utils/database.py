from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func, text
from sqlalchemy.orm import declarative_base, sessionmaker
import os
import logging
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("cosmicwatch.db")
_progress_logs = os.getenv("COSMICWATCH_REFRESH_PROGRESS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}

# Get database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Use SQLite as fallback if DATABASE_URL is not set
if not DATABASE_URL:
    # Get absolute path to the database in CosmicWatch folder
    COSMICWATCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH = os.path.join(COSMICWATCH_DIR, "space_debris.db")
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    logger.info("Using SQLite database db_path=%s", DB_PATH)

# Create engine and session
_engine_kwargs = {}
if DATABASE_URL.startswith("sqlite:///"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

@contextmanager
def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
    
    # Data freshness tracking
    celestrak_last_modified = Column(DateTime, nullable=True)  # When CelesTrak data was last updated
    data_source = Column(String, default='celestrak')  # Track data source
    
    # AI prediction caching for performance
    ai_risk_level = Column(String, nullable=True)  # CRITICAL/HIGH/MEDIUM/LOW
    ai_confidence = Column(Float, nullable=True)   # AI prediction confidence
    ai_last_predicted = Column(DateTime, nullable=True)  # When AI last analyzed this object
    ai_enhanced = Column(Integer, default=0)  # 1 if AI-enhanced, 0 if not

class DatabaseMetadata(Base):
    """Track database-wide metadata for smart caching and updates."""
    __tablename__ = "database_metadata"
    
    key = Column(String, primary_key=True)
    value = Column(String, nullable=True)
    updated_at = Column(DateTime, default=func.now())
    
    # Common keys we'll use:
    # 'last_celestrak_download' - When we last downloaded from CelesTrak
    # 'total_objects' - Number of objects in database
    # 'data_version' - Version of the data
    # 'ai_cache_version' - Version of AI predictions

def migrate_database_for_ai_caching():
    """
    OPTIONAL: Migrate database to add AI caching columns for better performance.
    This is safe to run and will improve loading speed significantly.
    """
    try:
        logger.info("Checking if database migration is needed")
        
        # Check if migration is needed
        with get_db_session() as db:
            try:
                db.execute(text("SELECT ai_risk_level FROM space_debris LIMIT 1"))
                logger.info("Database already has AI caching columns")
                return True
            except Exception:
                logger.info("AI caching columns not found - migration available")
        
        # Perform safe migration
        logger.info("Starting safe database migration")
        
        # Add AI caching columns
        migration_queries = [
            "ALTER TABLE space_debris ADD COLUMN ai_risk_level TEXT DEFAULT NULL",
            "ALTER TABLE space_debris ADD COLUMN ai_confidence REAL DEFAULT NULL", 
            "ALTER TABLE space_debris ADD COLUMN ai_last_predicted DATETIME DEFAULT NULL",
            "ALTER TABLE space_debris ADD COLUMN ai_enhanced INTEGER DEFAULT 0"
        ]
        
        with get_db_session() as db:
            for query in migration_queries:
                try:
                    db.execute(text(query))
                except Exception as e:
                    if "duplicate column name" not in str(e).lower():
                        raise
            db.commit()
            for query in migration_queries:
                logger.info("Executed migration query=%s", query)
        logger.info("Database migration completed successfully")
        
        return True
        
    except Exception as e:
        logger.exception("Migration failed: %s", e)
        return False

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
    """Populate database with comprehensive real-time space debris data from CelesTrak."""
    logger.info("Starting CelesTrak data population")
    try:
        # Import CelesTrak client
        from utils.celestrak_client import fetch_celestrak_data
        
        logger.info("Fetching satellite catalog from CelesTrak")
        
        # Get comprehensive global data including all satellites and debris
        debris_data = fetch_celestrak_data(
            include_debris=True,     # Include specific debris data
            include_starlink=True    # Include Starlink constellation
        )
        
        logger.info("Fetched celestrak_objects=%s", len(debris_data))
        
        if len(debris_data) < 1000:
            logger.warning("Only received celestrak_objects=%s expected=10000+", len(debris_data))
            raise Exception("Insufficient CelesTrak data - API may be unavailable")
        
        with get_db_session() as db:
            logger.info("Connected to database")

            # Clear existing data
            existing_count = db.query(SpaceDebris).count()
            db.query(SpaceDebris).delete()
            logger.info("Cleared existing_records=%s", existing_count)

            # Add new CelesTrak data with comprehensive error handling
            success_count = 0
            error_count = 0
        
            logger.info("Storing CelesTrak data")
        
            for i, item in enumerate(debris_data):
                try:
                    # Map CelesTrak data to our database schema
                    debris_record = {
                        'id': item.get('id', f"CT-{i}"),
                        'altitude': float(item.get('altitude', 400)),
                        'latitude': float(item.get('latitude', 0)),
                        'longitude': float(item.get('longitude', 0)),
                        'x': float(item.get('x', 0)),
                        'y': float(item.get('y', 0)),
                        'z': float(item.get('z', 0)),
                        'size': float(item.get('size', 1.0)),
                        'velocity': float(item.get('velocity', 7.8)),
                        'inclination': float(item.get('inclination', 0)),
                        'risk_score': float(item.get('risk_score', 0.5)),
                        'last_updated': datetime.fromisoformat(item.get('last_updated', datetime.now().isoformat()).replace('Z', '+00:00')) if isinstance(item.get('last_updated'), str) else item.get('last_updated', datetime.now())
                    }
                    
                    debris = SpaceDebris(**debris_record)
                    db.add(debris)
                    success_count += 1
                    
                    # Commit in batches for performance
                    if success_count % 500 == 0:
                        db.commit()
                        if _progress_logs:
                            logger.info("Committed records=%s", success_count)
                        
                except Exception as item_error:
                    error_count += 1
                    logger.debug("Error adding debris item index=%s error=%s", i, item_error)
                    # Skip this item but continue with others
                    db.rollback()
                
            # Final commit for remaining items
            try:
                db.commit()
                logger.info("Updated database objects=%s errors=%s", success_count, error_count)
                if error_count > 0:
                    logger.warning("Skipped objects_due_to_errors=%s", error_count)
                
                return True
                
            except Exception as commit_error:
                logger.exception("Final commit error: %s", commit_error)
                db.rollback()
                raise Exception(f"Failed to commit CelesTrak data: {str(commit_error)}")

    except Exception as e:
        logger.exception("Error updating database with CelesTrak data: %s", e)
        if 'db' in locals():
            db.rollback()
        # NO FALLBACK TO MOCK DATA - System requires real data only
        raise Exception(f"CelesTrak data population failed: {str(e)}. No fallback available.")

def get_metadata_value(key, default=None):
    """Get a metadata value from the database."""
    try:
        with get_db_session() as db:
            meta = db.query(DatabaseMetadata).filter(DatabaseMetadata.key == key).first()
            return meta.value if meta else default
    except Exception as e:
        logger.debug("Error getting metadata key=%s error=%s", key, e)
        return default

def set_metadata_value(key, value):
    """Set a metadata value in the database."""
    try:
        with get_db_session() as db:
            meta = db.query(DatabaseMetadata).filter(DatabaseMetadata.key == key).first()
            if meta:
                meta.value = str(value)
                meta.updated_at = datetime.now()
            else:
                meta = DatabaseMetadata(key=key, value=str(value))
                db.add(meta)
            db.commit()
            return True
    except Exception as e:
        logger.debug("Error setting metadata key=%s error=%s", key, e)
        return False

def is_data_fresh(max_age_hours=2):
    """Check if the database data is fresh enough to avoid re-downloading."""
    try:
        last_download = get_metadata_value('last_celestrak_download')
        if not last_download:
            logger.info("No previous download timestamp found")
            return False
        
        last_download_time = datetime.fromisoformat(last_download)
        age_hours = (datetime.now() - last_download_time).total_seconds() / 3600
        
        is_fresh = age_hours < max_age_hours
        logger.info("Data age_hours=%.1f max_age_hours=%s is_fresh=%s", age_hours, max_age_hours, is_fresh)
        return is_fresh
        
    except Exception as e:
        logger.debug("Error checking data freshness: %s", e)
        return False

def get_cached_objects_count():
    """Get the number of objects currently in the database."""
    try:
        with get_db_session() as db:
            return db.query(SpaceDebris).count()
    except Exception as e:
        logger.debug("Error counting cached objects: %s", e)
        return 0

def populate_real_data_smart():
    """Smart data population that only downloads when necessary."""
    logger.info("Checking if data refresh is needed")
    
    # Check if we have existing fresh data
    cached_count = get_cached_objects_count()
    
    if cached_count > 0:
        logger.info("Found cached objects=%s", cached_count)
        
        if is_data_fresh(max_age_hours=2):
            logger.info("Using cached data")
            # Update metadata to track this access
            set_metadata_value('last_access', datetime.now().isoformat())
            return True
        else:
            logger.info("Cached data is stale - refreshing from CelesTrak")
    else:
        logger.info("No cached data found - initial download from CelesTrak")
    
    # Data refresh needed - download from CelesTrak
    try:
        logger.info("Starting CelesTrak data download")
        return populate_real_data_force_refresh()
    except Exception as e:
        if cached_count > 0:
            logger.warning("CelesTrak download failed using cached objects=%s error=%s", cached_count, e)
            return True
        else:
            raise Exception(f"No cached data available and CelesTrak download failed: {e}")

def populate_real_data_force_refresh():
    """Force refresh data from CelesTrak (original populate_real_data logic)."""
    logger.info("Starting CelesTrak data population (force refresh)")
    try:
        # Import CelesTrak client
        from utils.celestrak_client import fetch_celestrak_data
        
        logger.info("Fetching satellite catalog from CelesTrak")
        
        # Get comprehensive global data including all satellites and debris
        debris_data = fetch_celestrak_data(
            include_debris=True,     # Include specific debris data
            include_starlink=True    # Include Starlink constellation
        )
        
        logger.info("Fetched celestrak_objects=%s", len(debris_data))
        
        if len(debris_data) < 1000:
            logger.warning("Only received celestrak_objects=%s expected=10000+", len(debris_data))
            raise Exception("Insufficient CelesTrak data - API may be unavailable")
        
        with get_db_session() as db:
            logger.info("Connected to database")

            # Clear existing data (only when doing force refresh)
            existing_count = db.query(SpaceDebris).count()
            db.query(SpaceDebris).delete()
            logger.info("Cleared existing_records=%s", existing_count)

            # Add new CelesTrak data with comprehensive error handling
            success_count = 0
            error_count = 0
        
            logger.info("Storing CelesTrak data")
        
            for i, item in enumerate(debris_data):
                try:
                    # Map CelesTrak data to our database schema
                    debris_record = {
                        'id': item.get('id', f"CT-{i}"),
                        'altitude': float(item.get('altitude', 400)),
                        'latitude': float(item.get('latitude', 0)),
                        'longitude': float(item.get('longitude', 0)),
                        'x': float(item.get('x', 0)),
                        'y': float(item.get('y', 0)),
                        'z': float(item.get('z', 0)),
                        'size': float(item.get('size', 1.0)),
                        'velocity': float(item.get('velocity', 7.8)),
                        'inclination': float(item.get('inclination', 0)),
                        'risk_score': float(item.get('risk_score', 0.5)),
                        'last_updated': datetime.fromisoformat(item.get('last_updated', datetime.now().isoformat()).replace('Z', '+00:00')) if isinstance(item.get('last_updated'), str) else item.get('last_updated', datetime.now()),
                        'celestrak_last_modified': datetime.now(),  # Track when we got this from CelesTrak
                        'data_source': 'celestrak'
                    }
                    
                    debris = SpaceDebris(**debris_record)
                    db.add(debris)
                    success_count += 1
                    
                    # Commit in batches for performance
                    if success_count % 500 == 0:
                        db.commit()
                        if _progress_logs:
                            logger.info("Committed records=%s", success_count)
                        
                except Exception as item_error:
                    error_count += 1
                    logger.debug("Error adding debris item index=%s error=%s", i, item_error)
                    # Skip this item but continue with others
                    db.rollback()
                
            # Final commit for remaining items
            try:
                db.commit()
                
                # Update metadata to track successful download
                set_metadata_value('last_celestrak_download', datetime.now().isoformat())
                set_metadata_value('total_objects', str(success_count))
                set_metadata_value('data_version', f"celestrak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                logger.info("Updated database objects=%s errors=%s", success_count, error_count)
                if error_count > 0:
                    logger.warning("Skipped objects_due_to_errors=%s", error_count)
                
                return True
                
            except Exception as commit_error:
                logger.exception("Final commit error: %s", commit_error)
                db.rollback()
                raise Exception(f"Failed to commit CelesTrak data: {str(commit_error)}")

    except Exception as e:
        logger.exception("CelesTrak data population failed: %s", e)
        # NO FALLBACK TO MOCK DATA - System requires real data only
        raise Exception(f"CelesTrak data population failed: {str(e)}. No fallback available.")
