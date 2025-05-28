import streamlit as st
# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Space Debris Tracker",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import time
from datetime import datetime
from components.globe import create_globe
from components.sidebar import create_sidebar
from components.alerts import show_alerts
from utils.orbital_calculations import check_collisions, hmm_model, pnn_model # Added imports
from utils.database import init_db, get_db, SpaceDebris, populate_real_data
# Import the actual model implementations
from utils.hmm_model import TrajectoryHMM
from utils.pnn_model import RiskClassifier
from utils.validation import validate_and_update # Added import for validation
from utils.csv_importer import import_csv_data # Import the CSV data importer

# Initialize models with real implementations
def initialize_models():
    global hmm_model, pnn_model  # Declare as global to modify
    hmm_model = TrajectoryHMM(n_components=3)  # Real HMM initialization
    pnn_model = RiskClassifier(input_dim=10)   # Real PNN initialization

def get_debris_data():
    db = next(get_db())
    debris_data = [
        {
            'id': debris.id,
            'altitude': debris.altitude,
            'latitude': debris.latitude,
            'longitude': debris.longitude,
            'x': debris.x,
            'y': debris.y,
            'z': debris.z,
            'size': debris.size,
            'velocity': debris.velocity,
            'inclination': debris.inclination,
            'risk_score': debris.risk_score, # Placeholder - should use PNN prediction
            'last_updated': debris.last_updated
        }
        for debris in db.query(SpaceDebris).all()
    ]
    return debris_data

# Function to train models on CSV data
def train_models_with_csv_data(csv_files):
    """
    Train the HMM and PNN models using data from CSV files.
    
    Args:
        csv_files: List of paths to CSV files
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Import CSV data
        debris_data = import_csv_data(csv_files)
        
        if not debris_data or len(debris_data) < 50:
            print(f"Not enough data for training: {len(debris_data)} objects")
            return False
            
        # Prepare training data for HMM (trajectory data)
        hmm_training_data = []
        for debris in debris_data:
            # Create feature vector for trajectory prediction
            feature_vector = [
                debris['x'], debris['y'], debris['z'],
                debris['velocity'], debris['inclination']
            ]
            hmm_training_data.append(feature_vector)
            
        # Prepare training data for PNN (risk prediction)
        pnn_training_data = []
        pnn_labels = []
        for debris in debris_data:
            # Create feature vector for risk prediction
            feature_vector = [
                debris['altitude'], debris['velocity'], 
                debris['inclination'], debris['size'],
                debris['latitude'], debris['longitude']
            ]
            pnn_training_data.append(feature_vector)
            pnn_labels.append(1 if debris['risk_score'] > 0.5 else 0)  # Binary risk classification
            
        # Train the models
        print(f"Training HMM with {len(hmm_training_data)} samples...")
        hmm_success = hmm_model.train(hmm_training_data)
        
        print(f"Training PNN with {len(pnn_training_data)} samples...")
        pnn_success = pnn_model.train(pnn_training_data, pnn_labels)
        
        # Update session state
        st.session_state.hmm_training_attempted = True
        st.session_state.pnn_training_attempted = True
        
        return hmm_success and pnn_success
        
    except Exception as e:
        print(f"Error training models with CSV data: {str(e)}")
        return False

# Initialize database
init_db()

try:
    # Try to populate with real data first
    with st.spinner("Initializing space debris tracking system..."):
        populate_real_data()
except Exception as e:
    st.error(f"Error initializing tracking system: {str(e)}. Using backup data.")

# Custom CSS
with open('styles/custom.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>Space Debris Tracking Dashboard</h1>", unsafe_allow_html=True)

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'csv_files_for_training' not in st.session_state:
    st.session_state.csv_files_for_training = []

# Initialize the HMM and PNN models with pre-training
initialize_models()

# Track model training status
if "hmm_training_attempted" not in st.session_state:
    st.session_state.hmm_training_attempted = False

if "pnn_training_attempted" not in st.session_state:
    st.session_state.pnn_training_attempted = False

# Force model status to be displayed initially
if "model_status_shown" in st.session_state:
    st.session_state.model_status_shown = False

# Display model status in sidebar
if "model_status_shown" not in st.session_state:
    st.session_state.model_status_shown = False

# Get the debris data
debris_data = get_debris_data()

# Show model status in a small box on the side
status_col1, status_col2 = st.columns([3, 1])
with status_col2:
    with st.expander("Model Status", expanded=not st.session_state.model_status_shown):
        st.markdown("### Hybrid Model Status")

        if hmm_model and hmm_model.is_trained:
            st.success("HMM: Trained and Active")
        else:
            st.info("HMM: Initializing (needs more data)")

        if pnn_model and pnn_model.is_trained:
            st.success("PNN: Trained and Active")
        else:
            st.info("PNN: Initializing (needs more data)")
            
        # Add option to train models with CSV data
        if st.session_state.csv_files_for_training:
            if st.button("Train Models with CSV Data"):
                with st.spinner("Training models with CSV data..."):
                    success = train_models_with_csv_data(st.session_state.csv_files_for_training)
                    if success:
                        st.success("Models successfully trained with CSV data!")
                        st.rerun()  # Refresh to update model status
                    else:
                        st.error("Failed to train models with CSV data")

        st.session_state.model_status_shown = True


# Main layout
col1, col2 = st.columns([7, 3])

with col1:
    # Globe visualization
    globe_fig = create_globe(debris_data)
    st.plotly_chart(globe_fig, use_container_width=True)

    # Statistics
    st.markdown("<h2 class='section-header'>Tracking Statistics</h2>", unsafe_allow_html=True)
    stats_cols = st.columns(4)
    with stats_cols[0]:
        st.metric("Total Objects", len(debris_data))
    with stats_cols[1]:
        st.metric("High Risk", len([d for d in debris_data if d['risk_score'] > 0.7]))
    with stats_cols[2]:
        st.metric("Medium Risk", len([d for d in debris_data if 0.3 < d['risk_score'] <= 0.7]))
    with stats_cols[3]:
        st.metric("Low Risk", len([d for d in debris_data if d['risk_score'] <= 0.3]))

    # Data source info
    st.info("Data Source: NASA Orbital Parameters - Real-time space debris simulation")

    # Update data every 3 minutes (reduced from 5)
    current_time = time.time()
    if current_time - st.session_state.last_update > 180:  # 3 minutes in seconds
        st.session_state.last_update = current_time

        # Log the auto-refresh
        st.toast("Auto-refreshing debris data...")

        # Fetch new data from NASA
        try:
            with st.spinner("Refreshing space debris data..."):
                populate_real_data()  # This updates debris_data in the database
                debris_data = get_debris_data()  # Get the updated data
                st.success("Data refreshed automatically")
        except Exception as e:
            st.error(f"Error refreshing data: {str(e)}")

        # Force page reload
        st.rerun()

with col2:
    # Alerts section
    st.markdown("<h2 class='section-header'>Collision Alerts</h2>", unsafe_allow_html=True)
    collision_risks = check_collisions(debris_data)
    show_alerts(collision_risks) #Modified to include HMM predictions

# Sidebar
create_sidebar(debris_data)