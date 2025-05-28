import streamlit as st
import time
from datetime import datetime
from components.globe import create_globe
from components.sidebar import create_sidebar
from components.alerts import show_alerts
from utils.orbital_calculations import check_collisions
from utils.mock_data import get_debris_data

# Page configuration
st.set_page_config(
    page_title="Space Debris Tracker",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Main layout
col1, col2 = st.columns([7, 3])

with col1:
    # Globe visualization
    debris_data = get_debris_data()
    globe_fig = create_globe(debris_data)
    st.plotly_chart(globe_fig, use_container_width=True)

    # Update data every 5 minutes
    if time.time() - st.session_state.last_update > 300:
        st.session_state.last_update = time.time()
        st.experimental_rerun()

with col2:
    # Alerts section
    st.markdown("<h2 class='section-header'>Collision Alerts</h2>", unsafe_allow_html=True)
    collision_risks = check_collisions(debris_data)
    show_alerts(collision_risks)

# Sidebar
create_sidebar(debris_data)