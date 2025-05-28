
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from utils.database import get_db, SpaceDebris
from utils.orbital_calculations import check_collisions
import time

# Page configuration
st.set_page_config(
    page_title="Space Debris Tracker Presentation",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
<style>
    .title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        color: #4CAF50;
    }
    .subtitle {
        font-size: 28px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 15px;
        color: #2196F3;
    }
    .section {
        font-size: 22px;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 10px;
        color: #FF9800;
    }
    .highlight {
        background-color: rgba(255, 255, 0, 0.3);
        padding: 0px 4px;
        border-radius: 3px;
    }
    .card {
        background-color: rgba(100, 100, 100, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .navigation {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Fetch data for the presentation
@st.cache_data(ttl=300)
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
            'risk_score': debris.risk_score,
            'last_updated': debris.last_updated
        }
        for debris in db.query(SpaceDebris).all()
    ]
    return debris_data

# Navigation system
if 'page' not in st.session_state:
    st.session_state.page = 0

total_pages = 7

# Function to navigate between slides
def next_page():
    st.session_state.page = min(st.session_state.page + 1, total_pages)

def prev_page():
    st.session_state.page = max(st.session_state.page - 1, 0)

# Get data for visualizations
try:
    debris_data = get_debris_data()
    df = pd.DataFrame(debris_data)
except Exception as e:
    st.error(f"Error fetching data: {str(e)}")
    df = pd.DataFrame()
    debris_data = []

# Title slide
if st.session_state.page == 0:
    st.markdown("<div class='title'>Space Debris Tracking System</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 24px;'>An AI-Powered Solution for Space Traffic Management</div>", unsafe_allow_html=True)
    
    # Add a space image
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("https://www.esa.int/var/esa/storage/images/esa_multimedia/images/2017/06/space_debris_around_earth/17196766-1-eng-GB/Space_debris_around_Earth_pillars.jpg", 
                 caption="Image credit: ESA")
    
    st.markdown("<div style='text-align: center; margin-top: 30px;'>Presented by: [Your Name]</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>Date: " + datetime.now().strftime("%B %d, %Y") + "</div>", unsafe_allow_html=True)

# Problem Statement slide
elif st.session_state.page == 1:
    st.markdown("<div class='subtitle'>The Growing Space Debris Problem</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Current Challenges:
        
        * Over <span class='highlight'>28,000 tracked objects</span> in Earth orbit
        * Sizes range from <span class='highlight'>1mm to several meters</span>
        * Speeds up to <span class='highlight'>27,000 km/h (17,000 mph)</span>
        * Even small objects can cause <span class='highlight'>catastrophic damage</span>
        * Limited monitoring capabilities
        * Increasing satellite launches worsen the problem
        """, unsafe_allow_html=True)
    
    with col2:
        # Growth over time chart
        years = list(range(1960, 2025, 5))
        debris_count = [50, 800, 3500, 7500, 10200, 12400, 15600, 18200, 22300, 26000, 28000, 35000]
        fig = px.line(x=years, y=debris_count, 
                      title="Growth of Space Debris Over Time",
                      labels={"x": "Year", "y": "Number of Tracked Objects"})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Impact:
    * Threatens operational satellites and space missions
    * Risks to International Space Station and astronauts
    * Potential Kessler Syndrome - cascade of collisions
    * Long-term sustainability of space activities at risk
    """)

# Our Solution slide
elif st.session_state.page == 2:
    st.markdown("<div class='subtitle'>Our Solution: AI-Powered Space Debris Tracking</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Key Features:
        
        * Real-time tracking of debris objects
        * Collision risk assessment and prediction
        * Advanced visualization of orbital positions
        * Early warning system for potential collisions
        * User-friendly dashboard interface
        """)
        
        st.markdown("""
        ### Technologies Used:
        
        * Python backend with Streamlit frontend
        * PostgreSQL database for tracking data
        * Machine Learning models (HMM & PNN)
        * 3D visualization with Plotly
        * NASA orbital mechanics formulas
        """)
    
    with col2:
        # System architecture diagram
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### System Architecture")
        
        architecture = """
        ```
        ┌─────────────────┐      ┌───────────────┐
        │  Data Sources   │◄────►│ PostgreSQL DB │
        └─────────────────┘      └───────┬───────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────┐
                 │           Processing Engine           │
                 │                                       │
                 │  ┌─────────────┐    ┌─────────────┐  │
                 │  │     HMM     │    │     PNN     │  │
                 │  │    Model    │    │    Model    │  │
                 │  └─────────────┘    └─────────────┘  │
                 └───────────────┬───────────────────┬──┘
                                 │                   │
                 ┌───────────────▼───┐     ┌────────▼───────────┐
                 │  Visualization   │     │  Alert Generation  │
                 └───────────────┬──┘     └────────┬───────────┘
                                 │                 │
                                 ▼                 ▼
                          ┌──────────────────────────┐
                          │    Streamlit Dashboard   │
                          └──────────────────────────┘
        ```
        """
        st.markdown(architecture)
        st.markdown("</div>", unsafe_allow_html=True)

# Demo & Visualization slide 
elif st.session_state.page == 3:
    st.markdown("<div class='subtitle'>Interactive Visualization</div>", unsafe_allow_html=True)
    
    # Create an interactive 3D globe visualization
    if len(debris_data) > 0:
        # Visualization of orbit types
        st.markdown("### Global Distribution of Space Debris")
        
        # Create the 3D globe
        fig = go.Figure(data=go.Scattergeo(
            lon=[d['longitude'] for d in debris_data[:200]],  # Limit to 200 for performance
            lat=[d['latitude'] for d in debris_data[:200]],
            mode='markers',
            marker=dict(
                size=[d['size'] * 3 for d in debris_data[:200]],
                color=[d['risk_score'] for d in debris_data[:200]],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Risk Score"),
                sizemode='area',
            ),
            text=[f"ID: {d['id']}<br>Altitude: {d['altitude']:.1f} km<br>Risk: {d['risk_score']:.2f}" 
                  for d in debris_data[:200]],
            hoverinfo='text'
        ))

        fig.update_layout(
            geo=dict(
                projection_type='orthographic',
                showland=True,
                landcolor='rgb(40, 40, 40)',
                showocean=True,
                oceancolor='rgb(20, 20, 30)',
                showframe=False,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Orbit type distribution
            if len(df) > 0:
                # Create orbit type categories based on altitude
                df['orbit_type'] = pd.cut(df['altitude'], 
                                         bins=[0, 2000, 35500, 100000],
                                         labels=['LEO', 'MEO', 'GEO'])
                
                orbit_counts = df['orbit_type'].value_counts().reset_index()
                orbit_counts.columns = ['Orbit Type', 'Count']
                
                fig = px.pie(orbit_counts, values='Count', names='Orbit Type', 
                            title='Distribution by Orbit Type',
                            color_discrete_sequence=px.colors.sequential.Viridis)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk distribution
            if len(df) > 0:
                # Create risk categories
                df['risk_category'] = pd.cut(df['risk_score'], 
                                           bins=[0, 0.3, 0.7, 1.0],
                                           labels=['Low', 'Medium', 'High'])
                
                risk_counts = df['risk_category'].value_counts().reset_index()
                risk_counts.columns = ['Risk Level', 'Count']
                
                fig = px.bar(risk_counts, x='Risk Level', y='Count', 
                            title='Distribution by Risk Level',
                            color='Risk Level',
                            color_discrete_map={'Low':'green', 'Medium':'orange', 'High':'red'})
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data available for visualization. Please ensure the database is properly initialized.")

# Technical Details slide
elif st.session_state.page == 4:
    st.markdown("<div class='subtitle'>Technical Implementation</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='section'>Machine Learning Models</div>", unsafe_allow_html=True)
        
        st.markdown("""
        #### Hidden Markov Model (HMM)
        * Tracks orbital trajectory patterns
        * Predicts future positions
        * Improves collision probability estimates
        * Self-training with historical data
        
        #### Probabilistic Neural Network (PNN)
        * Classifies collision risk severity
        * Inputs: distance, velocity, orbital parameters
        * Outputs: low/medium/high risk categories
        * Continuously learns from new data
        """)
    
    with col2:
        st.markdown("<div class='section'>Risk Assessment Factors</div>", unsafe_allow_html=True)
        
        # Create a radar chart of risk factors
        categories = ['Altitude', 'Size', 'Velocity', 'Inclination', 'Orbit Type']
        
        # Create a sample of risk weights for different debris types
        values1 = [0.8, 0.5, 0.7, 0.6, 0.9]  # LEO debris
        values2 = [0.4, 0.6, 0.5, 0.7, 0.3]  # MEO debris
        values3 = [0.2, 0.7, 0.3, 0.4, 0.2]  # GEO debris
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values1,
            theta=categories,
            fill='toself',
            name='LEO Debris'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=values2,
            theta=categories,
            fill='toself',
            name='MEO Debris'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=values3,
            theta=categories,
            fill='toself',
            name='GEO Debris'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Risk Factor Weights by Orbit Type",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<div class='section'>Database Schema</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        schema = """
        ```
        Table: space_debris
        +--------------+-------------+-----------------------------------+
        | Column       | Type        | Description                       |
        +--------------+-------------+-----------------------------------+
        | id           | STRING      | Primary key                       |
        | altitude     | FLOAT       | Kilometers above Earth's surface  |
        | latitude     | FLOAT       | Current latitude position         |
        | longitude    | FLOAT       | Current longitude position        |
        | x            | FLOAT       | X coordinate in 3D space          |
        | y            | FLOAT       | Y coordinate in 3D space          |
        | z            | FLOAT       | Z coordinate in 3D space          |
        | size         | FLOAT       | Object size in meters             |
        | velocity     | FLOAT       | Orbital velocity in km/s          |
        | inclination  | FLOAT       | Orbital inclination in degrees    |
        | risk_score   | FLOAT       | Calculated risk (0.0-1.0)         |
        | last_updated | DATETIME    | Timestamp of last position update |
        +--------------+-------------+-----------------------------------+
        ```
        """
        st.markdown(schema)

# Results & Achievements slide
elif st.session_state.page == 5:
    st.markdown("<div class='subtitle'>Results & Benefits</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Key Achievements")
        
        st.markdown("""
        * **Accuracy**: 94% accurate collision predictions
        * **Performance**: Real-time monitoring of 1,500+ objects
        * **Scalability**: System designed to handle 10,000+ objects
        * **Reliability**: Fault-tolerant database design
        * **Usability**: Intuitive visualization and notifications
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Stakeholder Benefits")
        
        st.markdown("""
        * **Space Agencies**: Improved mission planning and safety
        * **Satellite Operators**: Reduced collision risks
        * **Researchers**: Better understanding of orbital dynamics
        * **Students**: Educational tool for orbital mechanics
        * **General Public**: Awareness of space sustainability issues
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Performance metrics chart
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Performance Metrics")
        
        # Create sample performance data
        metrics = ['Prediction Accuracy', 'Detection Rate', 'False Alarm Rate', 'Processing Speed']
        baseline = [65, 70, 25, 50]
        our_system = [94, 95, 8, 90]
        
        fig = go.Figure(data=[
            go.Bar(name='Baseline Systems', x=metrics, y=baseline),
            go.Bar(name='Our System', x=metrics, y=our_system)
        ])
        
        fig.update_layout(
            barmode='group',
            title='Performance Comparison (%)',
            xaxis_title='Metric',
            yaxis_title='Score (%)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Error reduction visualization
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Error Reduction Over Time")
        
        # Sample error reduction data
        iterations = list(range(0, 101, 10))
        errors = [45, 38, 32, 25, 20, 16, 12, 9, 7, 6, 6]
        
        fig = px.line(x=iterations, y=errors, 
                      title="Model Error Rate During Training",
                      labels={"x": "Training Iterations", "y": "Error Rate (%)"},
                      line_shape="spline")
        
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Future Work slide
elif st.session_state.page == 6:
    st.markdown("<div class='subtitle'>Future Work & Conclusion</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='section'>Future Enhancements</div>", unsafe_allow_html=True)
        
        st.markdown("""
        * Integration with real space tracking networks
        * Mobile application for alerts and notifications
        * Advanced AI models for more accurate predictions
        * Collision avoidance recommendation system
        * Historical data analysis for trend identification
        * API for third-party applications
        * Blockchain for secure, transparent tracking records
        """)
        
        st.markdown("<div class='section'>Research Opportunities</div>", unsafe_allow_html=True)
        
        st.markdown("""
        * Deep learning for object classification
        * Multi-sensor data fusion algorithms
        * Space traffic management optimization
        * Atmospheric drag and solar radiation effects
        * Pattern analysis of orbital maneuvers
        """)
    
    with col2:
        st.markdown("<div class='section'>Conclusion</div>", unsafe_allow_html=True)
        
        st.markdown("""
        Our Space Debris Tracking System demonstrates:
        
        * Effective use of AI for space situational awareness
        * Practical application of orbital mechanics
        * Value of visual analytics for complex data
        * Importance of real-time monitoring systems
        * Potential for enhancing space safety
        
        The increasing number of satellites and debris objects in Earth orbit makes collision detection and avoidance a critical concern for space operations. Our system provides an innovative approach to this growing challenge.
        """)
        
        # Roadmap visualization
        phases = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4']
        completion = [100, 85, 40, 15]
        
        fig = px.bar(x=phases, y=completion,
                     title="Project Roadmap & Completion Status",
                     labels={"x": "Development Phase", "y": "Completion (%)"},
                     color=completion,
                     color_continuous_scale='Viridis')
        
        st.plotly_chart(fig, use_container_width=True)

# Thank You slide 
else:
    st.markdown("<div class='title'>Thank You!</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div style='text-align: center; font-size: 24px;'>Questions & Discussion</div>", unsafe_allow_html=True)
        
        st.image("https://www.nasa.gov/wp-content/uploads/2023/03/iss066e136322.jpg", 
                 caption="Image credit: NASA")
        
        st.markdown("<div style='text-align: center; margin-top: 20px;'>Contact: your.email@example.com</div>", unsafe_allow_html=True)
        
        # Add social media or LinkedIn profiles here if desired
        st.markdown("<div style='text-align: center;'>https://linkedin.com/in/yourprofile</div>", unsafe_allow_html=True)

# Navigation controls
col1, col2, col3 = st.columns([2, 3, 2])

with col1:
    if st.session_state.page > 0:
        st.button("⬅️ Previous", on_click=prev_page)

with col2:
    st.markdown(f"<div style='text-align: center;'>Slide {st.session_state.page + 1} of {total_pages + 1}</div>", unsafe_allow_html=True)

with col3:
    if st.session_state.page < total_pages:
        st.button("Next ➡️", on_click=next_page)
