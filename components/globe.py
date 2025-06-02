import plotly.graph_objects as go
import numpy as np
import plotly.express as px

def create_enhanced_globe(debris_data):
    """Create an AI-enhanced interactive 3D globe visualization of space debris with risk assessment."""

    if not debris_data:
        # Return empty plot if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No debris data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="white")
        )
        return fig

    # Enhanced color mapping based on AI risk levels
    def get_risk_color(risk_level, confidence=1.0):
        """Get color based on AI risk assessment"""
        color_map = {
            'CRITICAL': f'rgba(255, 68, 68, {confidence})',     # Bright red
            'HIGH': f'rgba(255, 140, 0, {confidence})',         # Orange
            'MEDIUM': f'rgba(255, 215, 0, {confidence})',       # Gold
            'LOW': f'rgba(50, 205, 50, {confidence})',          # Lime green
            'UNKNOWN': f'rgba(128, 128, 128, {confidence})'     # Gray
        }
        return color_map.get(risk_level, color_map['UNKNOWN'])

    # Enhanced size calculation based on altitude and risk
    def calculate_enhanced_size(debris):
        """Calculate marker size based on multiple factors"""
        base_size = max(4, debris['size'] * 3)  # Minimum size of 4
        
        # Risk-based size enhancement
        risk_multiplier = {
            'CRITICAL': 3.0,
            'HIGH': 2.0,
            'MEDIUM': 1.5,
            'LOW': 1.0,
            'UNKNOWN': 1.2
        }
        
        risk_level = debris.get('risk_level', 'UNKNOWN')
        enhanced_size = base_size * risk_multiplier.get(risk_level, 1.0)
        
        # Altitude-based adjustment (closer objects appear larger)
        if debris['altitude'] < 300:  # Very low orbit
            enhanced_size *= 1.5
        elif debris['altitude'] < 600:  # Low orbit
            enhanced_size *= 1.2
        
        return min(enhanced_size, 25)  # Cap at 25 for readability

    # Prepare enhanced data
    lons = []
    lats = []
    sizes = []
    colors = []
    hover_texts = []
    symbols = []

    for debris in debris_data:
        lons.append(debris['longitude'])
        lats.append(debris['latitude'])
        
        # Calculate enhanced size and color
        size = calculate_enhanced_size(debris)
        sizes.append(size)
        
        risk_level = debris.get('risk_level', 'UNKNOWN')
        confidence = debris.get('confidence', 1.0)
        color = get_risk_color(risk_level, min(confidence, 1.0))
        colors.append(color)
        
        # Enhanced symbol based on risk level
        if risk_level == 'CRITICAL':
            symbols.append('triangle-up')
        elif risk_level == 'HIGH':
            symbols.append('diamond')
        elif risk_level == 'MEDIUM':
            symbols.append('circle')
        else:
            symbols.append('circle-open')
        
        # Enhanced hover text with AI insights
        ai_enhanced = debris.get('ai_enhanced', False)
        probabilities = debris.get('probabilities', {})
        
        hover_text = f"""
        <b>{debris['id']}</b><br>
        🌍 Altitude: {debris['altitude']:.1f} km<br>
        📐 Position: {debris['latitude']:.2f}°, {debris['longitude']:.2f}°<br>
        🚀 Velocity: {debris['velocity']:.2f} km/s<br>
        📏 Size: {debris['size']:.2f} m<br>
        {'🤖 AI Risk: ' + risk_level if ai_enhanced else '🔍 Risk Score: ' + str(debris.get('risk_score', 'N/A'))}<br>
        {'🎯 Confidence: ' + f"{confidence:.1%}" if ai_enhanced and confidence > 0 else ''}
        """
        
        if probabilities and ai_enhanced:
            hover_text += "<br><b>AI Probabilities:</b><br>"
            for level, prob in probabilities.items():
                if prob > 0.01:  # Only show significant probabilities
                    hover_text += f"  {level}: {prob:.1%}<br>"
        
        hover_texts.append(hover_text)

    # Create the enhanced globe visualization
    fig = go.Figure()

    # Add orbital paths for high-risk objects (optional enhancement)
    critical_debris = [d for d in debris_data if d.get('risk_level') == 'CRITICAL']
    for debris in critical_debris[:5]:  # Show paths for top 5 critical objects
        # Generate simple orbital path
        orbit_lats = []
        orbit_lons = []
        
        # Simple circular orbit approximation
        for angle in np.linspace(0, 360, 36):
            # Approximate orbital path (simplified)
            lat_offset = 10 * np.sin(np.radians(angle))
            lon_offset = angle - 180  # Full orbit
            
            orbit_lat = max(-90, min(90, debris['latitude'] + lat_offset))
            orbit_lon = (debris['longitude'] + lon_offset) % 360 - 180
            
            orbit_lats.append(orbit_lat)
            orbit_lons.append(orbit_lon)
        
        # Add orbital path trace
        fig.add_trace(go.Scattergeo(
            lon=orbit_lons,
            lat=orbit_lats,
            mode='lines',
            line=dict(
                color='rgba(255, 68, 68, 0.3)',
                width=1,
                dash='dot'
            ),
            name=f'Orbit: {debris["id"]}',
            showlegend=False,
            hoverinfo='skip'
        ))

    # Main debris scatter plot
    fig.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            symbol=symbols,
            line=dict(
                color='rgba(255, 255, 255, 0.8)',
                width=1
            ),
            sizemode='diameter',
            opacity=0.8
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        name='Space Debris'
    ))

    # Enhanced layout with better styling
    fig.update_layout(
        title={
            'text': '🌍 AI-Enhanced Global Space Debris Tracking',
            'x': 0.5,
            'font': {'size': 16, 'color': '#4ECDC4'}
        },
        showlegend=False,
        paper_bgcolor='rgba(10, 10, 20, 0.9)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        geo=dict(
            projection_type='orthographic',
            showland=True,
            landcolor='rgba(60, 60, 60, 0.8)',
            showocean=True,
            oceancolor='rgba(20, 30, 50, 0.9)',
            showlakes=True,
            lakecolor='rgba(30, 50, 80, 0.7)',
            showcountries=True,
            countrycolor='rgba(100, 100, 100, 0.3)',
            showframe=False,
            bgcolor='rgba(0, 0, 0, 0)',
            projection=dict(
                rotation=dict(lon=0, lat=0, roll=0)
            )
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        font=dict(color='white', family='Arial, sans-serif')
    )

    # Add enhanced legend information
    legend_text = """
    <div style='position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.7); 
                 padding: 10px; border-radius: 5px; color: white; font-size: 12px;'>
        <b>🤖 AI Risk Levels</b><br>
        🔴 CRITICAL: Immediate reentry risk<br>
        🟠 HIGH: High atmospheric drag<br>
        🟡 MEDIUM: Moderate risk factors<br>
        🟢 LOW: Stable orbit<br><br>
        
        <b>📊 Symbols</b><br>
        ▲ Critical objects<br>
        ♦ High risk objects<br>
        ● Medium/Low risk<br>
        ○ Unknown risk<br><br>
        
        <b>📈 Size:</b> Risk × Altitude factor<br>
        <b>🎯 Accuracy:</b> 90.6% AI model
    </div>
    """

    # Add annotations for statistics
    ai_enhanced_count = len([d for d in debris_data if d.get('ai_enhanced', False)])
    critical_count = len([d for d in debris_data if d.get('risk_level') == 'CRITICAL'])
    
    fig.add_annotation(
        text=f"🤖 AI Analysis: {ai_enhanced_count}/{len(debris_data)} objects<br>"
             f"🔴 Critical Objects: {critical_count}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=12, color="#4ECDC4"),
        bgcolor="rgba(0, 0, 0, 0.7)",
        bordercolor="#4ECDC4",
        borderwidth=1,
        align="left"
    )

    return fig

def create_globe(debris_data):
    """Backward compatibility wrapper"""
    return create_enhanced_globe(debris_data)