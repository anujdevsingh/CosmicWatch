import plotly.graph_objects as go
import numpy as np

def create_globe(debris_data):
    """Create an interactive 3D globe visualization of space debris."""

    # Create the base globe
    fig = go.Figure(data=go.Scattergeo(
        lon=[d['longitude'] for d in debris_data],
        lat=[d['latitude'] for d in debris_data],
        mode='markers',
        marker=dict(
            size=[d['size'] * 5 for d in debris_data],
            color=[d['risk_score'] for d in debris_data],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=dict(
                    text='Risk Score',
                    side='right'
                ),
                x=1.1
            ),
            line=dict(color='rgb(40, 40, 40)', width=0.5),
            sizemode='area',
        ),
        text=[f"ID: {d['id']}<br>Altitude: {d['altitude']:.1f} km<br>Risk: {d['risk_score']:.2f}" 
              for d in debris_data],
        hoverinfo='text'
    ))

    # Update layout
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        geo=dict(
            projection_type='orthographic',
            showland=True,
            landcolor='rgb(40, 40, 40)',
            showocean=True,
            oceancolor='rgb(20, 20, 30)',
            showframe=False,
            bgcolor='rgba(0,0,0,0)',
        ),
        margin=dict(l=0, r=50, t=0, b=0),
        height=600
    )

    return fig