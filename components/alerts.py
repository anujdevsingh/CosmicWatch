import streamlit as st
from datetime import datetime

def show_alerts(collision_risks):
    """Display collision risk alerts."""

    if not collision_risks:
        st.info("No collision risks detected at this time.")
        return

    # Group by severity
    high_risks = [r for r in collision_risks if r['severity'] == 'high']
    medium_risks = [r for r in collision_risks if r['severity'] == 'medium']
    low_risks = [r for r in collision_risks if r['severity'] == 'low']

    # Show high risks first
    if high_risks:
        with st.container():
            st.markdown("<h3 class='alert-high'>⚠️ High Risk Collisions</h3>", unsafe_allow_html=True)
            for risk in high_risks:
                with st.expander(f"{risk['object1_id']} and {risk['object2_id']} - {risk['time_to_approach']:.1f} hours"):
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown(f"**Minimum Distance:** {risk['min_distance']:.2f} km")
                        st.markdown(f"**Probability:** {risk['probability']:.2%}")
                        st.markdown(f"**Time to Approach:** {risk['time_to_approach']:.1f} hours")
                    with cols[1]:
                        st.markdown(f"**Relative Velocity:** {risk['relative_velocity']:.2f} km/s")
                        st.markdown(f"**Combined Size:** {risk['combined_size']:.2f} m")
                        st.markdown(f"**Average Altitude:** {risk['altitude']:.2f} km")

                    # Add model prediction details
                    st.markdown("### Model Predictions")
                    pred_cols = st.columns(2)
                    with pred_cols[0]:
                        st.markdown(f"**HMM Prediction:** {risk.get('hmm_probability', 0):.2%}")
                    with pred_cols[1]:
                        if 'pnn_severity' in risk:
                            st.markdown(f"**PNN Prediction:** {risk['pnn_severity']}")
                            if 'pnn_probabilities' in risk:
                                # Show detailed probabilities
                                probs = risk['pnn_probabilities']
                                st.markdown(f"*Low: {probs[0]:.2%}, Medium: {probs[1]:.2%}, High: {probs[2]:.2%}*")

    # Medium risks
    if medium_risks:
        with st.container():
            st.markdown("<h3 class='alert-medium'>🔶 Medium Risk Collisions</h3>", unsafe_allow_html=True)
            for risk in medium_risks[:3]:  # Limit to top 3
                with st.expander(f"{risk['object1_id']} and {risk['object2_id']} - {risk['time_to_approach']:.1f} hours"):
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown(f"**Minimum Distance:** {risk['min_distance']:.2f} km")
                        st.markdown(f"**Probability:** {risk['probability']:.2%}")
                        st.markdown(f"**Time to Approach:** {risk['time_to_approach']:.1f} hours")
                    with cols[1]:
                        st.markdown(f"**Relative Velocity:** {risk['relative_velocity']:.2f} km/s")
                        st.markdown(f"**Combined Size:** {risk['combined_size']:.2f} m")
                        st.markdown(f"**Average Altitude:** {risk['altitude']:.2f} km")

                    # Add model prediction details
                    st.markdown("### Model Predictions")
                    pred_cols = st.columns(2)
                    with pred_cols[0]:
                        st.markdown(f"**HMM Prediction:** {risk.get('hmm_probability', 0):.2%}")
                    with pred_cols[1]:
                        if 'pnn_severity' in risk:
                            st.markdown(f"**PNN Prediction:** {risk['pnn_severity']}")
                            if 'pnn_probabilities' in risk:
                                # Show detailed probabilities
                                probs = risk['pnn_probabilities']
                                st.markdown(f"*Low: {probs[0]:.2%}, Medium: {probs[1]:.2%}, High: {probs[2]:.2%}*")

    # Low risks (collapsed)
    if low_risks:
        with st.expander(f"🔷 Low Risk Collisions ({len(low_risks)})"):
            for i, risk in enumerate(low_risks[:5]):  # Limit to top 5
                st.markdown(f"**{risk['object1_id']} and {risk['object2_id']}** - {risk['min_distance']:.2f} km, {risk['probability']:.2%} probability")

                # Add compact model predictions
                hmm_prob = risk.get('hmm_probability', 0)
                pnn_sev = risk.get('pnn_severity', 'Unknown')
                st.markdown(f"*HMM: {hmm_prob:.2%}, PNN: {pnn_sev}*")

                if i < len(low_risks) - 1:
                    st.divider()

            if len(low_risks) > 5:
                st.markdown(f"*...and {len(low_risks) - 5} more low-risk collisions*")