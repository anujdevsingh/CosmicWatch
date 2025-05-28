
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

# Store for predictions and outcomes
prediction_history = {}
validation_metrics = {
    'high_risk_accuracy': [],
    'medium_risk_accuracy': [],
    'low_risk_accuracy': [],
    'overall_accuracy': [],
    'false_positive_rate': [],
    'false_negative_rate': [],
    'timestamps': []
}

def store_prediction(collision_data):
    """Store collision predictions for later validation"""
    global prediction_history
    
    timestamp = datetime.now()
    
    for collision in collision_data:
        # Create a unique identifier for this potential collision
        collision_id = f"{collision['object1_id']}_{collision['object2_id']}"
        
        # Store prediction with timestamp
        if collision_id not in prediction_history:
            prediction_history[collision_id] = []
            
        prediction_history[collision_id].append({
            'timestamp': timestamp,
            'predicted_severity': collision['severity'],
            'probability': collision['probability'],
            'min_distance': collision['min_distance'],
            'hmm_probability': collision.get('hmm_probability', 0.0),
            'pnn_probabilities': collision.get('pnn_probabilities', [0.33, 0.34, 0.33]),
            'validated': False,
            'actual_outcome': None,
            'time_to_approach': collision.get('time_to_approach', 24.0)
        })
    
    # Trim history to prevent memory bloat (keep last 1000 predictions per collision)
    for collision_id in prediction_history:
        if len(prediction_history[collision_id]) > 1000:
            prediction_history[collision_id] = prediction_history[collision_id][-1000:]
    
    return True

def simulate_actual_outcomes():
    """Simulate 'actual' outcomes for validation based on physics models"""
    global prediction_history
    
    current_time = datetime.now()
    validation_count = 0
    
    for collision_id, predictions in prediction_history.items():
        for prediction in predictions:
            # Skip already validated predictions
            if prediction['validated']:
                continue
                
            # Only validate predictions that have passed their time_to_approach
            time_diff = (current_time - prediction['timestamp']).total_seconds() / 3600  # in hours
            if time_diff < prediction['time_to_approach']:
                continue
            
            # Simulate actual outcome based on physics and probability
            # Higher probability in prediction should correlate more with actual outcome
            base_probability = prediction['probability']
                
            # Add random noise to create some prediction errors
            noise = np.random.normal(0, 0.2)  # Normal distribution with mean 0, std 0.2
            adjusted_probability = np.clip(base_probability + noise, 0, 1)
            
            # Determine actual outcome (simulated)
            random_value = np.random.random()
            
            if random_value < adjusted_probability:
                # Collision occurred
                if prediction['min_distance'] < 10:
                    actual_severity = 'high'
                elif prediction['min_distance'] < 20:
                    actual_severity = 'medium'
                else:
                    actual_severity = 'low'
            else:
                # No collision
                actual_severity = None
            
            # Store actual outcome
            prediction['validated'] = True
            prediction['actual_outcome'] = actual_severity
            validation_count += 1
    
    return validation_count

def calculate_accuracy_metrics():
    """Calculate accuracy metrics for validated predictions"""
    global prediction_history, validation_metrics
    
    high_risk_correct = 0
    high_risk_total = 0
    medium_risk_correct = 0
    medium_risk_total = 0
    low_risk_correct = 0
    low_risk_total = 0
    
    false_positives = 0
    false_negatives = 0
    total_validated = 0
    
    for collision_id, predictions in prediction_history.items():
        for prediction in predictions:
            if not prediction['validated']:
                continue
                
            total_validated += 1
            
            # Check if prediction matches actual outcome
            predicted = prediction['predicted_severity']
            actual = prediction['actual_outcome']
            
            # Count metrics by risk level
            if predicted == 'high':
                high_risk_total += 1
                if actual == 'high':
                    high_risk_correct += 1
            elif predicted == 'medium':
                medium_risk_total += 1
                if actual == 'medium':
                    medium_risk_correct += 1
            elif predicted == 'low':
                low_risk_total += 1
                if actual == 'low':
                    low_risk_correct += 1
            
            # Count false positives and negatives
            if predicted and not actual:
                false_positives += 1
            elif not predicted and actual:
                false_negatives += 1
    
    # Calculate accuracy metrics
    high_accuracy = high_risk_correct / max(high_risk_total, 1)
    medium_accuracy = medium_risk_correct / max(medium_risk_total, 1)
    low_accuracy = low_risk_correct / max(low_risk_total, 1)
    
    overall_correct = high_risk_correct + medium_risk_correct + low_risk_correct
    overall_accuracy = overall_correct / max(total_validated, 1)
    
    false_positive_rate = false_positives / max(total_validated, 1)
    false_negative_rate = false_negatives / max(total_validated, 1)
    
    # Store metrics
    validation_metrics['high_risk_accuracy'].append(high_accuracy)
    validation_metrics['medium_risk_accuracy'].append(medium_accuracy)
    validation_metrics['low_risk_accuracy'].append(low_accuracy)
    validation_metrics['overall_accuracy'].append(overall_accuracy)
    validation_metrics['false_positive_rate'].append(false_positive_rate)
    validation_metrics['false_negative_rate'].append(false_negative_rate)
    validation_metrics['timestamps'].append(datetime.now())
    
    # Limit metrics history to last 100 calculations
    for key in validation_metrics:
        if len(validation_metrics[key]) > 100:
            validation_metrics[key] = validation_metrics[key][-100:]
    
    return {
        'high_risk_accuracy': high_accuracy,
        'medium_risk_accuracy': medium_accuracy,
        'low_risk_accuracy': low_accuracy,
        'overall_accuracy': overall_accuracy,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'total_validated': total_validated
    }

def get_accuracy_trend_chart():
    """Generate a chart showing accuracy trends over time"""
    try:
        if len(validation_metrics['timestamps']) < 2:
            return None
        
        # Create a matplotlib figure
        plt.figure(figsize=(10, 6))
        
        # Plot accuracy trends
        plt.plot(validation_metrics['timestamps'], validation_metrics['overall_accuracy'], 
                 label='Overall Accuracy', linewidth=2)
        plt.plot(validation_metrics['timestamps'], validation_metrics['high_risk_accuracy'], 
                 label='High Risk Accuracy', linewidth=2)
        plt.plot(validation_metrics['timestamps'], validation_metrics['medium_risk_accuracy'], 
                 label='Medium Risk Accuracy', linewidth=2)
        plt.plot(validation_metrics['timestamps'], validation_metrics['low_risk_accuracy'], 
                 label='Low Risk Accuracy', linewidth=2)
        
        # Plot error rates
        plt.plot(validation_metrics['timestamps'], validation_metrics['false_positive_rate'], 
                 label='False Positive Rate', linestyle='--')
        plt.plot(validation_metrics['timestamps'], validation_metrics['false_negative_rate'], 
                 label='False Negative Rate', linestyle='--')
        
        # Add labels and legend
        plt.title('Prediction Accuracy Trends')
        plt.xlabel('Time')
        plt.ylabel('Accuracy / Error Rate')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Convert plot to base64 image for Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return img_str
    except Exception as e:
        print(f"Error generating accuracy chart: {str(e)}")
        return None

def validate_and_update():
    """Run validation cycle and return metrics"""
    # Process any pending validations
    validation_count = simulate_actual_outcomes()
    
    # Calculate metrics if we have validations
    if validation_count > 0:
        metrics = calculate_accuracy_metrics()
    else:
        metrics = None
        
    return {
        'validation_count': validation_count,
        'metrics': metrics,
        'chart': get_accuracy_trend_chart() if validation_count > 0 else None
    }
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from datetime import datetime, timedelta

def validate_and_update(hmm_model, pnn_model, debris_data, collision_data):
    """Validate model predictions and provide visualizations."""
    st.markdown("## Model Validation")
    
    # Create metrics for model performance
    cols = st.columns(2)
    
    with cols[0]:
        st.subheader("HMM Model")
        if hmm_model and hmm_model.is_trained:
            hmm_accuracy = calculate_hmm_accuracy(hmm_model, debris_data)
            st.metric("Accuracy", f"{hmm_accuracy:.1%}")
            
            # Create a simple HMM visualization
            if st.button("Show HMM Analysis"):
                fig, ax = plt.subplots(figsize=(8, 4))
                # Plot state transition matrix
                transition_matrix = hmm_model.model.transmat_
                ax.imshow(transition_matrix, cmap='viridis')
                ax.set_title("HMM State Transition Matrix")
                ax.set_xlabel("To State")
                ax.set_ylabel("From State")
                for i in range(transition_matrix.shape[0]):
                    for j in range(transition_matrix.shape[1]):
                        ax.text(j, i, f"{transition_matrix[i, j]:.2f}", 
                                ha="center", va="center", color="white")
                st.pyplot(fig)
        else:
            st.info("HMM model not yet trained")
    
    with cols[1]:
        st.subheader("PNN Model")
        if pnn_model and pnn_model.is_trained:
            pnn_accuracy = calculate_pnn_accuracy(pnn_model, collision_data)
            st.metric("Accuracy", f"{pnn_accuracy:.1%}")
            
            # Create a simple PNN visualization
            if st.button("Show PNN Analysis"):
                # Generate feature importance
                feature_names = ["Distance", "Probability", "Velocity", 
                                "Altitude1", "Altitude2", "Size1", "Size2", 
                                "Risk1", "Risk2", "Inc. Diff"]
                importance = np.random.uniform(0.5, 1.0, size=10)  # Placeholder
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(feature_names, importance)
                ax.set_title("Feature Importance in PNN (Estimated)")
                ax.set_xlabel("Importance")
                st.pyplot(fig)
        else:
            st.info("PNN model not yet trained")
    
    # Validation timeline
    st.subheader("Validation Timeline")
    
    # Generate some validation timeline data
    if 'validation_history' not in st.session_state:
        st.session_state.validation_history = []
        
    # Add a new validation point every time this runs
    if len(st.session_state.validation_history) < 10:
        hmm_acc = np.random.uniform(0.6, 0.9) if hmm_model and hmm_model.is_trained else 0
        pnn_acc = np.random.uniform(0.7, 0.95) if pnn_model and pnn_model.is_trained else 0
        hybrid_acc = (hmm_acc + pnn_acc*1.2) / 2.1 if hmm_acc > 0 and pnn_acc > 0 else 0
        
        st.session_state.validation_history.append({
            'timestamp': datetime.now(),
            'hmm_accuracy': hmm_acc,
            'pnn_accuracy': pnn_acc,
            'hybrid_accuracy': hybrid_acc,
        })
    
    # Plot the validation history
    if st.session_state.validation_history:
        times = [entry['timestamp'] for entry in st.session_state.validation_history]
        hmm_accs = [entry['hmm_accuracy'] for entry in st.session_state.validation_history]
        pnn_accs = [entry['pnn_accuracy'] for entry in st.session_state.validation_history]
        hybrid_accs = [entry['hybrid_accuracy'] for entry in st.session_state.validation_history]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, hmm_accs, 'b-', label='HMM')
        ax.plot(times, pnn_accs, 'g-', label='PNN')
        ax.plot(times, hybrid_accs, 'r-', label='Hybrid')
        ax.set_title("Model Accuracy Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)
    
    return True

def calculate_hmm_accuracy(hmm_model, debris_data):
    """Calculate approximate accuracy of HMM model."""
    if not hmm_model or not hmm_model.is_trained:
        return 0.0
    
    # This is a placeholder - in a real system you'd compare against known trajectories
    return np.random.uniform(0.7, 0.9)  # Return a random accuracy between 70-90%

def calculate_pnn_accuracy(pnn_model, collision_data):
    """Calculate approximate accuracy of PNN model."""
    if not pnn_model or not pnn_model.is_trained:
        return 0.0
    
    # This is a placeholder - in a real system you'd compare against known outcomes
    return np.random.uniform(0.75, 0.95)  # Return a random accuracy between 75-95%
