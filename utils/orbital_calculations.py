import numpy as np
from datetime import datetime, timedelta
from utils.hmm_model import TrajectoryHMM
from utils.pnn_model import RiskClassifier

# Global model instances
hmm_model = None
pnn_model = None

# Debris history dictionary to store historical positions
debris_history = {}

def initialize_models():
    """Initialize HMM and PNN models if not already done"""
    global hmm_model, pnn_model
    
    if hmm_model is None:
        hmm_model = TrajectoryHMM(n_components=3)
        
    if pnn_model is None:
        pnn_model = RiskClassifier(input_dim=10)
        
    # Pre-train models with more realistic data if needed
    if not hmm_model.is_trained:
        # Create enhanced synthetic training data for HMM
        minimal_history = {}
        # Generate more objects for better training
        for i in range(20):  # Increased from 10 to 20
            obj_id = f"SEED-{i}"
            # Generate more sequential position records for each object
            minimal_history[obj_id] = []
            
            # Create realistic orbital parameters
            altitude = np.random.uniform(300, 1000)  # km
            inclination = np.random.uniform(0, 90)  # degrees
            r = altitude + 6371  # Earth radius + altitude
            velocity = np.sqrt(398600.4418 / r)  # km/s (realistic orbital velocity)
            
            # Generate trajectory points with realistic orbital motion
            for j in range(10):  # Increased from 5 to 10 points
                # Calculate position for time step j
                angle = j * 0.1  # Angular position increases with each step
                
                # Calculate cartesian coordinates for a satellite in orbit
                x = float(r * np.cos(angle))
                y = float(r * np.sin(angle))
                z = float(r * np.sin(inclination * np.pi/180) * np.sin(angle))
                
                minimal_history[obj_id].append({
                    'x': x,
                    'y': y,
                    'z': z,
                    'velocity': velocity + np.random.normal(0, 0.05),
                    'size': np.random.uniform(0.5, 3.0),
                    'timestamp': datetime.now() + timedelta(minutes=j*5)
                })
        
        # Try to pre-train with improved data
        try:
            hmm_model.train(minimal_history)
            print("HMM pre-trained with enhanced seed data")
        except Exception as e:
            print(f"Error pre-training HMM: {e}")
            
    # Pre-train PNN with minimal synthetic data if needed
    if not pnn_model.is_trained:
        # Create minimal feature data
        minimal_features = []
        minimal_labels = []
        
        # Generate 15 synthetic collision scenarios
        for i in range(15):
            # High risk (close distance, high probability)
            if i < 5:
                feature = [
                    np.random.uniform(1, 10),         # min_distance
                    np.random.uniform(0.6, 0.9),      # probability
                    np.random.uniform(5, 10),         # relative_velocity
                    np.random.uniform(300, 500),      # alt1
                    np.random.uniform(300, 500),      # alt2
                    np.random.uniform(0.5, 2),        # size1
                    np.random.uniform(0.5, 2),        # size2
                    np.random.uniform(0.7, 0.9),      # risk1
                    np.random.uniform(0.7, 0.9),      # risk2
                    np.random.uniform(0, 5)           # inclination diff
                ]
                label = 2  # high
            # Medium risk
            elif i < 10:
                feature = [
                    np.random.uniform(10, 30),        # min_distance
                    np.random.uniform(0.3, 0.6),      # probability
                    np.random.uniform(3, 7),          # relative_velocity
                    np.random.uniform(500, 800),      # alt1
                    np.random.uniform(500, 800),      # alt2
                    np.random.uniform(0.5, 2),        # size1
                    np.random.uniform(0.5, 2),        # size2
                    np.random.uniform(0.4, 0.7),      # risk1
                    np.random.uniform(0.4, 0.7),      # risk2
                    np.random.uniform(5, 15)          # inclination diff
                ]
                label = 1  # medium
            # Low risk
            else:
                feature = [
                    np.random.uniform(30, 80),        # min_distance
                    np.random.uniform(0.1, 0.3),      # probability
                    np.random.uniform(1, 4),          # relative_velocity
                    np.random.uniform(800, 1500),     # alt1
                    np.random.uniform(800, 1500),     # alt2
                    np.random.uniform(0.5, 2),        # size1
                    np.random.uniform(0.5, 2),        # size2
                    np.random.uniform(0.1, 0.4),      # risk1
                    np.random.uniform(0.1, 0.4),      # risk2
                    np.random.uniform(15, 30)         # inclination diff
                ]
                label = 0  # low
                
            minimal_features.append(feature)
            minimal_labels.append(label)
            
        # Generate synthetic debris and collision data
        minimal_debris = []
        minimal_collisions = []
        
        for i in range(15):
            # Create two debris objects
            obj1 = {
                'id': f"SEED-A{i}",
                'altitude': minimal_features[i][3],
                'size': minimal_features[i][5],
                'risk_score': minimal_features[i][7],
                'inclination': np.random.uniform(0, 90)
            }
            
            obj2 = {
                'id': f"SEED-B{i}",
                'altitude': minimal_features[i][4],
                'size': minimal_features[i][6],
                'risk_score': minimal_features[i][8],
                'inclination': obj1['inclination'] + minimal_features[i][9]
            }
            
            minimal_debris.extend([obj1, obj2])
            
            # Create corresponding collision data
            severity_map = {0: 'low', 1: 'medium', 2: 'high'}
            minimal_collisions.append({
                'object1_id': obj1['id'],
                'object2_id': obj2['id'],
                'min_distance': minimal_features[i][0],
                'probability': minimal_features[i][1],
                'relative_velocity': minimal_features[i][2],
                'severity': severity_map[minimal_labels[i]]
            })
            
        # Try to pre-train PNN with minimal data
        try:
            pnn_model.train(minimal_debris, minimal_collisions, epochs=50, batch_size=5)
            print("PNN pre-trained with seed data")
        except Exception as e:
            print(f"Error pre-training PNN: {e}")
        
def update_debris_history(debris_data):
    """Update historical position data for all debris objects"""
    global debris_history
    
    current_time = datetime.now()
    
    for obj in debris_data:
        obj_id = obj['id']
        
        # Create entry if it doesn't exist
        if obj_id not in debris_history:
            debris_history[obj_id] = []
            
        # Add current position to history
        # Keep only the last 10 positions to avoid memory bloat
        debris_history[obj_id].append({
            'x': obj['x'],
            'y': obj['y'],
            'z': obj['z'],
            'velocity': obj['velocity'],
            'size': obj['size'],
            'timestamp': current_time
        })
        
        # Limit history size
        if len(debris_history[obj_id]) > 10:
            debris_history[obj_id] = debris_history[obj_id][-10:]

def check_collisions(debris_data):
    """Calculate potential collisions between debris objects using hybrid approach."""
    global hmm_model, pnn_model, debris_history
    
    # Initialize models if needed
    initialize_models()
    
    # Update history with current positions
    update_debris_history(debris_data)
    
    # Try to train HMM if we have enough history
    if len(debris_history) > 3:
        for obj_id, history in debris_history.items():
            if len(history) >= 5:
                hmm_model.train(debris_history)
                break
    
    collision_risks = []

    for i, obj1 in enumerate(debris_data[:-1]):
        for obj2 in debris_data[i+1:]:
            # Step 1: Calculate deterministic minimum distance (baseline)
            min_distance = calculate_minimum_distance(obj1, obj2)
            
            # Only process objects within reasonable range
            if min_distance < 100:  # Increased threshold to catch more potential risks
                # Step 2: Calculate base probability using deterministic method
                base_probability = calculate_collision_probability(obj1, obj2, min_distance)
                
                # Step 3: Use HMM for enhanced probability if we have history
                hmm_probability = 0.0
                if hmm_model.is_trained and obj1['id'] in debris_history and obj2['id'] in debris_history:
                    # Reduced minimum history requirement from 3 to 2
                    if len(debris_history[obj1['id']]) >= 2 and len(debris_history[obj2['id']]) >= 2:
                        try:
                            hmm_probability = hmm_model.get_collision_probability(
                                debris_history[obj1['id']], 
                                debris_history[obj2['id']]
                            )
                        except Exception as e:
                            print(f"Error getting HMM probability: {e}")
                            # Use a fallback method if the HMM fails
                            try:
                                # Simplified probability based on past trajectory
                                last_pos1 = np.array([debris_history[obj1['id']][-1]['x'], 
                                                     debris_history[obj1['id']][-1]['y'], 
                                                     debris_history[obj1['id']][-1]['z']])
                                last_pos2 = np.array([debris_history[obj2['id']][-1]['x'], 
                                                     debris_history[obj2['id']][-1]['y'], 
                                                     debris_history[obj2['id']][-1]['z']])
                                
                                # Use a simplified distance-based probability
                                last_distance = np.linalg.norm(last_pos1 - last_pos2)
                                hmm_probability = max(0.0, min(1.0, 100 / (last_distance + 10)))
                            except:
                                hmm_probability = 0.1  # Fallback value
                
                # Combine probabilities (weighted average)
                # Start with more weight on deterministic, increase HMM weight as model improves
                hmm_weight = 0.3 if hmm_model.is_trained else 0.0
                probability = (1 - hmm_weight) * base_probability + hmm_weight * hmm_probability
                
                # Step 4: Traditional severity determination (will be replaced by PNN later)
                severity = determine_severity(min_distance, probability)

                if severity:  # Only add if there's a meaningful risk
                    time_to_approach = calculate_time_to_approach(obj1, obj2)

                    # Calculate 3D relative velocity vector
                    pos1 = np.array([obj1['x'], obj1['y'], obj1['z']])
                    pos2 = np.array([obj2['x'], obj2['y'], obj2['z']])
                    
                    # We don't have velocity vectors, so create a more realistic calculation
                    # by combining scalar velocity with positional differences
                    vel_magnitude = abs(obj1['velocity'] - obj2['velocity'])
                    
                    # Add random component to ensure non-zero relative velocity (based on object positions)
                    rel_vel = vel_magnitude
                    if rel_vel < 0.01:  # If velocity difference is near zero
                        # Create velocity based on position differences
                        pos_diff = np.linalg.norm(pos1 - pos2)
                        rel_vel = max(0.5, pos_diff / 1000)  # At least 0.5 km/s
                    
                    # Ensure hmm_probability is always set - for display purposes
                    if not hmm_model.is_trained or hmm_probability == 0.0:
                        hmm_probability = 0.0
                    
                    # Create collision risk record
                    collision_risk = {
                        'object1_id': obj1['id'],
                        'object2_id': obj2['id'],
                        'min_distance': min_distance,
                        'probability': probability,
                        'hmm_probability': hmm_probability,
                        'severity': severity,
                        'time_to_approach': time_to_approach,
                        'relative_velocity': rel_vel,
                        'combined_size': obj1['size'] + obj2['size'],
                        'altitude': (obj1['altitude'] + obj2['altitude']) / 2
                    }
                    
                    # Prepare feature for PNN model
                    if pnn_model.is_trained:
                        feature = np.array([
                            min_distance,
                            probability,
                            abs(obj1['velocity'] - obj2['velocity']),
                            obj1['altitude'],
                            obj2['altitude'],
                            obj1['size'],
                            obj2['size'],
                            obj1['risk_score'],
                            obj2['risk_score'],
                            abs(obj1['inclination'] - obj2.get('inclination', 0))
                        ])
                        
                        # Get PNN prediction
                        pnn_result = pnn_model.predict_risk(feature)
                        collision_risk['pnn_severity'] = pnn_result['class']
                        collision_risk['pnn_probabilities'] = pnn_result['probabilities']
                        
                        # Override severity with PNN if trained
                        collision_risk['severity'] = pnn_result['class']
                    
                    collision_risks.append(collision_risk)

    # Train PNN model on current collision data if we have enough
    if len(collision_risks) >= 10 and not pnn_model.is_trained:
        pnn_model.train(debris_data, collision_risks)

    # Sort by probability and then by minimum distance
    return sorted(collision_risks, 
                 key=lambda x: (x['probability'], -x['min_distance']), 
                 reverse=True)

def calculate_minimum_distance(obj1, obj2):
    """Calculate the minimum distance between two objects."""
    pos1 = np.array([obj1['x'], obj1['y'], obj1['z']])
    pos2 = np.array([obj2['x'], obj2['y'], obj2['z']])
    return float(np.linalg.norm(pos1 - pos2))

def calculate_collision_probability(obj1, obj2, min_distance):
    """Calculate the probability of collision based on multiple factors."""
    # Consider multiple factors for collision probability
    combined_size = obj1['size'] + obj2['size']
    relative_velocity = abs(obj1['velocity'] - obj2['velocity'])

    # Base probability increases with size and decreases with distance
    base_probability = combined_size / (min_distance + combined_size)

    # Velocity factor - higher relative velocities increase risk
    velocity_factor = min(relative_velocity / 10.0, 1.0)

    # Altitude factor - lower altitudes have more atmospheric effects
    altitude = (obj1['altitude'] + obj2['altitude']) / 2
    altitude_factor = np.exp(-altitude/1000)  # Exponential decay with altitude

    # Combine factors
    probability = base_probability * (0.4 + 0.3 * velocity_factor + 0.3 * altitude_factor)

    return float(min(max(probability, 0.0), 1.0))

def determine_severity(distance, probability):
    """Determine the severity level of a potential collision."""
    if probability > 0.6 and distance < 10:  # High risk if very close and high probability
        return 'high'
    elif probability > 0.3 or distance < 20:  # Medium risk if moderately close or moderate probability
        return 'medium'
    elif probability > 0.1 or distance < 40:  # Low risk if somewhat close or low probability
        return 'low'
    return None

def calculate_time_to_approach(obj1, obj2):
    """Calculate approximate time until closest approach."""
    # Calculate based on relative positions and velocities
    pos1 = np.array([obj1['x'], obj1['y'], obj1['z']])
    pos2 = np.array([obj2['x'], obj2['y'], obj2['z']])

    # Simplified time calculation based on current positions
    distance = np.linalg.norm(pos1 - pos2)
    relative_velocity = abs(obj1['velocity'] - obj2['velocity'])

    if relative_velocity > 0:
        time = distance / (relative_velocity * 3600)  # Convert to hours
        return float(min(max(time, 1.0), 48.0))  # Limit between 1 and 48 hours
    return 24.0  # Default to 24 hours if can't calculate