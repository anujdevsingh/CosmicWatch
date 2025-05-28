
import numpy as np
from hmmlearn import hmm
from datetime import datetime, timedelta
import pandas as pd

class TrajectoryHMM:
    """Hidden Markov Model for space debris trajectory prediction with uncertainty"""
    
    def __init__(self, n_components=3):
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type="full")
        self.is_trained = False
        self.n_components = n_components
        
    def prepare_data(self, debris_history):
        """Prepare sequence data for HMM from debris history"""
        # Extract position and velocity data
        sequences = []
        for obj_id, history in debris_history.items():
            if len(history) >= 5:  # Need enough history points
                # Create feature vector: [x, y, z, velocity]
                sequence = np.array([[h['x'], h['y'], h['z'], h['velocity']] 
                                     for h in history])
                sequences.append(sequence)
        
        if not sequences:
            raise ValueError("Not enough historical data for training")
            
        # Concatenate all sequences for training
        self.X_train = np.vstack(sequences)
        self.lengths = [len(s) for s in sequences]
        
        return self.X_train, self.lengths
    
    def train(self, debris_history):
        """Train the HMM model on debris historical data"""
        try:
            X_train, lengths = self.prepare_data(debris_history)
            self.model.fit(X_train, lengths)
            self.is_trained = True
            print(f"HMM model trained with {self.n_components} hidden states")
            return True
        except Exception as e:
            print(f"Error training HMM model: {str(e)}")
            return False
    
    def predict_trajectory(self, object_history, time_steps=10):
        """Predict future trajectory with uncertainty bounds"""
        if not self.is_trained:
            print("Model not trained yet")
            return None
            
        if len(object_history) < 2:  # Reduced from 3 to 2
            print("Not enough history for prediction")
            return None
            
        try:
            # Create sequence from recent history
            recent_seq = np.array([[h['x'], h['y'], h['z'], h['velocity']] 
                                  for h in object_history[-5:]])
            
            # Get the most likely hidden state sequence
            hidden_states = self.model.predict(recent_seq)
            last_state = hidden_states[-1]
            
            # Generate future steps using transition matrix and emission probabilities
            predictions = []
            uncertainties = []
            
            # Start with the last known position
            current = recent_seq[-1]
            
            for _ in range(time_steps):
                # Transition to next hidden state based on transition probability
                next_state = np.random.choice(
                    self.n_components, 
                    p=self.model.transmat_[last_state]
                )
                
                # Sample from the emission distribution of the next state
                means = self.model.means_[next_state]
                covars = self.model.covars_[next_state]
                
                # Generate next position with uncertainty
                next_point = np.random.multivariate_normal(means, covars)
                
                # Calculate uncertainty as standard deviation
                uncertainty = np.sqrt(np.diag(covars))
                
                predictions.append(next_point)
                uncertainties.append(uncertainty)
                
                # Update for next iteration
                current = next_point
                last_state = next_state
                
            return {
                'predictions': np.array(predictions),
                'uncertainties': np.array(uncertainties)
            }
            
        except Exception as e:
            print(f"Error in trajectory prediction: {str(e)}")
            return None
    
    def get_collision_probability(self, obj1_history, obj2_history, future_steps=5):
        """Calculate collision probability using HMM predictions"""
        # Get trajectory predictions for both objects
        obj1_pred = self.predict_trajectory(obj1_history, future_steps)
        obj2_pred = self.predict_trajectory(obj2_history, future_steps)
        
        if obj1_pred is None or obj2_pred is None:
            return 0.0
        
        # Calculate probability of collision for each future time step
        collision_probs = []
        
        for t in range(future_steps):
            # Get predicted positions and uncertainties at time t
            pos1 = obj1_pred['predictions'][t][:3]  # x, y, z
            pos2 = obj2_pred['predictions'][t][:3]  # x, y, z
            
            # Combined uncertainty (add variances)
            unc1 = obj1_pred['uncertainties'][t][:3]
            unc2 = obj2_pred['uncertainties'][t][:3]
            combined_uncertainty = np.sqrt(unc1**2 + unc2**2)
            
            # Distance between predicted positions
            distance = np.linalg.norm(pos1 - pos2)
            
            # Combined object sizes (assumed to be 1.0 if not available)
            size1 = obj1_history[-1].get('size', 1.0)
            size2 = obj2_history[-1].get('size', 1.0)
            combined_size = size1 + size2
            
            # Probability of collision based on distance, size and uncertainty
            # Using a normal distribution approximation
            try:
                # Handle cases where combined_uncertainty might be a vector
                if hasattr(combined_uncertainty, "__len__") and len(combined_uncertainty) > 1:
                    combined_uncertainty = np.mean(combined_uncertainty)
                
                # Calculate probability and ensure it's a scalar
                prob_val = np.exp(-0.5 * ((distance - combined_size) / max(combined_uncertainty, 0.001))**2)
                # Convert to scalar before clipping
                if hasattr(prob_val, "__len__"):
                    prob_val = float(np.mean(prob_val))
                
                # Ensure we have a scalar value between 0 and 1
                prob = max(0.0, min(float(prob_val), 1.0))
            except Exception as e:
                print(f"Error calculating collision probability: {e}")
                prob = 0.0  # Default to zero probability on error
            
            collision_probs.append(prob)
        
        # Return maximum probability across time steps
        return float(max(collision_probs))
