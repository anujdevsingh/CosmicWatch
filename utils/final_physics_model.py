#!/usr/bin/env python3
"""
Final Optimized Physics-Informed Space Debris Risk Model
Balanced risk classification with proper edge case handling
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FinalPhysicsFeatures:
    """Final optimized physics-based features"""
    
    @staticmethod
    def calculate_physics_features(orbital_elements: np.ndarray) -> np.ndarray:
        """
        Calculate optimized physics features from orbital elements
        """
        features = []
        
        # Extract basic orbital elements
        a = orbital_elements[0]  # semi-major axis (km)
        e = orbital_elements[1]  # eccentricity
        i = orbital_elements[2]  # inclination (rad)
        perigee_alt = orbital_elements[8]  # perigee altitude (km)
        apogee_alt = orbital_elements[9]   # apogee altitude (km)
        bstar = orbital_elements[10]       # drag coefficient
        period = orbital_elements[11]      # orbital period (min)
        
        # 1. Perigee altitude (most critical factor)
        features.append(perigee_alt)
        
        # 2. Log-scaled atmospheric drag factor
        # More sensitive to very low altitudes
        if perigee_alt < 300:
            drag_factor = 1.0  # Maximum drag
        elif perigee_alt < 600:
            drag_factor = np.exp(-(perigee_alt - 200) / 100)
        else:
            drag_factor = 0.0  # Minimal drag
        features.append(drag_factor)
        
        # 3. Eccentricity (orbital stability)
        features.append(e)
        
        # 4. Altitude difference (orbital shape indicator)
        alt_diff = apogee_alt - perigee_alt
        features.append(alt_diff)
        
        # 5. Ballistic coefficient (drag sensitivity)
        bstar_factor = min(1.0, abs(bstar) * 1e4)
        features.append(bstar_factor)
        
        # 6. Orbital period (related to altitude)
        period_normalized = min(2.0, period / 100)  # Normalize to ~2 hours
        features.append(period_normalized)
        
        # 7. Inclination risk (polar orbits more congested)
        inclination_deg = np.degrees(i)
        if 80 <= inclination_deg <= 100:
            inclination_risk = 1.0  # Polar orbit
        elif 45 <= inclination_deg <= 135:
            inclination_risk = 0.7  # Inclined orbit
        else:
            inclination_risk = 0.3  # Equatorial orbit
        features.append(inclination_risk)
        
        # 8. Combined risk indicator
        # Combines multiple factors for better discrimination
        combined_risk = 0
        if perigee_alt < 250:
            combined_risk += 3
        elif perigee_alt < 400:
            combined_risk += 2
        elif perigee_alt < 600:
            combined_risk += 1
        
        if e > 0.1:
            combined_risk += 1
        if abs(bstar) > 1e-4:
            combined_risk += 1
        
        features.append(combined_risk / 5.0)  # Normalize to 0-1
        
        return np.array(features, dtype=np.float32)

class FinalPhysicsNet(nn.Module):
    """Final optimized neural network"""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 128, num_classes: int = 4):
        super(FinalPhysicsNet, self).__init__()
        
        # Optimized architecture
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class FinalPhysicsInformedModel:
    """Final Optimized Physics-Informed Space Debris Risk Assessment Model"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_extractor = FinalPhysicsFeatures()
        self.training_history = {}
        
        print(f"🚀 Final Physics Model initialized on {self.device}")
    
    def final_risk_classification(self, orbital_elements: np.ndarray) -> str:
        """
        Final optimized physics-based risk classification
        More balanced and realistic classification
        """
        perigee_alt = orbital_elements[8]
        apogee_alt = orbital_elements[9]
        eccentricity = orbital_elements[1]
        bstar = orbital_elements[10]
        
        # More balanced risk thresholds
        if perigee_alt < 200:
            return 'CRITICAL'  # Immediate reentry
        elif perigee_alt < 350:
            return 'HIGH'      # High atmospheric drag
        elif perigee_alt < 700:
            # Check for additional risk factors
            risk_factors = 0
            if eccentricity > 0.1:
                risk_factors += 1
            if abs(bstar) > 1e-4:
                risk_factors += 1
            if apogee_alt > perigee_alt * 2:
                risk_factors += 1
            
            if risk_factors >= 2:
                return 'HIGH'
            elif risk_factors >= 1:
                return 'MEDIUM'
            else:
                return 'MEDIUM'  # Default for this range
        elif perigee_alt < 1200:
            return 'LOW' if eccentricity < 0.05 and abs(bstar) < 1e-5 else 'MEDIUM'
        else:
            return 'LOW'       # High altitude, stable
    
    def parse_tle_data(self, tle_file: str) -> pd.DataFrame:
        """Parse TLE data with final risk classification"""
        print(f"📡 Parsing TLE data from {tle_file}")
        
        satellites = []
        
        try:
            with open(tle_file, 'r') as f:
                lines = f.readlines()
            
            # Process TLE data in groups of 3 lines
            for i in range(0, len(lines) - 2, 3):
                try:
                    name_line = lines[i].strip()
                    line1 = lines[i + 1].strip()
                    line2 = lines[i + 2].strip()
                    
                    if not line1.startswith('1 ') or not line2.startswith('2 '):
                        continue
                    
                    # Parse orbital elements from TLE
                    # Line 1: epoch, mean motion derivatives
                    epoch_year = int(line1[18:20])
                    epoch_day = float(line1[20:32])
                    mean_motion_dot = float(line1[33:43].replace(' ', '0'))
                    bstar = self._parse_scientific_notation(line1[53:61])
                    
                    # Line 2: orbital elements
                    inclination = float(line2[8:16])  # degrees
                    raan = float(line2[17:25])        # degrees
                    eccentricity = float('0.' + line2[26:33])
                    arg_perigee = float(line2[34:42])  # degrees
                    mean_anomaly = float(line2[43:51]) # degrees
                    mean_motion = float(line2[52:63])  # revolutions per day
                    
                    # Convert to standard units
                    inclination_rad = np.radians(inclination)
                    raan_rad = np.radians(raan)
                    arg_perigee_rad = np.radians(arg_perigee)
                    mean_anomaly_rad = np.radians(mean_anomaly)
                    
                    # Calculate derived orbital parameters
                    # Semi-major axis from mean motion
                    mu = 398600.4418  # km³/s²
                    n = mean_motion * 2 * np.pi / 86400  # rad/s
                    semi_major_axis = (mu / (n**2))**(1/3)
                    
                    # Calculate altitudes
                    perigee_altitude = semi_major_axis * (1 - eccentricity) - 6371
                    apogee_altitude = semi_major_axis * (1 + eccentricity) - 6371
                    
                    # Orbital period
                    period = 2 * np.pi * np.sqrt(semi_major_axis**3 / mu) / 60  # minutes
                    
                    # Orbital velocities
                    vel_perigee = np.sqrt(mu * (2/(semi_major_axis*(1-eccentricity)) - 1/semi_major_axis))
                    vel_apogee = np.sqrt(mu * (2/(semi_major_axis*(1+eccentricity)) - 1/semi_major_axis))
                    
                    # Create orbital elements array
                    orbital_elements = np.array([
                        semi_major_axis, eccentricity, inclination_rad, raan_rad,
                        arg_perigee_rad, mean_anomaly_rad, mean_motion, mean_motion_dot,
                        perigee_altitude, apogee_altitude, bstar, period,
                        vel_perigee, vel_apogee
                    ])
                    
                    # Final physics-based risk classification
                    risk_level = self.final_risk_classification(orbital_elements)
                    
                    satellites.append({
                        'name': name_line,
                        'semi_major_axis': semi_major_axis,
                        'eccentricity': eccentricity,
                        'inclination': inclination_rad,
                        'raan': raan_rad,
                        'arg_perigee': arg_perigee_rad,
                        'mean_anomaly': mean_anomaly_rad,
                        'mean_motion': mean_motion,
                        'mean_motion_dot': mean_motion_dot,
                        'perigee_altitude': perigee_altitude,
                        'apogee_altitude': apogee_altitude,
                        'bstar': bstar,
                        'orbital_period': period,
                        'orbital_velocity_perigee': vel_perigee,
                        'orbital_velocity_apogee': vel_apogee,
                        'risk_level': risk_level,
                        'orbital_elements': orbital_elements
                    })
                    
                except (ValueError, IndexError) as e:
                    continue
            
            df = pd.DataFrame(satellites)
            print(f"✅ Parsed {len(df)} satellites successfully")
            
            # Print final risk distribution
            risk_dist = df['risk_level'].value_counts()
            print(f"📊 Final Risk Distribution: {risk_dist.to_dict()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error parsing TLE file: {e}")
            return None
    
    def _parse_scientific_notation(self, sci_str: str) -> float:
        """Parse scientific notation from TLE format"""
        try:
            sci_str = sci_str.strip()
            if sci_str == '' or sci_str == '00000+0' or sci_str == '00000-0':
                return 0.0
            
            # Handle TLE scientific notation format
            if '+' in sci_str or '-' in sci_str:
                if sci_str.endswith('+0') or sci_str.endswith('-0'):
                    return float(sci_str[:-2]) * 1e-5
                else:
                    # Extract mantissa and exponent
                    if '+' in sci_str:
                        parts = sci_str.split('+')
                        mantissa = float('0.' + parts[0])
                        exponent = int(parts[1])
                    else:
                        parts = sci_str.split('-')
                        mantissa = float('0.' + parts[0])
                        exponent = -int(parts[1])
                    
                    return mantissa * (10 ** exponent)
            else:
                return float(sci_str) * 1e-5
                
        except:
            return 0.0
    
    def train(self, tle_file: str, epochs: int = 150, batch_size: int = 256, learning_rate: float = 0.001):
        """Train the final optimized model"""
        print("🎯 Training Final Optimized Physics-Informed Model")
        print("=" * 55)
        
        # Parse data
        df = self.parse_tle_data(tle_file)
        if df is None or len(df) == 0:
            print("❌ No data to train on")
            return False
        
        # Extract features and labels
        X = []
        y = []
        
        for _, row in df.iterrows():
            # Extract final physics features
            physics_features = self.feature_extractor.calculate_physics_features(row['orbital_elements'])
            X.append(physics_features)
            y.append(row['risk_level'])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 Training data shape: {X.shape}")
        print(f"📊 Feature names: ['perigee_alt', 'drag_factor', 'eccentricity', 'alt_diff', 'bstar_factor', 'period_norm', 'inclination_risk', 'combined_risk']")
        
        # Check for NaN values
        if np.isnan(X).any():
            print("⚠️ Found NaN values in features, replacing with 0")
            X = np.nan_to_num(X)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        print(f"📊 Classes: {self.label_encoder.classes_}")
        print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Print class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"   {cls}: {count} samples ({count/len(y_train)*100:.1f}%)")
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        print(f"📊 Class weights: {dict(zip(self.label_encoder.classes_, class_weights))}")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_encoded).to(self.device)
        y_test_tensor = torch.LongTensor(y_test_encoded).to(self.device)
        
        # Initialize final model
        self.model = FinalPhysicsNet(
            input_size=X.shape[1],
            hidden_size=128,
            num_classes=len(self.label_encoder.classes_)
        ).to(self.device)
        
        # Loss and optimizer with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate*2, epochs=epochs, steps_per_epoch=1)
        
        # Training loop
        self.training_history = {'loss': [], 'accuracy': [], 'val_accuracy': []}
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            
            # Training
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Calculate training accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                train_accuracy = (predicted == y_train_tensor).float().mean().item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test_tensor)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (val_predicted == y_test_tensor).float().mean().item()
            
            # Store history
            self.training_history['loss'].append(loss.item())
            self.training_history['accuracy'].append(train_accuracy)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 15 == 0:
                print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}")
            
            # Early stopping
            if patience_counter >= 25:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test_tensor)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_accuracy = (test_predicted == y_test_tensor).float().mean().item()
        
        print(f"\n🎯 Final Results:")
        print(f"   Best Validation Accuracy: {best_val_accuracy:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        
        # Detailed classification report
        y_test_pred = test_predicted.cpu().numpy()
        y_test_true = y_test_encoded
        
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test_true, y_test_pred, 
                                  target_names=self.label_encoder.classes_))
        
        return test_accuracy > 0.85  # Success if > 85% accuracy
    
    def create_fallback_model(self):
        """Create a simple fallback model when loading fails"""
        # Create minimal scalers and encoders
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Fit with dummy data representing the actual feature ranges
        dummy_features = np.array([
            [200, 0.8, 0.01, 20, 0.5, 1.5, 1.0, 2.0],    # Critical example
            [350, 0.6, 0.05, 30, 0.3, 1.8, 0.7, 1.5],    # High example
            [600, 0.4, 0.1, 50, 0.1, 2.0, 0.5, 1.0],     # Medium example
            [1000, 0.1, 0.15, 100, 0.05, 2.5, 0.3, 0.5], # Low example
        ] * 25)  # Repeat to have 100 samples
        
        dummy_labels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] * 25
        
        self.feature_scaler.fit(dummy_features)
        self.label_encoder.fit(dummy_labels)
        
        # Create simple model
        self.model = FinalPhysicsNet(
            input_size=8,
            hidden_size=128,
            num_classes=4
        ).to(self.device)
        
        self.model.eval()
        print("Fallback model created successfully")
    
    def predict(self, orbital_elements: List[float]) -> Dict:
        """Make prediction for a single satellite"""
        if self.model is None:
            return {'error': 'Model not trained'}
        
        try:
            # Extract final physics features
            orbital_array = np.array(orbital_elements)
            physics_features = self.feature_extractor.calculate_physics_features(orbital_array)
            
            # If using fallback model, use simple altitude-based logic
            if not hasattr(self, 'training_history') or not self.training_history:
                # Simple fallback prediction based on perigee altitude
                perigee_alt = physics_features[0]  # First feature is perigee altitude
                
                if perigee_alt < 200:
                    risk_level = 'CRITICAL'
                    confidence = 0.9
                    probabilities = {'CRITICAL': 0.9, 'HIGH': 0.08, 'MEDIUM': 0.015, 'LOW': 0.005}
                elif perigee_alt < 350:
                    risk_level = 'HIGH'
                    confidence = 0.8
                    probabilities = {'CRITICAL': 0.1, 'HIGH': 0.8, 'MEDIUM': 0.08, 'LOW': 0.02}
                elif perigee_alt < 700:
                    risk_level = 'MEDIUM'
                    confidence = 0.7
                    probabilities = {'CRITICAL': 0.02, 'HIGH': 0.18, 'MEDIUM': 0.7, 'LOW': 0.1}
                else:
                    risk_level = 'LOW'
                    confidence = 0.85
                    probabilities = {'CRITICAL': 0.01, 'HIGH': 0.04, 'MEDIUM': 0.1, 'LOW': 0.85}
                
                return {
                    'risk_level': risk_level,
                    'confidence': float(confidence),
                    'probabilities': probabilities,
                    'physics_features': physics_features.tolist(),
                    'model_type': 'fallback'
                }
            
            # Normal model prediction
            # Scale features
            features_scaled = self.feature_scaler.transform([physics_features])
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
            
            # Convert to readable format
            risk_level = self.label_encoder.inverse_transform([predicted.cpu().numpy()[0]])[0]
            confidence = probabilities[0][predicted[0]].cpu().numpy()
            
            # Get all class probabilities
            class_probs = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                class_probs[class_name] = probabilities[0][i].cpu().numpy()
            
            return {
                'risk_level': risk_level,
                'confidence': float(confidence),
                'probabilities': {k: float(v) for k, v in class_probs.items()},
                'physics_features': physics_features.tolist(),
                'model_type': 'trained'
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'model_state_dict': self.model.state_dict() if self.model else None,
                'feature_scaler': self.feature_scaler,
                'label_encoder': self.label_encoder,
                'training_history': self.training_history,
                'model_type': 'final_physics_v1.0'
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model"""
        try:
            # Load the pickle file directly without custom unpickler
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.feature_scaler = model_data['feature_scaler']
            self.label_encoder = model_data['label_encoder']
            self.training_history = model_data.get('training_history', {})
            
            if model_data['model_state_dict']:
                # Recreate model architecture
                input_size = len(self.feature_scaler.mean_)
                num_classes = len(self.label_encoder.classes_)
                
                self.model = FinalPhysicsNet(
                    input_size=input_size,
                    hidden_size=128,
                    num_classes=num_classes
                ).to(self.device)
                
                # Load state dict with complete CPU mapping
                state_dict = model_data['model_state_dict']
                
                # Force all tensors to CPU if needed
                if self.device.type == 'cpu':
                    cpu_state_dict = {}
                    for key, value in state_dict.items():
                        if hasattr(value, 'cpu'):
                            cpu_state_dict[key] = value.cpu()
                        else:
                            cpu_state_dict[key] = value
                    state_dict = cpu_state_dict
                
                # Load the state dict
                self.model.load_state_dict(state_dict)
                self.model.eval()
            
            print(f"Final physics model loaded: {model_data.get('model_type', 'unknown')}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # If loading fails, let's create a simple fallback model
            try:
                print("Creating fallback model...")
                self.create_fallback_model()
                return True
            except Exception as e2:
                print(f"Fallback model creation failed: {e2}")
                return False 