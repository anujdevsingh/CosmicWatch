import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import traceback
from torch.utils.data import TensorDataset
import torch.optim as optim

class RiskDataset(Dataset):
    """Dataset for training the PNN risk classifier"""

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class ProbabilisticNeuralNetwork(nn.Module):
    """Probabilistic Neural Network for space debris risk classification"""

    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super(ProbabilisticNeuralNetwork, self).__init__()

        # Network architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Dropout for uncertainty
        self.dropout = nn.Dropout(0.2)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        # First layer with ReLU activation and batch normalization
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))

        # Second layer
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))

        # Output layer with softmax for probabilities
        x = self.fc3(x)

        return x

    def predict_proba(self, x):
        """Return class probabilities"""
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            logits = self.forward(x_tensor)
            return F.softmax(logits, dim=1).numpy()

class RiskClassifier:
    """Class for training and using the PNN risk classifier"""

    def __init__(self, input_dim=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ProbabilisticNeuralNetwork(input_dim).to(self.device)
        self.is_trained = False
        self.input_dim = input_dim

    def prepare_features(self, debris_data, collision_data):
        """Prepare features for the risk classifier"""
        features = []
        labels = []

        for collision in collision_data:
            # Find the debris objects
            obj1 = next((x for x in debris_data if x['id'] == collision['object1_id']), None)
            obj2 = next((x for x in debris_data if x['id'] == collision['object2_id']), None)

            if obj1 is None or obj2 is None:
                continue

            # Create feature vector
            feature = [
                collision['min_distance'],         # Minimum distance
                collision['probability'],          # Base probability
                collision['relative_velocity'],    # Relative velocity
                obj1['altitude'],                  # Altitude of first object
                obj2['altitude'],                  # Altitude of second object
                obj1['size'],                      # Size of first object
                obj2['size'],                      # Size of second object
                obj1['risk_score'],                # Risk score of first object
                obj2['risk_score'],                # Risk score of second object
                obj1['inclination'] - obj2['inclination']  # Inclination difference
            ]

            # Convert severity to numerical label
            if collision['severity'] == 'high':
                label = 2
            elif collision['severity'] == 'medium':
                label = 1
            else:  # low
                label = 0

            features.append(feature)
            labels.append(label)

        return np.array(features), np.array(labels)

    def train(self, debris_data, collision_data, epochs=100, batch_size=16):
        """Train the PNN model"""
        try:
            # Prepare data
            X, y = self.prepare_features(debris_data, collision_data)

            # Changed requirement from 10 to 5 samples 
            if len(X) < 5:
                print(f"Not enough data for training (only {len(X)} samples)")
                return False

            print(f"Training PNN with {len(X)} samples")

            # Create dataset and dataloader
            dataset = RiskDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            # Training loop
            self.model.train()
            for epoch in range(epochs):
                running_loss = 0.0

                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss: {running_loss/len(dataloader):.4f}')

            self.is_trained = True
            print("PNN model training completed")
            return True

        except Exception as e:
            print(f"Error training PNN model: {str(e)}")
            traceback.print_exc()
            return False

    def predict_risk(self, feature):
        """Predict risk class and probabilities for a single feature vector"""
        if not self.is_trained:
            print("Model not trained yet")
            return {'class': 'medium', 'probabilities': [0.33, 0.34, 0.33]}

        try:
            # Ensure feature is the right dimension
            if len(feature) != self.input_dim:
                print(f"Feature has wrong dimension: {len(feature)}, expected {self.input_dim}")
                # Pad with zeros or truncate if necessary
                if len(feature) < self.input_dim:
                    feature = np.pad(feature, (0, self.input_dim - len(feature)))
                else:
                    feature = feature[:self.input_dim]

            # Convert feature to tensor
            feature_tensor = torch.tensor([feature], dtype=torch.float32).to(self.device)

            # Get predictions
            with torch.no_grad():
                self.model.eval()  # Set to evaluation mode
                probabilities = self.model(feature_tensor)

            # Convert to numpy for processing
            probs = probabilities.cpu().numpy()[0]
            pred_class = np.argmax(probs)

            # Map to severity labels
            severity_labels = ['low', 'medium', 'high']
            severity = severity_labels[pred_class]

            return {
                'class': severity,
                'probabilities': probs.tolist()
            }
        except Exception as e:
            print(f"Error in PNN prediction: {str(e)}")
            traceback.print_exc()
            return {'class': 'medium', 'probabilities': [0.33, 0.34, 0.33]}