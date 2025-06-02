import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import sqlite3
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

# Import our modules
from mini_tot_model import MiniTemporalOrbitalTransformer, PhysicsInformedLoss, create_mini_tot_model
from enhanced_features import EnhancedPhysicsFeatureExtractor

class SpaceDebrisDataset(Dataset):
    """Dataset class for space debris data with temporal sequences"""
    
    def __init__(self, data_path="../space_debris.db", sequence_length=7, device='cuda'):
        self.sequence_length = sequence_length
        self.device = device
        self.feature_extractor = EnhancedPhysicsFeatureExtractor(device=device)
        
        # Load data from database
        self.data, self.labels = self.load_data_from_db(data_path)
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Label distribution: {np.bincount(self.labels)}")
    
    def load_data_from_db(self, db_path: str) -> Tuple[List[np.ndarray], List[int]]:
        """Load data from SQLite database"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query space debris data
            query = """
            SELECT OBJECT_NAME, MEAN_MOTION, ECCENTRICITY, INCLINATION, 
                   RA_OF_ASC_NODE, ARG_OF_PERICENTER, MEAN_ANOMALY, 
                   RCS_SIZE, OBJECT_TYPE
            FROM space_debris
            WHERE MEAN_MOTION IS NOT NULL AND MEAN_MOTION > 0
            LIMIT 5000
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()
            
            data = []
            labels = []
            
            for row in rows:
                try:
                    object_name, mean_motion, eccentricity, inclination, \
                    ra_asc_node, arg_pericenter, mean_anomaly, rcs_size, object_type = row
                    
                    # Create TLE data dictionary
                    tle_data = {
                        'MEAN_MOTION': float(mean_motion or 15.0),
                        'ECCENTRICITY': float(eccentricity or 0.0),
                        'INCLINATION': float(inclination or 51.6),
                        'RA_OF_ASC_NODE': float(ra_asc_node or 0.0),
                        'ARG_OF_PERICENTER': float(arg_pericenter or 0.0),
                        'MEAN_ANOMALY': float(mean_anomaly or 0.0),
                        'RCS_SIZE': float(rcs_size or 1.0)
                    }
                    
                    # Create temporal sequence (simulate 7 days of data)
                    sequence = self.create_temporal_sequence(tle_data)
                    
                    # Create risk label based on orbital parameters
                    risk_label = self.calculate_risk_label(tle_data)
                    
                    data.append(sequence)
                    labels.append(risk_label)
                    
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            return data, labels
            
        except Exception as e:
            print(f"Error loading data from database: {e}")
            # Create dummy data for testing
            return self.create_dummy_data()
    
    def create_temporal_sequence(self, tle_data: Dict) -> np.ndarray:
        """Create temporal sequence for a single object"""
        sequence = []
        
        for day in range(self.sequence_length):
            # Simulate slight variations over time
            varied_tle = tle_data.copy()
            
            # Add small temporal variations
            varied_tle['MEAN_MOTION'] += np.random.normal(0, 0.001)
            varied_tle['MEAN_ANOMALY'] += day * (varied_tle['MEAN_MOTION'] * 360.0 / 1440.0)  # Daily progression
            varied_tle['MEAN_ANOMALY'] = varied_tle['MEAN_ANOMALY'] % 360.0
            
            # Extract enhanced features
            features = self.feature_extractor.extract_all_features(varied_tle)
            sequence.append(features)
        
        return np.array(sequence)
    
    def calculate_risk_label(self, tle_data: Dict) -> int:
        """Calculate risk label based on orbital parameters"""
        try:
            mean_motion = tle_data['MEAN_MOTION']
            eccentricity = tle_data['ECCENTRICITY']
            inclination = tle_data['INCLINATION']
            
            # Calculate altitude
            n = mean_motion * 2 * np.pi / 86400  # rad/s
            mu = 398600.4418  # Earth's gravitational parameter
            a = (mu / n**2)**(1/3)  # Semi-major axis
            altitude = a - 6371.0  # Altitude above Earth
            
            # Risk classification based on altitude and orbital characteristics
            if altitude < 300:
                return 3  # Very High Risk (atmospheric decay)
            elif altitude < 600:
                return 2  # High Risk (significant drag)
            elif 700 <= altitude <= 1000:
                return 2  # High Risk (debris belt)
            elif altitude < 2000:
                if eccentricity > 0.1 or inclination > 80:
                    return 1  # Medium Risk
                else:
                    return 0  # Low Risk
            else:
                return 0  # Low Risk (high altitude)
                
        except Exception as e:
            print(f"Error calculating risk label: {e}")
            return 0  # Default to low risk
    
    def create_dummy_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Create dummy data for testing"""
        print("Creating dummy data for testing...")
        
        data = []
        labels = []
        
        for i in range(1000):
            # Create random temporal sequence
            sequence = np.random.randn(self.sequence_length, 30)
            
            # Random label
            label = np.random.randint(0, 4)
            
            data.append(sequence)
            labels.append(label)
        
        return data, labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.data[idx]).to(self.device)
        label = torch.LongTensor([self.labels[idx]]).to(self.device).squeeze()
        return sequence, label


class MiniTOTTrainer:
    """GPU-optimized trainer for Mini Temporal-Orbital Transformer"""
    
    def __init__(self, model, device='cuda', save_dir='./models'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'physics_loss': [],
            'uncertainty_loss': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_physics_loss = 0.0
        total_uncertainty_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions, uncertainty = self.model(sequences)
            
            # Compute loss
            loss, class_loss, physics_loss, uncertainty_loss = criterion(
                predictions, uncertainty, labels
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_physics_loss += physics_loss.item()
            total_uncertainty_loss += uncertainty_loss.item()
            
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: '
                      f'Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        avg_physics_loss = total_physics_loss / len(train_loader)
        avg_uncertainty_loss = total_uncertainty_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, avg_physics_loss, avg_uncertainty_loss
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                
                # Forward pass
                predictions, uncertainty = self.model(sequences)
                
                # Compute loss
                loss, _, _, _ = criterion(predictions, uncertainty, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_uncertainties.extend(uncertainty.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, all_predictions, all_labels, all_uncertainties
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, weight_decay=1e-5):
        """Full training loop with GPU optimization"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Model size: {self.model.get_model_size():.2f} MB")
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=lr*10, 
            epochs=epochs, 
            steps_per_epoch=len(train_loader)
        )
        
        # Loss function
        criterion = PhysicsInformedLoss(alpha=1.0, beta=0.1, gamma=0.05)
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train epoch
            train_loss, train_acc, physics_loss, uncertainty_loss = self.train_epoch(
                train_loader, optimizer, criterion, epoch
            )
            
            # Validate epoch
            val_loss, val_acc, val_predictions, val_labels, val_uncertainties = self.validate_epoch(
                val_loader, criterion
            )
            
            # Update learning rate
            scheduler.step()
            
            # Store history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['physics_loss'].append(physics_loss)
            self.train_history['uncertainty_loss'].append(uncertainty_loss)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_path = os.path.join(self.save_dir, f'best_mini_tot_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_history': self.train_history
                }, self.best_model_path)
                print(f"New best model saved: {val_acc:.2f}%")
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Physics Loss: {physics_loss:.4f}, Uncertainty Loss: {uncertainty_loss:.4f}')
            print(f'  Time: {epoch_time:.2f}s, LR: {scheduler.get_last_lr()[0]:.6f}')
            print('-' * 60)
            
            # Early stopping check
            if epoch > 20 and val_acc < max(self.train_history['val_acc'][-10:]) - 5:
                print("Early stopping triggered due to validation accuracy drop")
                break
        
        print(f"Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")
        return self.train_history
    
    def evaluate_model(self, test_loader):
        """Comprehensive model evaluation"""
        if self.best_model_path:
            # Load best model
            checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model with validation accuracy: {checkpoint['val_acc']:.2f}%")
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        inference_times = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                predictions, uncertainty = self.model(sequences)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                _, predicted = torch.max(predictions.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_uncertainties.extend(uncertainty.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        avg_inference_time = np.mean(inference_times)
        predictions_per_second = len(test_loader.dataset) / sum(inference_times)
        
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        print(f"Average Inference Time: {avg_inference_time:.4f}s per batch")
        print(f"Predictions per Second: {predictions_per_second:.1f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low', 'Medium', 'High', 'Very High'],
                   yticklabels=['Low', 'Medium', 'High', 'Very High'])
        plt.title('Confusion Matrix - Mini TOT Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), dpi=300)
        plt.show()
        
        return {
            'accuracy': accuracy,
            'predictions_per_second': predictions_per_second,
            'avg_inference_time': avg_inference_time,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels,
            'uncertainties': all_uncertainties
        }
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.train_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.train_history['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(self.train_history['val_acc'], label='Validation Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Physics loss
        axes[1, 0].plot(self.train_history['physics_loss'], label='Physics Loss')
        axes[1, 0].set_title('Physics Constraint Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Physics Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Uncertainty loss
        axes[1, 1].plot(self.train_history['uncertainty_loss'], label='Uncertainty Loss')
        axes[1, 1].set_title('Uncertainty Regularization Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Uncertainty Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=300)
        plt.show()


def main():
    """Main training function"""
    print("🚀 Starting RSDRAS-Lite Mini-TOT Training")
    print("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = SpaceDebrisDataset(sequence_length=7, device=device)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    batch_size = 32 if device.type == 'cuda' else 16
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    
    # Create model
    print("\nCreating Mini-TOT model...")
    model = create_mini_tot_model(
        device=device,
        input_features=30,
        hidden_dim=64,
        num_heads=4,
        sequence_length=7,
        num_classes=4,
        dropout=0.2
    )
    
    # Create trainer
    trainer = MiniTOTTrainer(model, device=device)
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=0.001,
        weight_decay=1e-5
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    print("\nEvaluating model...")
    results = trainer.evaluate_model(test_loader)
    
    # Save results
    results_summary = {
        'model_type': 'Mini Temporal-Orbital Transformer',
        'test_accuracy': float(results['accuracy']),
        'predictions_per_second': float(results['predictions_per_second']),
        'model_size_mb': float(model.get_model_size()),
        'total_parameters': int(model.count_parameters()),
        'best_val_accuracy': float(trainer.best_val_acc),
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(trainer.save_dir, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n🎉 Training completed successfully!")
    print(f"Results saved to: {trainer.save_dir}")
    print(f"Best model: {trainer.best_model_path}")


if __name__ == "__main__":
    main() 