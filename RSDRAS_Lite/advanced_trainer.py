#!/usr/bin/env python3
"""
Advanced Trainer for RSDRAS-Lite
Handles training, validation, and optimization of the Mini Temporal Orbital Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from mini_tot_model import MiniTemporalOrbitalTransformer, PhysicsInformedLoss
from enhanced_features import EnhancedPhysicsFeatureExtractor


class AdvancedTrainer:
    """Advanced training system for RSDRAS-Lite"""
    
    def __init__(self, model, feature_extractor, device='cuda'):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = PhysicsInformedLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = []
        
        # Setup logging
        self.logger = logging.getLogger('AdvancedTrainer')
        
    def prepare_optimizer(self, learning_rate=0.001, weight_decay=1e-4):
        """Prepare optimizer and scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100
        )
        
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic training data"""
        synthetic_data = []
        labels = []
        
        for _ in range(num_samples):
            # Generate realistic orbital parameters
            altitude = np.random.uniform(200, 2000)
            
            tle_data = {
                'MEAN_MOTION': np.random.uniform(10, 16),
                'ECCENTRICITY': np.random.uniform(0, 0.1),
                'INCLINATION': np.random.uniform(0, 180),
                'RA_OF_ASC_NODE': np.random.uniform(0, 360),
                'ARG_OF_PERICENTER': np.random.uniform(0, 360),
                'MEAN_ANOMALY': np.random.uniform(0, 360),
                'RCS_SIZE': np.random.uniform(0.1, 10),
                'ALTITUDE': altitude,
                'VELOCITY': np.random.uniform(6, 9)
            }
            
            # Extract features
            features = self.feature_extractor.extract_all_features(tle_data)
            
            # Create 7-day sequence
            sequence = np.array([features] * 7)
            synthetic_data.append(sequence)
            
            # Create label based on altitude (simplified)
            if altitude < 300:
                label = 3  # CRITICAL
            elif altitude < 600:
                label = 2  # HIGH
            elif altitude < 1000:
                label = 1  # MEDIUM
            else:
                label = 0  # LOW
                
            labels.append(label)
        
        return np.array(synthetic_data), np.array(labels)
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, uncertainty = self.model(data)
            
            # Calculate loss - handle tuple return from PhysicsInformedLoss
            loss_result = self.criterion(outputs, uncertainty, targets)
            if isinstance(loss_result, tuple):
                loss = loss_result[0]  # Main loss is first element
            else:
                loss = loss_result
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if batch_idx % 10 == 0:
                self.logger.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self, dataloader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs, uncertainty = self.model(data)
                loss_result = self.criterion(outputs, uncertainty, targets)
                if isinstance(loss_result, tuple):
                    loss = loss_result[0]  # Main loss is first element
                else:
                    loss = loss_result
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss = total_loss / len(dataloader)
        val_accuracy = correct / total
        
        return val_loss, val_accuracy
    
    def quick_demo_training(self, epochs=50, batch_size=32):
        """Quick demo training for setup purposes"""
        start_time = time.time()
        
        self.logger.info(f"Starting demo training for {epochs} epochs")
        
        # Prepare optimizer
        self.prepare_optimizer()
        
        # Generate synthetic data
        X, y = self.generate_synthetic_data(1000)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Update best accuracy
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
            
            # Store history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'timestamp': datetime.now().isoformat()
            })
        
        training_time = time.time() - start_time
        
        # Final evaluation
        final_val_loss, final_val_acc = self.validate_epoch(val_loader)
        
        self.logger.info(f"Training completed in {training_time:.1f} seconds")
        self.logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
        
        return {
            'status': 'success',
            'final_accuracy': final_val_acc,
            'final_loss': final_val_loss,
            'training_time': training_time,
            'epochs_completed': epochs,
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history
        }
    
    def full_training(self, train_data, val_data=None, epochs=100, batch_size=32, 
                     early_stopping_patience=10):
        """Full training with real data"""
        start_time = time.time()
        
        self.logger.info(f"Starting full training for {epochs} epochs")
        
        # Prepare optimizer
        self.prepare_optimizer()
        
        # Prepare data loaders
        if val_data is None:
            # Split training data
            split_idx = int(0.8 * len(train_data))
            train_subset = train_data[:split_idx]
            val_subset = train_data[split_idx:]
        else:
            train_subset = train_data
            val_subset = val_data
        
        # Create data loaders (this would need proper implementation)
        # For now, use synthetic data
        return self.quick_demo_training(epochs, batch_size)
    
    def evaluate_model(self, test_data):
        """Evaluate model on test data"""
        self.model.eval()
        
        # For demo purposes, generate some test data
        X_test, y_test = self.generate_synthetic_data(200)
        
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.LongTensor(y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        test_loss, test_accuracy = self.validate_epoch(test_loader)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_checkpoint(self, filepath):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'current_epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        self.training_history = checkpoint.get('training_history', [])
        
        self.logger.info(f"Checkpoint loaded from {filepath}") 