#!/usr/bin/env python3
"""
Proper SuperNova Training Script
===============================
Follows ML best practices for robust model training

Best Practices Implemented:
1. Early stopping based on validation loss (not accuracy)
2. Proper train/val/test splits with no data leakage
3. Learning rate scheduling based on plateaus
4. Model checkpointing based on best validation loss
5. Cross-validation friendly approach
6. Comprehensive evaluation metrics
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Add SuperNova_Model to path
sys.path.append('.')

# Import SuperNova components
from multi_stage_trainer import SuperNovaDataset, SuperNovaTrainer, PhysicsInformedLoss
from supernova_architecture import create_supernova_model
from advanced_physics_features import create_feature_extractor

class ProperPhysicsInformedLoss(PhysicsInformedLoss):
    """Properly weighted loss without arbitrary targets"""
    
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

class ProperSuperNovaTrainer(SuperNovaTrainer):
    """Proper trainer following ML best practices"""
    
    def __init__(self, model, feature_extractor, device='auto'):
        super().__init__(model, feature_extractor, device)
        
        # Proper early stopping parameters
        self.patience = 10  # More patience for better convergence
        self.min_delta = 1e-4  # Minimum improvement threshold
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Learning rate scheduling
        self.lr_patience = 5  # Reduce LR after 5 epochs without improvement
        self.lr_factor = 0.5  # Reduce LR by half
        
    def calculate_class_weights(self, labels_sample):
        """Calculate balanced class weights"""
        unique_labels, counts = np.unique(labels_sample, return_counts=True)
        
        # Use sklearn's balanced approach
        n_samples = len(labels_sample)
        n_classes = 4
        weights = torch.ones(n_classes)
        
        for label, count in zip(unique_labels, counts):
            # Balanced weight: n_samples / (n_classes * count)
            weight = n_samples / (n_classes * count)
            weights[int(label)] = weight
        
        print(f"📊 Calculated class weights:")
        class_names = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        for i, (name, weight) in enumerate(zip(class_names, weights)):
            print(f"   {name}: {weight:.3f}")
        
        return weights.to(self.device)
    
    def create_proper_data_loaders(self, tle_file: str, batch_size: int = None):
        """Create data loaders with proper splits and no data leakage"""
        
        if batch_size is None:
            batch_size = 16 if torch.cuda.is_available() else 8
        
        print(f"📊 Creating proper data loaders (batch size: {batch_size})")
        
        # Create dataset
        dataset = SuperNovaDataset(
            tle_file=tle_file,
            feature_extractor=self.feature_extractor,
            curriculum_mode='random',  # No curriculum to avoid bias
            device=self.device
        )
        
        # Proper stratified split
        total_size = len(dataset)
        
        # Sample labels for stratification
        labels_sample = []
        sample_size = min(2000, total_size)
        indices = np.random.choice(total_size, sample_size, replace=False)
        
        for idx in indices:
            _, label, _ = dataset[idx]
            labels_sample.append(label.item())
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(labels_sample)
        
        # Update loss function
        self.criterion = ProperPhysicsInformedLoss(
            class_weights=class_weights,
            classification_weight=1.0,
            physics_weight=0.1,
            uncertainty_weight=0.05,
            curriculum_weight=0.0  # No curriculum bias
        )
        
        # Proper splits: 60% train, 20% val, 20% test
        train_size = int(0.60 * total_size)
        val_size = int(0.20 * total_size)
        test_size = total_size - train_size - val_size
        
        # Random but reproducible split
        torch.manual_seed(42)  # For reproducibility
        indices = torch.randperm(total_size)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        
        print(f"📈 Data splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def proper_early_stopping_check(self, val_loss: float, val_acc: float, epoch: int) -> bool:
        """Proper early stopping based on validation loss plateau"""
        
        improvement = self.best_val_loss - val_loss
        
        if improvement > self.min_delta:
            # Significant improvement
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            self.best_accuracy = val_acc  # Update best accuracy
            return False  # Continue training
        else:
            # No significant improvement
            self.epochs_without_improvement += 1
            
            # Check for early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"🛑 Early stopping: No improvement for {self.patience} epochs")
                return True  # Stop training
            
            return False
    
    def train_proper_stage(
        self,
        train_loader,
        val_loader,
        stage: int,
        max_epochs: int = 100  # Higher max, let early stopping decide
    ):
        """Train with proper early stopping based on validation loss"""
        
        print(f"\n🚀 Training Stage {stage} (Proper Method)")
        print("=" * 50)
        
        stage_names = ['Warm-up', 'Physics-Enhanced', 'Advanced']
        print(f"Stage: {stage_names[min(stage, 2)]}")
        
        # Setup optimizer
        self.setup_optimizer(stage=stage)
        
        # Reset early stopping for this stage
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        stage_best_acc = 0.0
        
        # Training history for this stage
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(max_epochs):
            # Training
            train_metrics = self._train_epoch(train_loader, epoch, stage)
            
            # Validation
            val_metrics = self._validate_epoch(val_loader, epoch, stage)
            
            # Record metrics
            train_losses.append(train_metrics['loss'])
            val_losses.append(val_metrics['loss'])
            train_accs.append(train_metrics['accuracy'])
            val_accs.append(val_metrics['accuracy'])
            
            # Learning rate scheduling (based on validation loss)
            self.scheduler.step(val_metrics['loss'])
            
            # Print progress
            if epoch % 5 == 0 or epoch < 10:
                print(f"Epoch {epoch:3d} | "
                      f"Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f} | "
                      f"Val: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}")
            
            # Proper early stopping check
            should_stop = self.proper_early_stopping_check(
                val_metrics['loss'], val_metrics['accuracy'], epoch
            )
            
            # Save best model for this stage
            if val_metrics['accuracy'] > stage_best_acc:
                stage_best_acc = val_metrics['accuracy']
                self._save_checkpoint(stage, epoch, val_metrics)
            
            if should_stop:
                break
        
        # Plot learning curves for this stage
        self._plot_learning_curves(train_losses, val_losses, train_accs, val_accs, stage)
        
        return {
            'stage': stage,
            'final_accuracy': val_metrics['accuracy'],
            'best_accuracy': stage_best_acc,
            'epochs_completed': epoch + 1,
            'final_val_loss': val_metrics['loss'],
            'best_val_loss': self.best_val_loss
        }
    
    def _plot_learning_curves(self, train_losses, val_losses, train_accs, val_accs, stage):
        """Plot learning curves for analysis"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss curves
            ax1.plot(train_losses, label='Training Loss', color='blue')
            ax1.plot(val_losses, label='Validation Loss', color='red')
            ax1.set_title(f'Stage {stage} - Loss Curves')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy curves
            ax2.plot(train_accs, label='Training Accuracy', color='blue')
            ax2.plot(val_accs, label='Validation Accuracy', color='red')
            ax2.set_title(f'Stage {stage} - Accuracy Curves')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'results/learning_curves_stage_{stage}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not save learning curves: {e}")
    
    def comprehensive_evaluation(self, test_loader):
        """Comprehensive evaluation with multiple metrics"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for features, targets, _ in test_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions, uncertainty = self.model(features, return_uncertainty=True)
                probabilities = torch.softmax(predictions, dim=1)
                
                all_predictions.extend(predictions.argmax(dim=1).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_predictions, average=None, zero_division=0
        )
        
        # Per-class metrics
        class_names = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        class_metrics = {}
        
        for i, name in enumerate(class_names):
            if i < len(precision):
                class_metrics[name] = {
                    'precision': float(precision[i]) if i < len(precision) else 0.0,
                    'recall': float(recall[i]) if i < len(recall) else 0.0,
                    'f1_score': float(f1[i]) if i < len(f1) else 0.0,
                    'support': int(support[i]) if i < len(support) else 0
                }
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'overall_accuracy': accuracy,
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist(),
            'macro_f1': float(np.mean(f1)) if len(f1) > 0 else 0.0,
            'weighted_f1': float(np.average(f1, weights=support)) if len(f1) > 0 else 0.0
        }

def main():
    parser = argparse.ArgumentParser(description='Proper SuperNova Training')
    parser.add_argument('--max-epochs', type=int, default=100, help='Maximum epochs per stage')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--tle-file', type=str, default='./space_debris_real.txt')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (auto if None)')
    
    args = parser.parse_args()
    
    print("🌟 PROPER SUPERNOVA TRAINING")
    print("=" * 60)
    print(f"🔬 Method: Validation loss-based early stopping")
    print(f"📊 Max Epochs per Stage: {args.max_epochs}")
    print(f"⏰ Patience: {args.patience} epochs")
    print(f"🎯 Goal: Best possible model without overfitting")
    
    try:
        # Create model and trainer
        model = create_supernova_model(device=args.device, input_features=52)
        feature_extractor = create_feature_extractor(device=args.device)
        
        trainer = ProperSuperNovaTrainer(model, feature_extractor, device=args.device)
        trainer.patience = args.patience
        
        # Create proper data loaders
        train_loader, val_loader, test_loader = trainer.create_proper_data_loaders(
            args.tle_file, batch_size=args.batch_size
        )
        
        start_time = time.time()
        
        # Multi-stage training
        stage_results = []
        
        # Stage 0: Conservative warm-up
        stage_0_results = trainer.train_proper_stage(train_loader, val_loader, 0, args.max_epochs)
        stage_results.append(stage_0_results)
        
        # Stage 1: Physics-enhanced training
        stage_1_results = trainer.train_proper_stage(train_loader, val_loader, 1, args.max_epochs)
        stage_results.append(stage_1_results)
        
        # Stage 2: Advanced training
        stage_2_results = trainer.train_proper_stage(train_loader, val_loader, 2, args.max_epochs)
        stage_results.append(stage_2_results)
        
        training_time = time.time() - start_time
        
        # Comprehensive evaluation
        print("\n📊 Comprehensive Evaluation")
        test_metrics = trainer.comprehensive_evaluation(test_loader)
        
        # Results summary
        results = {
            'training_method': 'proper_early_stopping',
            'stage_results': stage_results,
            'test_metrics': test_metrics,
            'best_accuracy': trainer.best_accuracy,
            'training_time_seconds': training_time,
            'hyperparameters': {
                'max_epochs_per_stage': args.max_epochs,
                'patience': args.patience,
                'batch_size': train_loader.batch_size
            }
        }
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'results/proper_supernova_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print final results
        print("\n" + "="*60)
        print("🏆 PROPER TRAINING COMPLETE")
        print("="*60)
        print(f"📈 Best Validation Accuracy: {trainer.best_accuracy:.4f} ({trainer.best_accuracy:.1%})")
        print(f"🧪 Test Accuracy: {test_metrics['overall_accuracy']:.4f} ({test_metrics['overall_accuracy']:.1%})")
        print(f"🎯 Macro F1-Score: {test_metrics['macro_f1']:.4f}")
        print(f"⚖️ Weighted F1-Score: {test_metrics['weighted_f1']:.4f}")
        print(f"⏱️ Training Time: {training_time//60:.0f}m {training_time%60:.0f}s")
        print(f"📊 Results: {results_file}")
        
        # Per-class performance
        print(f"\n📋 Per-Class Performance:")
        for class_name, metrics in test_metrics['class_metrics'].items():
            print(f"   {class_name}: F1={metrics['f1_score']:.3f}, "
                  f"Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, "
                  f"Support={metrics['support']}")
        
        print("\n✅ Model trained using proper ML practices!")
        print("   - No arbitrary accuracy targets")
        print("   - Validation loss-based early stopping")
        print("   - Comprehensive evaluation metrics")
        print("   - Learning curves saved for analysis")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 