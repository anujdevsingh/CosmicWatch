#!/usr/bin/env python3
"""
🚀 COSMIC INTELLIGENCE MODEL TRAINING SCRIPT 🚀
==============================================
Comprehensive training script for the revolutionary CIM model

Features:
- Multi-stage training with physics-informed constraints
- Advanced data preprocessing and augmentation
- Real-time progress monitoring
- Automatic model saving and validation
- Performance benchmarking against existing models

For IIT Madras Space Technology Competition
Target: >98% accuracy
"""

import os
import sys
import time
import json
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# Import our Cosmic Intelligence Model
from cosmic_intelligence_model import (
    CosmicIntelligenceWrapper,
    get_cosmic_intelligence_model,
    CosmicConfig
)

def print_banner():
    """Print the awesome CIM training banner"""
    banner = """
    🌌🚀 COSMIC INTELLIGENCE MODEL TRAINING 🚀🌌
    ===============================================
    
          ⭐ Revolutionary Space Debris AI ⭐
          🎯 Target Accuracy: >98%
          🏆 IIT Madras Competition Ready
          
    Features:
    • Physics-Informed Neural Networks
    • Multi-Modal Transformer Architecture  
    • Real-time Uncertainty Quantification
    • Advanced Space Debris Risk Assessment
    
    ===============================================
    """
    print(banner)

def check_system_requirements():
    """Check if system meets training requirements"""
    print("🔍 Checking system requirements...")
    
    # Check PyTorch and CUDA
    print(f"  PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA available: ✅ {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  CUDA available: ❌ (will use CPU)")
    
    # Check memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / 1e9
        print(f"  System RAM: {memory_gb:.1f} GB")
        if memory_gb < 8:
            print("  ⚠️ Warning: Low RAM detected. Training may be slow.")
    except ImportError:
        print("  System RAM: Unknown (psutil not available)")
    
    # Check disk space
    try:
        import shutil
        free_space_gb = shutil.disk_usage('.').free / 1e9
        print(f"  Free disk space: {free_space_gb:.1f} GB")
        if free_space_gb < 5:
            print("  ⚠️ Warning: Low disk space. May affect model saving.")
    except:
        print("  Free disk space: Unknown")
    
    print("✅ System check completed!\n")

def verify_data_availability():
    """Verify that all required data files are available"""
    print("📊 Verifying training data availability...")
    
    required_files = [
        'space_debris.csvobjects_catalog_20250427.csv',
        'space_debris_real.txt',
        'space_debris.db'
    ]
    
    missing_files = []
    total_size = 0
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / 1e6
            total_size += size_mb
            print(f"  ✅ {file_path} ({size_mb:.1f} MB)")
        else:
            missing_files.append(file_path)
            print(f"  ❌ {file_path} (missing)")
    
    if missing_files:
        print(f"\n⚠️ Warning: {len(missing_files)} data files missing")
        print("Training will proceed with available data only.")
    else:
        print(f"\n✅ All data files found! Total size: {total_size:.1f} MB")
    
    return len(missing_files) == 0

def setup_training_environment():
    """Setup the training environment and directories"""
    print("🔧 Setting up training environment...")
    
    # Create necessary directories
    directories = ['models', 'logs', 'checkpoints', 'results']
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"  📁 Created directory: {dir_name}")
        else:
            print(f"  📁 Directory exists: {dir_name}")
    
    # Setup logging
    log_file = f"logs/cim_training_{int(time.time())}.log"
    print(f"  📝 Log file: {log_file}")
    
    return log_file

def create_training_config():
    """Create and display training configuration"""
    print("⚙️ Creating training configuration...")
    
    config = CosmicConfig()
    
    # Display configuration
    print("  Configuration:")
    print(f"    Model: {config.hidden_dim}D transformer with {config.num_transformer_layers} layers")
    print(f"    Attention heads: {config.num_attention_heads}")
    print(f"    Sequence length: {config.sequence_length}")
    print(f"    Batch size: {config.batch_size}")
    print(f"    Learning rate: {config.learning_rate}")
    print(f"    Max epochs: {config.max_epochs}")
    print(f"    Early stopping patience: {config.patience}")
    
    return config

def train_cosmic_model():
    """Main training function"""
    start_time = time.time()
    
    try:
        print("🚀 Initializing Cosmic Intelligence Model...")
        cosmic_model = get_cosmic_intelligence_model()
        
        print("📊 Model Information:")
        model_info = cosmic_model.get_model_info()
        print(f"  Model: {model_info['model_name']} v{model_info['model_version']}")
        print(f"  Device: {model_info['device']}")
        print(f"  Parameters: {model_info['num_parameters']:,}")
        print(f"  Trainable parameters: {model_info['num_trainable_params']:,}")
        
        print("\n🎯 Starting training process...")
        training_results = cosmic_model.train_model()
        
        if not training_results:
            print("❌ Training failed - no results returned")
            return False
        
        # Display training results
        print("\n🎉 Training completed successfully!")
        print("📊 Final Results:")
        print(f"  Best Accuracy: {training_results.get('best_accuracy', 0):.4f}")
        
        if 'final_metrics' in training_results:
            metrics = training_results['final_metrics']
            print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
            print(f"  AUC Score: {metrics.get('auc_score', 0):.4f}")
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall: {metrics.get('recall', 0):.4f}")
        
        # Save training results
        results_file = f"results/cim_training_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            # Convert any numpy types to regular Python types for JSON serialization
            json_results = {}
            for key, value in training_results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                       for k, v in value.items()}
                elif isinstance(value, (np.float32, np.float64)):
                    json_results[key] = float(value)
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2)
        print(f"💾 Results saved to: {results_file}")
        
        training_time = time.time() - start_time
        print(f"⏱️ Total training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_against_existing_models():
    """Benchmark CIM against existing models"""
    print("\n🏁 Benchmarking against existing models...")
    
    try:
        # Try to load the CIM model
        cosmic_model = get_cosmic_intelligence_model()
        
        if cosmic_model.is_loaded:
            cim_accuracy = cosmic_model.accuracy
            cim_f1 = cosmic_model.f1_score
            
            print("📊 Model Comparison:")
            print(f"  🌌 Cosmic Intelligence Model: {cim_accuracy:.4f} accuracy, {cim_f1:.4f} F1")
            
            # Compare with previous models
            previous_models = {
                "SuperNova Fixed": {"accuracy": 0.9606, "f1": 0.9610},
                "RSDRAS_Lite": {"accuracy": 0.85, "f1": 0.83},  # Estimated
                "Physics-Informed": {"accuracy": 0.78, "f1": 0.76}  # Estimated
            }
            
            for model_name, metrics in previous_models.items():
                accuracy_improvement = (cim_accuracy - metrics["accuracy"]) * 100
                f1_improvement = (cim_f1 - metrics["f1"]) * 100
                
                print(f"  📈 vs {model_name}: +{accuracy_improvement:.2f}% accuracy, +{f1_improvement:.2f}% F1")
            
            if cim_accuracy > 0.98:
                print("🏆 TARGET ACHIEVED: >98% accuracy reached!")
            else:
                print(f"🎯 Target Progress: {cim_accuracy/0.98*100:.1f}% towards 98% goal")
        else:
            print("⚠️ CIM model not loaded - skipping benchmark")
            
    except Exception as e:
        print(f"⚠️ Benchmarking failed: {e}")

def main():
    """Main training pipeline"""
    print_banner()
    
    # Pre-training checks
    check_system_requirements()
    data_available = verify_data_availability()
    log_file = setup_training_environment()
    config = create_training_config()
    
    if not data_available:
        print("⚠️ Proceeding with limited data - results may be suboptimal")
    
    print("\n" + "="*60)
    print("🚀 STARTING COSMIC INTELLIGENCE MODEL TRAINING")
    print("="*60)
    
    # Train the model
    success = train_cosmic_model()
    
    if success:
        # Benchmark against existing models
        benchmark_against_existing_models()
        
        print("\n" + "="*60)
        print("🎉 COSMIC INTELLIGENCE MODEL TRAINING COMPLETE!")
        print("="*60)
        print("✅ Model successfully trained and ready for deployment")
        print("🚀 Ready for IIT Madras competition submission!")
        print("📄 Check 'results' directory for detailed training metrics")
        print("💾 Model saved as 'cosmic_intelligence_best.pth'")
        
    else:
        print("\n" + "="*60)
        print("❌ TRAINING FAILED")
        print("="*60)
        print("Please check the error messages above and try again.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 