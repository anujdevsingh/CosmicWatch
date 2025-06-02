import torch
import torch.nn as nn
import numpy as np
import time
from mini_tot_model import create_mini_tot_model
from enhanced_features import EnhancedPhysicsFeatureExtractor

def test_gpu_setup():
    """Test GPU setup and CUDA availability"""
    print("🚀 RSDRAS-Lite GPU Setup Test")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        
        # GPU memory info
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        
        print(f"GPU Memory - Total: {memory_total:.1f} GB")
        print(f"GPU Memory - Reserved: {memory_reserved:.1f} GB")
        print(f"GPU Memory - Allocated: {memory_allocated:.1f} GB")
        print(f"GPU Memory - Available: {memory_total - memory_reserved:.1f} GB")
        
        device = torch.device('cuda')
    else:
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    
    return device

def test_feature_extraction(device):
    """Test enhanced feature extraction"""
    print("\n📊 Testing Enhanced Feature Extraction")
    print("-" * 40)
    
    # Create feature extractor
    feature_extractor = EnhancedPhysicsFeatureExtractor(device=device)
    
    # Sample TLE data
    sample_tle = {
        'MEAN_MOTION': 15.5,
        'ECCENTRICITY': 0.01,
        'INCLINATION': 51.6,
        'RA_OF_ASC_NODE': 45.0,
        'ARG_OF_PERICENTER': 90.0,
        'MEAN_ANOMALY': 180.0,
        'RCS_SIZE': 1.0
    }
    
    # Test single feature extraction
    start_time = time.time()
    features = feature_extractor.extract_all_features(sample_tle, "TEST-001")
    extraction_time = time.time() - start_time
    
    print(f"✅ Feature extraction successful!")
    print(f"   Features extracted: {len(features)}")
    print(f"   Extraction time: {extraction_time:.4f}s")
    print(f"   Feature range: [{features.min():.6f}, {features.max():.6f}]")
    
    # Test temporal sequence creation
    start_time = time.time()
    tle_list = [sample_tle] * 7
    sequence = feature_extractor.create_temporal_sequence(tle_list)
    sequence_time = time.time() - start_time
    
    print(f"✅ Temporal sequence creation successful!")
    print(f"   Sequence shape: {sequence.shape}")
    print(f"   Sequence time: {sequence_time:.4f}s")
    
    return features, sequence

def test_model_creation(device):
    """Test Mini-TOT model creation and forward pass"""
    print("\n🧠 Testing Mini-TOT Model Creation")
    print("-" * 40)
    
    # Create model
    start_time = time.time()
    model = create_mini_tot_model(
        device=device,
        input_features=30,
        hidden_dim=64,
        num_heads=4,
        sequence_length=7,
        num_classes=4,
        dropout=0.2
    )
    creation_time = time.time() - start_time
    
    print(f"✅ Model creation successful!")
    print(f"   Creation time: {creation_time:.4f}s")
    print(f"   Model size: {model.get_model_size():.2f} MB")
    print(f"   Total parameters: {model.count_parameters():,}")
    
    # Test model on device
    model.to(device)
    print(f"✅ Model moved to {device}")
    
    return model

def test_model_inference(model, device):
    """Test model inference with GPU acceleration"""
    print("\n⚡ Testing Model Inference")
    print("-" * 40)
    
    # Create test batch
    batch_size = 32
    sequence_length = 7
    input_features = 30
    
    # Generate random input
    test_input = torch.randn(batch_size, sequence_length, input_features).to(device)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input device: {test_input.device}")
    
    # Warm-up runs
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            predictions, uncertainty = model(test_input)
    
    # Timed inference
    num_runs = 10
    inference_times = []
    
    model.eval()
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            predictions, uncertainty = model(test_input)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
    
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    samples_per_second = batch_size / avg_inference_time
    
    print(f"✅ Model inference successful!")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Uncertainty shape: {uncertainty.shape}")
    print(f"   Average inference time: {avg_inference_time:.4f}s ± {std_inference_time:.4f}s")
    print(f"   Samples per second: {samples_per_second:.1f}")
    print(f"   Predictions per second: {samples_per_second:.1f}")
    
    # Check GPU memory usage
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"   GPU memory allocated: {memory_allocated:.2f} GB")
        print(f"   GPU memory reserved: {memory_reserved:.2f} GB")
    
    return avg_inference_time, samples_per_second

def test_training_step(model, device):
    """Test a single training step"""
    print("\n🏋️ Testing Training Step")
    print("-" * 40)
    
    # Create optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create training batch
    batch_size = 16  # Smaller batch for testing
    sequence_length = 7
    input_features = 30
    
    # Generate random training data
    sequences = torch.randn(batch_size, sequence_length, input_features).to(device)
    labels = torch.randint(0, 4, (batch_size,)).to(device)
    
    print(f"Training batch shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Training step
    model.train()
    
    start_time = time.time()
    
    # Forward pass
    optimizer.zero_grad()
    predictions, uncertainty = model(sequences)
    loss = criterion(predictions, labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    training_time = time.time() - start_time
    
    # Calculate accuracy
    _, predicted = torch.max(predictions.data, 1)
    accuracy = (predicted == labels).sum().item() / labels.size(0)
    
    print(f"✅ Training step successful!")
    print(f"   Training time: {training_time:.4f}s")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    
    return training_time, loss.item(), accuracy

def test_memory_efficiency(device):
    """Test memory efficiency with different batch sizes"""
    print("\n💾 Testing Memory Efficiency")
    print("-" * 40)
    
    if device.type != 'cuda':
        print("Skipping memory test - not using GPU")
        return
    
    batch_sizes = [8, 16, 32, 64, 128]
    successful_batch_sizes = []
    
    for batch_size in batch_sizes:
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Create model
            model = create_mini_tot_model(device=device, input_features=30, hidden_dim=64)
            
            # Create test data
            sequences = torch.randn(batch_size, 7, 30).to(device)
            labels = torch.randint(0, 4, (batch_size,)).to(device)
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                predictions, uncertainty = model(sequences)
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated(0) / 1e9
            
            print(f"   Batch size {batch_size:3d}: ✅ Memory used: {memory_used:.2f} GB")
            successful_batch_sizes.append(batch_size)
            
            # Clean up
            del model, sequences, labels, predictions, uncertainty
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   Batch size {batch_size:3d}: ❌ Out of memory")
                break
            else:
                print(f"   Batch size {batch_size:3d}: ❌ Error: {e}")
    
    max_batch_size = max(successful_batch_sizes) if successful_batch_sizes else 0
    print(f"✅ Maximum batch size: {max_batch_size}")
    
    return max_batch_size

def main():
    """Main test function"""
    print("🧪 RSDRAS-Lite Complete System Test")
    print("=" * 60)
    
    # Test GPU setup
    device = test_gpu_setup()
    
    # Test feature extraction
    features, sequence = test_feature_extraction(device)
    
    # Test model creation
    model = test_model_creation(device)
    
    # Test model inference
    inference_time, throughput = test_model_inference(model, device)
    
    # Test training step
    train_time, loss, accuracy = test_training_step(model, device)
    
    # Test memory efficiency
    max_batch_size = test_memory_efficiency(device)
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 60)
    print(f"✅ Device: {device}")
    print(f"✅ Feature extraction: {len(features)} features")
    print(f"✅ Model parameters: {model.count_parameters():,}")
    print(f"✅ Model size: {model.get_model_size():.2f} MB")
    print(f"✅ Inference speed: {throughput:.1f} samples/second")
    print(f"✅ Training step: {train_time:.4f}s per step")
    
    if device.type == 'cuda':
        print(f"✅ Maximum batch size: {max_batch_size}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU memory available: {total_memory:.1f} GB")
    
    print("\n🎉 All tests completed successfully!")
    print("Ready to start RSDRAS-Lite training!")

if __name__ == "__main__":
    main() 