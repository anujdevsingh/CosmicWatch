#!/usr/bin/env python3
"""
RSDRAS-Lite Quick Start Script
Runs GPU tests first, then optionally starts training
"""

import sys
import os
import subprocess
import time

def print_header():
    """Print the RSDRAS-Lite header"""
    print("🚀" + "=" * 58 + "🚀")
    print("🚀  RSDRAS-Lite: Revolutionary Space Debris AI System  🚀")
    print("🚀         Hardware-Optimized for 4GB GPU + 8GB RAM    🚀")
    print("🚀" + "=" * 58 + "🚀")
    print()

def check_dependencies():
    """Check if required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_modules = [
        'torch', 'numpy', 'pandas', 'matplotlib', 
        'seaborn', 'sklearn', 'sqlite3'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            if module == 'sklearn':
                __import__('sklearn')
            else:
                __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"   ❌ {module}")
    
    if missing_modules:
        print(f"\n❌ Missing dependencies: {', '.join(missing_modules)}")
        print("Please install them using:")
        print("pip install torch numpy pandas matplotlib seaborn scikit-learn")
        return False
    
    print("✅ All dependencies available!")
    return True

def run_gpu_test():
    """Run the GPU setup test"""
    print("\n🧪 Running GPU Setup Test...")
    print("-" * 40)
    
    try:
        # Import and run the test
        from test_gpu_setup import main as test_main
        test_main()
        return True
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def run_feature_test():
    """Run feature extraction test"""
    print("\n📊 Running Feature Extraction Test...")
    print("-" * 40)
    
    try:
        from enhanced_features import test_feature_extraction
        test_feature_extraction()
        return True
    except Exception as e:
        print(f"❌ Feature test failed: {e}")
        return False

def run_model_test():
    """Run model creation test"""
    print("\n🧠 Running Model Test...")
    print("-" * 40)
    
    try:
        from mini_tot_model import main as model_main
        model_main()
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def start_training():
    """Start the full training process"""
    print("\n🏋️ Starting RSDRAS-Lite Training...")
    print("=" * 60)
    
    try:
        from train_mini_tot import main as train_main
        train_main()
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    """Main function"""
    print_header()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        return
    
    print("\n🔧 RSDRAS-Lite System Ready!")
    print("\nWhat would you like to do?")
    print("1. Run complete system test (GPU + Model + Features)")
    print("2. Run GPU test only")
    print("3. Start training immediately")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                print("\n🧪 Running Complete System Test...")
                success = run_gpu_test()
                if success:
                    print("\n✅ All tests passed!")
                    
                    proceed = input("\nStart training now? (y/n): ").strip().lower()
                    if proceed in ['y', 'yes']:
                        start_training()
                    else:
                        print("Training can be started later with: python train_mini_tot.py")
                else:
                    print("\n❌ Tests failed. Please check your setup.")
                break
                
            elif choice == "2":
                print("\n🧪 Running GPU Test Only...")
                run_gpu_test()
                break
                
            elif choice == "3":
                print("\n🏋️ Starting Training Immediately...")
                start_training()
                break
                
            elif choice == "4":
                print("\n👋 Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main() 