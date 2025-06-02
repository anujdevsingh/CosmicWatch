#!/usr/bin/env python3
"""
Simple test script to debug CIM training issues
"""

import sys
import traceback

try:
    print("🔍 Step 1: Importing cosmic intelligence model...")
    from cosmic_intelligence_model import get_cosmic_intelligence_model
    print("✅ Import successful")
    
    print("🔍 Step 2: Initializing model...")
    cosmic_model = get_cosmic_intelligence_model()
    print("✅ Model initialized")
    
    print("🔍 Step 3: Getting model info...")
    info = cosmic_model.get_model_info()
    print(f"✅ Model info: {info['model_name']} - {info['num_parameters']:,} parameters")
    
    print("🔍 Step 4: Starting training...")
    training_results = cosmic_model.train_model()
    
    if training_results:
        print("✅ Training completed successfully!")
        print(f"Best accuracy: {training_results.get('best_accuracy', 'N/A')}")
    else:
        print("❌ Training returned empty results")
        
except Exception as e:
    print(f"❌ Error occurred: {e}")
    print("\n🔍 Full traceback:")
    traceback.print_exc()
    sys.exit(1)

print("🎉 Test completed successfully!") 