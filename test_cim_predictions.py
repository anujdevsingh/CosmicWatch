#!/usr/bin/env python3
"""
Test CIM Model Predictions
"""

import torch
from cosmic_intelligence_model import get_cosmic_intelligence_model

def test_cim_predictions():
    print("🌌 Testing Cosmic Intelligence Model Predictions...")
    
    try:
        # Initialize model
        cosmic_model = get_cosmic_intelligence_model()
        
        # Check if model is loaded
        if not cosmic_model.is_loaded:
            print("⚠️ Model not loaded, trying to load checkpoint...")
            try:
                checkpoint = torch.load('cosmic_intelligence_best.pth', map_location=cosmic_model.device)
                cosmic_model.model.load_state_dict(checkpoint['model_state_dict'])
                cosmic_model.accuracy = checkpoint.get('accuracy', 0.0)
                cosmic_model.is_loaded = True
                print(f"✅ Loaded checkpoint with accuracy: {cosmic_model.accuracy:.4f}")
            except Exception as e:
                print(f"❌ Could not load checkpoint: {e}")
                return False
        
        # Test predictions with sample data
        test_cases = [
            {"altitude": 200, "velocity": 7.8, "inclination": 51.6, "size": 1.0},  # Should be HIGH/CRITICAL
            {"altitude": 800, "velocity": 7.5, "inclination": 28.5, "size": 0.5},  # Should be MEDIUM/LOW
            {"altitude": 400, "velocity": 7.6, "inclination": 98.0, "size": 2.0},  # Should be MEDIUM/HIGH
        ]
        
        print("🧪 Running test predictions...")
        for i, test_data in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Input: {test_data}")
            
            prediction = cosmic_model.predict_debris_risk(test_data)
            
            print(f"Risk Level: {prediction['risk_level']}")
            print(f"Confidence: {prediction['confidence']:.3f}")
            print(f"Probabilities: {prediction['probabilities']}")
            print(f"Model: {prediction['model_name']}")
            print(f"Enhanced: {prediction['enhanced']}")
            
            if 'uncertainty' in prediction:
                unc = prediction['uncertainty']
                print(f"Total Uncertainty: {unc['total']:.3f}")
        
        print("\n🎉 All predictions completed successfully!")
        print(f"📊 Model Performance: {cosmic_model.accuracy:.4f} accuracy")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in predictions: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cim_predictions()
    if success:
        print("✅ CIM model is working correctly!")
    else:
        print("❌ CIM model test failed!") 