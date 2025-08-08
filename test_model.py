#!/usr/bin/env python3
"""
Test script to verify the SVM model works correctly
"""

import joblib
import numpy as np

def test_model():
    try:
        # Load the model
        model = joblib.load("best_svm_breast_cancer.joblib")
        print("✅ Model loaded successfully!")
        
        # Test with sample data (30 features for Wisconsin Breast Cancer Dataset)
        # These are sample values - in a real scenario, these would come from medical tests
        sample_data = np.array([[
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
            1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
            25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ]])
        
        # Make prediction
        prediction = model.predict(sample_data)[0]
        prediction_proba = model.predict_proba(sample_data)[0]
        
        print(f"✅ Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
        print(f"✅ Confidence: {max(prediction_proba) * 100:.2f}%")
        print(f"✅ Probabilities - Benign: {prediction_proba[0]*100:.2f}%, Malignant: {prediction_proba[1]*100:.2f}%")
        
        # Test model attributes
        print(f"✅ Model type: {type(model).__name__}")
        if hasattr(model, 'n_features_in_'):
            print(f"✅ Expected features: {model.n_features_in_}")
        
        return True
        
    except FileNotFoundError:
        print("❌ Model file 'best_svm_breast_cancer.joblib' not found!")
        print("   Make sure the model file is in the same directory as this script.")
        return False
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Breast Cancer SVM Model...")
    print("=" * 50)
    
    success = test_model()
    
    print("=" * 50)
    if success:
        print("🎉 Model test completed successfully!")
        print("   Your Flask app should work correctly.")
    else:
        print("💥 Model test failed!")
        print("   Please check the model file and try again.")
