from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pickle
import json

app = Flask(__name__)
CORS(app)

# Try to load the model, but provide fallback if it fails
model = None
model_loaded = False

try:
    import joblib
    import numpy as np
    model = joblib.load("best_svm_breast_cancer.joblib")
    model_loaded = True
    print("Model loaded successfully!")
except ImportError as e:
    print(f"Scientific packages not available: {e}")
    print("Running in demo mode...")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Running in demo mode...")

# Feature names for the breast cancer dataset
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

def demo_prediction(features):
    """
    Demo prediction function when model is not available
    This uses simple heuristics based on medical knowledge
    """
    # Simple heuristic: larger, more irregular tumors are more likely malignant
    radius = features[0] if len(features) > 0 else 10
    texture = features[1] if len(features) > 1 else 10
    perimeter = features[2] if len(features) > 2 else 50
    area = features[3] if len(features) > 3 else 300
    
    # Simple scoring based on size and texture
    score = 0
    if radius > 15: score += 0.3
    if texture > 20: score += 0.2
    if perimeter > 100: score += 0.3
    if area > 800: score += 0.2
    
    # Return prediction (1 = malignant, 0 = benign)
    prediction = 1 if score > 0.5 else 0
    confidence = min(0.95, max(0.55, score + 0.3))
    
    return prediction, [1-confidence, confidence]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'success': False
            }), 400

        # Extract features in the correct order
        features = []
        missing_features = []
        
        for feature_name in FEATURE_NAMES:
            if feature_name in data:
                try:
                    value = float(data[feature_name])
                    features.append(value)
                except (ValueError, TypeError):
                    return jsonify({
                        'error': f'Invalid value for {feature_name}. Please provide a numeric value.',
                        'success': False
                    }), 400
            else:
                missing_features.append(feature_name)
        
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {", ".join(missing_features)}',
                'success': False
            }), 400

        # Make prediction
        if model_loaded and model is not None:
            # Use the actual trained model
            import numpy as np
            features_array = np.array(features).reshape(1, -1)
            prediction = model.predict(features_array)[0]
            prediction_proba = model.predict_proba(features_array)[0]
        else:
            # Use demo prediction
            prediction, prediction_proba = demo_prediction(features)
        
        # Convert prediction to human-readable format
        result = "Malignant" if prediction == 1 else "Benign"
        confidence = max(prediction_proba) * 100
        
        response_data = {
            'prediction': result,
            'confidence': round(confidence, 2),
            'prediction_numeric': int(prediction),
            'probabilities': {
                'benign': round(prediction_proba[0] * 100, 2),
                'malignant': round(prediction_proba[1] * 100, 2)
            },
            'success': True,
            'demo_mode': not model_loaded
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'success': False
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Endpoint to get feature names"""
    return jsonify({
        'features': FEATURE_NAMES,
        'success': True
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'demo_mode': not model_loaded,
        'success': True
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
