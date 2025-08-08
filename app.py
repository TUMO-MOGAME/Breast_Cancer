from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
try:
    model = joblib.load("best_svm_breast_cancer.joblib")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Feature names for the breast cancer dataset (Wisconsin Breast Cancer Dataset)
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

# Feature descriptions for better user understanding
FEATURE_DESCRIPTIONS = {
    'radius_mean': 'Mean of distances from center to points on the perimeter',
    'texture_mean': 'Standard deviation of gray-scale values',
    'perimeter_mean': 'Mean size of the core tumor',
    'area_mean': 'Mean area of the tumor',
    'smoothness_mean': 'Mean of local variation in radius lengths',
    'compactness_mean': 'Mean of perimeter^2 / area - 1.0',
    'concavity_mean': 'Mean of severity of concave portions of the contour',
    'concave_points_mean': 'Mean for number of concave portions of the contour',
    'symmetry_mean': 'Mean symmetry',
    'fractal_dimension_mean': 'Mean for "coastline approximation" - 1'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please check if the model file exists.',
                'success': False
            }), 500

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

        # Convert to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        prediction_proba = model.predict_proba(features_array)[0]
        
        # Convert prediction to human-readable format
        # Assuming 0 = Benign, 1 = Malignant (standard for breast cancer datasets)
        result = "Malignant" if prediction == 1 else "Benign"
        confidence = max(prediction_proba) * 100
        
        return jsonify({
            'prediction': result,
            'confidence': round(confidence, 2),
            'prediction_numeric': int(prediction),
            'probabilities': {
                'benign': round(prediction_proba[0] * 100, 2),
                'malignant': round(prediction_proba[1] * 100, 2)
            },
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'success': False
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Endpoint to get feature names and descriptions"""
    return jsonify({
        'features': FEATURE_NAMES,
        'descriptions': FEATURE_DESCRIPTIONS,
        'success': True
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'success': True
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
