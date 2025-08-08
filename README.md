# Breast Cancer Prediction Web Application

A beautiful, AI-powered web application for breast cancer prediction using a trained Support Vector Machine (SVM) model. Created by **Tumo Olorato Mogame**.

## 🌟 Features

- **Beautiful Medical-themed UI** with gradient backgrounds and professional styling
- **AI-Powered Predictions** using a trained SVM model
- **Real-time Analysis** with confidence scores
- **Responsive Design** that works on all devices
- **Medical Professional Attribution** and branding
- **Easy Deployment** to Vercel

## 🏥 About

This application uses machine learning to analyze tumor characteristics and predict whether a breast tumor is benign or malignant. The model is trained on the Wisconsin Breast Cancer Dataset and uses Support Vector Machine algorithms for classification.

**⚠️ Important Medical Disclaimer**: This tool is for educational and research purposes only. Always consult with qualified medical professionals for actual medical diagnosis and treatment decisions.

## 🚀 Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the Model** (Optional but recommended)
   ```bash
   python test_model.py
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Open in Browser**
   Navigate to `http://localhost:5000`

### 🌐 Deploy to Vercel

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   vercel
   ```

   Follow the prompts:
   - Set up and deploy? **Y**
   - Which scope? Choose your account
   - Link to existing project? **N**
   - Project name: `breast-cancer-prediction` (or your preferred name)
   - Directory: `./` (current directory)

4. **Production Deployment**
   ```bash
   vercel --prod
   ```

## 📁 Project Structure

```
breast-cancer-prediction/
├── app.py                          # Flask backend application
├── requirements.txt                # Python dependencies
├── vercel.json                     # Vercel deployment configuration
├── best_svm_breast_cancer.joblib   # Trained SVM model
├── test_model.py                   # Model testing script
├── templates/
│   └── index.html                  # Beautiful frontend interface
└── README.md                       # This file
```

## 🔧 API Endpoints

### `GET /`
Returns the main web interface

### `POST /predict`
Accepts tumor characteristics and returns prediction

**Request Body:**
```json
{
  "radius_mean": 17.99,
  "texture_mean": 10.38,
  "perimeter_mean": 122.8,
  "area_mean": 1001.0,
  "smoothness_mean": 0.1184,
  "compactness_mean": 0.2776,
  "concavity_mean": 0.3001,
  "concave_points_mean": 0.1471,
  "symmetry_mean": 0.2419,
  "fractal_dimension_mean": 0.07871
}
```

**Response:**
```json
{
  "prediction": "Malignant",
  "confidence": 95.67,
  "prediction_numeric": 1,
  "probabilities": {
    "benign": 4.33,
    "malignant": 95.67
  },
  "success": true
}
```

### `GET /features`
Returns available features and their descriptions

### `GET /health`
Health check endpoint

## 🎨 UI Features

- **Medical Professional Branding** with creator attribution
- **Gradient Backgrounds** with medical color schemes
- **Interactive Form** with real-time validation
- **Loading Animations** during prediction
- **Result Visualization** with color-coded outcomes
- **Responsive Design** for mobile and desktop
- **Medical Icons** and professional typography

## 🧪 Model Information

- **Algorithm**: Support Vector Machine (SVM)
- **Dataset**: Wisconsin Breast Cancer Dataset
- **Features**: 30 tumor characteristics
- **Classes**: Benign (0) and Malignant (1)
- **Performance**: Optimized for accuracy and reliability

## 🛠️ Customization

### Styling
The CSS is embedded in the HTML template for easy customization. Key color variables:
- Primary gradient: `#667eea` to `#764ba2`
- Success (Benign): `#48bb78` to `#38a169`
- Warning (Malignant): `#f56565` to `#e53e3e`

### Features
To modify the input features, update the `FEATURES` object in the JavaScript section of `index.html` and ensure the backend `FEATURE_NAMES` list matches.

## 📝 License

This project is for educational and research purposes. Please ensure compliance with medical software regulations in your jurisdiction.

## 👨‍💻 Creator

**Tumo Olorato Mogame**  
Medical AI Research & Development

---

*This application demonstrates the power of machine learning in medical diagnosis assistance while maintaining the importance of professional medical consultation.*
