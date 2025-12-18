"""
Flask Backend API for Diabetes Prediction
Provides REST API endpoints for predictions and model information
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
import json
from data_preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'diabetes-prediction-secret-key-2024'

# Global variables for models
models = {}
preprocessor = None
model_results = {}
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


def load_models():
    """Load all trained models and preprocessor"""
    global models, preprocessor, model_results
    
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        print("Models directory not found. Please train models first.")
        return False
    
    # Load preprocessor
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        print("Preprocessor loaded successfully")
    
    # Load all models
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'SVM': 'svm.pkl',
        'XGBoost': 'xgboost.pkl',
        'K-Nearest Neighbors': 'k-nearest_neighbors.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Gradient Boosting': 'gradient_boosting.pkl'
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
            print(f"Loaded {model_name}")
    
    # Load model results
    results_path = os.path.join(models_dir, 'model_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            model_results = json.load(f)
        print("Model results loaded successfully")
    
    return len(models) > 0


def preprocess_input(data):
    """Preprocess input data for prediction"""
    # Create DataFrame
    df = pd.DataFrame([data], columns=feature_names)
    
    # Apply feature engineering (same as training)
    df_eng = df.copy()
    
    # BMI categories
    df_eng['BMI_Category'] = pd.cut(df_eng['BMI'], 
                                    bins=[0, 18.5, 25, 30, 100],
                                    labels=[0, 1, 2, 3])
    df_eng['BMI_Category'] = df_eng['BMI_Category'].astype(float)
    
    # Age groups
    df_eng['Age_Group'] = pd.cut(df_eng['Age'],
                                bins=[0, 30, 45, 60, 100],
                                labels=[0, 1, 2, 3])
    df_eng['Age_Group'] = df_eng['Age_Group'].astype(float)
    
    # Glucose categories
    df_eng['Glucose_Category'] = pd.cut(df_eng['Glucose'],
                                       bins=[0, 100, 126, 200],
                                       labels=[0, 1, 2])
    df_eng['Glucose_Category'] = df_eng['Glucose_Category'].astype(float)
    
    # Blood Pressure categories
    df_eng['BP_Category'] = pd.cut(df_eng['BloodPressure'],
                                  bins=[0, 80, 90, 150],
                                  labels=[0, 1, 2])
    df_eng['BP_Category'] = df_eng['BP_Category'].astype(float)
    
    # Interaction features
    df_eng['Glucose_BMI_Interaction'] = df_eng['Glucose'] * df_eng['BMI']
    df_eng['Age_Pregnancy_Ratio'] = df_eng['Age'] / (df_eng['Pregnancies'] + 1)
    
    # Polynomial features
    df_eng['Glucose_Squared'] = df_eng['Glucose'] ** 2
    df_eng['BMI_Squared'] = df_eng['BMI'] ** 2
    
    # Scale features
    if preprocessor:
        X_scaled = preprocessor.scale_features(df_eng, method='standard')
        return pd.DataFrame(X_scaled, columns=df_eng.columns)
    
    return df_eng


def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Moderate Risk"
    else:
        return "High Risk"


def get_recommendations(data, prediction, probability):
    """Generate personalized health recommendations"""
    recommendations = []
    
    # Glucose recommendations
    if data['Glucose'] > 140:
        recommendations.append("⚠️ Your glucose level is elevated. Consider consulting a doctor for diabetes screening.")
    elif data['Glucose'] > 100:
        recommendations.append("📊 Your glucose level is in the pre-diabetic range. Monitor your blood sugar regularly.")
    
    # BMI recommendations
    if data['BMI'] > 30:
        recommendations.append("🏃 Your BMI indicates obesity. Regular exercise and a balanced diet are recommended.")
    elif data['BMI'] > 25:
        recommendations.append("💪 Your BMI is slightly elevated. Maintaining a healthy weight can reduce diabetes risk.")
    
    # Blood Pressure
    if data['BloodPressure'] > 90:
        recommendations.append("❤️ Your blood pressure is high. Consider lifestyle changes and consult a healthcare provider.")
    
    # Age factor
    if data['Age'] > 45:
        recommendations.append("👴 Regular health check-ups are important at your age, especially for diabetes screening.")
    
    # General recommendations
    if prediction == 1:
        recommendations.extend([
            "🥗 Adopt a low-sugar, high-fiber diet",
            "🏋️ Engage in at least 150 minutes of moderate exercise per week",
            "💊 Consider consulting an endocrinologist",
            "📝 Monitor your blood glucose levels regularly",
            "🧘 Manage stress through meditation or yoga"
        ])
    else:
        recommendations.extend([
            "✅ Maintain a healthy lifestyle to prevent diabetes",
            "🥦 Eat a balanced diet rich in vegetables and whole grains",
            "🚶 Stay physically active",
            "⚖️ Maintain a healthy weight"
        ])
    
    return recommendations


@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction using the best model"""
    try:
        # Get input data
        data = request.json
        
        # Validate input
        required_fields = feature_names
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Extract values
        input_values = [float(data[field]) for field in feature_names]
        
        # Get selected model (default to Random Forest)
        selected_model = data.get('model', 'Random Forest')
        
        if selected_model not in models:
            return jsonify({'error': f'Model {selected_model} not found'}), 400
        
        # Preprocess input
        input_dict = {field: value for field, value in zip(feature_names, input_values)}
        X_processed = preprocess_input(input_dict)
        
        # Make prediction
        model = models[selected_model]
        prediction = int(model.predict(X_processed)[0])
        probability = float(model.predict_proba(X_processed)[0][1]) if hasattr(model, 'predict_proba') else 0.5
        
        # Get risk level
        risk_level = get_risk_level(probability)
        
        # Get recommendations
        recommendations = get_recommendations(input_dict, prediction, probability)
        
        # Prepare response
        response = {
            'prediction': prediction,
            'prediction_label': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'model_used': selected_model,
            'recommendations': recommendations,
            'input_data': input_dict
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict_all', methods=['POST'])
def predict_all_models():
    """Make predictions using all available models"""
    try:
        # Get input data
        data = request.json
        
        # Validate input
        required_fields = feature_names
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Extract values
        input_values = [float(data[field]) for field in feature_names]
        input_dict = {field: value for field, value in zip(feature_names, input_values)}
        
        # Preprocess input
        X_processed = preprocess_input(input_dict)
        
        # Make predictions with all models
        predictions = {}
        for model_name, model in models.items():
            pred = int(model.predict(X_processed)[0])
            prob = float(model.predict_proba(X_processed)[0][1]) if hasattr(model, 'predict_proba') else 0.5
            
            predictions[model_name] = {
                'prediction': pred,
                'prediction_label': 'Diabetic' if pred == 1 else 'Non-Diabetic',
                'probability': round(prob * 100, 2),
                'risk_level': get_risk_level(prob)
            }
        
        # Calculate consensus
        diabetic_count = sum(1 for p in predictions.values() if p['prediction'] == 1)
        consensus = 'Diabetic' if diabetic_count > len(predictions) / 2 else 'Non-Diabetic'
        
        response = {
            'predictions': predictions,
            'consensus': consensus,
            'consensus_percentage': round(diabetic_count / len(predictions) * 100, 2),
            'input_data': input_dict
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'available_models': list(models.keys())
    })


if __name__ == '__main__':
    print("=" * 80)
    print("DIABETES PREDICTION SYSTEM - BACKEND SERVER")
    print("=" * 80)
    
    # Load models
    print("\nLoading models...")
    if load_models():
        print(f"\n✓ Successfully loaded {len(models)} models")
        print(f"Available models: {', '.join(models.keys())}\n")
        
        print("=" * 80)
        print("Starting Flask server...")
        print("Access the application at: http://localhost:5000")
        print("API Documentation:")
        print("  - GET  /api/health          - Health check")
        print("  - POST /api/predict         - Single model prediction")
        print("  - POST /api/predict_all     - All models prediction")
        print("=" * 80)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Failed to load models. Please run train_models.py first.")
