# Diabetes Prediction System

A comprehensive machine learning-based diabetes prediction system with advanced data algorithms and AI/ML concepts.

## Features

- **Multiple ML Algorithms**: Logistic Regression, Random Forest, SVM, XGBoost, KNN
- **Data Preprocessing**: Handling missing values, outliers, feature scaling
- **Feature Engineering**: Creation of derived features, PCA analysis
- **Class Imbalance Handling**: SMOTE, class weighting
- **Model Evaluation**: Cross-validation, ROC-AUC, confusion matrix
- **Interactive Web Interface**: Real-time predictions with visualization
- **Model Comparison**: Performance metrics for all algorithms

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Train models:
```bash
python train_models.py
```

2. Run the web application:
```bash
python app.py
```

3. Open browser and navigate to `http://localhost:5000`

## Project Structure

- `app.py` - Flask backend API
- `train_models.py` - ML model training and evaluation
- `data_preprocessing.py` - Data cleaning and feature engineering
- `templates/` - HTML templates
- `static/` - CSS, JavaScript files
- `models/` - Saved ML models
- `data/` - Dataset

## Algorithms Used

1. **Logistic Regression** - Linear classifier with L2 regularization
2. **Random Forest** - Ensemble of decision trees
3. **Support Vector Machine** - Non-linear classification with RBF kernel
4. **XGBoost** - Gradient boosting algorithm
5. **K-Nearest Neighbors** - Instance-based learning

## Data Processing Techniques

- Missing value imputation (mean/median)
- Outlier detection and handling (IQR method)
- Feature scaling (StandardScaler)
- Feature selection (SelectKBest, feature importance)
- Dimensionality reduction (PCA)
- SMOTE for handling class imbalance
