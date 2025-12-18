# Diabetes Prediction System - Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
cd diabetes_predictor
pip install -r requirements.txt
```

### 2. Train the Models
```bash
python train_models.py
```

This will:
- Create a sample diabetes dataset (or use existing data/diabetes.csv)
- Apply advanced data preprocessing and feature engineering
- Train 8 different ML algorithms
- Evaluate and compare all models
- Save trained models to the `models/` directory
- Generate performance visualizations

### 3. Run the Web Application
```bash
python app.py
```

Access the application at: `http://localhost:5000`

## Features Overview

### Data Preprocessing (`data_preprocessing.py`)
- **Missing Value Handling**: Median/mean/mode imputation
- **Outlier Detection**: IQR method with configurable thresholds
- **Outlier Handling**: Capping, removal, or winsorization
- **Feature Engineering**:
  - BMI categories (Underweight, Normal, Overweight, Obese)
  - Age groups (Young, Middle-aged, Senior)
  - Glucose categories (Normal, Pre-diabetic, Diabetic)
  - Blood pressure categories
  - Interaction features (Glucose × BMI)
  - Polynomial features (Glucose², BMI²)
  - Ratio features (Age/Pregnancy ratio)
- **Feature Scaling**: StandardScaler and RobustScaler
- **Feature Selection**: SelectKBest with F-test and Chi-squared
- **Dimensionality Reduction**: PCA
- **Class Imbalance**: SMOTE oversampling

### Machine Learning Models (`train_models.py`)

Eight algorithms with optimized hyperparameters:

1. **Logistic Regression**
   - L2 regularization
   - Balanced class weights
   - Max iterations: 1000

2. **Random Forest**
   - 100 trees
   - Max depth: 10
   - Balanced class weights
   - Min samples split: 5

3. **Support Vector Machine (SVM)**
   - RBF kernel
   - C=1.0, gamma='scale'
   - Probability estimates enabled
   - Balanced class weights

4. **XGBoost**
   - 100 estimators
   - Learning rate: 0.1
   - Max depth: 6
   - Subsample: 0.8
   - Column subsample: 0.8

5. **K-Nearest Neighbors**
   - 5 neighbors
   - Distance-weighted voting
   - Minkowski distance

6. **Decision Tree**
   - Max depth: 5
   - Balanced class weights
   - Min samples split: 10

7. **Naive Bayes**
   - Gaussian distribution
   - Fast probabilistic classifier

8. **Gradient Boosting**
   - 100 estimators
   - Learning rate: 0.1
   - Max depth: 5

### Model Evaluation Metrics

Each model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: TP, TN, FP, FN counts
- **Cross-Validation**: 5-fold CV with mean and std

### Web Application (`app.py`)

#### API Endpoints

**Health Check**
```
GET /api/health
```
Returns server status and loaded models.

**Single Model Prediction**
```
POST /api/predict
Content-Type: application/json

{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 125,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
    "model": "Random Forest"
}
```

**All Models Prediction**
```
POST /api/predict_all
Content-Type: application/json

{
    "Pregnancies": 6,
    "Glucose": 148,
    ...
}
```

**Get All Models Info**
```
GET /api/models
```

**Get Specific Model Info**
```
GET /api/model/<model_name>
```

### Frontend Features

#### Main Prediction Page (`/`)
- Interactive form with input validation
- Real-time range indicators
- Single model prediction
- All models comparison
- Risk level assessment (Low/Moderate/High)
- Personalized health recommendations
- Visual result displays

#### Dashboard Page (`/dashboard`)
- Performance comparison charts
- Accuracy, F1-score, Precision/Recall charts
- ROC-AUC comparison
- Detailed metrics table
- Confusion matrices for all models
- Model information cards

## Data Algorithms Concepts Used

### 1. **Statistical Methods**
- Mean, median, mode for imputation
- Interquartile Range (IQR) for outlier detection
- Standard deviation and variance
- Correlation analysis

### 2. **Feature Engineering Algorithms**
- Binning and discretization
- Feature interaction creation
- Polynomial feature generation
- Feature scaling (z-score normalization)
- Robust scaling (median and IQR)

### 3. **Feature Selection**
- ANOVA F-test (f_classif)
- Chi-squared test
- Feature importance from tree models

### 4. **Dimensionality Reduction**
- Principal Component Analysis (PCA)
- Variance retention strategies

### 5. **Sampling Techniques**
- Stratified train-test split
- SMOTE (Synthetic Minority Over-sampling)
- Cross-validation (k-fold)

### 6. **Classification Algorithms**
- Linear models (Logistic Regression)
- Tree-based models (Decision Tree, Random Forest)
- Ensemble methods (Gradient Boosting, XGBoost)
- Instance-based (KNN)
- Kernel methods (SVM)
- Probabilistic (Naive Bayes)

### 7. **Evaluation Metrics**
- Confusion matrix analysis
- ROC curve and AUC
- Precision-Recall tradeoff
- F1-score optimization

### 8. **Optimization Techniques**
- Hyperparameter tuning with GridSearchCV
- Class weight balancing
- Regularization (L1, L2)
- Early stopping (XGBoost)

## Input Features Explanation

1. **Pregnancies**: Number of times pregnant (0-17)
2. **Glucose**: Plasma glucose concentration (mg/dL) - 2 hours oral glucose tolerance test
3. **BloodPressure**: Diastolic blood pressure (mm Hg)
4. **SkinThickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (μU/mL)
6. **BMI**: Body mass index (weight in kg/(height in m)²)
7. **DiabetesPedigreeFunction**: Diabetes pedigree function (genetic factor)
8. **Age**: Age in years

## Advanced Features

### Hyperparameter Tuning
The system includes GridSearchCV for Random Forest, SVM, and XGBoost:

```python
# Example usage in train_models.py
trainer = DiabetesModelTrainer()
X, y, _ = load_and_prepare_data()
best_model = trainer.hyperparameter_tuning('Random Forest', X, y)
```

### Custom Dataset
Replace `data/diabetes.csv` with your own dataset following the same format:
- Columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
- Outcome: 0 (non-diabetic) or 1 (diabetic)

### Model Persistence
All trained models are saved using joblib:
- Individual models: `models/<model_name>.pkl`
- Best model: `models/best_model.pkl`
- Preprocessor: `models/preprocessor.pkl`
- Results: `models/model_results.json`

## Troubleshooting

**Models not loading:**
- Ensure you've run `train_models.py` first
- Check that `models/` directory exists and contains .pkl files

**Import errors:**
- Install all requirements: `pip install -r requirements.txt`
- Use Python 3.8 or higher

**Poor predictions:**
- Retrain models with more data
- Adjust hyperparameters in `train_models.py`
- Try different feature engineering techniques

## Performance Tips

1. **For better accuracy**: Use ensemble methods (Random Forest, XGBoost)
2. **For faster predictions**: Use Logistic Regression or Naive Bayes
3. **For interpretability**: Use Decision Tree or Logistic Regression
4. **For handling non-linear relationships**: Use SVM or XGBoost

## Security Notes

- This is a demonstration system
- **NOT** for medical diagnosis
- Always consult healthcare professionals
- Input validation is implemented
- No patient data is stored

## Future Enhancements

- Deep learning models (Neural Networks)
- Real-time monitoring dashboard
- Patient history tracking
- Integration with medical databases
- Mobile application
- Batch prediction for multiple patients
- Model retraining pipeline
- A/B testing framework
