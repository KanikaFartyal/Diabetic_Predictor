"""
Data Preprocessing Module
Handles data cleaning, feature engineering, and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Advanced data preprocessing with multiple techniques"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.feature_selector = None
        self.pca = None
        self.feature_names = None
        
    def handle_missing_values(self, df, strategy='median'):
        """
        Handle missing values using different strategies
        - median: Replace with median (robust to outliers)
        - mean: Replace with mean
        - mode: Replace with mode
        """
        df_clean = df.copy()
        
        for column in df_clean.columns:
            if df_clean[column].isnull().sum() > 0:
                if strategy == 'median':
                    df_clean[column].fillna(df_clean[column].median(), inplace=True)
                elif strategy == 'mean':
                    df_clean[column].fillna(df_clean[column].mean(), inplace=True)
                elif strategy == 'mode':
                    df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
        
        return df_clean
    
    def detect_outliers_iqr(self, df, columns, threshold=1.5):
        """
        Detect outliers using IQR (Interquartile Range) method
        threshold: typically 1.5 for outliers, 3.0 for extreme outliers
        """
        outlier_indices = []
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            outlier_indices.extend(outliers)
        
        return list(set(outlier_indices))
    
    def handle_outliers(self, df, columns, method='cap'):
        """
        Handle outliers using different methods
        - cap: Cap values at boundaries
        - remove: Remove outlier rows
        - winsorize: Winsorization
        """
        df_clean = df.copy()
        
        if method == 'cap':
            for col in columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'remove':
            outlier_indices = self.detect_outliers_iqr(df_clean, columns)
            df_clean = df_clean.drop(outlier_indices)
        
        return df_clean
    
    def create_engineered_features(self, df):
        """
        Feature Engineering: Create new features from existing ones
        """
        df_eng = df.copy()
        
        # BMI categories
        if 'BMI' in df_eng.columns:
            df_eng['BMI_Category'] = pd.cut(df_eng['BMI'], 
                                            bins=[0, 18.5, 25, 30, 100],
                                            labels=[0, 1, 2, 3])
            df_eng['BMI_Category'] = df_eng['BMI_Category'].astype(float)
        
        # Age groups
        if 'Age' in df_eng.columns:
            df_eng['Age_Group'] = pd.cut(df_eng['Age'],
                                        bins=[0, 30, 45, 60, 100],
                                        labels=[0, 1, 2, 3])
            df_eng['Age_Group'] = df_eng['Age_Group'].astype(float)
        
        # Glucose categories
        if 'Glucose' in df_eng.columns:
            df_eng['Glucose_Category'] = pd.cut(df_eng['Glucose'],
                                               bins=[0, 100, 126, 200],
                                               labels=[0, 1, 2])
            df_eng['Glucose_Category'] = df_eng['Glucose_Category'].astype(float)
        
        # Blood Pressure categories
        if 'BloodPressure' in df_eng.columns:
            df_eng['BP_Category'] = pd.cut(df_eng['BloodPressure'],
                                          bins=[0, 80, 90, 150],
                                          labels=[0, 1, 2])
            df_eng['BP_Category'] = df_eng['BP_Category'].astype(float)
        
        # Interaction features
        if 'Glucose' in df_eng.columns and 'BMI' in df_eng.columns:
            df_eng['Glucose_BMI_Interaction'] = df_eng['Glucose'] * df_eng['BMI']
        
        if 'Age' in df_eng.columns and 'Pregnancies' in df_eng.columns:
            df_eng['Age_Pregnancy_Ratio'] = df_eng['Age'] / (df_eng['Pregnancies'] + 1)
        
        # Polynomial features (squares)
        if 'Glucose' in df_eng.columns:
            df_eng['Glucose_Squared'] = df_eng['Glucose'] ** 2
        
        if 'BMI' in df_eng.columns:
            df_eng['BMI_Squared'] = df_eng['BMI'] ** 2
        
        return df_eng
    
    def select_features(self, X, y, k=10, method='f_classif'):
        """
        Feature selection using statistical tests
        - f_classif: ANOVA F-value
        - chi2: Chi-squared test
        """
        if method == 'f_classif':
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'chi2':
            # Chi2 requires non-negative features
            X = X - X.min() + 1
            self.feature_selector = SelectKBest(score_func=chi2, k=k)
        
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        return X_selected, selected_features
    
    def apply_pca(self, X, n_components=None, variance_ratio=0.95):
        """
        Apply PCA for dimensionality reduction
        """
        if n_components is None:
            self.pca = PCA(n_components=variance_ratio)
        else:
            self.pca = PCA(n_components=n_components)
        
        X_pca = self.pca.fit_transform(X)
        
        return X_pca
    
    def scale_features(self, X, method='standard'):
        """
        Scale features using different methods
        - standard: StandardScaler (z-score normalization)
        - robust: RobustScaler (uses median and IQR)
        """
        if method == 'standard':
            X_scaled = self.scaler.fit_transform(X)
        elif method == 'robust':
            X_scaled = self.robust_scaler.fit_transform(X)
        
        return X_scaled
    
    def handle_class_imbalance(self, X, y):
        """
        Handle class imbalance using class weights (no oversampling)
        """
        # Just return original data - class weights will be used in models
        return X, y
    
    def get_feature_statistics(self, df):
        """
        Get comprehensive statistics about features
        """
        stats = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numeric_stats': df.describe().to_dict(),
            'correlation': df.corr().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {}
        }
        return stats


def load_and_prepare_data(filepath='data/diabetes.csv', use_feature_engineering=True):
    """
    Load and prepare diabetes dataset with comprehensive preprocessing
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Handle zero values as missing in certain columns (domain knowledge)
    zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_as_missing:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df, strategy='median')
    
    # Feature engineering
    if use_feature_engineering:
        df = preprocessor.create_engineered_features(df)
    
    # Separate features and target
    if 'Outcome' in df.columns:
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
    else:
        raise ValueError("Target column 'Outcome' not found in dataset")
    
    # Handle outliers
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    X = pd.DataFrame(
        preprocessor.handle_outliers(X, numeric_columns, method='cap'),
        columns=X.columns
    )
    
    return X, y, preprocessor


if __name__ == "__main__":
    # Test preprocessing
    print("Testing data preprocessing...")
    try:
        X, y, preprocessor = load_and_prepare_data()
        print(f"Data shape: {X.shape}")
        print(f"Features: {X.columns.tolist()}")
        print(f"Target distribution:\n{y.value_counts()}")
    except FileNotFoundError:
        print("Dataset file not found. Please ensure data/diabetes.csv exists.")
