"""
Machine Learning Model Training
Multiple algorithms with hyperparameter tuning and evaluation
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_prepare_data, DataPreprocessor
from graph_algorithms import PatientSimilarityGraph, FeatureCorrelationGraph, Graph
from sorting_algorithms import SortingAlgorithms, SearchAlgorithms, compare_sorting_algorithms
import os
import warnings
warnings.filterwarnings('ignore')


class DiabetesModelTrainer:
    """Train and evaluate multiple ML models for diabetes prediction"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.preprocessor = None
        
    def initialize_models(self):
        """Initialize multiple ML algorithms"""
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='minkowski'
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            )
        }
    
    def train_model(self, model, X_train, y_train):
        """Train a single model"""
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
    
    def cross_validate_model(self, model, X, y, cv=5):
        """Perform cross-validation"""
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def train_all_models(self, X, y):
        """Train and evaluate all models"""
        # Initialize models
        self.initialize_models()
        
        # Create preprocessor
        self.preprocessor = DataPreprocessor()
        
        # Scale features
        X_scaled = self.preprocessor.scale_features(X, method='standard')
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
        
        print("=" * 80)
        print("TRAINING MULTIPLE ML MODELS FOR DIABETES PREDICTION")
        print("=" * 80)
        
        # Train and evaluate each model
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            print("-" * 40)
            
            # Train
            trained_model = self.train_model(model, X_train, y_train)
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(trained_model, X_test, y_test, model_name)
            
            # Cross-validation
            cv_metrics = self.cross_validate_model(trained_model, X_scaled, y, cv=5)
            
            # Store results
            self.results[model_name] = {
                'test_metrics': test_metrics,
                'cv_metrics': cv_metrics,
                'model': trained_model
            }
            
            # Print results
            print(f"Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Precision: {test_metrics['precision']:.4f}")
            print(f"Recall: {test_metrics['recall']:.4f}")
            print(f"F1-Score: {test_metrics['f1_score']:.4f}")
            print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
            print(f"CV Score: {cv_metrics['mean_cv_score']:.4f} (+/- {cv_metrics['std_cv_score']:.4f})")
        
        # Find best model
        self.find_best_model()
        
        return X_train, X_test, y_train, y_test
    
    def find_best_model(self):
        """Find the best performing model based on F1-score"""
        best_f1 = 0
        best_name = None
        
        for model_name, results in self.results.items():
            f1 = results['test_metrics']['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_name = model_name
        
        self.best_model = {
            'name': best_name,
            'model': self.results[best_name]['model'],
            'metrics': self.results[best_name]['test_metrics']
        }
        
        print("\n" + "=" * 80)
        print(f"BEST MODEL: {best_name}")
        print(f"F1-Score: {best_f1:.4f}")
        print("=" * 80)
    
    def hyperparameter_tuning(self, model_name, X, y):
        """Perform hyperparameter tuning for a specific model"""
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly']
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0]
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return None
        
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best F1-score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_models(self, output_dir='models'):
        """Save all trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual models
        for model_name, results in self.results.items():
            safe_name = model_name.replace(' ', '_').lower()
            model_path = os.path.join(output_dir, f'{safe_name}.pkl')
            joblib.dump(results['model'], model_path)
            print(f"Saved {model_name} to {model_path}")
        
        # Save preprocessor
        preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"Saved preprocessor to {preprocessor_path}")
        
        # Save best model separately
        if self.best_model:
            best_model_path = os.path.join(output_dir, 'best_model.pkl')
            joblib.dump(self.best_model['model'], best_model_path)
            print(f"Saved best model to {best_model_path}")
        
        # Save results summary
        results_summary = {}
        for model_name, results in self.results.items():
            results_summary[model_name] = {
                'test_metrics': results['test_metrics'],
                'cv_metrics': results['cv_metrics']
            }
        
        results_path = os.path.join(output_dir, 'model_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=4)
        print(f"Saved results summary to {results_path}")
    
    def plot_model_comparison(self, output_dir='static/images'):
        """Create visualization comparing all models"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for plotting
        model_names = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        data = {metric: [] for metric in metrics}
        
        for model_name in model_names:
            for metric in metrics:
                data[metric].append(self.results[model_name]['test_metrics'][metric])
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(range(len(model_names)), data[metric], color='skyblue', edgecolor='navy')
            ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Remove extra subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'model_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {plot_path}")
        plt.close()
        
        # Create confusion matrix for best model
        if self.best_model:
            cm = np.array(self.best_model['metrics']['confusion_matrix'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Diabetes', 'Diabetes'],
                       yticklabels=['No Diabetes', 'Diabetes'])
            plt.title(f'Confusion Matrix - {self.best_model["name"]}', fontweight='bold')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix to {cm_path}")
            plt.close()


def create_sample_dataset():
    """Create a sample diabetes dataset if none exists"""
    from sklearn.datasets import make_classification
    
    # Generate synthetic diabetes-like data
    X, y = make_classification(
        n_samples=768,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=[0.65, 0.35],
        random_state=42
    )
    
    # Create DataFrame with meaningful column names
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    df = pd.DataFrame(X, columns=columns)
    
    # Scale to realistic ranges
    df['Pregnancies'] = (df['Pregnancies'] * 5 + 5).clip(0, 17).astype(int)
    df['Glucose'] = (df['Glucose'] * 40 + 120).clip(0, 200)
    df['BloodPressure'] = (df['BloodPressure'] * 20 + 70).clip(0, 122)
    df['SkinThickness'] = (df['SkinThickness'] * 20 + 20).clip(0, 99)
    df['Insulin'] = (df['Insulin'] * 200 + 100).clip(0, 846)
    df['BMI'] = (df['BMI'] * 10 + 30).clip(0, 67.1)
    df['DiabetesPedigreeFunction'] = (df['DiabetesPedigreeFunction'] * 0.5 + 0.5).clip(0.078, 2.42)
    df['Age'] = (df['Age'] * 20 + 33).clip(21, 81).astype(int)
    df['Outcome'] = y
    
    return df


if __name__ == "__main__":
    print("Diabetes Prediction Model Training\n")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Check if dataset exists, if not create sample
    data_path = 'data/diabetes.csv'
    if not os.path.exists(data_path):
        print("Dataset not found. Creating sample dataset...")
        df = create_sample_dataset()
        df.to_csv(data_path, index=False)
        print(f"Sample dataset saved to {data_path}\n")
    
    # Load and prepare data
    print("Loading and preprocessing data...")
    X, y, preprocessor = load_and_prepare_data(data_path)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Class distribution:\n{y.value_counts()}\n")
    
    # ============= DATA STRUCTURE ALGORITHMS IMPLEMENTATION =============
    print("\n" + "=" * 80)
    print("APPLYING DATA STRUCTURE ALGORITHMS")
    print("=" * 80)
    
    # 1. SORTING ALGORITHMS - Sort feature importances
    print("\n1. SORTING ALGORITHMS - Ranking Features by Variance")
    sorter = SortingAlgorithms()
    feature_variances = X.var().values
    feature_names = X.columns.tolist()
    
    print(f"Original feature variances: {feature_variances[:5]}...")
    sorted_variances = sorter.quick_sort(feature_variances.copy())
    print(f"Sorted using Quick Sort: {sorted_variances[:5]}...")
    
    # Compare sorting algorithms on feature variances
    print("\nComparing different sorting algorithms:")
    compare_sorting_algorithms(feature_variances[:10])
    
    # 2. BINARY SEARCH - Find features with specific variance
    print("\n2. BINARY SEARCH - Finding Features")
    searcher = SearchAlgorithms()
    target_variance = np.median(sorted_variances)
    print(f"Searching for feature with variance close to median: {target_variance:.4f}")
    idx = searcher.binary_search(sorted_variances, target_variance)
    
    # 3. GRAPH ALGORITHMS - Build Patient Similarity Network
    print("\n3. GRAPH ALGORITHMS - Patient Similarity Analysis")
    
    # Use subset of data for demonstration (first 50 patients)
    X_subset = X.iloc[:50].values
    y_subset = y.iloc[:50].values
    
    patient_graph = PatientSimilarityGraph(X_subset, y_subset)
    similarity_graph = patient_graph.build_similarity_graph(threshold=0.85)
    
    # BFS - Find similar patients
    print("\n=== BFS: Finding Similar Patients ===")
    similar_patients = patient_graph.find_similar_patients_bfs(patient_id=0, max_distance=2)
    print(f"Patients similar to Patient 0 (within 2 hops): {len(similar_patients)}")
    
    # DFS - Find patient clusters
    print("\n=== DFS: Identifying Patient Clusters ===")
    clusters = patient_graph.find_patient_clusters_dfs()
    print(f"Number of patient clusters found: {len(clusters)}")
    for i, cluster in enumerate(clusters[:3]):
        print(f"Cluster {i+1}: {len(cluster)} patients")
    
    # 4. DIJKSTRA'S ALGORITHM - Find shortest path between patients
    print("\n=== DIJKSTRA: Shortest Path Between Patients ===")
    if len(similar_patients) > 0:
        distances, parent = similarity_graph.dijkstra(0)
        print(f"Distances from Patient 0: {distances[:10]}")
        
        # Find path to a connected patient
        for patient_id, _ in similar_patients[:1]:
            path = similarity_graph.get_shortest_path(parent, 0, patient_id)
            print(f"Shortest path from Patient 0 to Patient {patient_id}: {path}")
    
    # 5. FEATURE CORRELATION GRAPH with MST
    print("\n4. GRAPH ALGORITHMS - Feature Correlation Network")
    
    # Use original features only (not engineered ones)
    original_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X_original = X[original_features]
    
    feature_graph = FeatureCorrelationGraph(X_original.values, original_features)
    corr_graph = feature_graph.build_correlation_graph(threshold=0.3)
    
    # KRUSKAL'S ALGORITHM - Find MST of feature correlations
    print("\n=== KRUSKAL'S MST: Most Important Feature Relationships ===")
    feature_mst_kruskal = feature_graph.find_feature_mst_kruskal()
    print("\nTop feature relationships (Kruskal's MST):")
    for rel in feature_mst_kruskal[:5]:
        print(f"  {rel['feature1']} <-> {rel['feature2']}: correlation = {rel['correlation']:.4f}")
    
    # PRIM'S ALGORITHM - Alternative MST
    print("\n=== PRIM'S MST: Alternative MST Construction ===")
    feature_mst_prim = feature_graph.find_feature_mst_prim()
    print("\nTop feature relationships (Prim's MST):")
    for rel in feature_mst_prim[:5]:
        print(f"  {rel['feature1']} <-> {rel['feature2']}: correlation = {rel['correlation']:.4f}")
    
    # 6. BELLMAN-FORD - Check for negative cycles (using inverse correlation)
    print("\n=== BELLMAN-FORD: Negative Cycle Detection ===")
    if corr_graph.V > 0:
        distances, parent, valid = corr_graph.bellman_ford(0)
        print(f"Graph is valid (no negative cycles): {valid}")
        print(f"Sample distances from feature 0: {distances[:5]}")
    
    print("\n" + "=" * 80)
    print("DATA STRUCTURE ALGORITHMS APPLIED SUCCESSFULLY")
    print("=" * 80)
    
    # ============= MACHINE LEARNING MODEL TRAINING =============
    
    # Initialize trainer
    trainer = DiabetesModelTrainer(test_size=0.2, random_state=42)
    
    # Train all models
    X_train, X_test, y_train, y_test = trainer.train_all_models(X, y)
    
    # Save models
    print("\nSaving models...")
    trainer.save_models()
    
    # Create visualizations
    print("\nCreating visualizations...")
    trainer.plot_model_comparison()
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 80)
