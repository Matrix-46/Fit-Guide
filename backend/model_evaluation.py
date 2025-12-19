# backend/model_evaluation.py
"""
Model Evaluation Module for FitGuide KNN Diet Recommendation System.

This module provides comprehensive ML evaluation metrics including:
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- Classification Report
- Cross-validation with k-fold

Author: FitGuide Team
"""

import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class DietModelEvaluator:
    """
    Evaluates the KNN-based diet recommendation model using standard ML metrics.
    
    Metrics provided:
        - Accuracy: Overall correct predictions
        - Precision: True positives / predicted positives (per class & weighted)
        - Recall: True positives / actual positives (per class & weighted)
        - F1 Score: Harmonic mean of precision and recall
        - Confusion Matrix: Per-class prediction breakdown
        - Cross-validation: K-fold validation with multiple scoring metrics
    """
    
    def __init__(self, dataset_path=None, n_neighbors=5, test_size=0.2, random_state=42):
        """
        Initialize the evaluator.
        
        Args:
            dataset_path: Path to the diet dataset CSV file
            n_neighbors: Number of neighbors for KNN (default: 5)
            test_size: Proportion of data for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.n_neighbors = n_neighbors
        self.test_size = test_size
        self.random_state = random_state
        
        # Set default dataset path
        if dataset_path is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            dataset_path = os.path.join(base_dir, 'datasets', 'diet_dataset_1000.csv')
        
        self.dataset_path = dataset_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['Calories', 'Protein', 'Carbs', 'Fat']
        self.target_column = 'Diet_type'
        self.classes = None
        
        # Results storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        
    def load_data(self):
        """Load and preprocess the diet dataset."""
        try:
            self.df = pd.read_csv(self.dataset_path)
            
            # Clean data
            self.df.dropna(subset=self.feature_columns + [self.target_column], inplace=True)
            
            # Convert numeric columns
            for col in self.feature_columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
            
            # Filter out invalid entries
            self.df = self.df[self.df['Calories'] > 50]
            
            # Encode target labels
            self.classes = self.df[self.target_column].unique().tolist()
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_data(self):
        """Prepare feature matrix X and target vector y."""
        if self.df is None:
            if not self.load_data():
                raise ValueError("Failed to load dataset")
        
        X = self.df[self.feature_columns].values
        y = self.label_encoder.fit_transform(self.df[self.target_column].values)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Train the KNN classifier."""
        if self.X_train is None:
            self.prepare_data()
        
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(self.X_train, self.y_train)
        
        return self.model
    
    def predict(self):
        """Make predictions on test data."""
        if self.model is None:
            self.train_model()
        
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred
    
    def calculate_accuracy(self):
        """Calculate accuracy score."""
        if self.y_pred is None:
            self.predict()
        
        return accuracy_score(self.y_test, self.y_pred)
    
    def calculate_precision(self, average='weighted'):
        """
        Calculate precision score.
        
        Args:
            average: 'weighted', 'macro', 'micro', or None for per-class
        """
        if self.y_pred is None:
            self.predict()
        
        return precision_score(self.y_test, self.y_pred, average=average, zero_division=0)
    
    def calculate_recall(self, average='weighted'):
        """
        Calculate recall score.
        
        Args:
            average: 'weighted', 'macro', 'micro', or None for per-class
        """
        if self.y_pred is None:
            self.predict()
        
        return recall_score(self.y_test, self.y_pred, average=average, zero_division=0)
    
    def calculate_f1(self, average='weighted'):
        """
        Calculate F1 score.
        
        Args:
            average: 'weighted', 'macro', 'micro', or None for per-class
        """
        if self.y_pred is None:
            self.predict()
        
        return f1_score(self.y_test, self.y_pred, average=average, zero_division=0)
    
    def get_confusion_matrix(self):
        """Get confusion matrix."""
        if self.y_pred is None:
            self.predict()
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Get class labels
        class_labels = self.label_encoder.classes_.tolist()
        
        return {
            'matrix': cm.tolist(),
            'labels': class_labels
        }
    
    def get_classification_report(self, output_dict=True):
        """Get detailed classification report with per-class metrics."""
        if self.y_pred is None:
            self.predict()
        
        class_labels = self.label_encoder.classes_.tolist()
        
        if output_dict:
            report = classification_report(
                self.y_test, self.y_pred, 
                target_names=class_labels,
                output_dict=True,
                zero_division=0
            )
            return report
        else:
            return classification_report(
                self.y_test, self.y_pred,
                target_names=class_labels,
                zero_division=0
            )
    
    def cross_validate(self, cv=5, scoring_metrics=None):
        """
        Perform k-fold cross-validation.
        
        Args:
            cv: Number of folds (default: 5)
            scoring_metrics: List of metrics to evaluate (default: accuracy, precision, recall, f1)
        
        Returns:
            Dictionary with cross-validation scores for each metric
        """
        if self.df is None:
            self.load_data()
        
        X = self.df[self.feature_columns].values
        y = self.label_encoder.fit_transform(self.df[self.target_column].values)
        
        # Scale all data
        X_scaled = self.scaler.fit_transform(X)
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Default scoring metrics
        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        cv_results = {}
        model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        
        for metric in scoring_metrics:
            scores = cross_val_score(model, X_scaled, y, cv=skf, scoring=metric)
            cv_results[metric] = {
                'scores': scores.tolist(),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            }
        
        return cv_results
    
    def evaluate_all_metrics(self):
        """
        Run complete evaluation and return all metrics.
        
        Returns:
            Comprehensive dictionary with all evaluation metrics
        """
        # Ensure model is trained and predictions are made
        if self.y_pred is None:
            self.prepare_data()
            self.train_model()
            self.predict()
        
        # Get basic metrics
        accuracy = self.calculate_accuracy()
        precision_weighted = self.calculate_precision('weighted')
        recall_weighted = self.calculate_recall('weighted')
        f1_weighted = self.calculate_f1('weighted')
        
        # Get macro metrics (treats all classes equally)
        precision_macro = self.calculate_precision('macro')
        recall_macro = self.calculate_recall('macro')
        f1_macro = self.calculate_f1('macro')
        
        # Get confusion matrix
        conf_matrix = self.get_confusion_matrix()
        
        # Get classification report (per-class metrics)
        class_report = self.get_classification_report(output_dict=True)
        
        # Get cross-validation results
        cv_results = self.cross_validate(cv=5)
        
        return {
            'model_info': {
                'algorithm': 'K-Nearest Neighbors (KNN)',
                'n_neighbors': self.n_neighbors,
                'features': self.feature_columns,
                'target': self.target_column,
                'dataset_size': len(self.df) if self.df is not None else 0,
                'train_size': len(self.X_train) if self.X_train is not None else 0,
                'test_size': len(self.X_test) if self.X_test is not None else 0
            },
            'overall_metrics': {
                'accuracy': round(accuracy, 4),
                'precision_weighted': round(precision_weighted, 4),
                'recall_weighted': round(recall_weighted, 4),
                'f1_weighted': round(f1_weighted, 4),
                'precision_macro': round(precision_macro, 4),
                'recall_macro': round(recall_macro, 4),
                'f1_macro': round(f1_macro, 4)
            },
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'cross_validation': cv_results
        }
    
    def print_evaluation_report(self):
        """Print a formatted evaluation report to console."""
        results = self.evaluate_all_metrics()
        
        print("\n" + "="*70)
        print("       FITGUIDE KNN DIET RECOMMENDATION MODEL - EVALUATION REPORT")
        print("="*70)
        
        # Model Info
        print("\n📊 MODEL INFORMATION")
        print("-"*50)
        info = results['model_info']
        print(f"  Algorithm:     {info['algorithm']}")
        print(f"  K Neighbors:   {info['n_neighbors']}")
        print(f"  Features:      {', '.join(info['features'])}")
        print(f"  Target:        {info['target']}")
        print(f"  Dataset Size:  {info['dataset_size']} samples")
        print(f"  Train/Test:    {info['train_size']} / {info['test_size']}")
        
        # Overall Metrics
        print("\n📈 OVERALL METRICS (Weighted Average)")
        print("-"*50)
        metrics = results['overall_metrics']
        print(f"  Accuracy:      {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision:     {metrics['precision_weighted']:.4f}")
        print(f"  Recall:        {metrics['recall_weighted']:.4f}")
        print(f"  F1 Score:      {metrics['f1_weighted']:.4f}")
        
        print("\n📈 OVERALL METRICS (Macro Average)")
        print("-"*50)
        print(f"  Precision:     {metrics['precision_macro']:.4f}")
        print(f"  Recall:        {metrics['recall_macro']:.4f}")
        print(f"  F1 Score:      {metrics['f1_macro']:.4f}")
        
        # Per-Class Metrics
        print("\n📋 PER-CLASS METRICS")
        print("-"*50)
        class_report = results['classification_report']
        print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("  " + "-"*60)
        for class_name, class_metrics in class_report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"  {class_name:<20} {class_metrics['precision']:>10.4f} "
                      f"{class_metrics['recall']:>10.4f} {class_metrics['f1-score']:>10.4f} "
                      f"{int(class_metrics['support']):>10}")
        
        # Confusion Matrix
        print("\n🔲 CONFUSION MATRIX")
        print("-"*50)
        cm = results['confusion_matrix']
        labels = cm['labels']
        matrix = cm['matrix']
        
        # Print header
        print("  Predicted →")
        print(f"  {'Actual ↓':<15}", end="")
        for label in labels:
            print(f"{label[:12]:>12}", end="")
        print()
        print("  " + "-"*60)
        
        # Print matrix rows
        for i, label in enumerate(labels):
            print(f"  {label:<15}", end="")
            for j in range(len(labels)):
                print(f"{matrix[i][j]:>12}", end="")
            print()
        
        # Cross-Validation
        print("\n🔄 5-FOLD CROSS-VALIDATION")
        print("-"*50)
        cv = results['cross_validation']
        print(f"  {'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("  " + "-"*60)
        for metric_name, metric_values in cv.items():
            display_name = metric_name.replace('_', ' ').title()
            print(f"  {display_name:<25} {metric_values['mean']:>10.4f} "
                  f"{metric_values['std']:>10.4f} {metric_values['min']:>10.4f} "
                  f"{metric_values['max']:>10.4f}")
        
        print("\n" + "="*70)
        print("✅ Evaluation Complete!")
        print("="*70 + "\n")
        
        return results


def main():
    """Main function to run evaluation from command line."""
    print("\n🚀 Starting FitGuide KNN Model Evaluation...")
    
    evaluator = DietModelEvaluator()
    
    if evaluator.load_data():
        print(f"✓ Dataset loaded: {len(evaluator.df)} samples")
        results = evaluator.print_evaluation_report()
        return results
    else:
        print("❌ Failed to load dataset. Please check the file path.")
        return None


if __name__ == "__main__":
    main()
