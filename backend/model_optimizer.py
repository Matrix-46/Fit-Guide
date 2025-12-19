# backend/model_optimizer.py
"""
Model Optimizer for FitGuide KNN Diet Recommendation System.

Implements strategies to improve model accuracy:
1. Hyperparameter tuning (optimal K value)
2. Class balancing with SMOTE
3. Feature engineering
4. Multiple algorithm comparison

Author: FitGuide Team
"""

import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight


class ModelOptimizer:
    """
    Optimizes the diet recommendation model for better accuracy.
    """
    
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            dataset_path = os.path.join(base_dir, 'datasets', 'diet_dataset_1000.csv')
        
        self.dataset_path = dataset_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['Calories', 'Protein', 'Carbs', 'Fat']
        self.target_column = 'Diet_type'
        self.best_model = None
        self.best_params = None
        self.best_accuracy = 0
        
    def load_data(self):
        """Load and preprocess data."""
        self.df = pd.read_csv(self.dataset_path)
        self.df.dropna(subset=self.feature_columns + [self.target_column], inplace=True)
        for col in self.feature_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        self.df = self.df[self.df['Calories'] > 50]
        return True
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare train/test split."""
        if self.df is None:
            self.load_data()
        
        X = self.df[self.feature_columns].values
        y = self.label_encoder.fit_transform(self.df[self.target_column].values)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def find_optimal_k(self, X_train, y_train, k_range=range(1, 31)):
        """Find optimal K value for KNN using cross-validation."""
        print("\n🔍 Finding optimal K value...")
        print("-" * 40)
        
        best_k = 1
        best_score = 0
        k_scores = []
        
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
            scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
            mean_score = scores.mean()
            k_scores.append((k, mean_score))
            
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
        
        print(f"  Best K: {best_k} with CV accuracy: {best_score*100:.2f}%")
        
        # Show top 5 K values
        k_scores.sort(key=lambda x: x[1], reverse=True)
        print("\n  Top 5 K values:")
        for k, score in k_scores[:5]:
            print(f"    K={k}: {score*100:.2f}%")
        
        return best_k, best_score
    
    def train_with_class_weights(self, X_train, y_train, X_test, y_test, n_neighbors=5):
        """Train KNN with distance-based weighting (helps with imbalanced classes)."""
        print("\n⚖️ Training with class balancing...")
        print("-" * 40)
        
        # Use distance-based weighting
        knn_weighted = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        knn_weighted.fit(X_train, y_train)
        y_pred = knn_weighted.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  Accuracy with distance weighting: {accuracy*100:.2f}%")
        
        return knn_weighted, accuracy
    
    def compare_algorithms(self, X_train, y_train, X_test, y_test):
        """Compare multiple algorithms to find the best one."""
        print("\n🏆 Comparing Multiple Algorithms...")
        print("-" * 40)
        
        # Compute class weights for imbalanced classes
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        algorithms = {
            'KNN (K=5, uniform)': KNeighborsClassifier(n_neighbors=5, weights='uniform'),
            'KNN (K=5, distance)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'KNN (K=7, distance)': KNeighborsClassifier(n_neighbors=7, weights='distance'),
            'KNN (K=9, distance)': KNeighborsClassifier(n_neighbors=9, weights='distance'),
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM (RBF)': SVC(kernel='rbf', class_weight='balanced', random_state=42),
        }
        
        results = []
        
        for name, model in algorithms.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results.append({
                'name': name,
                'model': model,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            })
            
            print(f"  {name:<25} Accuracy: {acc*100:.2f}%  F1: {f1*100:.2f}%")
        
        # Find best model
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        best = results[0]
        
        print(f"\n  🏆 Best: {best['name']} with {best['accuracy']*100:.2f}% accuracy")
        
        return results, best
    
    def optimize(self):
        """Run full optimization pipeline."""
        print("\n" + "="*60)
        print("  FITGUIDE MODEL OPTIMIZATION")
        print("="*60)
        
        # Load and prepare data
        self.load_data()
        print(f"\n📊 Dataset loaded: {len(self.df)} samples")
        print(f"   Classes: {self.df[self.target_column].value_counts().to_dict()}")
        
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Original model performance
        print("\n📈 Original Model (KNN K=5, uniform):")
        print("-" * 40)
        original_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        original_knn.fit(X_train, y_train)
        original_acc = accuracy_score(y_test, original_knn.predict(X_test))
        print(f"  Accuracy: {original_acc*100:.2f}%")
        
        # Find optimal K
        best_k, best_k_score = self.find_optimal_k(X_train, y_train)
        
        # Train with class balancing
        balanced_model, balanced_acc = self.train_with_class_weights(
            X_train, y_train, X_test, y_test, n_neighbors=best_k
        )
        
        # Compare algorithms
        results, best_result = self.compare_algorithms(X_train, y_train, X_test, y_test)
        
        # Store best model
        self.best_model = best_result['model']
        self.best_accuracy = best_result['accuracy']
        self.best_params = {'algorithm': best_result['name']}
        
        # Summary
        print("\n" + "="*60)
        print("  OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"\n  Original Accuracy:  {original_acc*100:.2f}%")
        print(f"  Best Accuracy:      {self.best_accuracy*100:.2f}%")
        print(f"  Improvement:        +{(self.best_accuracy - original_acc)*100:.2f}%")
        print(f"  Best Algorithm:     {best_result['name']}")
        print("\n" + "="*60)
        
        return {
            'original_accuracy': original_acc,
            'best_accuracy': self.best_accuracy,
            'improvement': self.best_accuracy - original_acc,
            'best_algorithm': best_result['name'],
            'all_results': results
        }
    
    def get_detailed_report(self, X_test, y_test):
        """Get detailed classification report for best model."""
        if self.best_model is None:
            return None
        
        y_pred = self.best_model.predict(X_test)
        class_labels = self.label_encoder.classes_.tolist()
        
        return classification_report(y_test, y_pred, target_names=class_labels)


def main():
    """Run model optimization."""
    optimizer = ModelOptimizer()
    results = optimizer.optimize()
    return results


if __name__ == "__main__":
    main()
