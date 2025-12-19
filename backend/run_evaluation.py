# run_evaluation.py
# Simple script to run and display model evaluation metrics

from model_evaluation import DietModelEvaluator

print("\n" + "="*60)
print("  FITGUIDE KNN MODEL EVALUATION RESULTS")
print("="*60)

evaluator = DietModelEvaluator()
evaluator.load_data()
results = evaluator.evaluate_all_metrics()

# Overall Metrics
print("\n OVERALL METRICS:")
print("-"*40)
m = results['overall_metrics']
print(f"  Accuracy:   {m['accuracy']*100:.2f}%")
print(f"  Precision:  {m['precision_weighted']*100:.2f}%")
print(f"  Recall:     {m['recall_weighted']*100:.2f}%")
print(f"  F1 Score:   {m['f1_weighted']*100:.2f}%")

# Per-Class Metrics
print("\n PER-CLASS METRICS:")
print("-"*40)
cr = results['classification_report']
for cls in ['Vegetarian', 'Vegan', 'Non-vegetarian']:
    c = cr[cls]
    print(f"  {cls}:")
    print(f"    Precision: {c['precision']*100:.1f}%  Recall: {c['recall']*100:.1f}%  F1: {c['f1-score']*100:.1f}%")

# Confusion Matrix
print("\n CONFUSION MATRIX:")
print("-"*40)
cm = results['confusion_matrix']
print("                   Non-veg  Vegan  Vegetarian")
for i, label in enumerate(cm['labels']):
    row = '  '.join([str(x).rjust(6) for x in cm['matrix'][i]])
    print(f"  {label:<15} {row}")

# Cross-Validation
print("\n CROSS-VALIDATION (5-fold):")
print("-"*40)
cv = results['cross_validation']
print(f"  Mean Accuracy: {cv['accuracy']['mean']*100:.2f}% +/- {cv['accuracy']['std']*100:.2f}%")

print("\n" + "="*60)
print(" Evaluation Complete!")
print("="*60 + "\n")
