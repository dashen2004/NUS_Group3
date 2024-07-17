import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import numpy as np
import pandas as pd


# Plot ROC curve
def plot_roc_curve(y, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


# Plot distribution of predicted probabilities
def plot_predicted_probabilities(y_pred_prob):
    plt.figure(figsize=(10, 8))
    sns.histplot(y_pred_prob, bins=30, kde=True, color='blue')
    plt.title('Distribution of Predicted Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


# Plot confusion matrix and print classification report
def plot_confusion_matrix_and_report(y, y_pred_prob):
    y_pred = (y_pred_prob > 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    print("\nClassification Report:")
    print(classification_report(y, y_pred, digits=3))


# Plot feature importance
def plot_feature_importance(model, feature_names, X, y):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importance, color='blue')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.grid(True)
        plt.show()
    else:
        print("The model does not have feature_importances_ attribute. Calculating permutation importance instead.")
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()
        plt.figure(figsize=(10, 6))
        plt.barh(np.array(feature_names)[sorted_idx], result.importances_mean[sorted_idx], color='blue')
        plt.xlabel('Permutation Importance')
        plt.ylabel('Features')
        plt.title('Feature Importance')
        plt.grid(True)
        plt.show()
