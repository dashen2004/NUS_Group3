import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


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
    sns.histplot(y_pred_prob, bins=30, kde=True)
    plt.title('Distribution of Predicted Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.show()


# Plot confusion matrix and print classification report
def plot_confusion_matrix_and_report(y, y_pred_prob):
    y_pred = (y_pred_prob > 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(cm)
    cr = classification_report(y, y_pred)
    print("\nClassification Report:")
    print(cr)


# Plot feature importance
def plot_feature_importance(result):
    coef = result.params.drop('const')
    plt.figure(figsize=(10, 6))
    coef.plot(kind='bar')
    plt.title('Feature Importance')
    plt.ylabel('Coefficient')
    plt.xticks(rotation=45)
    plt.show()


# Plot boxplots for each feature by Finalist Weight
def plot_boxplots(df_cleaned, columns_to_analyze):
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(columns_to_analyze[:-1]):
        plt.subplot(3, 3, i + 1)
        sns.boxplot(x='Finalist_Weight', y=column, data=df_cleaned)
        plt.title(f'Boxplot of Finalist Weight by {column}')
        plt.xlabel('Finalist Weight')
        plt.ylabel(column)
    plt.tight_layout()
    plt.show()


# Plot predicted probabilities vs actual values
def plot_pred_vs_actual(y_pred_prob, y):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred_prob, y, color='blue', alpha=0.5)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Value')
    plt.title('Predicted Probability vs Actual Value')
    plt.grid(True)
    plt.show()
