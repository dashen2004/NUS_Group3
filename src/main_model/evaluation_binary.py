import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import numpy as np


# Evaluate the model and print classification report
def evaluate_model_binary(voting_classifier, X_test, y_test):
    y_pred = voting_classifier.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1, digits=3,
                                labels=[0, 1], target_names=['Class 0', 'Class 1']))

    plot_roc_curve_binary(voting_classifier, X_test, y_test)
    plot_confusion_matrix_binary(y_test, y_pred)


# Plot ROC curve for binary classification
def plot_roc_curve_binary(voting_classifier, X_test, y_test_bin):
    y_score = voting_classifier.predict_proba(X_test)[:, 1]  # 只取正类的概率得分

    fpr, tpr, _ = roc_curve(y_test_bin, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


# Plot the confusion matrix
def plot_confusion_matrix_binary(y_test_bin, y_pred_bin):
    cm = confusion_matrix(y_test_bin, y_pred_bin, labels=[0, 1])
    cmd = ConfusionMatrixDisplay(cm, display_labels=['9', '1-8'])
    cmd.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()


# Perform cross-validation and print accuracy scores
def cross_validate_model_binary(voting_classifier, X, y):
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(voting_classifier, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean cross-validation accuracy: {cv_scores.mean()}")
    return cv_scores
