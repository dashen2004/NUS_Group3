import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import numpy as np


# Evaluate the model and print classification report
def evaluate_model(voting_classifier, X_test, y_test, y):
    y_pred = voting_classifier.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1, digits=3,
                                labels=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                target_names=['1', '2', '3', '4', '5', '6', '7', '8', '9']))
    plot_roc_curves(voting_classifier, X_test, y_test, y)
    plot_confusion_matrix(y_test, y_pred, y)


# Plot ROC curves for each class
def plot_roc_curves(voting_classifier, X_test, y_test, y):
    y_score = voting_classifier.predict_proba(X_test)
    n_classes = len(np.unique(y))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure()
        plt.plot(fpr[i], tpr[i], color='darkorange', lw=2,
                 label='ROC curve (area = {:.2f})'.format(roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - Class {}'.format(i + 1))
        plt.legend(loc="lower right")
        plt.show()


# Plot the confusion matrix
def plot_confusion_matrix(y_test, y_pred, y):
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    cmd = ConfusionMatrixDisplay(cm, display_labels=np.unique(y))
    cmd.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()


# Perform cross-validation and print accuracy scores
def cross_validate_model(voting_classifier, X, y):
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(voting_classifier, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean cross-validation accuracy: {cv_scores.mean()}")
    return cv_scores
