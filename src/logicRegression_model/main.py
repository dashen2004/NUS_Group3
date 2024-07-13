from data_processing import load_and_clean_data
from logistic_regression import perform_logistic_regression
from visualization import plot_roc_curve, plot_predicted_probabilities, plot_confusion_matrix_and_report, \
    plot_feature_importance, plot_boxplots, plot_pred_vs_actual


def main():
    # Load and clean data
    file_path = 'data_processing2_updated3_new.csv'
    df_cleaned = load_and_clean_data(file_path)

    # Perform logistic regression
    result, X, y, y_pred_prob = perform_logistic_regression(df_cleaned)

    # Plot ROC curve
    plot_roc_curve(y, y_pred_prob)

    # Plot predicted probabilities
    plot_predicted_probabilities(y_pred_prob)

    # Plot confusion matrix and classification report
    plot_confusion_matrix_and_report(y, y_pred_prob)

    # Plot feature importance
    plot_feature_importance(result)

    # Plot boxplots
    columns_to_analyze = [
        'Average Age mean',
        'Average Net Result mean',
        'Average Mark var',
        'current_match_ranking mean',
        'current_match_ranking min',
        'Average Mark min',
        'Nation mean',
        'Years to Olympics',
        'Finalist_Weight'
    ]
    plot_boxplots(df_cleaned, columns_to_analyze)

    # Plot predicted vs actual values
    plot_pred_vs_actual(y_pred_prob, y)


if __name__ == "__main__":
    main()
