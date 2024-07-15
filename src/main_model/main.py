from data_processing import load_data, prepare_data, oversample_data, split_data, augment_data
from model_training import train_mlp_classifier, train_voting_classifier
from evaluation import evaluate_model, cross_validate_model, plot_roc_curves, plot_confusion_matrix
from prediction import predict_new_data


def main():
    # Load and prepare data
    # TODO: change the model data
    data = load_data('cleaned_all_data_processed.csv')
    X, y = prepare_data(data)
    X_resampled, y_resampled = oversample_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)

    # data_argument
    X_train, y_train = augment_data(X_train, y_train, target_size=8000)

    # Train classifiers
    mlp_classifier = train_mlp_classifier(X_train, y_train)
    voting_classifier = train_voting_classifier(X_train, y_train, mlp_classifier)

    # Evaluate model
    evaluate_model(voting_classifier, X_test, y_test, y)

    # Predict on new data
    predict_new_data(voting_classifier, 'candidate_processed4.csv')

    # Cross-validate model
    cross_validate_model(voting_classifier, X, y)


if __name__ == "__main__":
    main()
