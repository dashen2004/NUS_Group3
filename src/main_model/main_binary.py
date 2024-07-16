from data_processing import (load_data, prepare_data, oversample_data,
                             split_data, expand_data, transform_to_binary)
from model_training_binary import train_xgboost, train_mlp_classifier, train_voting_classifier_binary
from evaluation_binary import evaluate_model_binary, cross_validate_model_binary
from prediction import predict_new_data


def main():
    # Load and prepare data
    data = load_data('cleaned_all_data_processed.csv')
    X, y = prepare_data(data)

    # Converted into a binary classification problem if we need
    y = transform_to_binary(y)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, y_train = oversample_data(X_train, y_train)
    X_test = expand_data(X_test, 10)
    y_test = expand_data(y_test, 10)
    print("Load and prepare data completed")

    # Train classifiers
    print("Start training...")
    res_model = train_voting_classifier_binary(X_train, y_train)

    # Evaluate model
    print("Start evaluating...")
    evaluate_model_binary(res_model, X_test, y_test)

    # Predict on new data
    print("Predict on new data...")
    predict_new_data(res_model, 'candidate_processed4.csv')

    # Cross-validate model
    print("Cross-validate model..")
    cross_validate_model_binary(res_model, X, y)


if __name__ == "__main__":
    main()
