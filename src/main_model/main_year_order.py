from data_processing import load_data, oversample_data, split_data, augment_data, expand_data
from model_training import train_mlp_classifier, train_voting_classifier
from evaluation import evaluate_model, cross_validate_model
from function_year_order import prepare_data_beta, show_tendency, predict_new_data_beta


def main():

    data = load_data('data_year_order.csv')
    X, y = prepare_data_beta(data)

    years = range(2000, 2021, 4)
    results = []

    for start_year in years:

        if start_year == 2020:
            break

        # Load and prepare data
        X_train = X[data['Year'] <= start_year]
        y_train = y[data['Year'] <= start_year]
        X_test = X[data['Year'] > start_year]
        y_test = y[data['Year'] > start_year]

        X_train, y_train = oversample_data(X_train, y_train)
        X_test = expand_data(X_test, 10)
        y_test = expand_data(y_test, 10)

        # data_argument
        # X_train, y_train = augment_data(X_train, y_train, target_size=10000)
        print("Load and prepare data completed")

        # Train classifiers
        print("Start training...")
        mlp_classifier = train_mlp_classifier(X_train, y_train)
        voting_classifier = train_voting_classifier(X_train, y_train, mlp_classifier)

        # Evaluate model
        print("Start evaluating...")
        evaluate_model(voting_classifier, X_test, y_test, y)

        # Cross-validate model
        print("Cross-validate model..")
        accuracy = cross_validate_model(voting_classifier, X, y)

        results.append((start_year, accuracy))

    show_tendency(results)
    pass


if __name__ == "__main__":
    main()
