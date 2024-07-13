import pandas as pd


# Predict new data using the trained voting classifier
def predict_new_data(voting_classifier, file_path):
    # Load new data from a CSV file
    new_data = pd.read_csv(file_path)

    # Select features for prediction
    X_new = new_data[['age_at_event mean', 'Average Net Result mean', 'Mark mean',
                      'Mark min', 'Mark var', 'current_match_ranking mean',
                      'current_match_ranking min', 'Average Rank mean', 'efficiency mean', 'Nat', 'Years to Olympics']]

    # Predict using the trained voting classifier
    new_predictions = voting_classifier.predict(X_new)

    # Add predictions to the new data
    new_data['Predicted Olympic Ranking'] = new_predictions

    # Print the predictions
    print(new_data[['Predicted Olympic Ranking']])

    # Save the predictions to a new CSV file
    new_data.to_csv('candidate_predictions_improved.csv', index=False)
