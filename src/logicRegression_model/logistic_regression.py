import statsmodels.api as sm

# Perform logistic regression on cleaned data
def perform_logistic_regression(df_cleaned):
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

    # Select features for logistic regression
    X = df_cleaned[columns_to_analyze[:-1]]
    X = sm.add_constant(X)  # Add constant term to the model

    # Define target variable
    y = df_cleaned['Finalist_Weight']

    # Fit logistic regression model
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()

    # Print model summary
    print(result.summary())

    # Predict probabilities
    y_pred_prob = result.predict(X)

    return result, X, y, y_pred_prob
