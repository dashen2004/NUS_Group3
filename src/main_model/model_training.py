from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

# Train MLP classifier with hyperparameter tuning using GridSearchCV
def train_mlp_classifier(X_train, y_train):
    param_grid = {
        'hidden_layer_sizes': [(100,), (200,), (200, 100), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    mlp = MLPClassifier(max_iter=300, random_state=42)
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Train a voting classifier combining RandomForest and MLP classifiers
def train_voting_classifier(X_train, y_train, mlp_classifier):
    rf_classifier = RandomForestClassifier(random_state=42)
    voting_classifier = VotingClassifier(estimators=[('rf', rf_classifier), ('mlp', mlp_classifier)], voting='soft')
    voting_classifier.fit(X_train, y_train)
    return voting_classifier
