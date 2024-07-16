import lightgbm as lgb
import xgboost as xgb
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
    mlp = MLPClassifier(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


# Train a voting classifier combining RandomForest and MLP classifiers
def train_voting_classifier(X_train, y_train, mlp_classifier):
    rf_classifier = RandomForestClassifier(random_state=42)
    voting_classifier = VotingClassifier(estimators=[('rf', rf_classifier), ('mlp', mlp_classifier)], voting='soft')
    voting_classifier.fit(X_train, y_train)
    return voting_classifier


def train_lightgbm(X_train, y_train):
    lgb_classifier = lgb.LGBMClassifier(objective='binary', metric='binary_logloss')
    # 限制超参数网格搜索的范围
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'num_leaves': [31, 63]
    }
    grid_search = GridSearchCV(estimator=lgb_classifier, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1,
                               verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def train_voting_classifier_binary(X_train, y_train):
    lgb_model = train_lightgbm(X_train, y_train)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    voting_classifier = VotingClassifier(
        estimators=[
            ('lgb', lgb_model),
            ('rf', rf_model)
        ],
        voting='soft',
        n_jobs=-1
    )
    voting_classifier.fit(X_train, y_train)
    return voting_classifier


def train_xgboost(X_train, y_train):
    # 定义XGBoost分类器
    xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

    # 定义超参数网格
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }

    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1,
                               verbose=2)
    grid_search.fit(X_train, y_train)

    # 输出最佳参数
    print(f"Best parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_
